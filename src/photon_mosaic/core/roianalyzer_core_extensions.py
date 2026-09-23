import warnings
from typing import Any

import numpy as np
import sparse
from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.node_pipeline import PipelineNode, run_node_pipeline

from .baseimaging import BaseImaging
from .baserois import BaseRois
from .roianalyzer import AnalyzerExtension, register_result_extension


class FluorescenceExtension(AnalyzerExtension):
    """Extension to extract fluorescence traces from ROIs."""

    extension_name = "fluorescence"
    depend_on: list[str] = []
    need_imaging = True
    use_nodepipeline = True
    nodepipeline_variables = ["fluorescence"]
    need_job_kwargs = True

    @classmethod
    def get_optional_dependencies(cls, **params):
        return ["neuropil"]

    def _set_params(self, use_neuropil=True, neuropil_weight=0.7):
        return dict(use_neuropil=use_neuropil, neuropil_weight=neuropil_weight)

    def _run(self, verbose=False, **job_kwargs):
        gather_mode = "memory"
        gather_kwargs = {}

        job_kwargs = fix_job_kwargs(job_kwargs)
        nodes = self.get_pipeline_nodes()
        fluorescence = run_node_pipeline(
            self.roi_analyzer.imaging,
            nodes,
            job_kwargs=job_kwargs,
            job_name=self.extension_name,
            gather_mode=gather_mode,
            gather_kwargs=gather_kwargs,
            verbose=verbose,
        )
        self.data["fluorescence"] = fluorescence

    def _get_pipeline_nodes(self):
        neuropil = None
        neuropil_traces = None
        if self.params["use_neuropil"] and self.roi_analyzer.has_extension("neuropil"):
            ext = self.roi_analyzer.get_extension("neuropil")
            # .get() rather than [] so analyzers saved before 'method' existed still load.
            if ext.params.get("method", "surround") == "cnmf":
                # The CNMF background is low-rank in time (b @ f), not a fixed spatial mask, so the
                # per-ROI neuropil trace is precomputed by NeuropilExtension and sliced per chunk.
                neuropil_traces = ext.get_data("neuropil_traces")
            else:
                neuropil = ext.get_data()
        return [
            FluorescenceNode(
                self.roi_analyzer.imaging,
                self.roi_analyzer.rois,
                neuropil=neuropil,
                neuropil_traces=neuropil_traces,
                neuropil_weight=self.params["neuropil_weight"],
            )
        ]

    def _get_data(self, outputs="numpy"):
        fluorescence_traces = self.data["fluorescence"]
        if outputs == "numpy":
            return fluorescence_traces
        elif outputs == "recording":
            from spikeinterface.core import NumpyRecording

            return NumpyRecording(
                fluorescence_traces,
                sampling_frequency=self.roi_analyzer.imaging.sampling_frequency,
                channel_ids=self.roi_analyzer.rois.roi_ids,
            )
        else:
            raise ValueError(f"Unsupported output type: {outputs}. Supported types are 'numpy' and 'recording'.")

    def _select_extension_data(self, roi_ids):
        roi_indices = self.roi_analyzer.rois.ids_to_indices(roi_ids)
        return {"fluorescence": self.data["fluorescence"][:, roi_indices]}


class FluorescenceNode(PipelineNode):
    # Opt into run_node_pipeline's wider call signature, which additionally supplies
    # (start_frame, end_frame, segment_index, max_margin) -- needed to slice precomputed
    # per-ROI neuropil traces to the chunk being processed. Harmless when they go unused.
    _compute_has_extended_signature = True

    def __init__(
        self,
        imaging: BaseImaging,
        rois: BaseRois,
        neuropil: np.ndarray | None = None,
        neuropil_weight: float = 0.7,
        neuropil_traces: np.ndarray | None = None,
    ):
        """
        Pipeline node to extract fluorescence traces from ROIs, with optional neuropil subtraction.

        Each ROI's own mask (whatever ``rois.get_roi_image_masks()`` returns) is renormalized
        internally so the returned trace correctly reconstructs the movie via ``traces @
        masks`` (using the original, unnormalized masks) and, for non-overlapping ROIs,
        matches the least-squares solution of ``movie ~ traces @ masks``. For binary masks
        (Suite2pRois and generate_rois's default) this exactly recovers each ROI's own
        per-pixel value; masks themselves are untouched (`rois` isn't mutated).

        Parameters
        ----------
        imaging : BaseImaging
            The imaging data to analyze.
        rois : BaseRois
            The ROIs to extract fluorescence from.
        neuropil : np.ndarray, optional
            Optional neuropil mask(s) to subtract from the fluorescence traces.
            Should have shape (num_rois, height, width) or (height, width).
        neuropil_weight : float, optional
            Weight to apply to the neuropil signal before subtraction (default is 0.7).
        neuropil_traces : np.ndarray, optional
            Precomputed per-ROI neuropil traces of shape ``(n_frames_total, num_rois)``, with frames
            concatenated across epochs in epoch order (the same ordering ``run_node_pipeline``
            gathers). Used instead of ``neuropil`` when the neuropil model is not expressible as a
            fixed spatial mask -- e.g. :class:`NeuropilExtension`'s ``method='cnmf'``, whose
            background ``b @ f`` varies frame by frame. Mutually exclusive with ``neuropil``.
        """
        PipelineNode.__init__(
            self,
            imaging,
            parents=[],
            return_output=True,
        )
        self.rois = rois
        self.neuropil = neuropil
        self.neuropil_weight = neuropil_weight

        if neuropil is not None and neuropil_traces is not None:
            raise ValueError(
                "Pass either `neuropil` (spatial masks) or `neuropil_traces` (precomputed per-ROI traces), not both."
            )
        if neuropil_traces is None:
            self._neuropil_traces = None
            self._epoch_offsets = None
        else:
            self._neuropil_traces = np.asarray(neuropil_traces, dtype=np.float32)
            # run_node_pipeline reports start_frame/end_frame *per epoch*, so global row indices
            # need the cumulative frame count of the preceding epochs.
            frames_per_epoch = [imaging.get_num_frames(epoch_index=i) for i in range(imaging.get_num_epochs())]
            self._epoch_offsets = np.concatenate(([0], np.cumsum(frames_per_epoch))).astype(np.int64)

        # Precompute flattened masks for efficient matrix multiplication. masks may be a
        # dense ndarray or a sparse array (see BaseRois.get_roi_image_masks) -- reshape must
        # be called with a single shape tuple for sparse arrays to accept it (unlike ndarray,
        # which also accepts unpacked dimensions); a tuple works for both.
        masks = rois.get_roi_image_masks()  # (N, H, W) or (N, H, W, P)
        num_rois = masks.shape[0]
        masks_flat = masks.reshape((num_rois, -1)).astype(np.float32)  # (N, spatial)

        def _row_norm(x):
            x = x.todense() if isinstance(x, sparse.SparseArray) else x
            x = np.asarray(x).reshape(-1, 1)
            x[x == 0] = 1.0  # an all-zero mask ROI stays all-zero, not NaN, downstream
            return x

        # Extraction uses each ROI's mask normalized to sum to 1 (L1, "mean" convention) --
        # matching Suite2p's own F/Fneu convention (stat['lam'] weights sum to ~1;
        # NeuropilExtension's ring masks are likewise L1-normalized, see
        # _build_surround_neuropil_masks) so that F and the neuropil trace are on the same scale
        # and F - neuropil_weight * neuropil_trace is dimensionally meaningful.
        l1_norm = _row_norm(masks_flat.sum(axis=1))
        self._masks_flat = masks_flat / l1_norm

        # But L1 isn't the scale that correctly reconstructs the movie (or generalizes to
        # regression-based extraction of overlapping ROIs): movie ~ traces @ masks has the
        # least-squares solution traces = movie @ masks.T @ pinv(masks @ masks.T), which for
        # non-overlapping ROIs is diagonal with each entry ||mask_n||^2 (L2 norm squared, not
        # L1) -- so the correctly-scaled trace divides by L2 squared, not L1. This L1->L2
        # rescale step is itself a no-op for binary masks (mask**2 == mask, so L1 == L2^2 --
        # true for every current mask source: Suite2pRois, generate_rois's default). The
        # L1-normalization above is not a no-op for binary masks, though: it changes F from a
        # raw per-ROI pixel sum to a per-pixel mean, for every mask, binary included --
        # deliberate (matching Suite2p's F/Fneu mean convention so neuropil subtraction is
        # dimensionally meaningful), not specific to weighted masks. Harmless for dF/F (a
        # per-ROI constant scale factor cancels in (F - F0) / F0), but does change raw
        # `fluorescence` values for every binary mask. Extracting via L1 first (for
        # neuropil-subtraction consistency, above) and rescaling the result by L1/L2^2
        # afterwards is algebraically identical to extracting via L2^2 directly and rescaling
        # the neuropil term by the same factor -- scalar multiplication distributes over the
        # subtraction -- so this order also keeps the neuropil subtraction itself correctly in
        # Suite2p's mean-scale convention.
        l2sq_norm = _row_norm((masks_flat**2).sum(axis=1))
        self._rescale_to_l2 = (l1_norm / l2sq_norm).reshape(1, -1)  # (1, N), for compute()

        if self._neuropil_traces is not None:
            assert self._epoch_offsets is not None  # set together, just above
            expected = (int(self._epoch_offsets[-1]), num_rois)
            if self._neuropil_traces.shape != expected:
                raise ValueError(
                    f"neuropil_traces has shape {self._neuropil_traces.shape}, expected {expected} "
                    "(total frames across all epochs, num_rois)"
                )

        # Precompute flattened neuropil masks
        if neuropil is not None:
            if neuropil.ndim == 2:
                # Global neuropil (H, W) -> (1, spatial)
                self._neuropil_flat = neuropil.reshape((1, -1)).astype(np.float32)
            else:
                # Per-ROI neuropil (N, H, W) -> (N, spatial)
                self._neuropil_flat = neuropil.reshape((neuropil.shape[0], -1)).astype(np.float32)
        else:
            self._neuropil_flat = None

    def get_dtype(self):
        return np.float32

    def compute(self, chunk, start_frame=0, end_frame=None, segment_index=0, max_margin=0, *args):
        # chunk shape: (num_frames, H, W, P)
        num_frames = chunk.shape[0]
        chunk_flat = chunk.reshape(num_frames, -1).astype(np.float32)  # (T, spatial)

        # Weighted fluorescence per ROI: (T, N)
        fluorescence = chunk_flat @ self._masks_flat.T

        # Neuropil subtraction, in the same L1/mean scale as self._masks_flat
        if self._neuropil_traces is not None:
            # Precomputed per-ROI trace (e.g. CNMF's b @ f projected through each ROI's own mask):
            # slice the rows belonging to this chunk. compute() is only handed max_margin, not
            # left_margin, so a margin-bearing chunk cannot be mapped to global rows unambiguously;
            # FluorescenceNode.get_margin() is 0 and it is the only node in this pipeline, so this
            # never fires today, but it fails loudly rather than misaligning silently.
            if max_margin:
                raise NotImplementedError(
                    f"Precomputed neuropil traces require a zero-margin pipeline (got max_margin={max_margin})"
                )
            assert self._epoch_offsets is not None  # set together with _neuropil_traces
            start = int(self._epoch_offsets[segment_index]) + start_frame
            neuropil_trace = self._neuropil_traces[start : start + num_frames]
            fluorescence -= self.neuropil_weight * neuropil_trace
        elif self._neuropil_flat is not None:
            # (T, 1) for global or (T, N) for per-ROI
            neuropil_trace = chunk_flat @ self._neuropil_flat.T
            fluorescence -= self.neuropil_weight * neuropil_trace

        # Rescale from L1 to the L2-normalized scale (see __init__) -- a no-op (factor 1) for
        # binary masks (Suite2pRois and generate_rois's default).
        fluorescence *= self._rescale_to_l2

        return (fluorescence,)


_NEUROPIL_PRIMARY_DATA_KEY = {"surround": "neuropil_masks", "cnmf": "neuropil_traces"}


class NeuropilExtension(AnalyzerExtension):
    """Extension to model the neuropil / background contaminating each ROI's fluorescence.

    Two methods are supported:

    - ``'surround'``: Suite2p-style neuropil mask -- the region surrounding each ROI (excluding
      pixels belonging to any ROI), rectangular by default or circular when ``circular=True``,
      via :func:`suite2p.extraction.masks.create_cell_pix`/:func:`~suite2p.extraction.masks.create_neuropil_masks`.
      Ring pixels are weighted ``1 / n_ring_pixels`` so that the weighted-sum matmul in
      :class:`FluorescenceNode` reproduces suite2p's own unweighted-mean ``Fneu`` convention.
      Works with *any* :class:`~photon_mosaic.core.baserois.BaseRois` -- per-ROI pixel
      coordinates are derived from ``rois.get_roi_image_masks()`` (not suite2p-specific stat
      data), since ``RoiAnalyzer`` always stores its own in-memory/on-disk snapshot of the ROIs
      rather than the original object passed to ``create_roi_analyzer`` (e.g. ``format="memory"``
      always copies into a plain ``NumpyRois``, so a `Suite2pRois`-specific accessor would not be
      reachable via ``roi_analyzer.rois`` in the common case). Multi-plane ROIs are supported as
      long as each ROI's own mask is confined to a single plane (e.g. well-separated mesoscope
      planes) -- each plane's ROIs are then treated as an independent 2D problem. A genuinely
      volumetric ROI spanning multiple planes is not yet supported (would need a true 3D
      "shell" neuropil mask, e.g. as in `Suite3D <https://www.biorxiv.org/content/10.1101/2025.03.26.645628v2.full>`_
      (`code <https://github.com/alihaydaroglu/suite3d>`_), rather than this per-plane approach).

    - ``'cnmf'``: CNMF-style low-rank background. Fits ``Y ~= C A.T + f b.T`` with the ROI
      footprints ``A`` held **fixed** at ``rois.get_roi_image_masks()``, giving ``gnb`` spatial
      background components ``b`` with their own timecourses ``f``, plus demixed traces ``C``.
      Unlike ``'surround'`` this needs no suite2p and no CaImAn -- only the masks and the movie --
      so it works on CaImAn and Suite2p ROIs alike, and (unlike a ring mask) it can represent
      background that fluctuates over time. Weighted and binary masks are both accepted.

      Computed data keys:

      ==========================  ==============================  ==========
      Key                         Shape                           Per-ROI?
      ==========================  ==============================  ==========
      ``background_spatial``      ``(gnb, Ly, Lx, n_planes)``      no
      ``background_temporal``     ``(gnb, n_frames)``              no
      ``neuropil_traces``         ``(n_frames, n_rois)``           yes
      ``demixed_fluorescence``    ``(n_frames, n_rois)``           yes
      ``epoch_frame_offsets``     ``(n_epochs + 1,)``              no
      ``fit_info``                dict of diagnostics              no
      ==========================  ==============================  ==========

      Frames are concatenated across epochs in epoch order, matching what
      :class:`FluorescenceExtension` gathers. ``background_spatial`` is dense (the components are
      spatially broad, so sparsity would not pay) and L2-normalised per component, with the scale
      pushed into ``background_temporal`` and components ordered by descending temporal energy --
      otherwise ``b`` and ``f`` would only be determined up to a per-component positive factor.

      Cost: two streaming passes over the movie, so roughly twice the read time of
      ``fluorescence``. Peak memory is dominated by two ``(n_frames, n_rois)`` arrays plus one
      chunk and the frame subsample, i.e. independent of the movie's total size.

      **The trace/background split has an exact gauge freedom.** For any ``alpha``,

      .. code-block:: text

          b -> b + A alpha        C -> C - f alpha.T

      leaves ``C A.T + f b.T`` *bit-for-bit* unchanged, so the data cannot distinguish them: what is
      not identifiable is precisely ``b`` restricted to ROI-support pixels, and the component of
      each trace lying along ``f``. Everything else is. In practice that means ``f``, the background
      away from ROIs, and each trace *after* the ``f`` direction is projected out are all recovered
      to within noise, while a trace's absolute offset and slow ``f``-shaped drift are not pinned
      down. Non-negativity narrows the family but does not collapse it. This is the classic CNMF
      neuropil/trace tradeoff -- this extension gives a far better-conditioned background estimate
      than a ring mask, but does not resolve the gauge, which is why ``neuropil_weight`` remains a
      user knob rather than being fixed at 1.

      A single ``b`` is shared across epochs, so it assumes the spatial background structure is
      constant across them (``f`` still absorbs per-frame amplitude changes).

    Once computed, this extension is picked up automatically by :class:`FluorescenceExtension`
    (see its ``use_neuropil``/``neuropil_weight`` params) -- just call
    ``roi_analyzer.compute("neuropil")`` before ``roi_analyzer.compute("fluorescence")``. Both
    methods feed the same ``F - neuropil_weight * Fneu`` subtraction on the same scale: ``'surround'``
    applies its ring mask to the movie inside :class:`FluorescenceNode`, while ``'cnmf'`` hands the
    node its precomputed ``neuropil_traces`` (the modelled background projected through each ROI's
    own L1-normalised mask) to slice per chunk.
    """

    extension_name = "neuropil"
    depend_on: list[str] = []
    # Deliberately False even though method='cnmf' reads the movie: flipping it would newly break
    # method='surround' on analyzers with no imaging attached, which works today by design. The
    # cnmf branch checks for imaging itself in _run.
    need_imaging = False
    use_nodepipeline = False
    need_job_kwargs = True

    def _set_params(
        self,
        method: str = "surround",
        inner_neuropil_radius: int = 2,
        min_neuropil_pixels: int = 350,
        circular: bool = False,
        lam_percentile: float = 50.0,
        gnb: int = 1,
        max_iter: int = 20,
        tol: float = 1e-4,
        highpass_sigma: float | None = 20.0,
        lowpass_sigma: float | None = 1.0,
        init_method: str = "ramp",
        nonneg_background: bool = True,
        nonneg_traces: bool = False,
        ridge: float = 1e-6,
        subsample_frames: int | None = 1000,
        **params: Any,
    ) -> dict[str, Any]:
        """Set parameters for neuropil mask computation.

        Parameters
        ----------
        method : str, optional
            Neuropil mask construction method. Only ``'surround'`` (Suite2p-style neighborhood mask,
            rectangular or circular) is currently supported. Default is ``'surround'``.
        inner_neuropil_radius : int, optional
            Pixels around each ROI to exclude before the ring starts. Only used with
            ``method='surround'``. Default is ``2``.
        min_neuropil_pixels : int, optional
            Minimum ring pixel count; the ring grows outward until this many pixels are found.
            Only used with ``method='surround'``. Default is ``350``.
        circular : bool, optional
            Restrict the ring to a circular region instead of a rectangular bounding-box grow.
            Only used with ``method='surround'``. Default is ``False``.
        lam_percentile : float, optional
            Percentile threshold used to decide which weighted pixels count as "ROI" pixels,
            excluded from every ROI's ring. Only used with ``method='surround'``. Default is
            ``50.0``.
        gnb : int, optional
            Number of low-rank background components to fit. Only used with ``method='cnmf'``.
            Default is ``1``, which also makes the fit deterministic and the non-negativity
            projection exact (the Gram is then 1x1); ``2``-``3`` suits spatially structured
            background.
        max_iter : int, optional
            Maximum alternating iterations for the background factors. Only used with
            ``method='cnmf'``. Default is ``20``.
        tol : float, optional
            Relative change in the fit objective below which the alternation stops. Only used with
            ``method='cnmf'``. Default is ``1e-4``.
        highpass_sigma : float or None, optional
            Standard deviation **in pixels** of the Gaussian subtracted from each frame to remove
            broad background before the initial trace estimate. ``None`` or ``0`` disables it. Only
            used with ``method='cnmf'``. Default is ``20.0`` (roughly 2-3 soma radii).
        lowpass_sigma : float or None, optional
            Standard deviation in pixels of a mild Gaussian smoothing applied before the high-pass,
            to suppress shot noise. ``None`` or ``0`` disables it. Only used with
            ``method='cnmf'``. Default is ``1.0``.
        init_method : str, optional
            How the background factors are initialised: ``'ramp'`` (spatial bumps over the mean
            residual image; dependency-free, the default) or ``'svd'`` (truncated SVD of the
            residual). Only used with ``method='cnmf'``.
        nonneg_background : bool, optional
            Constrain both background factors to be non-negative, as CaImAn does. Only used with
            ``method='cnmf'``. Default is ``True``.
        nonneg_traces : bool, optional
            Also constrain the demixed traces to be non-negative. Only used with
            ``method='cnmf'``. Default is ``False``, since a band-passed seed legitimately goes
            negative.
        ridge : float, optional
            Tikhonov regularisation applied to the least-squares solves, relative to each Gram
            matrix's mean diagonal. Only used with ``method='cnmf'``. Default is ``1e-6``.
        subsample_frames : int or None, optional
            Number of evenly-spaced frames used for the alternating background fit (the final
            traces and background timecourses are always solved over *all* frames). ``None`` uses
            every frame. Only used with ``method='cnmf'``. Default is ``1000``.
        """
        if params:
            raise TypeError(f"_set_params() got unexpected keyword argument(s): {sorted(params)}")
        if method == "cnmf":
            if gnb < 1:
                raise ValueError(f"gnb must be >= 1, got {gnb}")
            if init_method not in _CNMF_INIT_METHODS:
                raise ValueError(f"Unknown init_method: '{init_method}'. Supported: {_CNMF_INIT_METHODS}.")
            if subsample_frames is not None and subsample_frames < 2:
                raise ValueError(f"subsample_frames must be None or >= 2, got {subsample_frames}")
            if max_iter < 1:
                raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        return dict(
            method=method,
            inner_neuropil_radius=inner_neuropil_radius,
            min_neuropil_pixels=min_neuropil_pixels,
            circular=circular,
            lam_percentile=lam_percentile,
            gnb=gnb,
            max_iter=max_iter,
            tol=tol,
            highpass_sigma=highpass_sigma,
            lowpass_sigma=lowpass_sigma,
            init_method=init_method,
            nonneg_background=nonneg_background,
            nonneg_traces=nonneg_traces,
            ridge=ridge,
            subsample_frames=subsample_frames,
        )

    def _run(self, verbose: bool = False, **job_kwargs: Any) -> None:
        method = self.params["method"]
        rois = self.roi_analyzer.rois

        if method == "surround":
            masks = rois.get_roi_image_masks()
            self.data["neuropil_masks"] = _build_surround_neuropil_masks(
                masks,
                inner_neuropil_radius=self.params["inner_neuropil_radius"],
                min_neuropil_pixels=self.params["min_neuropil_pixels"],
                circular=self.params["circular"],
                lam_percentile=self.params["lam_percentile"],
            )
        elif method == "cnmf":
            # Checked here rather than via need_imaging -- see the class attributes above.
            if not (self.roi_analyzer.has_imaging() or self.roi_analyzer.has_temporary_imaging()):
                raise ValueError("NeuropilExtension(method='cnmf') requires the imaging")
            from spikeinterface.core.job_tools import ensure_chunk_size

            imaging = self.roi_analyzer.imaging
            job_kwargs = fix_job_kwargs(job_kwargs)
            chunk_size = ensure_chunk_size(imaging, **job_kwargs)
            self.data.update(
                _fit_cnmf_background(
                    imaging,
                    rois.get_roi_image_masks(),
                    gnb=self.params["gnb"],
                    max_iter=self.params["max_iter"],
                    tol=self.params["tol"],
                    highpass_sigma=self.params["highpass_sigma"],
                    lowpass_sigma=self.params["lowpass_sigma"],
                    init_method=self.params["init_method"],
                    nonneg_background=self.params["nonneg_background"],
                    nonneg_traces=self.params["nonneg_traces"],
                    ridge=self.params["ridge"],
                    subsample_frames=self.params["subsample_frames"],
                    chunk_size=chunk_size,
                    verbose=verbose,
                )
            )
        else:
            raise ValueError(f"Unknown method: '{method}'. Supported: 'surround', 'cnmf'.")

    def _get_data(self, key: str | None = None):
        """Return a computed neuropil result.

        Parameters
        ----------
        key : str or None, optional
            Which stored array to return. ``None`` (the default) returns the method's primary
            output: the ring masks for ``method='surround'``, the per-ROI background traces for
            ``method='cnmf'``. The other keys available for ``'cnmf'`` are
            ``'background_spatial'``, ``'background_temporal'``, ``'demixed_fluorescence'``,
            ``'epoch_frame_offsets'`` and ``'fit_info'``.

        Returns
        -------
        sparse.GCXS or np.ndarray or dict
            For ``method='surround'``, a ``sparse.GCXS`` of shape ``(n_rois, Ly, Lx)`` -- or
            ``(n_rois, Ly, Lx, n_planes)`` for multi-plane ROIs. Each ROI's ring pixels sum to 1.0
            (an unweighted mean over the ring, matching suite2p's own ``Fneu`` convention), except
            ROIs whose ring ended up empty (e.g. fully surrounded by other ROIs), which get an
            all-zero row. For ``method='cnmf'``, the requested array (or the ``fit_info`` dict).
        """
        if key is None:
            key = _NEUROPIL_PRIMARY_DATA_KEY[self.params.get("method", "surround")]
        if key not in self.data:
            raise KeyError(f"No '{key}' in neuropil data; available keys: {sorted(self.data)}")
        return self.data[key]

    def get_background(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the fitted ``(background_spatial, background_temporal)`` for ``method='cnmf'``.

        Returns
        -------
        tuple of np.ndarray
            ``b`` of shape ``(gnb, Ly, Lx, n_planes)`` and ``f`` of shape ``(gnb, n_frames)``. The
            modelled background movie for frame ``t`` is ``sum_k f[k, t] * b[k]``.
        """
        if self.params.get("method") != "cnmf":
            raise ValueError(f"get_background() is only available for method='cnmf', not '{self.params.get('method')}'")
        return self.data["background_spatial"], self.data["background_temporal"]

    def _select_extension_data(self, roi_ids):
        roi_indices = self.roi_analyzer.rois.ids_to_indices(roi_ids)
        if self.params.get("method", "surround") == "cnmf":
            # Every key must be returned: copy() assigns the result straight over `data`, so an
            # omitted key would be silently dropped. The background components are properties of the
            # whole field of view, so they pass through unsliced; the ROI-indexed traces are sliced
            # out of the full-problem solution rather than refitted on the ROI subset (same
            # semantics as slicing `fluorescence`).
            return {
                "background_spatial": self.data["background_spatial"],
                "background_temporal": self.data["background_temporal"],
                "epoch_frame_offsets": self.data["epoch_frame_offsets"],
                "fit_info": dict(self.data["fit_info"]),
                "neuropil_traces": self.data["neuropil_traces"][:, roi_indices],
                "demixed_fluorescence": self.data["demixed_fluorescence"][:, roi_indices],
            }
        return {"neuropil_masks": self.data["neuropil_masks"][roi_indices]}


class DfOverFExtension(AnalyzerExtension):
    """Extension to compute dF/F (relative fluorescence change) from fluorescence traces.

    dF/F is defined as ``(F - F0) / F0``, where ``F0`` is an estimate of the
    baseline fluorescence. Two baseline estimation methods are supported:

    - ``'maximin'``: Gaussian smoothing followed by a rolling minimum and
      maximum filter, as used in Suite2p. Robust to slow drift and does not
      require setting a percentile level.
    - ``'percentile'`` (alias ``'running_percentile'``): Rolling percentile
      filter, as used in CaImAn. The percentile level can be fixed
      (``prctile_baseline=<float>``) or estimated automatically per ROI via a
      DCT-based KDE of the fluorescence distribution (``prctile_baseline=None``).

    The fitted baseline itself is kept alongside the dF/F traces, accessible via
    ``self.data["f0"]`` (shape ``(n_frames, n_rois)``, matching ``self.data["df_over_f"]``).
    """

    extension_name = "df_over_f"
    depend_on: list[str] = ["fluorescence"]
    need_imaging = False
    need_job_kwargs = True

    def _set_params(
        self,
        method: str = "percentile",
        win_baseline: float = 60.0,
        sig_baseline: float = 10.0,
        prctile_baseline: float | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Set parameters for dF/F computation.

        Parameters
        ----------
        method : str, optional
            Baseline estimation method. One of ``'maximin'`` (Suite2p-style) or
            ``'percentile'`` / ``'running_percentile'`` (CaImAn-style).
            Default is ``'percentile'``.
        win_baseline : float, optional
            Duration of the sliding window in seconds used to estimate the
            baseline. Default is ``60.0``.
        sig_baseline : float, optional
            Standard deviation of the Gaussian filter (in frames) applied
            before the min/max filters. Only used with ``method='maximin'``.
            Default is ``10.0``.
        prctile_baseline : float or None, optional
            Percentile level (0–100) used for the rolling percentile baseline.
            Only used with ``method='percentile'``. If ``None``, the percentile
            is estimated automatically per ROI using a DCT-based KDE of the
            fluorescence distribution (CaImAn-style), falling back to the 50th
            percentile if estimation fails. If a float, that value is used
            directly for all ROIs. Default is ``None``.
        """
        return dict(
            method=method,
            win_baseline=win_baseline,
            sig_baseline=sig_baseline,
            prctile_baseline=prctile_baseline,
        )

    def _run(self, verbose: bool = False, **job_kwargs) -> None:
        F = self.roi_analyzer.get_extension("fluorescence").get_data()
        method = self.params["method"]
        if method == "maximin":  # maximin baseline estimation as in Suite2p
            from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

            fs = self.roi_analyzer.sampling_frequency
            win = int(self.params["win_baseline"] * fs)
            win += 1 if win % 2 == 0 else 0  # ensure odd window
            F0 = gaussian_filter1d(F, sigma=self.params["sig_baseline"], axis=0)
            F0 = minimum_filter1d(F0, size=win, axis=0)
            F0 = maximum_filter1d(F0, size=win, axis=0)
        elif method in ("percentile", "running_percentile"):  # running percentile baseline as in CaImAn
            from concurrent.futures import ProcessPoolExecutor

            win = int(self.params["win_baseline"] * self.roi_analyzer.sampling_frequency)
            n_jobs = fix_job_kwargs(job_kwargs).get("n_jobs", 1)
            prctile_baseline = self.params["prctile_baseline"]
            args = [(F[:, i].copy(), win, prctile_baseline) for i in range(F.shape[1])]
            if n_jobs == 1:
                cols = [_percentile_filter_roi(a) for a in args]
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                    cols = list(ex.map(_percentile_filter_roi, args))
            F0 = np.stack(cols, axis=1)
        else:
            raise ValueError(f"Unknown method: '{method}'. Supported: 'maximin', 'percentile'.")

        self.data["df_over_f"] = ((F - F0) / (F0 + np.finfo(np.float32).eps)).astype(np.float32)
        self.data["f0"] = F0.astype(np.float32)

    def _get_data(self, outputs="numpy"):
        """Return the computed dF/F traces.

        Parameters
        ----------
        outputs : str, optional
            Output format. ``'numpy'`` returns an ``ndarray`` of shape ``(n_frames, n_rois)``.
            ``'recording'`` wraps the traces in a :class:`~spikeinterface.core.NumpyRecording`.
            Default is ``'numpy'``.

        Returns
        -------
        np.ndarray or NumpyRecording
            dF/F traces in the requested format.
        """
        df_over_f_traces = self.data["df_over_f"]
        if outputs == "numpy":
            return df_over_f_traces
        elif outputs == "recording":
            from spikeinterface.core import NumpyRecording

            return NumpyRecording(
                df_over_f_traces,
                sampling_frequency=self.roi_analyzer.sampling_frequency,
                channel_ids=self.roi_analyzer.rois.roi_ids,
            )
        else:
            raise ValueError(f"Unsupported output type: {outputs}. Supported types are 'numpy' and 'recording'.")

    def _select_extension_data(self, roi_ids):
        roi_indices = self.roi_analyzer.rois.ids_to_indices(roi_ids)
        return {
            "df_over_f": self.data["df_over_f"][:, roi_indices],
            "f0": self.data["f0"][:, roi_indices],
        }


class DeconvolutionExtension(AnalyzerExtension):
    """Extension to deconvolve neural activity from dF/F traces using OASIS.

    Solves the noise-constrained sparse non-negative deconvolution problem
    (Friedrich, Zhou & Paninski, PLOS Comput Biol 2017) to infer, for each
    ROI, the most likely deconvolved activity trace and denoised calcium
    trace underlying its dF/F signal.
    """

    extension_name = "deconvolution"
    depend_on: list[str] = ["df_over_f"]
    need_imaging = False
    need_job_kwargs = True

    def _set_params(
        self,
        decay_time: float | None = None,
        rise_time: float | None = 0,
        baseline: float | None = None,
        baseline_nonneg: bool = False,
        penalty: int | None = 1,
        **params: Any,
    ) -> dict[str, Any]:
        """Set parameters for OASIS deconvolution.

        Parameters
        ----------
        decay_time : float or None, optional
            Decay time constant in seconds. If provided, sets the decay kinetics
            directly instead of estimating it per ROI. Default is None.
        rise_time : float or None, optional
            Rise time constant in seconds, modeling the calcium kinetics as a
            rise-and-decay (double exponential) process instead of
            decay-only. Default is 0 (no rise, decay-only). Set both
            `decay_time` and `rise_time` to None to auto-estimate a
            rise-and-decay model instead of a decay-only one.
        baseline : float or None, optional
            Fixed baseline value. Optimized per ROI if not given. Unused
            when `penalty` is None (treated as a fixed offset, default 0).
            Default is None.
        baseline_nonneg : bool, optional
            Enforce a strictly non-negative estimated baseline. Default is
            False, since the input is dF/F (already baseline-subtracted and
            free to fluctuate below zero), unlike raw fluorescence. Unused
            when `penalty` is None.
        penalty : int or None, optional
            Sparsity penalty: 1 for L1 (convex), 0 for L0. If None, skips
            the noise-constrained optimization entirely and deconvolves
            without imposing sparsity by default (`noise_std` and
            `baseline_nonneg` are unused in this mode). Default is 1.

        Advanced, undocumented keyword arguments are also accepted and
        passed through to :func:`oasis.functions.deconvolve` for power
        users: ``noise_std`` (overrides the per-ROI noise estimate that
        otherwise controls the sparsity weight), ``refine_kinetics``
        (number of large, isolated events used to refine the time
        constant(s) per ROI — can make estimates worse on noisy or
        low-event-count traces), and, only when `penalty` is None,
        ``lam`` (fixed sparsity weight, default 0) and ``s_min`` (minimal
        non-zero activity per bin, default 0).
        """
        lam = params.pop("lam", 0.0)
        s_min = params.pop("s_min", 0.0)
        noise_std = params.pop("noise_std", None)
        refine_kinetics = params.pop("refine_kinetics", 0)
        if params:
            raise TypeError(f"_set_params() got unexpected keyword argument(s): {sorted(params)}")
        return dict(
            decay_time=decay_time,
            rise_time=rise_time,
            baseline=baseline,
            baseline_nonneg=baseline_nonneg,
            penalty=penalty,
            lam=lam,
            s_min=s_min,
            noise_std=noise_std,
            refine_kinetics=refine_kinetics,
        )

    def _run(self, verbose: bool = False, **job_kwargs) -> None:
        from concurrent.futures import ProcessPoolExecutor

        dff = self.roi_analyzer.get_extension("df_over_f").get_data()
        fs = self.roi_analyzer.sampling_frequency
        n_jobs = fix_job_kwargs(job_kwargs).get("n_jobs", 1)

        # deconvolve() never mutates its input, and pickling a column slice (for the parallel
        # path below) serializes just that slice's own data, not the whole dff matrix.
        args = [(dff[:, i], fs, self.params) for i in range(dff.shape[1])]
        if n_jobs == 1:
            results = [_deconvolve_roi(a) for a in args]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                results = list(ex.map(_deconvolve_roi, args))

        self.data["denoised"] = np.stack([r[0] for r in results], axis=1).astype(np.float32)
        self.data["deconvolved"] = np.stack([r[1] for r in results], axis=1).astype(np.float32)

    def _get_data(self, outputs="numpy"):
        """Return the deconvolved activity trace.

        Parameters
        ----------
        outputs : str, optional
            Output format. ``'numpy'`` returns an ``ndarray`` of shape ``(n_frames, n_rois)``.
            ``'recording'`` wraps the traces in a :class:`~spikeinterface.core.NumpyRecording`.
            Default is ``'numpy'``.

        Returns
        -------
        np.ndarray or NumpyRecording
            Deconvolved activity trace in the requested format. The
            denoised calcium traces are available via
            ``self.data["denoised"]``.
        """
        deconvolved = self.data["deconvolved"]
        if outputs == "numpy":
            return deconvolved
        elif outputs == "recording":
            from spikeinterface.core import NumpyRecording

            return NumpyRecording(
                deconvolved,
                sampling_frequency=self.roi_analyzer.sampling_frequency,
                channel_ids=self.roi_analyzer.rois.roi_ids,
            )
        else:
            raise ValueError(f"Unsupported output type: {outputs}. Supported types are 'numpy' and 'recording'.")

    def _select_extension_data(self, roi_ids):
        roi_indices = self.roi_analyzer.rois.ids_to_indices(roi_ids)
        return {
            "denoised": self.data["denoised"][:, roi_indices],
            "deconvolved": self.data["deconvolved"][:, roi_indices],
        }


def _deconvolve_roi(args: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Run OASIS deconvolution on a single ROI's dF/F trace.

    Unpacks ``(y, framerate, params)`` — required as a module-level function
    so it can be pickled by :class:`~concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    args : tuple
        ``(y, framerate, params)`` where:

        - ``y`` : dF/F trace for one ROI, shape ``(n_frames,)``.
        - ``framerate`` : imaging sampling frequency in Hz.
        - ``params`` : dict with keys ``decay_time``, ``rise_time``,
          ``noise_std``, ``baseline``, ``baseline_nonneg``, ``penalty``,
          ``refine_kinetics``, ``lam``, ``s_min``
          (see :meth:`DeconvolutionExtension._set_params`).

    Returns
    -------
    c : np.ndarray
        Denoised calcium trace, shape ``(n_frames,)``.
    s : np.ndarray
        Deconvolved activity trace, shape ``(n_frames,)``.
    """
    from oasis.functions import deconvolve

    y, framerate, params = args
    kwargs = dict(
        framerate=framerate,
        sn=params["noise_std"],
        b=params["baseline"],
        b_nonneg=params["baseline_nonneg"],
        optimize_g=params["refine_kinetics"],
        penalty=params["penalty"],
    )
    if params["penalty"] is None:
        # lam/s_min are only accepted by oasisAR1/oasisAR2 (via deconvolve's penalty=None
        # path), not by constrained_oasisAR1/constrained_onnlsAR2 used for the default penalty.
        kwargs["lam"] = params["lam"]
        kwargs["s_min"] = params["s_min"]

    c, s, _b, _g, _lam = deconvolve(y, tau_d=params["decay_time"], tau_r=params["rise_time"], **kwargs)
    return c, s


def _percentile_filter_roi(args: tuple) -> np.ndarray:
    """Estimate baseline percentile (if needed) and apply a rolling percentile filter to one ROI.

    Unpacks ``(col, size, prctile_baseline)`` — required as a module-level
    function so it can be pickled by
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    args : tuple
        ``(col, size, prctile_baseline)`` where:

        - ``col`` : full fluorescence trace (float32 or float64).
        - ``size`` : rolling window length in frames.
        - ``prctile_baseline`` : fixed percentile (float) or ``None`` to trigger
          automatic KDE estimation from the first ``size`` frames of ``col``.
    """
    col, size, prctile_baseline = args
    if prctile_baseline is None:
        window = col if size >= len(col) else col[:size]
        try:
            prct = _kde_mode_percentile(window.astype(np.float64))
        except Exception:
            prct = 50.0
    else:
        prct = float(prctile_baseline)

    if size >= len(col):
        # Window covers the whole trace: skip scipy.ndimage's boundary-reflection
        # padding, whose behavior here isn't reliably reproducible across environments.
        baseline = np.percentile(col, prct)
        return np.full_like(col, baseline)

    from scipy.ndimage import percentile_filter

    return percentile_filter(col, prct, size=size)


def _kde_mode_percentile(data: np.ndarray, N: int = 2**12) -> float:
    """Return the percentile rank of the mode of `data` using a DCT-based KDE.

    Implements the bandwidth-selection method of Botev et al. (2010),
    mirroring CaImAn's ``caiman.utils.stats.kde`` / ``df_percentile``.

    Parameters
    ----------
    data : np.ndarray
        1-D array of fluorescence values (float64 recommended).
    N : int, optional
        Number of histogram bins and DCT coefficients. Must be a power of 2
        for efficiency. Default is ``4096`` (``2**12``).

    Returns
    -------
    float
        Percentile rank (0–100) of the KDE mode within ``data``.

    Raises
    ------
    ValueError
        If the estimated percentile is NaN, negative, or ≥ 100.
    """
    from scipy import fftpack, optimize

    M = len(data)
    minimum, maximum = data.min(), data.max()
    R = maximum - minimum
    if R == 0:
        return 50.0
    MIN = minimum - R / 10
    MAX = maximum + R / 10
    R = MAX - MIN

    # Histogram → DCT
    hist, bins = np.histogram(data, bins=N, range=(MIN, MAX))
    hist = hist / M
    dct_data = fftpack.dct(hist, norm=None)

    i_sq = np.arange(1, N, dtype=np.float64) ** 2
    sq = (dct_data[1:] / 2) ** 2

    def fixed_point(t):
        ell = 7
        f = 2 * np.pi ** (2 * ell) * np.sum(i_sq**ell * sq * np.exp(-i_sq * np.pi**2 * t))
        for s in range(ell, 1, -1):
            K0 = np.prod(np.arange(1, 2 * s, 2, dtype=np.float64)) / np.sqrt(2 * np.pi)
            const = (1 + (0.5) ** (s + 0.5)) / 3
            time = (2 * const * K0 / M / f) ** (2 / (3 + 2 * s))
            f = 2 * np.pi ** (2 * s) * np.sum(i_sq**s * sq * np.exp(-i_sq * np.pi**2 * time))
        return t - (2 * M * np.sqrt(np.pi) * f) ** (-2 / 5)

    t_star = optimize.brentq(fixed_point, 0, 0.1)

    # Smooth DCT coefficients and invert
    smooth = dct_data * np.exp(-(np.arange(N, dtype=np.float64) ** 2) * np.pi**2 * t_star / 2)
    density = fftpack.idct(smooth, norm=None) * N / R

    mesh = (bins[:-1] + bins[1:]) / 2
    density = density / np.trapezoid(density, mesh)
    cdf = np.cumsum(density) * (mesh[1] - mesh[0])

    return float(cdf[np.argmax(density)] * 100)


def _build_surround_neuropil_masks(
    masks,
    inner_neuropil_radius: int = 2,
    min_neuropil_pixels: int = 350,
    circular: bool = False,
    lam_percentile: float = 50.0,
):
    """Build Suite2p-style ("surround") neuropil masks from ROI image masks.

    For each ROI, builds the ring of pixels surrounding it (excluding pixels belonging to any
    ROI) via :mod:`suite2p.extraction.masks`, then converts the flattened-index ring into a
    mask where each ring pixel has weight ``1 / n_ring_pixels``. This weighting is required (not
    optional): :meth:`FluorescenceNode.compute` consumes this mask via
    ``chunk_flat @ neuropil_flat.T``, a *weighted sum*. A binary ring mask would instead compute
    a sum scaled by ring pixel count (typically >=350), which does not match suite2p's own
    ``Fneu = mean(movie[neuropil_ipix], axis=0)`` convention and would make ``neuropil_weight``
    uninterpretable.

    Per-ROI pixel coordinates and weights are derived directly from ``masks`` (each ROI's own
    nonzero entries), rather than requiring suite2p's raw stat dicts -- this makes ``'surround'``
    usable with any :class:`~photon_mosaic.core.baserois.BaseRois`, not only
    :class:`~photon_mosaic.extractors.Suite2pRois`. Each ROI's ``radius`` (needed by
    ``create_cell_pix``'s internal smoothing) is estimated from its pixel count assuming a
    roughly circular shape (``sqrt(n_pixels / pi)``); ``lam`` is taken from the mask's own
    values, falling back to uniform weights for all-zero/binary masks.

    Multi-plane input (``masks.ndim == 4``, shape ``(n_rois, Ly, Lx, n_planes)``) is supported
    as long as every ROI's own mask is confined to a single plane -- each plane's ROIs are then
    treated as an independent 2D problem via suite2p's (inherently 2D) mask functions, so a
    ring only ever excludes/competes with same-plane neighbors. This is correct for
    well-separated planes (e.g. mesoscope acquisitions where each ROI shows up in exactly one
    plane) but not for genuinely volumetric ROIs spanning multiple planes -- that would need a
    true 3D "shell" neuropil mask (e.g. as in
    `Suite3D <https://www.biorxiv.org/content/10.1101/2025.03.26.645628v2.full>`_, code at
    https://github.com/alihaydaroglu/suite3d) and raises ``NotImplementedError``.

    Parameters
    ----------
    masks : np.ndarray | sparse.SparseArray
        ROI image masks, shape ``(n_rois, Ly, Lx)`` or ``(n_rois, Ly, Lx, n_planes)`` (e.g.
        from ``BaseRois.get_roi_image_masks()``).
    inner_neuropil_radius, min_neuropil_pixels, circular, lam_percentile
        Passed through to suite2p's ``create_cell_pix``/``create_neuropil_masks``.

    Returns
    -------
    sparse.GCXS
        Same shape as ``masks``, dtype float32. Ring pixels sum to 1.0 per ROI; ROIs whose
        ring ended up empty get an all-zero row (no neuropil subtraction for that ROI).
    """
    import sparse

    try:
        from suite2p.extraction.masks import create_cell_pix, create_neuropil_masks
    except ImportError as e:
        raise ImportError(
            "NeuropilExtension(method='surround') requires suite2p. Install it with "
            "'pip install \"photon-mosaic[suite2p-registration]\"'."
        ) from e

    if masks.ndim == 3:
        n_rois, Ly, Lx = masks.shape
        n_planes = 1
    elif masks.ndim == 4:
        n_rois, Ly, Lx, n_planes = masks.shape
    else:
        raise ValueError(f"Expected masks with 3 or 4 dimensions (n_rois, Ly, Lx[, n_planes]), got {masks.shape}")

    out_shape = (0, Ly, Lx) if masks.ndim == 3 else (0, Ly, Lx, n_planes)
    if n_rois == 0:
        return sparse.GCXS.from_numpy(np.zeros(out_shape, dtype=np.float32), compressed_axes=(0,))

    # roi_planes[i] is only meaningful (and only used) when masks.ndim == 4.
    roi_planes = np.zeros(n_rois, dtype=int)
    stats = []
    for i in range(n_rois):
        roi_mask = masks[i]
        if isinstance(roi_mask, sparse.SparseArray):
            coo = roi_mask.tocoo()
            coords, data = coo.coords, np.asarray(coo.data, dtype=np.float64)
        else:
            nz = np.nonzero(roi_mask)
            coords, data = np.stack(nz), np.asarray(roi_mask[nz], dtype=np.float64)

        if masks.ndim == 4:
            ypix, xpix, plane_idx = coords[0], coords[1], coords[2]
            planes_present = np.unique(plane_idx)
            if len(planes_present) > 1:
                raise NotImplementedError(
                    f"ROI {i} spans multiple planes ({planes_present.tolist()}). "
                    "NeuropilExtension(method='surround') only supports ROIs confined to a "
                    "single plane each (e.g. well-separated mesoscope planes); a genuinely "
                    "volumetric neuropil mask (Suite3D-style 3D shell) is not yet implemented."
                )
            if len(planes_present):
                roi_planes[i] = int(planes_present[0])
        else:
            ypix, xpix = coords[0], coords[1]

        lam = data
        if len(ypix) == 0 or lam.sum() <= 0:
            lam = np.ones(len(ypix))
        radius = np.sqrt(len(ypix) / np.pi) if len(ypix) > 0 else 1.0
        stats.append({"ypix": ypix, "xpix": xpix, "lam": lam, "radius": radius})

    def _cell_pix_and_neuropil_ipix(stats_subset):
        cell_pix = create_cell_pix(stats_subset, Ly, Lx, lam_percentile=lam_percentile)
        return create_neuropil_masks(
            ypixs=[s["ypix"] for s in stats_subset],
            xpixs=[s["xpix"] for s in stats_subset],
            cell_pix=cell_pix,
            inner_neuropil_radius=inner_neuropil_radius,
            min_neuropil_pixels=min_neuropil_pixels,
            circular=circular,
        )

    # An all-zero ROI (no pixels) gets an all-zero ring directly: suite2p's rectangular (the
    # default, non-circular) growth path calls .min()/.max() on a ROI's own pixel coordinates
    # while extending it, which raises on an empty array. Such a ROI can never contribute any
    # cell pixels either, so simply excluding it from the suite2p calls below changes nothing
    # for the other ROIs.
    neuropil_ipix: list = [np.zeros(0, dtype=np.intp)] * n_rois
    if n_planes == 1:
        non_empty = [i for i in range(n_rois) if len(stats[i]["ypix"])]
        if non_empty:
            ipix = _cell_pix_and_neuropil_ipix([stats[i] for i in non_empty])
            for local_i, global_i in enumerate(non_empty):
                neuropil_ipix[global_i] = ipix[local_i]
    else:
        # Each plane's ROIs are solved as an independent 2D problem (see docstring): a ring
        # only excludes/competes with pixels from the same plane's other ROIs.
        for p in range(n_planes):
            roi_indices_p = [i for i in np.flatnonzero(roi_planes == p) if len(stats[i]["ypix"])]
            if len(roi_indices_p) == 0:
                continue
            ipix_p = _cell_pix_and_neuropil_ipix([stats[i] for i in roi_indices_p])
            for local_i, global_i in enumerate(roi_indices_p):
                neuropil_ipix[global_i] = ipix_p[local_i]

    ring_masks = []
    for i, ipix in enumerate(neuropil_ipix):
        ipix = np.asarray(ipix)
        n_pixels = len(ipix)
        shape = (Ly, Lx) if masks.ndim == 3 else (Ly, Lx, n_planes)
        if n_pixels == 0:
            ring_masks.append(
                sparse.COO(np.zeros((len(shape), 0), dtype=np.intp), np.zeros(0, dtype=np.float32), shape=shape)
            )
            continue
        ring_y, ring_x = np.unravel_index(ipix, (Ly, Lx))
        weights = np.full(n_pixels, 1.0 / n_pixels, dtype=np.float32)
        if masks.ndim == 3:
            coords = np.stack([ring_y, ring_x])
        else:
            plane_coord = np.full(n_pixels, roi_planes[i], dtype=np.intp)
            coords = np.stack([ring_y, ring_x, plane_coord])
        ring_masks.append(sparse.COO(coords, weights, shape=shape))

    return sparse.GCXS.from_coo(sparse.stack(ring_masks, axis=0), compressed_axes=(0,))


_CNMF_INIT_METHODS = ("ramp", "svd")


def _masks_to_sparse_matrix(masks):
    """Flatten ROI image masks into a ``scipy.sparse`` CSR matrix of shape ``(n_rois, n_pixels)``.

    Returns a ``scipy.sparse.csr_matrix``; scipy is imported lazily, hence the untyped signature.

    Accepts whatever ``BaseRois.get_roi_image_masks()`` returns (dense ndarray or
    :class:`sparse.SparseArray`) and keeps the mask *values* -- CNMF footprints are weighted, so
    binarizing here would change the model being fit.
    """
    import scipy.sparse as sp

    num_rois = masks.shape[0]
    flat = masks.reshape((num_rois, -1))
    if isinstance(flat, sparse.SparseArray):
        coo = flat.tocoo()
        rows, cols = coo.coords
        return sp.csr_matrix(
            (np.asarray(coo.data, dtype=np.float32), (rows, cols)),
            shape=(num_rois, flat.shape[1]),
        )
    return sp.csr_matrix(np.asarray(flat, dtype=np.float32))


def _spatial_bandpass(chunk, highpass_sigma, lowpass_sigma):
    """Difference-of-Gaussians spatial band-pass, applied per frame and per plane.

    ``chunk`` has shape ``(n_frames, Ly, Lx, n_planes)``; ``sigma=0`` on the frame and plane axes
    makes this a strictly 2D filter without a Python loop, so multi-plane input is handled for free.
    ``mode="nearest"`` rather than scipy's default ``"reflect"`` (and definitely not ``"constant"``,
    which darkens the field-of-view border and so manufactures a spurious high-pass ring).
    """
    from scipy.ndimage import gaussian_filter

    x = np.asarray(chunk, dtype=np.float32)
    low = x
    if lowpass_sigma:
        low = gaussian_filter(x, sigma=(0, lowpass_sigma, lowpass_sigma, 0), mode="nearest")
    if not highpass_sigma:
        return low
    return low - gaussian_filter(x, sigma=(0, highpass_sigma, highpass_sigma, 0), mode="nearest")


def _ridge_inverse(gram, ridge: float):
    """Symmetric (pseudo-)inverse of a Gram matrix with a per-entry-relative ridge.

    The ridge is scaled by each *diagonal entry* rather than by the mean diagonal. That distinction
    is load-bearing for the joint ``[A, b]`` Gram: a background component's diagonal is orders of
    magnitude larger than an ROI's, so a mean-scaled ridge would apply a huge relative penalty to
    the ROI block and visibly shrink the recovered traces.
    """
    from scipy.linalg import pinvh

    gram = np.asarray(gram, dtype=np.float64)
    n = gram.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    diagonal = np.diag(gram).astype(np.float64).copy()
    invalid = ~np.isfinite(diagonal) | (diagonal <= 0)
    if invalid.any():
        fallback = float(np.mean(diagonal[~invalid])) if (~invalid).any() else 1.0
        diagonal[invalid] = fallback if fallback > 0 else 1.0
    regularized = gram + ridge * np.diag(diagonal)
    try:
        return pinvh(regularized)
    except Exception:  # pragma: no cover - pinvh is very hard to make fail
        return np.linalg.pinv(regularized)


def _nnls_block(gram, rhs, x0, n_iter: int = 50):
    """``argmin_{X >= 0} 0.5*||X||^2_gram - <X, rhs>`` by projected gradient descent.

    Simply clipping the unconstrained minimiser ``rhs @ inv(gram)`` at zero is *not* the constrained
    minimiser unless ``gram`` is diagonal (i.e. exact only for ``gnb == 1``, where it is 1x1).
    Projected gradient with step ``1/||gram||_2`` is monotone, which is what keeps the reported
    objective non-increasing.
    """
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    x = np.maximum(np.asarray(x0, dtype=np.float64), 0.0)
    norm = np.linalg.norm(gram, 2) if gram.size else 0.0
    if not np.isfinite(norm) or norm <= 0:
        return x
    step = 1.0 / norm
    for _ in range(n_iter):
        x = np.maximum(x - step * (x @ gram - rhs), 0.0)
    return x


def _project_float64(data, factor, block: int = 512):
    """``data @ factor`` accumulated in float64, in bounded-memory row blocks.

    The movie is float32, but the fit objective is a difference of terms of order ``||Y||^2`` that
    cancel down to something ~1e7 times smaller. float32 projections leave absolute noise big
    enough to make the objective visibly non-monotone, so every reduction against the movie is done
    in float64 -- blocked, so the float64 upcast never materialises the whole array at once.
    """
    out = np.empty((data.shape[0], factor.shape[1]), dtype=np.float64)
    for i in range(0, data.shape[0], block):
        out[i : i + block] = data[i : i + block].astype(np.float64) @ factor
    return out


def _project_float64_transposed(data, factor, block: int = 512):
    """``data.T @ factor`` accumulated in float64, in bounded-memory row blocks of ``data``."""
    out = np.zeros((data.shape[1], factor.shape[1]), dtype=np.float64)
    for i in range(0, data.shape[0], block):
        out += data[i : i + block].astype(np.float64).T @ factor[i : i + block]
    return out


def _partition_of_unity(spatial_shape, gnb: int):
    """``(gnb, n_pixels)`` non-negative spatial weights summing to 1 at every pixel.

    Used to split a single broad background estimate into ``gnb`` spatially distinct, deterministic
    starting components (triangular bumps along the x axis). Deterministic matters: extension data is
    cached on disk and compared across runs, so an RNG-based init would make results irreproducible.
    """
    n_pixels = int(np.prod(spatial_shape))
    if gnb == 1:
        return np.ones((1, n_pixels), dtype=np.float64)

    height, width, num_planes = spatial_shape
    centers = np.linspace(0.0, width - 1, gnb)
    coords = np.arange(width, dtype=np.float64)
    bump_width = max((width - 1) / max(gnb - 1, 1), 1.0)
    bumps = np.maximum(1.0 - np.abs(coords[None, :] - centers[:, None]) / bump_width, 0.0)
    totals = bumps.sum(axis=0)
    totals[totals == 0] = 1.0
    bumps = bumps / totals
    weights = np.broadcast_to(bumps[:, None, :, None], (gnb, height, width, num_planes))
    return np.ascontiguousarray(weights.reshape(gnb, -1))


def _inpaint_roi_support(image, roi_support, spatial_shape, sigma):
    """Replace ``image`` on ROI pixels by a smooth interpolation of its off-ROI neighbourhood.

    This is what picks the initial point along the model's gauge freedom (``b -> b + A alpha``,
    ``C -> C - f alpha.T``, see :class:`NeuropilExtension`). The objective is *exactly* flat along
    that direction, so which member of the family the fit lands on is decided by the
    initialisation, not by the data -- and the seeded traces come from a band-passed movie, so they
    carry no baseline and the raw residual image hands every ROI's baseline to ``b``.

    Assuming instead that the background is spatially smooth *through* each footprint -- which is
    what "neuropil" physically means -- starts the fit at the sensible member of the family, where
    each ROI's baseline stays in its own trace.
    """
    from scipy.ndimage import gaussian_filter

    if not roi_support.any() or not sigma:
        return image

    shaped = image.reshape(spatial_shape)
    keep = (~roi_support).reshape(spatial_shape).astype(np.float64)
    blur = (0 if spatial_shape[0] == 1 else sigma, sigma, 0)
    numerator = gaussian_filter(shaped * keep, sigma=blur, mode="nearest")
    denominator = gaussian_filter(keep, sigma=blur, mode="nearest")
    filled = np.where(denominator > 1e-8, numerator / np.maximum(denominator, 1e-8), shaped)

    out = image.copy()
    out[roi_support] = filled.reshape(-1)[roi_support]
    return out


def _init_cnmf_background(y_sub, masks_csr, traces_sub, gnb, spatial_shape, init_method, nonneg, ridge, inpaint_sigma):
    """Deterministic non-negative initialisation of the background factors ``(b, f)``.

    Both methods start from the *mean residual image* ``mean(Y) - A mean(C)``, smoothed across the
    ROI footprints (see :func:`_inpaint_roi_support`, which is what fixes the gauge), since that is
    the broad structure left over once the seeded ROI traces are accounted for.

    - ``'ramp'``: split that image into ``gnb`` spatial bumps (see :func:`_partition_of_unity`).
      Dependency-free and the default.
    - ``'svd'``: rank-``gnb`` truncated SVD of the residual with an NNDSVD-style sign fix.
    """
    residual_mean = y_sub.mean(axis=0).astype(np.float64) - np.asarray(
        masks_csr.T @ traces_sub.mean(axis=0), dtype=np.float64
    )
    roi_support = np.asarray(abs(masks_csr).sum(axis=0), dtype=np.float64).ravel() > 0
    residual_mean = _inpaint_roi_support(residual_mean, roi_support, spatial_shape, inpaint_sigma)

    if init_method == "ramp":
        b = (np.maximum(residual_mean, 0.0)[None, :] * _partition_of_unity(spatial_shape, gnb)).T
    elif init_method == "svd":
        from scipy.sparse.linalg import svds

        residual = y_sub.astype(np.float64) - (masks_csr.T @ traces_sub.T).T
        residual[:, roi_support] = 0.0  # same gauge choice as 'ramp': ignore ROI pixels
        k = min(gnb, min(residual.shape) - 1)
        if k < 1:
            b = np.maximum(residual_mean, 0.0)[:, None] * np.ones((1, gnb))
        else:
            _, _, vt = svds(residual, k=k)
            components = np.abs(vt[::-1])  # svds returns ascending singular values
            b = np.zeros((residual.shape[1], gnb), dtype=np.float64)
            for i in range(gnb):
                b[:, i] = components[min(i, k - 1)]
    else:  # pragma: no cover - validated in _set_params
        raise ValueError(f"Unknown init_method: '{init_method}'. Supported: {_CNMF_INIT_METHODS}.")

    b = np.asarray(b, dtype=np.float64).reshape(-1, gnb)
    # A background component that initialises to all-zero can never move (every update is
    # multiplicative in b's own support), so fall back to a flat component.
    for k in range(gnb):
        if not np.any(b[:, k] > 0):
            b[:, k] = 1.0

    gram_b = b.T @ b
    rhs = _project_float64(y_sub, b) - traces_sub @ np.asarray(masks_csr @ b, dtype=np.float64)
    f = rhs @ _ridge_inverse(gram_b, ridge)
    if nonneg:
        f = _nnls_block(gram_b, rhs, f)
    return b, f


def _cnmf_objective(ynorm_sq, traces, temporal, proj_masks, proj_background, gram_aa, gram_ab, gram_bb):
    """``||Y - C A.T - f b.T||_F^2``, computed entirely from cached Gram matrices.

    Needs no access to the movie, so convergence can be monitored for free every iteration. This is
    also the cheapest possible check on the update algebra: any sign or transpose slip shows up as a
    non-monotone objective.
    """
    return float(
        ynorm_sq
        + np.sum((traces.T @ traces) * gram_aa)
        + np.sum((temporal.T @ temporal) * gram_bb)
        - 2.0 * np.sum(proj_masks * traces)
        - 2.0 * np.sum(proj_background * temporal)
        + 2.0 * np.sum((traces.T @ temporal) * gram_ab)
    )


def _iter_movie_chunks(imaging, chunk_size, epoch_offsets):
    """Yield ``(global_start, flat_chunk, shaped_chunk)`` over the whole movie, in gather order.

    Uses SpikeInterface's own :func:`divide_time_series_into_chunks` -- the *same* slice list
    ``run_node_pipeline`` consumes -- so the frame ordering of everything computed here matches
    ``FluorescenceExtension``'s output by construction rather than by coincidence.
    """
    from spikeinterface.core.job_tools import divide_time_series_into_chunks

    for epoch_index, start, stop in divide_time_series_into_chunks(imaging, chunk_size):
        chunk = np.asarray(
            imaging.get_series(start_frame=start, end_frame=stop, epoch_index=epoch_index),
            dtype=np.float32,
        )
        if chunk.ndim == 3:  # (frames, Ly, Lx) -> (frames, Ly, Lx, 1)
            chunk = chunk[..., np.newaxis]
        yield int(epoch_offsets[epoch_index]) + start, chunk.reshape(chunk.shape[0], -1), chunk


def _fit_cnmf_background(
    imaging,
    masks,
    *,
    gnb: int,
    max_iter: int,
    tol: float,
    highpass_sigma: float | None,
    lowpass_sigma: float | None,
    init_method: str,
    nonneg_background: bool,
    nonneg_traces: bool,
    ridge: float,
    subsample_frames: int | None,
    chunk_size: int | None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fit a low-rank CNMF background ``b f`` and demixed traces ``C`` against *fixed* footprints.

    Model: ``Y ~= C A.T + f b.T`` for ``Y`` ``(n_frames, n_pixels)``, ``A`` ``(n_pixels, n_rois)``
    taken as given from the ROI masks, ``C`` ``(n_frames, n_rois)``, ``b`` ``(n_pixels, gnb)`` and
    ``f`` ``(n_frames, gnb)``.

    Because ``A`` is fixed, ``P_A = Y @ A`` can be accumulated once and the ``C`` update then never
    touches the movie again. Stacking ``D = [A, b]`` and ``X = [C, f]``, the joint least-squares
    solution is closed-form, ``X = [P_A, P_b] @ inv(D.T D)``, so only ``b`` actually needs
    iterating -- and that is done on a deterministic frame subsample. The whole fit is therefore
    **two streaming passes** over the movie; refining ``b`` at full temporal resolution would need
    two more and is deliberately not done.

    Returns the extension data dict; see :class:`NeuropilExtension` for the keys.
    """
    import scipy.sparse as sp

    num_rois = int(masks.shape[0])
    spatial_shape = tuple(int(s) for s in masks.shape[1:])
    if len(spatial_shape) == 2:
        spatial_shape = spatial_shape + (1,)
    n_pixels = int(np.prod(spatial_shape))

    num_epochs = imaging.get_num_epochs()
    frames_per_epoch = [imaging.get_num_frames(epoch_index=i) for i in range(num_epochs)]
    epoch_offsets = np.concatenate(([0], np.cumsum(frames_per_epoch))).astype(np.int64)
    num_frames = int(epoch_offsets[-1])

    if num_rois == 0:
        empty_traces = np.zeros((num_frames, 0), dtype=np.float32)
        return {
            "background_spatial": np.zeros((gnb,) + spatial_shape, dtype=np.float32),
            "background_temporal": np.zeros((gnb, num_frames), dtype=np.float32),
            "neuropil_traces": empty_traces,
            "demixed_fluorescence": empty_traces,
            "epoch_frame_offsets": epoch_offsets,
            "fit_info": _cnmf_fit_info(gnb, 0, True, [], 0.0, 0.0, 0.0, 0.0, 0, init_method),
        }

    # float64 throughout: see _project_float64 on why float32 reductions are not good enough here.
    masks_csr = _masks_to_sparse_matrix(masks).astype(np.float64)  # (n_rois, n_pixels)
    gram_aa = np.asarray((masks_csr @ masks_csr.T).todense(), dtype=np.float64)
    gram_aa_inv = _ridge_inverse(gram_aa, ridge)

    # L1-normalised masks, matching FluorescenceNode's "mean over the footprint" convention, so the
    # neuropil trace produced here is on the same scale as the fluorescence it gets subtracted from.
    # Same all-zero-mask guard as FluorescenceNode._row_norm: a zero mask stays zero, not NaN.
    l1_norm = np.asarray(masks_csr.sum(axis=1), dtype=np.float64).ravel()
    l1_norm[l1_norm == 0] = 1.0
    masks_l1 = sp.diags(1.0 / l1_norm) @ masks_csr

    if chunk_size is None:
        chunk_size = max(num_frames, 1)

    # ---- Pass 1: P_A, the band-passed P_A for the seed, ||Y||^2, and the frame subsample --------
    if subsample_frames is None or subsample_frames >= num_frames:
        sub_rows = np.arange(num_frames)
    else:
        # Deterministic and evenly spread over the whole recording (never an RNG: extension data is
        # cached on disk and compared across runs, so the fit has to be reproducible).
        sub_rows = np.unique(np.round(np.linspace(0, num_frames - 1, subsample_frames)).astype(int))

    proj_masks = np.zeros((num_frames, num_rois), dtype=np.float64)
    proj_masks_hp = np.zeros((num_frames, num_rois), dtype=np.float64)
    y_sub = np.zeros((len(sub_rows), n_pixels), dtype=np.float32)
    ynorm_sq = 0.0

    for global_start, flat, shaped in _iter_movie_chunks(imaging, chunk_size, epoch_offsets):
        n = flat.shape[0]
        proj_masks[global_start : global_start + n] = (masks_csr @ flat.T).T
        highpassed = _spatial_bandpass(shaped, highpass_sigma, lowpass_sigma)
        proj_masks_hp[global_start : global_start + n] = (masks_csr @ highpassed.reshape(n, -1).T).T
        # float64 accumulation: over ~1e9 elements a float32 sum drifts far enough to break both the
        # objective's monotonicity and reproducibility across chunk sizes.
        ynorm_sq += float(np.einsum("ij,ij->", flat, flat, dtype=np.float64))
        in_chunk = (sub_rows >= global_start) & (sub_rows < global_start + n)
        if np.any(in_chunk):
            y_sub[np.flatnonzero(in_chunk)] = flat[sub_rows[in_chunk] - global_start]

    # ---- Alternating fit for b, on the subsample only (no further movie access) -----------------
    # The band-passed seed removes the broad background before the first trace estimate so it does
    # not leak into C. Note pinv(A) @ (L Y) is not strictly scale-consistent with
    # pinv(L A) @ (L Y); that is tolerated because this seed only forms the residual the b/f init
    # works from, and C is re-solved exactly in closed form at the end.
    traces_sub = proj_masks_hp[sub_rows] @ gram_aa_inv
    if nonneg_traces:
        traces_sub = np.maximum(traces_sub, 0.0)

    # A footprint-sized smoothing scale: wide enough to bridge a soma, narrow enough to keep real
    # background structure. Falls back to the high-pass scale when the masks are degenerate.
    inpaint_sigma = float(np.sqrt(max(gram_aa.diagonal().max(), 1.0) / np.pi)) + 1.0
    if highpass_sigma:
        inpaint_sigma = min(inpaint_sigma, float(highpass_sigma))
    background, temporal_sub = _init_cnmf_background(
        y_sub,
        masks_csr,
        traces_sub,
        gnb,
        spatial_shape,
        init_method,
        nonneg_background,
        ridge,
        inpaint_sigma,
    )

    proj_masks_sub = proj_masks[sub_rows]
    ysub_norm_sq = float(np.sum(y_sub.astype(np.float64) ** 2))
    objective: list[float] = []
    converged = False
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        gram_ab = np.asarray(masks_csr @ background, dtype=np.float64)

        traces_sub = (proj_masks_sub - temporal_sub @ gram_ab.T) @ gram_aa_inv
        if nonneg_traces:
            traces_sub = np.maximum(traces_sub, 0.0)

        gram_hh = temporal_sub.T @ temporal_sub
        rhs_b = _project_float64_transposed(y_sub, temporal_sub) - np.asarray(
            masks_csr.T @ (traces_sub.T @ temporal_sub), dtype=np.float64
        )
        background = (
            _nnls_block(gram_hh, rhs_b, background) if nonneg_background else rhs_b @ _ridge_inverse(gram_hh, ridge)
        )

        gram_ab = np.asarray(masks_csr @ background, dtype=np.float64)
        gram_bb = background.T @ background
        proj_background_sub = _project_float64(y_sub, background)
        rhs_f = proj_background_sub - traces_sub @ gram_ab
        temporal_sub = (
            _nnls_block(gram_bb, rhs_f, temporal_sub) if nonneg_background else rhs_f @ _ridge_inverse(gram_bb, ridge)
        )

        objective.append(
            _cnmf_objective(
                ysub_norm_sq,
                traces_sub,
                temporal_sub,
                proj_masks_sub,
                proj_background_sub,
                gram_aa,
                gram_ab,
                gram_bb,
            )
        )
        if verbose:
            print(f"neuropil(cnmf) iteration {n_iter}: objective={objective[-1]:.6g}")
        if len(objective) > 1:
            previous = objective[-2]
            denom = abs(previous) if previous else 1.0
            if abs(previous - objective[-1]) / denom < tol:
                converged = True
                break

    # ---- Pass 2: P_b = Y @ b, then the exact joint solve over all frames -----------------------
    proj_background = np.zeros((num_frames, gnb), dtype=np.float64)
    for global_start, flat, _ in _iter_movie_chunks(imaging, chunk_size, epoch_offsets):
        proj_background[global_start : global_start + flat.shape[0]] = _project_float64(flat, background)

    gram_ab = np.asarray(masks_csr @ background, dtype=np.float64)
    gram_bb = background.T @ background
    gram_joint = np.block([[gram_aa, gram_ab], [gram_ab.T, gram_bb]])
    joint = np.concatenate([proj_masks, proj_background], axis=1) @ _ridge_inverse(gram_joint, ridge)
    traces, temporal = joint[:, :num_rois], joint[:, num_rois:]

    if nonneg_background:
        temporal = _nnls_block(gram_bb, proj_background - traces @ gram_ab, temporal)
        traces = (proj_masks - temporal @ gram_ab.T) @ gram_aa_inv  # keep C optimal given f
    if nonneg_traces:
        traces = np.maximum(traces, 0.0)

    objective_full = _cnmf_objective(ynorm_sq, traces, temporal, proj_masks, proj_background, gram_aa, gram_ab, gram_bb)

    # ---- Pin the scale/permutation ambiguity so results are comparable across runs -------------
    # b f is invariant under b_k -> alpha_k b_k, f_k -> f_k / alpha_k for any alpha_k > 0, and under
    # permutation of k. Fix it: unit-L2 spatial components, ordered by descending temporal energy.
    # Purely cosmetic for the fit itself, which is why the objective is evaluated just above.
    for k in range(gnb):
        scale = float(np.linalg.norm(background[:, k]))
        if scale > 0:
            background[:, k] /= scale
            temporal[:, k] *= scale
    order = np.argsort(-np.linalg.norm(temporal, axis=0))
    background, temporal = background[:, order], temporal[:, order]

    cond_aa = float(np.linalg.cond(gram_aa))
    cond_joint = float(np.linalg.cond(gram_joint))
    if max(cond_aa, cond_joint) > 1e6:
        warnings.warn(
            f"Ill-conditioned CNMF design (cond(A.T A)={cond_aa:.3g}, cond(D.T D)={cond_joint:.3g}); "
            "heavily overlapping ROIs make the trace/background split poorly determined. Consider "
            "raising `ridge` or reducing `gnb`.",
            stacklevel=2,
        )

    # The modelled background movie is f b.T; projecting *it* through each ROI's own L1-normalised
    # mask is the drop-in replacement for the surround method's mask-times-movie neuropil trace:
    #   (f b.T) @ A_L1.T == f @ (A_L1 @ b).T
    neuropil_traces = temporal @ np.asarray(masks_l1 @ background, dtype=np.float64).T

    return {
        "background_spatial": np.ascontiguousarray(background.T.reshape((gnb,) + spatial_shape), dtype=np.float32),
        "background_temporal": np.ascontiguousarray(temporal.T, dtype=np.float32),
        "neuropil_traces": np.ascontiguousarray(neuropil_traces, dtype=np.float32),
        "demixed_fluorescence": np.ascontiguousarray(traces, dtype=np.float32),
        "epoch_frame_offsets": epoch_offsets,
        "fit_info": _cnmf_fit_info(
            gnb,
            n_iter,
            converged,
            objective,
            objective_full,
            cond_aa,
            cond_joint,
            ynorm_sq,
            len(sub_rows),
            init_method,
        ),
    }


def _cnmf_fit_info(
    gnb,
    n_iter,
    converged,
    objective,
    objective_full,
    cond_gram_masks,
    cond_gram_joint,
    ynorm_sq,
    subsample_frames,
    init_method,
) -> dict[str, Any]:
    """Diagnostics for a CNMF background fit.

    Everything is cast to a plain Python scalar: the zarr backend serialises this dict with
    ``numcodecs.JSON()`` and the ``binary_folder`` backend with ``check_json``, and neither accepts
    numpy scalars.
    """
    return {
        "gnb": int(gnb),
        "n_iter": int(n_iter),
        "converged": bool(converged),
        "objective": [float(v) for v in objective],
        "objective_full": float(objective_full),
        "cond_gram_masks": float(cond_gram_masks),
        "cond_gram_joint": float(cond_gram_joint),
        "ynorm_sq": float(ynorm_sq),
        "subsample_frames": int(subsample_frames),
        "init_method": str(init_method),
    }


register_result_extension(FluorescenceExtension)
register_result_extension(NeuropilExtension)
register_result_extension(DfOverFExtension)
register_result_extension(DeconvolutionExtension)
