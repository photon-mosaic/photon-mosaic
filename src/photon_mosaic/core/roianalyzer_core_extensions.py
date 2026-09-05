from typing import Any

import numpy as np
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
    need_job_kwargs = True

    @classmethod
    def get_optional_dependencies(cls, **params):
        return ["neuropil"]

    def _set_params(self, use_neuropil=True, neuropil_weight=1.0):
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
        if self.params["use_neuropil"] and self.roi_analyzer.has_extension("neuropil"):
            neuropil = self.roi_analyzer.get_extension("neuropil").get_data()
        else:
            neuropil = None
        return [FluorescenceNode(self.roi_analyzer.imaging, self.roi_analyzer.rois, neuropil=neuropil)]

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
    def __init__(
        self,
        imaging: BaseImaging,
        rois: BaseRois,
        neuropil: np.ndarray | None = None,
        neuropil_weight: float = 0.7,
    ):
        """
        Pipeline node to extract fluorescence traces from ROIs, with optional neuropil subtraction.

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

        # Precompute flattened masks for efficient matrix multiplication. masks may be a
        # dense ndarray or a sparse array (see BaseRois.get_roi_image_masks) -- reshape must
        # be called with a single shape tuple for sparse arrays to accept it (unlike ndarray,
        # which also accepts unpacked dimensions); a tuple works for both.
        masks = rois.get_roi_image_masks()  # (N, H, W) or (N, H, W, P)
        num_rois = masks.shape[0]
        self._masks_flat = masks.reshape((num_rois, -1)).astype(np.float32)  # (N, spatial)

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

    def compute(self, chunk, *args):
        # chunk shape: (num_frames, H, W, P)
        num_frames = chunk.shape[0]
        chunk_flat = chunk.reshape(num_frames, -1).astype(np.float32)  # (T, spatial)

        # Weighted fluorescence per ROI: (T, N)
        fluorescence = chunk_flat @ self._masks_flat.T

        # Neuropil subtraction
        if self._neuropil_flat is not None:
            # (T, 1) for global or (T, N) for per-ROI
            neuropil_trace = chunk_flat @ self._neuropil_flat.T
            fluorescence -= self.neuropil_weight * neuropil_trace

        return (fluorescence,)


register_result_extension(FluorescenceExtension)


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
        return {"df_over_f": self.data["df_over_f"][:, roi_indices]}


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


register_result_extension(DfOverFExtension)


class NeuropilExtension(AnalyzerExtension):
    """Extension to compute neuropil masks for background/contamination subtraction.

    Currently one method is supported:

    - ``'halo'``: Suite2p-style ring/annulus neuropil mask. For each ROI, builds the ring of
      pixels surrounding it (excluding pixels belonging to any ROI), via
      :func:`suite2p.extraction.masks.create_cell_pix`/:func:`~suite2p.extraction.masks.create_neuropil_masks`.
      Ring pixels are weighted ``1 / n_ring_pixels`` so that the weighted-sum matmul in
      :class:`FluorescenceNode` reproduces suite2p's own unweighted-mean ``Fneu`` convention.
      Works with *any* :class:`~photon_mosaic.core.baserois.BaseRois` -- per-ROI pixel
      coordinates are derived from ``rois.get_roi_image_masks()`` (not suite2p-specific stat
      data), since ``RoiAnalyzer`` always stores its own in-memory/on-disk snapshot of the ROIs
      rather than the original object passed to ``create_roi_analyzer`` (e.g. ``format="memory"``
      always copies into a plain ``NumpyRois``, so a `Suite2pRois`-specific accessor would not be
      reachable via ``roi_analyzer.rois`` in the common case). Multi-plane ROIs are not yet
      supported.

    Once computed, this extension is picked up automatically by :class:`FluorescenceExtension`
    (see its ``use_neuropil``/``neuropil_weight`` params) -- just call
    ``roi_analyzer.compute("neuropil")`` before ``roi_analyzer.compute("fluorescence")``.
    """

    extension_name = "neuropil"
    depend_on: list[str] = []
    need_imaging = False
    use_nodepipeline = False
    need_job_kwargs = False

    def _set_params(
        self,
        method: str = "halo",
        inner_neuropil_radius: int = 2,
        min_neuropil_pixels: int = 350,
        circular: bool = False,
        lam_percentile: float = 50.0,
        **params: Any,
    ) -> dict[str, Any]:
        """Set parameters for neuropil mask computation.

        Parameters
        ----------
        method : str, optional
            Neuropil mask construction method. Only ``'halo'`` (Suite2p-style ring/annulus) is
            currently supported. Default is ``'halo'``.
        inner_neuropil_radius : int, optional
            Pixels around each ROI to exclude before the ring starts. Only used with
            ``method='halo'``. Default is ``2``.
        min_neuropil_pixels : int, optional
            Minimum ring pixel count; the ring grows outward until this many pixels are found.
            Only used with ``method='halo'``. Default is ``350``.
        circular : bool, optional
            Restrict the ring to a circular region instead of a rectangular bounding-box grow.
            Only used with ``method='halo'``. Default is ``False``.
        lam_percentile : float, optional
            Percentile threshold used to decide which weighted pixels count as "ROI" pixels,
            excluded from every ROI's ring. Only used with ``method='halo'``. Default is
            ``50.0``.
        """
        if params:
            raise TypeError(f"_set_params() got unexpected keyword argument(s): {sorted(params)}")
        return dict(
            method=method,
            inner_neuropil_radius=inner_neuropil_radius,
            min_neuropil_pixels=min_neuropil_pixels,
            circular=circular,
            lam_percentile=lam_percentile,
        )

    def _run(self, verbose: bool = False, **kwargs: Any) -> None:
        method = self.params["method"]
        rois = self.roi_analyzer.rois

        if rois.num_planes > 1:
            raise NotImplementedError(
                f"NeuropilExtension currently only supports single-plane ROIs "
                f"(got num_planes={rois.num_planes}). Multi-plane halo neuropil masks are not "
                "yet implemented."
            )

        if method == "halo":
            masks = rois.get_roi_image_masks()
            self.data["neuropil_masks"] = _build_halo_neuropil_masks(
                masks,
                inner_neuropil_radius=self.params["inner_neuropil_radius"],
                min_neuropil_pixels=self.params["min_neuropil_pixels"],
                circular=self.params["circular"],
                lam_percentile=self.params["lam_percentile"],
            )
        else:
            raise ValueError(f"Unknown method: '{method}'. Supported: 'halo'.")

    def _get_data(self):
        """Return the computed neuropil masks.

        Returns
        -------
        sparse.GCXS
            Shape ``(n_rois, Ly, Lx)``. Each ROI's ring pixels sum to 1.0 (an unweighted mean
            over the ring, matching suite2p's own ``Fneu`` convention), except ROIs whose ring
            ended up empty (e.g. fully surrounded by other ROIs), which get an all-zero row.
        """
        return self.data["neuropil_masks"]

    def _select_extension_data(self, roi_ids):
        roi_indices = self.roi_analyzer.rois.ids_to_indices(roi_ids)
        return {"neuropil_masks": self.data["neuropil_masks"][roi_indices]}


def _build_halo_neuropil_masks(
    masks,
    inner_neuropil_radius: int = 2,
    min_neuropil_pixels: int = 350,
    circular: bool = False,
    lam_percentile: float = 50.0,
):
    """Build Suite2p-style ring/annulus ("halo") neuropil masks from ROI image masks.

    For each ROI, builds the ring of pixels surrounding it (excluding pixels belonging to any
    ROI) via :mod:`suite2p.extraction.masks`, then converts the flattened-index ring into a
    mask where each ring pixel has weight ``1 / n_ring_pixels``. This weighting is required (not
    optional): :meth:`FluorescenceNode.compute` consumes this mask via
    ``chunk_flat @ neuropil_flat.T``, a *weighted sum*. A binary ring mask would instead compute
    a sum scaled by ring pixel count (typically >=350), which does not match suite2p's own
    ``Fneu = mean(movie[neuropil_ipix], axis=0)`` convention and would make ``neuropil_weight``
    uninterpretable.

    Per-ROI pixel coordinates and weights are derived directly from ``masks`` (each ROI's own
    nonzero entries), rather than requiring suite2p's raw stat dicts -- this makes ``'halo'``
    usable with any :class:`~photon_mosaic.core.baserois.BaseRois`, not only
    :class:`~photon_mosaic.extractors.Suite2pRois`. Each ROI's ``radius`` (needed by
    ``create_cell_pix``'s internal smoothing) is estimated from its pixel count assuming a
    roughly circular shape (``sqrt(n_pixels / pi)``); ``lam`` is taken from the mask's own
    values, falling back to uniform weights for all-zero/binary masks.

    Parameters
    ----------
    masks : np.ndarray | sparse.SparseArray
        ROI image masks, shape ``(n_rois, Ly, Lx)`` (e.g. from ``BaseRois.get_roi_image_masks()``).
    inner_neuropil_radius, min_neuropil_pixels, circular, lam_percentile
        Passed through to suite2p's ``create_cell_pix``/``create_neuropil_masks``.

    Returns
    -------
    sparse.GCXS
        Shape ``(n_rois, Ly, Lx)``, dtype float32. Ring pixels sum to 1.0 per ROI; ROIs whose
        ring ended up empty get an all-zero row (no neuropil subtraction for that ROI).
    """
    import sparse

    try:
        from suite2p.extraction.masks import create_cell_pix, create_neuropil_masks
    except ImportError as e:
        raise ImportError(
            "NeuropilExtension(method='halo') requires suite2p. Install it with "
            "'pip install \"photon-mosaic[suite2p-registration]\"'."
        ) from e

    n_rois, Ly, Lx = masks.shape
    if n_rois == 0:
        return sparse.GCXS.from_numpy(np.zeros((0, Ly, Lx), dtype=np.float32), compressed_axes=(0,))

    stats = []
    for i in range(n_rois):
        roi_mask = masks[i]
        if isinstance(roi_mask, sparse.SparseArray):
            coo = roi_mask.tocoo()
            ypix, xpix = coo.coords
            lam = np.asarray(coo.data, dtype=np.float64)
        else:
            ypix, xpix = np.nonzero(roi_mask)
            lam = np.asarray(roi_mask[ypix, xpix], dtype=np.float64)
        if len(ypix) == 0 or lam.sum() <= 0:
            lam = np.ones(len(ypix))
        radius = np.sqrt(len(ypix) / np.pi) if len(ypix) > 0 else 1.0
        stats.append({"ypix": ypix, "xpix": xpix, "lam": lam, "radius": radius})

    cell_pix = create_cell_pix(stats, Ly, Lx, lam_percentile=lam_percentile)
    neuropil_ipix = create_neuropil_masks(
        ypixs=[s["ypix"] for s in stats],
        xpixs=[s["xpix"] for s in stats],
        cell_pix=cell_pix,
        inner_neuropil_radius=inner_neuropil_radius,
        min_neuropil_pixels=min_neuropil_pixels,
        circular=circular,
    )

    ring_masks = []
    for ipix in neuropil_ipix:
        ipix = np.asarray(ipix)
        n_pixels = len(ipix)
        if n_pixels == 0:
            ring_masks.append(
                sparse.COO(np.zeros((2, 0), dtype=np.intp), np.zeros(0, dtype=np.float32), shape=(Ly, Lx))
            )
            continue
        ring_y, ring_x = np.unravel_index(ipix, (Ly, Lx))
        weights = np.full(n_pixels, 1.0 / n_pixels, dtype=np.float32)
        ring_masks.append(sparse.COO(np.stack([ring_y, ring_x]), weights, shape=(Ly, Lx)))

    return sparse.GCXS.from_coo(sparse.stack(ring_masks, axis=0), compressed_axes=(0,))


register_result_extension(NeuropilExtension)
register_result_extension(DeconvolutionExtension)
