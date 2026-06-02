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

        # Precompute flattened masks for efficient matrix multiplication
        masks = rois.get_roi_image_masks()  # (N, H, W) or (N, H, W, P)
        num_rois = masks.shape[0]
        self._masks_flat = masks.reshape(num_rois, -1).astype(np.float32)  # (N, spatial)

        # Precompute flattened neuropil masks
        if neuropil is not None:
            if neuropil.ndim == 2:
                # Global neuropil (H, W) -> (1, spatial)
                self._neuropil_flat = neuropil.reshape(1, -1).astype(np.float32)
            else:
                # Per-ROI neuropil (N, H, W) -> (N, spatial)
                self._neuropil_flat = neuropil.reshape(neuropil.shape[0], -1).astype(np.float32)
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
        if params:
            unexpected = ", ".join(sorted(params))
            raise TypeError(f"Unexpected parameter(s) for {self.__class__.__name__}: {unexpected}")

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

            fs = self.roi_analyzer.imaging.sampling_frequency
            win = int(self.params["win_baseline"] * fs)
            win += 1 if win % 2 == 0 else 0  # ensure odd window
            F0 = gaussian_filter1d(F, sigma=self.params["sig_baseline"], axis=0)
            F0 = minimum_filter1d(F0, size=win, axis=0)
            F0 = maximum_filter1d(F0, size=win, axis=0)
        elif method in ("percentile", "running_percentile"):  # running percentile baseline as in CaImAn
            from concurrent.futures import ProcessPoolExecutor

            win = int(self.params["win_baseline"] * self.roi_analyzer.imaging.sampling_frequency)
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
                sampling_frequency=self.roi_analyzer.imaging.sampling_frequency,
                channel_ids=self.roi_analyzer.rois.roi_ids,
            )
        else:
            raise ValueError(f"Unsupported output type: {outputs}. Supported types are 'numpy' and 'recording'.")

    def _select_extension_data(self, roi_ids):
        roi_indices = self.roi_analyzer.rois.ids_to_indices(roi_ids)
        return {"df_over_f": self.data["df_over_f"][:, roi_indices]}


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
    from scipy.ndimage import percentile_filter

    col, size, prctile_baseline = args
    if prctile_baseline is None:
        try:
            prct = _kde_mode_percentile(col[:size].astype(np.float64))
        except Exception:
            prct = 50.0
    else:
        prct = float(prctile_baseline)
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
