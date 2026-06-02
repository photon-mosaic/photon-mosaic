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
    """Extension to compute DfOverF from fluorescence traces."""

    extension_name = "df_over_f"
    depend_on: list[str] = ["fluorescence"]
    need_imaging = False
    need_job_kwargs = True  # potentially, to parallelize over chunks

    def _set_params(
        self,
        method="maximin",
        win_baseline=60.0,
        sig_baseline=10.0,
    ):
        return dict(
            method=method,
            win_baseline=win_baseline,
            sig_baseline=sig_baseline,
        )

    def _run(self, verbose=False, **job_kwargs):
        F = self.roi_analyzer.get_extension("fluorescence").get_data()
        # compute
        method = self.params["method"]
        if method == "maximin":  # maximin baseline estimation as in Suite2p
            from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
            fs = self.roi_analyzer.imaging.sampling_frequency
            win = int(self.params["win_baseline"] * fs)
            win += 1 if win % 2 == 0 else 0  # ensure odd window
            F0 = gaussian_filter1d(F, sigma=self.params["sig_baseline"], axis=0)
            F0 = minimum_filter1d(F0, size=win, axis=0)
            F0 = maximum_filter1d(F0, size=win, axis=0)
        else:
            raise ValueError(f"Unknown baseline_method: '{method}'. Supported: 'maximin'.")

        self.data["df_over_f"] = ((F - F0) / (F0 + np.finfo(np.float32).eps)).astype(np.float32)


    def _get_data(self, outputs="numpy"):
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


register_result_extension(DfOverFExtension)
