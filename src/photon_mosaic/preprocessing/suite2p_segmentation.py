import logging
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from photon_mosaic.core import BaseImaging, BaseRois

from .baseregistrationsettings import BaseRegistrationSettings


class Suite2pSegmentationSettings(BaseRegistrationSettings):
    """Settings for Suite2P ROI detection. Passed directly to
    detection_wrapper"""

    diameter: list[float] = Field(
        default=[12.0, 12.0],
        description="Expected cell diameter in pixels [y, x]. 0 lets CellPose estimate it.",
    )
    tau: float = Field(
        default=1.0,
        description="Timescale of the indicator (seconds). GCaMP6s ~1.5, GCaMP6f ~0.7.",
    )
    fs: float = Field(
        default=30.0,
        description="Sampling frequency of the imaging data in Hz.",
    )
    preclassify: float = Field(
        default=0.0,
        description="Apply classifier before refinement with probability threshold. 0 disables.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device for detection: 'cpu', 'cuda', or 'mps'.",
    )

    algorithm: str = Field(
        default="sparsery",
        description=(
            "ROI detection algorithm. Options: 'sparsery' (suite2p sparse mode), "
            "'sourcery' (suite2p dense mode), 'cellpose' (CellPose)."
        ),
    )
    denoise: bool = Field(
        default=False,
        description="Apply denoising to binned movie before cell detection.",
    )
    threshold_scaling: float = Field(
        default=1.0,
        description="Scale the automatically determined detection threshold.",
    )
    max_overlap: float = Field(
        default=0.75,
        description="ROIs with more overlap than this are removed during triage.",
    )
    soma_crop: bool = Field(
        default=True,
        description="Crop dendrites for cell classification stats like compactness.",
    )
    allow_overlap: bool = Field(
        default=False,
        description="If False, overlapping pixels are discarded; if True, added to both ROIs.",
    )
    nbins: int = Field(
        default=5000,
        description="Number of bins for the activity histogram.",
    )
    highpass_time: int = Field(
        default=100,
        description="Window size (frames) for temporal high-pass filter before detection.",
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SUITE2P_SEGMENTATION_", case_sensitive=False, extra="ignore"
    )

    @field_validator("algorithm")
    @classmethod
    def lowercase_algorithm(cls, v: str) -> str:
        return v.lower()

    def to_detection_settings(self) -> dict[str, Any]:
        """Integrate ``settings`` dict expected by ``detection_wrapper``.

        Starts from the defaults of the installed suite2p version so any fields
        added in newer releases are preserved, then overlays the values defined here.
        """
        import inspect

        from suite2p.detection import detection_wrapper  # noqa: PLC0415

        defaults: dict[str, Any] = inspect.signature(detection_wrapper).parameters["settings"].default
        overrides = {
            "algorithm": self.algorithm,
            "denoise": self.denoise,
            "threshold_scaling": self.threshold_scaling,
            "max_overlap": self.max_overlap,
            "soma_crop": self.soma_crop,
            "nbins": self.nbins,
            "highpass_time": self.highpass_time,
        }
        return {**defaults, **overrides}


class Suite2pDetectedRois(BaseRois):
    """ROIs produced by running suite2p detection on a registered imaging object.

    Unlike :class:`~photon_mosaic.extractors.Suite2pRois`, this class stores the
    ``stat`` array in memory rather than loading it from a saved folder.
    """

    def __init__(
        self,
        stats: list[dict[str, Any]],
        shape: tuple[int, int, int],
        sampling_frequency: float,
        plane_assignments: NDArray[np.intp] | None = None,
    ) -> None:
        """Create ROIs from an in-memory suite2p ``stat`` list.

        Parameters
        ----------
        stats : list[dict]
            List of per-ROI stat dicts from ``detection_wrapper``.
            Each dict must contain ``ypix``, ``xpix``, and ``lam``.
        shape : tuple[int, int, int]
            Spatial shape ``(height, width, n_planes)``.
        sampling_frequency : float
            Imaging sampling rate in Hz.
        plane_assignments : NDArray | None, optional
            Integer plane index for each ROI. Required when ``n_planes > 1``.
        """
        roi_ids = np.arange(len(stats))
        BaseRois.__init__(self, sampling_frequency=sampling_frequency, shape=shape, roi_ids=roi_ids)

        self._stats = stats
        self._plane_assignments = (
            plane_assignments if plane_assignments is not None else np.zeros(len(stats), dtype=int)
        )

        # Expose scalar stat fields as properties
        skip = {"xpix", "ypix", "lam", "soma_crop", "overlap", "neuropil_mask"}
        if stats:
            for key in stats[0]:
                if key in skip:
                    continue
                values = [s[key] for s in stats]
                try:
                    self.set_property(key, values)
                except Exception:
                    pass

        self._kwargs = dict(
            stats=stats,
            shape=shape,
            sampling_frequency=sampling_frequency,
            plane_assignments=plane_assignments,
        )

    def get_roi_image_masks(self, roi_ids: list[int] | None = None) -> NDArray:
        """Return binary image masks shaped ``(n_rois, H, W)`` or ``(n_rois, H, W, n_planes)``."""
        if roi_ids is None:
            roi_ids = self.roi_ids.tolist()

        H, W, n_planes = self.shape
        masks = []
        for roi_id in roi_ids:
            stat = self._stats[roi_id]
            if n_planes == 1:
                mask = np.zeros((H, W), dtype=bool)
                mask[stat["ypix"], stat["xpix"]] = True
            else:
                mask = np.zeros((H, W, n_planes), dtype=bool)
                p = int(self._plane_assignments[roi_id])
                mask[stat["ypix"], stat["xpix"], p] = True
            masks.append(mask)
        return np.array(masks)


def detect_rois_suite2p(
    imaging: BaseImaging,
    yrange: Sequence[int] | None = None,
    xrange: Sequence[int] | None = None,
    badframes: NDArray[np.bool_] | None = None,
    settings: Suite2pSegmentationSettings | dict[str, Any] | None = None,
) -> Suite2pDetectedRois:
    """Detect ROIs using suite2p's detection pipeline on a registered imaging object.

    This function is intentionally decoupled from :class:`Suite2PMotion`.
    ``yrange``, ``xrange``, and ``badframes`` can be sourced from anywhere —
    e.g. ``motion.yranges[epoch][plane]`` or computed independently.

    Parameters
    ----------
    imaging : BaseImaging
        Motion-corrected imaging object. All epochs are averaged to build the
        activity movie passed to ``detection_wrapper``.
    yrange : sequence of int | None, optional
        Valid pixel row range ``[ymin, ymax]`` from motion correction.
        If None, the full image height is used.
    xrange : sequence of int | None, optional
        Valid pixel column range ``[xmin, xmax]`` from motion correction.
        If None, the full image width is used.
    badframes : NDArray[bool] | None, optional
        Boolean mask ``(n_frames,)`` of frames to exclude from detection.
    settings : Suite2pSegmentationSettings | dict | None, optional
        Detection settings. Dicts are coerced into ``Suite2pSegmentationSettings``.

    Returns
    -------
    Suite2pDetectedRois
        Detected ROIs with pixel masks and suite2p stat properties.
    """
    import torch
    from suite2p.detection import detection_wrapper

    if settings is None:
        cfg = Suite2pSegmentationSettings()
    elif isinstance(settings, dict):
        cfg = Suite2pSegmentationSettings(**settings)
    else:
        cfg = settings

    device = torch.device(cfg.device)
    H, W, n_planes = imaging.shape

    yrange = list(yrange) if yrange is not None else [0, H]
    xrange = list(xrange) if xrange is not None else [0, W]

    all_stats: list[dict[str, Any]] = []
    plane_assignments: list[int] = []

    for p in range(n_planes):
        logging.info(f"Suite2p detection — plane {p + 1}/{n_planes}")
        t0 = time.time()

        # Collect frames for this plane across all epochs
        plane_chunks = []
        for epoch_idx in range(imaging.get_num_epochs()):
            n_frames = imaging.get_num_samples(segment_index=epoch_idx)
            frames = imaging.epochs[epoch_idx].get_series(0, n_frames)
            plane_frames = frames[:, :, :, p] if frames.ndim == 4 else frames
            plane_chunks.append(plane_frames.astype(np.float32))
        f_reg = np.concatenate(plane_chunks, axis=0)  # (total_frames, H, W)

        # Align badframes length to concatenated frame count
        bf = None
        if badframes is not None:
            bf = badframes.astype(bool)
            if bf.shape[0] != f_reg.shape[0]:
                logging.warning(
                    f"badframes length {bf.shape[0]} != n_frames {f_reg.shape[0]}; ignoring"
                )
                bf = None

        diameter = np.array(cfg.diameter, dtype=float)

        _, stat, _ = detection_wrapper(
            f_reg,
            diameter=diameter,
            tau=cfg.tau,
            fs=cfg.fs,
            yrange=yrange,
            xrange=xrange,
            badframes=bf,
            settings=cfg.to_detection_settings(),
            device=device,
        )

        logging.info(f"  found {len(stat)} ROIs in {time.time() - t0:.1f}s")
        all_stats.extend(stat)
        plane_assignments.extend([p] * len(stat))

    return Suite2pDetectedRois(
        stats=all_stats,
        shape=(H, W, n_planes),
        sampling_frequency=imaging.sampling_frequency,
        plane_assignments=np.array(plane_assignments, dtype=int),
    )
