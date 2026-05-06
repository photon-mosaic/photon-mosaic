import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

from photon_mosaic.core import BaseImaging, BaseImagingEpoch

from .basepreprocessor import BasePreprocessor, BasePreprocessorEpoch


class Suite2pRegistrationSettings(BaseSettings):
    """Settings for Suite2P motion correction.

    This class defines all configuration parameters for motion correction using Suite2P.
    Values can be provided via constructor, environment variables, or .env file.
    """

    debug: bool = Field(default=False, description="Run with partial dataset")
    tmp_dir: str | Path = Field(
        default=Path("/scratch"),
        description="Directory into which to write temporary files produced by Suite2P",
    )
    data_type: str = Field(default="h5", description="Processing h5 (default) or TIFF timeseries")
    do_registration: bool = Field(
        default=True,
        description="whether to register data (2 forces re-registration)",
    )
    batch_size: int = Field(default=500, description="Number of frames per batch")
    align_by_chan: int = Field(
        default=1,
        description="when multi-channel, you can align by non-functional channel (1-based)",
    )
    maxregshift: float = Field(
        default=0.1,
        description="max allowed registration shift, as a fraction of "
        "frame max(width and height). This will be ignored if force_refImg is set to True",
    )
    force_refImg: bool = Field(default=True, description="Force the use of an external reference image")
    nonrigid: bool = Field(default=True, description="Whether to use non-rigid registration")
    block_size: list = Field(default_factory=lambda: [128, 128], description="Block size for non-rigid registration.")
    snr_thresh: float = Field(
        default=1.2,
        description="if any nonrigid block is below this threshold, it gets smoothed "
        "until above this threshold. 1.0 results in no smoothing",
    )
    maxregshiftNR: int = Field(
        default=5,
        description="maximum pixel shift allowed for nonrigid, relative to rigid",
    )
    outlier_detrend_window: float = Field(
        default=3.0,
        description="For outlier rejection in the xoff/yoff outputs of suite2p, the offsets are first de-trended "
        "with a median filter of this duration [seconds]. "
        "This value is ~30 or 90 samples in size for 11 and 31 Hz sampling rates respectively.",
    )
    outlier_maxregshift: float = Field(
        default=0.05,
        description="Units [fraction FOV dim]. After median-filter detrending, outliers more than this value are "
        "clipped to this value in x and y offset, independently. "
        "This is similar to Suite2P's internal maxregshift, but allows for low-frequency drift. "
        "Default value of 0.05 is typically clipping outliers to "
        "512 * 0.05 = 25 pixels above or below the median trend.",
    )
    clip_negative: bool = Field(
        default=False,
        description="Whether or not to clip negative pixel values in output. Because the pixel values "
        "in the raw movies are set by the current coming off a photomultiplier tube, there can "
        "be pixels with negative values (current has a sign), possibly due to noise in the rig. "
        "Some segmentation algorithms cannot handle negative values in the movie, so we have this "
        "option to artificially set those pixels to zero.",
    )
    max_reference_iterations: int = Field(
        default=8,
        description="Maximum number of iterations for creating a reference image",
    )
    auto_remove_empty_frames: bool = Field(
        default=True,
        description="Automatically detect empty noise frames at the start and end of the movie. "
        "Overrides values set in "
        "trim_frames_start and trim_frames_end. Some movies arrive with otherwise quality data but contain a set of "
        "frames that are empty and contain pure noise. When processed, these frames tend to receive "
        "large random shifts that throw off motion border calculation. Turning on this setting automatically "
        "detects these frames before processing and removes them from reference image creation, automated smoothing "
        "parameter searches, and finally the motion border calculation. The frames are still written however any "
        "shift estimated is removed and their shift is set to 0 to avoid large motion borders.",
    )
    trim_frames_start: int = Field(
        default=0,
        description="Number of frames to remove from the start of the movie if known. "
        "Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and trim_frames_start > 0",
    )
    trim_frames_end: int = Field(
        default=0,
        description="Number of frames to remove from the end of the movie if known. "
        "Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still written to motion correction. Raises an error if "
        "auto_remove_empty_frames is set and trim_frames_start > 0",
    )
    do_optimize_motion_params: bool = Field(
        default=False,
        description="Do a search for best parameters of smooth_sigma and smooth_sigma_time. "
        "Adds significant runtime cost to "
        "motion correction and should only be run once per experiment with the resulting parameters being stored "
        "for later use.",
    )
    smooth_sigma_time: int = Field(
        default=0,
        description="gaussian smoothing in time. If do_optimize_motion_params is set, this will be overridden",
    )
    smooth_sigma: float = Field(
        default=1.15,
        description="~1 good for 2P recordings, recommend 3-5 for 1P recordings. "
        "If do_optimize_motion_params is set, this will be overridden",
    )
    use_ave_image_as_reference: bool = Field(
        default=False,
        description="Only available if `do_optimize_motion_params` is set. "
        "After the a best set of smoothing parameters is found, "
        "use the resulting average image as the reference for the full registration. This can be used as two step "
        "registration by setting by setting smooth_sigma_min=smooth_sigma_max and "
        "smooth_sigma_time_min=smooth_sigma_time_max and steps=1.",
    )
    # Additional parameters that were hardcoded in the original code
    movie_lower_quantile: float = Field(
        default=0.1,
        description="Lower quantile threshold for avg projection histogram adjustment of movie",
    )
    movie_upper_quantile: float = Field(
        default=0.999,
        description="Upper quantile threshold for avg projection histogram adjustment of movie",
    )
    preview_frame_bin_seconds: float = Field(
        default=2.0,
        description="Before creating the webm, the movies will be averaged into bins of this many seconds",
    )
    preview_playback_factor: float = Field(
        default=10.0,
        description="The preview movie will playback at this factor times real-time",
    )
    n_batches: int = Field(
        default=20,
        description="Number of batches to load from the movie for smoothing parameter testing. "
        "Batches are evenly spaced throughout the movie.",
    )
    smooth_sigma_min: float = Field(
        default=0.65,
        description="Minimum value of the parameter search for smooth_sigma",
    )
    smooth_sigma_max: float = Field(
        default=2.15,
        description="Maximum value of the parameter search for smooth_sigma",
    )
    smooth_sigma_steps: int = Field(
        default=4,
        description="Number of steps to grid between smooth_sigma and smooth_sigma_max",
    )
    smooth_sigma_time_min: float = Field(
        default=0,
        description="Minimum value of the parameter search for smooth_sigma_time",
    )
    smooth_sigma_time_max: float = Field(
        default=6,
        description="Maximum value of the parameter search for smooth_sigma_time",
    )
    smooth_sigma_time_steps: int = Field(
        default=7,
        description="Number of steps to grid between smooth_sigma and smooth_sigma_time_max. "
        "Large values will add significant time to motion correction",
    )
    device: str = Field(
        default="cpu",
        description="Torch device for registration: 'cpu', 'cuda', or 'mps'.",
    )

    model_config = ConfigDict(env_prefix="SUITE2P_REGISTRATION_", case_sensitive=False, env_file=".env")


class Suite2PMotion:
    """Container for Suite2P motion correction artifacts."""

    def __init__(
        self,
        imaging: BaseImaging,
        displacements: Sequence[NDArray[np.floating[Any]]],
        refAndMasks: Any,
        ops: dict[str, Any],
        nonrigid_offsets: Sequence[Sequence[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]] | None]
        | Sequence[Sequence[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None]]
        | None = None,
        blocks: Sequence[Any] | None = None,
        yranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        xranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        corrected_badframes: Sequence[Sequence[NDArray[np.bool_]]] | None = None,
        registration_outputs: dict | None = None,
    ) -> None:
        """Store displacement fields and metadata produced by Suite2P.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging object associated with the computed motion.
        displacements : Sequence[NDArray]
            Per-epoch displacement arrays shaped ``(frames, planes, 2)`` or ``(frames, 2)``.
        refAndMasks : Any
            Suite2P reference images returned by ``_compute_reference_wrapper``.
        ops : dict[str, Any]
            Suite2P options used during registration.
        nonrigid_offsets : Sequence | None, optional
            Per-epoch, per-plane non-rigid offsets.
            Structure: ``[epoch][plane] -> (yoff1, xoff1)`` each ``(n_frames, n_blocks)``.
        blocks : Sequence | None, optional
            Suite2P block definitions when non-rigid registration is enabled.
        yranges : Sequence | None, optional
            Per-epoch, per-plane valid pixel row range ``[epoch][plane] -> (ymin, ymax)``.
        xranges : Sequence | None, optional
            Per-epoch, per-plane valid pixel column range ``[epoch][plane] -> (xmin, xmax)``.
        corrected_badframes : Sequence | None, optional
            Per-epoch, per-plane boolean bad-frame mask ``[epoch][plane] -> (n_frames,)``.
            Combines the input ``badframes`` with frames flagged by large registration shifts.
        registration_outputs : dict | None, optional
            Additional outputs from the registration pipeline.
        """

        self.imaging = imaging
        self.displacements = displacements
        self.refAndMasks = refAndMasks
        self.ops = ops
        self.nonrigid_offsets = nonrigid_offsets
        self.blocks = blocks
        self.yranges = yranges
        self.xranges = xranges
        self.corrected_badframes = corrected_badframes
        self.registration_outputs = registration_outputs

    @property
    def num_epochs(self) -> int:
        """Number of epochs represented in the stored displacements."""

        return len(self.displacements)

    def get_displacement_at_frames(
        self,
        frame_indices: int | NDArray[np.integer],
        plane_index: int | None = None,
        epoch_index: int = 0,
    ) -> NDArray[np.floating[Any]]:
        """Return displacement vectors for the requested frames.

        With ``plane_index=None`` the planes axis is preserved
        (``(..., n_planes, 2)``); with an integer ``plane_index`` it is dropped
        (``(..., 2)``).
        """

        disps = self.displacements[epoch_index]
        if plane_index is None:
            return disps[frame_indices]
        return disps[frame_indices, plane_index]


def _compute_reference_wrapper(
    f_align_in: NDArray,
    badframes: NDArray | None,
    settings: dict,
    refImg: NDArray | None = None,
    device=None,
) -> tuple[NDArray, int]:
    """Compute the reference image for a single plane using Suite2p's algorithm.

    Parameters
    ----------
    f_align_in : NDArray
        Frames shaped ``(n_frames, Ly, Lx)``.
    badframes : NDArray | None
        Boolean mask of bad frames, or None.
    settings : dict
        Suite2p settings dict (merged with suite2p defaults).
    refImg : NDArray | None, optional
        Pre-computed reference image. If provided, skips computation.
    device : torch.device | None, optional
        Torch device to use. Defaults to CPU.

    Returns
    -------
    refImg : NDArray
        Reference image shaped ``(Ly, Lx)``.
    bidiphase : int
        Bidirectional phase offset in pixels.
    """
    import suite2p.registration.bidiphase as bidi
    import torch
    from suite2p.registration.register import compute_reference

    if device is None:
        device = torch.device("cpu")

    n_frames = f_align_in.shape[0]
    compute_bidi = settings.get("do_bidiphase", False) and settings.get("bidiphase", 0) == 0

    if refImg is None or compute_bidi:
        ix_frames = np.linspace(0, n_frames, 1 + min(settings.get("nimg_init", 200), n_frames), dtype=int)[:-1]
        frames = f_align_in[ix_frames].copy()

    if compute_bidi:
        bidiphase = bidi.compute(frames)
        logging.info("Estimated bidiphase offset from data: %d pixels" % bidiphase)
    else:
        bidiphase = settings.get("bidiphase", 0)

    if bidiphase != 0 and refImg is None:
        frames = bidi.shift(frames, bidiphase)

    if refImg is None:
        t0 = time.time()
        refImg = compute_reference(frames, settings=settings, device=device)
        logging.info("Reference frame computed in %0.2f sec." % (time.time() - t0))

    return refImg, bidiphase


def compute_motion_suite2p(
    imaging: BaseImaging,
    settings: Suite2pRegistrationSettings | dict[str, Any] | None = None,
    badframes: NDArray | None = None,
    **kwargs: Any,
) -> Suite2PMotion:
    """Pre-compute Suite2P displacements for all planes and epochs.

    Computes the reference image and per-frame rigid (and optionally nonrigid)
    shifts without applying them to the data. Shifts are stored in the returned
    ``Suite2PMotion`` object and applied lazily by ``RegisterSuite2PImagingEpoch``.

    Parameters
    ----------
    imaging : BaseImaging
        Imaging object containing one or more epochs/planes to be registered.
    settings : Suite2pRegistrationSettings | dict | None, optional
        Registration settings. Dicts and None are coerced into
        ``Suite2pRegistrationSettings``. Extra keyword arguments override.
    badframes : NDArray | None, optional
        Boolean array of shape ``(n_frames,)`` marking frames to exclude from
        reference image computation.
    **kwargs : Any
        Extra options forwarded to ``Suite2pRegistrationSettings``.

    Returns
    -------
    Suite2PMotion
        Motion container with per-epoch displacements.
    """
    import torch
    from suite2p.registration.register import compute_crop, default_settings, register_frames

    # Merge suite2p defaults → our settings → caller overrides
    ops = default_settings()["registration"]
    if settings is None:
        user_settings = Suite2pRegistrationSettings()
    elif isinstance(settings, dict):
        user_settings = Suite2pRegistrationSettings(**settings)
    else:
        user_settings = settings
    ops.update(user_settings.model_dump())
    ops.update(kwargs)

    device = torch.device(ops.get("device", "cpu"))

    n_planes = imaging.num_planes
    n_epochs = imaging.get_num_epochs()

    # Reference images and bidiphase offsets computed once per plane from epoch 0
    refImgs: list[NDArray | None] = [None] * n_planes
    bidiphases: list[int] = [0] * n_planes
    # Block definitions are the same for all epochs (geometry doesn't change)
    all_blocks: list[Any] | None = None

    all_displacements: list[NDArray] = []
    all_nonrigid_offsets: list[list | None] = []
    all_yranges: list[list[tuple[int, int]]] = []
    all_xranges: list[list[tuple[int, int]]] = []
    all_corrected_badframes: list[list[NDArray]] = []

    for epoch_idx in range(n_epochs):
        epoch = imaging.epochs[epoch_idx]
        n_frames = imaging.get_num_samples(segment_index=epoch_idx)

        # Load entire epoch: (n_frames, H, W, n_planes) or (n_frames, H, W)
        all_frames = epoch.get_series(0, n_frames)
        Ly, Lx = imaging.shape[0], imaging.shape[1]

        badframes0 = np.zeros(n_frames, dtype=bool) if badframes is None else badframes.copy()

        epoch_yoff: list[NDArray] = []
        epoch_xoff: list[NDArray] = []
        epoch_corrXY: list[NDArray] = []
        epoch_nr: list[tuple[NDArray, NDArray] | None] = []
        epoch_blocks: list[Any] = []

        for p in range(n_planes):
            if all_frames.ndim == 4:
                plane_frames = all_frames[:, :, :, p].astype(np.float32)
            else:
                plane_frames = all_frames.astype(np.float32)

            # Compute reference once from the first epoch
            if epoch_idx == 0:
                refImgs[p], bidiphases[p] = _compute_reference_wrapper(
                    plane_frames, badframes, ops, refImg=None, device=device
                )

            # Compute shifts without applying them to the frames
            _, _, _, offsets_all, blocks = register_frames(
                plane_frames,
                refImg=refImgs[p],
                f_align_out=None,
                batch_size=ops["batch_size"],
                bidiphase=bidiphases[p],
                norm_frames=ops.get("norm_frames", True),
                smooth_sigma=ops["smooth_sigma"],
                spatial_taper=ops.get("spatial_taper", 3.45),
                block_size=ops["block_size"],
                nonrigid=ops["nonrigid"],
                maxregshift=ops["maxregshift"],
                smooth_sigma_time=ops["smooth_sigma_time"],
                snr_thresh=ops["snr_thresh"],
                maxregshiftNR=ops["maxregshiftNR"],
                device=device,
                apply_shifts=False,
            )
            yoff, xoff, corrXY, yoff1, xoff1, corrXY1, zest, cmax_all = offsets_all

            epoch_yoff.append(yoff)
            epoch_xoff.append(xoff)
            epoch_corrXY.append(corrXY)
            epoch_nr.append((yoff1, xoff1) if yoff1 is not None else None)

            if epoch_idx == 0 and blocks is not None:
                epoch_blocks.append(blocks)

        if epoch_idx == 0 and epoch_blocks:
            all_blocks = epoch_blocks

        # Stack planes → always (n_frames, n_planes, 2)
        disps = np.stack(
            [np.stack(epoch_yoff, axis=1), np.stack(epoch_xoff, axis=1)], axis=-1
        )

        all_displacements.append(disps)

        has_nonrigid = any(o is not None for o in epoch_nr)
        all_nonrigid_offsets.append(epoch_nr if has_nonrigid else None)

        # Compute valid FOV region and refined bad-frame mask for each plane
        epoch_yranges: list[tuple[int, int]] = []
        epoch_xranges: list[tuple[int, int]] = []
        epoch_cbf: list[NDArray] = []
        for p in range(n_planes):
            bf, yrange, xrange = compute_crop(
                xoff=epoch_xoff[p],
                yoff=epoch_yoff[p],
                corrXY=epoch_corrXY[p],
                th_badframes=ops.get("th_badframes", 1.0),
                badframes=badframes0.copy(),
                maxregshift=ops["maxregshift"],
                Ly=Ly,
                Lx=Lx,
            )
            epoch_yranges.append(tuple(yrange))
            epoch_xranges.append(tuple(xrange))
            epoch_cbf.append(bf)

        all_yranges.append(epoch_yranges)
        all_xranges.append(epoch_xranges)
        all_corrected_badframes.append(epoch_cbf)

    nonrigid_offsets = all_nonrigid_offsets if any(o is not None for o in all_nonrigid_offsets) else None

    return Suite2PMotion(
        imaging=imaging,
        displacements=all_displacements,
        refAndMasks=refImgs,
        ops=ops,
        nonrigid_offsets=nonrigid_offsets,
        blocks=all_blocks,
        yranges=all_yranges,
        xranges=all_xranges,
        corrected_badframes=all_corrected_badframes,
    )


class RegisterSuite2PImaging(BasePreprocessor):
    """Apply pre-computed Suite2P motion correction on-the-fly."""

    def __init__(self, imaging: BaseImaging, motion: Suite2PMotion, **kwargs: Any) -> None:
        """Build an imaging view that applies stored motion fields lazily."""
        BasePreprocessor.__init__(self, imaging)

        if motion.num_epochs != len(imaging.epochs):
            raise ValueError(
                f"Number of epochs in motion ({motion.num_epochs}) does not match imaging ({len(imaging.epochs)})"
            )

        for epoch_idx, parent_epoch in enumerate(imaging.epochs):
            epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, epoch_idx, **kwargs)
            self.add_epoch(epoch)

        self._kwargs = dict(imaging=imaging, motion=motion, **kwargs)


class RegisterSuite2PImagingEpoch(BasePreprocessorEpoch):
    """Epoch-level preprocessor that applies stored Suite2P displacements."""

    def __init__(
        self,
        parent_imaging_epoch: BaseImagingEpoch,
        motion: Suite2PMotion,
        epoch_index: int,
        **kwargs: Any,
    ) -> None:
        """Create an epoch preprocessor for a specific epoch and displacement set."""
        BasePreprocessorEpoch.__init__(self, parent_imaging_epoch)
        self.motion = motion
        self.epoch_index = epoch_index
        self.kwargs = kwargs

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: int | slice | Sequence[int] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return motion-corrected frames for the requested interval and planes."""

        import torch
        from suite2p.registration import register

        video = self.parent_imaging_epoch.get_series(start_frame, end_frame)
        num_planes = video.shape[3] if video.ndim == 4 else 1

        if plane_indices is None:
            planes_to_process = list(range(num_planes))
        elif isinstance(plane_indices, int):
            planes_to_process = [plane_indices]
        elif isinstance(plane_indices, slice):
            planes_to_process = list(range(*plane_indices.indices(num_planes)))
        else:
            planes_to_process = list(plane_indices)

        disps = self.motion.displacements[self.epoch_index]
        output_planes = []

        for p in planes_to_process:
            plane_video = video[:, :, :, p] if video.ndim == 4 else video
            plane_video = plane_video.astype("float32", copy=True)

            # Ensure plane_video is always 3D (frames, height, width)
            if plane_video.ndim == 2:
                plane_video = plane_video[np.newaxis, :, :]

            yoff = disps[start_frame:end_frame, p, 0].astype(int)
            xoff = disps[start_frame:end_frame, p, 1].astype(int)

            yoff1 = xoff1 = blocks = None
            if self.motion.nonrigid_offsets is not None:
                epoch_offsets = self.motion.nonrigid_offsets[self.epoch_index]
                if epoch_offsets is not None and p < len(epoch_offsets):
                    plane_offsets = epoch_offsets[p]
                    if plane_offsets is not None:
                        yoff1_full, xoff1_full = plane_offsets
                        yoff1 = yoff1_full[start_frame:end_frame]
                        xoff1 = xoff1_full[start_frame:end_frame]
                        if self.motion.blocks is not None:
                            blocks = self.motion.blocks[p]

            # Apply motion correction shifts — always CPU for on-the-fly application
            device = torch.device("cpu")

            plane_video_torch = torch.from_numpy(plane_video).to(device)
            yoff_torch = torch.from_numpy(yoff.astype(np.int64)).to(device)
            xoff_torch = torch.from_numpy(xoff.astype(np.int64)).to(device)

            yoff1_torch = None
            xoff1_torch = None
            if yoff1 is not None:
                yoff1_torch = torch.from_numpy(yoff1).to(device)
                xoff1_torch = torch.from_numpy(xoff1).to(device)

            registered_plane = register.shift_frames(
                plane_video_torch,
                yoff_torch,
                xoff_torch,
                yoff1_torch,
                xoff1_torch,
                blocks=blocks,
                device=device,
            )
            # Ensure registered_plane is always 3D (frames, height, width)
            if registered_plane.ndim == 2:
                registered_plane = registered_plane[np.newaxis, :, :]

            output_planes.append(registered_plane)

        return np.stack(output_planes, axis=-1)


register_suite2p = RegisterSuite2PImaging
