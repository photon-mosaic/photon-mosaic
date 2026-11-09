import logging
import math
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from photon_mosaic.core import (
    BaseImaging,
    Motion,
    register_motion_class,
    register_registration_class,
    split_epoch_at_frames,
)
from photon_mosaic.core.selectplanes import select_planes

from .registration import RegisterImaging, RegisterImagingEpoch, RegistrationSettings


class Suite2pRegistrationSettings(RegistrationSettings):
    """Settings for Suite2P motion correction.

    Adds the Suite2P specific parameters on top of the shared
    :class:`photon_mosaic.preprocessing.registration.RegistrationSettings`
    (``debug``, ``tmp_dir``, ``batch_size``, ``device``).
    Values can be provided via constructor, environment variables, or .env file.
    """

    data_type: str = Field(default="h5", description="Processing h5 (default) or TIFF timeseries")
    do_registration: bool = Field(
        default=True,
        description="whether to register data (2 forces re-registration)",
    )
    align_by_chan: int = Field(
        default=1,
        description="when multi-channel, you can align by non-functional channel (1-based). "
        "Must be 1 for now: multi-channel imaging is not supported",
    )
    maxregshift: float = Field(
        default=0.1,
        description="max allowed registration shift, as a fraction of frame max(width and height)",
    )
    force_refImg: bool = Field(default=True, description="Force the use of an external reference image")
    nimg_init: int = Field(default=400, description="Number of evenly spaced frames used to build the reference image")
    do_bidiphase: bool = Field(
        default=False,
        description="Estimate the bidirectional scanning phase offset. Must be False: suite2p 1.x corrupts odd scan "
        "lines when the offset is not 0 (https://github.com/MouseLand/suite2p/issues/1265).",
    )
    bidiphase: int = Field(
        default=0,
        description="Bidirectional scanning phase offset in pixels. Must be 0, see do_bidiphase.",
    )
    norm_frames: bool = Field(default=True, description="Normalize frames when estimating shifts")
    spatial_taper: float = Field(default=3.45, description="Edge taper width in pixels; larger for 1P recordings")
    th_badframes: float = Field(
        default=1.0,
        description="Threshold for the frames excluded when computing the valid region; smaller excludes more",
    )
    block_size: list = Field(default_factory=lambda: [128, 128], description="Block size for non-rigid registration.")
    snr_thresh: float = Field(
        default=1.2,
        description="if any nonrigid block is below this threshold, it gets smoothed "
        "until above this threshold. 1.0 results in no smoothing",
    )
    outlier_detrend_window: float = Field(
        default=3.0,
        description="For outlier rejection in the xoff/yoff outputs of suite2p, the offsets are first de-trended "
        "with a median filter of this duration [seconds]. "
        "This value is ~30 or 90 samples in size for 11 and 31 Hz sampling rates respectively.",
    )
    outlier_maxregshift: float | None = Field(
        default=None,
        description="Units [fraction FOV dim]. After median-filter detrending, outliers more than this value are "
        "clipped to this value in x and y offset, independently. "
        "This is similar to Suite2P's internal maxregshift, but allows for low-frequency drift. "
        "A value of 0.05 is typically clipping outliers to "
        "512 * 0.05 = 25 pixels above or below the median trend. None (default) turns it off; rigid only.",
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
        description="Maximum number of iterations for creating a reference image. Must be 8: suite2p 1.x hard-codes it",
    )
    auto_remove_empty_frames: bool = Field(
        default=True,
        description="Not implemented yet: it does nothing. "
        "Automatically detect empty noise frames at the start and end of the movie. "
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
        ge=0,
        description="Number of frames to remove from the start of each epoch if known. "
        "Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still written to motion correction.",
    )
    trim_frames_end: int = Field(
        default=0,
        ge=0,
        description="Number of frames to remove from the end of each epoch if known. "
        "Removes frames from motion border calculation "
        "and resets the frame shifts found. Frames are still written to motion correction.",
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
        description="gaussian smoothing in time. Must be 0: suite2p 1.x crashes for values > 0 "
        "(https://github.com/MouseLand/suite2p/issues/1271).",
    )
    smooth_sigma: float = Field(
        default=1.15,
        description="~1 good for 2P recordings, recommend 3-5 for 1P recordings.",
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
    # suite2p options the backend does not support: each must keep its default
    subpixel: int = Field(default=10, description="Must be 10: suite2p 1.x non-rigid phase correlation assumes 10")
    two_step_registration: bool = Field(default=False, description="Not supported: must be False")
    model_config = ConfigDict(env_prefix="SUITE2P_REGISTRATION_", case_sensitive=False, env_file=".env")


def _compute_reference_wrapper(
    f_align_in: NDArray,
    settings: Suite2pRegistrationSettings,
    refImg: NDArray | None = None,
    device=None,
) -> tuple[NDArray, int]:
    """Compute the reference image for a single plane using Suite2p's algorithm.

    Parameters
    ----------
    f_align_in : NDArray
        Frames shaped ``(n_frames, Ly, Lx)``.
    settings : Suite2pRegistrationSettings
        Registration settings.
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
    compute_bidi = settings.do_bidiphase and settings.bidiphase == 0

    if refImg is None or compute_bidi:
        ix_frames = np.linspace(0, n_frames, 1 + min(settings.nimg_init, n_frames), dtype=int)[:-1]
        frames = f_align_in[ix_frames].copy()

    if compute_bidi:
        bidiphase = bidi.compute(frames)
        logging.info("Estimated bidiphase offset from data: %d pixels" % bidiphase)
    else:
        bidiphase = settings.bidiphase

    if bidiphase != 0 and refImg is None:
        frames = bidi.shift(frames, bidiphase)

    if refImg is None:
        t0 = time.time()
        refImg = compute_reference(frames, settings=settings.model_dump(), device=device)
        logging.info("Reference frame computed in %0.2f sec." % (time.time() - t0))

    return refImg, bidiphase


def _clip_outliers(offsets: NDArray, size: int, limit: int) -> NDArray:
    """Clip offsets to within ``limit`` pixels of their running median over ``size`` frames."""
    from scipy.ndimage import median_filter

    trend = median_filter(offsets, size, mode="nearest")
    return np.clip(offsets, trend - limit, trend + limit)


class Suite2PMotion(Motion):
    """Motion artifacts produced by Suite2P registration.

    Adds Suite2P-specific fields (``settings``, block coordinates, non-rigid offsets)
    on top of the algorithm-agnostic :class:`Motion` container.
    """

    method_name = "suite2p"
    settings_class = Suite2pRegistrationSettings

    def __init__(
        self,
        imaging: BaseImaging,
        displacements: Sequence[NDArray[np.floating[Any]]],
        settings: Suite2pRegistrationSettings | None = None,
        reference: Any = None,
        nonrigid_offsets: Sequence[Sequence[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]] | None]
        | Sequence[Sequence[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None]]
        | None = None,
        blocks: Sequence[Any] | None = None,
        yranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        xranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        corrected_badframes: Sequence[NDArray[np.bool_]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store Suite2P registration outputs.

        Parameters
        ----------
        imaging, displacements, reference, yranges, xranges, metadata
            See :class:`photon_mosaic.core.motion.Motion`.
        corrected_badframes : Sequence | None, optional
            Per-epoch bad-frame mask (see :class:`photon_mosaic.core.motion.Motion`).
            For Suite2P this is the union of any caller-supplied ``badframes``
            with frames whose registration shift exceeded ``maxregshift``,
            then combined across planes (a frame flagged on any plane is
            flagged for the whole volume).
        settings : Suite2pRegistrationSettings | None, optional
            The registration settings used; dumped to a dict only when saved.
        nonrigid_offsets : Sequence | None, optional
            Per-epoch, per-plane non-rigid offsets.
            Structure: ``[epoch][plane] -> (yoff1, xoff1)`` each ``(n_frames, n_blocks)``.
        blocks : Sequence | None, optional
            Suite2P block definitions when non-rigid registration is enabled.
        """
        super().__init__(
            imaging=imaging,
            displacements=displacements,
            reference=reference,
            yranges=yranges,
            xranges=xranges,
            corrected_badframes=corrected_badframes,
            metadata=metadata,
        )
        self.settings = settings if settings is not None else self.settings_class()
        self.nonrigid_offsets = nonrigid_offsets
        self.blocks = blocks

    def _to_dict_kwargs(self) -> dict[str, Any]:
        """Add the Suite2P-specific fields to :meth:`Motion._to_dict_kwargs`."""

        kwargs = super()._to_dict_kwargs()
        blocks = None if self.blocks is None else [_compact_blocks(b) for b in self.blocks]
        kwargs.update(settings=self.settings.model_dump(), nonrigid_offsets=self.nonrigid_offsets, blocks=blocks)
        return kwargs

    def patch_motion(self, epoch_index: int, plane_index: int) -> tuple[NDArray, tuple[NDArray, NDArray]] | None:
        """See :meth:`Motion.patch_motion`: each block's offset plus the rigid shift, at the block centres."""

        epoch_offsets = self.nonrigid_offsets[epoch_index] if self.nonrigid_offsets is not None else None
        if self.blocks is None or not epoch_offsets or epoch_offsets[plane_index] is None:
            return None
        yblock, xblock, (ny, nx) = self.blocks[plane_index][:3]
        rigid = self.displacements[epoch_index][:, plane_index]
        shifts = np.stack(epoch_offsets[plane_index], axis=-1).reshape(-1, ny, nx, 2) + rigid[:, None, None, :]
        return shifts, (np.mean(yblock[::nx], axis=1), np.mean(xblock[:nx], axis=1))

    @classmethod
    def _from_dict_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Rebuild the settings object and the upsampling matrix dropped by :meth:`_to_dict_kwargs`."""

        kwargs = super()._from_dict_kwargs(kwargs)
        kwargs["settings"] = cls.settings_class.model_validate(kwargs.get("settings") or {})
        if kwargs.get("blocks") is not None:
            kwargs["blocks"] = [_expand_blocks(b) for b in kwargs["blocks"]]
        return kwargs

    @classmethod
    def _compute(
        cls,
        imaging: BaseImaging,
        *,
        settings: Suite2pRegistrationSettings | None = None,
        badframes: NDArray | None = None,
        **params: Any,
    ) -> "Suite2PMotion":
        """Pre-compute Suite2P displacements for all planes and epochs.

        Computes the reference image and per-frame rigid (and optionally nonrigid)
        shifts without applying them to the data. Shifts are stored in the returned
        ``Motion`` object and applied lazily by ``RegisterSuite2PImagingEpoch``.

        Called through :meth:`photon_mosaic.core.Motion.compute` (or its
        ``compute_motion`` alias), which resolves ``settings`` before calling this.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging object containing one or more epochs/planes to be registered.
        settings : Suite2pRegistrationSettings | None, optional
            Registration settings; ``None`` uses the defaults.
        badframes : NDArray | None, optional
            Boolean array of shape ``(n_frames,)`` marking frames to exclude from
            reference image computation.
        **params : Any
            Settings overriding ``settings``, by field name.

        Returns
        -------
        Suite2PMotion
            Motion container with per-epoch displacements and Suite2P-specific
            fields (``settings``, ``blocks``, ``nonrigid_offsets``).
        """
        import torch
        from suite2p.registration.register import compute_crop, register_frames

        if "maxregshiftNR" in params:
            raise ValueError("'maxregshiftNR' is now 'max_nonrigid_shift'.")
        base = settings if settings is not None else cls.settings_class()
        settings = cls.settings_class(**{**base.model_dump(), **params})
        if settings.smooth_sigma_time > 0:
            raise ValueError("smooth_sigma_time > 0 crashes suite2p 1.x registration.")
        if settings.do_bidiphase or settings.bidiphase:
            raise ValueError("Bidiphase correction corrupts odd scan lines in suite2p 1.x (MouseLand/suite2p#1265).")
        unsupported = ["subpixel", "two_step_registration"]
        unsupported += ["align_by_chan", "max_reference_iterations"]
        if changed := [k for k in unsupported if getattr(settings, k) != cls.settings_class.model_fields[k].default]:
            raise ValueError(f"Not supported by the suite2p backend: {changed}. See each setting's description.")
        if settings.outlier_maxregshift is not None and settings.nonrigid:
            raise ValueError("outlier_maxregshift clips rigid shifts only: set nonrigid=False.")
        bounds = settings.bounds
        if bounds is not None and settings.nonrigid:
            raise ValueError(
                "'bounds' (an estimation crop) is supported by the suite2p backend for rigid registration only: "
                "its non-rigid block geometry is defined on the frame the shifts are applied to."
            )
        rows, cols = (slice(None), slice(None)) if bounds is None else (slice(*bounds[0]), slice(*bounds[1]))

        device = torch.device(settings.device)

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
        all_corrected_badframes: list[NDArray] = []

        trims = settings.trim_frames_start + settings.trim_frames_end
        if trims and trims >= min(imaging.get_num_samples(segment_index=e) for e in range(n_epochs)):
            raise ValueError("trim_frames_start + trim_frames_end must be shorter than every epoch.")
        for epoch_idx in range(n_epochs):
            n_frames = imaging.get_num_samples(segment_index=epoch_idx)
            Ly, Lx = imaging.shape[0], imaging.shape[1]

            badframes0 = np.zeros(n_frames, dtype=bool) if badframes is None else badframes.copy()
            # trimmed frames, at the start and end of each epoch, keep zero shift and leave the valid region
            start, end = settings.trim_frames_start, n_frames - settings.trim_frames_end
            badframes0[:start] = badframes0[end:] = True

            epoch_yoff: list[NDArray] = []
            epoch_xoff: list[NDArray] = []
            epoch_corrXY: list[NDArray] = []
            epoch_nr: list[tuple[NDArray, NDArray] | None] = []
            epoch_blocks: list[Any] = []

            # sub-epochs of batch_size frames: each is read whole, so one batch is in memory at a time
            batch_size = settings.batch_size
            batched = split_epoch_at_frames(imaging, epoch_idx, range(batch_size, n_frames, batch_size))

            for p, plane_id in enumerate(imaging.plane_ids):
                # Reference once, from the first epoch: read only the nimg_init evenly spaced frames suite2p
                # samples (the wrapper then uses all of them)
                if epoch_idx == 0:
                    picks = np.linspace(start, end, 1 + min(settings.nimg_init, end - start), dtype=int)[:-1]
                    plane = select_planes(imaging, [plane_id])
                    sample = np.concatenate([plane.get_series(i, i + 1, epoch_index=0) for i in picks])
                    sample = sample[:, rows, cols, 0].astype(np.float32)
                    refImgs[p], bidiphases[p] = _compute_reference_wrapper(sample, settings, refImg=None, device=device)

                # Compute shifts without applying them, one sub-epoch (= one suite2p batch) per call.
                # Only the offsets are kept per batch: the other returns include a mean image per call.
                # Select the plane after splitting: selection is applied by get_series, not by the epochs.
                plane_batches = select_planes(batched, [plane_id])
                batch_offsets = []
                for b in range(plane_batches.get_num_epochs()):
                    _, _, _, offsets, blocks, _, _, _ = register_frames(
                        plane_batches.get_series(epoch_index=b)[:, rows, cols, 0].astype(np.float32),
                        refImg=refImgs[p],
                        f_align_out=None,
                        batch_size=batch_size,
                        bidiphase=bidiphases[p],
                        norm_frames=settings.norm_frames,
                        smooth_sigma=settings.smooth_sigma,
                        spatial_taper=settings.spatial_taper,
                        block_size=settings.block_size,
                        nonrigid=settings.nonrigid,
                        maxregshift=settings.maxregshift,
                        smooth_sigma_time=settings.smooth_sigma_time,
                        snr_thresh=settings.snr_thresh,
                        maxregshiftNR=settings.max_nonrigid_shift,
                        device=device,
                        apply_shifts=False,
                    )
                    batch_offsets.append(offsets)
                # suite2p's eight offset arrays, concatenated across batches as register_frames does internally
                yoff, xoff, corrXY, yoff1, xoff1, corrXY1, zest, cmax_all = [
                    None if parts[0] is None else np.concatenate(parts, axis=0) for parts in zip(*batch_offsets)
                ]
                if settings.outlier_maxregshift is not None:
                    size = max(1, int(imaging.sampling_frequency * settings.outlier_detrend_window))
                    yoff = _clip_outliers(yoff, size, int(Ly * settings.outlier_maxregshift))
                    xoff = _clip_outliers(xoff, size, int(Lx * settings.outlier_maxregshift))
                for off in (yoff, xoff, yoff1, xoff1):
                    if off is not None:
                        off[:start] = off[end:] = 0

                epoch_yoff.append(yoff)
                epoch_xoff.append(xoff)
                epoch_corrXY.append(corrXY)
                epoch_nr.append((yoff1, xoff1) if yoff1 is not None else None)

                if epoch_idx == 0 and blocks is not None:
                    epoch_blocks.append(blocks)

            if epoch_idx == 0 and epoch_blocks:
                all_blocks = epoch_blocks

            # Stack planes → always (n_frames, n_planes, 2)
            disps = np.stack([np.stack(epoch_yoff, axis=1), np.stack(epoch_xoff, axis=1)], axis=-1)

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
                    th_badframes=settings.th_badframes,
                    badframes=badframes0.copy(),
                    maxregshift=settings.maxregshift,
                    Ly=Ly,
                    Lx=Lx,
                )
                epoch_yranges.append(tuple(yrange))
                epoch_xranges.append(tuple(xrange))
                epoch_cbf.append(bf)

            all_yranges.append(epoch_yranges)
            all_xranges.append(epoch_xranges)
            # Bad frames are a property of the time axis (a corrupted volume frame
            # is bad on all planes) — collapse Suite2P's per-plane masks into one.
            all_corrected_badframes.append(np.logical_or.reduce(epoch_cbf))

        nonrigid_offsets = all_nonrigid_offsets if any(o is not None for o in all_nonrigid_offsets) else None

        return cls(
            imaging=imaging,
            displacements=all_displacements,
            settings=settings,
            reference=refImgs,
            nonrigid_offsets=nonrigid_offsets,
            blocks=all_blocks,
            yranges=all_yranges,
            xranges=all_xranges,
            corrected_badframes=all_corrected_badframes,
        )


# suite2p's make_blocks returns (yblock, xblock, nblocks, block_size, NRsm, Kmat, nup). Kmat is a
# ((2*lpad+1)**2, nup**2) float32 tensor -- ~0.7 MB, ~11 MB as json, per plane -- but it is fully
# determined by lpad and subpixel, so serialize those two ints instead and recompute it on load.
_KMAT_INDEX = 5


def _compact_blocks(blocks: Any) -> Any:
    """Replace the ``Kmat`` of one plane's ``blocks`` by the ``mat_upsample`` arguments that produce it."""

    if not isinstance(blocks, tuple) or len(blocks) != 7:
        return blocks  # rigid registration stores no blocks for the plane
    kmat, nup = blocks[_KMAT_INDEX], blocks[6]
    lpad = (math.isqrt(kmat.shape[0]) - 1) // 2
    subpixel = (nup - 1) // (2 * lpad)
    return blocks[:_KMAT_INDEX] + ({"lpad": lpad, "subpixel": subpixel},) + blocks[_KMAT_INDEX + 1 :]


def _expand_blocks(blocks: Any) -> Any:
    """Inverse of :func:`_compact_blocks`."""

    if not isinstance(blocks, tuple) or len(blocks) != 7 or not isinstance(blocks[_KMAT_INDEX], dict):
        return blocks
    from suite2p.registration.utils import mat_upsample  # noqa: PLC0415

    kmat, _ = mat_upsample(**blocks[_KMAT_INDEX])
    return blocks[:_KMAT_INDEX] + (kmat,) + blocks[_KMAT_INDEX + 1 :]


class RegisterSuite2PImaging(RegisterImaging):
    """Apply pre-computed Suite2P motion correction on-the-fly.

    Thin alias of :class:`photon_mosaic.preprocessing.registration.RegisterImaging`,
    kept for discoverability and backwards compatibility; the generic class
    already dispatches on ``method``.
    """


class RegisterSuite2PImagingEpoch(RegisterImagingEpoch):
    """Epoch-level preprocessor that applies stored Suite2P displacements.

    Frame bounds, plane selection and output allocation come from
    :class:`~photon_mosaic.preprocessing.registration.RegisterImagingEpoch`; this class
    only shifts one plane's frames with Suite2P's ``shift_frames``.
    """

    def _correct_plane(
        self,
        plane_video: NDArray[np.floating[Any]],
        plane_index: int,
        start_frame: int,
        end_frame: int,
    ) -> NDArray[np.floating[Any]]:
        """Shift one plane's frames by the stored rigid (and non-rigid) offsets.

        Reads the motion through :meth:`Motion.patch_motion`, so it applies any backend's
        motion: suite2p rolls by whole pixels (another backend's rigid shifts are rounded)
        and leaves the rest to the block offsets.
        """

        import torch
        from suite2p.registration import register

        n_frames = end_frame - start_frame
        rigid = np.round(self.motion.displacements[self.epoch_index][start_frame:end_frame, plane_index])
        rigid = rigid.astype(np.int64)
        yoff1 = xoff1 = blocks = None
        patches = self.motion.patch_motion(self.epoch_index, plane_index)
        if patches is not None:
            shifts, (cy, cx) = patches
            residual = (shifts[start_frame:end_frame] - rigid[:, None, None, :]).astype(np.float32)
            yoff1, xoff1 = (torch.from_numpy(residual[..., i].reshape(n_frames, -1).copy()) for i in (0, 1))
            # suite2p's transform_data reads only each block's centre, the mean of its extent
            blocks = (
                [np.array([y, y]) for y in cy for _ in cx],
                [np.array([x, x]) for _ in cy for x in cx],
                [len(cy), len(cx)],
            )

        frames = register.shift_frames(
            torch.from_numpy(plane_video),
            torch.from_numpy(rigid[:, 0].copy()),
            torch.from_numpy(rigid[:, 1].copy()),
            yoff1,
            xoff1,
            blocks=blocks,
            device=torch.device("cpu"),
        )
        # another backend's motion carries no suite2p settings: apply it with the defaults
        settings = self.motion.settings if isinstance(self.motion, Suite2PMotion) else Suite2pRegistrationSettings()
        return np.clip(frames, 0, None) if settings.clip_negative else frames


register_motion_class(Suite2PMotion)
register_registration_class("suite2p", RegisterSuite2PImagingEpoch)
