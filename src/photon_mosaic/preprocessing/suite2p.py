import logging
import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from photon_mosaic.core import BaseImaging, BaseImagingEpoch

from .basepreprocessor import BasePreprocessor, BasePreprocessorEpoch
from .baseregistrationsettings import Suite2pRegistrationSettings


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
            Per-epoch, per-plane non-rigid offsets if available.
            Structure: ``[epoch][plane] -> (yoff1, xoff1)`` each ``(n_frames, n_blocks)``.
        blocks : Sequence | None, optional
            Suite2P block definitions when non-rigid registration is enabled.
        registration_outputs : dict | None, optional
            Additional outputs from the registration pipeline.
        """

        self.imaging = imaging
        self.displacements = displacements
        self.refAndMasks = refAndMasks
        self.ops = ops
        self.nonrigid_offsets = nonrigid_offsets
        self.blocks = blocks
        self.registration_outputs = registration_outputs

    @property
    def num_epochs(self) -> int:
        """Number of epochs represented in the stored displacements."""

        return len(self.displacements)


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
        ix_frames = np.linspace(
            0, n_frames, 1 + min(settings.get("nimg_init", 200), n_frames), dtype=int
        )[:-1]
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
    from suite2p.registration.register import default_settings, register_frames

    # Merge suite2p defaults → our settings → caller overrides
    ops = default_settings()
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

    for epoch_idx in range(n_epochs):
        epoch = imaging.epochs[epoch_idx]
        n_frames = imaging.get_num_samples(segment_index=epoch_idx)

        # Load entire epoch: (n_frames, H, W, n_planes) or (n_frames, H, W)
        all_frames = epoch.get_series(0, n_frames)

        epoch_yoff: list[NDArray] = []
        epoch_xoff: list[NDArray] = []
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
            epoch_nr.append((yoff1, xoff1) if yoff1 is not None else None)

            if epoch_idx == 0 and blocks is not None:
                epoch_blocks.append(blocks)

        if epoch_idx == 0 and epoch_blocks:
            all_blocks = epoch_blocks

        # Stack planes → (n_frames, n_planes, 2) or (n_frames, 2) for single-plane
        if n_planes > 1:
            disps = np.stack(
                [np.stack(epoch_yoff, axis=1), np.stack(epoch_xoff, axis=1)], axis=-1
            )  # (n_frames, n_planes, 2)
        else:
            disps = np.stack([epoch_yoff[0], epoch_xoff[0]], axis=-1)  # (n_frames, 2)

        all_displacements.append(disps)

        has_nonrigid = any(o is not None for o in epoch_nr)
        all_nonrigid_offsets.append(epoch_nr if has_nonrigid else None)

    nonrigid_offsets = (
        all_nonrigid_offsets
        if any(o is not None for o in all_nonrigid_offsets)
        else None
    )

    return Suite2PMotion(
        imaging=imaging,
        displacements=all_displacements,
        refAndMasks=refImgs,
        ops=ops,
        nonrigid_offsets=nonrigid_offsets,
        blocks=all_blocks,
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

            if disps.ndim == 3:
                yoff = disps[start_frame:end_frame, p, 0].astype(int)
                xoff = disps[start_frame:end_frame, p, 1].astype(int)
            else:
                yoff = disps[start_frame:end_frame, 0].astype(int)
                xoff = disps[start_frame:end_frame, 1].astype(int)

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
