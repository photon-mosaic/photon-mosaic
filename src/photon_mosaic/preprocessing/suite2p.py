from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from photon_mosaic.core import BaseImaging, BaseImagingEpoch

from .basepreprocessor import BasePreprocessor, BasePreprocessorEpoch
from .baseregistrationsettings import Suite2pRegistrationSettings

import torch


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
    ) -> None:
        """Store displacement fields and metadata produced by Suite2P.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging object associated with the computed motion.
        displacements : Sequence[NDArray]
            Per-epoch displacement arrays shaped ``(frames, planes, 2)`` or ``(frames, 2)``.
        refAndMasks : Any
            Suite2P reference images and masks returned by ``compute_reference_masks``.
        ops : dict[str, Any]
            Suite2P options used during registration.
        nonrigid_offsets : Sequence | None, optional
            Per-epoch, per-plane non-rigid offsets if available.
        blocks : Sequence | None, optional
            Suite2P block definitions when non-rigid registration is enabled.
        """

        self.imaging = imaging
        self.displacements = displacements
        self.refAndMasks = refAndMasks
        self.ops = ops
        self.nonrigid_offsets = nonrigid_offsets
        self.blocks = blocks

    @property
    def num_epochs(self) -> int:
        """Number of epochs represented in the stored displacements."""

        return len(self.displacements)

    def get_displacement_at_frames(
        self,
        frame_idx: int | NDArray[np.integer[Any]],
        epoch_index: int = 0,
        plane_index: int | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return displacements for selected frames and plane.

        Parameters
        ----------
        frame_idx : int | NDArray[np.integer]
            Frame index or indices to extract.
        epoch_index : int, optional
            Epoch to query. Defaults to 0.
        plane_index : int | None, optional
            Specific plane to slice when multi-plane displacements exist. If None, returns
            the full displacement slice for the requested frames.
        """

        disp = self.displacements[epoch_index]
        if plane_index is None:
            return disp[frame_idx]
        if disp.ndim == 2:
            return disp[frame_idx]
        return disp[frame_idx, plane_index, :]


def compute_motion_suite2p(
    imaging: BaseImaging,
    settings: Suite2pRegistrationSettings | dict[str, Any] | None = None,
    **kwargs: Any,
) -> Suite2PMotion:
    """Pre-compute Suite2P references and displacements for all planes/epochs.

    Parameters
    ----------
    imaging : BaseImaging
        Imaging object containing one or more epochs/planes to be registered.
    settings : Suite2pRegistrationSettings | dict | None, optional
        Registration settings. Dicts are validated into ``Suite2pRegistrationSettings``.
        If None, defaults are used. Additional keyword arguments override provided settings.
    **kwargs : Any
        Extra options forwarded to ``Suite2pRegistrationSettings``.

    Returns
    -------
    Suite2PMotion
        Motion container with per-epoch displacements and reference metadata.
    """

    # Import for suite2p 1.0.0.1+
    from suite2p.default_ops import default_ops as default_settings
    from suite2p.registration import register

    if settings is None:
        resolved_settings = Suite2pRegistrationSettings(**kwargs)
    elif isinstance(settings, dict):
        resolved_settings = Suite2pRegistrationSettings.model_validate({**settings, **kwargs})
    elif kwargs:
        resolved_settings = settings.model_copy(update=kwargs)
    else:
        resolved_settings = settings

    # Use settings for suite2p 1.0.0.1+
    ops = default_settings()
    ops.update(resolved_settings.model_dump(exclude={"debug", "tmp_dir", "data_type"}))

    num_planes = imaging.get_num_planes()
    num_epochs = len(imaging.epochs)

    all_refAndMasks = []
    plane_displacements = []
    plane_nonrigid = []
    blocks_per_plane = []

    def _register_single_plane(plane_idx: int):
        first_epoch = imaging.epochs[0]
        n_ref = min(resolved_settings.max_reference_iterations, first_epoch.get_num_samples())
        ref_frames = first_epoch.get_series(0, n_ref)
        if ref_frames.ndim == 4:
            ref_frames = ref_frames[:, :, :, plane_idx]
        # Compute reference image with device parameter for suite2p 1.0.0.1+
        device = torch.device("cpu")  # Using CPU for consistency, can be made configurable
        device_str = "cpu"  # String version for functions that need it
        reference = register.compute_reference(ref_frames, settings=ops, device=device)

        # suite2p 1.0.0.1+: use compute_filters_and_norm
        # Returns 9 values: added rmin, rmax at the end
        refAndMasks = register.compute_filters_and_norm(
            reference,
            norm_frames=ops.get("norm_frames", True),
            spatial_smooth=ops.get("smooth_sigma", 1.15),
            spatial_taper=ops.get("spatial_taper", 3 * ops.get("smooth_sigma", 1.15)),
            block_size=tuple(ops.get("block_size", [128, 128])),
            lpad=3,  # Default value from suite2p
            subpixel=10,  # Default value from suite2p
            device=device_str  # Use string for this function
        )

        displacements: list[NDArray[np.floating[Any]]] = []
        nonrigid_offsets_list: list[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None] = []

        for epoch in imaging.epochs:
            epoch_yoff = []
            epoch_xoff = []
            epoch_yoff1 = []
            epoch_xoff1 = []
            num_frames = epoch.get_num_samples()

            for start in range(0, num_frames, resolved_settings.batch_size):
                end = min(start + resolved_settings.batch_size, num_frames)
                batch = epoch.get_series(start, end)
                if batch.ndim == 4:
                    batch = batch[:, :, :, plane_idx]

                # suite2p 1.0.0.1+: use compute_shifts for computing displacements
                # Convert batch to torch tensor
                batch_torch = torch.from_numpy(batch.astype('float32')).to(device_str)

                # Call compute_shifts - it will unpack refAndMasks internally
                yoff, xoff, cmax, yoff1, xoff1, cmax1, zest, cmax_all = register.compute_shifts(
                    refAndMasks,
                    batch_torch,
                    maxregshift=ops.get("maxregshift", 0.1),
                    smooth_sigma_time=ops.get("smooth_sigma_time", 0),
                    snr_thresh=ops.get("snr_thresh", 1.2),
                    maxregshiftNR=ops.get("maxregshiftNR", 5),
                    nZ=1
                )

                # Convert torch tensors back to numpy
                yoff = yoff.cpu().numpy() if hasattr(yoff, 'cpu') else yoff
                xoff = xoff.cpu().numpy() if hasattr(xoff, 'cpu') else xoff
                if yoff1 is not None:
                    yoff1 = yoff1.cpu().numpy() if hasattr(yoff1, 'cpu') else yoff1
                    xoff1 = xoff1.cpu().numpy() if hasattr(xoff1, 'cpu') else xoff1

                epoch_yoff.append(yoff)
                epoch_xoff.append(xoff)
                if yoff1 is not None and xoff1 is not None:
                    epoch_yoff1.append(yoff1)
                    epoch_xoff1.append(xoff1)

            disp = np.column_stack([np.concatenate(epoch_yoff), np.concatenate(epoch_xoff)])
            displacements.append(disp)

            if epoch_yoff1:
                nonrigid_offsets_list.append((np.concatenate(epoch_yoff1), np.concatenate(epoch_xoff1)))
            else:
                nonrigid_offsets_list.append(None)

        blocks = refAndMasks[6]
        if all(item is None for item in nonrigid_offsets_list):
            nonrigid_offsets_result: list[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None] | None = (
                None
            )
        else:
            nonrigid_offsets_result = nonrigid_offsets_list

        return refAndMasks, displacements, nonrigid_offsets_result, blocks

    for plane_idx in range(num_planes):
        ref_masks, plane_disp, nonrigid, blocks = _register_single_plane(plane_idx)
        all_refAndMasks.append(ref_masks)
        plane_displacements.append(plane_disp)
        plane_nonrigid.append(nonrigid)
        blocks_per_plane.append(blocks)

    displacements = []
    has_nonrigid = any(nonrigid is not None for nonrigid in plane_nonrigid)
    nonrigid_offsets: list[list[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None]] | None
    if has_nonrigid:
        nonrigid_offsets = []
    else:
        nonrigid_offsets = None

    for epoch_idx in range(num_epochs):
        epoch_plane_disps = [plane_displacements[p][epoch_idx] for p in range(num_planes)]
        displacements_epoch = np.stack(epoch_plane_disps, axis=1)
        displacements.append(displacements_epoch)

        if has_nonrigid and nonrigid_offsets is not None:
            epoch_offsets: list[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None] = []
            for plane_idx in range(num_planes):
                plane_nr = plane_nonrigid[plane_idx]
                epoch_offsets.append(None if plane_nr is None else plane_nr[epoch_idx])
            nonrigid_offsets.append(epoch_offsets)

    return Suite2PMotion(
        imaging,
        displacements,
        all_refAndMasks,
        ops,
        nonrigid_offsets=nonrigid_offsets,
        blocks=blocks_per_plane,
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

        from suite2p.registration import register
        import torch

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

            # suite2p 1.0.0.1+: shift_frames with device parameter
            device_str = "cpu"  # Using CPU for consistency

            # Convert to torch tensors
            plane_video_torch = torch.from_numpy(plane_video).to(device_str)
            yoff_torch = torch.from_numpy(yoff.astype(np.int64)).to(device_str)
            xoff_torch = torch.from_numpy(xoff.astype(np.int64)).to(device_str)

            yoff1_torch = None
            xoff1_torch = None
            if yoff1 is not None:
                yoff1_torch = torch.from_numpy(yoff1).to(device_str)
                xoff1_torch = torch.from_numpy(xoff1).to(device_str)

            registered_plane = register.shift_frames(
                plane_video_torch,
                yoff_torch,
                xoff_torch,
                yoff1_torch,
                xoff1_torch,
                blocks=blocks,
                device=device_str
            )
            # shift_frames in 1.0.0.1+ returns numpy array, no conversion needed
            output_planes.append(registered_plane)

        return np.stack(output_planes, axis=-1)


# Convenience function for backwards compatibility
register_suite2p = RegisterSuite2PImaging
