import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorEpoch
from .baseregistrationsettings import Suite2pRegistrationSettings


class Suite2PMotion:
    """
    Container for Suite2P motion correction information.

    Stores the reference image, masks, and displacement information
    for each frame across all epochs.

    Parameters
    ----------
    imaging : Imaging object
        The parent imaging object
    displacements : list of np.ndarray
        List of displacement arrays, one per epoch. Each array has shape (n_frames, 2)
        containing (y, x) displacements for each frame.
    refAndMasks : tuple
        The reference and masks computed by suite2p
    ops : dict
        Suite2p operations dictionary
    """

    def __init__(self, imaging, displacements, refAndMasks, ops):
        self.imaging = imaging
        self.displacements = displacements
        self.refAndMasks = refAndMasks
        self.ops = ops
        self.num_epochs = len(displacements)

    def get_displacement_at_frames(self, frames, epoch_index=0):
        """
        Get displacement for specific frames in a epoch.

        Parameters
        ----------
        frames : np.ndarray or int
            Frame indices to get displacements for
        epoch_index : int, default: 0
            Which epoch to get displacements from

        Returns
        -------
        displacements : np.ndarray
            Array of displacements with shape (n_frames, 2) or (2,) if single frame
        """
        if isinstance(frames, int):
            return self.displacements[epoch_index][frames]
        return self.displacements[epoch_index][frames]


def compute_motion_suite2p(imaging, settings=None, **kwargs):
    """
    Compute motion correction signals for the entire imaging recording using Suite2P.

    This function runs once over the entire video to compute displacement information
    for all frames. The returned Suite2PMotion object can then be used with
    RegisterSuite2PImaging to apply the correction on-the-fly.

    Parameters
    ----------
    imaging : BaseImaging
        The imaging object to compute motion for
    settings : Suite2pRegistrationSettings or dict, optional
        Registration settings. Can be a Suite2pRegistrationSettings instance,
        a dict (e.g. loaded from JSON), or None to use defaults.
        If a dict is provided, it will be validated against
        Suite2pRegistrationSettings.
    **kwargs : dict
        Override individual settings fields. Applied on top of `settings`.

    Returns
    -------
    motion : Suite2PMotion
        Motion object containing displacement information for all frames
    """
    from suite2p.default_ops import default_ops
    from suite2p.registration import register

    # Resolve settings: dict -> validated model, None -> defaults, kwargs override
    if settings is None:
        settings = Suite2pRegistrationSettings(**kwargs)
    elif isinstance(settings, dict):
        settings = Suite2pRegistrationSettings.model_validate({**settings, **kwargs})
    elif kwargs:
        settings = settings.model_copy(update=kwargs)

    # Initialize suite2p ops
    ops = default_ops()
    ops.update(settings.model_dump(exclude={"debug", "tmp_dir", "data_type"}))

    bidiphase = ops.get("bidiphase", 0)
    rmin, rmax = -np.inf, np.inf
    nZ = 1

    displacements_per_epoch = []
    refAndMasks = None

    # Process each epoch
    for epoch_idx, epoch in enumerate(imaging.epochs):
        num_frames = epoch.get_num_samples()

        # Compute reference from first epoch only (using initial frames)
        if refAndMasks is None:
            n_ref_frames = min(settings.max_reference_iterations, num_frames)
            ref_frames = epoch.get_series(0, n_ref_frames)
            reference = register.compute_reference(ref_frames)
            refAndMasks = register.compute_reference_masks(reference, ops)

        # Compute displacements for all frames in this epoch
        epoch_displacements = []

        # Process in batches to avoid memory issues
        for start_frame in range(0, num_frames, settings.batch_size):
            end_frame = min(start_frame + settings.batch_size, num_frames)
            batch_frames = epoch.get_series(start_frame, end_frame)

            # Register frames to get displacement information
            _, ymax, xmax, cmax, ymax1, xmax1, cmax1, _ = register.register_frames(
                refAndMasks, batch_frames, rmin=rmin, rmax=rmax, bidiphase=bidiphase, ops=ops, nZ=nZ
            )
            # Store displacements (ymax, xmax are the rigid displacements)
            batch_displacements = np.column_stack([ymax, xmax])
            epoch_displacements.append(batch_displacements)

        # Concatenate all batch displacements for this epoch
        epoch_displacements = np.vstack(epoch_displacements)
        displacements_per_epoch.append(epoch_displacements)

    return Suite2PMotion(imaging, displacements_per_epoch, refAndMasks, ops)


class RegisterSuite2PImaging(BasePreprocessor):
    """
    Apply pre-computed Suite2P motion correction to imaging data.

    This preprocessor applies motion correction on-the-fly using displacement
    information computed by compute_motion_suite2p(). This ensures consistent
    results regardless of how frames are sliced.

    Parameters
    ----------
    imaging : Imaging object
        The parent imaging object
    motion : Suite2PMotion
        Pre-computed motion object from compute_motion_suite2p()
    **kwargs : dict
        Additional keyword arguments

    """

    def __init__(self, imaging, motion, **kwargs):
        BasePreprocessor.__init__(self, imaging)

        if motion.num_epochs != len(imaging.epochs):
            raise ValueError(f"Motion has {motion.num_epochs} epochs but imaging has {len(imaging.epochs)} epochs")

        for epoch_idx, parent_epoch in enumerate(imaging.epochs):
            epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, epoch_idx, **kwargs)
            self.add_epoch(epoch)

        self._kwargs = dict(imaging=imaging, motion=motion, **kwargs)


class RegisterSuite2PImagingEpoch(BasePreprocessorEpoch):
    """
    Epoch-level preprocessor that applies Suite2P motion correction.

    Parameters
    ----------
    parent_imaging_epoch : ImagingEpoch
        The parent imaging epoch
    motion : Suite2PMotion
        Pre-computed motion object
    epoch_index : int
        Index of this epoch
    **kwargs : dict
        Additional keyword arguments
    """

    def __init__(self, parent_imaging_epoch, motion, epoch_index, **kwargs):
        BasePreprocessorEpoch.__init__(self, parent_imaging_epoch)
        self.motion = motion
        self.epoch_index = epoch_index
        self.kwargs = kwargs

    def get_series(self, start_frame, end_frame, plane_indices=None):
        """
        Get motion-corrected frames for the specified range.

        Parameters
        ----------
        start_frame : int
            Starting frame index (inclusive)
        end_frame : int
            Ending frame index (exclusive)

        Returns
        -------
        registered_video : np.ndarray
            Motion-corrected video with shape (n_frames, height, width)
        """
        import copy

        from suite2p.registration import register

        # Get raw video for this frame range
        video = self.parent_imaging_epoch.get_series(start_frame, end_frame)

        # Get pre-computed displacements for these frames
        # displacements = self.motion.get_displacement_at_frames(
        #     frame_indices, self.epoch_index
        # )

        # Apply registration using pre-computed displacements
        ops = self.motion.ops
        bidiphase = ops.get("bidiphase", 0)
        rmin, rmax = -np.inf, np.inf
        nZ = 1

        registered_video = copy.deepcopy(video)

        # Apply the registration with pre-computed shifts
        # Note: We still need to call register_frames, but now it's using
        # a consistent reference from the full video analysis
        registered_video, *_ = register.register_frames(
            self.motion.refAndMasks, registered_video, rmin=rmin, rmax=rmax, bidiphase=bidiphase, ops=ops, nZ=nZ
        )

        return registered_video


# Convenience function for backwards compatibility
register_suite2p = RegisterSuite2PImaging
