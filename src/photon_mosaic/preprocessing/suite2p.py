import numpy as np

from .basepreprocessor import BasePreprocessor, BasePreprocessorEpoch
from .baseregistrationsettings import Suite2pRegistrationSettings


class Suite2PMotion:
    """
    Container for Suite2P motion correction reference data.

    Stores the reference image and masks needed to apply motion correction
    on-the-fly via register_frames.

    Parameters
    ----------
    refAndMasks : tuple
        The reference and masks computed by suite2p
    ops : dict
        Suite2p operations dictionary
    """

    def __init__(self, refAndMasks, ops):
        self.refAndMasks = refAndMasks
        self.ops = ops


def compute_motion_suite2p(imaging, settings=None, **kwargs):
    """
    Pre-compute the Suite2P reference image and masks for motion correction.

    This function computes the reference and masks from the first epoch's
    initial frames. The returned Suite2PMotion object can then be used with
    RegisterSuite2PImaging, which applies registration on-the-fly in get_series.

    Parameters
    ----------
    imaging : BaseImaging
        The imaging object to compute the reference from
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
        Motion object containing the reference and masks for on-the-fly registration
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

    # Compute reference from first epoch
    first_epoch = imaging.epochs[0]
    num_frames = first_epoch.get_num_samples()
    n_ref_frames = min(settings.max_reference_iterations, num_frames)
    ref_frames = first_epoch.get_series(0, n_ref_frames)
    reference = register.compute_reference(ref_frames)
    refAndMasks = register.compute_reference_masks(reference, ops)

    return Suite2PMotion(refAndMasks, ops)


class RegisterSuite2PImaging(BasePreprocessor):
    """
    Apply pre-computed Suite2P motion correction to imaging data.

    This preprocessor applies motion correction on-the-fly using the
    reference and masks computed by compute_motion_suite2p().

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

        Computes and applies displacement on-the-fly using the pre-computed
        reference and masks.

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
        from suite2p.registration import register

        video = self.parent_imaging_epoch.get_series(start_frame, end_frame)

        ops = self.motion.ops
        bidiphase = ops.get("bidiphase", 0)
        rmin, rmax = -np.inf, np.inf
        nZ = 1

        registered_video, *_ = register.register_frames(
            self.motion.refAndMasks, video, rmin=rmin, rmax=rmax, bidiphase=bidiphase, ops=ops, nZ=nZ
        )

        return registered_video


# Convenience function for backwards compatibility
register_suite2p = RegisterSuite2PImaging
