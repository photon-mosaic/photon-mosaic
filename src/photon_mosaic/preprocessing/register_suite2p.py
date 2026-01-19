from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment
import numpy as np


class Suite2PMotion:
    """
    Container for Suite2P motion correction information.
    
    Stores the reference image, masks, and displacement information 
    for each frame across all segments.
    
    Parameters
    ----------
    imaging : Imaging object
        The parent imaging object
    displacements : list of np.ndarray
        List of displacement arrays, one per segment. Each array has shape (n_frames, 2)
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
        self.num_segments = len(displacements)
        
    def get_displacement_at_frames(self, frames, plane_index=0):
        """
        Get displacement for specific frames in a segment.
        
        Parameters
        ----------
        frames : np.ndarray or int
            Frame indices to get displacements for
        segment_index : int, default: 0
            Which segment to get displacements from
            
        Returns
        -------
        displacements : np.ndarray
            Array of displacements with shape (n_frames, 2) or (2,) if single frame
        """
        if isinstance(frames, int):
            return self.displacements[plane_index][frames]
        return self.displacements[plane_index][frames]


def compute_motion_suite2p(imaging, reference_frames=150, batch_size=500, **suite2p_kwargs):
    """
    Compute motion correction signals for the entire imaging recording using Suite2P.
    
    This function runs once over the entire video to compute displacement information
    for all frames. The returned Suite2PMotion object can then be used with 
    RegisterSuite2PImaging to apply the correction on-the-fly.
    
    Parameters
    ----------
    imaging : Imaging object
        The imaging object to compute motion for
    reference_frames : int, default: 150
        Number of initial frames to use for computing the reference image
    batch_size : int, default: 500
        Number of frames to process at once during motion estimation
    **suite2p_kwargs : dict
        Additional keyword arguments to pass to suite2p registration
        
    Returns
    -------
    motion : Suite2PMotion
        Motion object containing displacement information for all frames
    """
    from suite2p.registration import register
    from suite2p.default_ops import default_ops
    
    # Initialize suite2p ops
    ops = default_ops()
    ops.update(suite2p_kwargs)
    
    bidiphase = ops.get('bidiphase', 0)
    rmin, rmax = -np.inf, np.inf
    nZ = 1
    
    displacements_per_segment = []
    refAndMasks = None
    
    # Process each plane, the last axis of imaging.get_series() is assumed to be planes
    for plane_index in range(imaging.get_series().shape[-1]):
        plane = imaging.get_series()[..., plane_index]
        num_frames = imaging.get_num_samples()
        
        # Compute reference from first segment only (using initial frames)
        if refAndMasks is None:
            n_ref_frames = min(reference_frames, num_frames)
            ref_frames = plane[0:n_ref_frames].compute()
            reference = register.compute_reference(ref_frames)
            refAndMasks = register.compute_reference_masks(reference, ops)
        
        # Compute displacements for all frames in this segment
        segment_displacements = []
        
        # Process in batches to avoid memory issues
        for start_frame in range(0, num_frames, batch_size):
            end_frame = min(start_frame + batch_size, num_frames)
            batch_frames = plane[start_frame:end_frame].compute()
            
            # Register frames to get displacement information
            _, ymax, xmax, cmax, ymax1, xmax1, cmax1, _ = register.register_frames(
                refAndMasks, batch_frames, rmin=rmin, rmax=rmax, 
                bidiphase=bidiphase, ops=ops, nZ=nZ
            )
            # Store displacements (ymax, xmax are the rigid displacements)
            batch_displacements = np.column_stack([ymax, xmax])
            segment_displacements.append(batch_displacements)
        
        # Concatenate all batch displacements for this segment
        segment_displacements = np.vstack(segment_displacements)
        displacements_per_segment.append(segment_displacements)
    
    return Suite2PMotion(imaging, displacements_per_segment, refAndMasks, ops)


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
        
        # if motion.num_segments != len(imaging.segments):
        #     raise ValueError(
        #         f"Motion has {motion.num_segments} segments but imaging has "
        #         f"{len(imaging.segments)} segments"
        #     )
        
        # for segment_idx, parent_segment in enumerate(imaging.segments):
        #     segment = RegisterSuite2PImagingSegment(
        #         parent_segment, motion, segment_idx, **kwargs
        #     )
        #     self.add_imaging_segment(segment)

        self._kwargs = dict(imaging=imaging, motion=motion, **kwargs)


# class RegisterSuite2PImagingSegment(BasePreprocessorSegment):
#     """
#     Segment-level preprocessor that applies Suite2P motion correction.
    
#     Parameters
#     ----------
#     parent_imaging_segment : ImagingSegment
#         The parent imaging segment
#     motion : Suite2PMotion
#         Pre-computed motion object
#     segment_index : int
#         Index of this segment
#     **kwargs : dict
#         Additional keyword arguments
#     """
    
#     def __init__(self, parent_imaging_segment, motion, segment_index, **kwargs):
#         BasePreprocessorSegment.__init__(self, parent_imaging_segment)
#         self.motion = motion
#         self.segment_index = segment_index
#         self.kwargs = kwargs

    def get_series(self, start_frame, end_frame):
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
        from suite2p.registration import register
        import copy
        
        # Get raw video for this frame range
        video = self.imaging.get_series(start_frame, end_frame)
        
        # Get pre-computed displacements for these frames
        frame_indices = np.arange(start_frame, end_frame)
        # displacements =self.motion.get_displacement_at_frames(
        #     frame_indices, plane_index=0
        # )
        
        # Apply registration using pre-computed displacements
        ops = self.motion.ops
        bidiphase = ops.get('bidiphase', 0)
        rmin, rmax = -np.inf, np.inf
        nZ = 1
        
        registered_video = copy.deepcopy(video)
        
        # Apply the registration with pre-computed shifts
        # Note: We still need to call register_frames, but now it's using 
        # a consistent reference from the full video analysis
        for plane_index in range(registered_video.shape[-1]):
            registered_video[..., plane_index], *_ = register.register_frames(
                self.motion.refAndMasks[...,plane_index], registered_video[..., plane_index].compute(), 
                rmin=rmin, rmax=rmax, bidiphase=bidiphase, 
                ops=ops, nZ=nZ
            )

        return registered_video

# Convenience function for backwards compatibility
register_suite2p = RegisterSuite2PImaging