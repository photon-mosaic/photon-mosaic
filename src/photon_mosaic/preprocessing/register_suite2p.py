from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment

class RegisterSuite2PImaging(BasePreprocessor):

    def __init__(self, imaging, **kwargs):

        BasePreprocessor.__init__(self, imaging)
        for parent_segment in imaging.segments:
            segment = RegisterSuite2PImagingSegment(parent_segment, **kwargs)
            self.add_imaging_segment(segment)

        self._kwargs = dict(imaging=imaging, **kwargs)

class RegisterSuite2PImagingSegment(BasePreprocessorSegment):
    def __init__(self, parent_imaging_segment, **kwargs):
        BasePreprocessorSegment.__init__(self, parent_imaging_segment)
        self.kwargs = kwargs
        self.refAndMasks = None

    def get_series(self, start_frame, end_frame):

        from suite2p.registration import register
        from suite2p.default_ops import default_ops
        import numpy as np
        import copy

        video = self.parent_imaging_segment.get_series(start_frame, end_frame)

        ops = default_ops()
        bidiphase = 0
        rmin, rmax = -np.inf, np.inf
        nZ = 1

        if not self.refAndMasks:
            #  we can do also bidiphase correction first
            first_150_frames = np.zeros((150, video.shape[1], video.shape[2]), dtype=video.dtype)

            first_150_frames[:] = self.parent_imaging_segment.get_series(0,150)[:]
            
            reference = register.compute_reference(first_150_frames)
            self.refAndMasks = register.compute_reference_masks(reference, ops)
        

        registered_video = copy.deepcopy(video)
        registered_video, *_ = register.register_frames(
            self.refAndMasks, registered_video, rmin=rmin, rmax=rmax, bidiphase=bidiphase, ops=ops,
            nZ=nZ)
        
        return registered_video
    

register_suite2p = RegisterSuite2PImaging