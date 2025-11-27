from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class LaplaceFilterImaging(BasePreprocessor):

    def __init__(self, imaging, ksize=3):

        BasePreprocessor.__init__(self, imaging)
        for parent_segment in imaging._imaging_segments:
            segment = LaplaceFilterImagingSegment(parent_segment, ksize=ksize)
            self.add_imaging_segment(segment)

        self._kwargs = dict(imaging=imaging, ksize=ksize)


class LaplaceFilterImagingSegment(BasePreprocessorSegment):

    def __init__(self, parent_imaging_segment, ksize):
        BasePreprocessorSegment.__init__(self, parent_imaging_segment)
        self.ksize = ksize

    def get_series(self, start_frame, end_frame):
        import numpy as np
        from skimage.filters import laplace

        video = self.parent_imaging_segment.get_series(start_frame, end_frame)
        filtered = np.zeros_like(video, dtype=bool)

        for i in range(video.shape[0]):
            filtered[i] = laplace(video[i], self.ksize)

        return filtered.astype(video.dtype)


laplace_filter = LaplaceFilterImaging
