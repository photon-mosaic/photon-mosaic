from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class MedianFilterImaging(BasePreprocessor):

    def __init__(self, imaging, size: int, **kwargs):

        BasePreprocessor.__init__(self, imaging)
        for parent_segment in imaging._imaging_segments:
            segment = MedianFilterImagingSegment(parent_segment, size)
            self.add_imaging_segment(segment)

        self._kwargs = dict(imaging=imaging, size=size)


class MedianFilterImagingSegment(BasePreprocessorSegment):

    def __init__(self, parent_imaging_segment, size: int):
        BasePreprocessorSegment.__init__(self, parent_imaging_segment)
        self.size = size

    def get_series(self, start_frame, end_frame):
        import numpy as np
        from skimage.morphology import disk
        from skimage.filters import median

        data = self.parent_imaging_segment.get_series(start_frame, end_frame)
        out = np.zeros_like(data)
        for i in range(data.shape[0]):
            out[i] = median(data[i], footprint=disk(self.size))

        return out



median_filter = MedianFilterImaging
