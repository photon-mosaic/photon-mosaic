from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class EdgeDetectorImaging(BasePreprocessor):

    def __init__(self, imaging, **kwargs):

        BasePreprocessor.__init__(self, imaging)
        for parent_segment in imaging._imaging_segments:
            segment = EdgeDetectorImagingSegment(parent_segment)
            self.add_imaging_segment(segment)

        self._kwargs = dict(imaging=imaging)


# TODO: use canny or other methods
class EdgeDetectorImagingSegment(BasePreprocessorSegment):

    def __init__(self, parent_imaging_segment):
        BasePreprocessorSegment.__init__(self, parent_imaging_segment)

    def get_series(self, start_frame, end_frame):
        import numpy as np
        from skimage.feature import canny

        video = self.parent_imaging_segment.get_series(start_frame, end_frame)
        edges = np.zeros_like(video, dtype=bool)

        for i in range(video.shape[0]):
            edges[i] = canny(video[i]) #, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

        return edges.astype(video.dtype)


edge_detector = EdgeDetectorImaging
