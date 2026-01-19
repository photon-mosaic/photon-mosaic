"""Imaging and Segmentation Extractors for .npy files.

Classes
-------
NumpyImagingExtractor
    An ImagingExtractor specified by timeseries .npy file, sampling frequency, and channel names.
NumpySegmentationExtractor
    A Segmentation extractor specified by image masks and traces .npy files.
"""

from pathlib import Path
from warnings import warn

import numpy as np

from .baseimaging import BaseImaging, BaseImagingSegment
from .baserois import BaseRois
from .utils import FloatType, ArrayType


class NumpyImaging(BaseImaging):
    """An single-segment Imaging specified by timeseries .npy or numpy array"""

    def __init__(
        self,
        imaging_series: ArrayType | list[ArrayType],
        sampling_frequency: FloatType,
        time_vectors: ArrayType | list[ArrayType] | None = None,
        plane_ids: list[int] | None = None,
        seed=None,
    ):
        """Create a NumpyImagingExtractor from a numpy array or list of numpy arrays.

        If a list of numpy arrays is provided, each array is treated as a separate segment.
        Individual segments can have one or more planes. In the former case, the shape of each 
        array should be (num_frames, height, width). In the latter case, the shape should be 
        (num_frames, height, width, num_planes).

        Parameters
        ----------
        imaging_series: ArrayType | list[ArrayType]
            Numpy array or list of numpy arrays representing the video.
        sampling_frequency: FloatType
            Sampling frequency of the video in Hz.
        time_vectors: ArrayType | list[ArrayType] | None, default: None
            Optional time vector(s) for the video.
        plane_ids: list[int] | None, default: None
            Optional list of plane IDs for the video.
        """
        if isinstance(imaging_series, np.ndarray):
            videos = [imaging_series]
        elif isinstance(imaging_series, list) and all(isinstance(ts, np.ndarray) for ts in imaging_series):
            videos = imaging_series
        else:
            raise ValueError("'timeseries' must be a numpy array or a list of numpy arrays")

        num_segments = len(videos)
        self._sampling_frequency = float(sampling_frequency)

        # Check that all shapes and number of planes are consistent across segments
        shapes = []
        for video in videos:
            if len(video.shape) not in [3, 4]:
                raise ValueError("'timeseries' must be a 3D or 4D numpy array (num_frames, height, width, [num_channels])")
            shapes.append(video.shape[1:])
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All segments must have the same image shape (height, width) and number of planes")
        height, width = shapes[0][0:2]
        num_planes = shapes[0][2] if len(shapes[0]) == 3 else 1

        if num_planes > 1:
            if plane_ids is None:
                plane_ids = list(range(num_planes))
            else:
                assert len(plane_ids) == num_planes, "plane_ids length must match num_planes"

        # Check consistency of time vectors
        if time_vectors is not None:
            if num_segments == 1 and isinstance(time_vectors, np.ndarray):
                time_vectors = [time_vectors]
            assert len(time_vectors) == num_segments, "Number of time vectors must match number of segments"
        else:
            time_vectors = [None] * num_segments
        
        BaseImaging.__init__(self, shape=(height, width), sampling_frequency=sampling_frequency, plane_ids=plane_ids)

        for video, time_vector in zip(videos, time_vectors):
            self.add_imaging_segment(
                NumpyImagingSegment(
                    video=video,
                    sampling_frequency=self._sampling_frequency,
                    time_vector=time_vector,
                )
            )

        self._kwargs = {
            "imaging_series": imaging_series,
            "sampling_frequency": self._sampling_frequency,
            "time_vectors": time_vectors,
            "plane_ids": plane_ids,
            "seed": seed,
        }


class NumpyImagingSegment(BaseImagingSegment):
    """A single segment of an Imaging specified by a numpy array"""

    def __init__(
        self,
        video: np.ndarray,
        sampling_frequency: float,
        time_vector: ArrayType | None = None,
    ):
        super().__init__(sampling_frequency=sampling_frequency, time_vector=time_vector)
        self._video = video

    def get_series(self, start_frame: int | None = None, end_frame: int | None = None, plane_indices: list | None = None) -> np.ndarray:
        """Get the raw series, optionally for a subset of samples.

        Parameters
        ----------
        start_frame : int | None, default: None
            start frame index, or zero if None
        end_frame : int | None, default: None
            end frame, or number of frames if None
        plane_indices : list | None, default: None
            List of plane indices to include, or all planes if None

        Returns
        -------
        series: np.ndarray
            The raw series for the specified frame range.
        """
        start = start_frame if start_frame is not None else 0
        end = end_frame if end_frame is not None else self._video.shape[0]
        print(self._video.shape)
        if self._video.ndim == 4 and plane_indices is not None:
            return self._video[start:end, :, :, plane_indices]
        else:
            return self._video[start:end, ...]

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal segment

        Returns:
            SampleIndex : Number of samples in the signal segment
        """
        return self._video.shape[0]


class NumpyRois(BaseRois):
    """A ROIs object specified by numpy arrays for masks and traces."""

    def __init__(
        self,
        roi_image_masks: list[np.ndarray],
        sampling_frequency: FloatType,
        roi_ids: ArrayType | None = None,
    ):
        """Create a NumpyRois object from numpy arrays.

        Parameters
        ----------
        roi_image_masks : np.ndarray
            Numpy array representing the image masks for each ROI (dimensions: num_rois x height x width).
        sampling_frequency : FloatType
            Sampling frequency of the ROIs in Hz.
        roi_ids : ArrayType | None, default: None
            Optional array of ROI IDs.
        """
        shape = roi_image_masks[0].shape
        BaseRois.__init__(self, sampling_frequency=sampling_frequency, shape=shape, roi_ids=roi_ids)
        self._roi_image_masks = roi_image_masks

        self._kwargs = {
            "roi_image_masks": roi_image_masks,
            "sampling_frequency": sampling_frequency,
            "roi_ids": roi_ids,
        }

    def get_roi_image_masks(self, roi_ids: list[int | str] | None = None) -> list[np.ndarray]:
        """Get the image masks for specific ROIs.

        Parameters
        ----------
        roi_ids : list[int | str] | None
            The IDs of the ROIs.

        Returns
        -------
        list[np.ndarray]
            The image masks for the specified ROIs.
        """
        if roi_ids is None:
            return self._roi_image_masks
        else:
            roi_indices = self.ids_to_indices(roi_ids)
            return self._roi_image_masks[roi_indices]
