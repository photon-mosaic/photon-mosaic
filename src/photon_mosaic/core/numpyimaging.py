"""Imaging and Segmentation Extractors for .npy files.

Classes
-------
NumpyImagingExtractor
    An ImagingExtractor specified by timeseries .npy file, sampling frequency, and channel names.
NumpySegmentationExtractor
    A Segmentation extractor specified by image masks and traces .npy files.
"""

import numpy as np
from numpy.typing import ArrayLike

from .baseimaging import BaseImaging, BaseImagingSegment
from .baserois import BaseRois


class NumpyImaging(BaseImaging):
    """An single-segment or multi-segment Imaging specified by a numpy array or list of arrays"""

    def __init__(
        self,
        imaging_series: ArrayLike | list[ArrayLike],
        sampling_frequency: float,
        time_vectors: ArrayLike | list[ArrayLike] | None = None,
        seed=None,
    ):
        """Create a NumpyImagingExtractor from a numpy array or list of numpy arrays.

        If a list of numpy arrays is provided, each array is treated as a separate segment.
        Individual segments can have one or more planes. In the former case, the shape of each
        array should be (num_frames, height, width). In the latter case, the shape should be
        (num_frames, height, width, num_planes).

        Parameters
        ----------
        imaging_series: ArrayLike | list[ArrayLike]
            Numpy array or list of numpy arrays representing the video.
        sampling_frequency: float
            Sampling frequency of the video in Hz.
        time_vectors: ArrayLike | list[ArrayLike] | None, default: None
            Optional time vector(s) for the video.
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
                raise ValueError(
                    "'timeseries' must be a 3D or 4D numpy array (num_frames, height, width, [num_planes])"
                )
            if len(video.shape) == 3:
                video = video[:, :, :, np.newaxis]  # Add a planes dimension
            shapes.append(video.shape[1:])
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All segments must have the same image shape (height, width, planes)")

        # Check consistency of time vectors
        if time_vectors is not None:
            if num_segments == 1 and isinstance(time_vectors, np.ndarray):
                time_vectors = [time_vectors]
            assert len(time_vectors) == num_segments, "Number of time vectors must match number of segments"
        else:
            time_vectors = [None] * num_segments

        BaseImaging.__init__(self, shape=shapes[0], sampling_frequency=sampling_frequency)

        for video, time_vector in zip(videos, time_vectors):
            self.add_segment(
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
            "seed": seed,
        }


class NumpyImagingSegment(BaseImagingSegment):
    """A single segment of an Imaging specified by a numpy array"""

    def __init__(
        self,
        video: np.ndarray,
        sampling_frequency: float,
        time_vector: ArrayLike | None = None,
    ):
        super().__init__(sampling_frequency=sampling_frequency, time_vector=time_vector)
        self._video = video

    def get_series(
        self, start_frame: int | None = None, end_frame: int | None = None, plane_indices: list | None = None
    ) -> np.ndarray:
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
        return self._video[start:end, :, :, plane_indices] if plane_indices is not None else self._video[start:end]

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
        roi_image_masks: ArrayLike,
        sampling_frequency: float,
        roi_ids: ArrayLike | None = None,
    ):
        """Create a NumpyRois object from numpy arrays.

        Parameters
        ----------
        roi_image_masks : ArrayLike
            Numpy array representing the image masks for each ROI
            Accepted dimensions are: (num_rois x height x width) for single-plane and
            (num_rois x height x width x num_planes) for multi-plane.
        sampling_frequency : float
            Sampling frequency of the ROIs in Hz.
        roi_ids : ArrayLike | None, default: None
            Optional array of ROI IDs. If None, IDs will be assigned as integers from 0 to num_rois-1.
        """
        num_rois = roi_image_masks.shape[0]
        mask_shape = roi_image_masks[0].shape
        if len(mask_shape) == 2 or (len(mask_shape) == 3 and mask_shape[2] == 1):
            num_planes = 1
        elif len(mask_shape) == 3:
            num_planes = mask_shape[2]
        else:
            raise ValueError(
                "'roi_image_masks' must contain 2D (height, width) or 3D (height, width, num_planes) masks"
            )
        if roi_ids is None:
            roi_ids = np.arange(num_rois)

        image_shape = mask_shape[:2]
        BaseRois.__init__(
            self,
            sampling_frequency=sampling_frequency,
            shape=image_shape,
            roi_ids=roi_ids,
            num_planes=num_planes,
        )
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
