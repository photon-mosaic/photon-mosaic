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
from .utils import PathType, FloatType, ArrayType


class NumpyImaging(BaseImaging):
    """An single-segment Imaging specified by timeseries .npy or numpy array"""

    def __init__(
        self,
        timeseries: ArrayType,
        sampling_frequency: FloatType,
        channel_index: int = None,
        time_vector: ArrayType | None = None,
        seed=None,
    ):
        """Create a NumpyImagingExtractor from a .npy file.

        Parameters
        ----------
        timeseries: ArrayType
            Numpy array representing the video.
        sampling_frequency: FloatType
            Sampling frequency of the video in Hz.
        channel_index: int, default: None
            Index of the channel to load (for multi-channel videos).
        time_vector: ArrayType | None, default: None
            Optional time vector for the video.
        """
        if isinstance(timeseries, np.ndarray):
            self._video = timeseries
            timeseries_kwarg = timeseries
        else:
            raise TypeError("'timeseries' must be a numpy array")

        self._sampling_frequency = float(sampling_frequency)

        if seed is None:
            rng = np.random.default_rng(seed=seed)
            seed = rng.integers(0, 1e6)

        if len(self._video.shape) not in [3, 4]:
            raise ValueError("'timeseries' must be a 3D or 4D numpy array (num_frames, height, width, [num_channels])")
        _, height, width = self._video.shape[0:3]
        num_channels = 1 if len(self._video.shape) == 3 else self._video.shape[3]
        if num_channels > 1:
            assert channel_index is not None, "'channel_index' must be provided for multi-channel videos"
            self.channel_index = channel_index
        else:
            self.channel_index = 0

        if len(self._video.shape) == 4:
            # check if this converts to np.ndarray
            self._video = self._video[:, :, :, self.channel_index]

        BaseImaging.__init__(self, shape=(height, width), sampling_frequency=sampling_frequency)

        self.add_imaging_segment(
            NumpyImagingSegment(
                video=self._video,
                sampling_frequency=self._sampling_frequency,
                time_vector=time_vector,
            )
        )

        self._kwargs = {
            "timeseries": timeseries_kwarg,
            "sampling_frequency": self._sampling_frequency,
            "channel_index": self.channel_index,
            "time_vector": time_vector,
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

    def get_series(self, start_frame: int | None = None, end_frame: int | None = None) -> np.ndarray:
        """Get the raw series, optionally for a subset of samples.

        Parameters
        ----------
        start_frame : int | None, default: None
            start frame index, or zero if None
        end_frame : int | None, default: None
            end frame, or number of frames if None

        Returns
        -------
        series: np.ndarray
            The raw series for the specified frame range.
        """
        start = start_frame if start_frame is not None else 0
        end = end_frame if end_frame is not None else self._video.shape[0]
        return self._video[start:end, ...]

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal segment

        Returns:
            SampleIndex : Number of samples in the signal segment
        """
        return self._video.shape[0]
