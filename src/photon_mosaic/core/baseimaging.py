from typing import Optional
import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from math import prod
import warnings


from roiextractors.core_utils import _convert_bytes_to_str, _convert_seconds_to_str
from spikeinterface.core.base import BaseExtractor, BaseSegment

from .imaging_tools import write_binary_imaging, get_random_data_chunks

# TODO: frames instead of samples
# TODO: epoch instead of segment (segmentation is another thing)


class BaseImaging(BaseExtractor):
    """Base class for imaging extractors."""

    def __init__(self, sampling_frequency: float, shape: tuple | list | np.ndarray, channel_ids: list | None = None):
        if channel_ids is None:
            channel_ids = [0]  # fake channel
        BaseExtractor.__init__(self, channel_ids)
        self._sampling_frequency = float(sampling_frequency)
        assert len(shape) == 2, "Shape must be a tuple/list/array of length 2 (height, width)"
        self._image_shape = np.array(shape)
        self._imaging_segments: list[BaseImagingSegment] = []
        self._average_image = None

    def _repr_header(self, display_name=True):
        """Generate text representation of the BaseImaging object."""
        num_samples = [self.get_num_samples(segment_index=i) for i in range(self.get_num_segments())]
        image_shape = self.image_shape
        dtype = self.get_dtype()
        sf_hz = self.sampling_frequency

        # Format sampling frequency
        if not sf_hz.is_integer():
            sampling_frequency_repr = f"{sf_hz:f} Hz"
        else:
            sampling_frequency_repr = f"{sf_hz:0.1f}Hz"

        # Calculate duration
        durations = [ns / sf_hz for ns in num_samples]
        duration_repr = [_convert_seconds_to_str(duration) for duration in durations]

        # Calculate memory size using product of all dimensions in image_size
        memory_sizes = [ns * prod(image_shape) * dtype.itemsize for ns in num_samples]
        memory_repr = [_convert_bytes_to_str(memory_size) for memory_size in memory_sizes]

        if self.get_num_segments() == 1:
            num_samples = num_samples[0]
            duration_repr = duration_repr[0]
            memory_repr = memory_repr[0]

        if display_name and self.name != self.__class__.__name__:
            name = f"{self.name} ({self.__class__.__name__})"
        else:
            name = self.__class__.__name__

        # Format shape string based on whether data is volumetric or not
        image_shape_repr = f"{image_shape[0]} rows x {image_shape[1]} columns "
        return (
            f"{name}:\n"
            f"{sampling_frequency_repr} - "
            f"{self.get_num_segments()} segments - "
            f"{image_shape_repr} - "
            f"{duration_repr} - "
            f"{dtype} dtype - "
            f"{memory_repr}"
        )

    def __repr__(self):
        return self._repr_header()

    def _repr_html_(self, display_name=True):
        common_style = "margin-left: 10px;"
        border_style = "border:1px solid #ddd; padding:10px;"

        html_header = f"<div style='{border_style}'><strong>{self._repr_header(display_name)}</strong></div>"

        html_segments = ""
        if self.get_num_segments() > 1:
            html_segments += f"<details style='{common_style}'>  <summary><strong>Segments</strong></summary><ol>"
            for segment_index in range(self.get_num_segments()):
                samples = self.get_num_samples(segment_index)
                duration = self.get_duration(segment_index)
                memory_size = self.get_memory_size(segment_index)
                samples_str = f"{samples:,}"
                duration_str = _convert_seconds_to_str(duration)
                memory_size_str = _convert_bytes_to_str(memory_size)
                html_segments += (
                    f"<li> Samples: {samples_str}, Duration: {duration_str}, Memory: {memory_size_str}</li>"
                )

            html_segments += "</ol></details>"

        html_extra = self._get_common_repr_html(common_style)
        # remove properties from html_extra
        if "<summary><strong>Properties</strong></summary>" in html_extra:
            start = html_extra.find("<details style='")
            # Find the Properties section specifically
            properties_start = html_extra.find("<summary><strong>Properties</strong></summary>")
            if properties_start != -1:
                # Find the start of the details tag containing Properties
                details_start = html_extra.rfind("<details", 0, properties_start)
                # Find the end of that details section
                details_end = html_extra.find("</details>", properties_start) + len("</details>")
                html_extra = html_extra[:details_start] + html_extra[details_end:]
        html_repr = html_header + html_segments + html_extra
        return html_repr

    @property
    def image_shape(self):
        """Get the shape of the images (height, width).

        Returns
        -------
        tuple
            The shape of the images as (height, width).
        """
        return self._image_shape

    def get_image_shape(self):
        return self._image_shape

    def get_sample_size_in_bytes(self):
        return self.get_num_pixels() * np.dtype(self.get_dtype()).itemsize

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_num_samples(self, segment_index: int | None = None) -> int:
        """Get the total number of samples (frames) in the imaging data.

        Parameters
        ----------

        Returns
        -------
        int
            The total number of samples (frames).
        """
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        return self._imaging_segments[segment_index].get_num_samples()

    def get_num_frames(self, segment_index: int | None = None) -> int:
        """Get the total number of frames in the imaging data.

        Parameters
        ----------

        Returns
        -------
        int
            The total number of frames.
        """
        return self.get_num_samples(segment_index=segment_index)

    def get_num_segments(self) -> int:
        """Get the number of imaging segments.

        Returns
        -------
        int
            The number of imaging segments.
        """
        return len(self._imaging_segments)

    def get_dtype(self) -> DTypeLike:
        """Get the data type of the video.

        Returns
        -------
        dtype: dtype
            Data type of the video.
        """
        return self.get_series(start_frame=0, end_frame=2, segment_index=0).dtype

    def get_num_pixels(self) -> int:
        return np.prod(self.image_shape)

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        segment_index: int | None = None,
    ) -> np.ndarray:
        """Get a series of frames from the imaging data.

        Parameters
        ----------
        start_sample : int
            The starting frame index (inclusive).
        end_sample : int
            The ending frame index (exclusive).
        segment_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.

        Returns
        -------
        np.ndarray
            The requested series of frames as a NumPy array.
        """
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_samples(segment_index=segment_index)
        return self._imaging_segments[segment_index].get_series(start_frame, end_frame)

    def get_average_image(
        self,
        num_chunks: int = 20,
        chunk_duration: str = "1s",
        chunk_size: int | None = None,
        recompute: bool = False,
    ) -> np.ndarray:
        if self._average_image is not None and not recompute:
            return self._average_image
        else:
            data = get_random_data_chunks(
                self,
                num_chunks_per_segment=num_chunks,
                chunk_duration=chunk_duration,
                chunk_size=chunk_size,
                concatenated=True,
            )
            self._average_image = np.mean(data, axis=0)
            return self._average_image

    def add_imaging_segment(self, imaging_segment):
        """Adds an imaging segment.

        Parameters
        ----------
        imaging_segment : BaseImagingSegment
            The imaging segment to add.
        """
        self._imaging_segments.append(imaging_segment)
        imaging_segment.set_parent_extractor(self)

    def get_times(self, segment_index: int | None = None) -> np.ndarray:
        """Get the timestamps for each frame in the imaging data.

        Parameters
        ----------
        segment_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.

        Returns
        -------
        np.ndarray
            The timestamps for each frame.
        """
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        return self._imaging_segments[segment_index].get_times()

    def set_times(self, times: ArrayLike, segment_index: int | None = None):
        """Set the timestamps for each frame in the imaging data.

        Parameters
        ----------
        times : ArrayLike
            The timestamps to set.
        segment_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.
        """
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        self._imaging_segments[segment_index].time_vector = np.asarray(times)

    def get_start_time(self, segment_index=None) -> float:
        """Get the start time of the recording segment.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)

        Returns
        -------
        float
            The start time in seconds
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._imaging_segments[segment_index]
        return rs.get_start_time()

    def get_end_time(self, segment_index=None) -> float:
        """Get the stop time of the recording segment.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)

        Returns
        -------
        float
            The stop time in seconds
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._imaging_segments[segment_index]
        return rs.get_end_time()

    def has_time_vector(self, segment_index: Optional[int] = None):
        """Check if the segment of the recording has a time vector.

        Parameters
        ----------
        segment_index : int or None, default: None
            The segment index (required for multi-segment)

        Returns
        -------
        bool
            True if the recording has time vectors, False otherwise
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._imaging_segments[segment_index]
        d = rs.get_times_kwargs()
        return d["time_vector"] is not None

    def reset_times(self):
        """
        Reset time information in-memory for all segments that have a time vector.
        If the timestamps come from a file, the files won't be modified. but only the in-memory
        attributes of the recording objects are deleted. Also `t_start` is set to None and the
        segment's sampling frequency is set to the recording's sampling frequency.
        """
        for segment_index in range(self.get_num_segments()):
            rs = self._imaging_segments[segment_index]
            if self.has_time_vector(segment_index):
                rs.time_vector = None
            rs.t_start = None
            rs.sampling_frequency = self.sampling_frequency

    def shift_times(self, shift: int | float, segment_index: int | None = None) -> None:
        """
        Shift all times by a scalar value.

        Parameters
        ----------
        shift : int | float
            The shift to apply. If positive, times will be increased by `shift`.
            e.g. shifting by 1 will be like the recording started 1 second later.
            If negative, the start time will be decreased i.e. as if the recording
            started earlier.

        segment_index : int | None
            The segment on which to shift the times.
            If `None`, all segments will be shifted.
        """
        if segment_index is None:
            segments_to_shift = range(self.get_num_segments())
        else:
            segments_to_shift = (segment_index,)

        for segment_index in segments_to_shift:
            rs = self._imaging_segments[segment_index]

            if self.has_time_vector(segment_index=segment_index):
                rs.time_vector += shift
            else:
                new_start_time = 0 + shift if rs.t_start is None else rs.t_start + shift
                rs.t_start = new_start_time

    def sample_index_to_time(self, sample_ind, segment_index=None):
        """
        Transform sample index into time in seconds
        """
        segment_index = self._check_segment_index(segment_index)
        rs = self._imaging_segments[segment_index]
        return rs.sample_index_to_time(sample_ind)

    def time_to_sample_index(self, time_s, segment_index=None):
        segment_index = self._check_segment_index(segment_index)
        rs = self._imaging_segments[segment_index]
        return rs.time_to_sample_index(time_s)

    def is_binary_compatible(self) -> bool:
        """
        Checks if the recording is "binary" compatible.
        To be used before calling `rec.get_binary_description()`

        Returns
        -------
        bool
            True if the underlying recording is binary
        """
        # has to be changed in subclass if yes
        return False

    def get_binary_description(self):
        """
        When `rec.is_binary_compatible()` is True
        this returns a dictionary describing the binary format.
        """
        if not self.is_binary_compatible:
            raise NotImplementedError

    def _get_t_starts(self):
        # handle t_starts
        t_starts = []
        for rs in self._imaging_segments:
            d = rs.get_times_kwargs()
            t_starts.append(d["t_start"])

        if all(t_start is None for t_start in t_starts):
            t_starts = None
        return t_starts

    def _get_time_vectors(self):
        time_vectors = []
        for rs in self._imaging_segments:
            d = rs.get_times_kwargs()
            time_vectors.append(d["time_vector"])
        if all(time_vector is None for time_vector in time_vectors):
            time_vectors = None
        return time_vectors

    def _save(self, format="binary", verbose: bool = False, **save_kwargs):
        from spikeinterface.core.job_tools import split_job_kwargs

        kwargs, job_kwargs = split_job_kwargs(save_kwargs)

        if format == "binary":
            folder = kwargs["folder"]
            file_paths = [folder / f"video_cached_seg{i}.raw" for i in range(self.get_num_segments())]
            dtype = kwargs.get("dtype", None) or self.get_dtype()
            t_starts = self._get_t_starts()

            write_binary_imaging(self, file_paths=file_paths, dtype=dtype, verbose=verbose, **job_kwargs)

            from .binaryimaging import BinaryFolderImaging, BinaryImaging

            # This is created so it can be saved as json because the `BinaryFolderRecording` requires it loading
            # See the __init__ of `BinaryFolderRecording`
            binary_imaging = BinaryImaging(
                file_paths=file_paths,
                sampling_frequency=self.get_sampling_frequency(),
                image_shape=self.image_shape,
                dtype=dtype,
                t_starts=t_starts,
                file_offset=0,
            )
            binary_imaging.dump(folder / "binary.json", relative_to=folder)

            cached = BinaryFolderImaging(folder_path=folder)

        elif format == "memory":
            raise NotImplementedError
        elif format == "zarr":
            raise NotImplementedError
        elif format == "nwb":
            # TODO implement a format based on zarr
            raise NotImplementedError

        else:
            raise ValueError(f"format {format} not supported")

        for segment_index in range(self.get_num_segments()):
            if self.has_time_vector(segment_index):
                # the use of get_times is preferred since timestamps are converted to array
                time_vector = self.get_times(segment_index=segment_index)
                cached.set_times(time_vector, segment_index=segment_index)

        return cached


class BaseImagingSegment(BaseSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """

    def __init__(self, sampling_frequency=None, t_start=None, time_vector=None):
        # sampling_frequency and time_vector are exclusive
        if sampling_frequency is None:
            assert time_vector is not None, "Pass either 'sampling_frequency' or 'time_vector'"
            assert time_vector.ndim == 1, "time_vector should be a 1D array"

        if time_vector is None:
            assert sampling_frequency is not None, "Pass either 'sampling_frequency' or 'time_vector'"

        self.sampling_frequency = sampling_frequency
        self.t_start = t_start
        self.time_vector = time_vector

        BaseSegment.__init__(self)

    def get_times(self) -> np.ndarray:
        if self.time_vector is not None:
            self.time_vector = np.asarray(self.time_vector)
            return self.time_vector
        else:
            time_vector = np.arange(self.get_num_samples(), dtype="float64")
            time_vector /= self.sampling_frequency
            if self.t_start is not None:
                time_vector += self.t_start
            return time_vector

    def get_start_time(self) -> float:
        if self.time_vector is not None:
            return self.time_vector[0]
        else:
            return self.t_start if self.t_start is not None else 0.0

    def get_end_time(self) -> float:
        if self.time_vector is not None:
            return self.time_vector[-1]
        else:
            t_stop = (self.get_num_samples() - 1) / self.sampling_frequency
            if self.t_start is not None:
                t_stop += self.t_start
            return t_stop

    def get_times_kwargs(self) -> dict:
        """
        Retrieves the timing attributes characterizing a RecordingSegment

        Returns
        -------
        dict
            A dictionary containing the following key-value pairs:

            - "sampling_frequency" : The sampling frequency of the RecordingSegment.
            - "t_start" : The start time of the RecordingSegment.
            - "time_vector" : The time vector of the RecordingSegment.

        Notes
        -----
        The keys are always present, but the values may be None.
        """
        time_kwargs = dict(
            sampling_frequency=self.sampling_frequency,
            t_start=self.t_start,
            time_vector=self.time_vector,
        )
        return time_kwargs

    def sample_index_to_time(self, sample_ind):
        """
        Transform sample index into time in seconds
        """
        if self.time_vector is None:
            time_s = sample_ind / self.sampling_frequency
            if self.t_start is not None:
                time_s += self.t_start
        else:
            time_s = self.time_vector[sample_ind]
        return time_s

    def time_to_sample_index(self, time_s):
        """
        Transform time in seconds into sample index
        """
        if self.time_vector is None:
            if self.t_start is None:
                sample_index = time_s * self.sampling_frequency
            else:
                sample_index = (time_s - self.t_start) * self.sampling_frequency
            sample_index = np.round(sample_index).astype(np.int64)
        else:
            sample_index = np.searchsorted(self.time_vector, time_s, side="right") - 1

        return sample_index

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal segment

        Returns:
            SampleIndex : Number of samples in the signal segment
        """
        # must be implemented in subclass
        raise NotImplementedError

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> np.ndarray:
        """
        Return the raw series, optionally for a subset of samples

        Parameters
        ----------
        start_frame : int | None, default: None
            start sample index, or zero if None
        end_frame : int | None, default: None
            end_sample, or number of samples if None

        Returns
        -------
        series : np.ndarray
            Array of series, num_samples x height x width
        """
        # must be implemented in subclass
        raise NotImplementedError
