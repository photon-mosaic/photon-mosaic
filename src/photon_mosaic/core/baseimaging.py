from math import prod

import numpy as np
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.chunkable import ChunkableMixin, ChunkableSegment
from spikeinterface.core.chunkable_tools import get_chunks, write_binary
from spikeinterface.core.core_tools import convert_bytes_to_str, convert_seconds_to_str

from photon_mosaic.core.utils import DtypeType

# TODO: frames instead of samples
# TODO: epoch instead of segment (segmentation is another thing)


class BaseImaging(BaseExtractor, ChunkableMixin):
    """
    Base class for imaging extractors.

    The class inherits from `BaseExtractor` and `ChunkableMixin` to provide common functionality
    for imaging data handling.

    Each `BaseImaging` instance is associated to a single "channel".
    The `_main_ids` attribute is used here for multi-plane imaging objects.
    """

    def __init__(self, sampling_frequency: float, dtype: DtypeType, shape: tuple | list | np.ndarray, plane_ids: list | None = None):
        if plane_ids is None:
            plane_ids = [0]  # dummy single plane
        BaseExtractor.__init__(self, plane_ids)
        self._sampling_frequency = float(sampling_frequency)
        assert len(shape) == 2, "Shape must be a tuple/list/array of length 2 (height, width)"
        self._image_shape = np.array(shape)
        self._dtype = dtype
        self._average_image = None
        self._segments_dask = []

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
            sampling_frequency_repr = f"{sf_hz:0.1f} Hz"

        # Calculate duration
        durations = [ns / sf_hz for ns in num_samples]
        duration_repr = [convert_seconds_to_str(duration) for duration in durations]

        # Calculate memory size using product of all dimensions in image_size
        memory_sizes = [ns * prod(image_shape) * dtype.itemsize for ns in num_samples]
        memory_repr = [convert_bytes_to_str(memory_size) for memory_size in memory_sizes]

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
                duration_str = convert_seconds_to_str(duration)
                memory_size_str = convert_bytes_to_str(memory_size)
                html_segments += (
                    f"<li> Samples: {samples_str}, Duration: {duration_str}, Memory: {memory_size_str}</li>"
                )

            html_segments += "</ol></details>"

        html_extra = self._get_common_repr_html(common_style)
        # remove properties from html_extra
        if "<summary><strong>Properties</strong></summary>" in html_extra:
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

    def add_segment(self, segment) -> None:
        import dask.array as da

        darr_segment = da.from_array(
            segment,
            asarray=False         # critical: don't auto-coerce eagerly
        )
        self._segments_dask.append(darr_segment)
        BaseExtractor.add_segment(self, segment)

    @property
    def image_shape(self):
        """Get the shape of the images (height, width).

        Returns
        -------
        tuple
            The shape of the images as (height, width).
        """
        return self._image_shape

    @property
    def plane_ids(self):
        """Get the plane IDs associated with the imaging data.

        Returns
        -------
        list
            A list of plane IDs.
        """
        return self._main_ids

    @property
    def sampling_frequency(self):
        """Get the sampling frequency of the imaging object.

        Returns
        -------
        float
            The sampling frequency in Hz.
        """
        return self._sampling_frequency

    @property
    def num_planes(self):
        """Get the number of planes in the imaging data.

        Returns
        -------
        int
            The number of planes.
        """
        return len(self.plane_ids)

    @property
    def dask_segments(self):
        """Get the Dask arrays for each imaging segment.

        Returns
        -------
        list
            A list of Dask arrays, one for each imaging segment.
        """
        return self._segments_dask

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_sample_size_in_bytes(self):
        return self.get_num_pixels() * self.get_num_planes() * np.dtype(self.get_dtype()).itemsize

    def get_shape(self, segment_index: int | None = None) -> tuple:
        """Get the shape of the imaging data as (num_samples, height, width).

        Parameters
        ----------
        segment_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.

        Returns
        -------
        tuple
            The shape of the imaging data as (num_samples, height, width).
        """
        if segment_index is None:
            if self.get_num_segments() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        num_samples = self.get_num_samples(segment_index=segment_index)
        if self.get_num_planes() > 1:
            return (num_samples, *self.image_shape, self.get_num_planes())
        else:
            return (num_samples, *self.image_shape)

    def get_data(self, start_frame: int, end_frame: int, segment_index: int | None = None, **kwargs) -> np.ndarray:
        return self.get_series(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)

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
        return self.segments[segment_index].get_num_samples()

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
        return len(self.segments)

    def get_dtype(self) -> DtypeType:
        """Get the data type of the video.

        Returns
        -------
        dtype: dtype
            Data type of the video.
        """
        return self._dtype

    def get_num_pixels(self) -> int:
        """Get the number of pixels in the image.

        Returns
        -------
        int
            Number of pixels in the image.
        """
        return np.prod(self.image_shape)

    def get_num_planes(self) -> int:
        """Get the number of planes in the imaging data.

        Returns
        -------
        int
            The number of planes.
        """
        return len(self.plane_ids)

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        plane_ids: list[int] | None = None,
        segment_index: int | None = None,
    ) -> np.ndarray:
        """Get a series of frames from the imaging data.

        Parameters
        ----------
        start_frame : int
            The starting frame index (inclusive).
        end_frame : int
            The ending frame index (exclusive).
        plane_ids : list[int] | None
            The list of plane IDs to include. If None, all planes are included.
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
        if plane_ids is None:
            plane_indices = slice(self.get_num_planes())
        else:
            plane_indices = self.ids_to_indices(plane_ids)
        return self._segments_dask[segment_index][start_frame:end_frame, ..., plane_indices].compute()

    def __getitem__(self, idx):
        """
        Get item(s) from the imaging data using numpy-like indexing.

        The first index corresponds to the segment index, and subsequent indices correspond to
        frame and spatial dimensions.
        """
        segment_index = idx[0] if len(idx) > 0 else 0
        dask_segment = self._segments_dask[segment_index]
        return dask_segment[idx[1:]].compute()

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
            data = get_chunks(
                self,
                num_chunks_per_segment=num_chunks,
                chunk_duration=chunk_duration,
                chunk_size=chunk_size,
                concatenated=True,
            )
            self._average_image = np.mean(data, axis=0)
            return self._average_image

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

    def get_binary_description(self):  # pragma: no cover
        """
        When `rec.is_binary_compatible()` is True
        this returns a dictionary describing the binary format.
        """
        if not self.is_binary_compatible:
            raise NotImplementedError

    def _save(self, format="binary", verbose: bool = False, **save_kwargs):  # pragma: no cover
        from spikeinterface.core.job_tools import split_job_kwargs

        kwargs, job_kwargs = split_job_kwargs(save_kwargs)

        if format == "binary":
            folder = kwargs["folder"]
            file_paths = [folder / f"video_cached_seg{i}.raw" for i in range(self.get_num_segments())]
            dtype = kwargs.get("dtype", None) or self.get_dtype()
            t_starts = self._get_t_starts()

            write_binary(self, file_paths=file_paths, dtype=dtype, verbose=verbose, **job_kwargs)

            from .binaryimaging import BinaryFolderImaging, BinaryImaging

            # This is created so it can be saved as json because the `BinaryFolderRecording` requires it loading
            # See the __init__ of `BinaryFolderRecording`
            binary_imaging = BinaryImaging(
                file_paths=file_paths,
                sampling_frequency=self.get_sampling_frequency(),
                image_shape=self.image_shape,
                num_planes=self.get_num_planes(),
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


class BaseImagingSegment(ChunkableSegment):
    """
    Abstract class representing a multichannel timeseries, or block of raw ephys traces
    """
    def __init__(self, dtype, sample_shape, sampling_frequency=None, t_start=None, time_vector=None):
        super().__init__(sampling_frequency, t_start, time_vector)
        self._dtype = dtype
        self._sample_shape = sample_shape

    @property
    def dtype(self) -> DtypeType:
        """
        Data type of the imaging segment.

        Returns
        -------
        dtype: dtype
            Data type of the imaging segment.
        """
        return self._dtype

    @property
    def shape(self) -> tuple:
        """
        Shape of the imaging segment as (num_samples, height, width).

        Returns
        -------
        tuple
            Shape of the imaging segment.
        """
        return (self.get_num_samples(),) + self._sample_shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions of the imaging segment.

        Returns
        -------
        int
            Number of dimensions.
        """
        return len(self.shape)

    def __getitem__(self, idx):
        """
        Implement numpy-like fancy indexing for the imaging segment.
        
        Supports indexing along dimensions for both 3D (samples, height, width) 
        and 4D (samples, height, width, depth) data.
        
        Parameters
        ----------
        idx : int, slice, tuple, list, np.ndarray
            Index specification following numpy conventions. Can be:
            - Single integer: select one frame
            - Slice: select a range of frames
            - Tuple of indices: multi-dimensional indexing
            - List or array: fancy indexing
            
        Returns
        -------
        np.ndarray
            The indexed data.
            
        Examples
        --------
        >>> segment[0]  # First frame
        >>> segment[10:20]  # Frames 10-20
        >>> segment[10:20, :, :]  # Frames 10-20, all spatial dims
        >>> segment[:, 5:10, 5:10]  # All frames, spatial crop
        >>> segment[[0, 5, 10]]  # Specific frames via fancy indexing
        """
        # Get the full data - subclasses should implement get_series efficiently
        # For now, we need to determine the range from the index
        
        # Normalize idx to a tuple
        if not isinstance(idx, tuple):
            idx = (idx,)
        
        # Determine which frames to retrieve
        frame_idx = idx[0] if len(idx) > 0 else slice(None)
        
        # Convert frame index to start/end range
        num_samples = self.get_num_samples()
        
        if isinstance(frame_idx, int):
            # Single frame
            if frame_idx < 0:
                frame_idx = num_samples + frame_idx
            start_frame = frame_idx
            end_frame = frame_idx + 1
        elif isinstance(frame_idx, slice):
            # Slice of frames
            start_frame, end_frame, step = frame_idx.indices(num_samples)
            # Note: if step != 1, we'll need to handle it after retrieval
        elif isinstance(frame_idx, (list, np.ndarray)):
            # Fancy indexing - we need to get data and then index
            # For efficiency, determine the min/max range
            frame_array = np.asarray(frame_idx)
            if frame_array.dtype == bool:
                # Boolean indexing
                frame_array = np.where(frame_array)[0]
            start_frame = 0
            end_frame = num_samples
        else:
            start_frame = 0
            end_frame = num_samples
        
        # Get the data
        data = self.get_series(start_frame=start_frame, end_frame=end_frame)
        
        # Apply the full indexing
        if isinstance(frame_idx, int):
            # Already got the right frame, just need to apply spatial indexing
            result = data[0]  # Remove the frame dimension
            if len(idx) > 1:
                # Apply remaining indices to spatial dimensions
                result = result[idx[1:]]
        elif isinstance(frame_idx, slice):
            # Handle step for slice
            _, _, step = frame_idx.indices(num_samples)
            if step != 1:
                data = data[::step]
            # Apply full indexing
            if len(idx) > 1:
                result = data[(slice(None),) + idx[1:]]
            else:
                result = data
        elif isinstance(frame_idx, (list, np.ndarray)):
            # Fancy indexing for frames
            frame_array = np.asarray(frame_idx)
            if frame_array.dtype == bool:
                frame_array = np.where(frame_array)[0]
            # Adjust indices if we didn't start from 0
            adjusted_indices = frame_array - start_frame
            data = data[adjusted_indices]
            if len(idx) > 1:
                result = data[(slice(None),) + idx[1:]]
            else:
                result = data
        else:
            # Default: apply all indices
            result = data[idx]
        
        return result

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: list[int] | None = None,
    ) -> np.ndarray:  # pragma: no cover
        """
        Return the raw series, optionally for a subset of samples

        Parameters
        ----------
        start_frame : int | None, default: None
            start sample index, or zero if None
        end_frame : int | None, default: None
            end_sample, or number of samples if None
        plane_indices : list[int] | None, default: None
            List of plane indices to include, or all planes if None

        Returns
        -------
        series : np.ndarray
            Array of series, num_samples x height x width
        """
        # must be implemented in subclass
        raise NotImplementedError
