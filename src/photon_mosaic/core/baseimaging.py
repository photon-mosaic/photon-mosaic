from math import prod

import numpy as np
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.chunkable import ChunkableMixin, ChunkableSegment
from spikeinterface.core.chunkable_tools import get_chunks, write_binary

from photon_mosaic.core.utils import DtypeType, _convert_bytes_to_str, _convert_seconds_to_str

# from .imaging_tools import write_binary_imaging, get_random_data_chunks

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

    def __init__(self, sampling_frequency: float, shape: tuple | list | np.ndarray, plane_ids: list | None = None):
        if plane_ids is None:
            plane_ids = [0]  # dummy single plane
        BaseExtractor.__init__(self, plane_ids)
        self._sampling_frequency = float(sampling_frequency)
        assert len(shape) == 2, "Shape must be a tuple/list/array of length 2 (height, width)"
        self._image_shape = np.array(shape)
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
            sampling_frequency_repr = f"{sf_hz:0.1f} Hz"

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
        return self._sampling_frequency

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
        return self.get_series(start_frame=0, end_frame=2, segment_index=0).dtype

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
            plane_indices = range(self.get_num_planes())
        else:
            plane_indices = self.ids_to_indices(plane_ids)
        return self.segments[segment_index].get_series(start_frame, end_frame, plane_indices)

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

    def add_imaging_segment(self, imaging_segment):
        """Adds an imaging segment.

        Parameters
        ----------
        imaging_segment : BaseImagingSegment
            The imaging segment to add.
        """
        self.segments.append(imaging_segment)
        imaging_segment.set_parent_extractor(self)

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
