from math import prod

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.chunkable import ChunkableMixin, ChunkableSegment
from spikeinterface.core.chunkable_tools import get_chunks, write_binary
from spikeinterface.core.core_tools import convert_bytes_to_str, convert_seconds_to_str


class BaseImaging(BaseExtractor, ChunkableMixin):
    """
    Base class for imaging extractors.

    The class inherits from `BaseExtractor` and `ChunkableMixin` to provide common functionality
    for imaging data handling.

    Each `BaseImaging` instance is associated to a single "channel".
    The `_main_ids` attribute is used here for multi-plane imaging objects.
    """

    def __init__(self, sampling_frequency: float, shape: tuple | list | ArrayLike):
        # Should we allow users to provide 2D shape (H, W) for single plane imaging?
        if len(shape) == 2:
            shape = (shape[0], shape[1], 1)
        assert len(shape) == 3, "Shape must be a tuple/list/array of length 3 (height, width, planes)"
        num_planes = shape[2]
        BaseExtractor.__init__(self, range(0, num_planes))
        self._sampling_frequency = float(sampling_frequency)
        self._shape = tuple(shape)  # Image is intended as a volume (H, W, planes)
        self._average_image = None

    def _repr_header(self, display_name=True):
        """Generate text representation of the BaseImaging object."""
        num_frames = [self.get_num_frames(epoch_index=i) for i in range(self.get_num_epochs())]
        shape = self._shape
        dtype = self.get_dtype()
        sf_hz = self.sampling_frequency

        # Format sampling frequency
        if not sf_hz.is_integer():
            sampling_frequency_repr = f"{sf_hz:f} Hz"
        else:
            sampling_frequency_repr = f"{sf_hz:0.1f} Hz"

        # Calculate duration
        durations = [ns / sf_hz for ns in num_frames]
        duration_repr = [convert_seconds_to_str(duration) for duration in durations]

        # Calculate memory size using product of all dimensions in image_size
        memory_sizes = [ns * prod(shape) * dtype.itemsize for ns in num_frames]
        memory_repr = [convert_bytes_to_str(memory_size) for memory_size in memory_sizes]

        if self.get_num_epochs() == 1:
            num_frames = num_frames[0]
            duration_repr = duration_repr[0]
            memory_repr = memory_repr[0]

        if display_name and self.name != self.__class__.__name__:
            name = f"{self.name} ({self.__class__.__name__})"
        else:
            name = self.__class__.__name__

        # Format shape string based on whether data is volumetric or not
        shape_repr = f"{shape[0]} rows x {shape[1]} columns "
        return (
            f"{name}:\n"
            f"{sampling_frequency_repr} - "
            f"{self.get_num_epochs()} epochs - "
            f"{shape_repr} - "
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

        html_epochs = ""
        if self.get_num_epochs() > 1:
            html_epochs += f"<details style='{common_style}'>  <summary><strong>Epochs</strong></summary><ol>"
            for epoch_index in range(self.get_num_epochs()):
                samples = self.get_num_samples(epoch_index)
                duration = self.get_duration(epoch_index)
                memory_size = self.get_memory_size(epoch_index)
                samples_str = f"{samples:,}"
                duration_str = convert_seconds_to_str(duration)
                memory_size_str = convert_bytes_to_str(memory_size)
                html_epochs += f"<li> Samples: {samples_str}, Duration: {duration_str}, Memory: {memory_size_str}</li>"

            html_epochs += "</ol></details>"

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
        html_repr = html_header + html_epochs + html_extra
        return html_repr

    @property
    def shape(self):
        """Get the shape of the images (height, width).

        Returns
        -------
        tuple
            The shape of the images as (height, width).
        """
        return self._shape

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
    def epochs(self):
        """Get the epochs (segments) of the imaging data.

        Returns
        -------
        list
            A list of epochs (segments) in the imaging data.
        """
        return self.segments

    def add_epoch(self, epoch: "BaseImagingEpoch"):
        """Add an epoch (segment) to the imaging data.

        Parameters
        ----------
        epoch : BaseImagingEpoch
            The epoch (segment) to add to the imaging data.
        """
        self.add_segment(epoch)

    def get_sampling_frequency(self):
        return self._sampling_frequency

    def get_sample_size_in_bytes(self):
        return self.get_num_pixels() * np.dtype(self.get_dtype()).itemsize

    def get_shape(self, segment_index: int | None = None) -> tuple:
        """Get the shape of the imaging data as (num_samples, height, width, planes).
        Used internally for SpikeInterface chunk processing.

        Parameters
        ----------
        segment_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.

        Returns
        -------
        tuple
            The shape of the imaging data as (num_samples, height, width, planes).
        """
        if segment_index is None:
            if self.get_num_epochs() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        num_samples = self.get_num_samples(segment_index=segment_index)

        return (num_samples, *self.shape)

    def get_data(self, start_frame: int, end_frame: int, segment_index: int | None = None, **kwargs) -> np.ndarray:
        """Internal function to return data for SpikeInterface chunk processing.

        Parameters
        ----------
        start_frame : int
            The starting frame index (inclusive).
        end_frame : int
            The ending frame index (exclusive).
        segment_index : int | None, optional
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.

        Returns
        -------
        np.ndarray
            The requested series of frames as a NumPy array, with shape (num_samples, height, width, planes).
        """
        return self.get_series(start_frame=start_frame, end_frame=end_frame, epoch_index=segment_index)

    def get_num_samples(self, segment_index: int | None = None) -> int:
        """Get the number of samples (frames) in the imaging segment.

        Parameters
        ----------
        segment_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.
        Returns
        -------
        int
            The number of samples (frames) in the segment.
        """
        if segment_index is None:
            if self.get_num_epochs() == 1:
                segment_index = 0
            else:
                raise ValueError("segment_index must be provided for multi-segment imaging data.")
        return self.segments[segment_index].get_num_samples()

    def get_num_frames(self, epoch_index: int | None = None) -> int:
        """Get the total number of frames in the imaging data.

        Parameters
        ----------

        Returns
        -------
        int
            The total number of frames.
        """
        return self.get_num_samples(segment_index=epoch_index)

    def get_total_frames(self) -> int:
        """Get the total number of frames across all segments.

        Returns
        -------
        int
            The total number of frames across all segments.
        """
        return self.get_total_samples()

    def get_num_segments(self) -> int:  # pragma: no cover
        """Get the number of imaging segments.
        This is needed for SpikeInterface compatibility, but the photon-mosaic nomenclature is
        "epochs" instead. Use `get_num_epochs()` is preferred.

        Returns
        -------
        int
            The number of imaging segments.
        """
        return len(self.segments)

    def get_num_epochs(self) -> int:
        """Get the number of imaging epochs.

        Returns
        -------
        int
            The number of imaging epochs.
        """
        return len(self.segments)

    def get_dtype(self) -> DTypeLike:
        """Get the data type of the video.

        Returns
        -------
        dtype: dtype
            Data type of the video.
        """
        return self.get_series(start_frame=0, end_frame=2, epoch_index=0).dtype

    def get_num_pixels(self) -> int:
        """Get the number of pixels in the image.

        Returns
        -------
        int
            Number of pixels in the image.
        """
        return np.prod(self.shape)

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
        epoch_index: int | None = None,
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
        epoch_index : int | None
            The index of the imaging segment. If None and there is only one segment, it defaults to 0.

        Returns
        -------
        np.ndarray
            The requested series of frames as a NumPy array.
        """
        if epoch_index is None:
            if self.get_num_epochs() == 1:
                epoch_index = 0
            else:
                raise ValueError("epoch_index must be provided for multi-segment imaging data.")
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else self.get_num_frames(epoch_index=epoch_index)
        if plane_ids is None:
            plane_indices = range(self.get_num_planes())
        else:
            plane_indices = self.ids_to_indices(plane_ids)
        return self.epochs[epoch_index].get_series(start_frame, end_frame, plane_indices)

    def get_average_image(
        self,
        num_chunks: int = 20,
        chunk_duration: str = "1s",
        chunk_size: int | None = None,
        recompute: bool = False,
    ) -> np.ndarray:
        """Compute the average image across all frames in the imaging data.

        Parameters
        ----------
        num_chunks : int, default: 20
            The number of chunks to use for computing the average image. The data will be divided into
            this many chunks, and the average will be computed across the chunks to save memory.
        chunk_duration : str, default: "1s"
            The duration of each chunk, specified as a string (e.g., "1s" for 1 second, "500ms" for 500 milliseconds).
        chunk_size : int | None, default: None
            The number of frames in each chunk. If specified, this will override the chunk_duration.
        recompute : bool, default: False
            If True, forces recomputation of the average image even if it has been computed before.

        Returns
        -------
        np.ndarray
            The average image (height, width, num_planes)computed across sampled frames in the imaging data.
        """
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
        Checks if the imaging object is "binary" compatible.
        To be used before calling `imaging.get_binary_description()`

        Returns
        -------
        bool
            True if the underlying imaging object is binary
        """
        # has to be changed in subclass if yes
        return False

    def get_binary_description(self) -> dict:  # pragma: no cover
        """
        When `imaging.is_binary_compatible()` is True
        this returns a dictionary describing the binary format.
        """
        if not self.is_binary_compatible():
            raise NotImplementedError
        return {}

    def _save(self, format="binary", verbose: bool = False, **save_kwargs):  # pragma: no cover
        from spikeinterface.core.job_tools import split_job_kwargs

        kwargs, job_kwargs = split_job_kwargs(save_kwargs)

        if format == "binary":
            folder = kwargs["folder"]
            file_paths = [folder / f"video_cached_seg{i}.raw" for i in range(self.get_num_epochs())]
            dtype = kwargs.get("dtype", None) or self.get_dtype()
            t_starts = self._get_t_starts()

            write_binary(self, file_paths=file_paths, dtype=dtype, verbose=verbose, **job_kwargs)

            from .binaryimaging import BinaryFolderImaging, BinaryImaging

            # This is created so it can be saved as json because the `BinaryFolderImaging` requires it loading
            # See the __init__ of `BinaryFolderImaging`
            binary_imaging = BinaryImaging(
                file_paths=file_paths,
                sampling_frequency=self.get_sampling_frequency(),
                shape=self.shape,
                dtype=dtype,
                t_starts=t_starts,
                file_offset=0,
            )
            binary_imaging.dump(folder / "binary.json", relative_to=folder)

            cached = BinaryFolderImaging(folder_path=folder)

        elif format == "memory":
            raise NotImplementedError
        elif format == "zarr":
            import zarr

            from .zarrimaging import ZarrImaging, add_imaging_to_zarr_group

            folder_path = kwargs["folder"]
            storage_options = kwargs.get("storage_options", None)
            zarr_root = zarr.open(str(folder_path), mode="w", storage_options=storage_options)
            add_imaging_to_zarr_group(self, zarr_root, **kwargs)

            cached = ZarrImaging(zarr_root)
        elif format == "nwb":
            raise NotImplementedError

        else:
            raise ValueError(f"format {format} not supported")

        for epoch_index in range(self.get_num_epochs()):
            if self.has_time_vector(epoch_index):
                # the use of get_times is preferred since timestamps are converted to array
                time_vector = self.get_times(segment_index=epoch_index)
                cached.set_times(time_vector, segment_index=epoch_index)

        return cached


class BaseImagingEpoch(ChunkableSegment):
    """
    Abstract class representing a video epoch.
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
