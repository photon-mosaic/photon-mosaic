from math import prod

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.core_tools import convert_bytes_to_str, convert_seconds_to_str
from spikeinterface.core.job_tools import split_job_kwargs
from spikeinterface.core.time_series import TimeSeries, TimeSeriesSegment
from spikeinterface.core.time_series_tools import get_chunks


class BaseImaging(BaseExtractor, TimeSeries):
    """
    Base class for imaging extractors.

    The class inherits from `BaseExtractor` and `TimeSeries` to provide common functionality
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
        TimeSeries.__init__(self)
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

    def get_sample_size_in_bytes(self, dtype=None):
        return self.get_num_pixels() * np.dtype(self.get_dtype() if dtype is None else dtype).itemsize

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
        plane_ids: list | np.ndarray | None = None,
        epoch_index: int | None = None,
    ) -> np.ndarray:
        """Get a series of frames from the imaging data.

        Parameters
        ----------
        start_frame : int
            The starting frame index (inclusive).
        end_frame : int
            The ending frame index (exclusive).
        plane_ids : list | np.ndarray | None
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
            plane_indices = slice(self.get_num_planes())
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

    def save(self, format: str = "binary", verbose: bool = False, **save_kwargs):
        """
        Save a `BaseImaging` object to a specified format:

        * "binary"
        * "zarr"
        * "memory" (not implemented)

        Parameters
        ----------
        format : str, default: "binary"
            The format to save the imaging in. Options are:

            - "binary": Saves the imaging in binary format.
            - "zarr": Saves the imaging in Zarr format.
            - "memory": Saves the imaging in memory (shared memory or numpy array).
        verbose : bool, default: False
            If True, prints additional information during the save process.
        **save_kwargs : dict
            Additional keyword arguments specific to the chosen format.
            All formats support job_kwargs for parallel processing
            (see `si.get_global_job_kwargs()` for default values).

            * "binary" format:
                - folder : str or Path
                    The folder where the binary files will be saved.
                - overwrite : bool, default: False
                    If True, existing files in the folder will be overwritten.
                - dtype : str, optional
                    The data type to use for saving the imaging. If not provided, the imaging's dtype
                    will be used.
            * "zarr" format:
                - folder : str or Path
                    The folder where the Zarr files will be saved.
                - overwrite: bool, default: False
                    If True, the folder is removed if it already exists
                - storage_options: dict or None, default: None
                    Storage options for zarr `store`. E.g., if "s3://" or "gcs://" they can
                    provide authentication methods, etc.
                    For cloud storage locations, this should not be None (in case of default values, use an empty dict)
                - dtype: np.dtype or None, default: None
                    The dtype to use for the video datasets. If None, the imaging's dtype is used
                - compressor: numcodecs.Codec or None, default: None
                    Global compressor. If None, Blosc-zstd, level 5, with bit shuffle is used
                - filters: list[numcodecs.Codec] or None, default: None
                    Global filters for zarr (global)
                - compressor_by_dataset: dict or None, default: None
                    Optional compressor per dataset:

                        - videos
                        - times

                    If None, the global compressor is used
                - filters_by_dataset: dict or None, default: None
                    Optional filters per dataset:

                        - videos
                        - times

                    If None, the global filters are used
                - extra_chunks: dict or None, default: None
                    Extra chunk specification passed to the zarr writer
            * "memory" format:
                - sharedmem : bool, default: True
                    If True, the imaging is saved in shared memory. If False, it is saved as
                    a numpy array in memory.

        Returns
        -------
        Baseimaging
            The saved imaging object in the specified format.
        """
        kwargs, job_kwargs = split_job_kwargs(save_kwargs)

        if format == "binary":
            if "folder" not in kwargs:
                raise ValueError("Missing folder in imaging.save(folder='...')")

            from .binaryimaging import BinaryFolderImaging

            folder = kwargs.pop("folder")
            cached = BinaryFolderImaging.write_imaging(
                self, folder_path=folder, verbose=verbose, **kwargs, **job_kwargs
            )
        elif format == "zarr":
            if "folder" not in kwargs:
                raise ValueError("Missing folder in imaging.save(folder='...')")
            folder_path = kwargs.pop("folder")

            from .zarrimaging import ZarrImaging

            cached = ZarrImaging.write_imaging(self, folder_path=folder_path, verbose=verbose, **kwargs, **job_kwargs)
        elif format == "memory":
            # if kwargs.get("sharedmem", True):
            #     from .numpyextractors import SharedMemoryRecording

            #     cached = SharedMemoryRecording.from_recording(
            #         self, with_metadata=True, with_time_vector=True, **job_kwargs
            #     )
            # else:
            #     from spikeinterface.core import NumpyRecording

            #     cached = NumpyRecording.from_recording(self, with_metadata=True, with_time_vector=True, **job_kwargs)
            raise NotImplementedError("Memory format is not implemented yet.")

        else:
            raise ValueError(f"format {format} not supported")

        return cached

    def _extra_metadata_to_dict(self, dump_dict):
        super()._extra_metadata_to_dict(dump_dict)

        # Add times_kwargs if the recording has been modified in memory (e.g. by set_times / shift_times / reset_times)
        if self._time_info_modified:
            dump_dict["times_kwargs"] = []
            for segment_index in range(self.get_num_segments()):
                times_kwargs = self.segments[segment_index].get_times_kwargs()
                dump_dict["times_kwargs"].append(times_kwargs)

    def _extra_metadata_from_dict(self, dump_dict):
        super()._extra_metadata_from_dict(dump_dict)

        if "times_kwargs" in dump_dict:
            # When serializing, dump timestamps information because this could have been
            # set in memory
            times_kwargs_list = dump_dict["times_kwargs"]
            for segment_index, times_kwargs in enumerate(times_kwargs_list):
                self.segments[segment_index]._sampling_frequency = times_kwargs["sampling_frequency"]
                self.segments[segment_index]._t_start = times_kwargs["t_start"]
                self.segments[segment_index]._time_vector = times_kwargs["time_vector"]


class BaseImagingEpoch(TimeSeriesSegment):
    """
    Abstract class representing a video epoch.
    """

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: slice | np.ndarray | None = None,
    ) -> np.ndarray:  # pragma: no cover
        """
        Return the raw series, optionally for a subset of samples

        Parameters
        ----------
        start_frame : int | None, default: None
            start sample index, or zero if None
        end_frame : int | None, default: None
            end_sample, or number of samples if None
        plane_indices : slice | list[int] | None, default: None
            List of plane indices to include, or all planes if None

        Returns
        -------
        series : np.ndarray
            Array of series, num_samples x height x width
        """
        # must be implemented in subclass
        raise NotImplementedError

    def get_data(self, start_frame: int, end_frame: int, indices: list | np.ndarray | None = None) -> np.ndarray:
        """
        General retrieval function for time series objects
        """
        return self.get_series(start_frame=start_frame, end_frame=end_frame, plane_indices=indices)
