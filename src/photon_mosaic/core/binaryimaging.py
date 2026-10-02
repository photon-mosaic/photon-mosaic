import json
import mmap
import shutil
import warnings
from pathlib import Path

import numpy as np
from spikeinterface.core.core_tools import (
    load_annotations_from_folder,
    load_properties_from_folder,
    save_annotations_to_folder,
    save_extractor_provenance,
    save_properties_to_folder,
)

from .baseimaging import BaseImaging, BaseImagingEpoch


class BinaryImaging(BaseImaging):
    """
    ImagingExtractor for a binary format

    Parameters
    ----------
    file_paths : str or Path or list
        Path to the binary file
    sampling_frequency : float
        The sampling frequency
    shape : tuple(int, int) | tuple(int, int, int)
        Image height, width, and optionally number of planes
    dtype : str or dtype
        The dtype of the binary file
    t_starts : None or list of float, default: None
        Times in seconds of the first sample for each epoch. If None, defaults to 0 for all epochs.
    file_offset : int, default: 0
        Number of bytes in the file to offset by during memmap instantiation.
    file_timestamps_paths : str or Path or list, default: None
        Path to the binary file containing timestamps for each segment. If None, timestamps are not loaded

    Returns
    -------
    imaging : BinaryImaging
        The imaging object
    """

    def __init__(
        self,
        file_paths,
        sampling_frequency,
        shape,
        dtype,
        t_starts=None,
        file_offset=0,
        file_timestamps_paths: str | Path | list[str | Path] | None = None,
    ):
        BaseImaging.__init__(self, sampling_frequency, shape)

        if isinstance(file_paths, list):
            # several epochs
            file_path_list = [Path(p) for p in file_paths]
        else:
            # one epoch
            file_path_list = [Path(file_paths)]

        if t_starts is not None:
            assert len(t_starts) == len(file_path_list), "t_starts must be a list of the same size as file_paths"
            t_starts = [float(t_start) for t_start in t_starts]

        dtype = np.dtype(dtype)

        if file_timestamps_paths is None:
            timestamps_path_list: list[str | Path] | None = None
        elif isinstance(file_timestamps_paths, list):
            timestamps_path_list = file_timestamps_paths
        else:
            timestamps_path_list = [file_timestamps_paths]

        for i, file_path in enumerate(file_path_list):
            if t_starts is None:
                t_start = None
            else:
                t_start = t_starts[i]
            if timestamps_path_list is None:
                file_timestamps_path = None
            else:
                file_timestamps_path = timestamps_path_list[i]
            imaging_epoch = BinaryImagingEpoch(
                file_path, sampling_frequency, t_start, shape, dtype, file_offset, file_timestamps_path
            )
            self.add_epoch(imaging_epoch)

        self._kwargs = {
            "file_paths": [str(Path(e).absolute()) for e in file_path_list],
            "sampling_frequency": sampling_frequency,
            "t_starts": t_starts,
            "shape": shape,
            "dtype": dtype.str,
            "file_offset": file_offset,
        }

    def is_binary_compatible(self) -> bool:
        return True

    def get_binary_description(self):
        d = dict(
            file_paths=self._kwargs["file_paths"],
            dtype=np.dtype(self._kwargs["dtype"]),
            shape=self._kwargs["shape"],
            file_offset=self._kwargs["file_offset"],
        )
        return d

    def __del__(self):  # pragma: no cover
        """
        Ensures that all epoch resources are properly cleaned up when this imaging extractor is deleted.
        Closes any open file handles in the imaging epochs.
        """
        # Close all imaging epochs
        if hasattr(self, "epochs"):
            for epoch in self.epochs:
                # This will trigger the __del__ method of the BaseImagingEpoch
                # which will close the file handle
                del epoch


class BinaryImagingEpoch(BaseImagingEpoch):
    def __init__(self, file_path, sampling_frequency, t_start, shape, dtype, file_offset, file_timestamps_path):
        BaseImagingEpoch.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.file_offset = file_offset
        self.file_path = file_path
        self.file = open(self.file_path, "rb")
        self.bytes_per_sample = np.prod(shape) * self.dtype.itemsize
        self.data_size_in_bytes = Path(file_path).stat().st_size - file_offset
        self.num_samples = self.data_size_in_bytes // self.bytes_per_sample
        if file_timestamps_path is not None:
            self._time_vector = np.memmap(file_timestamps_path, dtype="float64", mode="r", shape=(self.num_samples,))

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return self.num_samples

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: slice | np.ndarray | None = None,
    ) -> np.ndarray:
        # Calculate byte offsets for start and end frames
        start_byte = self.file_offset + start_frame * self.bytes_per_sample
        end_byte = self.file_offset + end_frame * self.bytes_per_sample

        # Calculate the length of the data chunk to load into memory
        length = end_byte - start_byte

        # The mmap offset must be a multiple of mmap.ALLOCATIONGRANULARITY
        memmap_offset, start_offset = divmod(start_byte, mmap.ALLOCATIONGRANULARITY)
        memmap_offset *= mmap.ALLOCATIONGRANULARITY

        # Adjust the length so it includes the extra data from rounding down
        # the memmap offset to a multiple of ALLOCATIONGRANULARITY
        length += start_offset

        # Create the mmap object
        memmap_obj = mmap.mmap(
            self.file.fileno(),
            length=length,
            access=mmap.ACCESS_READ,
            offset=memmap_offset,
        )

        # Create a numpy array using the mmap object as the buffer
        # Note that the shape must be recalculated based on the new data chunk
        shape: tuple[int, int, int, int]
        shape = (
            (end_frame - start_frame),
            self.shape[0],
            self.shape[1],
            self.shape[2],
            # We could also read only the specific planes here
            # if we implemented more complex memory mapping offsets
        )

        # Now the entire array should correspond to the data between start_frame and end_frame,
        # so we can use it directly
        series = np.ndarray(
            shape=shape,
            dtype=self.dtype,
            buffer=memmap_obj,
            offset=start_offset,
        )

        # Slice planes if needed
        series = series[:, :, :, plane_indices] if plane_indices is not None else series

        return series

    def __del__(self):  # pragma: no cover
        # Ensure that the file handle is closed when the epoch is garbage-collected
        try:
            if hasattr(self, "file") and self.file and not self.file.closed:
                self.file.close()
        except Exception as e:
            warnings.warn(f"Error closing file handle in BaseImagingEpoch: {e}")
            pass


# For backward compatibility (old good time)
read_binary = BinaryImaging


class BinaryFolderImaging(BinaryImaging):
    """
    BinaryFolderImaging is an internal format used in photon-mosaic.
    It is a BinaryImaging + metadata contained in a folder.

    It is created with the function: `imaging.save(format="binary", folder="/myfolder")`

    Parameters
    ----------
    folder_path : str or Path

    Returns
    -------
    imaging : BinaryFolderImaging
        The imaging object
    """

    def __init__(self, folder_path):
        from spikeinterface.core.core_tools import make_paths_absolute

        folder_path = Path(folder_path)

        with open(folder_path / "binary.json", "r") as f:
            d = json.load(f)

        if not d["class"].endswith(".BinaryImaging"):
            raise ValueError("This folder is not a binary photon-mosaic folder")

        assert d["relative_paths"]

        d = make_paths_absolute(d, folder_path)

        BinaryImaging.__init__(self, **d["kwargs"])

        # Load properties and annotations
        load_properties_from_folder(folder_path / "properties", self)
        load_annotations_from_folder(folder_path, self)

        self._kwargs = dict(folder_path=str(Path(folder_path).absolute()))
        self._bin_kwargs = d["kwargs"]

    def is_binary_compatible(self) -> bool:
        return True

    def get_binary_description(self):
        d = dict(
            file_paths=self._bin_kwargs["file_paths"],
            dtype=np.dtype(self._bin_kwargs["dtype"]),
            shape=self._bin_kwargs["shape"],
            file_offset=self._bin_kwargs["file_offset"],
        )
        return d

    @staticmethod
    def write_imaging(
        imaging: BaseImaging,
        folder_path: str | Path,
        verbose: bool = False,
        overwrite: bool = False,
        dtype=None,
        **job_kwargs,
    ):
        """Write imaging data to a folder in binary format.

        Each epoch is written to its own `.raw` file.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging object to write.
        folder_path : str | Path
            Destination folder where binary files are saved.
        verbose : bool, default: False
            If ``True``, enables verbose output during writing.
        overwrite : bool, default: False
            If ``True``, removes an existing destination folder before writing.
        dtype : dtype, optional
            Data type used to store trace data. If ``None``, uses
            ``imaging.get_dtype()``.
        **job_kwargs
            Additional keyword arguments forwarded to
            :func:`spikeinterface.core.time_series_tools.write_binary`.

        Returns
        -------
        Path
            Path to the written `BinaryFolderImaging`.

        Notes
        -----
        Implemented as a static method so it can be called by
        :meth:`BaseImaging.save` without instantiating :class:`BinaryImaging`.
        """
        from spikeinterface.core.time_series_tools import write_binary

        folder_path = Path(folder_path)
        if folder_path.is_dir():
            if not overwrite:
                raise FileExistsError(f"Folder {folder_path} already exists. Use overwrite=True to overwrite it.")
            else:
                shutil.rmtree(folder_path)
        folder_path.mkdir(exist_ok=False, parents=True)

        file_paths = [folder_path / f"traces_cached_seg{i}.raw" for i in range(imaging.get_num_epochs())]
        if dtype is None:
            dtype = imaging.get_dtype()
        # Check if there are any time vectors
        t_starts = imaging.get_segment_t_starts()
        if imaging.has_any_time_vector():
            file_timestamps_paths: list[str | Path] | None = [
                folder_path / f"times_cached_seg{i}.raw" for i in range(imaging.get_num_epochs())
            ]
        else:
            file_timestamps_paths = None

        write_binary(
            imaging,
            file_paths=file_paths,
            file_timestamps_paths=file_timestamps_paths,
            dtype=dtype,
            verbose=verbose,
            **job_kwargs,
        )

        save_extractor_provenance(folder_path, imaging)
        save_properties_to_folder(folder_path / "properties", imaging)
        save_annotations_to_folder(folder_path, imaging)

        # This is created so it can be saved as json because the `BinaryFolderRecording` requires it loading
        # See the __init__
        binary_imaging = BinaryImaging(
            file_paths=file_paths,
            file_timestamps_paths=file_timestamps_paths,
            shape=imaging.shape,
            sampling_frequency=imaging.get_sampling_frequency(),
            dtype=dtype,
            t_starts=t_starts,
            file_offset=0,
        )
        binary_imaging.dump(folder_path / "binary.json", relative_to=folder_path)

        # Create the si_folder file to make the load() easier until version 0.105.0
        # All properties, annotations, and probe information are already saved in the folder,
        # so we don't need to include them in the si_folder.json
        cached = BinaryFolderImaging(folder_path=folder_path)
        si_folder_path = folder_path / "si_folder.json"
        cached.dump_to_json(
            file_path=si_folder_path,
            relative_to=folder_path,
            include_properties=False,
            include_annotations=False,
            include_extra_metadata=False,
        )

        return cached


read_binary_folder = BinaryFolderImaging
