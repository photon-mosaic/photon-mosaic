from __future__ import annotations
import mmap
import warnings
import json
from pathlib import Path

import numpy as np

from .baseimaging import BaseImaging, BaseImagingSegment
from .imaging_tools import write_binary_imaging
from spikeinterface.core.job_tools import _shared_job_kwargs_doc


class BinaryImaging(BaseImaging):
    """
    RecordingExtractor for a binary format

    Parameters
    ----------
    file_paths : str or Path or list
        Path to the binary file
    sampling_frequency : float
        The sampling frequency
    image_shape : tuple(int, int)
        Image height and width
    dtype : str or dtype
        The dtype of the binary file
    time_axis : int, default: 0
        The axis of the time dimension
    t_starts : None or list of float, default: None
        Times in seconds of the first sample for each segment
    file_offset : int, default: 0
        Number of bytes in the file to offset by during memmap instantiation.

    Returns
    -------
    recording : BinaryImaging
        The imaging object
    """

    def __init__(
        self,
        file_paths,
        sampling_frequency,
        dtype,
        image_shape,
        t_starts=None,
        file_offset=0,
    ):
        BaseImaging.__init__(self, sampling_frequency, image_shape)

        if isinstance(file_paths, list):
            # several segment
            file_path_list = [Path(p) for p in file_paths]
        else:
            # one segment
            file_path_list = [Path(file_paths)]

        if t_starts is not None:
            assert len(t_starts) == len(file_path_list), "t_starts must be a list of the same size as file_paths"
            t_starts = [float(t_start) for t_start in t_starts]

        dtype = np.dtype(dtype)

        for i, file_path in enumerate(file_path_list):
            if t_starts is None:
                t_start = None
            else:
                t_start = t_starts[i]
            imaging_segment = BinaryImagingSegment(
                file_path, sampling_frequency, t_start, image_shape, dtype, file_offset
            )
            self.add_imaging_segment(imaging_segment)

        self._kwargs = {
            "file_paths": [str(Path(e).absolute()) for e in file_path_list],
            "sampling_frequency": sampling_frequency,
            "t_starts": t_starts,
            "image_shape": image_shape,
            "dtype": dtype.str,
            "file_offset": file_offset,
        }

    @staticmethod
    def write_imaging(imaging, file_paths, dtype=None, **job_kwargs):
        """
        Save the traces of a recording extractor in binary .dat format.

        Parameters
        ----------
        recording : RecordingExtractor
            The recording extractor object to be saved in .dat format
        file_paths : str
            The path to the file.
        dtype : dtype, default: None
            Type of the saved data
        {}
        """
        write_binary_imaging(imaging, file_paths=file_paths, dtype=dtype, **job_kwargs)

    def is_binary_compatible(self) -> bool:
        return True

    def get_binary_description(self):
        d = dict(
            file_paths=self._kwargs["file_paths"],
            dtype=np.dtype(self._kwargs["dtype"]),
            image_shape=self._kwargs["image_shape"],
            time_axis=self._kwargs["time_axis"],
            file_offset=self._kwargs["file_offset"],
        )
        return d

    def __del__(self):
        """
        Ensures that all segment resources are properly cleaned up when this recording extractor is deleted.
        Closes any open file handles in the recording segments.
        """
        # Close all recording segments
        if hasattr(self, "_recording_segments"):
            for segment in self._imaging_segments:
                # This will trigger the __del__ method of the BinaryRecordingSegment
                # which will close the file handle
                del segment


BinaryImaging.write_imaging.__doc__ = BinaryImaging.write_imaging.__doc__.format(_shared_job_kwargs_doc)


class BinaryImagingSegment(BaseImagingSegment):
    def __init__(self, file_path, sampling_frequency, t_start, image_shape, dtype, file_offset):
        BaseImagingSegment.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        self.image_shape = image_shape
        self.dtype = np.dtype(dtype)
        self.file_offset = file_offset
        self.file_path = file_path
        self.file = open(self.file_path, "rb")
        self.bytes_per_sample = np.prod(image_shape) * self.dtype.itemsize
        self.data_size_in_bytes = Path(file_path).stat().st_size - file_offset
        self.num_samples = self.data_size_in_bytes // self.bytes_per_sample

    def get_num_samples(self) -> int:
        """Returns the number of samples in this signal block

        Returns:
            SampleIndex : Number of samples in the signal block
        """
        return self.num_samples

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
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
        shape = ((end_frame - start_frame), self.image_shape[0], self.image_shape[1])

        # Now the entire array should correspond to the data between start_frame and end_frame, so we can use it directly
        series = np.ndarray(
            shape=shape,
            dtype=self.dtype,
            buffer=memmap_obj,
            offset=start_offset,
        )

        return series

    def __del__(self):
        # Ensure that the file handle is closed when the segment is garbage-collected
        try:
            if hasattr(self, "file") and self.file and not self.file.closed:
                self.file.close()
        except Exception as e:
            warnings.warn(f"Error closing file handle in BinaryImagingSegment: {e}")
            pass


# For backward compatibility (old good time)
read_binary = BinaryImaging


class BinaryFolderImaging(BinaryImaging):
    """
    BinaryFolderImaging is an internal format used in spikeinterface.
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

        folder_metadata = folder_path
        self.load_metadata_from_folder(folder_metadata)

        self._kwargs = dict(folder_path=str(Path(folder_path).absolute()))
        self._bin_kwargs = d["kwargs"]

    def is_binary_compatible(self) -> bool:
        return True

    def get_binary_description(self):
        d = dict(
            file_paths=self._bin_kwargs["file_paths"],
            dtype=np.dtype(self._bin_kwargs["dtype"]),
            image_shape=self._bin_kwargs["image_shape"],
            file_offset=self._bin_kwargs["file_offset"],
        )
        return d


read_binary_folder = BinaryFolderImaging
