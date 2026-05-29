import json
import mmap
import warnings
from pathlib import Path

import numpy as np

from .baseimaging import BaseImaging, BaseImagingEpoch


def _serialize_file_paths(epoch_entries):
    """Convert per-epoch path entries to JSON-friendly absolute strings.

    Each entry is either a Path (interleaved blob) or a list[Path] (per-plane).
    """
    out = []
    for entry in epoch_entries:
        if isinstance(entry, list):
            out.append([str(Path(p).absolute()) for p in entry])
        else:
            out.append(str(Path(entry).absolute()))
    return out


class BinaryImaging(BaseImaging):
    """
    ImagingExtractor for a binary format.

    Each epoch is either backed by a single interleaved ``(T, H, W, n_planes)``
    binary file, or by ``n_planes`` per-plane ``(T, H, W)`` binary files that
    are stitched along the plane axis on read. The two modes can be mixed
    across epochs (one epoch may be interleaved while another is per-plane),
    though all epochs must share the same ``shape`` and ``dtype``.

    Parameters
    ----------
    file_paths : str | Path | list[str | Path | list[str | Path]]
        Path(s) to the binary file(s). A single ``str``/``Path`` is one epoch
        with an interleaved blob. A list with one entry per epoch may contain
        either a ``str``/``Path`` (interleaved blob) or a ``list`` of
        ``n_planes`` paths, one per plane (per-plane mode).
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
        Applied identically to every file (including per-plane files).

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
    ):
        BaseImaging.__init__(self, sampling_frequency, shape)

        per_epoch = file_paths if isinstance(file_paths, list) else [file_paths]
        # Normalize each entry: a list element is per-plane mode, anything else is a single interleaved file
        epoch_entries: list[Path | list[Path]] = []
        for entry in per_epoch:
            if isinstance(entry, list):
                epoch_entries.append([Path(p) for p in entry])
            else:
                epoch_entries.append(Path(entry))

        if t_starts is not None:
            assert len(t_starts) == len(epoch_entries), "t_starts must be a list of the same size as file_paths"
            t_starts = [float(t_start) for t_start in t_starts]

        dtype = np.dtype(dtype)
        n_planes = self._shape[2]

        for i, entry in enumerate(epoch_entries):
            t_start = None if t_starts is None else t_starts[i]
            if isinstance(entry, list):
                if len(entry) != n_planes:
                    raise ValueError(
                        f"Per-plane file list for epoch {i} has {len(entry)} files "
                        f"but shape declares {n_planes} planes"
                    )
                imaging_epoch = BinaryMultiPlaneImagingEpoch(
                    entry,
                    sampling_frequency,
                    t_start,
                    self._shape,
                    dtype,
                    file_offset,
                )
            else:
                imaging_epoch = BinaryImagingEpoch(
                    entry,
                    sampling_frequency,
                    t_start,
                    self._shape,
                    dtype,
                    file_offset,
                )
            self.add_epoch(imaging_epoch)

        self._kwargs = {
            "file_paths": _serialize_file_paths(epoch_entries),
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


class BinaryMultiPlaneImagingEpoch(BaseImagingEpoch):
    """A binary epoch whose planes live in separate ``(T, H, W)`` files.

    Each plane is memory-mapped independently and stacked along the plane axis
    in :meth:`get_series`. Used to back imaging objects whose source format
    stores planes as parallel binary files (e.g. suite2p's per-plane
    ``data.bin``) without copying into an interleaved blob.
    """

    def __init__(self, file_paths, sampling_frequency, t_start, shape, dtype, file_offset):
        BaseImagingEpoch.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        if len(shape) == 2:
            shape = (shape[0], shape[1], 1)
        if len(shape) != 3:
            raise ValueError(f"shape must be (H, W) or (H, W, n_planes); got {shape}")
        self.shape = tuple(shape)
        n_planes = self.shape[2]
        if len(file_paths) != n_planes:
            raise ValueError(f"Got {len(file_paths)} per-plane files but shape declares {n_planes} planes")

        self.dtype = np.dtype(dtype)
        self.file_offset = file_offset
        self.file_paths = [Path(p) for p in file_paths]
        self.files = [open(p, "rb") for p in self.file_paths]
        self.bytes_per_plane_sample = self.shape[0] * self.shape[1] * self.dtype.itemsize

        plane_sample_counts = []
        for p in self.file_paths:
            data_size = p.stat().st_size - file_offset
            plane_sample_counts.append(data_size // self.bytes_per_plane_sample)
        if len(set(plane_sample_counts)) != 1:
            raise ValueError(
                f"Per-plane binary files disagree on sample count: {plane_sample_counts} (paths={self.file_paths})"
            )
        self.num_samples = int(plane_sample_counts[0])

    def get_num_samples(self) -> int:
        return self.num_samples

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: slice | np.ndarray | None = None,
    ) -> np.ndarray:
        n_planes = self.shape[2]
        if plane_indices is None:
            planes_to_read = list(range(n_planes))
        elif isinstance(plane_indices, slice):
            planes_to_read = list(range(*plane_indices.indices(n_planes)))
        else:
            planes_to_read = list(plane_indices)

        n_frames = end_frame - start_frame
        out = np.empty((n_frames, self.shape[0], self.shape[1], len(planes_to_read)), dtype=self.dtype)

        if n_frames == 0:
            return out

        start_byte = self.file_offset + start_frame * self.bytes_per_plane_sample
        length = n_frames * self.bytes_per_plane_sample

        for i, plane_idx in enumerate(planes_to_read):
            f = self.files[plane_idx]
            memmap_offset, start_offset = divmod(start_byte, mmap.ALLOCATIONGRANULARITY)
            memmap_offset *= mmap.ALLOCATIONGRANULARITY
            mm_length = length + start_offset
            memmap_obj = mmap.mmap(
                f.fileno(),
                length=mm_length,
                access=mmap.ACCESS_READ,
                offset=memmap_offset,
            )
            plane_block = np.ndarray(
                shape=(n_frames, self.shape[0], self.shape[1]),
                dtype=self.dtype,
                buffer=memmap_obj,
                offset=start_offset,
            )
            out[..., i] = plane_block

        return out

    def __del__(self):  # pragma: no cover
        try:
            for f in getattr(self, "files", []):
                if f and not f.closed:
                    f.close()
        except Exception as e:
            warnings.warn(f"Error closing file handle in BinaryMultiPlaneImagingEpoch: {e}")


class BinaryImagingEpoch(BaseImagingEpoch):
    def __init__(self, file_path, sampling_frequency, t_start, shape, dtype, file_offset):
        BaseImagingEpoch.__init__(self, sampling_frequency=sampling_frequency, t_start=t_start)
        # Always normalize to a 3-tuple (H, W, n_planes)
        if len(shape) == 2:
            shape = (shape[0], shape[1], 1)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.file_offset = file_offset
        self.file_path = file_path
        self.file = open(self.file_path, "rb")
        self.bytes_per_sample = int(np.prod(self.shape)) * self.dtype.itemsize
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
            shape=self._bin_kwargs["shape"],
            file_offset=self._bin_kwargs["file_offset"],
        )
        return d


read_binary_folder = BinaryFolderImaging
