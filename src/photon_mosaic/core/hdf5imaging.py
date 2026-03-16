"""Imaging classes for HDF5-backed data.

Classes
-------
HDF5Imaging
    An Imaging object backed by an HDF5 dataset on disk.
    Supports lazy loading — data is only read when ``get_series()`` is called.
HDF5ImagingEpoch
    A single epoch of HDF5-backed imaging data.
"""

import h5py
import numpy as np

from .baseimaging import BaseImaging, BaseImagingEpoch


class HDF5ImagingEpoch(BaseImagingEpoch):
    """A single epoch of imaging data backed by an h5py dataset.

    Data is read lazily — only the requested hyperslab is loaded from disk
    when ``get_series()`` is called.

    Parameters
    ----------
    h5_dataset : h5py.Dataset
        An open h5py dataset with shape ``(num_frames, height, width)``.
    sampling_frequency : float
        Sampling frequency in Hz.
    """

    def __init__(self, h5_dataset: h5py.Dataset, sampling_frequency: float):
        super().__init__(sampling_frequency=sampling_frequency)
        self._dset = h5_dataset

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        plane_indices: list | None = None,
        row_range: tuple[int, int] | None = None,
        col_range: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Read a subset of frames from the HDF5 dataset.

        Parameters
        ----------
        start_frame : int | None
            Starting frame index (inclusive). Defaults to 0.
        end_frame : int | None
            Ending frame index (exclusive). Defaults to total number of frames.
        plane_indices : list | None
            Plane indices to select. Defaults to all planes.
        row_range : tuple[int, int] | None
            ``(row_start, row_end)`` spatial row selection. Defaults to full height.
        col_range : tuple[int, int] | None
            ``(col_start, col_end)`` spatial column selection. Defaults to full width.

        Returns
        -------
        np.ndarray
            Array with shape ``(num_frames, height, width, 1)``.
        """
        s = start_frame if start_frame is not None else 0
        e = end_frame if end_frame is not None else self._dset.shape[0]
        r0, r1 = row_range if row_range is not None else (0, self._dset.shape[1])
        c0, c1 = col_range if col_range is not None else (0, self._dset.shape[2])

        # h5py reads only the requested hyperslab from disk
        data = self._dset[s:e, r0:r1, c0:c1]
        # Expand to 4D: (frames, H, W) -> (frames, H, W, 1)
        data = data[:, :, :, np.newaxis]

        if plane_indices is not None:
            data = data[:, :, :, plane_indices]
        return data

    def get_num_samples(self) -> int:
        """Return the number of frames in the dataset."""
        return self._dset.shape[0]


class HDF5Imaging(BaseImaging):
    """An Imaging object backed by an HDF5 file on disk.

    The HDF5 dataset is expected to have shape ``(num_frames, height, width)``
    (single-plane). Data is loaded lazily via ``get_series()``.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 file.
    dataset_key : str
        Key of the dataset within the HDF5 file. Default: ``"data"``.
    sampling_frequency : float
        Sampling frequency in Hz. Default: ``30.0``.
    """

    def __init__(self, h5_path: str, dataset_key: str = "data", sampling_frequency: float = 30.0):
        self._h5_file = h5py.File(h5_path, "r")
        self._dset = self._h5_file[dataset_key]
        h, w = self._dset.shape[1], self._dset.shape[2]

        BaseImaging.__init__(self, sampling_frequency=sampling_frequency, shape=(h, w, 1))
        self.add_epoch(HDF5ImagingEpoch(self._dset, sampling_frequency=sampling_frequency))

        self._kwargs = dict(
            h5_path=h5_path,
            dataset_key=dataset_key,
            sampling_frequency=sampling_frequency,
        )

    def __del__(self):
        """Close the HDF5 file handle when the object is garbage collected."""
        if hasattr(self, "_h5_file") and self._h5_file:
            self._h5_file.close()
