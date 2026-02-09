"""Imaging classes for zarr-backed data, read lazily via dask.

Classes
-------
ZarrImaging
    An Imaging backed by a zarr store, read lazily through dask arrays.
"""

from pathlib import Path

import dask.array as da
import numpy as np
import zarr

from .baseimaging import BaseImaging, BaseImagingEpoch


class ZarrImaging(BaseImaging):
    """A single- or multi-epoch Imaging backed by zarr stores.

    Each epoch corresponds to one zarr array on disk, read lazily via
    ``dask.array.from_zarr``.

    Parameters
    ----------
    zarr_paths : str | Path | list[str | Path]
        Path(s) to the zarr store(s). Each store must contain a single array
        with shape ``(num_frames, height, width)`` or
        ``(num_frames, height, width, num_planes)``.
    sampling_frequency : float
        Sampling frequency in Hz.
    chunks : str | tuple | None, default: None
        Chunk specification forwarded to ``dask.array.from_zarr``. If *None*,
        the zarr array's own chunking is used.
    t_starts : list[float] | None, default: None
        Start times (in seconds) for each epoch. Must match the number of
        zarr paths if provided.
    """

    def __init__(
        self,
        zarr_paths,
        sampling_frequency: float,
        chunks=None,
        t_starts=None,
    ):
        if isinstance(zarr_paths, (str, Path)):
            zarr_path_list = [Path(zarr_paths)]
        elif isinstance(zarr_paths, list):
            zarr_path_list = [Path(p) for p in zarr_paths]
        else:
            raise ValueError("'zarr_paths' must be a path or list of paths")

        if t_starts is not None:
            assert len(t_starts) == len(zarr_path_list), (
                "t_starts must have the same length as zarr_paths"
            )
            t_starts = [float(t) for t in t_starts]

        # Peek at the first store to determine image shape
        first_z = zarr.open(str(zarr_path_list[0]), mode="r")
        ndim = first_z.ndim
        if ndim not in (3, 4):
            raise ValueError(
                "Zarr array must be 3-D (T, H, W) or 4-D (T, H, W, P)"
            )
        if ndim == 3:
            shape = (first_z.shape[1], first_z.shape[2], 1)
        else:
            shape = (first_z.shape[1], first_z.shape[2], first_z.shape[3])

        BaseImaging.__init__(
            self, sampling_frequency=sampling_frequency, shape=shape
        )

        for i, zarr_path in enumerate(zarr_path_list):
            t_start = t_starts[i] if t_starts is not None else None
            epoch = ZarrImagingEpoch(
                zarr_path=zarr_path,
                sampling_frequency=sampling_frequency,
                t_start=t_start,
                chunks=chunks,
            )
            self.add_epoch(epoch)

        self._kwargs = {
            "zarr_paths": [str(p.absolute()) for p in zarr_path_list],
            "sampling_frequency": sampling_frequency,
            "chunks": chunks,
            "t_starts": t_starts,
        }


class ZarrImagingEpoch(BaseImagingEpoch):
    """A single epoch of imaging data backed by a zarr store.

    The data is exposed as a dask array for lazy, chunked access.
    """

    def __init__(
        self,
        zarr_path,
        sampling_frequency: float,
        t_start: float | None = None,
        chunks=None,
    ):
        BaseImagingEpoch.__init__(
            self, sampling_frequency=sampling_frequency, t_start=t_start
        )
        self.zarr_path = Path(zarr_path)

        # Open lazily via dask
        self._video = da.from_zarr(str(self.zarr_path), chunks=chunks)

        if self._video.ndim == 3:
            self._video = self._video[:, :, :, np.newaxis]

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        plane_indices: list[int] | None = None,
    ) -> da.Array:
        """Get a lazy dask series for the requested frame range and planes.

        Parameters
        ----------
        start_frame : int | None
            Start frame (inclusive). Defaults to 0.
        end_frame : int | None
            End frame (exclusive). Defaults to total frames.
        plane_indices : list[int] | None
            Plane indices to select. Defaults to all planes.

        Returns
        -------
        da.Array
            Lazy dask array of shape (frames, H, W, planes).
        """
        start = start_frame if start_frame is not None else 0
        end = end_frame if end_frame is not None else self._video.shape[0]

        if plane_indices is not None:
            return self._video[start:end, :, :, plane_indices]
        return self._video[start:end]

    def get_num_samples(self) -> int:
        """Return the number of frames in this epoch."""
        return self._video.shape[0]
