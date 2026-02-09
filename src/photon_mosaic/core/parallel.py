import shutil
from pathlib import Path

import numpy as np  # Add this import
from numpy.typing import DTypeLike

from photon_mosaic.core.baseimaging import BaseImaging


class DaskImaging(BaseImaging):
    """DaskImaging is a BaseImaging that can be instantiated as a Dask array.
    It needs to point to a single epoch and implements the array API.

    Parameters
    ----------
    segments : list of BaseImagingEpoch
        A list of imaging epochs (segments) that make up the complete imaging dataset.

    Returns
    -------
    imaging : DaskImaging
        The DaskImaging object representing the complete imaging dataset.
    """

    def __init__(self, imaging: BaseImaging, epoch_index: int = 0):
        BaseImaging.__init__(self, shape=imaging.shape, sampling_frequency=imaging.sampling_frequency)
        self._segment = imaging.epochs[epoch_index]
        self.add_epoch(imaging.epochs[epoch_index])
        self._dtype = imaging.get_dtype()
        self._kwargs = dict(imaging=imaging, epoch_index=epoch_index)

    # DASK ARRAY API METHODS
    @property
    def dtype(self) -> DTypeLike:
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
        return (self.get_num_samples(segment_index=0),) + self._shape

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

    # def __getitem__(self, idx):
    #     if not isinstance(idx, tuple):
    #         idx = (idx,)

    #     # Pad idx to ndim
    #     idx = idx + (slice(None),) * (self.ndim - len(idx))

    #     # Only handle first dimension for slicing into samples
    #     frame_idx = idx[0]
    #     num_samples = self.get_num_samples()

    #     # Convert to valid start/end
    #     if isinstance(frame_idx, int):
    #         start, stop = frame_idx, frame_idx + 1
    #     elif isinstance(frame_idx, slice):
    #         start, stop, step = frame_idx.indices(num_samples)
    #     else:
    #         # list/array indexing
    #         start, stop = 0, num_samples

    #     data = self.get_series(start_frame=start, end_frame=stop)
    #     if data is None:
    #         raise ValueError(f"get_series returned None for slice {idx}")

    #     print("GETITEM:", idx, "returning shape", None if data is None else getattr(data, "shape", "None"))

    #     # Apply full slice to remaining dimensions
    #     try:
    #         return data[idx if isinstance(frame_idx, slice) else (slice(None),) + idx[1:]]
    #     except Exception as e:
    #         print(f"Error applying slice {idx} to data shape {data.shape}")
    #         raise e

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
        data = self._segment.get_series(start_frame=start_frame, end_frame=end_frame)

        # Add this check immediately after get_series
        if data is None:
            print(
                f"get_series returned None for frames [{start_frame}:{end_frame}]. Index: {idx}, segment has {num_samples} samples.",
                flush=True,
            )
            raise ValueError(
                f"get_series returned None for frames [{start_frame}:{end_frame}]. "
                f"Index: {idx}, segment has {num_samples} samples."
            )

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

        if result is None:
            print(f"{idx} resulted in None data", flush=True)
            raise IndexError(f"Index {idx} is out of bounds for imaging segment with shape {self.shape}")

        return result


def save_to_zarr(
    imaging: BaseImaging,
    file_path: str | Path,
    client=None,
    overwrite: bool = True,
    chunks=None,
    **client_kwargs,
) -> None:
    """Save the imaging data to a Zarr file for efficient storage and access.

    Parameters
    ----------
    imaging : BaseImaging
        The imaging data to save.
    file_path : str or Path
        The path to the Zarr file where the data will be saved.
    client : dask.distributed.Client, optional
        A Dask distributed client for parallel processing. If None, the default Dask scheduler will be used.
    client_kwargs : dict, optional
        Additional keyword arguments to pass to the Dask client when saving the data.
    """
    import dask.array as da
    from dask import compute
    from dask.distributed import Client

    if client is None:
        client = Client(**client_kwargs)
    elif not isinstance(client, Client):
        raise ValueError("client must be a dask.distributed.Client instance or None")

    file_path = Path(file_path)
    if file_path.suffix != ".zarr":
        raise ValueError("file_path must have a .zarr extension")

    if file_path.is_dir():
        if overwrite:
            print(f"Directory {file_path} already exists. Overwriting it.")
            shutil.rmtree(file_path)
        else:
            raise FileExistsError(f"File {file_path} already exists. Set overwrite=True to overwrite it.")

    if chunks is None:
        # Default chunk size: 100 frames per chunk, full spatial dimensions
        chunks = (100,) + imaging.shape

    delayed = []
    for epoch_index in range(imaging.get_num_epochs()):
        segment = DaskImaging(imaging, epoch_index=epoch_index)
        dask_array = da.from_array(segment, chunks=chunks)
        # Save the segment to Zarr using Dask's to_zarr method
        writes = dask_array.to_zarr(
            file_path / f"video_epoch{epoch_index}",
            compute=False,
        )
        delayed.extend(writes)

    # Compute all the delayed writes to save the data to Zarr
    compute(*delayed)
