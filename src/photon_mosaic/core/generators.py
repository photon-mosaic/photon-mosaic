"""Module to generate synthetic imaging and ROI objects for testing and example purposes."""

from pathlib import Path

import numpy as np

from photon_mosaic.core import BaseRois
from photon_mosaic.core.arrayimaging import ArrayImaging, NumpyRois


# ---------------------------------------------------------------------------
# In-memory generators
# ---------------------------------------------------------------------------


def generate_random_imaging(
    num_frames: int | tuple[int, ...] = 1000,
    height: int = 256,
    width: int = 256,
    num_planes: int = 1,
    sampling_frequency: float = 30.0,
    seed: int | None = None,
) -> ArrayImaging:
    """Generate a random ArrayImaging object for testing.

    Parameters
    ----------
    num_frames : int | tuple[int, ...], default: 1000
        Number of frames for each epoch in the imaging data.
    height : int, default: 256
        Height of each frame in pixels.
    width : int, default: 256
        Width of each frame in pixels.
    sampling_frequency : float, default: 30.0
        Sampling frequency in Hz.

    Returns
    -------
    ArrayImaging
        An ArrayImaging object containing the generated random imaging data.
    """
    if isinstance(num_frames, int):
        num_frames = (num_frames,)
    rng = np.random.default_rng(seed)
    videos = []
    for n_frames in num_frames:
        video = rng.random((n_frames, height, width, num_planes))
        videos.append(video)
    return ArrayImaging(imaging_series=videos, sampling_frequency=sampling_frequency)



def write_random_npy(
    path: str | Path,
    shape: tuple,
    dtype=np.uint16,
    seed: int = 0,
) -> Path:
    """Write a ``.npy`` file filled with random data.

    Data is written one frame at a time via an on-disk memory-mapped
    ``.npy`` file so that peak RAM usage stays bounded regardless of the
    total dataset size.

    Parameters
    ----------
    path : str or Path
        Destination ``.npy`` file path (will be created / overwritten).
    shape : tuple
        ``(num_frames, height, width, num_planes)``
    dtype : numpy dtype
        Element data type.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Path
        The path that was written to.
    """
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create a memory-mapped .npy file on disk
    mmap = np.lib.format.open_memmap(
        str(path), mode="w+", dtype=dtype, shape=shape
    )

    num_frames = shape[0]
    frame_shape = shape[1:]  # (H, W, P)
    for i in range(num_frames):
        mmap[i] = rng.integers(0, 4000, size=frame_shape, dtype=dtype)

    # Flush to disk and release the memmap
    del mmap
    return path


def write_random_binary(
    path: str | Path,
    shape: tuple,
    dtype=np.uint16,
    seed: int = 0,
) -> Path:
    """Write a contiguous raw binary file filled with random data.

    The file is written one frame at a time so that peak memory usage stays
    bounded regardless of the total dataset size.

    Parameters
    ----------
    path : str or Path
        Destination file path (will be created / overwritten).
    shape : tuple
        ``(num_frames, height, width, num_planes)``
    dtype : numpy dtype
        Element data type.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Path
        The path that was written to.
    """
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_frames = shape[0]
    frame_shape = shape[1:]  # (H, W, P)

    with open(path, "wb") as f:
        for _ in range(num_frames):
            frame = rng.integers(0, 4000, size=frame_shape, dtype=dtype)
            f.write(frame.tobytes())

    return path


def write_random_zarr(
    path: str | Path,
    shape: tuple,
    dtype=np.uint16,
    chunks: tuple | None = None,
    seed: int = 0,
) -> Path:
    """Write a zarr store filled with random data.

    Data is written one temporal chunk at a time to keep memory bounded.

    Parameters
    ----------
    path : str or Path
        Destination ``.zarr`` directory.
    shape : tuple
        ``(num_frames, height, width, num_planes)``
    dtype : numpy dtype
        Element data type.
    chunks : tuple | None
        Zarr chunk layout.  Defaults to ``(100, H, W, 1)`` — chunked
        along *both* time and planes, which is the layout that lets dask
        load a single plane without touching the others.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    Path
        The path that was written to.
    """
    import zarr

    rng = np.random.default_rng(seed)
    path = Path(path)

    num_frames, height, width, num_planes = shape
    if chunks is None:
        chunks = (100, height, width, 1)

    z = zarr.open(
        str(path),
        mode="w",
        shape=shape,
        dtype=dtype,
        chunks=chunks,
    )

    # Write in temporal slabs matching the temporal chunk size
    t_chunk = chunks[0]
    for t0 in range(0, num_frames, t_chunk):
        t1 = min(t0 + t_chunk, num_frames)
        slab = rng.integers(0, 4000, size=(t1 - t0, height, width, num_planes), dtype=dtype)
        z[t0:t1] = slab

    return path



def generate_rois(
    num_rois: int = 20,
    height: int = 256,
    width: int = 256,
    radius_range: tuple[int, int] | tuple[int, int, int] = (5, 15),
    sampling_frequency: float = 30.0,
    roi_ids: np.ndarray | None = None,
    weighted: bool = False,
    num_planes: int = 1,
    seed: int | None = None,
) -> BaseRois:
    """Generate circular ROIs for testing.

    Parameters
    ----------
    num_rois : int, default: 20
        Number of ROIs to generate, by default 20
    height : int, default: 256
        Height of the imaging field, by default 256
    width : int, default: 256
        Width of the imaging field, by default 256
    radius_range : tuple[int, int] | tuple[int, int, int], default: (5, 15)
        Range of radii for the circular ROIs, by default (5, 15) (for 2D).
        If num_planes > 1 and a tuple of three ints is provided, the third int is used as the radius for the
        z-dimension. If a tuple of two ints is provided, the depth radius will be half of the num_planes.
    sampling_frequency : float, default: 30.
        Sampling frequency, by default 30.0
    roi_ids : np.ndarray | None, default: None
        Array of ROI IDs. If None, defaults to np.arange(num_rois)
    weighted : bool, default: False
        Whether to create weighted masks (values between 0 and 1) or binary masks (0 or 1), by default False
    num_planes : int, default: 1
        Number of planes for the ROIs, by default 1 (2D masks). If >1, creates 3D masks.
    """
    if num_planes == 1:
        roi_masks = np.zeros((num_rois, height, width))
        rng = np.random.default_rng(seed)
    else:
        roi_masks = np.zeros((num_rois, height, width, num_planes))
        rng = np.random.default_rng(seed)
        if len(radius_range) == 2:
            depth_radius = max(1, num_planes // 2)
        else:
            depth_radius = radius_range[2]

    assert radius_range[0] < radius_range[1], "Invalid radius range"
    assert radius_range[1] < width - radius_range[1], "ROIs may not fit in the image with the given radius range"
    assert radius_range[1] < height - radius_range[1], "ROIs may not fit in the image with the given radius range"

    for roi_idx in range(num_rois):
        center_x = rng.integers(radius_range[1], width - radius_range[1])
        center_y = rng.integers(radius_range[1], height - radius_range[1])
        radius = rng.integers(radius_range[0], radius_range[1])

        if num_planes == 1:
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2

            if not weighted:
                roi_masks[roi_idx] = mask
            else:
                # Create a weighted mask with values decreasing from center to edge
                distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                weighted_mask = np.clip(1 - (distance_from_center / radius), 0, 1) * mask
                roi_masks[roi_idx] = weighted_mask
        else:
            # Choose z-center so the ROI fits in depth; fall back to middle plane if too shallow
            if num_planes > 2 * depth_radius:
                center_z = rng.integers(depth_radius, num_planes - depth_radius)
            else:
                center_z = num_planes // 2

            y, x, z = np.ogrid[:height, :width, :num_planes]

            # Ellipsoidal ROI: (dx^2+dy^2)/r^2 + dz^2/drz^2 <= 1
            dx2_dy2 = (x - center_x) ** 2 + (y - center_y) ** 2
            dz2 = (z - center_z) ** 2
            ellipsoid_distance = np.sqrt((dx2_dy2 / (radius**2)) + (dz2 / (depth_radius**2)))
            mask = ellipsoid_distance <= 1

            if not weighted:
                roi_masks[roi_idx] = mask
            else:
                weighted_mask = np.clip(1 - ellipsoid_distance, 0, 1) * mask
                roi_masks[roi_idx] = weighted_mask

    roi_ids = np.arange(num_rois) if roi_ids is None else roi_ids
    return NumpyRois(roi_image_masks=roi_masks, roi_ids=roi_ids, sampling_frequency=sampling_frequency)

