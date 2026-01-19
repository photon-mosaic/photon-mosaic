"""Module to generate synthetic imaging and ROI objects for testing purposes."""

import numpy as np

from photon_mosaic.core import BaseRois
from photon_mosaic.core.numpyimaging import NumpyImaging, NumpyRois


def generate_random_imaging(
    num_frames: int | tuple[int, ...] = 1000,
    height: int = 256,
    width: int = 256,
    sampling_frequency: float = 30.0,
    num_planes: int = 1,
    plane_ids: list[int] | None = None,
    seed: int | None = None,
) -> NumpyImaging:
    """Generate a random NumpyImaging object for testing.

    Parameters
    ----------
    num_frames : int | tuple[int, ...], default: 1000
        Number of frames for each segment in the imaging data.
    height : int, default: 256
        Height of each frame in pixels.
    width : int, default: 256
        Width of each frame in pixels.
    sampling_frequency : float, default: 30.0
        Sampling frequency in Hz.

    Returns
    -------
    NumpyImaging
        A NumpyImaging object containing the generated random imaging data.
    """
    if isinstance(num_frames, int):
        num_frames = (num_frames,)
    rng = np.random.default_rng(seed)
    videos = []
    for n_frames in num_frames:
        video = rng.random((n_frames, height, width, num_planes))
        video = np.squeeze(video)
        videos.append(video)
    return NumpyImaging(imaging_series=videos, sampling_frequency=sampling_frequency, plane_ids=plane_ids)


def generate_rois(
    num_rois: int = 20,
    height: int = 256,
    width: int = 256,
    radius_range: tuple[int, int] = (5, 15),
    sampling_frequency: float = 30.0,
    roi_ids: np.ndarray | None = None,
    weighted: bool = False,
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
    radius_range : tuple[int, int], default: (5, 15)
        Range of radii for the circular ROIs, by default (5, 15)
    """
    roi_masks = np.zeros((num_rois, height, width))
    rng = np.random.default_rng(0)

    assert radius_range[0] < radius_range[1], "Invalid radius range"
    assert radius_range[1] < width - radius_range[1], "ROIs may not fit in the image with the given radius range"
    assert radius_range[1] < height - radius_range[1], "ROIs may not fit in the image with the given radius range"

    for roi_idx in range(num_rois):
        center_x = rng.integers(radius_range[1], width - radius_range[1])
        center_y = rng.integers(radius_range[1], height - radius_range[1])
        radius = rng.integers(radius_range[0], radius_range[1])

        y, x = np.ogrid[:height, :width]
        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        if not weighted:
            roi_masks[roi_idx] = mask
        else:
            # Create a weighted mask with values decreasing from center to edge
            distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            weighted_mask = np.clip(1 - (distance_from_center / radius), 0, 1) * mask
            roi_masks[roi_idx] = weighted_mask

    roi_ids = np.arange(num_rois) if roi_ids is None else roi_ids
    return NumpyRois(roi_image_masks=roi_masks, roi_ids=roi_ids, sampling_frequency=sampling_frequency)
