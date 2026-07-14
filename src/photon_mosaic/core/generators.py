"""Module to generate synthetic imaging and ROI objects for testing and example purposes."""

from typing import NamedTuple

import numpy as np

from photon_mosaic.core import BaseRois
from photon_mosaic.core.numpyimaging import NumpyImaging, NumpyRois


class FluorescenceData(NamedTuple):
    """Return type of :func:`generate_fluorescence`.

    Attributes
    ----------
    traces : np.ndarray
        Final fluorescence traces ``(num_frames, num_rois)``, float32.
        Without bleaching equals ``clean_traces`` (dF/F). With bleaching equals
        ``(1 + clean_traces) * bleach``, i.e. absolute fluorescence normalised by F0.
    spikes : np.ndarray
        Binary spike trains ``(num_frames, num_rois)``, float32.
    clean_traces : np.ndarray
        Convolved traces before bleaching and noise ``(num_frames, num_rois)``, float32.
    """

    traces: np.ndarray
    spikes: np.ndarray
    clean_traces: np.ndarray


def generate_random_imaging(
    num_frames: int | tuple[int, ...] = 1000,
    height: int = 256,
    width: int = 256,
    num_planes: int = 1,
    sampling_frequency: float = 30.0,
    seed: int | None = None,
) -> NumpyImaging:
    """Generate a random NumpyImaging object for testing.

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
    NumpyImaging
        A NumpyImaging object containing the generated random imaging data.
    """
    if isinstance(num_frames, int):
        num_frames = (num_frames,)
    rng = np.random.default_rng(seed)
    videos = []
    for n_frames in num_frames:
        video = rng.random((n_frames, height, width, num_planes))
        videos.append(video)
    return NumpyImaging(imaging_series=videos, sampling_frequency=sampling_frequency)


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


def generate_imaging_with_rois(
    num_frames: int = 1000,
    height: int = 256,
    width: int = 256,
    num_planes: int = 1,
    num_rois: int = 20,
    radius_range: tuple[int, int] | tuple[int, int, int] = (5, 15),
    sampling_frequency: float = 30.0,
    decay_seconds: float = 2.0,
    weighted_rois: bool = False,
    noise_std: float = 1.0,
    seed: int | None = None,
) -> tuple[BaseRois, NumpyImaging]:
    """Generate a random NumpyImaging object and corresponding ROIs with fluorescence activity.

    Creates synthetic imaging data with exponentially decaying fluorescence bumps
    injected at random times for each ROI.

    Parameters
    ----------
    num_frames : int, default: 1000
        Number of frames in the imaging data.
    height : int, default: 256
        Height of each frame in pixels.
    width : int, default: 256
        Width of each frame in pixels.
    num_planes : int, default: 1
        Number of imaging planes.
    num_rois : int, default: 20
        Number of ROIs to generate.
    radius_range : tuple[int, int] | tuple[int, int, int], default: (5, 15)
        Range of radii for circular ROIs.
    sampling_frequency : float, default: 30.0
        Sampling frequency in Hz.
    decay_seconds : float, default: 2.0
        Duration of exponential decay for fluorescence events in seconds.
    weighted_rois : bool, default: False
        Whether to create weighted masks.
    noise_std : float, default: 1.0
        Scale factor applied to the random background video from
        `generate_random_imaging` before injecting the clean ROI masks x
        clean_traces tensor product. 1.0 preserves the default background
        noise amplitude; use values >1 to increase noise, <1 to decrease it.
    seed : int | None, default: None
        Random seed for reproducibility.

    Returns
    -------
    rois : BaseRois
        The generated ROIs.
    imaging : NumpyImaging
        The imaging data with injected fluorescence activity.
    """
    rng = np.random.default_rng(seed)
    imaging_seed = int(rng.integers(0, 2**31))
    rois_seed = int(rng.integers(0, 2**31))
    fluorescence_seed = int(rng.integers(0, 2**31))

    imaging = generate_random_imaging(
        num_frames=num_frames,
        height=height,
        width=width,
        num_planes=num_planes,
        sampling_frequency=sampling_frequency,
        seed=imaging_seed,
    )
    rois = generate_rois(
        num_rois=num_rois,
        height=height,
        width=width,
        radius_range=radius_range,
        sampling_frequency=sampling_frequency,
        weighted=weighted_rois,
        num_planes=num_planes,
        seed=rois_seed,
    )
    fluorescence = generate_fluorescence(
        num_frames=num_frames,
        num_rois=num_rois,
        sampling_frequency=sampling_frequency,
        decay_seconds=decay_seconds,
        seed=fluorescence_seed,
    )
    # Inject clean traces into video: (T, N) @ (N, H*W*P) -> (T, H*W*P) -> (T, H, W, P)
    video = imaging.epochs[0]._video
    video *= noise_std
    masks = rois.get_roi_image_masks()  # (N, H, W) or (N, H, W, P)
    masks_flat = masks.reshape(num_rois, -1).astype(video.dtype)
    video += (fluorescence.clean_traces @ masks_flat).reshape(video.shape)

    rois.register_imaging(imaging)  # Link the ROIs to the imaging data

    return rois, imaging


def generate_fluorescence(
    num_frames: int,
    num_rois: int = 1,
    sampling_frequency: float = 30.0,
    decay_seconds: float = 2.0,
    noise_std: float = 0.0,
    bleaching_tau: float | None = None,
    seed: int | None = None,
) -> FluorescenceData:
    """Generate synthetic fluorescence traces by convolving random spike events with an exponential kernel.

    Parameters
    ----------
    num_frames : int
        Number of frames (time points) to generate.
    num_rois : int, default: 1
        Number of independent fluorescence traces to generate.
    sampling_frequency : float, default: 30.0
        Sampling frequency in Hz.
    decay_seconds : float, default: 2.0
        Time constant of the exponential decay kernel in seconds.
    noise_std : float, default: 0.0
        Standard deviation of additive Gaussian noise. 0 means no noise.
    bleaching_tau : float | None, default: None
        Time constant of multiplicative photobleaching in seconds.
        If None, no bleaching is applied.
    seed : int | None, default: None
        Random seed for reproducibility.

    Returns
    -------
    FluorescenceData
        Named tuple with fields ``traces``, ``spikes``, and ``clean_traces``,
        each of shape ``(num_frames, num_rois)`` as float32.
    """
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)

    # Generate binary spike trains
    spikes = np.zeros((num_frames, num_rois), dtype=np.float32)
    for roi_idx in range(num_rois):
        num_events = rng.integers(5, 15)
        event_times = rng.choice(num_frames, size=num_events, replace=False)
        spikes[event_times, roi_idx] = 1.0

    # Convolve spikes with exponential kernel via exact IIR recurrence: traces[t] = spikes[t] + g * traces[t-1]
    g = np.exp(-1.0 / (decay_seconds * sampling_frequency))
    clean_traces = lfilter([1.0], [1.0, -g], spikes, axis=0).astype(np.float32)

    # Apply bleaching and noise
    traces = clean_traces.copy()
    if bleaching_tau is not None:
        # bleach acts as F0(t): F = (1 + dF/F) * F0(t), with F0 normalised to 1 at t=0
        bleach = np.exp(-np.arange(num_frames) / (bleaching_tau * sampling_frequency), dtype=np.float32)
        traces = (1.0 + clean_traces) * bleach[:, np.newaxis]
    if noise_std > 0:
        traces = traces + rng.normal(0, noise_std, (num_frames, num_rois)).astype(np.float32)

    return FluorescenceData(traces=traces, spikes=spikes, clean_traces=clean_traces)
