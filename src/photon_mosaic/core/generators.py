"""Module to generate synthetic imaging and ROI objects for testing and example purposes."""

from typing import Literal, NamedTuple

import numpy as np

from photon_mosaic.core import BaseRois
from photon_mosaic.core.numpyimaging import NumpyImaging, NumpyRois


class FluorescenceData(NamedTuple):
    """Return type of :func:`generate_fluorescence`.

    Attributes
    ----------
    traces : np.ndarray
        Final fluorescence traces ``(num_frames, num_rois)``, float32. Always
        ``(1 + clean_traces) * bleach(t)`` (plus additive noise if `noise_std` > 0), i.e.
        absolute fluorescence normalised so ``F0(0) == 1``; ``bleach(t) == 1`` for all
        frames when there is no photobleaching (`bleaching_time` is ``inf``).
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
    decay_time: float = 2.0,
    event_rate: float = 0.3,
    weighted_rois: bool = False,
    background: float = 3.0,
    baseline_range: tuple[float, float] = (2.0, 4.0),
    noise_std: float | Literal["poisson"] = 2.5,
    bleaching_time: float = np.inf,
    seed: int | None = None,
) -> tuple[BaseRois, NumpyImaging, FluorescenceData]:
    """Generate a random NumpyImaging object and corresponding ROIs with fluorescence activity.

    Creates synthetic imaging data with exponentially decaying fluorescence bumps
    injected at random times for each ROI, on top of a background with Gaussian or
    Poisson (shot) noise.

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
    decay_time : float, default: 2.0
        Duration of exponential decay for fluorescence events in seconds.
    event_rate : float, default: 0.3
        Mean spike-event rate in Hz, passed through to :func:`generate_fluorescence`. Each
        ROI's number of events scales with the recording's duration (`num_frames` /
        `sampling_frequency`) rather than being fixed regardless of length.
    weighted_rois : bool, default: False
        Whether to create weighted masks.
    background : float, default: 3.0
        Mean pixel intensity present everywhere in the frame, including under
        the ROIs (e.g. neuropil, out-of-focus light). 0 means a dark background
        with only noise. Subject to the same `bleaching_time` decay as the ROI
        signal, since it mostly represents genuine fluorescence rather than
        non-bleaching dark counts.
    baseline_range : tuple[float, float], default: (2.0, 4.0)
        Range from which each ROI's baseline fluorescence (F0) is drawn
        uniformly at random, modeling cell-to-cell brightness variability.
        Recovered dF/F matches `clean_traces` when `background` is 0; a
        nonzero `background` attenuates it (see `FluorescenceNode`'s
        `neuropil` argument to correct for this).
    noise_std : float or "poisson", default: 2.5
        Standard deviation of additive Gaussian noise on the video, per pixel
        and frame. 0 means no noise. Pass ``"poisson"`` instead to draw
        physically realistic shot noise (variance equals the local mean
        signal) rather than fixed-variance Gaussian noise. The default of
        2.5 gives roughly the same recovered dF/F quality (SNR) as
        ``noise_std="poisson"`` at the default `background`/`baseline_range`.
    bleaching_time : float, default: inf
        Time constant of multiplicative photobleaching in seconds, passed through to
        :func:`generate_fluorescence`. The default of ``inf`` means no photobleaching.
    seed : int | None, default: None
        Random seed for reproducibility.

    Returns
    -------
    rois : BaseRois
        The generated ROIs.
    imaging : NumpyImaging
        The imaging data with injected fluorescence activity.
    fluorescence : FluorescenceData
        The ground-truth fluorescence (traces, spikes, clean_traces) injected
        into the video, e.g. for comparison against values recovered from
        `imaging` via an :class:`~photon_mosaic.core.roianalyzer.RoiAnalyzer`.
    """
    rng = np.random.default_rng(seed)
    imaging_seed = int(rng.integers(0, 2**31))
    rois_seed = int(rng.integers(0, 2**31))
    fluorescence_seed = int(rng.integers(0, 2**31))
    noise_seed = int(rng.integers(0, 2**31))
    roi_baseline = rng.uniform(baseline_range[0], baseline_range[1], size=num_rois)

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
        decay_time=decay_time,
        event_rate=event_rate,
        bleaching_time=bleaching_time,
        seed=fluorescence_seed,
    )
    # Compute the mean signal (background everywhere, plus each ROI's fluorescence trace --
    # already F0-normalised as (1 + clean_traces) * bleach(t) -- scaled by its own F0 (baseline)):
    # (T, N) @ (N, H*W*P) -> (T, H*W*P) -> (T, H, W, P). background bleaches too (see the
    # `background` docs above), so bleaching cancels out of the dF/F ratio instead of drifting.
    video = imaging.epochs[0]._video
    masks = rois.get_roi_image_masks()  # (N, H, W) or (N, H, W, P)
    masks_flat = masks.reshape(num_rois, -1).astype(video.dtype)
    signal = roi_baseline[np.newaxis, :] * fluorescence.traces  # (T, N)
    bleach = np.exp(-np.arange(num_frames) / (bleaching_time * sampling_frequency), dtype=np.float32)

    # `video` is about to be fully overwritten, so it doubles as scratch space for the matmul
    # and background/noise below; noise is added one slab at a time to avoid a full-size array.
    flat = video.reshape(num_frames, -1)
    np.matmul(signal, masks_flat, out=flat)
    flat += (background * bleach)[:, np.newaxis]

    noise_rng = np.random.default_rng(noise_seed)
    slab_size = 256
    for t0 in range(0, num_frames, slab_size):
        sl = flat[t0 : t0 + slab_size]
        if noise_std == "poisson":
            sl[:] = noise_rng.poisson(np.clip(sl, 0, None))
        else:
            sl += noise_rng.normal(0, noise_std, sl.shape)

    rois.register_imaging(imaging)  # Link the ROIs to the imaging data

    return rois, imaging, fluorescence


def generate_fluorescence(
    num_frames: int,
    num_rois: int = 1,
    sampling_frequency: float = 30.0,
    decay_time: float = 2.0,
    event_rate: float = 0.3,
    noise_std: float = 0.0,
    bleaching_time: float = np.inf,
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
    decay_time : float, default: 2.0
        Time constant of the exponential decay kernel in seconds.
    event_rate : float, default: 0.3
        Mean spike-event rate in Hz. Each ROI's number of events is drawn as
        ``round(uniform(0.5, 1.5) * event_rate * num_frames / sampling_frequency)``, i.e.
        it scales with the recording's duration rather than being fixed regardless of length.
        Clamped to ``[0, num_frames]``.
    noise_std : float, default: 0.0
        Standard deviation of additive Gaussian noise. 0 means no noise.
    bleaching_time : float, default: inf
        Time constant of multiplicative photobleaching in seconds. The default of ``inf``
        means no photobleaching.
    seed : int | None, default: None
        Random seed for reproducibility.

    Returns
    -------
    FluorescenceData
        Named tuple with fields ``traces``, ``spikes``, and ``clean_traces``,
        each of shape ``(num_frames, num_rois)`` as float32. ``traces`` is
        ``(1 + clean_traces) * bleach(t)``, i.e. absolute fluorescence
        normalised so ``F0(0) == 1``; ``bleach(t) == 1`` for all frames when
        `bleaching_time` is ``inf`` (no photobleaching decay, but the baseline
        offset of 1 still applies).
    """
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)

    # Generate binary spike trains. Each event has unit amplitude, i.e. a single, isolated
    # spike produces a dF/F peak of 1 (100%); this is close to single-AP responses reported
    # for newer, high-sensitivity indicators like jGCaMP8m/8s (Zhang et al. 2023, Nature).
    spikes = np.zeros((num_frames, num_rois), dtype=np.float32)
    for roi_idx in range(num_rois):
        num_events = round(rng.uniform(0.5, 1.5) * event_rate * num_frames / sampling_frequency)
        num_events = min(max(num_events, 0), num_frames)
        event_times = rng.choice(num_frames, size=num_events, replace=False)
        spikes[event_times, roi_idx] = 1.0

    # Convolve spikes with exponential kernel via exact IIR recurrence: traces[t] = spikes[t] + g * traces[t-1]
    g = np.exp(-1.0 / (decay_time * sampling_frequency))
    clean_traces = lfilter([1.0], [1.0, -g], spikes, axis=0).astype(np.float32)

    # traces acts as F = (1 + dF/F) * F0(t), with F0 normalised to 1 at t=0; bleaching_time=inf
    # is exactly the no-bleaching limit, since bleach(t) = exp(-t / (inf * sf)) == 1 for all t.
    bleach = np.exp(-np.arange(num_frames) / (bleaching_time * sampling_frequency), dtype=np.float32)
    traces = (1.0 + clean_traces) * bleach[:, np.newaxis]
    if noise_std > 0:
        traces = traces + rng.normal(0, noise_std, (num_frames, num_rois)).astype(np.float32)

    return FluorescenceData(traces=traces, spikes=spikes, clean_traces=clean_traces)
