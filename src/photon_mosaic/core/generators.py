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
    dtype: type = np.float32,
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
    dtype : type, default: np.float32
        Dtype of the generated video (``np.float32``/``np.float64``). Float32 halves memory
        with no precision cost -- nothing downstream reads above float32 precision anyway.

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
        video = rng.random((n_frames, height, width, num_planes), dtype=dtype)
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
    sparse: bool = False,
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
    sparse : bool, default: False
        If True, return the masks as a sparse.GCXS array instead of a dense np.ndarray.
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
    if sparse:
        import sparse as sparse_module

        # Compress along the ROI axis so per-ROI indexing (e.g. select_rois) stays fast --
        # the default heuristic often picks a different axis, making it ~40x slower.
        roi_masks = sparse_module.GCXS.from_numpy(roi_masks, compressed_axes=(0,))
    return NumpyRois(roi_image_masks=roi_masks, roi_ids=roi_ids, sampling_frequency=sampling_frequency)


# Internal constants for the "vignette"/"diffuse" neuropil models (see `generate_imaging_with_rois`).
_VIGNETTE_FALLOFF = 0.7  # fraction of center brightness lost at the frame corners
_NEUROPIL_TAU_SECONDS = 5.0  # OU mean-reversion timescale -- slow drift, not frame-to-frame noise
_DIFFUSE_DENSITY = 8  # diffuse sources per ROI
# Relative to ROI radius -- Zhou et al. 2018 (CNMF-E) simulate background footprints ~5x neuron width.
_DIFFUSE_RADIUS_MULTIPLIER = 5.0


def _generate_vignette_profile(height: int, width: int, num_planes: int = 1) -> np.ndarray:
    """Static radial illumination falloff, brighter at the frame center than the edges.

    Simple linear falloff from the frame center -- real 2P vignetting (Gaussian beam, finite-NA
    lens falloff) isn't modeled exactly. Renormalized to spatial mean 1, so multiplying it into
    `background` preserves that parameter's "mean photon count per pixel per frame" semantics
    regardless of `neuropil_model`.

    Parameters
    ----------
    height, width : int
        Frame dimensions in pixels.
    num_planes : int, default: 1
        Number of imaging planes; the same 2D profile is broadcast across all planes.

    Returns
    -------
    np.ndarray
        ``(height, width)`` or ``(height, width, num_planes)``, float32, spatial mean 1.
    """
    y, x = np.ogrid[:height, :width]
    center_y, center_x = (height - 1) / 2.0, (width - 1) / 2.0
    distance = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    max_distance = np.sqrt(center_y**2 + center_x**2)  # center-to-corner distance
    profile = 1.0 - _VIGNETTE_FALLOFF * (distance / max_distance)
    profile = profile / profile.mean()  # renormalize to spatial mean 1
    if num_planes > 1:
        profile = np.broadcast_to(profile[:, :, np.newaxis], (height, width, num_planes)).copy()
    return profile.astype(np.float32)


def _generate_ou_process(
    num_frames: int,
    sampling_frequency: float,
    rng: np.random.Generator,
    num_traces: int = 1,
) -> np.ndarray:
    """Ornstein-Uhlenbeck-like fluctuation(s): mean 0, asymptotically unit variance.

    Discrete-time AR(1) via exact IIR recurrence (same `lfilter` style as
    `generate_fluorescence`'s exponential-kernel convolution): ``x[t] = phi * x[t-1] +
    sqrt(1 - phi**2) * white[t]``, with ``phi = exp(-1 / (_NEUROPIL_TAU_SECONDS *
    sampling_frequency))``. Starts at 0 and reaches unit variance within a few
    `_NEUROPIL_TAU_SECONDS` -- an accepted simplification (the brief initial transient is
    negligible for the frame counts this is meant for) rather than seeding the recurrence from
    its stationary distribution.

    Parameters
    ----------
    num_frames : int
        Number of time points to generate.
    sampling_frequency : float
        Sampling frequency in Hz.
    rng : np.random.Generator
        Source of randomness.
    num_traces : int, default: 1
        Number of independent traces to generate at once.

    Returns
    -------
    np.ndarray
        ``(num_frames, num_traces)``, float32.
    """
    from scipy.signal import lfilter

    phi = np.exp(-1.0 / (_NEUROPIL_TAU_SECONDS * sampling_frequency))
    white = rng.normal(0, 1, size=(num_frames, num_traces)).astype(np.float32)
    ou = lfilter([np.sqrt(1 - phi**2)], [1.0, -phi], white, axis=0)
    return ou.astype(np.float32)


def _generate_diffuse_footprints(
    n_sources: int,
    height: int,
    width: int,
    num_planes: int,
    radius_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Broad, overlapping, low-amplitude spatial footprints for the "diffuse" neuropil model.

    Same linear radial-falloff shape as `generate_rois`'s weighted masks (``1 -
    distance/radius``, clipped to ``[0, 1]``), but centers are unconstrained by frame edges
    (these blobs are meant to be broad relative to the frame, unlike per-cell ROIs) and returned
    as a plain array rather than a `BaseRois` -- these aren't ROIs, and `generate_rois`'s
    edge-margin ``assert`` would otherwise reject radii this large relative to typical frame
    sizes. Each source spans a single plane (drawn uniformly) rather than `generate_rois`'
    ellipsoidal 3D masks.

    Parameters
    ----------
    n_sources : int
        Number of diffuse sources to place.
    height, width : int
        Frame dimensions in pixels.
    num_planes : int
        Number of imaging planes.
    radius_range : tuple[float, float]
        Range of radii (pixels) from which each source's radius is drawn uniformly.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        ``(n_sources, height, width)`` or ``(n_sources, height, width, num_planes)``, float32,
        each source's own footprint in ``[0, 1]``.
    """
    y, x = np.ogrid[:height, :width]
    shape = (n_sources, height, width) if num_planes == 1 else (n_sources, height, width, num_planes)
    footprints = np.zeros(shape, dtype=np.float32)
    centers_x = rng.uniform(0, width, size=n_sources)
    centers_y = rng.uniform(0, height, size=n_sources)
    radii = rng.uniform(radius_range[0], radius_range[1], size=n_sources)
    planes = rng.integers(0, num_planes, size=n_sources) if num_planes > 1 else None
    for k in range(n_sources):
        distance = np.sqrt((x - centers_x[k]) ** 2 + (y - centers_y[k]) ** 2)
        footprint = np.clip(1.0 - distance / radii[k], 0.0, 1.0)
        if num_planes == 1:
            footprints[k] = footprint
        else:
            footprints[k, :, :, planes[k]] = footprint
    return footprints


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
    background: float = 0.2,
    baseline_range: tuple[float, float] = (0.5, 1.0),
    noise_std: float | Literal["poisson"] = 1.3,
    bleaching_time: float = np.inf,
    neuropil_model: Literal["constant", "vignette", "diffuse"] = "constant",
    neuropil_fluctuation_std: float = 0.3,
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
        `sampling_frequency`).
    weighted_rois : bool, default: False
        Whether to create weighted masks.
    background : float, default: 0.2
        Mean photon count per pixel per frame, exact only under
        ``noise_std="poisson"`` (otherwise just an intensity scale) -- present
        everywhere in the frame, including under the ROIs (e.g. neuropil, out-of-
        focus light). 0 means a dark background with only noise. Subject to the same
        `bleaching_time` decay as the ROI signal, since it mostly represents genuine
        fluorescence rather than non-bleaching dark counts. This is the true
        frame-and-time-averaged mean regardless of `neuropil_model`: each model's spatial
        profile and/or temporal fluctuation is normalized to mean 1, so `background` always
        means the same thing.
    baseline_range : tuple[float, float], default: (0.5, 1.0)
        Range from which each ROI's baseline fluorescence (F0) is drawn uniformly at
        random, modeling cell-to-cell brightness variability -- same photon-count
        semantics as `background` (exact only under ``noise_std="poisson"``).
        Recovered dF/F matches `clean_traces` when `background` is 0; a nonzero
        `background` attenuates it (see `FluorescenceNode`'s `neuropil` argument to
        correct for this).
    noise_std : float or "poisson", default: 1.3
        Standard deviation of additive Gaussian noise on the video, per pixel
        and frame. 0 means no noise. Pass ``"poisson"`` instead to draw
        physically realistic shot noise (variance equals the local mean
        signal) rather than fixed-variance Gaussian noise. The default of
        1.3 gives roughly the same recovered dF/F noise level (measured via
        ``aind_ophys_utils.signal_utils.noise_std(method="welch")``) as
        ``noise_std="poisson"`` at the default `background`/`baseline_range`.
    bleaching_time : float, default: inf
        Time constant of multiplicative photobleaching in seconds, passed through to
        :func:`generate_fluorescence`. The default of ``inf`` means no photobleaching.
    neuropil_model : {"constant", "vignette", "diffuse"}, default: "constant"
        How `background` varies over space and time:

        - ``"constant"``: spatially uniform, no fluctuation beyond `bleaching_time` decay.
          Neuropil subtraction only ever corrects this constant attenuation, never a genuine
          fluctuation.
        - ``"vignette"``: one shared Ornstein-Uhlenbeck-like fluctuation (slow, ~5s timescale)
          modulated by a static radial illumination falloff (brighter center, dimmer edges --
          real 2P vignetting). Every background pixel shares the same fluctuation, just scaled
          by its own position.
        - ``"diffuse"``: many (``8 * num_rois``) broad, overlapping, independently-fluctuating
          sources mixed together, so nearby background pixels are correlated but distant ones
          aren't -- a spatially-varying mixture rather than one shared signal. Each source's
          radius is drawn from `radius_range` scaled by 5 (Zhou et al. 2018's CNMF-E
          simulations use the same ratio between background and neuron footprint size), so it
          scales with ROI size rather than frame size. Also carries the same vignette falloff
          as ``"vignette"``.

        Both new models are normalized to preserve `background`'s mean-photon-count semantics
        (see `background` above); only their *spatial and temporal structure* differs from
        ``"constant"``. Kept intentionally simple -- just enough realism to make neuropil
        subtraction demonstrably matter, not an optically/biologically precise model.
    neuropil_fluctuation_std : float, default: 0.3
        How strongly the background fluctuates relative to `background` itself, for
        ``neuropil_model="vignette"`` or ``"diffuse"``. Ignored for ``"constant"``.
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
    # Drawn unconditionally (even for "constant", which doesn't use it) so switching
    # `neuropil_model` never perturbs the other seeds' consumption/downstream RNG streams.
    neuropil_seed = int(rng.integers(0, 2**31))
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
    # background bleaches too, so bleaching cancels out of the dF/F ratio instead of drifting.
    # (T, N) @ (N, H*W*P) -> (T, H*W*P) -> (T, H, W, P).
    video = imaging.epochs[0]._video
    masks = rois.get_roi_image_masks()  # (N, H, W) or (N, H, W, P)
    masks_flat = masks.reshape((num_rois, -1)).astype(video.dtype)
    # roi_baseline is float64 by default; cast to video.dtype, or the matmul below (out=flat)
    # silently allocates a full movie-sized float64 temporary for the mixed-precision multiply.
    signal = (roi_baseline[np.newaxis, :] * fluorescence.traces).astype(video.dtype)  # (T, N)
    bleach = np.exp(-np.arange(num_frames) / (bleaching_time * sampling_frequency), dtype=np.float32)

    # `video` is reused as scratch (about to be overwritten); background+noise are added slab by
    # slab to avoid a full (num_frames, height*width*num_planes) temporary for either.
    flat = video.reshape(num_frames, -1)
    np.matmul(signal, masks_flat, out=flat)

    neuropil_rng = np.random.default_rng(neuropil_seed)
    if neuropil_model != "constant":
        profile_flat = _generate_vignette_profile(height, width, num_planes).reshape(-1).astype(video.dtype)
    if neuropil_model == "vignette":
        fluctuation = 1.0 + neuropil_fluctuation_std * _generate_ou_process(
            num_frames, sampling_frequency, neuropil_rng
        ).reshape(-1)
    elif neuropil_model == "diffuse":
        n_sources = max(1, round(_DIFFUSE_DENSITY * num_rois))
        diffuse_radius_range = (
            radius_range[0] * _DIFFUSE_RADIUS_MULTIPLIER,
            radius_range[1] * _DIFFUSE_RADIUS_MULTIPLIER,
        )
        footprints = _generate_diffuse_footprints(
            n_sources, height, width, num_planes, diffuse_radius_range, neuropil_rng
        )
        footprints_flat = footprints.reshape(n_sources, -1).astype(video.dtype)
        # Typical per-pixel variance if every source's own trace had unit variance -- each
        # source's trace *is* unit variance (`_generate_ou_process`), so this scales
        # `neuropil_fluctuation_std` to a comparable magnitude to "vignette"'s single shared
        # trace, exactly rather than approximately.
        typical_variance = (footprints_flat**2).sum(axis=0).mean()
        modulation_scale = 1.0 / np.sqrt(max(typical_variance, 1e-12))
        ou_traces = _generate_ou_process(num_frames, sampling_frequency, neuropil_rng, num_traces=n_sources)

    noise_rng = np.random.default_rng(noise_seed)
    slab_size = 256
    for t0 in range(0, num_frames, slab_size):
        sl = flat[t0 : t0 + slab_size]
        bleach_sl = bleach[t0 : t0 + slab_size]
        if neuropil_model == "constant":
            sl += (background * bleach_sl)[:, np.newaxis]
        elif neuropil_model == "vignette":
            sl += (background * bleach_sl * fluctuation[t0 : t0 + slab_size])[:, np.newaxis] * profile_flat[
                np.newaxis, :
            ]
        else:  # "diffuse"
            # Zero-mean-at-every-pixel fluctuation (each source's own trace has mean 0), so the
            # background's mean level is exactly `background * bleach * vignette_profile`
            # everywhere, same as "vignette" -- only the *fluctuation pattern* differs (a
            # spatially-varying mixture of many sources, rather than one shared trace). A pixel
            # far from every source simply doesn't fluctuate (behaves like "vignette" with
            # `neuropil_fluctuation_std=0` there), rather than losing its background entirely.
            # Built as a (slab, K) @ (K, H*W*P) matmul, K = n_sources -- the same pattern the
            # ROI signal itself uses above -- so no full (num_frames, H*W*P) array is ever
            # materialized.
            modulation_sl = (ou_traces[t0 : t0 + slab_size] @ footprints_flat) * modulation_scale
            sl += (
                (background * bleach_sl)[:, np.newaxis]
                * profile_flat[np.newaxis, :]
                * (1.0 + neuropil_fluctuation_std * modulation_sl)
            )

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
        ``round(uniform(0.5, 1.5) * event_rate * num_frames / sampling_frequency)``, scaling
        with the recording's duration. Clamped to ``[0, num_frames]``.
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
        See :class:`FluorescenceData` for field descriptions.
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
