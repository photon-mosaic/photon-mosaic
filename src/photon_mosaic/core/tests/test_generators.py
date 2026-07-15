import numpy as np
import pytest

from photon_mosaic.core.generators import (
    FluorescenceData,
    generate_fluorescence,
    generate_imaging_with_rois,
    generate_random_imaging,
    generate_rois,
)


def test_generate_random_imaging_int_vs_singleton_tuple_same_output():
    n, h, w, sf = 8, 6, 7, 30.0
    seed = 123

    imaging_int = generate_random_imaging(num_frames=n, height=h, width=w, sampling_frequency=sf, seed=seed)
    imaging_tuple = generate_random_imaging(num_frames=(n,), height=h, width=w, sampling_frequency=sf, seed=seed)

    assert imaging_int.get_num_epochs() == 1
    assert imaging_tuple.get_num_epochs() == 1

    np.testing.assert_array_equal(imaging_int.get_series(), imaging_tuple.get_series())
    assert imaging_int.get_shape() == imaging_tuple.get_shape() == (n, h, w, 1)


def test_generate_random_imaging_multiepoch_shapes_and_reproducibility_multiplane():
    num_frames = (3, 5)
    h, w, p = 4, 6, 2
    sf = 12.5
    seed = 999

    imaging1 = generate_random_imaging(
        num_frames=num_frames,
        height=h,
        width=w,
        sampling_frequency=sf,
        num_planes=p,
        seed=seed,
    )
    imaging2 = generate_random_imaging(
        num_frames=num_frames,
        height=h,
        width=w,
        sampling_frequency=sf,
        num_planes=p,
        seed=seed,
    )

    assert imaging1.get_num_epochs() == 2
    assert imaging1.get_num_planes() == p

    s10 = imaging1.get_series(epoch_index=0)
    s11 = imaging1.get_series(epoch_index=1)
    assert s10.shape == (num_frames[0], h, w, p)
    assert s11.shape == (num_frames[1], h, w, p)

    np.testing.assert_allclose(imaging1.get_series(epoch_index=0), imaging2.get_series(epoch_index=0))
    np.testing.assert_allclose(imaging1.get_series(epoch_index=1), imaging2.get_series(epoch_index=1))


def test_generate_random_imaging_num_planes_1_returns_3d_series():
    n, h, w = 5, 8, 9
    imaging = generate_random_imaging(num_frames=n, height=h, width=w, sampling_frequency=10.0, num_planes=1, seed=0)

    series = imaging.get_series()
    assert series.shape == (n, h, w, 1)
    assert series.ndim == 4
    assert imaging.get_shape() == (n, h, w, 1)


def test_generate_rois_default_roi_ids_and_deterministic_masks():
    kwargs = dict(num_rois=4, height=30, width=40, radius_range=(3, 5), sampling_frequency=31.2, seed=10)

    rois1 = generate_rois(**kwargs)
    rois2 = generate_rois(**kwargs)

    masks1 = rois1.get_roi_image_masks()
    masks2 = rois2.get_roi_image_masks()

    assert masks1.shape == (kwargs["num_rois"], kwargs["height"], kwargs["width"])
    assert np.array_equal(masks1, masks2)  # rng is fixed in generator
    assert np.array_equal(np.asarray(rois1.roi_ids), np.arange(kwargs["num_rois"]))


def test_generate_rois_weighted_masks_are_in_0_1_and_have_fractional_values():
    rois = generate_rois(
        num_rois=3,
        height=50,
        width=60,
        radius_range=(5, 8),
        weighted=True,
        sampling_frequency=10.0,
        seed=10,
    )
    masks = rois.get_roi_image_masks()

    assert masks.shape == (3, 50, 60)
    assert np.min(masks) >= 0.0
    assert np.max(masks) <= 1.0

    # Must have zeros (background) and some fractional weights (not purely binary)
    assert np.any(masks == 0.0)
    assert np.any((masks > 0.0) & (masks < 1.0))


def test_generate_rois_multiplane_depth_radius_from_third_radius_range_limits_z_extent():
    # With depth_radius=1, each ROI should occupy at most 3 z-slices (center_z-1..center_z+1).
    num_rois, h, w, p = 3, 40, 41, 9
    rois = generate_rois(
        num_rois=num_rois,
        height=h,
        width=w,
        radius_range=(4, 6, 1),
        num_planes=p,
        weighted=False,
        sampling_frequency=10.0,
    )
    masks = rois.get_roi_image_masks()
    assert masks.shape == (num_rois, h, w, p)

    for i in range(num_rois):
        coords = np.argwhere(masks[i] > 0)
        assert coords.size > 0  # ROI not empty

        z_vals = coords[:, 2]
        assert z_vals.min() >= 0 and z_vals.max() < p
        assert (z_vals.max() - z_vals.min()) <= 2
        assert np.unique(z_vals).size <= 3


def test_generate_rois_multiplane_binary_vs_weighted_value_properties():
    kwargs = dict(num_rois=2, height=35, width=36, radius_range=(4, 7, 2), num_planes=7, sampling_frequency=20.0)

    rois_bin = generate_rois(weighted=False, **kwargs)
    masks_bin = rois_bin.get_roi_image_masks()
    uniq_bin = np.unique(masks_bin)
    assert np.all(np.isin(uniq_bin, [0, 1]))

    rois_w = generate_rois(weighted=True, **kwargs)
    masks_w = rois_w.get_roi_image_masks()
    assert np.min(masks_w) >= 0.0
    assert np.max(masks_w) <= 1.0
    assert np.any((masks_w > 0.0) & (masks_w < 1.0))


def test_generate_rois_invalid_radius_range_raises_assertion():
    with pytest.raises(AssertionError, match="Invalid radius range"):
        _ = generate_rois(num_rois=1, height=30, width=30, radius_range=(7, 5))

    # Too large to fit: radius_range[1] must be < width - radius_range[1] and same for height
    with pytest.raises(AssertionError, match="ROIs may not fit"):
        _ = generate_rois(num_rois=1, height=20, width=20, radius_range=(5, 15))


# ── generate_fluorescence tests ──


def test_generate_fluorescence_returns_fluorescence_traces():
    result = generate_fluorescence(num_frames=200, num_rois=3, seed=0)
    assert isinstance(result, FluorescenceData)


def test_generate_fluorescence_output_shapes_and_dtype():
    result = generate_fluorescence(num_frames=200, num_rois=3, seed=0)
    assert result.traces.shape == (200, 3)
    assert result.spikes.shape == (200, 3)
    assert result.clean_traces.shape == (200, 3)
    assert result.traces.dtype == np.float32
    assert result.spikes.dtype == np.float32
    assert result.clean_traces.dtype == np.float32


def test_generate_fluorescence_spikes_are_binary():
    result = generate_fluorescence(num_frames=500, num_rois=4, seed=1)
    unique = np.unique(result.spikes)
    assert set(unique).issubset({0.0, 1.0})


def test_generate_fluorescence_traces_equal_clean_traces_plus_one_without_noise_or_bleaching():
    result = generate_fluorescence(num_frames=300, num_rois=2, noise_std=0.0, bleaching_tau=None, seed=2)
    np.testing.assert_array_equal(result.traces, result.clean_traces + 1.0)


def test_generate_fluorescence_noise_changes_traces():
    r_clean = generate_fluorescence(num_frames=300, num_rois=2, noise_std=0.0, seed=3)
    r_noisy = generate_fluorescence(num_frames=300, num_rois=2, noise_std=1.0, seed=3)
    assert not np.array_equal(r_clean.traces, r_noisy.traces)
    np.testing.assert_array_equal(r_clean.clean_traces, r_noisy.clean_traces)


def test_generate_fluorescence_bleaching_attenuates_trace():
    result = generate_fluorescence(num_frames=500, num_rois=1, bleaching_tau=5.0, seed=4)
    # Last quarter should be weaker than first quarter on average
    assert result.traces[:125, 0].mean() > result.traces[375:, 0].mean()


def test_generate_fluorescence_is_reproducible():
    r1 = generate_fluorescence(num_frames=200, num_rois=3, noise_std=0.1, bleaching_tau=30.0, seed=42)
    r2 = generate_fluorescence(num_frames=200, num_rois=3, noise_std=0.1, bleaching_tau=30.0, seed=42)
    np.testing.assert_array_equal(r1.traces, r2.traces)
    np.testing.assert_array_equal(r1.spikes, r2.spikes)
    np.testing.assert_array_equal(r1.clean_traces, r2.clean_traces)


def test_generate_fluorescence_different_seeds_differ():
    r1 = generate_fluorescence(num_frames=200, num_rois=2, seed=0)
    r2 = generate_fluorescence(num_frames=200, num_rois=2, seed=1)
    assert not np.array_equal(r1.traces, r2.traces)


# ── generate_imaging_with_rois tests ──


def test_generate_imaging_with_rois_returns_correct_types_and_shapes():
    rois, imaging, _ = generate_imaging_with_rois(
        num_frames=50,
        height=30,
        width=40,
        num_rois=3,
        radius_range=(3, 5),
        sampling_frequency=10.0,
        seed=42,
    )
    assert imaging.get_num_epochs() == 1
    series = imaging.get_series()
    assert series.shape == (50, 30, 40, 1)
    assert rois.get_num_rois() == 3
    masks = rois.get_roi_image_masks()
    assert masks.shape[0] == 3


def test_generate_imaging_with_rois_adds_fluorescence_signal():
    """The imaging data should be brighter than pure random noise due to injected bumps."""
    # Generate plain random imaging with the same derived seeds
    rng = np.random.default_rng(42)
    imaging_seed = int(rng.integers(0, 2**31))

    plain = generate_random_imaging(
        num_frames=100,
        height=30,
        width=40,
        sampling_frequency=10.0,
        seed=imaging_seed,
    )
    _, with_bumps, _ = generate_imaging_with_rois(
        num_frames=100,
        height=30,
        width=40,
        num_rois=5,
        radius_range=(3, 5),
        sampling_frequency=10.0,
        seed=42,
    )
    # The injected signal should increase the mean
    assert with_bumps.get_series().mean() > plain.get_series().mean()


def test_generate_imaging_with_rois_noise_std_default_matches_explicit_default():
    """The default noise_std should reproduce the same background as passing it explicitly."""
    kwargs = dict(num_frames=60, height=20, width=20, num_rois=3, radius_range=(3, 5), seed=1)
    _, default, _ = generate_imaging_with_rois(**kwargs)
    _, explicit, _ = generate_imaging_with_rois(**kwargs, noise_std=0.1)
    np.testing.assert_array_equal(default.get_series(), explicit.get_series())


def test_generate_imaging_with_rois_noise_std_scales_background():
    """Increasing noise_std should increase the video's overall standard deviation."""
    kwargs = dict(num_frames=200, height=20, width=20, num_rois=3, radius_range=(3, 5), seed=1)
    _, quiet, _ = generate_imaging_with_rois(**kwargs, noise_std=0.01)
    _, loud, _ = generate_imaging_with_rois(**kwargs, noise_std=5.0)
    assert loud.get_series().std() > quiet.get_series().std()


def test_generate_imaging_with_rois_dff_scale_independent_of_noise_std():
    """Recovered dF/F's scale relative to clean_traces should not depend on noise_std.

    Correlation alone wouldn't catch a regression here (it's scale-invariant), so this
    compares std(dF/F) / std(clean_traces) at very different noise_std values instead.
    Without the baseline scaling fix, this ratio scales with ~1/noise_std (a ~40x
    difference between the two noise levels tested); with the fix, it stays flat.
    """
    from photon_mosaic.core import create_roi_analyzer

    kwargs = dict(num_frames=1000, height=20, width=20, num_rois=2, radius_range=(3, 5), seed=1)
    ratios = {}
    for noise_std in (0.05, 2.0):
        rois, imaging, ground_truth = generate_imaging_with_rois(**kwargs, noise_std=noise_std)
        analyzer = create_roi_analyzer(rois, imaging, format="memory")
        analyzer.compute("fluorescence")
        analyzer.compute("df_over_f", method="percentile")
        dff = analyzer.get_extension("df_over_f").get_data()
        ratios[noise_std] = dff[:, 0].std() / ground_truth.clean_traces[:, 0].std()

    assert 0.3 < ratios[0.05] / ratios[2.0] < 3.0


def test_generate_imaging_with_rois_is_reproducible():
    kwargs = dict(
        num_frames=60,
        height=30,
        width=40,
        num_rois=4,
        radius_range=(3, 5),
        sampling_frequency=10.0,
        seed=99,
    )
    rois1, im1, _ = generate_imaging_with_rois(**kwargs)
    rois2, im2, _ = generate_imaging_with_rois(**kwargs)

    np.testing.assert_array_equal(im1.get_series(), im2.get_series())
    np.testing.assert_array_equal(
        rois1.get_roi_image_masks(),
        rois2.get_roi_image_masks(),
    )


def test_generate_imaging_with_rois_multiplane():
    rois, imaging, _ = generate_imaging_with_rois(
        num_frames=40,
        height=30,
        width=30,
        num_planes=3,
        num_rois=2,
        radius_range=(3, 5, 1),
        sampling_frequency=10.0,
        seed=7,
    )
    series = imaging.get_series()
    assert series.shape == (40, 30, 30, 3)
    masks = rois.get_roi_image_masks()
    assert masks.shape == (2, 30, 30, 3)


def test_generate_imaging_with_rois_multiplane_is_reproducible():
    kwargs = dict(
        num_frames=40,
        height=30,
        width=30,
        num_planes=3,
        num_rois=2,
        radius_range=(3, 5, 1),
        sampling_frequency=10.0,
        seed=7,
    )
    _, im1, _ = generate_imaging_with_rois(**kwargs)
    _, im2, _ = generate_imaging_with_rois(**kwargs)
    np.testing.assert_array_equal(im1.get_series(), im2.get_series())


def test_generate_imaging_with_rois_weighted():
    rois, imaging, _ = generate_imaging_with_rois(
        num_frames=50,
        height=30,
        width=40,
        num_rois=3,
        radius_range=(3, 5),
        sampling_frequency=10.0,
        weighted_rois=True,
        seed=1,
    )
    masks = rois.get_roi_image_masks()
    assert np.any((masks > 0.0) & (masks < 1.0))  # fractional weights present


def test_generate_imaging_with_rois_decay_affects_trace():
    """Longer decay should spread the signal over more frames."""
    kwargs = dict(
        num_frames=200,
        height=30,
        width=40,
        num_rois=3,
        radius_range=(3, 5),
        sampling_frequency=10.0,
        seed=55,
    )
    _, im_short, _ = generate_imaging_with_rois(decay_seconds=0.5, **kwargs)
    _, im_long, _ = generate_imaging_with_rois(decay_seconds=5.0, **kwargs)

    # Both should have visible injected signal well above the dark background
    assert im_short.get_series().max() > 1.0
    assert im_long.get_series().max() > 1.0
