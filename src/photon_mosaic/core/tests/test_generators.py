import numpy as np
import pytest

from photon_mosaic.core.generators import generate_random_imaging, generate_rois


def test_generate_random_imaging_int_vs_singleton_tuple_same_output():
    n, h, w, sf = 8, 6, 7, 30.0
    seed = 123

    imaging_int = generate_random_imaging(num_frames=n, height=h, width=w, sampling_frequency=sf, seed=seed)
    imaging_tuple = generate_random_imaging(num_frames=(n,), height=h, width=w, sampling_frequency=sf, seed=seed)

    assert imaging_int.get_num_segments() == 1
    assert imaging_tuple.get_num_segments() == 1

    np.testing.assert_allclose(imaging_int.get_series(), imaging_tuple.get_series())
    assert imaging_int.get_shape() == imaging_tuple.get_shape() == (n, h, w)


def test_generate_random_imaging_multisegment_shapes_and_reproducibility_multiplane():
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

    assert imaging1.get_num_segments() == 2
    assert imaging1.get_num_planes() == p

    s10 = imaging1.get_series(segment_index=0)
    s11 = imaging1.get_series(segment_index=1)
    assert s10.shape == (num_frames[0], h, w, p)
    assert s11.shape == (num_frames[1], h, w, p)

    np.testing.assert_allclose(imaging1.get_series(segment_index=0), imaging2.get_series(segment_index=0))
    np.testing.assert_allclose(imaging1.get_series(segment_index=1), imaging2.get_series(segment_index=1))


def test_generate_random_imaging_num_planes_1_returns_3d_series():
    n, h, w = 5, 8, 9
    imaging = generate_random_imaging(num_frames=n, height=h, width=w, sampling_frequency=10.0, num_planes=1, seed=0)

    series = imaging.get_series()
    assert series.shape == (n, h, w)
    assert series.ndim == 3
    assert imaging.get_shape() == (n, h, w)


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
