from __future__ import annotations

import numpy as np
import pytest

from photon_mosaic.core.baserois import BaseRois
from photon_mosaic.core.generators import generate_rois


@pytest.fixture
def make_rois_for_tests() -> tuple[BaseRois, dict]:
    roi_kwargs = dict(
        num_rois=3,
        height=50,
        width=50,
        radius_range=(5, 7),
        sampling_frequency=31.2,
        roi_ids=[10, 11, 12],
    )
    rois = generate_rois(
        **roi_kwargs,
    )
    return rois, roi_kwargs


class MockImaging:
    def __init__(self, sampling_frequency):
        self.sampling_frequency = sampling_frequency


def test_init_stores_sampling_frequency_shape_and_roi_ids(make_rois_for_tests):
    rois, roi_kwargs = make_rois_for_tests
    assert rois.sampling_frequency == roi_kwargs["sampling_frequency"]
    assert np.all(np.asarray(rois.image_shape) == np.array([roi_kwargs["height"], roi_kwargs["width"]]))
    assert np.all(np.asarray(rois.roi_ids) == np.array(roi_kwargs["roi_ids"]))
    assert rois.get_num_rois() == roi_kwargs["num_rois"]
    assert rois.get_num_segments() == 1
    assert rois.imaging is None
    assert rois.has_imaging() is False


def test_repr_contains_roi_count_and_shape(make_rois_for_tests):
    rois, roi_kwargs = make_rois_for_tests
    text = repr(rois)
    assert "ROIs" in text
    assert f"{roi_kwargs['num_rois']} ROIs" in text
    assert f"{roi_kwargs['height']} rows x {roi_kwargs['width']} columns" in text

    # test with Name
    rois.name = "TestRois"
    text = repr(rois)
    assert "TestRois" in text

    text_html = rois._repr_html_()
    assert "ROIs" in text_html
    assert f"{roi_kwargs['num_rois']} ROIs" in text_html
    assert f"{roi_kwargs['height']} rows x {roi_kwargs['width']} columns" in text_html


def test_roi_image_masks(make_rois_for_tests):
    rois, roi_kwargs = make_rois_for_tests

    # This returns an array of shape (num_rois, height, width)
    image_masks = rois.get_roi_image_masks()
    assert isinstance(image_masks, np.ndarray)
    assert len(image_masks) == roi_kwargs["num_rois"]
    assert image_masks.shape[1:] == (roi_kwargs["height"], roi_kwargs["width"])

    for mask in image_masks:
        # Check that mask values are either 0 or 1
        unique_values = np.unique(mask)
        assert np.all(np.isin(unique_values, [0, 1]))

    with pytest.raises(ValueError):
        rois.get_roi_image_masks(roi_ids=[11, 99])


def test_get_roi_pixel_masks_default_all_rois_and_weights_preserved(make_rois_for_tests):
    rois, roi_kwargs = make_rois_for_tests
    pixel_masks = rois.get_roi_pixel_masks()
    assert isinstance(pixel_masks, list)
    assert len(pixel_masks) == roi_kwargs["num_rois"]

    for pm in pixel_masks:
        assert isinstance(pm, np.ndarray)
        assert pm.shape[1] == 3  # y, x, weight columns
        assert len(np.unique(pm[:, 2])) == 1 and np.all(pm[:, 2] == 1)  # not wiughted
        # Check that all y,x are within image bounds
        assert np.all(pm[:, 0] >= 0) and np.all(pm[:, 0] < roi_kwargs["height"])
        assert np.all(pm[:, 1] >= 0) and np.all(pm[:, 1] < roi_kwargs["width"])

    with pytest.raises(ValueError):
        rois.get_roi_pixel_masks(roi_ids=[12, 999])

    # test with weights
    rois_weighted = generate_rois(weighted=True, **roi_kwargs)
    pixel_masks_weighted = rois_weighted.get_roi_pixel_masks()
    assert isinstance(pixel_masks_weighted, list)
    assert len(pixel_masks_weighted) == roi_kwargs["num_rois"]

    for pm in pixel_masks_weighted:
        assert isinstance(pm, np.ndarray)
        assert pm.shape[1] == 3  # y, x, weight columns
        assert len(np.unique(pm[:, 2])) > 1  # weighted
        # Check that all y,x are within image bounds
        assert np.all(pm[:, 0] >= 0) and np.all(pm[:, 0] < roi_kwargs["height"])
        assert np.all(pm[:, 1] >= 0) and np.all(pm[:, 1] < roi_kwargs["width"])


def test_get_roi_pixel_masks_subset_order_is_respected(make_rois_for_tests):
    rois, roi_kwargs = make_rois_for_tests

    roi_ids = roi_kwargs["roi_ids"]
    pixel_masks = rois.get_roi_pixel_masks(roi_ids=[roi_ids[1], roi_ids[0]])
    assert len(pixel_masks) == 2

    all_pixel_masks = rois.get_roi_pixel_masks()

    # first returned corresponds to ROI 11
    pm_first = pixel_masks[0]
    assert np.array_equal(pm_first, all_pixel_masks[1])

    # second corresponds to ROI 10 (two pixels)
    pm_second = pixel_masks[1]
    assert np.array_equal(pm_second, all_pixel_masks[0])


def test_select_rois_returns_selected_rois_and_masks_match(make_rois_for_tests):
    rois, roi_kwargs = make_rois_for_tests

    selected = rois.select_rois([11])

    assert selected.get_num_rois() == 1
    assert np.all(np.asarray(selected.roi_ids) == np.array([11]))
    assert np.all(np.asarray(selected.image_shape) == np.array([roi_kwargs["height"], roi_kwargs["width"]]))
    assert selected.sampling_frequency == rois.sampling_frequency

    # ensure it proxies masks from the source
    mask = selected.get_roi_image_masks()[0]
    original_mask = rois.get_roi_image_masks(roi_ids=[11])[0]
    assert np.array_equal(mask, original_mask)

    with pytest.raises(ValueError):
        _ = rois.select_rois([999])


def test_register_imaging_sets_imaging_and_has_imaging_true(make_rois_for_tests):
    rois, roi_kwargs = make_rois_for_tests

    imaging = MockImaging(roi_kwargs["sampling_frequency"])
    rois.register_imaging(imaging)

    assert rois.has_imaging() is True
    assert rois.imaging is imaging

    # select ROIS preserves imaging association
    selected = rois.select_rois([10])
    assert selected.has_imaging() is True
    assert selected.imaging is imaging

    imaging = MockImaging(35.0)
    with pytest.raises(AssertionError, match="different sampling frequency"):
        rois.register_imaging(imaging)
