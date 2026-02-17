import numpy as np
import pytest

from photon_mosaic.extractors.suite2prois import Suite2pRois, read_suite2p_rois


def _make_suite2p_folder(tmp_path, num_rois=5, height=64, width=64, fs=31.2, include_iscell=True):
    """Create a minimal Suite2p output folder with ops.npy, stat.npy, and optionally iscell.npy."""
    ops = {
        "Lyc": height,
        "Lxc": width,
        "fs": fs,
    }
    np.save(tmp_path / "ops.npy", ops)

    rng = np.random.default_rng(42)
    stats = []
    for _ in range(num_rois):
        num_pixels = rng.integers(5, 20)
        ypix = rng.integers(0, height, size=num_pixels)
        xpix = rng.integers(0, width, size=num_pixels)
        stat = {
            "ypix": ypix,
            "xpix": xpix,
            "lam": rng.random(num_pixels),
            "soma_crop": np.array([True]),
            "overlap": np.array([False]),
            "neuropil_mask": np.array([]),
            "compact": 1.2,
            "footprint": 0,
            "radius": 5.0,
            "skew": 0.3,
        }
        stats.append(stat)
    np.save(tmp_path / "stat.npy", np.array(stats, dtype=object))

    if include_iscell:
        iscell = np.column_stack(
            [
                rng.integers(0, 2, size=num_rois).astype(float),
                rng.random(num_rois),
            ]
        )
        np.save(tmp_path / "iscell.npy", iscell)

    return ops, stats


@pytest.fixture
def suite2p_folder(tmp_path):
    ops, stats = _make_suite2p_folder(tmp_path)
    return tmp_path, ops, stats


@pytest.fixture
def suite2p_folder_no_iscell(tmp_path):
    ops, stats = _make_suite2p_folder(tmp_path, include_iscell=False)
    return tmp_path, ops, stats


def test_init_loads_sampling_frequency_shape_and_roi_ids(suite2p_folder):
    folder, ops, stats = suite2p_folder
    rois = Suite2pRois(folder)

    assert rois.sampling_frequency == ops["fs"]
    assert rois.shape[0] == ops["Lyc"]
    assert rois.shape[1] == ops["Lxc"]
    assert rois.get_num_rois() == len(stats)
    assert np.array_equal(rois.roi_ids, np.arange(len(stats)))


def test_init_raises_if_ops_missing(tmp_path):
    np.save(tmp_path / "stat.npy", np.array([{}], dtype=object))
    with pytest.raises(FileNotFoundError, match="ops.npy"):
        Suite2pRois(tmp_path)


def test_init_raises_if_stat_missing(tmp_path):
    np.save(tmp_path / "ops.npy", {"Lyc": 64, "Lxc": 64, "fs": 30.0})
    with pytest.raises(FileNotFoundError, match="stat.npy"):
        Suite2pRois(tmp_path)


def test_get_roi_image_masks_returns_correct_shape(suite2p_folder):
    folder, ops, stats = suite2p_folder
    rois = Suite2pRois(folder)
    masks = rois.get_roi_image_masks()

    assert isinstance(masks, np.ndarray)
    assert masks.shape == (len(stats), ops["Lyc"], ops["Lxc"])
    assert masks.dtype == bool


def test_get_roi_image_masks_pixels_match_stat(suite2p_folder):
    folder, ops, stats = suite2p_folder
    rois = Suite2pRois(folder)
    masks = rois.get_roi_image_masks()

    for i, stat in enumerate(stats):
        mask = masks[i]
        ypix, xpix = stat["ypix"], stat["xpix"]
        # All stat pixels should be True
        assert np.all(mask[ypix, xpix])
        # Total True count should match number of unique pixel positions
        unique_coords = set(zip(ypix.tolist(), xpix.tolist()))
        assert mask.sum() == len(unique_coords)


def test_get_roi_image_masks_subset(suite2p_folder):
    folder, ops, stats = suite2p_folder
    rois = Suite2pRois(folder)

    subset_ids = [0, 2]
    masks_subset = rois.get_roi_image_masks(roi_ids=subset_ids)
    masks_all = rois.get_roi_image_masks()

    assert masks_subset.shape[0] == len(subset_ids)
    np.testing.assert_array_equal(masks_subset[0], masks_all[0])
    np.testing.assert_array_equal(masks_subset[1], masks_all[2])


def test_properties_set_from_stats_excluding_skipped(suite2p_folder):
    folder, ops, stats = suite2p_folder
    rois = Suite2pRois(folder)

    skip_properties = {"xpix", "ypix", "lam", "soma_crop", "overlap", "neuropil_mask"}
    expected_properties = {k for k in stats[0].keys() if k not in skip_properties}

    for prop in expected_properties:
        values = rois.get_property(prop)
        assert values is not None, f"Property '{prop}' should be set"
        assert len(values) == len(stats)

    # Skipped properties should not be set
    for prop in skip_properties:
        values = rois.get_property(prop)
        assert values is None, f"Skipped property '{prop}' should not be set"


def test_iscell_properties_loaded(suite2p_folder):
    folder, ops, stats = suite2p_folder
    rois = Suite2pRois(folder)

    iscell_bool = rois.get_property("iscell")
    iscell_prob = rois.get_property("iscell_probability")

    assert iscell_bool is not None
    assert iscell_prob is not None
    assert len(iscell_bool) == len(stats)
    assert len(iscell_prob) == len(stats)
    assert iscell_bool.dtype == bool
    assert np.all((iscell_prob >= 0) & (iscell_prob <= 1))


def test_no_iscell_file_works(suite2p_folder_no_iscell):
    folder, ops, stats = suite2p_folder_no_iscell
    rois = Suite2pRois(folder)

    assert rois.get_property("iscell") is None
    assert rois.get_property("iscell_probability") is None
    assert rois.get_num_rois() == len(stats)


def test_read_suite2p_rois_is_suite2p_rois_class():
    assert read_suite2p_rois is Suite2pRois


def test_get_roi_pixel_masks_from_suite2p(suite2p_folder):
    folder, ops, stats = suite2p_folder
    rois = Suite2pRois(folder)
    pixel_masks = rois.get_roi_pixel_masks()

    assert isinstance(pixel_masks, list)
    assert len(pixel_masks) == len(stats)

    for pm in pixel_masks:
        assert isinstance(pm, np.ndarray)
        assert pm.shape[1] == 3  # y, x, weight (single plane)
        assert np.all(pm[:, 0] >= 0) and np.all(pm[:, 0] < ops["Lyc"])
        assert np.all(pm[:, 1] >= 0) and np.all(pm[:, 1] < ops["Lxc"])


def test_suite2p_rois_select_rois(suite2p_folder):
    folder, ops, stats = suite2p_folder
    rois = Suite2pRois(folder)

    selected = rois.select_rois([1, 3])
    assert selected.get_num_rois() == 2
    assert np.array_equal(selected.roi_ids, np.array([1, 3]))

    mask_selected = selected.get_roi_image_masks()
    mask_original = rois.get_roi_image_masks(roi_ids=[1, 3])
    np.testing.assert_array_equal(mask_selected, mask_original)
