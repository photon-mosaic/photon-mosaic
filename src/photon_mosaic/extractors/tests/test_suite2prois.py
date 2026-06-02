import numpy as np
import pytest

from photon_mosaic.core import BaseRois
from photon_mosaic.extractors.suite2prois import Suite2pRois, read_suite2p_rois


def _make_stats(rng, num_rois, height, width):
    stats = []
    for _ in range(num_rois):
        num_pixels = rng.integers(5, 20)
        stat = {
            "ypix": rng.integers(0, height, size=num_pixels),
            "xpix": rng.integers(0, width, size=num_pixels),
            "lam": rng.random(num_pixels),
            "soma_crop": np.array([True]),
            "overlap": np.array([False]),
            "neuropil_mask": np.array([]),
            "compact": 1.2,
            "footprint": 0,
            "radius": 5.0,
            "skew": 0.3,
            "npix": int(num_pixels),
        }
        stats.append(stat)
    return stats


def _make_suite2p_folder(
    tmp_path,
    num_rois=5,
    height=64,
    width=64,
    fs=31.2,
    include_iscell=True,
):
    """Write a minimal suite2p output folder for ROI tests (no binary movie)."""
    ops = {"Ly": height, "Lx": width, "Lyc": height, "Lxc": width, "fs": fs}
    np.save(tmp_path / "ops.npy", ops)

    rng = np.random.default_rng(42)
    stats = _make_stats(rng, num_rois, height, width)
    np.save(tmp_path / "stat.npy", np.array(stats, dtype=object))

    if include_iscell:
        iscell = np.column_stack(
            [rng.integers(0, 2, size=num_rois).astype(float), rng.random(num_rois)]
        )
        np.save(tmp_path / "iscell.npy", iscell)

    return ops, stats


@pytest.fixture
def suite2p_folder(tmp_path):
    ops, stats = _make_suite2p_folder(tmp_path)
    return tmp_path, ops, stats


# ---------------------------------------------------------------------------
# Folder constructor
# ---------------------------------------------------------------------------


class TestFolderConstructor:
    def test_is_baserois_only(self, suite2p_folder):
        folder, _, _ = suite2p_folder
        rois = Suite2pRois(folder)
        assert isinstance(rois, BaseRois)

    def test_read_suite2p_rois_alias(self):
        assert read_suite2p_rois is Suite2pRois

    def test_loads_sampling_frequency_shape_and_roi_ids(self, suite2p_folder):
        folder, ops, stats = suite2p_folder
        rois = Suite2pRois(folder)

        assert rois.sampling_frequency == ops["fs"]
        assert rois.shape == (ops["Ly"], ops["Lx"], 1)
        assert rois.get_num_rois() == len(stats)
        assert np.array_equal(rois.roi_ids, np.arange(len(stats)))

    def test_raises_if_ops_missing(self, tmp_path):
        np.save(tmp_path / "stat.npy", np.array([{}], dtype=object))
        with pytest.raises(FileNotFoundError, match="ops.npy"):
            Suite2pRois(tmp_path)

    def test_raises_if_stat_missing(self, tmp_path):
        np.save(tmp_path / "ops.npy", {"Ly": 64, "Lx": 64, "fs": 30.0})
        with pytest.raises(FileNotFoundError, match="stat.npy"):
            Suite2pRois(tmp_path)

    def test_falls_back_to_cropped_dims(self, tmp_path):
        np.save(tmp_path / "ops.npy", {"Lyc": 40, "Lxc": 48, "fs": 30.0})
        np.save(
            tmp_path / "stat.npy",
            np.array(_make_stats(np.random.default_rng(0), 3, 40, 48), dtype=object),
        )
        rois = Suite2pRois(tmp_path)
        assert rois.shape[:2] == (40, 48)

    def test_image_masks_shape_and_pixels(self, suite2p_folder):
        folder, ops, stats = suite2p_folder
        rois = Suite2pRois(folder)
        masks = rois.get_roi_image_masks()

        assert masks.shape == (len(stats), ops["Ly"], ops["Lx"])
        assert masks.dtype == bool
        for i, stat in enumerate(stats):
            assert np.all(masks[i][stat["ypix"], stat["xpix"]])

    def test_image_masks_subset(self, suite2p_folder):
        folder, _, _ = suite2p_folder
        rois = Suite2pRois(folder)
        subset = rois.get_roi_image_masks(roi_ids=[0, 2])
        all_masks = rois.get_roi_image_masks()

        assert subset.shape[0] == 2
        np.testing.assert_array_equal(subset[0], all_masks[0])
        np.testing.assert_array_equal(subset[1], all_masks[2])

    def test_pixel_masks(self, suite2p_folder):
        folder, _, stats = suite2p_folder
        rois = Suite2pRois(folder)
        pixel_masks = rois.get_roi_pixel_masks()
        assert len(pixel_masks) == len(stats)
        for pm in pixel_masks:
            assert pm.shape[1] == 3  # y, x, weight (single plane)

    def test_properties_set_excluding_skipped(self, suite2p_folder):
        folder, _, stats = suite2p_folder
        rois = Suite2pRois(folder)
        skip = {"xpix", "ypix", "lam", "soma_crop", "overlap", "neuropil_mask"}

        for prop in (k for k in stats[0] if k not in skip):
            assert rois.get_property(prop) is not None
            assert len(rois.get_property(prop)) == len(stats)
        for prop in skip:
            assert rois.get_property(prop) is None

    def test_iscell_loaded(self, suite2p_folder):
        folder, _, _ = suite2p_folder
        rois = Suite2pRois(folder)
        assert rois.get_property("iscell").dtype == bool
        prob = rois.get_property("iscell_probability")
        assert np.all((prob >= 0) & (prob <= 1))

    def test_no_iscell_file(self, tmp_path):
        _make_suite2p_folder(tmp_path, include_iscell=False)
        rois = Suite2pRois(tmp_path)
        assert rois.get_property("iscell") is None

    def test_select_rois(self, suite2p_folder):
        folder, _, _ = suite2p_folder
        rois = Suite2pRois(folder)
        selected = rois.select_rois([1, 3])
        assert selected.get_num_rois() == 2
        np.testing.assert_array_equal(
            selected.get_roi_image_masks(), rois.get_roi_image_masks(roi_ids=[1, 3])
        )

    def test_no_imaging_registered_by_default(self, suite2p_folder):
        folder, _, _ = suite2p_folder
        rois = Suite2pRois(folder)
        assert not rois.has_imaging()


# ---------------------------------------------------------------------------
# from_stat classmethod
# ---------------------------------------------------------------------------


class TestFromStat:
    def test_builds_rois_from_in_memory_stats(self):
        rng = np.random.default_rng(0)
        stats = _make_stats(rng, num_rois=4, height=32, width=40)

        rois = Suite2pRois.from_stat(
            stats=stats,
            shape=(32, 40, 1),
            sampling_frequency=30.0,
        )

        assert isinstance(rois, Suite2pRois)
        assert isinstance(rois, BaseRois)
        assert rois.sampling_frequency == 30.0
        assert rois.shape == (32, 40, 1)
        assert rois.get_num_rois() == 4
        # Properties from stat are set on the instance
        assert rois.get_property("npix") is not None
        assert len(rois.get_property("npix")) == 4

    def test_image_masks_match_stat_pixels(self):
        rng = np.random.default_rng(1)
        stats = _make_stats(rng, num_rois=3, height=20, width=24)
        rois = Suite2pRois.from_stat(stats, shape=(20, 24, 1), sampling_frequency=15.0)

        masks = rois.get_roi_image_masks()
        assert masks.shape == (3, 20, 24)
        for i, stat in enumerate(stats):
            assert np.all(masks[i][stat["ypix"], stat["xpix"]])

    def test_multiplane_with_assignments(self):
        rng = np.random.default_rng(2)
        stats = _make_stats(rng, num_rois=4, height=10, width=12)
        plane_assignments = np.array([0, 1, 0, 1])

        rois = Suite2pRois.from_stat(
            stats=stats,
            shape=(10, 12, 2),
            sampling_frequency=20.0,
            plane_assignments=plane_assignments,
        )

        masks = rois.get_roi_image_masks()
        assert masks.shape == (4, 10, 12, 2)
        # ROI 0 lights up plane 0 only
        assert masks[0][..., 0].any() and not masks[0][..., 1].any()
        # ROI 1 lights up plane 1 only
        assert masks[1][..., 1].any() and not masks[1][..., 0].any()

    def test_no_imaging_registered_by_default(self):
        rng = np.random.default_rng(3)
        stats = _make_stats(rng, num_rois=2, height=16, width=16)
        rois = Suite2pRois.from_stat(stats, shape=(16, 16, 1), sampling_frequency=10.0)
        assert not rois.has_imaging()
