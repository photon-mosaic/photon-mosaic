"""Tests for RoiAnalyzer and the extension system."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from photon_mosaic.core import (
    AnalyzerExtension,
    RoiAnalyzer,
    create_roi_analyzer,
    load_roi_analyzer,
    register_result_extension,
)
from photon_mosaic.core.generators import generate_random_imaging, generate_rois

# ---------------------------------------------------------------------------
# Dummy extension for testing the extension system
# ---------------------------------------------------------------------------


class DummyAnalyzerExtension(AnalyzerExtension):
    extension_name = "dummy"
    depend_on = []
    need_imaging = False
    use_nodepipeline = False
    need_job_kwargs = False

    def _set_params(self, param1=5):
        return dict(param1=param1)

    def _run(self, **kwargs):
        roi_ids = self.roi_analyzer.roi_ids
        self.data["result"] = np.arange(len(roi_ids)) * self.params["param1"]

    def _select_extension_data(self, roi_ids):
        all_ids = self.roi_analyzer.roi_ids
        mask = np.isin(all_ids, roi_ids)
        return {"result": self.data["result"][mask]}

    def _get_data(self):
        return self.data["result"]


register_result_extension(DummyAnalyzerExtension)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def imaging():
    return generate_random_imaging(num_frames=100, height=64, width=64, sampling_frequency=30.0, seed=42)


@pytest.fixture
def rois():
    return generate_rois(num_rois=10, height=64, width=64, sampling_frequency=30.0, seed=42)


@pytest.fixture
def analyzer(rois, imaging):
    return create_roi_analyzer(rois, imaging, format="memory")


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Basic creation and properties
# ---------------------------------------------------------------------------


class TestCreateMemory:
    def test_create_memory(self, analyzer, rois, imaging):
        assert isinstance(analyzer, RoiAnalyzer)
        assert analyzer.format == "memory"
        assert analyzer.has_imaging()
        assert not analyzer.has_temporary_imaging()

    def test_properties(self, analyzer, rois, imaging):
        assert analyzer.get_num_rois() == 10
        assert analyzer.get_num_epochs() == 1
        assert analyzer.shape == (64, 64, 1)
        assert analyzer.num_planes == 1
        assert np.array_equal(analyzer.roi_ids, rois.roi_ids)
        assert analyzer.sampling_frequency == 30.0

    def test_repr(self, analyzer):
        r = repr(analyzer)
        assert "RoiAnalyzer" in r
        assert "10 ROIs" in r
        assert "memory" in r

    def test_get_num_samples(self, analyzer):
        assert analyzer.get_num_samples() == 100
        assert analyzer.get_total_samples() == 100

    def test_get_dtype(self, analyzer):
        assert analyzer.get_dtype() == np.float64

    def test_is_read_only(self, analyzer):
        assert not analyzer.is_read_only()

    def test_create_memory_with_sparse_masks(self, imaging):
        """create_roi_analyzer should work with sparse-backed ROIs (e.g. Suite2p output)
        without forcing an implicit (and error-prone) densification -- see photon-mosaic#103."""
        import sparse

        from photon_mosaic.extractors.suite2prois import Suite2pRois

        rng = np.random.default_rng(0)
        stats = [{"ypix": rng.integers(0, 64, size=8), "xpix": rng.integers(0, 64, size=8)} for _ in range(5)]
        sparse_rois = Suite2pRois.from_stat(stats, shape=(64, 64, 1), sampling_frequency=30.0)

        analyzer = create_roi_analyzer(sparse_rois, imaging, format="memory")
        assert analyzer.get_num_rois() == 5
        masks = analyzer.rois.get_roi_image_masks()
        assert isinstance(masks, sparse.SparseArray)
        np.testing.assert_array_equal(masks.todense(), sparse_rois.get_roi_image_masks().todense())


class TestValidation:
    def test_sampling_frequency_mismatch(self, imaging):
        rois = generate_rois(num_rois=5, height=64, width=64, sampling_frequency=15.0, seed=0)
        with pytest.raises(ValueError, match="Sampling frequency mismatch"):
            create_roi_analyzer(rois, imaging)

    def test_shape_mismatch(self, imaging):
        rois = generate_rois(num_rois=5, height=32, width=32, sampling_frequency=30.0, seed=0)
        with pytest.raises(ValueError, match="Spatial shape mismatch"):
            create_roi_analyzer(rois, imaging)


# ---------------------------------------------------------------------------
# Select / remove ROIs
# ---------------------------------------------------------------------------


class TestSelectRemove:
    def test_select_rois(self, analyzer):
        selected_ids = analyzer.roi_ids[:5]
        new_analyzer = analyzer.select_rois(selected_ids)
        assert new_analyzer.get_num_rois() == 5
        assert np.array_equal(new_analyzer.roi_ids, selected_ids)

    def test_remove_rois(self, analyzer):
        remove_ids = analyzer.roi_ids[:3]
        new_analyzer = analyzer.remove_rois(remove_ids)
        assert new_analyzer.get_num_rois() == 7
        assert not np.any(np.isin(remove_ids, new_analyzer.roi_ids))

    def test_select_propagates_extensions(self, analyzer):
        analyzer.compute("dummy", param1=3)
        selected_ids = analyzer.roi_ids[:5]
        new_analyzer = analyzer.select_rois(selected_ids)
        assert new_analyzer.has_extension("dummy")
        data = new_analyzer.get_extension("dummy").get_data()
        assert len(data) == 5


# ---------------------------------------------------------------------------
# Temporary imaging
# ---------------------------------------------------------------------------


class TestTemporaryImaging:
    def test_set_temporary_imaging(self, analyzer, imaging):
        # Remove the imaging reference, then set a temporary one
        analyzer._imaging = None
        assert not analyzer.has_imaging()
        analyzer.set_temporary_imaging(imaging)
        assert analyzer.has_temporary_imaging()
        assert analyzer.imaging is imaging

    def test_temporary_imaging_mismatch(self, analyzer):
        bad_imaging = generate_random_imaging(num_frames=50, height=32, width=32, seed=0)
        with pytest.raises(ValueError):
            analyzer.set_temporary_imaging(bad_imaging)


# ---------------------------------------------------------------------------
# ROI properties
# ---------------------------------------------------------------------------


class TestRoiProperties:
    def test_set_get_property(self, analyzer):
        values = np.arange(analyzer.get_num_rois(), dtype=float)
        analyzer.set_roi_property("quality", values)
        retrieved = analyzer.get_roi_property("quality")
        np.testing.assert_array_equal(values, retrieved)


# ---------------------------------------------------------------------------
# Extension system
# ---------------------------------------------------------------------------


class TestExtensions:
    def test_compute_extension(self, analyzer):
        ext = analyzer.compute("dummy", param1=2)
        assert ext is not None
        data = ext.get_data()
        expected = np.arange(10) * 2
        np.testing.assert_array_equal(data, expected)

    def test_has_extension(self, analyzer):
        assert not analyzer.has_extension("dummy")
        analyzer.compute("dummy")
        assert analyzer.has_extension("dummy")

    def test_delete_extension(self, analyzer):
        analyzer.compute("dummy")
        assert analyzer.has_extension("dummy")
        analyzer.delete_extension("dummy")
        assert not analyzer.has_extension("dummy")

    def test_compute_several(self, analyzer):
        # Only one extension registered, but test the dict path
        analyzer.compute({"dummy": {"param1": 7}})
        assert analyzer.has_extension("dummy")
        data = analyzer.get_extension("dummy").get_data()
        np.testing.assert_array_equal(data, np.arange(10) * 7)

    def test_compute_list(self, analyzer):
        analyzer.compute(["dummy"])
        assert analyzer.has_extension("dummy")


# ---------------------------------------------------------------------------
# Binary folder backend
# ---------------------------------------------------------------------------


class TestBinaryFolder:
    def test_create_and_load(self, rois, imaging, tmp_dir):
        folder = tmp_dir / "test_binary"
        analyzer = create_roi_analyzer(rois, imaging, format="binary_folder", folder=folder)
        assert analyzer.format == "binary_folder"
        assert analyzer.folder == folder

        # Reload
        loaded = load_roi_analyzer(folder)
        assert loaded.get_num_rois() == 10
        assert loaded.shape == (64, 64, 1)
        assert loaded.get_num_epochs() == 1

    def test_overwrite(self, rois, imaging, tmp_dir):
        folder = tmp_dir / "test_overwrite"
        create_roi_analyzer(rois, imaging, format="binary_folder", folder=folder)
        with pytest.raises(ValueError, match="already exists"):
            create_roi_analyzer(rois, imaging, format="binary_folder", folder=folder)
        # Should succeed with overwrite
        create_roi_analyzer(rois, imaging, format="binary_folder", folder=folder, overwrite=True)

    def test_extension_persistence(self, rois, imaging, tmp_dir):
        folder = tmp_dir / "test_ext_persist"
        analyzer = create_roi_analyzer(rois, imaging, format="binary_folder", folder=folder)
        analyzer.compute("dummy", param1=3)

        # Reload and check extension
        loaded = load_roi_analyzer(folder)
        assert loaded.has_extension("dummy")
        data = loaded.get_extension("dummy").get_data()
        np.testing.assert_array_equal(data, np.arange(10) * 3)

    def test_save_as_binary(self, analyzer, tmp_dir):
        folder = tmp_dir / "saved_binary"
        saved = analyzer.save_as(format="binary_folder", folder=folder)
        assert saved.format == "binary_folder"
        assert saved.get_num_rois() == 10

    def test_roi_property_persistence(self, rois, imaging, tmp_dir):
        folder = tmp_dir / "test_prop"
        analyzer = create_roi_analyzer(rois, imaging, format="binary_folder", folder=folder)
        values = np.random.default_rng(0).random(10)
        analyzer.set_roi_property("quality", values)

        loaded = load_roi_analyzer(folder)
        np.testing.assert_array_almost_equal(loaded.get_roi_property("quality"), values)


# ---------------------------------------------------------------------------
# Zarr backend
# ---------------------------------------------------------------------------


class TestZarr:
    def test_create_and_load(self, rois, imaging, tmp_dir):
        folder = tmp_dir / "test.zarr"
        analyzer = create_roi_analyzer(rois, imaging, format="zarr", folder=folder)
        assert analyzer.format == "zarr"

        loaded = load_roi_analyzer(folder)
        assert loaded.get_num_rois() == 10
        assert loaded.shape == (64, 64, 1)

    def test_extension_persistence(self, rois, imaging, tmp_dir):
        folder = tmp_dir / "test_ext.zarr"
        analyzer = create_roi_analyzer(rois, imaging, format="zarr", folder=folder)
        analyzer.compute("dummy", param1=4)

        loaded = load_roi_analyzer(folder)
        assert loaded.has_extension("dummy")
        data = loaded.get_extension("dummy").get_data()
        np.testing.assert_array_equal(data, np.arange(10) * 4)

    def test_save_as_zarr(self, analyzer, tmp_dir):
        folder = tmp_dir / "saved.zarr"
        saved = analyzer.save_as(format="zarr", folder=folder)
        assert saved.format == "zarr"
        assert saved.get_num_rois() == 10


# ---------------------------------------------------------------------------
# Multi-epoch
# ---------------------------------------------------------------------------


class TestMultiEpoch:
    def test_multi_epoch(self):
        imaging = generate_random_imaging(num_frames=(50, 60), height=32, width=32, seed=0)
        rois = generate_rois(num_rois=5, height=32, width=32, sampling_frequency=30.0, seed=0)
        analyzer = create_roi_analyzer(rois, imaging)
        assert analyzer.get_num_epochs() == 2
        assert analyzer.get_num_samples(epoch_index=0) == 50
        assert analyzer.get_num_samples(epoch_index=1) == 60
        assert analyzer.get_total_samples() == 110


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy(self, analyzer):
        analyzer.compute("dummy")
        copied = analyzer.copy()
        assert copied.format == "memory"
        assert copied.get_num_rois() == 10
        assert copied.has_extension("dummy")
