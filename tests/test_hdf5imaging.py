"""Tests for HDF5Imaging and HDF5ImagingEpoch."""


import h5py
import numpy as np
import pytest

from photon_mosaic.core.hdf5imaging import HDF5Imaging, HDF5ImagingEpoch


@pytest.fixture
def h5_path(tmp_path):
    """Create a temporary HDF5 file with test data."""
    path = tmp_path / "test.h5"
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, size=(50, 32, 64), dtype=np.int16)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data)
    return str(path), data


class TestHDF5Imaging:
    def test_shape_and_dtype(self, h5_path):
        path, raw = h5_path
        imaging = HDF5Imaging(path, dataset_key="data", sampling_frequency=30.0)
        assert imaging.shape == (32, 64, 1)
        assert imaging.get_num_frames() == 50
        assert imaging.get_num_epochs() == 1
        assert imaging.get_dtype() == np.int16

    def test_get_series_full(self, h5_path):
        path, raw = h5_path
        imaging = HDF5Imaging(path, sampling_frequency=30.0)
        series = imaging.get_series()
        assert series.shape == (50, 32, 64, 1)
        np.testing.assert_array_equal(series[:, :, :, 0], raw)

    def test_get_series_time_selection(self, h5_path):
        path, raw = h5_path
        imaging = HDF5Imaging(path, sampling_frequency=30.0)
        series = imaging.get_series(start_frame=10, end_frame=20)
        assert series.shape == (10, 32, 64, 1)
        np.testing.assert_array_equal(series[:, :, :, 0], raw[10:20])

    def test_get_series_spatial_row_range(self, h5_path):
        path, raw = h5_path
        imaging = HDF5Imaging(path, sampling_frequency=30.0)
        series = imaging.get_series(row_range=(5, 15))
        assert series.shape == (50, 10, 64, 1)
        np.testing.assert_array_equal(series[:, :, :, 0], raw[:, 5:15, :])

    def test_get_series_spatial_col_range(self, h5_path):
        path, raw = h5_path
        imaging = HDF5Imaging(path, sampling_frequency=30.0)
        series = imaging.get_series(col_range=(10, 30))
        assert series.shape == (50, 32, 20, 1)
        np.testing.assert_array_equal(series[:, :, :, 0], raw[:, :, 10:30])

    def test_get_series_spatial_both(self, h5_path):
        path, raw = h5_path
        imaging = HDF5Imaging(path, sampling_frequency=30.0)
        series = imaging.get_series(row_range=(5, 15), col_range=(10, 30))
        assert series.shape == (50, 10, 20, 1)
        np.testing.assert_array_equal(series[:, :, :, 0], raw[:, 5:15, 10:30])

    def test_get_series_combined(self, h5_path):
        """Test time + spatial selection together."""
        path, raw = h5_path
        imaging = HDF5Imaging(path, sampling_frequency=30.0)
        series = imaging.get_series(start_frame=5, end_frame=15, row_range=(2, 20), col_range=(10, 50))
        assert series.shape == (10, 18, 40, 1)
        np.testing.assert_array_equal(series[:, :, :, 0], raw[5:15, 2:20, 10:50])

    def test_get_series_plane_indices(self, h5_path):
        """Test plane selection (single-plane HDF5 expanded to 4D)."""
        path, raw = h5_path
        imaging = HDF5Imaging(path, sampling_frequency=30.0)
        series = imaging.get_series(start_frame=0, end_frame=5, plane_ids=[0])
        assert series.shape == (5, 32, 64, 1)

    def test_custom_dataset_key(self, tmp_path):
        """Test loading from a non-default dataset key."""
        path = tmp_path / "custom.h5"
        rng = np.random.default_rng(0)
        data = rng.random((10, 8, 8)).astype(np.float32)
        with h5py.File(path, "w") as f:
            f.create_dataset("my_video", data=data)
        imaging = HDF5Imaging(str(path), dataset_key="my_video", sampling_frequency=15.0)
        assert imaging.shape == (8, 8, 1)
        assert imaging.sampling_frequency == 15.0

    def test_repr(self, h5_path):
        path, _ = h5_path
        imaging = HDF5Imaging(path, sampling_frequency=30.0)
        r = repr(imaging)
        assert "HDF5Imaging" in r
        assert "32 rows x 64 columns" in r


class TestHDF5ImagingEpoch:
    def test_get_num_samples(self, h5_path):
        path, _ = h5_path
        with h5py.File(path, "r") as f:
            epoch = HDF5ImagingEpoch(f["data"], sampling_frequency=30.0)
            assert epoch.get_num_samples() == 50

    def test_get_series_defaults(self, h5_path):
        path, raw = h5_path
        with h5py.File(path, "r") as f:
            epoch = HDF5ImagingEpoch(f["data"], sampling_frequency=30.0)
            series = epoch.get_series(0, 10)
            assert series.shape == (10, 32, 64, 1)
            np.testing.assert_array_equal(series[:, :, :, 0], raw[0:10])
