import numpy as np
import pytest

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.zarrimaging import ZarrImaging


@pytest.fixture
def imaging():
    return generate_random_imaging(num_frames=[100, 20, 30], height=4, width=4, sampling_frequency=10.0, seed=3)


def test_zarr_writing(imaging, tmp_path):
    zarr_path = tmp_path / "test_imaging.zarr"

    imaging_zarr = imaging.save(format="zarr", folder=zarr_path)

    imaging_zarr_loaded = ZarrImaging(zarr_path)

    assert imaging_zarr_loaded.sampling_frequency == imaging.sampling_frequency
    assert len(imaging_zarr_loaded.epochs) == len(imaging.epochs)
    assert imaging_zarr.get_num_epochs() == imaging.get_num_epochs()
    assert imaging_zarr_loaded.get_num_epochs() == imaging.get_num_epochs()
    assert imaging_zarr.shape == imaging.shape
    assert imaging_zarr_loaded.shape == imaging.shape

    for epoch_index in range(imaging.get_num_epochs()):
        np.testing.assert_array_equal(
            imaging_zarr.get_series(epoch_index=epoch_index), imaging.get_series(epoch_index=epoch_index)
        )
        np.testing.assert_array_equal(
            imaging_zarr_loaded.get_series(epoch_index=epoch_index), imaging.get_series(epoch_index=epoch_index)
        )

    # load_compression_ratio
    imaging_with_cr = ZarrImaging(zarr_path, load_compression_ratio=True)
    assert "compression_ratio" in imaging_with_cr.get_annotation_keys()


def test_zarr_writing_with_t_starts(imaging, tmp_path):
    t_starts = []
    for epoch_index in range(imaging.get_num_segments()):
        t_start = epoch_index * 10
        imaging.shift_times(t_start, segment_index=epoch_index)
        t_starts.append(t_start)

    zarr_path = tmp_path / "test_imaging_t_starts.zarr"
    imaging_zarr = imaging.save(format="zarr", folder=zarr_path)

    for epoch_index in range(imaging_zarr.get_num_segments()):
        assert imaging_zarr.get_start_time(epoch_index) == t_starts[epoch_index]


def test_zarr_writing_with_times(imaging, tmp_path):
    times_list = []
    for epoch_index in range(imaging.get_num_segments()):
        times = imaging.get_times(epoch_index) + epoch_index * 10 + 20
        imaging.set_times(times, segment_index=epoch_index)
        times_list.append(times)

    zarr_path = tmp_path / "test_imaging_times.zarr"
    imaging_zarr = imaging.save(format="zarr", folder=zarr_path)

    for epoch_index in range(imaging_zarr.get_num_segments()):
        np.testing.assert_array_equal(imaging_zarr.get_times(epoch_index), times_list[epoch_index])
