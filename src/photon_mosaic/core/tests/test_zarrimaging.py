import numpy as np
import pytest

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.testingtools import assert_imaging_equal
from photon_mosaic.core.zarrimaging import ZarrImaging


@pytest.fixture
def imaging():
    return generate_random_imaging(num_frames=[40, 20, 30], height=4, width=4, sampling_frequency=10.0, seed=3)


@pytest.fixture
def imaging_multi_plane():
    imaging = generate_random_imaging(
        num_frames=[100, 20, 30], height=4, width=4, sampling_frequency=10.0, seed=3, num_planes=3
    )
    imaging.set_property("depth", np.arange(100, 103))
    return imaging


def test_zarr_writing(imaging, tmp_path):
    zarr_path = tmp_path / "test_imaging.zarr"

    imaging_zarr = imaging.save(format="zarr", folder=zarr_path, n_jobs=2)

    imaging_zarr_loaded = ZarrImaging(zarr_path)

    assert_imaging_equal(imaging, imaging_zarr)
    assert_imaging_equal(imaging, imaging_zarr_loaded)

    # load_compression_ratio
    imaging_with_cr = ZarrImaging(zarr_path, load_compression_ratio=True)
    assert "compression_ratio" in imaging_with_cr.get_annotation_keys()


def test_zarr_multi_plane_writing(imaging_multi_plane, tmp_path):
    zarr_path = tmp_path / "test_imaging.zarr"

    imaging = imaging_multi_plane
    imaging_zarr = imaging.save(format="zarr", folder=zarr_path, n_jobs=2)

    imaging_zarr_loaded = ZarrImaging(zarr_path)

    assert_imaging_equal(imaging, imaging_zarr)
    assert_imaging_equal(imaging, imaging_zarr_loaded)

    # load_compression_ratio
    imaging_with_cr = ZarrImaging(zarr_path, load_compression_ratio=True)
    assert "compression_ratio" in imaging_with_cr.get_annotation_keys()


def test_zarr_writing_with_t_starts(imaging, tmp_path):
    t_starts = []
    for epoch_index in range(imaging.get_num_epochs()):
        t_start = epoch_index * 10
        imaging.shift_times(t_start, segment_index=epoch_index)
        t_starts.append(t_start)

    zarr_path = tmp_path / "test_imaging_t_starts.zarr"
    imaging_zarr = imaging.save(format="zarr", folder=zarr_path)

    for epoch_index in range(imaging_zarr.get_num_epochs()):
        assert imaging_zarr.get_start_time(epoch_index) == t_starts[epoch_index]


def test_zarr_writing_with_times(imaging, tmp_path):
    # Test all time vectors
    times_list = []
    for epoch_index in range(imaging.get_num_epochs()):
        times = imaging.get_times(epoch_index) + epoch_index * 10 + 20
        imaging.set_times(times, segment_index=epoch_index)
        times_list.append(times)

    zarr_path = tmp_path / "test_imaging_times.zarr"
    imaging_zarr = imaging.save(format="zarr", folder=zarr_path)

    for epoch_index in range(imaging_zarr.get_num_epochs()):
        np.testing.assert_array_equal(imaging_zarr.get_times(epoch_index), times_list[epoch_index])

    # Test partial time vectors
    imaging.reset_times()  # reset to default time vectors
    for epoch_index in range(imaging.get_num_epochs()):
        times = imaging.get_times(epoch_index) + epoch_index * 10 + 20
        if epoch_index % 2 == 0:
            imaging.set_times(times, segment_index=epoch_index)

    zarr_path = tmp_path / "test_imaging_partial_times.zarr"
    imaging_zarr = imaging.save(format="zarr", folder=zarr_path)

    for epoch_index in range(imaging_zarr.get_num_epochs()):
        assert imaging_zarr.has_time_vector(epoch_index) == imaging.has_time_vector(epoch_index)
        np.testing.assert_array_equal(imaging_zarr.get_times(epoch_index), imaging.get_times(epoch_index))


def test_zarr_writing_extra_chunks(imaging, tmp_path):
    zarr_path = tmp_path / "test_imaging_extra_chunks.zarr"
    imaging_zarr = imaging.save(format="zarr", folder=zarr_path, extra_chunks=(2, 2, 1))

    imaging_zarr_loaded = ZarrImaging(zarr_path)

    assert_imaging_equal(imaging, imaging_zarr)
    assert_imaging_equal(imaging, imaging_zarr_loaded)

    # check chunking
    for epoch_index in range(imaging_zarr.get_num_epochs()):
        chunk_shape = imaging_zarr._root[f"video_epoch{epoch_index}"].chunks
        assert chunk_shape[1:] == (2, 2, 1)


def test_zarr_writing_multi_plane_extra_chunks(imaging_multi_plane, tmp_path):
    zarr_path = tmp_path / "test_imaging_extra_chunks.zarr"
    imaging_zarr = imaging_multi_plane.save(format="zarr", folder=zarr_path, extra_chunks=(2, 2, 2))

    imaging_zarr_loaded = ZarrImaging(zarr_path)

    assert_imaging_equal(imaging_multi_plane, imaging_zarr)
    assert_imaging_equal(imaging_multi_plane, imaging_zarr_loaded)

    # check chunking
    for epoch_index in range(imaging_zarr.get_num_epochs()):
        chunk_shape = imaging_zarr._root[f"video_epoch{epoch_index}"].chunks
        assert chunk_shape[1:] == (2, 2, 2)
