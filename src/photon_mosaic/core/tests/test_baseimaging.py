import numpy as np
import pytest

from photon_mosaic.core.baseimaging import BaseImaging
from photon_mosaic.core.generators import generate_random_imaging


def test_random_imaging_basic_properties():
    num_frames, h, w, sf = 10, 8, 9, 30.0
    imaging = generate_random_imaging(num_frames=num_frames, height=h, width=w, sampling_frequency=sf, seed=0)

    assert imaging.get_num_epochs() == 1
    assert imaging.get_num_frames() == num_frames
    assert imaging.get_num_samples() == num_frames
    assert tuple(imaging.shape) == (h, w, 1)
    assert imaging.get_shape() == (num_frames, h, w, 1)

    dt = imaging.get_dtype()
    assert np.dtype(dt).kind == "f"
    assert not imaging.is_binary_compatible()


def test_random_imaging_get_series_default_and_slicing():
    imaging = generate_random_imaging(num_frames=12, height=6, width=7, sampling_frequency=20.0, seed=1)

    full = imaging.get_series()
    assert full.shape == (12, 6, 7, 1)

    sl = imaging.get_series(start_frame=2, end_frame=5)
    assert sl.shape == (3, 6, 7, 1)
    np.testing.assert_allclose(sl, full[2:5])


def test_random_imaging_times_without_time_vector():
    num_frames, sf = 11, 25.0
    imaging = generate_random_imaging(num_frames=num_frames, height=5, width=6, sampling_frequency=sf, seed=2)

    times = imaging.get_times()
    assert times.shape == (num_frames,)
    assert times.dtype.kind == "f"
    np.testing.assert_allclose(times[0], 0.0)
    np.testing.assert_allclose(times[1], 1.0 / sf)

    # Roundtrip helpers
    t3 = imaging.sample_index_to_time(3)
    np.testing.assert_allclose(t3, 3.0 / sf)
    idx = imaging.time_to_sample_index(t3)
    assert int(idx) == 3


def test_random_imaging_shift_times_without_time_vector_updates_start_time():
    imaging = generate_random_imaging(num_frames=10, height=4, width=4, sampling_frequency=10.0, seed=3)

    np.testing.assert_allclose(imaging.get_start_time(), 0.0)
    imaging.shift_times(1.5)
    np.testing.assert_allclose(imaging.get_start_time(), 1.5)

    # No time_vector: sample_index_to_time should reflect new t_start
    np.testing.assert_allclose(imaging.sample_index_to_time(0), 1.5)
    np.testing.assert_allclose(imaging.sample_index_to_time(2), 1.5 + 2 / imaging.sampling_frequency)


def test_random_imaging_set_times_has_time_vector_shift_and_reset():
    num_frames, sf = 9, 30.0
    imaging = generate_random_imaging(num_frames=num_frames, height=5, width=5, sampling_frequency=sf, seed=4)

    base_times = np.arange(num_frames, dtype="float64") / sf + 0.1
    imaging.set_times(base_times.copy(), with_warning=False)
    assert imaging.has_time_vector() is True
    np.testing.assert_allclose(imaging.get_times(), base_times)

    imaging.shift_times(2.0)
    np.testing.assert_allclose(imaging.get_times(), base_times + 2.0)
    np.testing.assert_allclose(imaging.get_start_time(), base_times[0] + 2.0)

    imaging.reset_times()
    assert imaging.has_time_vector() is False
    np.testing.assert_allclose(imaging.get_start_time(), 0.0)
    # After reset, times should be reconstructed from sampling_frequency
    np.testing.assert_allclose(imaging.get_times()[1], 1.0 / sf)


def test_random_imaging_repr_contains_expected_fields():
    imaging = generate_random_imaging(num_frames=3, height=2, width=4, sampling_frequency=30.0, seed=5)
    s = repr(imaging)
    assert "rows x" in s
    assert "columns" in s
    assert "epochs" in s
    assert "dtype" in s
    assert "Hz" in s

    # set name
    imaging.name = "TestImaging"
    s = repr(imaging)
    assert "TestImaging" in s

    # test with float sampling frequency
    sf_float = 30.01920948
    imaging = generate_random_imaging(num_frames=3, height=2, width=4, sampling_frequency=sf_float, seed=5)
    s = repr(imaging)
    assert f"{sf_float:f} Hz" in s

    s_html = imaging._repr_html_()
    assert "rows x" in s_html
    assert "columns" in s_html
    assert "epochs" in s_html
    assert "dtype" in s_html
    assert "Hz" in s_html


def test_baseimaging_constructor_with_2d_dhape():
    shape = (50, 50)
    sampling_frequency = 15.0
    base_imaging = BaseImaging(sampling_frequency=sampling_frequency, shape=shape)

    assert base_imaging.shape == (50, 50, 1)


def test_baseimaging_multi_epoch_requires_epoch_index():
    sf = 10.0
    h, w = 3, 4
    num_frames = [10, 20]
    imaging = generate_random_imaging(num_frames=num_frames, height=h, width=w, sampling_frequency=sf, seed=5)

    assert imaging.get_num_epochs() == 2

    with pytest.raises(ValueError):
        _ = imaging.get_num_samples()

    with pytest.raises(ValueError):
        _ = imaging.get_series()

    with pytest.raises(ValueError):
        _ = imaging.get_shape()

    # Works when epoch_index is provided
    assert imaging.get_num_frames(epoch_index=0) == 10
    assert imaging.get_num_frames(epoch_index=1) == 20
    assert imaging.get_total_frames() == 30
    assert imaging.get_series(epoch_index=1).shape == (20, h, w, 1)
    assert imaging.get_shape(segment_index=0) == (10, h, w, 1)
    # Run repr_html
    str_html = imaging._repr_html_()
    assert "epochs" in str_html


def test_get_average_image_caches_and_recompute_replaces():
    imaging = generate_random_imaging(num_frames=25, height=8, width=6, sampling_frequency=30.0, seed=6)

    avg1 = imaging.get_average_image(num_chunks=2, chunk_size=5)
    assert avg1.shape == (8, 6, 1)

    avg2 = imaging.get_average_image(num_chunks=2, chunk_size=5)
    assert avg2 is avg1  # cached

    avg3 = imaging.get_average_image(num_chunks=2, chunk_size=5, recompute=True)
    assert avg3.shape == (8, 6, 1)
    assert avg3 is not avg1

# -------------------------
# Multi-plane specific tests
# -------------------------


def test_multiplane_basic_shape_and_plane_ids():
    n, h, w, p = 7, 5, 6, 3
    sf = 20.0
    imaging = generate_random_imaging(
        num_frames=n,
        height=h,
        width=w,
        sampling_frequency=sf,
        num_planes=p,
        seed=0,
    )

    assert imaging.get_num_epochs() == 1
    assert imaging.get_num_planes() == p and imaging.num_planes == p
    assert np.array_equal(imaging.plane_ids, list(range(p)))
    assert imaging.get_shape() == (n, h, w, p)
    assert imaging.get_series().shape == (n, h, w, p)
    # slice get_Series by plane_ids
    assert imaging.get_series(plane_ids=[1, 2]).shape == (n, h, w, 2)


def test_multiplane_average_image_shape_and_caching():
    n, h, w, p = 30, 8, 6, 2
    sf = 30.0
    imaging = generate_random_imaging(
        num_frames=n,
        height=h,
        width=w,
        num_planes=p,
        sampling_frequency=sf,
        seed=42,
    )

    avg1 = imaging.get_average_image(num_chunks=3, chunk_size=4)
    assert avg1.shape == (h, w, p)

    avg2 = imaging.get_average_image(num_chunks=3, chunk_size=4)
    assert avg2 is avg1  # cached

    avg3 = imaging.get_average_image(num_chunks=3, chunk_size=4, recompute=True)
    assert avg3.shape == (h, w, p)
    assert avg3 is not avg1


def test_generate_random_imaging_multiplane_has_expected_plane_dimension_and_selection():
    n, h, w, p = 12, 6, 7, 3
    imaging = generate_random_imaging(num_frames=n, height=h, width=w, sampling_frequency=20.0, num_planes=p, seed=123)

    assert imaging.get_num_planes() == p
    assert imaging.get_shape() == (n, h, w, p)

    full = imaging.get_series()
    assert full.shape == (n, h, w, p)

    sel = imaging.get_series(plane_ids=[1])
    assert sel.shape == (n, h, w, 1)
    np.all(sel[..., 0] == full[..., 1])
