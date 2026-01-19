import numpy as np
import pytest

from photon_mosaic.core.numpyimaging import NumpyImaging, NumpyImagingSegment


def test_numpyimaging_rejects_invalid_imaging_series_type():
    with pytest.raises(ValueError, match="timeseries"):
        NumpyImaging(imaging_series="not an array", sampling_frequency=10.0)


def test_numpyimaging_accepts_single_3d_array_and_wraps_time_vector():
    video = np.zeros((5, 8, 9), dtype=np.float32)
    t = np.arange(video.shape[0], dtype=float) / 10.0

    im = NumpyImaging(imaging_series=video, sampling_frequency=10.0, time_vectors=t)

    assert isinstance(im._sampling_frequency, float)
    assert im._sampling_frequency == 10.0

    # time_vectors ndarray should be wrapped into a list for a single segment
    assert "time_vectors" in im._kwargs
    assert isinstance(im._kwargs["time_vectors"], list)
    assert len(im._kwargs["time_vectors"]) == 1
    np.testing.assert_allclose(im._kwargs["time_vectors"][0], t)

    assert len(im.segments) == 1


def test_numpyimaging_defaults_time_vectors_to_none_per_segment():
    v1 = np.zeros((3, 4, 5), dtype=np.uint8)
    v2 = np.zeros((7, 4, 5), dtype=np.uint8)

    im = NumpyImaging(imaging_series=[v1, v2], sampling_frequency=30.0, time_vectors=None)

    assert isinstance(im._kwargs["time_vectors"], list)
    assert im._kwargs["time_vectors"] == [None, None]

    assert len(im.segments) == 2


def test_numpyimaging_rejects_non_3d_or_4d_videos():
    bad2d = np.zeros((10, 10), dtype=np.float32)
    bad5d = np.zeros((2, 3, 4, 5, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="3D or 4D"):
        NumpyImaging(imaging_series=bad2d, sampling_frequency=1.0)

    with pytest.raises(ValueError, match="3D or 4D"):
        NumpyImaging(imaging_series=bad5d, sampling_frequency=1.0)


def test_numpyimaging_rejects_inconsistent_shapes_across_segments():
    v1 = np.zeros((5, 8, 9), dtype=np.float32)
    v2 = np.zeros((5, 8, 10), dtype=np.float32)  # different width

    with pytest.raises(ValueError, match="same image shape"):
        NumpyImaging(imaging_series=[v1, v2], sampling_frequency=10.0)


def test_numpyimaging_time_vectors_length_must_match_num_segments():
    v1 = np.zeros((5, 8, 9), dtype=np.float32)
    v2 = np.zeros((6, 8, 9), dtype=np.float32)

    tv = [np.arange(5), np.arange(6), np.arange(7)]
    with pytest.raises(AssertionError, match="Number of time vectors"):
        NumpyImaging(imaging_series=[v1, v2], sampling_frequency=10.0, time_vectors=tv)


def test_numpyimaging_plane_ids_generated_for_4d_video_and_validated():
    # 4D video: (frames, height, width, planes)
    video = np.zeros((4, 6, 7, 3), dtype=np.float32)

    im = NumpyImaging(imaging_series=video, sampling_frequency=20.0, plane_ids=None)
    assert im._kwargs["plane_ids"] == [0, 1, 2]

    with pytest.raises(AssertionError, match="plane_ids length must match num_planes"):
        NumpyImaging(imaging_series=video, sampling_frequency=20.0, plane_ids=[0, 1])


def test_numpyimagingsegment_get_num_samples():
    video = np.zeros((11, 2, 3), dtype=np.float32)
    seg = NumpyImagingSegment(video=video, sampling_frequency=5.0)
    assert seg.get_num_samples() == 11


def test_numpyimagingsegment_get_series_3d_basic_slicing():
    video = np.arange(5 * 3 * 4, dtype=np.int32).reshape(5, 3, 4)
    seg = NumpyImagingSegment(video=video, sampling_frequency=1.0)

    out_all = seg.get_series()
    np.testing.assert_array_equal(out_all, video)

    out_sub = seg.get_series(start_frame=1, end_frame=4)
    np.testing.assert_array_equal(out_sub, video[1:4, ...])

    # plane_indices should be ignored for 3D data
    out_pi = seg.get_series(start_frame=0, end_frame=2, plane_indices=[0])
    np.testing.assert_array_equal(out_pi, video[0:2, ...])


def test_numpyimagingsegment_get_series_4d_plane_selection():
    video = np.arange(6 * 2 * 3 * 4, dtype=np.int32).reshape(6, 2, 3, 4)
    seg = NumpyImagingSegment(video=video, sampling_frequency=1.0)

    out_all = seg.get_series()
    np.testing.assert_array_equal(out_all, video)

    out_subset = seg.get_series(start_frame=2, end_frame=5, plane_indices=[1, 3])
    np.testing.assert_array_equal(out_subset, video[2:5, :, :, [1, 3]])
    assert out_subset.shape == (3, 2, 3, 2)

    out_none_planes = seg.get_series(start_frame=1, end_frame=3, plane_indices=None)
    np.testing.assert_array_equal(out_none_planes, video[1:3, ...])
    assert out_none_planes.shape == (2, 2, 3, 4)
