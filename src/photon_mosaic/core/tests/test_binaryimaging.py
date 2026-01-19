import pytest
import os
from pathlib import Path
import numpy as np

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.binaryimaging import BinaryFolderImaging, BinaryImaging


def _write_binary_file(path: Path, array: np.ndarray, file_offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        if file_offset:
            f.write(b"\x00" * file_offset)
        f.write(array.tobytes(order="C"))


def test_binaryimaging_single_segment_single_plane_roundtrip(tmp_path: Path):
    n_frames, h, w = 11, 3, 4
    dtype = np.uint16
    data = (np.arange(n_frames * h * w, dtype=dtype).reshape(n_frames, h, w))

    fpath = tmp_path / "video.dat"
    _write_binary_file(fpath, data)

    imaging = BinaryImaging(
        file_paths=str(fpath),
        sampling_frequency=10.0,
        image_shape=(h, w),
        dtype=dtype,
        num_planes=1,
    )

    assert imaging.get_num_segments() == 1
    assert imaging.get_num_frames() == n_frames
    assert tuple(imaging.image_shape) == (h, w)
    assert imaging.is_binary_compatible()
    binary_desc = imaging.get_binary_description()
    assert binary_desc is not None

    out = imaging.get_series(start_frame=0, end_frame=5)
    assert out.shape == (5, h, w)
    np.testing.assert_array_equal(out, data[:5])

    out2 = imaging.get_series(start_frame=None, end_frame=None)
    assert out2.shape == (n_frames, h, w)
    np.testing.assert_array_equal(out2, data)

    # test deleting
    del imaging


def test_binaryimaging_multi_plane_and_plane_selection(tmp_path: Path):
    n_frames, h, w, p = 7, 2, 3, 4
    dtype = np.int16
    data = np.arange(n_frames * h * w * p, dtype=dtype).reshape(n_frames, h, w, p)

    fpath = tmp_path / "video4d.dat"
    _write_binary_file(fpath, data)

    imaging = BinaryImaging(
        file_paths=str(fpath),
        sampling_frequency=20.0,
        image_shape=(h, w),
        dtype=dtype,
        num_planes=p,
        plane_ids=list(range(p)),
    )

    out_all = imaging.get_series(0, n_frames)
    assert out_all.shape == (n_frames, h, w, p)
    np.testing.assert_array_equal(out_all, data)

    out_plane_1 = imaging.get_series(1, 6, plane_ids=[1])
    assert out_plane_1.shape == (5, h, w, 1)
    np.testing.assert_array_equal(out_plane_1[..., 0], data[1:6, ..., 1])

    out_planes_3_0 = imaging.get_series(0, 3, plane_ids=[3, 0])
    assert out_planes_3_0.shape == (3, h, w, 2)
    np.testing.assert_array_equal(out_planes_3_0[..., 0], data[:3, ..., 3])
    np.testing.assert_array_equal(out_planes_3_0[..., 1], data[:3, ..., 0])


def test_binaryimaging_with_tstarts(tmp_path: Path):
    h, w = 2, 2
    dtype = np.float32

    data0 = np.arange(4 * h * w, dtype=dtype).reshape(4, h, w)
    data1 = (np.arange(6 * h * w, dtype=dtype).reshape(6, h, w) + 10)

    f0 = tmp_path / "seg0.dat"
    f1 = tmp_path / "seg1.dat"
    _write_binary_file(f0, data0)
    _write_binary_file(f1, data1)

    t_starts = [0.2, 0.5]  # seconds

    imaging = BinaryImaging(
        file_paths=[str(f0), str(f1)],
        sampling_frequency=10.0,
        image_shape=(h, w),
        dtype=dtype,
        t_starts=t_starts,
        num_planes=1,
    )

    assert imaging.get_num_segments() == 2

    time_vector_0 = imaging.get_times(segment_index=0)
    expected_time_0 = np.arange(4) / 10.0 + t_starts[0]
    np.testing.assert_allclose(time_vector_0, expected_time_0)

    time_vector_1 = imaging.get_times(segment_index=1)
    expected_time_1 = np.arange(6) / 10.0 + t_starts[1]
    np.testing.assert_allclose(time_vector_1, expected_time_1)


def test_binaryimaging_multiple_segments(tmp_path: Path):
    h, w = 3, 3
    dtype = np.uint8

    data0 = np.arange(5 * h * w, dtype=dtype).reshape(5, h, w)
    data1 = (np.arange(8 * h * w, dtype=dtype).reshape(8, h, w) + 100)

    f0 = tmp_path / "seg0.dat"
    f1 = tmp_path / "seg1.dat"
    _write_binary_file(f0, data0)
    _write_binary_file(f1, data1)

    imaging = BinaryImaging(
        file_paths=[str(f0), str(f1)],
        sampling_frequency=5.0,
        image_shape=(h, w),
        dtype=dtype,
        num_planes=1,
    )

    assert imaging.get_num_segments() == 2
    assert imaging.get_num_samples(segment_index=0) == 5
    assert imaging.get_num_samples(segment_index=1) == 8

    s0 = imaging.get_series(0, 5, segment_index=0)
    s1 = imaging.get_series(0, 8, segment_index=1)
    np.testing.assert_array_equal(s0, data0)
    np.testing.assert_array_equal(s1, data1)


def test_binaryimaging_file_offset(tmp_path: Path):
    n_frames, h, w = 4, 2, 5
    dtype = np.float32
    file_offset = 64

    data = np.linspace(0, 1, n_frames * h * w, dtype=dtype).reshape(n_frames, h, w)
    fpath = tmp_path / "offset.dat"
    _write_binary_file(fpath, data, file_offset=file_offset)

    imaging = BinaryImaging(
        file_paths=str(fpath),
        sampling_frequency=30.0,
        image_shape=(h, w),
        dtype=dtype,
        num_planes=1,
        file_offset=file_offset,
    )

    out = imaging.get_series(0, n_frames)
    np.testing.assert_allclose(out, data)


def test_baseimaging_save_binary_with_multiprocessing(tmp_path: Path):
    # Create a small source BinaryImaging (acts as a generic BaseImaging instance for save()).
    n_frames, h, w = 23, 4, 6
    dtype = np.int16
    data = np.arange(n_frames * h * w, dtype=dtype).reshape(n_frames, h, w)

    src_path = tmp_path / "src.dat"
    _write_binary_file(src_path, data)

    imaging = BinaryImaging(
        file_paths=str(src_path),
        sampling_frequency=10.0,
        image_shape=(h, w),
        dtype=dtype,
        num_planes=1,
    )

    out_folder = tmp_path / "saved_binary"
    n_jobs = 2 if (os.cpu_count() or 1) > 1 else 1

    # Exercise BaseImaging.save() path with multiprocessing-enabled job kwargs.
    saved = imaging.save(
        format="binary",
        folder=str(out_folder),
        n_jobs=n_jobs,
        chunk_size=7,
        progress_bar=False,
    )

    # Ensure a binary folder was created and is loadable
    assert (out_folder / "binary.json").exists()

    loaded = BinaryFolderImaging(out_folder)

    got = loaded.get_series(0, n_frames)
    assert got.shape == data.shape
    np.testing.assert_array_equal(got, data)
    assert loaded.is_binary_compatible()
    binary_desc = loaded.get_binary_description()
    assert binary_desc is not None

    # test failure with incompatible JSON files
    with pytest.raises(ValueError):
        bad_folder = tmp_path / "bad_binary"
        bad_folder.mkdir()
        (bad_folder / "binary.json").write_text('{"class": "SomeFunkyImagingClass"}')
        BinaryFolderImaging(bad_folder)


def test_base_imaging_multi_segment_multiplane_save(tmp_path: Path):
    n, h, w, p = (15, 30), 5, 5, 3
    sf = 20.0
    imaging = generate_random_imaging(
        num_frames=n,
        height=h,
        width=w,
        sampling_frequency=sf,
        num_planes=p,
        seed=0,
    )

    out_folder = tmp_path / "multiplane_binary"
    saved = imaging.save(
        format="binary",
        folder=str(out_folder),
        n_jobs=1,
        chunk_size=5,
        progress_bar=False,
    )

    loaded = BinaryFolderImaging(out_folder)
    assert loaded.get_num_planes() == p

    assert imaging.get_num_segments() == loaded.get_num_segments()
    for segment_index in range(imaging.get_num_segments()):
        assert imaging.get_num_samples(segment_index=segment_index) == loaded.get_num_samples(
            segment_index=segment_index
        )
        full = imaging.get_series(segment_index=segment_index)
        got = loaded.get_series(segment_index=segment_index)
        assert got.shape == full.shape
        np.testing.assert_array_equal(got, full)