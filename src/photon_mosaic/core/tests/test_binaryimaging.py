import os
from pathlib import Path

import numpy as np
import pytest

from photon_mosaic.core.binaryimaging import BinaryFolderImaging, BinaryImaging
from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.testingtools import assert_imaging_equal


def _write_binary_file(path: Path, array: np.ndarray, file_offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        if file_offset:
            f.write(b"\x00" * file_offset)
        f.write(array.tobytes(order="C"))


def test_binaryimaging_single_epoch_single_plane_roundtrip(tmp_path: Path):
    n_frames, h, w = 11, 3, 4
    dtype = np.uint16
    data = np.arange(n_frames * h * w, dtype=dtype).reshape(n_frames, h, w, 1)

    fpath = tmp_path / "video.dat"
    _write_binary_file(fpath, data)

    imaging = BinaryImaging(
        file_paths=str(fpath),
        sampling_frequency=10.0,
        shape=(h, w, 1),
        dtype=dtype,
    )

    assert imaging.get_num_epochs() == 1
    assert imaging.get_num_frames() == n_frames
    assert tuple(imaging.shape) == (h, w, 1)
    assert imaging.is_binary_compatible()
    binary_desc = imaging.get_binary_description()
    assert binary_desc is not None

    out = imaging.get_series(start_frame=0, end_frame=5)
    assert out.shape == (5, h, w, 1)
    np.testing.assert_array_equal(out, data[:5])

    out2 = imaging.get_series(start_frame=None, end_frame=None)
    assert out2.shape == (n_frames, h, w, 1)
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
        shape=(h, w, p),
        dtype=dtype,
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
    data1 = np.arange(6 * h * w, dtype=dtype).reshape(6, h, w) + 10

    f0 = tmp_path / "seg0.dat"
    f1 = tmp_path / "seg1.dat"
    _write_binary_file(f0, data0)
    _write_binary_file(f1, data1)

    t_starts = [0.2, 0.5]  # seconds

    imaging = BinaryImaging(
        file_paths=[str(f0), str(f1)],
        sampling_frequency=10.0,
        shape=(h, w, 1),
        dtype=dtype,
        t_starts=t_starts,
    )

    assert imaging.get_num_epochs() == 2

    time_vector_0 = imaging.get_times(segment_index=0)
    expected_time_0 = np.arange(4) / 10.0 + t_starts[0]
    np.testing.assert_allclose(time_vector_0, expected_time_0)

    time_vector_1 = imaging.get_times(segment_index=1)
    expected_time_1 = np.arange(6) / 10.0 + t_starts[1]
    np.testing.assert_allclose(time_vector_1, expected_time_1)


def test_binaryimaging_multiple_epochs(tmp_path: Path):
    h, w = 3, 3
    dtype = np.uint8

    data0 = np.arange(5 * h * w, dtype=dtype).reshape(5, h, w, 1)
    data1 = np.arange(8 * h * w, dtype=dtype).reshape(8, h, w, 1) + 100

    f0 = tmp_path / "seg0.dat"
    f1 = tmp_path / "seg1.dat"
    _write_binary_file(f0, data0)
    _write_binary_file(f1, data1)

    imaging = BinaryImaging(
        file_paths=[str(f0), str(f1)],
        sampling_frequency=5.0,
        shape=(h, w, 1),
        dtype=dtype,
    )

    assert imaging.get_num_epochs() == 2
    assert imaging.get_num_frames(epoch_index=0) == 5
    assert imaging.get_num_frames(epoch_index=1) == 8

    s0 = imaging.get_series(0, 5, epoch_index=0)
    s1 = imaging.get_series(0, 8, epoch_index=1)
    np.testing.assert_array_equal(s0, data0)
    np.testing.assert_array_equal(s1, data1)


def test_binaryimaging_file_offset(tmp_path: Path):
    n_frames, h, w = 4, 2, 5
    dtype = np.float32
    file_offset = 64

    data = np.linspace(0, 1, n_frames * h * w, dtype=dtype).reshape(n_frames, h, w, 1)
    fpath = tmp_path / "offset.dat"
    _write_binary_file(fpath, data, file_offset=file_offset)

    imaging = BinaryImaging(
        file_paths=str(fpath),
        sampling_frequency=30.0,
        shape=(h, w, 1),
        dtype=dtype,
        file_offset=file_offset,
    )

    out = imaging.get_series(0, n_frames)
    np.testing.assert_allclose(out, data)


def test_baseimaging_save_binary_with_multiprocessing(tmp_path: Path):
    # Create a small source BinaryImaging (acts as a generic BaseImaging instance for save()).
    n_frames, h, w = 23, 4, 6
    dtype = np.int16
    data = np.arange(n_frames * h * w, dtype=dtype).reshape(n_frames, h, w, 1)

    src_path = tmp_path / "src.dat"
    _write_binary_file(src_path, data)

    imaging = BinaryImaging(
        file_paths=str(src_path),
        sampling_frequency=10.0,
        shape=(h, w, 1),
        dtype=dtype,
    )

    out_folder = tmp_path / "saved_binary"
    n_jobs = 2 if (os.cpu_count() or 1) > 1 else 1

    # Exercise BaseImaging.save() path with multiprocessing-enabled job kwargs.
    _ = imaging.save(
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


def test_binaryimaging_per_plane_files_roundtrip(tmp_path: Path):
    n_frames, h, w, p = 9, 3, 4, 3
    dtype = np.int16
    rng = np.random.default_rng(0)
    data = rng.integers(-10000, 10000, size=(n_frames, h, w, p)).astype(dtype)

    plane_files = []
    for plane_idx in range(p):
        fp = tmp_path / f"plane{plane_idx}.bin"
        _write_binary_file(fp, np.ascontiguousarray(data[..., plane_idx]))
        plane_files.append(str(fp))

    imaging = BinaryImaging(
        file_paths=[plane_files],
        sampling_frequency=15.0,
        shape=(h, w, p),
        dtype=dtype,
    )

    assert imaging.get_num_epochs() == 1
    assert imaging.get_num_frames() == n_frames
    out = imaging.get_series(0, n_frames)
    assert out.shape == (n_frames, h, w, p)
    np.testing.assert_array_equal(out, data)

    out_plane = imaging.get_series(2, 6, plane_ids=[2, 0])
    assert out_plane.shape == (4, h, w, 2)
    np.testing.assert_array_equal(out_plane[..., 0], data[2:6, ..., 2])
    np.testing.assert_array_equal(out_plane[..., 1], data[2:6, ..., 0])

    # binary description should reflect the per-plane layout
    desc = imaging.get_binary_description()
    assert isinstance(desc["file_paths"][0], list)
    assert len(desc["file_paths"][0]) == p


def test_binaryimaging_per_plane_file_count_mismatch_raises(tmp_path: Path):
    h, w, p = 2, 2, 2
    dtype = np.int16
    data = np.zeros((4, h, w), dtype=dtype)

    f0 = tmp_path / "plane0.bin"
    _write_binary_file(f0, data)
    # Only one plane file supplied but shape declares 2 planes
    with pytest.raises(ValueError):
        _ = BinaryImaging(
            file_paths=[[str(f0)]],
            sampling_frequency=10.0,
            shape=(h, w, p),
            dtype=dtype,
        )


def test_binaryimaging_per_plane_inconsistent_sample_counts_raises(tmp_path: Path):
    h, w, p = 2, 2, 2
    dtype = np.int16

    f0 = tmp_path / "plane0.bin"
    f1 = tmp_path / "plane1.bin"
    _write_binary_file(f0, np.zeros((5, h, w), dtype=dtype))
    _write_binary_file(f1, np.zeros((4, h, w), dtype=dtype))

    with pytest.raises(ValueError):
        _ = BinaryImaging(
            file_paths=[[str(f0), str(f1)]],
            sampling_frequency=10.0,
            shape=(h, w, p),
            dtype=dtype,
        )


def test_base_imaging_multi_epoch_multiplane_save(tmp_path: Path):
    n = (15, 30)
    h, w, p = 5, 5, 3
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
    imaging_saved = imaging.save(
        format="binary",
        folder=str(out_folder),
        n_jobs=1,
        chunk_size=5,
        progress_bar=False,
    )

    imaging_loaded = BinaryFolderImaging(out_folder)
    assert_imaging_equal(imaging, imaging_saved)
    assert_imaging_equal(imaging, imaging_loaded)
