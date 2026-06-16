"""Tests for the Suite2pImaging extractor.

These tests build synthetic suite2p output trees on disk (per-plane ``data.bin``
plus ``ops.npy``) so the extractor can be exercised end-to-end without a real
suite2p run. The on-disk layout matches what suite2p produces in v0.14+.
"""

from pathlib import Path

import numpy as np
import pytest

from photon_mosaic.core.split import split_epoch_at_frames
from photon_mosaic.extractors.suite2p import Suite2pImaging, read_suite2p

SUITE2P_DTYPE = np.int16


def _write_suite2p_run(
    root: Path,
    *,
    n_frames: int,
    Ly: int,
    Lx: int,
    nplanes: int,
    nchannels: int = 1,
    fs: float = 30.0,
    frames_per_file: list[int] | None = None,
    seed: int = 0,
) -> dict[int, dict[int, np.ndarray]]:
    """Write a synthetic suite2p output tree and return the source data per (plane, chan).

    Returns ``{plane_index: {chan_index: array}}`` where each array has shape
    ``(n_frames, Ly, Lx)`` and dtype int16, matching what the extractor should
    read back.
    """
    rng = np.random.default_rng(seed)
    written: dict[int, dict[int, np.ndarray]] = {}
    for plane_idx in range(nplanes):
        plane_dir = root / f"plane{plane_idx}"
        plane_dir.mkdir(parents=True, exist_ok=True)

        per_chan: dict[int, np.ndarray] = {}
        for chan in range(1, nchannels + 1):
            data = rng.integers(-10000, 10000, size=(n_frames, Ly, Lx)).astype(SUITE2P_DTYPE)
            bin_name = "data.bin" if chan == 1 else "data_chan2.bin"
            with open(plane_dir / bin_name, "wb") as f:
                f.write(np.ascontiguousarray(data).tobytes(order="C"))
            per_chan[chan] = data
        written[plane_idx] = per_chan

        ops = {
            "Ly": Ly,
            "Lx": Lx,
            "nframes": n_frames,
            "fs": fs,
            "nplanes": nplanes,
            "nchannels": nchannels,
        }
        if frames_per_file is not None:
            ops["frames_per_file"] = np.asarray(frames_per_file, dtype=int)
        np.save(plane_dir / "ops.npy", ops, allow_pickle=True)

    return written


def test_suite2p_imaging_single_plane_single_run(tmp_path: Path):
    root = tmp_path / "run0"
    n_frames, Ly, Lx = 12, 5, 7
    written = _write_suite2p_run(root, n_frames=n_frames, Ly=Ly, Lx=Lx, nplanes=1, seed=42)

    imaging = Suite2pImaging(root)
    assert imaging.get_num_epochs() == 1
    assert imaging.get_num_frames() == n_frames
    assert tuple(imaging.shape) == (Ly, Lx, 1)
    assert imaging.sampling_frequency == 30.0
    assert imaging.chan == 1
    assert imaging.nchannels == 1
    # suite2p binaries are the registered movie
    assert imaging.is_registered is True

    out = imaging.get_series(0, n_frames)
    np.testing.assert_array_equal(out[..., 0], written[0][1])

    assert imaging.is_binary_compatible()
    desc = imaging.get_binary_description()
    assert desc is not None


def test_suite2p_imaging_multi_plane_stitches_planes(tmp_path: Path):
    root = tmp_path / "run0"
    n_frames, Ly, Lx, nplanes = 8, 4, 6, 3
    written = _write_suite2p_run(root, n_frames=n_frames, Ly=Ly, Lx=Lx, nplanes=nplanes, seed=7)

    imaging = Suite2pImaging(root)
    assert tuple(imaging.shape) == (Ly, Lx, nplanes)

    out = imaging.get_series(0, n_frames)
    assert out.shape == (n_frames, Ly, Lx, nplanes)
    for p in range(nplanes):
        np.testing.assert_array_equal(out[..., p], written[p][1])

    # Plane selection should pull only the requested planes
    out_sel = imaging.get_series(2, 6, plane_ids=[2, 0])
    assert out_sel.shape == (4, Ly, Lx, 2)
    np.testing.assert_array_equal(out_sel[..., 0], written[2][1][2:6])
    np.testing.assert_array_equal(out_sel[..., 1], written[0][1][2:6])


def test_suite2p_imaging_two_channels_returns_pair(tmp_path: Path):
    root = tmp_path / "run0"
    n_frames, Ly, Lx, nplanes = 5, 3, 4, 2
    written = _write_suite2p_run(root, n_frames=n_frames, Ly=Ly, Lx=Lx, nplanes=nplanes, nchannels=2, seed=1)

    result = read_suite2p(root)
    assert isinstance(result, tuple)
    chan1, chan2 = result
    assert isinstance(chan1, Suite2pImaging)
    assert isinstance(chan2, Suite2pImaging)
    assert chan1.chan == 1 and chan2.chan == 2
    assert tuple(chan1.shape) == (Ly, Lx, nplanes)
    assert tuple(chan2.shape) == (Ly, Lx, nplanes)

    out1 = chan1.get_series(0, n_frames)
    out2 = chan2.get_series(0, n_frames)
    for p in range(nplanes):
        np.testing.assert_array_equal(out1[..., p], written[p][1])
        np.testing.assert_array_equal(out2[..., p], written[p][2])


def test_read_suite2p_single_channel_returns_one_object(tmp_path: Path):
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=6, Ly=3, Lx=3, nplanes=1, nchannels=1)
    obj = read_suite2p(root)
    assert isinstance(obj, Suite2pImaging)


def test_suite2p_imaging_multiple_runs_become_epochs(tmp_path: Path):
    root_a = tmp_path / "session_a"
    root_b = tmp_path / "session_b"
    Ly, Lx, nplanes = 4, 5, 2
    wa = _write_suite2p_run(root_a, n_frames=7, Ly=Ly, Lx=Lx, nplanes=nplanes, seed=10)
    wb = _write_suite2p_run(root_b, n_frames=11, Ly=Ly, Lx=Lx, nplanes=nplanes, seed=11)

    imaging = Suite2pImaging([root_a, root_b])
    assert imaging.get_num_epochs() == 2
    assert imaging.get_num_frames(epoch_index=0) == 7
    assert imaging.get_num_frames(epoch_index=1) == 11

    out0 = imaging.get_series(0, 7, epoch_index=0)
    out1 = imaging.get_series(0, 11, epoch_index=1)
    for p in range(nplanes):
        np.testing.assert_array_equal(out0[..., p], wa[p][1])
        np.testing.assert_array_equal(out1[..., p], wb[p][1])


def test_suite2p_imaging_records_frames_per_file_metadata(tmp_path: Path):
    root = tmp_path / "run0"
    n_frames, Ly, Lx = 10, 3, 3
    fpf = [3, 4, 3]
    _write_suite2p_run(root, n_frames=n_frames, Ly=Ly, Lx=Lx, nplanes=1, frames_per_file=fpf)

    imaging = Suite2pImaging(root)
    assert imaging.frames_per_file_per_epoch == [fpf]

    # The metadata should let us subdivide the epoch into per-source-tiff sub-epochs
    boundaries = np.cumsum(fpf)[:-1].tolist()
    sub = split_epoch_at_frames(imaging, 0, boundaries)
    assert sub.get_num_epochs() == len(fpf)
    assert [sub.get_num_samples(segment_index=i) for i in range(len(fpf))] == fpf


def test_suite2p_imaging_rejects_inconsistent_geometry_across_planes(tmp_path: Path):
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=5, Ly=3, Lx=3, nplanes=2)
    # Corrupt plane 1's ops.npy to claim a different Lx
    ops1 = np.load(root / "plane1" / "ops.npy", allow_pickle=True).item()
    ops1["Lx"] = 99
    np.save(root / "plane1" / "ops.npy", ops1, allow_pickle=True)

    with pytest.raises(ValueError, match="disagrees"):
        _ = Suite2pImaging(root)


def test_suite2p_imaging_rejects_missing_channel(tmp_path: Path):
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=5, Ly=3, Lx=3, nplanes=1, nchannels=1)
    with pytest.raises(FileNotFoundError):
        _ = Suite2pImaging(root, chan=2)
