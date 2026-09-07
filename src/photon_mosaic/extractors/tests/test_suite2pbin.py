"""Tests for the Suite2pImaging extractor.

These tests build synthetic suite2p output trees on disk (per-plane ``data.bin``
plus ``ops.npy``) so the extractor can be exercised end-to-end without a real
suite2p run. The on-disk layout matches what suite2p produces in v0.14+.

Following issue #77, ``Suite2pImaging`` loads a single plane; multi-plane
volumes are assembled with ``stack_planes`` (used internally by
``read_suite2p_full``).
"""

from pathlib import Path

import numpy as np
import pytest

from photon_mosaic.core.split import split_epoch_at_frames
from photon_mosaic.core.stack import stack_planes
from photon_mosaic.extractors.suite2pbin import (
    Suite2pImaging,
    read_suite2p_binary,
    read_suite2p_full,
)

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

    # A single plane still maps to a unique binary on disk.
    assert imaging.is_binary_compatible()
    desc = imaging.get_binary_description()
    assert desc is not None


def test_suite2p_imaging_accepts_plane_directory_directly(tmp_path: Path):
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=6, Ly=3, Lx=4, nplanes=2, seed=5)

    # A plane directory (contains ops.npy) is accepted directly...
    plane1 = Suite2pImaging(root / "plane1")
    assert tuple(plane1.shape) == (3, 4, 1)
    # ...while a multi-plane root is rejected with a pointer to read_suite2p_full.
    with pytest.raises(ValueError, match="single plane"):
        _ = Suite2pImaging(root)


def test_stack_planes_of_single_plane_suite2p_objects(tmp_path: Path):
    """Use case 1 of issue #77: load each plane, then stitch with stack_planes."""
    root = tmp_path / "run0"
    n_frames, Ly, Lx = 6, 4, 5
    written = _write_suite2p_run(root, n_frames=n_frames, Ly=Ly, Lx=Lx, nplanes=2, seed=3)

    plane0 = Suite2pImaging(root / "plane0")
    plane1 = Suite2pImaging(root / "plane1")
    assert tuple(plane0.shape) == (Ly, Lx, 1)

    volume = stack_planes(plane0, plane1)
    assert tuple(volume.shape) == (Ly, Lx, 2)
    assert volume.is_registered is True

    out = volume.get_series(0, n_frames)
    assert out.shape == (n_frames, Ly, Lx, 2)
    np.testing.assert_array_equal(out[..., 0], written[0][1])
    np.testing.assert_array_equal(out[..., 1], written[1][1])


def test_read_suite2p_stitches_planes(tmp_path: Path):
    root = tmp_path / "run0"
    n_frames, Ly, Lx, nplanes = 8, 4, 6, 3
    written = _write_suite2p_run(root, n_frames=n_frames, Ly=Ly, Lx=Lx, nplanes=nplanes, seed=7)

    imaging = read_suite2p_full(root)  # single channel -> one stitched volume
    assert tuple(imaging.shape) == (Ly, Lx, nplanes)
    assert imaging.is_registered is True

    out = imaging.get_series(0, n_frames)
    assert out.shape == (n_frames, Ly, Lx, nplanes)
    for p in range(nplanes):
        np.testing.assert_array_equal(out[..., p], written[p][1])


def test_read_suite2p_two_channels_returns_pair(tmp_path: Path):
    root = tmp_path / "run0"
    n_frames, Ly, Lx = 5, 3, 4
    written = _write_suite2p_run(root, n_frames=n_frames, Ly=Ly, Lx=Lx, nplanes=1, nchannels=2, seed=1)

    result = read_suite2p_full(root)
    assert isinstance(result, tuple)
    chan1, chan2 = result
    assert isinstance(chan1, Suite2pImaging) and isinstance(chan2, Suite2pImaging)
    assert chan1.chan == 1 and chan2.chan == 2
    assert tuple(chan1.shape) == (Ly, Lx, 1)
    assert tuple(chan2.shape) == (Ly, Lx, 1)

    np.testing.assert_array_equal(chan1.get_series(0, n_frames)[..., 0], written[0][1])
    np.testing.assert_array_equal(chan2.get_series(0, n_frames)[..., 0], written[0][2])


def test_read_suite2p_single_plane_single_channel_returns_one_object(tmp_path: Path):
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=6, Ly=3, Lx=3, nplanes=1, nchannels=1)
    obj = read_suite2p_full(root)
    assert isinstance(obj, Suite2pImaging)


def test_read_suite2p_binary_alias_is_the_class(tmp_path: Path):
    # Mirrors read_suite2p_rois = Suite2pRois: read_suite2p_binary IS the class.
    assert read_suite2p_binary is Suite2pImaging
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=4, Ly=3, Lx=3, nplanes=1)
    assert isinstance(read_suite2p_binary(root), Suite2pImaging)


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


def test_into_epochs_splits_at_file_boundaries(tmp_path: Path):
    root = tmp_path / "run0"
    n_frames, Ly, Lx = 10, 3, 4
    fpf = [3, 4, 3]
    written = _write_suite2p_run(root, n_frames=n_frames, Ly=Ly, Lx=Lx, nplanes=1, frames_per_file=fpf)

    imaging = Suite2pImaging(root)
    split = imaging.into_epochs()

    assert split.get_num_epochs() == len(fpf)
    assert [split.get_num_samples(segment_index=i) for i in range(len(fpf))] == fpf
    # registered flag carried over from the suite2p parent
    assert split.is_registered is True

    # Concatenating the per-file epochs back recovers the original movie
    recovered = np.concatenate([split.get_series(epoch_index=i)[..., 0] for i in range(len(fpf))], axis=0)
    np.testing.assert_array_equal(recovered, written[0][1])


def test_into_epochs_requires_frames_per_file(tmp_path: Path):
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=6, Ly=3, Lx=3, nplanes=1)  # no frames_per_file
    imaging = Suite2pImaging(root)
    with pytest.raises(ValueError, match="frames_per_file"):
        _ = imaging.into_epochs()


def test_into_epochs_rejects_frames_per_file_mismatch(tmp_path: Path):
    root = tmp_path / "run0"
    # ops declares per-file counts that do not sum to the actual frame count
    _write_suite2p_run(root, n_frames=10, Ly=3, Lx=3, nplanes=1, frames_per_file=[3, 4])
    imaging = Suite2pImaging(root)
    with pytest.raises(ValueError, match="frames_per_file"):
        _ = imaging.into_epochs()


def test_read_suite2p_rejects_inconsistent_geometry_across_planes(tmp_path: Path):
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=5, Ly=3, Lx=3, nplanes=2)
    # Corrupt plane 1's ops.npy to claim a different Lx
    ops1 = np.load(root / "plane1" / "ops.npy", allow_pickle=True).item()
    ops1["Lx"] = 99
    np.save(root / "plane1" / "ops.npy", ops1, allow_pickle=True)

    # stack_planes (via read_suite2p_full) refuses to stitch mismatched planes.
    with pytest.raises(ValueError, match="frame shape"):
        _ = read_suite2p_full(root)


def test_suite2p_imaging_rejects_missing_channel(tmp_path: Path):
    root = tmp_path / "run0"
    _write_suite2p_run(root, n_frames=5, Ly=3, Lx=3, nplanes=1, nchannels=1)
    with pytest.raises(FileNotFoundError):
        _ = Suite2pImaging(root, chan=2)
