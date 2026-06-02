import numpy as np
import pytest

from photon_mosaic.core import BaseImaging, BinaryImaging
from photon_mosaic.extractors import Suite2pImaging


def _make_suite2p_folder(tmp_path, height=8, width=10, fs=31.2, n_frames=7, include_binary=True):
    """Write a minimal suite2p output folder with ops.npy and (optionally) data.bin."""
    ops = {"Ly": height, "Lx": width, "Lyc": height, "Lxc": width, "fs": fs}
    np.save(tmp_path / "ops.npy", ops)

    movie = None
    if include_binary:
        rng = np.random.default_rng(42)
        movie = rng.integers(0, 1000, size=(n_frames, height, width)).astype(np.int16)
        movie.tofile(tmp_path / "data.bin")

    return ops, movie


class TestSuite2pImaging:
    def test_is_a_binary_imaging(self, tmp_path):
        _make_suite2p_folder(tmp_path)
        imaging = Suite2pImaging(tmp_path)
        assert isinstance(imaging, BinaryImaging)
        assert isinstance(imaging, BaseImaging)

    def test_loads_shape_and_sampling_frequency(self, tmp_path):
        ops, _ = _make_suite2p_folder(tmp_path, height=8, width=10, fs=31.2)
        imaging = Suite2pImaging(tmp_path)
        assert imaging.shape == (ops["Ly"], ops["Lx"], 1)
        assert imaging.sampling_frequency == ops["fs"]

    def test_reads_frames_from_binary(self, tmp_path):
        _make_suite2p_folder(tmp_path, height=8, width=10, n_frames=7, fs=31.2)
        imaging = Suite2pImaging(tmp_path)

        assert imaging.get_num_epochs() == 1
        assert imaging.get_num_samples(segment_index=0) == 7
        frames = imaging.get_series(0, 7)
        assert frames.shape == (7, 8, 10, 1)

    def test_raises_if_ops_missing(self, tmp_path):
        rng = np.random.default_rng(0)
        movie = rng.integers(0, 1000, size=(3, 8, 10)).astype(np.int16)
        movie.tofile(tmp_path / "data.bin")
        with pytest.raises(FileNotFoundError, match="ops.npy"):
            Suite2pImaging(tmp_path)

    def test_raises_if_binary_missing(self, tmp_path):
        _make_suite2p_folder(tmp_path, include_binary=False)
        with pytest.raises(FileNotFoundError, match="data.bin"):
            Suite2pImaging(tmp_path)

    def test_falls_back_to_cropped_dims(self, tmp_path):
        np.save(tmp_path / "ops.npy", {"Lyc": 8, "Lxc": 10, "fs": 30.0})
        rng = np.random.default_rng(0)
        rng.integers(0, 1000, size=(3, 8, 10)).astype(np.int16).tofile(tmp_path / "data.bin")
        imaging = Suite2pImaging(tmp_path)
        assert imaging.shape == (8, 10, 1)
