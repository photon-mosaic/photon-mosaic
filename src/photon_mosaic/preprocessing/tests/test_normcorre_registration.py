"""Tests for the NoRMCorre backend: the kernels on known shifts, then compute -> register end to end."""

import numpy as np
import pytest
import scipy.ndimage as ndi

from photon_mosaic.core import NumpyImaging
from photon_mosaic.preprocessing import normcorre_registration as nc

SIZE = 96
INNER = slice(12, -12)


def _base(seed=0):
    rng = np.random.default_rng(seed)
    return (ndi.gaussian_filter(rng.random((SIZE, SIZE)), 2) * 100).astype(np.float64)


def _shifted(base, shift):
    """``base`` moved by ``shift`` (y, x) pixels, exactly, by a Fourier phase ramp (periodic)."""
    return np.real(np.fft.ifft2(ndi.fourier_shift(np.fft.fft2(base), shift)))


def _movie(shifts, seed=0):
    base = _base(seed)
    return np.stack([_shifted(base, s) for s in shifts]).astype(np.float32)


@pytest.fixture(scope="module")
def true_shifts():
    return np.random.default_rng(1).uniform(-3, 3, (60, 2))


@pytest.fixture(scope="module")
def imaging(true_shifts):
    return NumpyImaging(imaging_series=[_movie(true_shifts)[..., None]], sampling_frequency=10.0)


class TestKernels:
    def test_register_translation_recovers_a_subpixel_shift(self):
        base = _base()
        shift = np.array([2.3, -1.7])
        rigid, _ = nc._register_translation(_shifted(base, shift)[None], base, (6, 6), 10)
        np.testing.assert_allclose(rigid[0], shift, atol=0.1)

    def test_register_translation_respects_bounds(self):
        base = _base()
        rigid, _ = nc._register_translation(
            _shifted(base, (4, 4))[None], base, (6, 6), 10, shifts_lb=np.array([-1, -1]), shifts_ub=np.array([1, 1])
        )
        assert np.all(np.abs(rigid[0]) <= 1.1)

    def test_high_pass_removes_a_constant(self):
        flat = np.full((1, 40, 40), 7.0)
        np.testing.assert_allclose(nc._high_pass_filter_space(flat, (3, 3)), 0, atol=1e-4)

    def test_bin_median_of_bins_spaced_through_the_movie(self):
        frames = np.arange(20, dtype=float)[:, None, None] * np.ones((1, 2, 2))
        # 2 bins of 10 frames each, frames 0,2,4,... and 1,3,5,...: means 9 and 10, median 9.5
        np.testing.assert_allclose(nc._bin_median(frames), 9.5)

    def test_register_sample_aligns_the_frames(self, true_shifts):
        registered = nc._register_sample(_movie(true_shifts), (6, 6))
        assert registered[:, INNER, INNER].std(0).mean() < 0.2 * _movie(true_shifts)[:, INNER, INNER].std(0).mean()


CONFIGS = {
    "rigid_warp": dict(nonrigid=False),
    "rigid_dft": dict(nonrigid=False, shifts_opencv=False),
    "nonrigid_remap": dict(nonrigid=True, strides=(32, 32), overlaps=(16, 16)),
    "nonrigid_fft": dict(nonrigid=True, strides=(32, 32), overlaps=(16, 16), shifts_opencv=False),
}
