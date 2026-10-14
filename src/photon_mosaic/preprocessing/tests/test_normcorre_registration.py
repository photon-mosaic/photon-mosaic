"""Tests for the NoRMCorre backend: the kernels on known shifts, then compute -> register end to end."""

import pickle

import numpy as np
import pytest
import scipy.ndimage as ndi

from photon_mosaic.core import NumpyImaging, compute_motion, get_registration_class
from photon_mosaic.preprocessing import NormcorreMotion, RegisterNormcorreImagingEpoch, register_motion
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

    @pytest.mark.parametrize("apply", ["warp", "dft"])
    def test_applying_the_correction_undoes_the_shift(self, apply):
        base = _base()
        shift = np.array([[2.3, -1.7]])
        moved = _shifted(base, shift[0])[None]
        if apply == "warp":
            corrected = nc._apply_shift_iteration(moved, -shift, "copy")
        else:
            corrected = nc._apply_shifts_dft(moved, -shift, None, "copy")
        np.testing.assert_allclose(corrected[0][INNER, INNER], base[INNER, INNER], atol=1.0)

    @pytest.mark.parametrize("border_nan", [True, False, "min", "copy"])
    def test_border_fill(self, border_nan):
        frame = _base()[None]
        out = nc._apply_shift_iteration(frame, np.array([[2.0, -3.0]]), border_nan)[0]
        strip, edge = out[:2], out[2]
        if border_nan is True:
            assert np.isnan(strip).all() and np.isnan(out[:, -3:]).all()
        elif border_nan is False:
            # nothing is filled: the warp's reflected pixels stay
            assert np.isfinite(strip).all() and not (strip == edge).all()
        elif border_nan == "min":
            assert (strip == frame.min()).all()
        else:
            np.testing.assert_array_equal(strip, np.broadcast_to(edge, strip.shape))

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
}


class TestEndToEnd:
    @pytest.mark.parametrize("config", CONFIGS)
    def test_recovers_the_shifts_and_registers_the_movie(self, imaging, true_shifts, config):
        motion = compute_motion(imaging, method="normcorre", num_frames_split=20, **CONFIGS[config])
        assert isinstance(motion, NormcorreMotion)
        # displacements are the frames' motion (suite2p's convention), not CaImAn's correction;
        # the template sits wherever the sample's mean position was, so compare up to a constant
        moved = motion.displacements[0][:, 0]
        np.testing.assert_allclose(moved - moved.mean(0), true_shifts - true_shifts.mean(0), atol=0.25)

        registered = register_motion(imaging, motion, method="normcorre").get_series(0, 60)[..., 0]
        raw = imaging.get_series(0, 60)[..., 0]
        assert registered[:, INNER, INNER].std(0).mean() < 0.2 * raw[:, INNER, INNER].std(0).mean()
        if CONFIGS[config]["nonrigid"]:
            patches = motion.patch_shifts[0][0]
            assert patches.shape[:1] == (60,)
            # patch shifts follow the same convention: on a rigid movie they track the rigid motion
            patch_mean = patches.mean(axis=(1, 2))
            assert all(np.corrcoef(patch_mean[:, i], moved[:, i])[0, 1] > 0.9 for i in range(2))

    def test_multi_epoch_carries_the_template(self, true_shifts):
        movie = _movie(true_shifts)[..., None]
        imaging = NumpyImaging(imaging_series=[movie[:30], movie[30:]], sampling_frequency=10.0)
        motion = compute_motion(imaging, method="normcorre", nonrigid=False, num_frames_split=10)
        moved = np.concatenate([d[:, 0] for d in motion.displacements])
        np.testing.assert_allclose(moved - moved.mean(0), true_shifts - true_shifts.mean(0), atol=0.25)

    def test_motion_pickles_and_registered_view_serialises(self, imaging):
        from spikeinterface.core.base import BaseExtractor

        motion = compute_motion(imaging, method="normcorre", nonrigid=False)
        restored = pickle.loads(pickle.dumps(motion))
        np.testing.assert_array_equal(restored.displacements[0], motion.displacements[0])
        registered = register_motion(imaging, motion, method="normcorre")
        rebuilt = BaseExtractor.from_dict(registered.to_dict())
        assert isinstance(rebuilt.epochs[0], RegisterNormcorreImagingEpoch)
        np.testing.assert_array_equal(rebuilt.get_series(0, 3), registered.get_series(0, 3))

    def test_registration_class_is_registered(self):
        assert get_registration_class("normcorre") is RegisterNormcorreImagingEpoch

    def test_errors_where_caiman_errors(self, imaging):
        with pytest.raises(ValueError, match="bad-frame"):
            compute_motion(imaging, method="normcorre", badframes=np.zeros(60, bool))
        with pytest.raises(ValueError, match="shifts_opencv=True"):
            compute_motion(imaging, method="normcorre", gSig_filt=(3, 3), shifts_opencv=False)
