"""Tests for the jnormcorre registration backend.

The end-to-end round-trip tests need jnormcorre (and JAX) installed; they
``importorskip`` when it is not. The lazy-loader adapter test exercises the
``BaseImagingEpoch`` -> jnormcorre ``lazy_data_loader`` adapter without jnormcorre
by checking the abstract-interface methods directly via a stub base class.
"""

import numpy as np
import pytest

from photon_mosaic.core import Motion, NumpyImaging
from photon_mosaic.preprocessing.jnormcorre_registration import (
    JNormcorreMotion,
    JNormcorreRegistrationSettings,
    RegisterJNormcorreImaging,
    compute_motion_jnormcorre,
    register_jnormcorre,
)

# --------------------------------------------------------------------------- #
# Synthetic shifted data (mirrors the suite2p backend's fixtures)
# --------------------------------------------------------------------------- #


def _make_shifted_plane(height, width, dy, dx, value):
    base = np.zeros((height, width), dtype=np.float32)
    y0 = height // 2 - 1
    x0 = width // 2 - 1
    frame = np.zeros_like(base)
    y_start, x_start = y0 + dy, x0 + dx
    if y_start < 0 or x_start < 0 or y_start + 2 > height or x_start + 2 > width:
        raise ValueError("Shift moves block out of bounds")
    frame[y_start : y_start + 2, x_start : x_start + 2] = value
    return frame


def _make_shifted_video(num_frames, num_planes=1, height=16, width=16, shifts=None):
    if shifts is None:
        shifts = [[(0, 0)] * num_frames for _ in range(num_planes)]
    if shifts and isinstance(shifts[0], tuple):
        shifts = [list(shifts)]
    video = np.zeros((num_frames, height, width, num_planes), dtype=np.float32)
    for plane_idx in range(num_planes):
        for frame_idx, (dy, dx) in enumerate(shifts[plane_idx]):
            video[frame_idx, :, :, plane_idx] = _make_shifted_plane(height, width, dy, dx, value=plane_idx + 1)
    return video


def _make_shifted_imaging(num_frames, num_planes=1, height=16, width=16, shifts=None, sampling_frequency=10.0):
    if isinstance(num_frames, int):
        num_frames = (num_frames,)
    if (
        len(num_frames) == 1
        and shifts
        and isinstance(shifts[0], list)
        and shifts[0]
        and isinstance(shifts[0][0], tuple)
    ):
        shifts = [shifts]
    videos = []
    if shifts is None:
        shifts = [None] * len(num_frames)
    for epoch_idx, n_frames in enumerate(num_frames):
        videos.append(
            _make_shifted_video(n_frames, num_planes=num_planes, height=height, width=width, shifts=shifts[epoch_idx])
        )
    return NumpyImaging(imaging_series=videos, sampling_frequency=sampling_frequency)


# --------------------------------------------------------------------------- #
# Settings resolution (no jnormcorre needed)
# --------------------------------------------------------------------------- #


class TestSettingsResolution:
    def test_none_settings_uses_defaults(self):
        resolved = JNormcorreRegistrationSettings()
        assert resolved.frames_per_split == 1000
        assert resolved.pw_rigid is False

    def test_dict_settings_validated(self):
        resolved = JNormcorreRegistrationSettings.model_validate({"frames_per_split": 300, "pw_rigid": True})
        assert resolved.frames_per_split == 300
        assert resolved.pw_rigid is True


# --------------------------------------------------------------------------- #
# Lazy-loader adapter (no jnormcorre needed: stub the abstract base)
# --------------------------------------------------------------------------- #


class TestPlaneLoaderAdapter:
    """The adapter must hand jnormcorre the right plane in (frames, Ly, Lx) form."""

    def _patched_loader(self, monkeypatch, epoch, plane, n_frames, Ly, Lx):
        # Stub jnormcorre's abstract base so the adapter can be built without the
        # real package. _make_plane_loader imports lazy_data_loader at call time.
        import sys
        import types

        class _StubBase:  # minimal stand-in for jnormcorre.utils.lazy_array.lazy_data_loader
            pass

        mod_root = types.ModuleType("jnormcorre")
        mod_utils = types.ModuleType("jnormcorre.utils")
        mod_lazy = types.ModuleType("jnormcorre.utils.lazy_array")
        mod_lazy.lazy_data_loader = _StubBase
        monkeypatch.setitem(sys.modules, "jnormcorre", mod_root)
        monkeypatch.setitem(sys.modules, "jnormcorre.utils", mod_utils)
        monkeypatch.setitem(sys.modules, "jnormcorre.utils.lazy_array", mod_lazy)

        from photon_mosaic.preprocessing.jnormcorre_registration import _make_plane_loader

        return _make_plane_loader(epoch, plane, n_frames, Ly, Lx)

    def test_slice_int_list_return_correct_plane(self, monkeypatch):
        # Two planes with distinct fill values so we can tell them apart.
        imaging = _make_shifted_imaging(num_frames=6, num_planes=2, height=12, width=12)
        epoch = imaging.epochs[0]
        loader = self._patched_loader(monkeypatch, epoch, plane=1, n_frames=6, Ly=12, Lx=12)

        assert loader.shape == (6, 12, 12)
        assert loader.dtype == "float32"

        sl = loader._compute_at_indices(slice(0, 4))
        assert sl.shape == (4, 12, 12)
        # plane 1 has fill value 2 (value = plane_idx + 1)
        assert sl.max() == pytest.approx(2.0)

        single = loader._compute_at_indices(2)
        assert single.shape == (12, 12)

        picked = loader._compute_at_indices([0, 2, 5])
        assert picked.shape == (3, 12, 12)


# --------------------------------------------------------------------------- #
# End-to-end (needs jnormcorre + jax)
# --------------------------------------------------------------------------- #

jnormcorre = pytest.importorskip("jnormcorre")


class TestComputeMotionJNormcorre:
    def test_returns_motion_and_registers_frames_rigid(self):
        shifts = [[(0, 0), (0, 0), (1, 0), (1, 0), (0, 1), (0, 1)]]
        imaging = _make_shifted_imaging(num_frames=6, num_planes=1, height=24, width=24, shifts=shifts)

        motion = compute_motion_jnormcorre(
            imaging, settings=JNormcorreRegistrationSettings(pw_rigid=False, max_shifts=(4, 4))
        )
        assert isinstance(motion, JNormcorreMotion)
        assert isinstance(motion, Motion)
        assert motion.num_epochs == 1
        assert motion.displacements[0].shape == (6, 1, 2)

        registered = RegisterJNormcorreImaging(imaging, motion).epochs[0].get_series(0, 6)
        assert registered.shape == (6, 24, 24, 1)
        # Registration should pull every frame close to the (shared) template.
        np.testing.assert_allclose(registered, registered[0:1].repeat(6, axis=0), atol=1.0)

    def test_register_alias_and_multi_epoch_shapes(self):
        shifts = [
            [(0, 0), (0, 0), (1, 0), (1, 0), (0, 1)],
            [(0, 0), (0, 0), (-1, 0), (-1, 0)],
        ]
        imaging = _make_shifted_imaging(num_frames=(5, 4), num_planes=1, height=24, width=24, shifts=shifts)
        motion = compute_motion_jnormcorre(imaging, settings={"pw_rigid": False, "max_shifts": (4, 4)})

        assert register_jnormcorre is RegisterJNormcorreImaging
        assert motion.num_epochs == 2
        assert motion.displacements[0].shape == (5, 1, 2)
        assert motion.displacements[1].shape == (4, 1, 2)

    def test_pw_rigid_runs_and_stores_patch_shifts(self):
        shifts = [[(0, 0), (0, 0), (1, 0), (1, 0), (0, 1), (0, 1)]]
        imaging = _make_shifted_imaging(num_frames=6, num_planes=1, height=64, width=64, shifts=shifts)
        settings = JNormcorreRegistrationSettings(pw_rigid=True, max_shifts=(4, 4), strides=(32, 32), overlaps=(16, 16))
        motion = compute_motion_jnormcorre(imaging, settings=settings)

        assert motion.pw_rigid is True
        assert motion.displacements[0].shape == (6, 1, 2)
        registered = RegisterJNormcorreImaging(imaging, motion).epochs[0].get_series(0, 6)
        assert registered.shape == (6, 64, 64, 1)

    def test_epoch_count_mismatch_raises(self):
        shifts = [[(0, 0)] * 4]
        imaging = _make_shifted_imaging(num_frames=4, num_planes=1, height=24, width=24, shifts=shifts)
        motion = compute_motion_jnormcorre(imaging, settings={"pw_rigid": False})

        smaller = _make_shifted_imaging(
            num_frames=(4, 4), num_planes=1, height=24, width=24, shifts=[shifts[0], shifts[0]]
        )
        with pytest.raises(ValueError):
            RegisterJNormcorreImaging(smaller, motion)
