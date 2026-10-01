"""Tests for the backend-agnostic registration layer."""

from pathlib import Path

import numpy as np
import pytest

from photon_mosaic.core import Motion, generate_random_imaging, register_motion_class, register_registration_class
from photon_mosaic.preprocessing import RegisterImaging, RegisterImagingEpoch, RegistrationSettings, register_motion


class DoublingEpoch(RegisterImagingEpoch):
    """Stand-in backend epoch: 'applies' motion by doubling each plane."""

    def _correct_plane(self, plane_video, plane_index, start_frame, end_frame):
        return plane_video * 2


class DoublingMotion(Motion):
    method_name = "doubling_test_backend"


register_motion_class(DoublingMotion)
register_registration_class("doubling_test_backend", DoublingEpoch)


@pytest.fixture()
def imaging():
    return generate_random_imaging(num_frames=6, height=8, width=9, num_planes=1, sampling_frequency=10.0, seed=0)


class TestRegistrationSettings:
    def test_shared_defaults(self):
        settings = RegistrationSettings()
        assert settings.debug is False
        assert settings.tmp_dir == Path("/scratch")
        assert settings.batch_size == 500
        assert settings.device == "cpu"

    def test_values_can_be_overridden(self):
        settings = RegistrationSettings(batch_size=32, device="cuda")
        assert settings.batch_size == 32
        assert settings.device == "cuda"

    def test_env_vars_need_the_prefix(self, monkeypatch):
        monkeypatch.setenv("DEVICE", "cuda")
        monkeypatch.setenv("BATCH_SIZE", "7")
        assert RegistrationSettings().device == "cpu"
        assert RegistrationSettings().batch_size == 500

    def test_prefixed_env_vars_are_read(self, monkeypatch):
        monkeypatch.setenv("REGISTRATION_DEVICE", "mps")
        assert RegistrationSettings().device == "mps"

    def test_suite2p_keeps_its_own_env_prefix(self, monkeypatch):
        from photon_mosaic.preprocessing import Suite2pRegistrationSettings

        monkeypatch.setenv("REGISTRATION_DEVICE", "mps")
        monkeypatch.setenv("SUITE2P_REGISTRATION_DEVICE", "cuda")
        assert Suite2pRegistrationSettings().device == "cuda"

    def test_suite2p_settings_inherit_the_shared_fields(self):
        from photon_mosaic.preprocessing import Suite2pRegistrationSettings

        settings = Suite2pRegistrationSettings(batch_size=42)
        assert isinstance(settings, RegistrationSettings)
        assert settings.batch_size == 42
        assert settings.nonrigid is True


class TestRegisterImaging:
    def test_applies_the_backend_registration_class(self, imaging):
        motion = DoublingMotion(imaging=imaging, displacements=[np.zeros((6, 1, 2))])
        registered = RegisterImaging(imaging, motion, method="doubling_test_backend")

        assert isinstance(registered.epochs[0], DoublingEpoch)
        expected = imaging.epochs[0].get_series(0, 6) * 2
        np.testing.assert_allclose(registered.epochs[0].get_series(0, 6), expected)

    def test_epoch_count_mismatch_raises(self, imaging):
        motion = DoublingMotion(imaging=imaging, displacements=[np.zeros((6, 1, 2)), np.zeros((6, 1, 2))])
        with pytest.raises(ValueError, match="does not match imaging"):
            RegisterImaging(imaging, motion, method="doubling_test_backend")

    def test_motion_without_registration_class_raises(self, imaging):
        motion = Motion(imaging=imaging, displacements=[np.zeros((6, 1, 2))])
        with pytest.raises(TypeError, match="registration_class"):
            RegisterImaging(imaging, motion, method="unregistered_test_backend")

    def test_serialisation_keeps_the_method(self, imaging):
        from spikeinterface.core.base import BaseExtractor

        motion = DoublingMotion(imaging=imaging, displacements=[np.zeros((6, 1, 2))])
        registered = RegisterImaging(imaging, motion, method="doubling_test_backend")
        restored = BaseExtractor.from_dict(registered.to_dict())
        assert isinstance(restored.epochs[0], DoublingEpoch)

    def test_kwargs_are_forwarded_to_the_epoch(self, imaging):
        motion = DoublingMotion(imaging=imaging, displacements=[np.zeros((6, 1, 2))])
        registered = RegisterImaging(imaging, motion, method="doubling_test_backend", flavour="test")
        assert registered.epochs[0].kwargs == {"flavour": "test"}


class TestRegisterImagingEpoch:
    """The shared bookkeeping of every backend's epoch class."""

    @pytest.fixture()
    def epoch(self):
        im = generate_random_imaging(num_frames=6, height=8, width=9, num_planes=3, sampling_frequency=10.0, seed=2)
        motion = DoublingMotion(imaging=im, displacements=[np.zeros((6, 3, 2))])
        return DoublingEpoch(im.epochs[0], motion, 0), im

    def test_output_shape_dtype_and_per_plane_dispatch(self, epoch):
        ep, im = epoch
        out = ep.get_series(1, 4)
        assert out.shape == (3, 8, 9, 3) and out.dtype == np.float32
        np.testing.assert_allclose(out, im.epochs[0].get_series(1, 4) * 2)

    def test_plane_indices_int_slice_and_list(self, epoch):
        ep, im = epoch
        full = im.epochs[0].get_series(0, 6) * 2
        np.testing.assert_allclose(ep.get_series(0, 6, plane_indices=2), full[..., 2:3])
        np.testing.assert_allclose(ep.get_series(0, 6, plane_indices=slice(0, 2)), full[..., 0:2])
        np.testing.assert_allclose(ep.get_series(0, 6, plane_indices=[2, 0]), full[..., [2, 0]])

    def test_end_frame_past_end_warns_and_clamps(self, epoch, caplog):
        ep, _ = epoch
        with caplog.at_level("WARNING"):
            out = ep.get_series(2, 60)
        assert out.shape[0] == 4
        assert any("exceeds recording length" in rec.message for rec in caplog.records)

    def test_start_past_end_raises(self, epoch):
        ep, _ = epoch
        with pytest.raises(ValueError, match="past end_frame"):
            ep.get_series(10, 20)

    def test_empty_range_returns_empty(self, epoch):
        ep, _ = epoch
        assert ep.get_series(3, 3).shape == (0, 8, 9, 3)

    def test_correct_plane_is_abstract(self, epoch):
        _, im = epoch
        motion = DoublingMotion(imaging=im, displacements=[np.zeros((6, 3, 2))])
        with pytest.raises(NotImplementedError):
            RegisterImagingEpoch(im.epochs[0], motion, 0).get_series(0, 2)


def test_register_motion_is_alias():
    assert register_motion is RegisterImaging
