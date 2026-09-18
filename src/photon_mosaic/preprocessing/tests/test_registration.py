"""Tests for the backend-agnostic registration layer."""

from pathlib import Path

import numpy as np
import pytest

from photon_mosaic.core import Motion, generate_random_imaging
from photon_mosaic.preprocessing import RegisterImaging, RegistrationSettings, register_motion
from photon_mosaic.preprocessing.basepreprocessor import BasePreprocessorEpoch


class DoublingEpoch(BasePreprocessorEpoch):
    """Stand-in backend epoch: 'applies' motion by doubling the frames."""

    def __init__(self, parent_imaging_epoch, motion, epoch_index, **kwargs):
        BasePreprocessorEpoch.__init__(self, parent_imaging_epoch)
        self.motion = motion
        self.epoch_index = epoch_index
        self.kwargs = kwargs

    def get_series(self, start_frame, end_frame, plane_indices=None):
        return self.parent_imaging_epoch.get_series(start_frame, end_frame) * 2


class DoublingMotion(Motion):
    method_name = "doubling_test_backend"
    epoch_class = DoublingEpoch


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
    def test_applies_the_backend_epoch_class(self, imaging):
        motion = DoublingMotion(imaging=imaging, displacements=[np.zeros((6, 1, 2))])
        registered = RegisterImaging(imaging, motion)

        assert isinstance(registered.epochs[0], DoublingEpoch)
        expected = imaging.epochs[0].get_series(0, 6) * 2
        np.testing.assert_allclose(registered.epochs[0].get_series(0, 6), expected)

    def test_epoch_count_mismatch_raises(self, imaging):
        motion = DoublingMotion(imaging=imaging, displacements=[np.zeros((6, 1, 2)), np.zeros((6, 1, 2))])
        with pytest.raises(ValueError, match="does not match imaging"):
            RegisterImaging(imaging, motion)

    def test_motion_without_epoch_class_raises(self, imaging):
        motion = Motion(imaging=imaging, displacements=[np.zeros((6, 1, 2))])
        with pytest.raises(TypeError, match="epoch_class"):
            RegisterImaging(imaging, motion)

    def test_kwargs_are_forwarded_to_the_epoch(self, imaging):
        motion = DoublingMotion(imaging=imaging, displacements=[np.zeros((6, 1, 2))])
        registered = RegisterImaging(imaging, motion, flavour="test")
        assert registered.epochs[0].kwargs == {"flavour": "test"}


def test_register_motion_is_alias():
    assert register_motion is RegisterImaging
