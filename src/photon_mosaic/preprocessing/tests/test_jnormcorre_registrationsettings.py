import pytest
from pydantic import ValidationError

from photon_mosaic.preprocessing.jnormcorre_registration import (
    JNormcorreRegistrationSettings,
)


class TestJNormcorreDefaults:
    @pytest.fixture()
    def settings(self):
        return JNormcorreRegistrationSettings()

    def test_max_shifts_default(self, settings):
        assert settings.max_shifts == (6, 6)

    def test_frames_per_split_default(self, settings):
        assert settings.frames_per_split == 1000

    def test_pw_rigid_default(self, settings):
        assert settings.pw_rigid is False

    def test_strides_default(self, settings):
        assert settings.strides == (96, 96)

    def test_overlaps_default(self, settings):
        assert settings.overlaps == (32, 32)

    def test_max_deviation_rigid_default(self, settings):
        assert settings.max_deviation_rigid == 3

    def test_niter_defaults(self, settings):
        assert settings.niter_rig == 1
        assert settings.niter_els == 1

    def test_split_count_defaults_none(self, settings):
        assert settings.num_splits_to_process_rig is None
        assert settings.num_splits_to_process_els is None

    def test_min_mov_default_none(self, settings):
        assert settings.min_mov is None

    def test_upsample_factor_grid_default(self, settings):
        assert settings.upsample_factor_grid == 4

    def test_batching_default(self, settings):
        assert settings.batching == 100


class TestJNormcorreCustomValues:
    def test_override_scalar_fields(self):
        settings = JNormcorreRegistrationSettings(
            pw_rigid=True,
            frames_per_split=500,
            max_deviation_rigid=5,
            niter_rig=2,
        )
        assert settings.pw_rigid is True
        assert settings.frames_per_split == 500
        assert settings.max_deviation_rigid == 5
        assert settings.niter_rig == 2

    def test_override_tuple_fields(self):
        settings = JNormcorreRegistrationSettings(max_shifts=(10, 10), strides=(48, 48), overlaps=(16, 16))
        assert settings.max_shifts == (10, 10)
        assert settings.strides == (48, 48)
        assert settings.overlaps == (16, 16)


class TestJNormcorreEnvVars:
    def test_env_var_int(self, monkeypatch):
        monkeypatch.setenv("JNORMCORRE_REGISTRATION_FRAMES_PER_SPLIT", "250")
        settings = JNormcorreRegistrationSettings()
        assert settings.frames_per_split == 250

    def test_env_var_bool(self, monkeypatch):
        monkeypatch.setenv("JNORMCORRE_REGISTRATION_PW_RIGID", "true")
        settings = JNormcorreRegistrationSettings()
        assert settings.pw_rigid is True

    def test_env_var_tuple_json(self, monkeypatch):
        monkeypatch.setenv("JNORMCORRE_REGISTRATION_MAX_SHIFTS", "[8, 8]")
        settings = JNormcorreRegistrationSettings()
        assert tuple(settings.max_shifts) == (8, 8)


class TestJNormcorreValidationAndRoundtrip:
    def test_invalid_value_raises(self):
        with pytest.raises(ValidationError):
            JNormcorreRegistrationSettings.model_validate({"frames_per_split": "not_a_number!"})

    def test_json_string_roundtrip(self):
        original = JNormcorreRegistrationSettings(frames_per_split=250, pw_rigid=True)
        restored = JNormcorreRegistrationSettings.model_validate_json(original.model_dump_json())
        assert restored.frames_per_split == 250
        assert restored.pw_rigid is True

    def test_model_dump_drops_no_motioncorrect_surprise(self):
        # model_dump (minus `batching`) is forwarded straight to MotionCorrect,
        # so its keys must stay aligned with the constructor argument names.
        cfg = JNormcorreRegistrationSettings().model_dump()
        expected = {
            "max_shifts",
            "frames_per_split",
            "num_splits_to_process_rig",
            "niter_rig",
            "pw_rigid",
            "strides",
            "overlaps",
            "max_deviation_rigid",
            "num_splits_to_process_els",
            "niter_els",
            "min_mov",
            "upsample_factor_grid",
            "batching",
        }
        assert set(cfg) == expected
