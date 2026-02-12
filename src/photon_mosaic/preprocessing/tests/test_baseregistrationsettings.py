from pathlib import Path

import pytest

from photon_mosaic.preprocessing.baseregistrationsettings import (
    BasePreprocessorSettings,
    CaImAnRegistrationSettings,
    Suite2pRegistrationSettings,
)

# ---------------------------------------------------------------------------
# BasePreprocessorSettings
# ---------------------------------------------------------------------------


class TestBasePreprocessorSettings:
    def test_instantiation(self):
        settings = BasePreprocessorSettings()
        assert isinstance(settings, BasePreprocessorSettings)

    def test_subclass_relationship(self):
        assert issubclass(Suite2pRegistrationSettings, BasePreprocessorSettings)
        assert issubclass(CaImAnRegistrationSettings, BasePreprocessorSettings)


# ---------------------------------------------------------------------------
# Suite2pRegistrationSettings – defaults
# ---------------------------------------------------------------------------


class TestSuite2pDefaults:
    @pytest.fixture()
    def settings(self):
        return Suite2pRegistrationSettings()

    def test_debug_default(self, settings):
        assert settings.debug is False

    def test_tmp_dir_default(self, settings):
        assert settings.tmp_dir == Path("/scratch")

    def test_data_type_default(self, settings):
        assert settings.data_type == "h5"

    def test_batch_size_default(self, settings):
        assert settings.batch_size == 500

    def test_align_by_chan_default(self, settings):
        assert settings.align_by_chan == 1

    def test_maxregshift_default(self, settings):
        assert settings.maxregshift == pytest.approx(0.1)

    def test_force_refImg_default(self, settings):
        assert settings.force_refImg is True

    def test_nonrigid_default(self, settings):
        assert settings.nonrigid is True

    def test_block_size_default(self, settings):
        assert settings.block_size == [128, 128]

    def test_snr_thresh_default(self, settings):
        assert settings.snr_thresh == pytest.approx(1.2)

    def test_maxregshiftNR_default(self, settings):
        assert settings.maxregshiftNR == 5

    def test_smooth_sigma_default(self, settings):
        assert settings.smooth_sigma == pytest.approx(1.15)

    def test_smooth_sigma_time_default(self, settings):
        assert settings.smooth_sigma_time == 0

    def test_clip_negative_default(self, settings):
        assert settings.clip_negative is False

    def test_trim_frames_defaults(self, settings):
        assert settings.trim_frames_start == 0
        assert settings.trim_frames_end == 0

    def test_auto_remove_empty_frames_default(self, settings):
        assert settings.auto_remove_empty_frames is True

    def test_preview_defaults(self, settings):
        assert settings.preview_frame_bin_seconds == pytest.approx(2.0)
        assert settings.preview_playback_factor == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Suite2pRegistrationSettings – custom values
# ---------------------------------------------------------------------------


class TestSuite2pCustomValues:
    def test_override_scalar_fields(self):
        settings = Suite2pRegistrationSettings(
            debug=True,
            batch_size=200,
            maxregshift=0.05,
            nonrigid=False,
            smooth_sigma=2.0,
        )
        assert settings.debug is True
        assert settings.batch_size == 200
        assert settings.maxregshift == pytest.approx(0.05)
        assert settings.nonrigid is False
        assert settings.smooth_sigma == pytest.approx(2.0)

    def test_override_tmp_dir(self):
        settings = Suite2pRegistrationSettings(tmp_dir=Path("/tmp/my_scratch"))
        assert settings.tmp_dir == Path("/tmp/my_scratch")

    def test_override_block_size(self):
        settings = Suite2pRegistrationSettings(block_size=[64, 64])
        assert settings.block_size == [64, 64]


# ---------------------------------------------------------------------------
# Suite2pRegistrationSettings – env var loading
# ---------------------------------------------------------------------------


class TestSuite2pEnvVars:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("SUITE2P_REGISTRATION_BATCH_SIZE", "250")
        settings = Suite2pRegistrationSettings()
        assert settings.batch_size == 250

    def test_env_var_bool(self, monkeypatch):
        monkeypatch.setenv("SUITE2P_REGISTRATION_DEBUG", "true")
        settings = Suite2pRegistrationSettings()
        assert settings.debug is True

    def test_env_var_float(self, monkeypatch):
        monkeypatch.setenv("SUITE2P_REGISTRATION_MAXREGSHIFT", "0.2")
        settings = Suite2pRegistrationSettings()
        assert settings.maxregshift == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# CaImAnRegistrationSettings – defaults
# ---------------------------------------------------------------------------


class TestCaImAnDefaults:
    @pytest.fixture()
    def settings(self):
        return CaImAnRegistrationSettings()

    def test_debug_default(self, settings):
        assert settings.debug is False

    def test_max_shifts_default(self, settings):
        assert settings.max_shifts == (6, 6)

    def test_niter_rig_default(self, settings):
        assert settings.niter_rig == 1

    def test_splits_rig_default(self, settings):
        assert settings.splits_rig == 14

    def test_num_splits_to_process_rig_default(self, settings):
        assert settings.num_splits_to_process_rig is None

    def test_strides_default(self, settings):
        assert settings.strides == (96, 96)

    def test_overlaps_default(self, settings):
        assert settings.overlaps == (32, 32)

    def test_splits_els_default(self, settings):
        assert settings.splits_els == 14

    def test_upsample_factor_grid_default(self, settings):
        assert settings.upsample_factor_grid == 4

    def test_max_deviation_rigid_default(self, settings):
        assert settings.max_deviation_rigid == 3

    def test_shifts_opencv_default(self, settings):
        assert settings.shifts_opencv is True

    def test_nonneg_movie_default(self, settings):
        assert settings.nonneg_movie is True

    def test_gSig_filt_default(self, settings):
        assert settings.gSig_filt == []

    def test_border_nan_default(self, settings):
        assert settings.border_nan == "copy"

    def test_pw_rigid_default(self, settings):
        assert settings.pw_rigid is False

    def test_num_frames_split_default(self, settings):
        assert settings.num_frames_split == 80

    def test_var_name_hdf5_default(self, settings):
        assert settings.var_name_hdf5 == "mov"

    def test_is3D_default(self, settings):
        assert settings.is3D is False

    def test_shifts_interpolate_default(self, settings):
        assert settings.shifts_interpolate is False


# ---------------------------------------------------------------------------
# CaImAnRegistrationSettings – custom values
# ---------------------------------------------------------------------------


class TestCaImAnCustomValues:
    def test_override_scalar_fields(self):
        settings = CaImAnRegistrationSettings(
            debug=True,
            niter_rig=3,
            max_deviation_rigid=5,
            pw_rigid=True,
            num_frames_split=120,
        )
        assert settings.debug is True
        assert settings.niter_rig == 3
        assert settings.max_deviation_rigid == 5
        assert settings.pw_rigid is True
        assert settings.num_frames_split == 120

    def test_override_tuple_fields(self):
        settings = CaImAnRegistrationSettings(
            max_shifts=(10, 10),
            strides=(48, 48),
            overlaps=(16, 16),
        )
        assert settings.max_shifts == (10, 10)
        assert settings.strides == (48, 48)
        assert settings.overlaps == (16, 16)

    def test_override_gSig_filt(self):
        settings = CaImAnRegistrationSettings(gSig_filt=[5, 5])
        assert settings.gSig_filt == [5, 5]

    def test_override_border_nan(self):
        settings = CaImAnRegistrationSettings(border_nan="min")
        assert settings.border_nan == "min"


# ---------------------------------------------------------------------------
# CaImAnRegistrationSettings – env var loading
# ---------------------------------------------------------------------------


class TestCaImAnEnvVars:
    def test_env_var_override_int(self, monkeypatch):
        monkeypatch.setenv("CAIMAN_REGISTRATION_NITER_RIG", "5")
        settings = CaImAnRegistrationSettings()
        assert settings.niter_rig == 5

    def test_env_var_override_bool(self, monkeypatch):
        monkeypatch.setenv("CAIMAN_REGISTRATION_PW_RIGID", "true")
        settings = CaImAnRegistrationSettings()
        assert settings.pw_rigid is True

    def test_env_var_override_str(self, monkeypatch):
        monkeypatch.setenv("CAIMAN_REGISTRATION_BORDER_NAN", "min")
        settings = CaImAnRegistrationSettings()
        assert settings.border_nan == "min"
