from pathlib import Path

import pytest

from photon_mosaic.preprocessing.suite2p_registration import (
    Suite2pRegistrationSettings,
)

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
