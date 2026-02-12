from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.preprocessing.baseregistrationsettings import (
    Suite2pRegistrationSettings,
)
from photon_mosaic.preprocessing.suite2p import (
    RegisterSuite2PImaging,
    RegisterSuite2PImagingEpoch,
    Suite2PMotion,
    compute_motion_suite2p,
    register_suite2p,
)

# ---------------------------------------------------------------------------
# Suite2PMotion
# ---------------------------------------------------------------------------


class TestSuite2PMotion:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(num_frames=10, height=8, width=9, sampling_frequency=30.0, seed=0)

    @pytest.fixture()
    def motion(self, imaging):
        displacements = [np.random.default_rng(0).random((10, 2))]
        return Suite2PMotion(
            imaging=imaging,
            displacements=displacements,
            refAndMasks=("ref", "masks"),
            ops={"bidiphase": 0},
        )

    def test_attributes(self, imaging, motion):
        assert motion.imaging is imaging
        assert motion.num_epochs == 1
        assert motion.refAndMasks == ("ref", "masks")
        assert motion.ops == {"bidiphase": 0}

    def test_num_epochs_matches_displacements(self, imaging):
        disps = [np.zeros((5, 2)), np.zeros((8, 2))]
        m = Suite2PMotion(imaging, disps, None, {})
        assert m.num_epochs == 2

    def test_get_displacement_at_frames_single_int(self, motion):
        d = motion.get_displacement_at_frames(3)
        assert d.shape == (2,)
        np.testing.assert_array_equal(d, motion.displacements[0][3])

    def test_get_displacement_at_frames_array(self, motion):
        indices = np.array([1, 4, 7])
        d = motion.get_displacement_at_frames(indices)
        assert d.shape == (3, 2)
        np.testing.assert_array_equal(d, motion.displacements[0][indices])

    def test_get_displacement_at_frames_epoch_index(self, imaging):
        d0 = np.random.default_rng(1).random((10, 2))
        d1 = np.random.default_rng(2).random((10, 2))
        m = Suite2PMotion(imaging, [d0, d1], None, {})
        np.testing.assert_array_equal(m.get_displacement_at_frames(0, epoch_index=1), d1[0])


# ---------------------------------------------------------------------------
# compute_motion_suite2p – settings resolution
# ---------------------------------------------------------------------------


class TestComputeMotionSettingsResolution:
    """Test that settings are resolved correctly without running suite2p."""

    def test_none_settings_creates_defaults_from_kwargs(self):
        """When settings=None, a Suite2pRegistrationSettings is created from kwargs."""
        settings = None
        kwargs = {"batch_size": 200}
        if settings is None:
            resolved = Suite2pRegistrationSettings(**kwargs)
        assert resolved.batch_size == 200
        # All other fields should have their defaults
        assert resolved.nonrigid is True

    def test_none_settings_no_kwargs_uses_all_defaults(self):
        """When settings=None and no kwargs, all defaults are used."""
        resolved = Suite2pRegistrationSettings()
        assert resolved.batch_size == 500

    def test_dict_settings_validated(self):
        """A dict is validated into a Suite2pRegistrationSettings."""
        d = {"batch_size": 300, "nonrigid": False}
        resolved = Suite2pRegistrationSettings.model_validate(d)
        assert resolved.batch_size == 300
        assert resolved.nonrigid is False

    def test_dict_settings_with_kwargs_merged(self):
        """kwargs override dict values."""
        d = {"batch_size": 300}
        kwargs = {"batch_size": 999}
        resolved = Suite2pRegistrationSettings.model_validate({**d, **kwargs})
        assert resolved.batch_size == 999

    def test_settings_object_with_kwargs_copied(self):
        """kwargs override fields on an existing settings object via model_copy."""
        settings = Suite2pRegistrationSettings(batch_size=100)
        updated = settings.model_copy(update={"batch_size": 777})
        assert updated.batch_size == 777
        assert settings.batch_size == 100  # original unchanged

    def test_invalid_dict_raises_validation_error(self):
        """A dict with invalid types raises a ValidationError."""
        with pytest.raises(ValidationError):
            Suite2pRegistrationSettings.model_validate({"batch_size": "not_a_number_at_all!"})

    def test_json_string_roundtrip(self):
        """Settings can be serialized to JSON and deserialized back."""
        original = Suite2pRegistrationSettings(batch_size=250, nonrigid=False)
        json_str = original.model_dump_json()
        restored = Suite2pRegistrationSettings.model_validate_json(json_str)
        assert restored.batch_size == 250
        assert restored.nonrigid is False


# ---------------------------------------------------------------------------
# compute_motion_suite2p – integration with mocked suite2p
# ---------------------------------------------------------------------------


class TestComputeMotionSuite2p:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(num_frames=20, height=8, width=9, sampling_frequency=30.0, seed=0)

    @pytest.fixture()
    def mock_suite2p(self):
        mock_register = MagicMock()
        mock_register.compute_reference.return_value = np.zeros((8, 9))
        mock_register.compute_reference_masks.return_value = ("ref", "masks")

        def fake_register_frames(refAndMasks, frames, **kw):
            n = frames.shape[0]
            return (
                frames,
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                None,
            )

        mock_register.register_frames.side_effect = fake_register_frames
        mock_default_ops = MagicMock(return_value={"bidiphase": 0})
        return mock_register, mock_default_ops

    def test_returns_suite2p_motion(self, imaging, mock_suite2p):
        mock_register, mock_default_ops = mock_suite2p
        with patch.dict(
            "sys.modules",
            {
                "suite2p": MagicMock(),
                "suite2p.registration": MagicMock(register=mock_register),
                "suite2p.default_ops": MagicMock(default_ops=mock_default_ops),
            },
        ):
            motion = compute_motion_suite2p(imaging)
        assert isinstance(motion, Suite2PMotion)
        assert motion.num_epochs == 1
        assert motion.displacements[0].shape == (20, 2)

    def test_with_dict_settings(self, imaging, mock_suite2p):
        mock_register, mock_default_ops = mock_suite2p
        with patch.dict(
            "sys.modules",
            {
                "suite2p": MagicMock(),
                "suite2p.registration": MagicMock(register=mock_register),
                "suite2p.default_ops": MagicMock(default_ops=mock_default_ops),
            },
        ):
            motion = compute_motion_suite2p(imaging, settings={"batch_size": 10})
        assert isinstance(motion, Suite2PMotion)
        # With batch_size=10 and 20 frames, register_frames should be called twice
        assert mock_register.register_frames.call_count == 2

    def test_with_settings_object(self, imaging, mock_suite2p):
        mock_register, mock_default_ops = mock_suite2p
        settings = Suite2pRegistrationSettings(batch_size=20)
        with patch.dict(
            "sys.modules",
            {
                "suite2p": MagicMock(),
                "suite2p.registration": MagicMock(register=mock_register),
                "suite2p.default_ops": MagicMock(default_ops=mock_default_ops),
            },
        ):
            motion = compute_motion_suite2p(imaging, settings=settings)
        assert isinstance(motion, Suite2PMotion)
        # batch_size=20 covers all 20 frames in one call
        assert mock_register.register_frames.call_count == 1

    def test_kwargs_override_settings(self, imaging, mock_suite2p):
        mock_register, mock_default_ops = mock_suite2p
        settings = Suite2pRegistrationSettings(batch_size=20)
        with patch.dict(
            "sys.modules",
            {
                "suite2p": MagicMock(),
                "suite2p.registration": MagicMock(register=mock_register),
                "suite2p.default_ops": MagicMock(default_ops=mock_default_ops),
            },
        ):
            motion = compute_motion_suite2p(imaging, settings=settings, batch_size=5)
        assert isinstance(motion, Suite2PMotion)
        # batch_size overridden to 5: 20 frames / 5 = 4 batches
        assert mock_register.register_frames.call_count == 4

    def test_multi_epoch(self, mock_suite2p):
        imaging = generate_random_imaging(
            num_frames=(10, 15),
            height=8,
            width=9,
            sampling_frequency=30.0,
            seed=1,
        )
        mock_register, mock_default_ops = mock_suite2p
        with patch.dict(
            "sys.modules",
            {
                "suite2p": MagicMock(),
                "suite2p.registration": MagicMock(register=mock_register),
                "suite2p.default_ops": MagicMock(default_ops=mock_default_ops),
            },
        ):
            motion = compute_motion_suite2p(imaging)
        assert motion.num_epochs == 2
        assert motion.displacements[0].shape == (10, 2)
        assert motion.displacements[1].shape == (15, 2)


# ---------------------------------------------------------------------------
# RegisterSuite2PImaging
# ---------------------------------------------------------------------------


class TestRegisterSuite2PImaging:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(num_frames=10, height=8, width=9, sampling_frequency=30.0, seed=0)

    @pytest.fixture()
    def motion(self, imaging):
        displacements = [np.zeros((10, 2))]
        return Suite2PMotion(imaging, displacements, ("ref", "masks"), {"bidiphase": 0})

    def test_construction(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg._parent is imaging
        assert reg.get_num_epochs() == 1

    def test_epoch_mismatch_raises(self, imaging):
        displacements = [np.zeros((10, 2)), np.zeros((10, 2))]
        motion = Suite2PMotion(imaging, displacements, None, {})
        with pytest.raises(ValueError, match="epochs"):
            RegisterSuite2PImaging(imaging, motion)

    def test_inherits_sampling_frequency(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg.sampling_frequency == imaging.sampling_frequency

    def test_inherits_shape(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg.shape == imaging.shape


# ---------------------------------------------------------------------------
# RegisterSuite2PImagingEpoch
# ---------------------------------------------------------------------------


class TestRegisterSuite2PImagingEpoch:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(num_frames=10, height=8, width=9, sampling_frequency=30.0, seed=0)

    @pytest.fixture()
    def motion(self, imaging):
        displacements = [np.zeros((10, 2))]
        return Suite2PMotion(imaging, displacements, ("ref", "masks"), {"bidiphase": 0})

    def test_construction(self, imaging, motion):
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)
        assert epoch.parent_imaging_epoch is parent_epoch
        assert epoch.motion is motion
        assert epoch.epoch_index == 0

    def test_get_num_samples(self, imaging, motion):
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)
        assert epoch.get_num_samples() == 10

    def test_get_series_calls_suite2p(self, imaging, motion):
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)

        mock_register = MagicMock()
        fake_video = np.zeros((5, 8, 9))
        mock_register.register_frames.return_value = (fake_video,)

        with patch.dict(
            "sys.modules",
            {
                "suite2p": MagicMock(),
                "suite2p.registration": MagicMock(register=mock_register),
            },
        ):
            result = epoch.get_series(0, 5)

        mock_register.register_frames.assert_called_once()
        assert result.shape[0] == 5


# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------


def test_register_suite2p_is_alias():
    assert register_suite2p is RegisterSuite2PImaging
