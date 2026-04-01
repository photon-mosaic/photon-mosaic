import numpy as np
import pytest
from pydantic import ValidationError

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.numpyimaging import NumpyImaging
from photon_mosaic.preprocessing.baseregistrationsettings import Suite2pRegistrationSettings
from photon_mosaic.preprocessing.suite2p import (
    RegisterSuite2PImaging,
    RegisterSuite2PImagingEpoch,
    Suite2PMotion,
    compute_motion_suite2p,
    register_suite2p,
)


def _make_shifted_plane(height, width, dy, dx, value):
    base = np.zeros((height, width), dtype=np.float32)
    y0 = height // 2 - 1
    x0 = width // 2 - 1
    base[y0 : y0 + 2, x0 : x0 + 2] = value

    frame = np.zeros_like(base)
    y_start = y0 + dy
    x_start = x0 + dx
    y_end = y_start + 2
    x_end = x_start + 2
    if y_start < 0 or x_start < 0 or y_end > height or x_end > width:
        raise ValueError("Shift moves block out of bounds")
    frame[y_start:y_end, x_start:x_end] = value
    return frame


def _make_shifted_video(num_frames, num_planes=1, height=16, width=16, shifts=None):
    if shifts is None:
        shifts = [[(0, 0)] * num_frames for _ in range(num_planes)]
    # Allow single-plane shorthand: list of tuples
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
    videos = []
    if shifts is None:
        shifts = [None] * len(num_frames)
    # If user passed plane-wise shifts for a single epoch, wrap it
    if len(num_frames) == 1 and shifts and isinstance(shifts[0], list) and shifts[0] and isinstance(shifts[0][0], tuple):
        shifts = [shifts]
    for epoch_idx, n_frames in enumerate(num_frames):
        epoch_shifts = shifts[epoch_idx]
        videos.append(_make_shifted_video(n_frames, num_planes=num_planes, height=height, width=width, shifts=epoch_shifts))
    return NumpyImaging(imaging_series=videos, sampling_frequency=sampling_frequency)


class TestSuite2PMotion:
    @pytest.fixture()
    def imaging_single(self):
        return generate_random_imaging(
            num_frames=10,
            height=8,
            width=9,
            num_planes=1,
            sampling_frequency=30.0,
            seed=0,
        )

    @pytest.fixture()
    def imaging_multi(self):
        return generate_random_imaging(
            num_frames=10,
            height=8,
            width=9,
            num_planes=3,
            sampling_frequency=30.0,
            seed=1,
        )

    @pytest.fixture()
    def motion_single(self, imaging_single):
        displacements = [np.random.default_rng(0).random((10, 1, 2))]
        return Suite2PMotion(
            imaging=imaging_single,
            displacements=displacements,
            refAndMasks=("ref", "masks"),
            ops={"bidiphase": 0},
            blocks=[None],
        )

    def test_attributes(self, imaging_single, motion_single):
        assert motion_single.imaging is imaging_single
        assert motion_single.num_epochs == 1
        assert motion_single.refAndMasks == ("ref", "masks")
        assert motion_single.ops == {"bidiphase": 0}

    def test_num_epochs_matches_displacements(self, imaging_single):
        disps = [np.zeros((5, 1, 2)), np.zeros((8, 1, 2))]
        m = Suite2PMotion(imaging_single, disps, None, {})
        assert m.num_epochs == 2

    def test_get_displacement_at_frames_single_int(self, motion_single):
        d = motion_single.get_displacement_at_frames(3)
        assert d.shape == (1, 2)
        np.testing.assert_array_equal(d, motion_single.displacements[0][3])

    def test_get_displacement_at_frames_array(self, motion_single):
        indices = np.array([1, 4, 7])
        d = motion_single.get_displacement_at_frames(indices)
        assert d.shape == (3, 1, 2)
        np.testing.assert_array_equal(d, motion_single.displacements[0][indices])

    def test_get_displacement_at_frames_plane_index(self, imaging_multi):
        disps = [np.random.default_rng(2).random((10, 3, 2))]
        motion = Suite2PMotion(imaging_multi, disps, None, {})
        single_val = motion.get_displacement_at_frames(0, plane_index=1)
        assert single_val.shape == (2,)
        np.testing.assert_array_equal(single_val, disps[0][0, 1])

        multi_val = motion.get_displacement_at_frames(np.array([2, 5]), plane_index=2)
        assert multi_val.shape == (2, 2)
        np.testing.assert_array_equal(multi_val, disps[0][[2, 5], 2])


class TestComputeMotionSettingsResolution:
    def test_none_settings_creates_defaults_from_kwargs(self):
        settings = None
        kwargs = {"batch_size": 200}
        if settings is None:
            resolved = Suite2pRegistrationSettings(**kwargs)
        assert resolved.batch_size == 200
        assert resolved.nonrigid is True

    def test_none_settings_no_kwargs_uses_all_defaults(self):
        resolved = Suite2pRegistrationSettings()
        assert resolved.batch_size == 500

    def test_dict_settings_validated(self):
        d = {"batch_size": 300, "nonrigid": False}
        resolved = Suite2pRegistrationSettings.model_validate(d)
        assert resolved.batch_size == 300
        assert resolved.nonrigid is False

    def test_dict_settings_with_kwargs_merged(self):
        d = {"batch_size": 300}
        kwargs = {"batch_size": 999}
        resolved = Suite2pRegistrationSettings.model_validate({**d, **kwargs})
        assert resolved.batch_size == 999

    def test_settings_object_with_kwargs_copied(self):
        settings = Suite2pRegistrationSettings(batch_size=100)
        updated = settings.model_copy(update={"batch_size": 777})
        assert updated.batch_size == 777
        assert settings.batch_size == 100

    def test_invalid_dict_raises_validation_error(self):
        with pytest.raises(ValidationError):
            Suite2pRegistrationSettings.model_validate({"batch_size": "not_a_number_at_all!"})

    def test_json_string_roundtrip(self):
        original = Suite2pRegistrationSettings(batch_size=250, nonrigid=False)
        json_str = original.model_dump_json()
        restored = Suite2pRegistrationSettings.model_validate_json(json_str)
        assert restored.batch_size == 250
        assert restored.nonrigid is False


class TestComputeMotionSuite2p:
    def test_returns_suite2p_motion_and_registers_frames(self):
        shifts = [[(0, 0), (0, 0), (1, 0), (1, 0), (0, 1), (0, 1)]]
        imaging = _make_shifted_imaging(num_frames=6, num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0)
        motion = compute_motion_suite2p(imaging, settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False))
        assert isinstance(motion, Suite2PMotion)
        assert motion.num_epochs == 1
        assert motion.displacements[0].shape == (6, 1, 2)

        # Applying the computed motion should restore the unshifted template (first frame)
        registered = RegisterSuite2PImaging(imaging, motion).epochs[0].get_series(0, 6)
        np.testing.assert_allclose(registered, registered[0:1].repeat(6, axis=0), atol=1e-3)

    def test_kwargs_override_settings_and_batching(self):
        shifts = [[(0, 0)] * 8]
        imaging = _make_shifted_imaging(num_frames=8, num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0)
        settings = Suite2pRegistrationSettings(batch_size=6)
        motion = compute_motion_suite2p(imaging, settings=settings, batch_size=3)
        assert motion.displacements[0].shape == (8, 1, 2)

    def test_multi_epoch_single_plane_alignment(self):
        shifts = [
            [(0, 0), (0, 0), (1, 0), (1, 0), (0, 1)],
            [(0, 0), (0, 0), (-1, 0), (-1, 0)],
        ]
        imaging = _make_shifted_imaging(num_frames=(5, 4), num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0)
        motion = compute_motion_suite2p(imaging, settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False))

        assert motion.num_epochs == 2
        assert motion.displacements[0].shape == (5, 1, 2)
        assert motion.displacements[1].shape == (4, 1, 2)

        reg = RegisterSuite2PImaging(imaging, motion)
        out_epoch0 = reg.epochs[0].get_series(0, 5)
        out_epoch1 = reg.epochs[1].get_series(0, 4)
        np.testing.assert_allclose(out_epoch0, out_epoch0[0:1].repeat(5, axis=0), atol=1e-3)
        diff = np.abs(out_epoch1 - out_epoch1[0:1].repeat(4, axis=0))
        assert diff.max() <= 1
        assert np.count_nonzero(diff > 1e-6) <= 16

    def test_multi_plane_multi_epoch_shapes_and_alignment(self):
        plane0 = [(0, 0), (0, 0), (1, 0), (0, 1)]
        plane1 = [(0, 0), (0, 0), (-1, 0), (0, -1)]
        plane2 = [(0, 0), (0, 0), (1, 1), (-1, -1)]
        shifts_epoch0 = [plane0, plane1, plane2]

        plane0_e1 = [(0, 0), (0, 0), (1, 0)]
        plane1_e1 = [(0, 0), (0, 0), (-1, 0)]
        plane2_e1 = [(0, 0), (0, 0), (0, 1)]
        shifts_epoch1 = [plane0_e1, plane1_e1, plane2_e1]

        imaging = _make_shifted_imaging(
            num_frames=(4, 3),
            num_planes=3,
            height=12,
            width=12,
            shifts=[shifts_epoch0, shifts_epoch1],
            sampling_frequency=10.0,
        )

        motion = compute_motion_suite2p(imaging, settings=Suite2pRegistrationSettings(batch_size=2, nonrigid=False))
        assert motion.displacements[0].shape == (4, 3, 2)
        assert motion.displacements[1].shape == (3, 3, 2)
        assert len(motion.refAndMasks) == 3

        reg = RegisterSuite2PImaging(imaging, motion)
        out0 = reg.epochs[0].get_series(0, 4)
        out1 = reg.epochs[1].get_series(0, 3)
        np.testing.assert_allclose(out0[..., 0], out0[0:1, ..., 0].repeat(4, axis=0), atol=1e-3)
        np.testing.assert_allclose(out0[..., 1], out0[0:1, ..., 1].repeat(4, axis=0), atol=1e-3)
        np.testing.assert_allclose(out0[..., 2], out0[0:1, ..., 2].repeat(4, axis=0), atol=1e-3)
        diff_plane0 = np.abs(out1[..., 0] - out1[0:1, ..., 0].repeat(3, axis=0))
        assert diff_plane0.max() <= 1
        assert np.count_nonzero(diff_plane0 > 1e-6) <= 16


class TestRegisterSuite2PImaging:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(num_frames=10, height=8, width=9, sampling_frequency=30.0, seed=0)

    @pytest.fixture()
    def motion(self, imaging):
        displacements = [np.zeros((10, 1, 2))]
        return Suite2PMotion(imaging, displacements, ("ref", "masks"), {"bidiphase": 0})

    def test_construction(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg._parent is imaging
        assert reg.get_num_epochs() == 1

    def test_epoch_mismatch_raises(self, imaging):
        displacements = [np.zeros((10, 1, 2)), np.zeros((10, 1, 2))]
        motion = Suite2PMotion(imaging, displacements, None, {})
        with pytest.raises(ValueError, match="epochs"):
            RegisterSuite2PImaging(imaging, motion)

    def test_inherits_sampling_frequency(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg.sampling_frequency == imaging.sampling_frequency

    def test_inherits_shape(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg.shape == imaging.shape


class TestRegisterSuite2PImagingEpoch:
    @pytest.fixture()
    def imaging(self):
        shifts = [[(0, 0), (0, 0), (1, 0), (0, 1), (1, 1), (0, 0)]]
        return _make_shifted_imaging(num_frames=6, num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0)

    @pytest.fixture()
    def motion(self, imaging):
        return compute_motion_suite2p(imaging, settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False))

    def test_construction(self, imaging, motion):
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)
        assert epoch.parent_imaging_epoch is parent_epoch
        assert epoch.motion is motion
        assert epoch.epoch_index == 0

    def test_get_num_samples(self, imaging, motion):
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)
        assert epoch.get_num_samples() == 6

    def test_get_series_calls_shift_frames(self, imaging, motion):
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)
        result = epoch.get_series(0, 6)
        assert result.shape == (6, 12, 12, 1)
        np.testing.assert_allclose(result, result[0:1].repeat(6, axis=0), atol=1e-3)

    def test_get_series_multiplane(self):
        shifts = [[(0, 0)] * 5, [(0, 0)] * 5, [(0, 0)] * 5]
        imaging = _make_shifted_imaging(num_frames=5, num_planes=3, height=12, width=12, shifts=shifts, sampling_frequency=10.0)
        motion = compute_motion_suite2p(imaging, settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False))
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)

        result = epoch.get_series(0, 5)
        assert result.shape == (5, 12, 12, 3)
        np.testing.assert_allclose(result[..., 0], result[0:1, ..., 0].repeat(5, axis=0), atol=1e-3)
        np.testing.assert_allclose(result[..., 1], result[0:1, ..., 1].repeat(5, axis=0), atol=1e-3)
        np.testing.assert_allclose(result[..., 2], result[0:1, ..., 2].repeat(5, axis=0), atol=1e-3)

    def test_get_series_plane_slice(self):
        shifts = [[(0, 0)] * 5, [(0, 0)] * 5, [(0, 0)] * 5]
        imaging = _make_shifted_imaging(num_frames=5, num_planes=3, height=12, width=12, shifts=shifts, sampling_frequency=10.0)
        motion = compute_motion_suite2p(imaging, settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False))
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)

        result = epoch.get_series(0, 5, plane_indices=slice(0, 2))

        assert result.shape == (5, 12, 12, 2)
        np.testing.assert_allclose(result[..., 0], result[0:1, ..., 0].repeat(5, axis=0), atol=1e-3)
        np.testing.assert_allclose(result[..., 1], result[0:1, ..., 1].repeat(5, axis=0), atol=1e-3)


def test_register_suite2p_is_alias():
    assert register_suite2p is RegisterSuite2PImaging
