import numpy as np
import pytest
from pydantic import ValidationError
from spikeinterface.core.base import BaseExtractor

from photon_mosaic.core import Motion, NumpyImaging, compute_motion, generate_random_imaging, get_registration_class
from photon_mosaic.preprocessing import suite2p_registration as s2p
from photon_mosaic.preprocessing.registration import RegisterImaging
from photon_mosaic.preprocessing.suite2p_registration import (
    RegisterSuite2PImaging,
    RegisterSuite2PImagingEpoch,
    Suite2PMotion,
    Suite2pRegistrationSettings,
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
    if (
        len(num_frames) == 1
        and shifts
        and isinstance(shifts[0], list)
        and shifts[0]
        and isinstance(shifts[0][0], tuple)
    ):
        shifts = [shifts]
    for epoch_idx, n_frames in enumerate(num_frames):
        epoch_shifts = shifts[epoch_idx]
        videos.append(
            _make_shifted_video(n_frames, num_planes=num_planes, height=height, width=width, shifts=epoch_shifts)
        )
    return NumpyImaging(imaging_series=videos, sampling_frequency=sampling_frequency)


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
        assert resolved.batch_size == 100

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
        imaging = _make_shifted_imaging(
            num_frames=6, num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0
        )
        motion = compute_motion(
            imaging, method="suite2p", settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False)
        )
        assert isinstance(motion, Suite2PMotion)
        assert isinstance(motion, Motion)
        assert motion.num_epochs == 1
        assert motion.displacements[0].shape == (6, 1, 2)

        # Applying the computed motion should restore the unshifted template (first frame)
        registered = RegisterSuite2PImaging(imaging, motion).epochs[0].get_series(0, 6)
        np.testing.assert_allclose(registered, registered[0:1].repeat(6, axis=0), atol=1e-3)

    def test_kwargs_override_settings_and_batching(self):
        shifts = [[(0, 0)] * 8]
        imaging = _make_shifted_imaging(
            num_frames=8, num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0
        )
        settings = Suite2pRegistrationSettings(batch_size=6)
        motion = compute_motion(imaging, method="suite2p", settings=settings, batch_size=3)
        assert motion.displacements[0].shape == (8, 1, 2)

    def test_former_kwargs_are_settings(self, monkeypatch):
        monkeypatch.setenv("SUITE2P_REGISTRATION_NIMG_INIT", "50")
        assert Suite2pRegistrationSettings().nimg_init == 50
        settings = {"nonrigid": False, "norm_frames": False, "spatial_taper": 2.0, "th_badframes": 0.5}
        motion = compute_motion(_make_shifted_imaging(4, height=12, width=12), method="suite2p", settings=settings)
        assert {k: getattr(motion.settings, k) for k in settings} == settings
        with pytest.raises(ValidationError):  # not a setting
            compute_motion(_make_shifted_imaging(4, height=12, width=12), method="suite2p", not_a_setting=5)

    def test_every_suite2p_option_is_a_setting(self):
        from suite2p.registration.register import default_settings

        ours = {"maxregshiftNR": "max_nonrigid_shift", "align_by_chan2": "align_by_chan"}  # suite2p's name -> ours
        saving = {"upsample_meanImg", "reg_tif", "reg_tif_chan2"}  # outputs written while applying shifts
        suite2p_options = {ours.get(k, k) for k in default_settings()["registration"]} - saving
        assert suite2p_options <= set(Suite2pRegistrationSettings.model_fields)

    def test_clip_negative(self):
        imaging = NumpyImaging(
            imaging_series=[_make_shifted_video(4, height=12, width=12) - 0.5], sampling_frequency=10.0
        )
        for clip in (False, True):
            motion = compute_motion(imaging, method="suite2p", nonrigid=False, clip_negative=clip)
            assert (RegisterSuite2PImaging(imaging, motion).epochs[0].get_series(0, 4).min() < 0) == (not clip)

    def test_trimmed_frames_keep_zero_shift_and_are_bad(self, monkeypatch):
        samples = []
        wrapper = s2p._compute_reference_wrapper
        monkeypatch.setattr(
            s2p, "_compute_reference_wrapper", lambda x, *a, **k: samples.append(x) or wrapper(x, *a, **k)
        )
        shifts = [[(1, 0), (0, 0), (1, 0), (0, 1), (1, 1), (0, 1)]]
        imaging = _make_shifted_imaging(6, height=12, width=12, shifts=shifts)
        motion = compute_motion(imaging, method="suite2p", nonrigid=False, trim_frames_start=1, trim_frames_end=2)
        assert not motion.displacements[0][[0, 4, 5]].any()
        assert motion.corrected_badframes[0][[0, 4, 5]].all()
        np.testing.assert_array_equal(samples[0], imaging.get_series(1, 4)[..., 0])  # frames 1 to 3 only

    def test_invalid_trims_and_outlier_settings_raise(self):
        imaging = _make_shifted_imaging((6, 3), height=12, width=12)
        for trim in ("trim_frames_start", "trim_frames_end"):
            with pytest.raises(ValidationError):
                compute_motion(imaging, method="suite2p", **{trim: -1})
        with pytest.raises(ValueError, match="shorter than every epoch"):
            compute_motion(imaging, method="suite2p", trim_frames_end=3)
        with pytest.raises(ValueError, match="rigid shifts only"):
            compute_motion(imaging, method="suite2p", outlier_maxregshift=0.1)

    def test_outlier_shifts_are_clipped(self):
        video = _make_shifted_video(8, height=12, width=12, shifts=[(0, 0)] * 4 + [(3, 3)] + [(0, 0)] * 3)
        imaging = NumpyImaging(imaging_series=[video * 1000], sampling_frequency=10.0)  # suite2p's reference is int16
        for limit, expected in ((None, 3), (0.1, 1)):  # 0.1 of 12 pixels: 1 pixel around the running median
            motion = compute_motion(
                imaging, method="suite2p", nonrigid=False, maxregshift=0.5, outlier_maxregshift=limit
            )
            assert motion.displacements[0][:, 0].tolist() == [[0, 0]] * 4 + [[expected, expected]] + [[0, 0]] * 3

    @pytest.mark.parametrize(
        "option",
        [
            {"subpixel": 5},
            {"two_step_registration": True},
            {"align_by_chan": 2},
            {"max_reference_iterations": 4},
        ],
    )
    def test_unsupported_options_are_rejected(self, option):
        with pytest.raises(ValueError, match="Not supported"):
            compute_motion(_make_shifted_imaging(4, height=12, width=12), method="suite2p", settings=option)

    @pytest.mark.parametrize("bidi", [{"do_bidiphase": True}, {"bidiphase": 2}])
    def test_bidiphase_is_rejected(self, bidi):
        imaging = _make_shifted_imaging(4, height=12, width=12)
        with pytest.raises(ValueError, match="suite2p#1265"):
            compute_motion(imaging, method="suite2p", settings=bidi)

    def test_smooth_sigma_time_kwarg_is_rejected(self):
        imaging = _make_shifted_imaging(
            num_frames=4, num_planes=1, height=12, width=12, shifts=[[(0, 0)] * 4], sampling_frequency=10.0
        )
        with pytest.raises(ValueError, match="smooth_sigma_time > 0"):
            compute_motion(imaging, method="suite2p", smooth_sigma_time=1)

    def test_multi_epoch_single_plane_alignment(self):
        shifts = [
            [(0, 0), (0, 0), (1, 0), (1, 0), (0, 1)],
            [(0, 0), (0, 0), (-1, 0), (-1, 0)],
        ]
        imaging = _make_shifted_imaging(
            num_frames=(5, 4), num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0
        )
        motion = compute_motion(
            imaging, method="suite2p", settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False)
        )

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

        motion = compute_motion(
            imaging, method="suite2p", settings=Suite2pRegistrationSettings(batch_size=2, nonrigid=False)
        )
        assert motion.displacements[0].shape == (4, 3, 2)
        assert motion.displacements[1].shape == (3, 3, 2)
        assert len(motion.reference) == 3

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
        return Suite2PMotion(
            imaging=imaging,
            displacements=displacements,
            reference=("ref", "masks"),
        )

    def test_construction(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg._parent is imaging
        assert reg.get_num_epochs() == 1

    def test_epoch_mismatch_raises(self, imaging):
        displacements = [np.zeros((10, 1, 2)), np.zeros((10, 1, 2))]
        motion = Suite2PMotion(imaging=imaging, displacements=displacements)
        with pytest.raises(ValueError, match="epochs"):
            RegisterSuite2PImaging(imaging, motion)

    def test_inherits_sampling_frequency(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg.sampling_frequency == imaging.sampling_frequency

    def test_inherits_shape(self, imaging, motion):
        reg = RegisterSuite2PImaging(imaging, motion)
        assert reg.shape == imaging.shape


class TestRegisterImagingSerialization:
    @pytest.mark.parametrize("nonrigid", [False, True])
    @pytest.mark.parametrize("dump", ["json", "pickle"])
    def test_reloaded_imaging_registers_identically(self, tmp_path, nonrigid, dump):
        shifts = [[(0, 0), (1, 0), (0, 1), (1, 1), (0, 0), (2, 1)]]
        # file-backed parent: an in-memory NumpyImaging does not reload from json by itself
        imaging = _make_shifted_imaging(num_frames=6, height=32, width=32, shifts=shifts).save(folder=tmp_path / "raw")
        motion = compute_motion(
            imaging,
            method="suite2p",
            settings=Suite2pRegistrationSettings(
                batch_size=3, nonrigid=nonrigid, block_size=[16, 16], clip_negative=True
            ),
        )
        registered = RegisterImaging(imaging, motion)

        file_path = tmp_path / f"registered.{dump}"
        assert registered.check_serializability(dump)
        if dump == "json":
            registered.dump_to_json(file_path)
        else:
            registered.dump_to_pickle(file_path)
        reloaded = BaseExtractor.load(file_path)

        assert isinstance(reloaded, RegisterImaging)
        assert type(reloaded.epochs[0].motion) is Suite2PMotion
        assert reloaded.epochs[0].motion.settings.clip_negative is True
        np.testing.assert_array_equal(reloaded.get_series(0, 6), registered.get_series(0, 6))


class TestRegisterSuite2PImagingEpoch:
    @pytest.fixture()
    def imaging(self):
        shifts = [[(0, 0), (0, 0), (1, 0), (0, 1), (1, 1), (0, 0)]]
        return _make_shifted_imaging(
            num_frames=6, num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0
        )

    @pytest.fixture()
    def motion(self, imaging):
        return compute_motion(
            imaging, method="suite2p", settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False)
        )

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
        diff = np.abs(result - result[0:1].repeat(6, axis=0))
        assert diff.max() <= 1
        assert np.count_nonzero(diff > 1e-6) <= 16

    def test_get_series_multiplane(self):
        shifts = [[(0, 0)] * 5, [(0, 0)] * 5, [(0, 0)] * 5]
        imaging = _make_shifted_imaging(
            num_frames=5, num_planes=3, height=12, width=12, shifts=shifts, sampling_frequency=10.0
        )
        motion = compute_motion(
            imaging, method="suite2p", settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False)
        )
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)

        result = epoch.get_series(0, 5)
        assert result.shape == (5, 12, 12, 3)
        np.testing.assert_allclose(result[..., 0], result[0:1, ..., 0].repeat(5, axis=0), atol=1e-3)
        np.testing.assert_allclose(result[..., 1], result[0:1, ..., 1].repeat(5, axis=0), atol=1e-3)
        np.testing.assert_allclose(result[..., 2], result[0:1, ..., 2].repeat(5, axis=0), atol=1e-3)

    def test_get_series_plane_slice(self):
        shifts = [[(0, 0)] * 5, [(0, 0)] * 5, [(0, 0)] * 5]
        imaging = _make_shifted_imaging(
            num_frames=5, num_planes=3, height=12, width=12, shifts=shifts, sampling_frequency=10.0
        )
        motion = compute_motion(
            imaging, method="suite2p", settings=Suite2pRegistrationSettings(batch_size=3, nonrigid=False)
        )
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)

        result = epoch.get_series(0, 5, plane_indices=slice(0, 2))

        assert result.shape == (5, 12, 12, 2)
        np.testing.assert_allclose(result[..., 0], result[0:1, ..., 0].repeat(5, axis=0), atol=1e-3)
        np.testing.assert_allclose(result[..., 1], result[0:1, ..., 1].repeat(5, axis=0), atol=1e-3)

    def test_get_series_end_frame_past_end_warns_and_clamps(self, imaging, motion, caplog):
        # Over-requested range is clamped to the recording length, with a warning.
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)
        n = epoch.get_num_samples()

        with caplog.at_level("WARNING"):
            result = epoch.get_series(0, n + 50)

        assert result.shape == (n, 12, 12, 1)
        assert any("exceeds recording length" in rec.message for rec in caplog.records)

    def test_get_series_start_past_end_raises(self, imaging, motion):
        # start_frame past the recording is a caller bug; raise instead of papering over.
        parent_epoch = imaging.epochs[0]
        epoch = RegisterSuite2PImagingEpoch(parent_epoch, motion, 0)
        n = epoch.get_num_samples()

        with pytest.raises(ValueError, match="past end_frame"):
            epoch.get_series(n + 10, n + 20)


class TestGenericEntryPoints:
    """The backend must be reachable through the generic API of issue #102."""

    @pytest.fixture()
    def imaging(self):
        shifts = [[(0, 0), (0, 0), (1, 0), (1, 0), (0, 1), (0, 1)]]
        return _make_shifted_imaging(
            num_frames=6, num_planes=1, height=12, width=12, shifts=shifts, sampling_frequency=10.0
        )

    def test_compute_motion_matches_the_suite2p_entry_point(self, imaging):
        settings = Suite2pRegistrationSettings(batch_size=3, nonrigid=False)
        generic = compute_motion(imaging, method="suite2p", settings=settings)
        specific = Suite2PMotion.compute(imaging, settings=settings)

        assert isinstance(generic, Suite2PMotion)
        np.testing.assert_allclose(generic.displacements[0], specific.displacements[0])

    def test_compute_motion_accepts_a_settings_dict(self, imaging):
        motion = compute_motion(imaging, method="suite2p", settings={"batch_size": 3, "nonrigid": False})
        assert motion.settings.batch_size == 3
        assert motion.settings.nonrigid is False

    def test_compute_motion_defaults_to_suite2p(self, imaging):
        motion = compute_motion(imaging, settings={"batch_size": 3, "nonrigid": False})
        assert isinstance(motion, Suite2PMotion)

    def test_register_imaging_dispatches_on_the_motion_object(self, imaging):
        motion = compute_motion(imaging, method="suite2p", settings={"batch_size": 3, "nonrigid": False})
        registered = RegisterImaging(imaging, motion).epochs[0].get_series(0, 6)
        np.testing.assert_allclose(registered, registered[0:1].repeat(6, axis=0), atol=1e-3)

    def test_motion_declares_its_backend_hooks(self):
        assert Suite2PMotion.method_name == "suite2p"
        assert Suite2PMotion.settings_class is Suite2pRegistrationSettings
        assert get_registration_class("suite2p") is RegisterSuite2PImagingEpoch


class TestSharedSettingsOnSuite2p:
    """The shared fields reach suite2p under the keys its own code reads."""

    @pytest.fixture()
    def imaging(self):
        shifts = [[(0, 0), (0, 0), (1, 0), (1, 0), (0, 1), (0, 1)]]
        return _make_shifted_imaging(num_frames=6, num_planes=1, height=12, width=12, shifts=shifts)

    def test_bounds_crop_estimates_the_same_rigid_shifts(self, imaging):
        settings = dict(batch_size=3, nonrigid=False)
        full = compute_motion(imaging, method="suite2p", settings=settings)
        cropped = compute_motion(imaging, method="suite2p", settings={**settings, "bounds": ((2, 10), (2, 10))})
        np.testing.assert_array_equal(cropped.displacements[0], full.displacements[0])
        assert cropped.reference[0].shape == (8, 8)
        assert cropped.yranges[0][0] == full.yranges[0][0]

    def test_bounds_with_nonrigid_is_rejected(self, imaging):
        with pytest.raises(ValueError, match="rigid registration only"):
            compute_motion(imaging, method="suite2p", settings={"nonrigid": True, "bounds": ((2, 10), (2, 10))})


def test_register_suite2p_imaging_is_the_generic_class():
    assert issubclass(RegisterSuite2PImaging, RegisterImaging)


class TestStreamingEstimation:
    """Estimation reads the movie a batch at a time instead of loading each epoch whole."""

    def test_estimation_never_reads_more_than_a_batch(self):
        shifts = [[(0, 0), (1, 0), (0, 1), (1, 1)] * 6]
        imaging = _make_shifted_imaging(num_frames=24, height=12, width=12, shifts=shifts)
        epoch = imaging.epochs[0]
        read_sizes = []
        get_series = epoch.get_series

        def spy(start, end, *args, **kwargs):
            read_sizes.append(end - start)
            return get_series(start, end, *args, **kwargs)

        epoch.get_series = spy
        motion = compute_motion(imaging, method="suite2p", batch_size=5, nonrigid=False)
        assert max(read_sizes) <= 5
        assert motion.displacements[0].shape == (24, 1, 2)

    def test_each_plane_reads_only_its_own_frames(self):
        imaging = generate_random_imaging(num_frames=12, height=12, width=12, num_planes=3, sampling_frequency=10.0)
        epoch = imaging.epochs[0]
        requested = []
        get_series = epoch.get_series

        def spy(start, end, plane_indices=None):
            requested.append(plane_indices)
            return get_series(start, end, plane_indices=plane_indices)

        epoch.get_series = spy
        compute_motion(imaging, method="suite2p", batch_size=4, nonrigid=False)
        assert all(np.arange(3)[planes].size == 1 for planes in requested)
