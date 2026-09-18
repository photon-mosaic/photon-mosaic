"""Test core Motion"""

import numpy as np
import pytest
from pydantic_settings import BaseSettings

from photon_mosaic.core import Motion, compute_motion, generate_random_imaging, register_motion_class
from photon_mosaic.core.motion import (
    _builtin_motion_modules,
    _get_motion_class,
    _registered_motion_classes,
    coerce_settings,
)


class TestMotion:
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
        return Motion(
            imaging=imaging_single,
            displacements=displacements,
            reference=("ref_plane_0",),
            metadata={"backend": "test"},
        )

    def test_attributes(self, imaging_single, motion_single):
        assert motion_single.imaging is imaging_single
        assert motion_single.num_epochs == 1
        assert motion_single.reference == ("ref_plane_0",)
        assert motion_single.metadata == {"backend": "test"}

    def test_metadata_defaults_to_empty_dict(self, imaging_single):
        disps = [np.zeros((5, 1, 2))]
        m = Motion(imaging_single, disps)
        assert m.metadata == {}

    def test_num_epochs_matches_displacements(self, imaging_single):
        disps = [np.zeros((5, 1, 2)), np.zeros((8, 1, 2))]
        m = Motion(imaging_single, disps)
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
        motion = Motion(imaging_multi, disps)
        single_val = motion.get_displacement_at_frames(0, plane_index=1)
        assert single_val.shape == (2,)
        np.testing.assert_array_equal(single_val, disps[0][0, 1])

        multi_val = motion.get_displacement_at_frames(np.array([2, 5]), plane_index=2)
        assert multi_val.shape == (2, 2)
        np.testing.assert_array_equal(multi_val, disps[0][[2, 5], 2])


class DummySettings(BaseSettings):
    value: int = 1


class DummyMotion(Motion):
    """Backend-less Motion subclass used to exercise the registry."""

    method_name = "dummy_test_backend"
    settings_class = DummySettings

    @classmethod
    def _compute(cls, imaging, *, settings=None, badframes=None, **params):
        motion = cls(imaging=imaging, displacements=[np.zeros((3, 1, 2))])
        motion.metadata = {"settings": settings, "badframes": badframes, "params": params}
        return motion


@pytest.fixture()
def registered_dummy():
    register_motion_class(DummyMotion)
    yield DummyMotion
    _registered_motion_classes.pop(DummyMotion.method_name, None)


class TestMotionRegistry:
    def test_suite2p_is_resolved_lazily(self):
        from photon_mosaic.preprocessing.suite2p_registration import Suite2PMotion

        assert _get_motion_class("suite2p") is Suite2PMotion

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown motion correction method"):
            _get_motion_class("not_a_backend")

    def test_register_and_resolve(self, registered_dummy):
        assert _get_motion_class("dummy_test_backend") is DummyMotion

    def test_register_requires_method_name(self):
        class Nameless(Motion):
            pass

        with pytest.raises(ValueError, match="method_name"):
            register_motion_class(Nameless)

    def test_missing_optional_dependency_reports_the_method(self, monkeypatch):
        monkeypatch.setitem(_builtin_motion_modules, "phantom", "photon_mosaic.preprocessing.not_a_module")
        with pytest.raises(ImportError, match="phantom"):
            _get_motion_class("phantom")


class TestMotionCompute:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(num_frames=3, height=8, width=9, num_planes=1, sampling_frequency=30.0, seed=3)

    def test_dispatches_on_method(self, imaging, registered_dummy):
        motion = compute_motion(imaging, method="dummy_test_backend")
        assert isinstance(motion, DummyMotion)

    def test_settings_are_coerced_and_params_forwarded(self, imaging, registered_dummy):
        motion = compute_motion(imaging, method="dummy_test_backend", settings={"value": 7}, extra=3)
        assert isinstance(motion.metadata["settings"], DummySettings)
        assert motion.metadata["settings"].value == 7
        assert motion.metadata["params"] == {"extra": 3}

    def test_subclass_compute_needs_no_method(self, imaging, registered_dummy):
        motion = DummyMotion.compute(imaging)
        assert isinstance(motion, DummyMotion)

    def test_subclass_rejects_a_foreign_method(self, imaging, registered_dummy):
        with pytest.raises(ValueError, match="implements method"):
            DummyMotion.compute(imaging, method="suite2p")

    def test_non_string_method_points_at_settings(self, imaging, registered_dummy):
        settings = DummySettings(value=2)
        with pytest.raises(TypeError, match="settings="):
            compute_motion(imaging, settings)
        with pytest.raises(TypeError, match="settings="):
            DummyMotion.compute(imaging, settings)

    def test_base_compute_is_not_implemented(self, imaging):
        class Bare(Motion):
            method_name = "bare_test_backend"

        with pytest.raises(NotImplementedError):
            Bare.compute(imaging)


class TestCoerceSettings:
    def test_none_gives_defaults(self):
        assert coerce_settings(None, DummySettings).value == 1

    def test_dict_is_validated(self):
        assert coerce_settings({"value": 5}, DummySettings).value == 5

    def test_instance_passes_through(self):
        settings = DummySettings(value=9)
        assert coerce_settings(settings, DummySettings) is settings

    def test_no_settings_class_passes_through(self):
        assert coerce_settings("anything", None) == "anything"

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            coerce_settings(3.14, DummySettings)
