"""Test core Motion"""

import numpy as np
import pytest

from photon_mosaic.core import Motion, generate_random_imaging


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
