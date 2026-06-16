import numpy as np
import pytest

from photon_mosaic.core.concatenate import ConcatenatePlanesImaging, concatenate_planes
from photon_mosaic.core.generators import generate_random_imaging


def _imaging(num_frames, planes, seed):
    return generate_random_imaging(
        num_frames=num_frames, height=5, width=6, sampling_frequency=20.0, num_planes=planes, seed=seed
    )


def test_concatenate_two_single_plane_objects():
    a = _imaging(8, 1, seed=0)
    b = _imaging(8, 1, seed=1)

    joined = concatenate_planes(a, b)

    assert isinstance(joined, ConcatenatePlanesImaging)
    assert joined.get_num_planes() == 2
    assert tuple(joined.shape) == (5, 6, 2)

    out = joined.get_series()
    assert out.shape == (8, 5, 6, 2)
    np.testing.assert_array_equal(out[..., 0], a.get_series()[..., 0])
    np.testing.assert_array_equal(out[..., 1], b.get_series()[..., 0])


def test_concatenate_multi_plane_objects_and_plane_selection():
    a = _imaging(7, 2, seed=2)
    b = _imaging(7, 3, seed=3)

    joined = concatenate_planes(a, b)
    assert joined.get_num_planes() == 5

    full = joined.get_series()
    assert full.shape == (7, 5, 6, 5)
    np.testing.assert_array_equal(full[..., 0:2], a.get_series())
    np.testing.assert_array_equal(full[..., 2:5], b.get_series())

    # plane selection crossing the boundary between the two inputs
    sel = joined.get_series(plane_ids=[1, 3])
    assert sel.shape == (7, 5, 6, 2)
    np.testing.assert_array_equal(sel[..., 0], a.get_series()[..., 1])
    np.testing.assert_array_equal(sel[..., 1], b.get_series()[..., 1])


def test_concatenate_multi_epoch():
    a = _imaging((4, 6), 1, seed=4)
    b = _imaging((4, 6), 2, seed=5)

    joined = concatenate_planes(a, b)
    assert joined.get_num_epochs() == 2
    assert joined.get_num_planes() == 3

    out1 = joined.get_series(epoch_index=1)
    assert out1.shape == (6, 5, 6, 3)
    np.testing.assert_array_equal(out1[..., 0:1], a.get_series(epoch_index=1))
    np.testing.assert_array_equal(out1[..., 1:3], b.get_series(epoch_index=1))


def test_concatenate_accepts_single_sequence():
    a = _imaging(5, 1, seed=6)
    b = _imaging(5, 1, seed=7)
    joined = concatenate_planes([a, b])
    assert joined.get_num_planes() == 2


def test_concatenate_is_registered_only_when_all_inputs_are():
    a = _imaging(5, 1, seed=8)
    b = _imaging(5, 1, seed=9)

    assert concatenate_planes(a, b).is_registered is False

    a.is_registered = True
    assert concatenate_planes(a, b).is_registered is False  # b still unregistered

    b.is_registered = True
    assert concatenate_planes(a, b).is_registered is True


def test_concatenate_rejects_mismatched_inputs():
    a = _imaging(8, 1, seed=10)

    with pytest.raises(ValueError, match="at least two"):
        _ = concatenate_planes(a)

    # mismatched frame counts
    b_short = _imaging(7, 1, seed=11)
    with pytest.raises(ValueError, match="frame counts"):
        _ = concatenate_planes(a, b_short)

    # mismatched number of epochs
    b_epochs = _imaging((8, 3), 1, seed=12)
    with pytest.raises(ValueError, match="epochs"):
        _ = concatenate_planes(a, b_epochs)

    # mismatched frame shape
    b_shape = generate_random_imaging(num_frames=8, height=9, width=6, sampling_frequency=20.0, num_planes=1, seed=13)
    with pytest.raises(ValueError, match="frame shape"):
        _ = concatenate_planes(a, b_shape)


def test_concatenate_rejects_non_imaging_input():
    a = _imaging(5, 1, seed=14)
    with pytest.raises(TypeError):
        _ = concatenate_planes(a, "not an imaging")
