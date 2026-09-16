import numpy as np
import pytest

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.stack import StackPlanesImaging, stack_planes


def _imaging(num_frames, planes, seed):
    return generate_random_imaging(
        num_frames=num_frames, height=5, width=6, sampling_frequency=20.0, num_planes=planes, seed=seed
    )


def test_stack_two_single_plane_objects():
    a = _imaging(8, 1, seed=0)
    b = _imaging(8, 1, seed=1)

    joined = stack_planes(a, b)

    assert isinstance(joined, StackPlanesImaging)
    assert joined.get_num_planes() == 2
    assert tuple(joined.shape) == (5, 6, 2)

    out = joined.get_series()
    assert out.shape == (8, 5, 6, 2)
    np.testing.assert_array_equal(out[..., 0], a.get_series()[..., 0])
    np.testing.assert_array_equal(out[..., 1], b.get_series()[..., 0])


def test_stack_multi_plane_objects_and_plane_selection():
    a = _imaging(7, 2, seed=2)
    b = _imaging(7, 3, seed=3)

    joined = stack_planes(a, b)
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


def test_stack_preserves_requested_plane_order():
    """An out-of-order request must come back in the requested order.

    Planes are gathered parent by parent, so without an explicit reorder a
    request for [3, 0] would silently return [0, 3].
    """
    a = _imaging(6, 2, seed=20)
    b = _imaging(6, 2, seed=21)

    joined = stack_planes(a, b)

    out = joined.get_series(plane_ids=[3, 0])
    assert out.shape == (6, 5, 6, 2)
    np.testing.assert_array_equal(out[..., 0], b.get_series()[..., 1])
    np.testing.assert_array_equal(out[..., 1], a.get_series()[..., 0])

    # within a single parent, too
    within = joined.get_series(plane_ids=[1, 0])
    np.testing.assert_array_equal(within[..., 0], a.get_series()[..., 1])
    np.testing.assert_array_equal(within[..., 1], a.get_series()[..., 0])


def test_stack_does_not_read_parents_without_requested_planes():
    """Planes are pulled only from the parents that hold them.

    Guards the push-down: a naive implementation reads every parent in full and
    slices afterwards, which costs one full read per plane of the volume.
    """
    a = _imaging(6, 1, seed=22)
    b = _imaging(6, 1, seed=23)
    c = _imaging(6, 1, seed=24)

    joined = stack_planes(a, b, c)

    calls = []
    for index, epoch in enumerate(joined.epochs[0]._parent_epochs):
        original = epoch.get_series

        def spy(start, end, plane_indices=None, _i=index, _f=original):
            calls.append(_i)
            return _f(start, end, plane_indices)

        epoch.get_series = spy

    out = joined.get_series(plane_ids=[1])

    assert calls == [1]
    np.testing.assert_array_equal(out[..., 0], b.get_series()[..., 0])


def test_stack_empty_plane_selection_returns_empty_plane_axis():
    a = _imaging(6, 1, seed=27)
    b = _imaging(6, 1, seed=28)
    joined = stack_planes(a, b)

    out = joined.get_series(plane_ids=[])
    assert out.shape == (6, 5, 6, 0)
    assert out.dtype == a.get_dtype()


def test_stack_accepts_negative_plane_indices():
    """Negative indices count from the end, as they do on the sibling epochs."""
    a = _imaging(6, 2, seed=29)
    b = _imaging(6, 1, seed=30)
    joined = stack_planes(a, b)

    last = joined.epochs[0].get_series(0, 6, np.array([-1]))
    np.testing.assert_array_equal(last[..., 0], b.get_series()[..., 0])

    mixed = joined.epochs[0].get_series(0, 6, np.array([-1, 0]))
    np.testing.assert_array_equal(mixed[..., 0], b.get_series()[..., 0])
    np.testing.assert_array_equal(mixed[..., 1], a.get_series()[..., 0])


def test_stack_treats_a_boolean_mask_as_a_mask():
    """A boolean array selects planes; it does not name them by index."""
    a = _imaging(6, 2, seed=33)
    b = _imaging(6, 2, seed=34)
    joined = stack_planes(a, b)

    out = joined.epochs[0].get_series(0, 6, np.array([True, False, True, False]))
    assert out.shape == (6, 5, 6, 2)
    np.testing.assert_array_equal(out[..., 0], a.get_series()[..., 0])
    np.testing.assert_array_equal(out[..., 1], b.get_series()[..., 0])


def test_stack_reads_whole_parents_by_slice_not_index_array():
    """A parent giving up all its planes is asked with a slice.

    An index array would force an advanced-indexing copy, making the common
    whole-volume read more expensive than it needs to be.
    """
    a = _imaging(6, 2, seed=31)
    b = _imaging(6, 2, seed=32)
    joined = stack_planes(a, b)

    seen = []
    for epoch in joined.epochs[0]._parent_epochs:
        original = epoch.get_series

        def spy(start, end, plane_indices=None, _f=original):
            seen.append(plane_indices)
            return _f(start, end, plane_indices)

        epoch.get_series = spy

    _ = joined.get_series()

    assert all(isinstance(s, slice) for s in seen), seen


def test_stack_rejects_out_of_range_plane_index():
    a = _imaging(5, 1, seed=25)
    b = _imaging(5, 1, seed=26)
    joined = stack_planes(a, b)

    with pytest.raises(IndexError):
        _ = joined.epochs[0].get_series(0, 5, np.array([2]))

    with pytest.raises(IndexError):
        _ = joined.epochs[0].get_series(0, 5, np.array([-3]))


def test_stack_multi_epoch():
    a = _imaging((4, 6), 1, seed=4)
    b = _imaging((4, 6), 2, seed=5)

    joined = stack_planes(a, b)
    assert joined.get_num_epochs() == 2
    assert joined.get_num_planes() == 3

    out1 = joined.get_series(epoch_index=1)
    assert out1.shape == (6, 5, 6, 3)
    np.testing.assert_array_equal(out1[..., 0:1], a.get_series(epoch_index=1))
    np.testing.assert_array_equal(out1[..., 1:3], b.get_series(epoch_index=1))


def test_stack_accepts_single_sequence():
    a = _imaging(5, 1, seed=6)
    b = _imaging(5, 1, seed=7)
    joined = stack_planes([a, b])
    assert joined.get_num_planes() == 2


def test_stack_rejects_mismatched_inputs():
    a = _imaging(8, 1, seed=10)

    with pytest.raises(ValueError, match="at least two"):
        _ = stack_planes(a)

    # mismatched frame counts
    b_short = _imaging(7, 1, seed=11)
    with pytest.raises(ValueError, match="frame counts"):
        _ = stack_planes(a, b_short)

    # mismatched number of epochs
    b_epochs = _imaging((8, 3), 1, seed=12)
    with pytest.raises(ValueError, match="epochs"):
        _ = stack_planes(a, b_epochs)

    # mismatched frame shape
    b_shape = generate_random_imaging(num_frames=8, height=9, width=6, sampling_frequency=20.0, num_planes=1, seed=13)
    with pytest.raises(ValueError, match="frame shape"):
        _ = stack_planes(a, b_shape)


def test_stack_rejects_non_imaging_input():
    a = _imaging(5, 1, seed=14)
    with pytest.raises(TypeError):
        _ = stack_planes(a, "not an imaging")
