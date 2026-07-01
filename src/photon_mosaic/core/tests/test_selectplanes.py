import numpy as np
import pytest

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.selectplanes import SelectPlanesImaging, select_planes


def test_select_planes_returns_lazy_subset_per_epoch():
    imaging = generate_random_imaging(
        num_frames=(4, 6),
        height=5,
        width=7,
        num_planes=4,
        sampling_frequency=20.0,
        seed=10,
    )

    selected = select_planes(imaging, [3, 1])

    assert isinstance(selected, SelectPlanesImaging)
    assert selected.get_num_planes() == 2
    assert selected.shape == (5, 7, 2)
    np.testing.assert_array_equal(selected.plane_ids, [3, 1])

    np.testing.assert_allclose(
        selected.get_series(epoch_index=0),
        imaging.get_series(epoch_index=0, plane_ids=[3, 1]),
    )
    np.testing.assert_allclose(
        selected.get_series(epoch_index=1),
        imaging.get_series(epoch_index=1, plane_ids=[3, 1]),
    )

    # Selection is lazy: epochs are shared with the source imaging.
    assert selected.epochs[0] is imaging.epochs[0]
    assert selected.epochs[1] is imaging.epochs[1]


def test_select_planes_allows_subselection_within_selected_proxy():
    imaging = generate_random_imaging(
        num_frames=5,
        height=4,
        width=6,
        num_planes=3,
        sampling_frequency=15.0,
        seed=11,
    )

    selected = select_planes(imaging, [2, 0])

    np.testing.assert_allclose(
        selected.get_series(plane_ids=[0]),
        imaging.get_series(plane_ids=[0]),
    )

    with pytest.raises(ValueError):
        _ = selected.get_series(plane_ids=[1])


def test_select_planes_raises_for_invalid_plane_ids():
    imaging = generate_random_imaging(
        num_frames=4,
        height=3,
        width=3,
        num_planes=2,
        sampling_frequency=10.0,
        seed=12,
    )

    with pytest.raises(ValueError):
        _ = select_planes(imaging, [2])

    with pytest.raises(ValueError):
        _ = select_planes(imaging, [])
