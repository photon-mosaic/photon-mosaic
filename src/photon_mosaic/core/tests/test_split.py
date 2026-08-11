import numpy as np
import pytest

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.split import SelectEpochImaging, split_epochs


def test_split_epoch_with_single_index_returns_single_epoch_proxy():
    imaging = generate_random_imaging(num_frames=(4, 6, 8), height=5, width=7, sampling_frequency=20.0, seed=0)

    split_imaging = split_epochs(imaging, 1)

    assert isinstance(split_imaging, SelectEpochImaging)
    assert split_imaging.get_num_epochs() == 1
    np.testing.assert_allclose(split_imaging.get_series(), imaging.get_series(epoch_index=1))
    assert split_imaging.epochs[0] is imaging.epochs[1]


def test_split_epoch_with_multiple_indices_preserves_order():
    imaging = generate_random_imaging(num_frames=(3, 5, 7), height=4, width=6, sampling_frequency=15.0, seed=1)

    split_imaging = split_epochs(imaging, [2, 0])

    assert split_imaging.get_num_epochs() == 2
    np.testing.assert_allclose(split_imaging.get_series(epoch_index=0), imaging.get_series(epoch_index=2))
    np.testing.assert_allclose(split_imaging.get_series(epoch_index=1), imaging.get_series(epoch_index=0))
    assert split_imaging.epochs[0] is imaging.epochs[2]
    assert split_imaging.epochs[1] is imaging.epochs[0]


def test_split_repeatly():
    imaging = generate_random_imaging(num_frames=(4, 6, 8, 10), height=5, width=7, sampling_frequency=20.0, seed=0)

    split_imaging = split_epochs(imaging, [1, 2, 3])  # take last three epochs
    split_split_imaging = split_epochs(split_imaging, [1])  # take middle of last three epochs

    assert split_split_imaging.get_num_epochs() == 1
    np.testing.assert_allclose(split_split_imaging.get_series(epoch_index=0), split_imaging.get_series(epoch_index=1))
    np.testing.assert_allclose(split_split_imaging.get_series(epoch_index=0), imaging.get_series(epoch_index=2))
    assert split_split_imaging.epochs[0] is split_imaging.epochs[1]
    assert split_split_imaging.epochs[0] is imaging.epochs[2]


def test_split_epoch_raises_on_invalid_indices():
    imaging = generate_random_imaging(num_frames=(3, 5), height=4, width=6, sampling_frequency=15.0, seed=2)

    with pytest.raises(IndexError):
        _ = split_epochs(imaging, 2)

    with pytest.raises(ValueError):
        _ = split_epochs(imaging, [])

    with pytest.raises(TypeError):
        _ = split_epochs(imaging, [0, 1.5])

    with pytest.raises(TypeError):
        _ = split_epochs(imaging, True)

    with pytest.raises(TypeError):
        _ = split_epochs(imaging, [0, False])
