import numpy as np
import pytest

from photon_mosaic.core.generators import generate_random_imaging
from photon_mosaic.core.split import (
    SelectEpochImaging,
    SplitEpochAtFramesImaging,
    select_epochs,
    split_epoch_at_frames,
)


def test_split_epoch_with_single_index_returns_single_epoch_proxy():
    imaging = generate_random_imaging(num_frames=(4, 6, 8), height=5, width=7, sampling_frequency=20.0, seed=0)

    split_imaging = select_epochs(imaging, 1)

    assert isinstance(split_imaging, SelectEpochImaging)
    assert split_imaging.get_num_epochs() == 1
    np.testing.assert_allclose(split_imaging.get_series(), imaging.get_series(epoch_index=1))
    assert split_imaging.epochs[0] is imaging.epochs[1]


def test_split_epoch_with_multiple_indices_preserves_order():
    imaging = generate_random_imaging(num_frames=(3, 5, 7), height=4, width=6, sampling_frequency=15.0, seed=1)

    split_imaging = select_epochs(imaging, [2, 0])

    assert split_imaging.get_num_epochs() == 2
    np.testing.assert_allclose(split_imaging.get_series(epoch_index=0), imaging.get_series(epoch_index=2))
    np.testing.assert_allclose(split_imaging.get_series(epoch_index=1), imaging.get_series(epoch_index=0))
    assert split_imaging.epochs[0] is imaging.epochs[2]
    assert split_imaging.epochs[1] is imaging.epochs[0]


def test_split_epoch_raises_on_invalid_indices():
    imaging = generate_random_imaging(num_frames=(3, 5), height=4, width=6, sampling_frequency=15.0, seed=2)

    with pytest.raises(IndexError):
        _ = select_epochs(imaging, 2)

    with pytest.raises(ValueError):
        _ = select_epochs(imaging, [])

    with pytest.raises(TypeError):
        _ = select_epochs(imaging, [0, 1.5])

    with pytest.raises(TypeError):
        _ = select_epochs(imaging, True)

    with pytest.raises(TypeError):
        _ = select_epochs(imaging, [0, False])


def test_split_epoch_at_frames_produces_contiguous_sub_epochs():
    imaging = generate_random_imaging(num_frames=(20,), height=4, width=5, sampling_frequency=10.0, seed=3)
    full = imaging.get_series(epoch_index=0)

    sub = split_epoch_at_frames(imaging, 0, [7, 13])

    assert isinstance(sub, SplitEpochAtFramesImaging)
    assert sub.get_num_epochs() == 3
    assert [sub.get_num_samples(segment_index=i) for i in range(3)] == [7, 6, 7]
    np.testing.assert_array_equal(sub.get_series(epoch_index=0), full[:7])
    np.testing.assert_array_equal(sub.get_series(epoch_index=1), full[7:13])
    np.testing.assert_array_equal(sub.get_series(epoch_index=2), full[13:])


def test_split_epoch_at_frames_validates_boundaries():
    imaging = generate_random_imaging(num_frames=(10,), height=3, width=3, sampling_frequency=10.0, seed=4)

    with pytest.raises(IndexError):
        _ = split_epoch_at_frames(imaging, 1, [3])
    with pytest.raises(ValueError):
        _ = split_epoch_at_frames(imaging, 0, [0])
    with pytest.raises(ValueError):
        _ = split_epoch_at_frames(imaging, 0, [10])
    with pytest.raises(ValueError):
        _ = split_epoch_at_frames(imaging, 0, [5, 5])
    with pytest.raises(ValueError):
        _ = split_epoch_at_frames(imaging, 0, [5, 3])
