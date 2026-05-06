import numpy as np
import pytest

from photon_mosaic.core import generate_random_imaging
from photon_mosaic.preprocessing.basepreprocessor import (
    BasePreprocessor,
    BasePreprocessorEpoch,
)

# ---------------------------------------------------------------------------
# BasePreprocessor
# ---------------------------------------------------------------------------


class TestBasePreprocessor:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(num_frames=10, height=8, width=9, sampling_frequency=30.0, seed=0)

    def test_init_inherits_sampling_frequency(self, imaging):
        reg = BasePreprocessor(imaging)
        assert reg.sampling_frequency == imaging.sampling_frequency

    def test_init_inherits_shape(self, imaging):
        reg = BasePreprocessor(imaging)
        assert reg.shape == imaging.shape

    def test_init_stores_parent(self, imaging):
        reg = BasePreprocessor(imaging)
        assert reg._parent is imaging

    def test_init_custom_sampling_frequency(self, imaging):
        reg = BasePreprocessor(imaging, sampling_frequency=15.0)
        assert reg.sampling_frequency == 15.0

    def test_init_custom_dtype(self, imaging):
        # Verify construction succeeds with an explicit dtype
        BasePreprocessor(imaging, dtype=np.float32)

    def test_init_rejects_non_imaging(self):
        with pytest.raises(AssertionError, match="must be a BaseImaging"):
            BasePreprocessor("not_an_imaging_object")

    def test_init_rejects_dict(self):
        with pytest.raises(AssertionError):
            BasePreprocessor({"data": [1, 2, 3]})


# ---------------------------------------------------------------------------
# BasePreprocessor – multi-epoch
# ---------------------------------------------------------------------------


class TestBasePreprocessorMultiEpoch:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(
            num_frames=(10, 20),
            height=6,
            width=7,
            sampling_frequency=20.0,
            seed=1,
        )

    def test_parent_has_two_epochs(self, imaging):
        assert imaging.get_num_epochs() == 2

    def test_init_with_multi_epoch_imaging(self, imaging):
        reg = BasePreprocessor(imaging)
        assert reg._parent is imaging
        assert reg.sampling_frequency == imaging.sampling_frequency


# ---------------------------------------------------------------------------
# BasePreprocessorEpoch
# ---------------------------------------------------------------------------


class TestBasePreprocessorEpoch:
    @pytest.fixture()
    def imaging(self):
        return generate_random_imaging(num_frames=12, height=5, width=6, sampling_frequency=25.0, seed=2)

    @pytest.fixture()
    def epoch(self, imaging):
        parent_epoch = imaging.epochs[0]
        return BasePreprocessorEpoch(parent_epoch)

    def test_stores_parent_epoch(self, imaging, epoch):
        assert epoch.parent_imaging_epoch is imaging.epochs[0]

    def test_get_num_samples_delegates(self, imaging, epoch):
        assert epoch.get_num_samples() == imaging.get_num_frames()

    def test_get_series_raises_not_implemented(self, epoch):
        with pytest.raises(NotImplementedError):
            epoch.get_series(0, 5)
