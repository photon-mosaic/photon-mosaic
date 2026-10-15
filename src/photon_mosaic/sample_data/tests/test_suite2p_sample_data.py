import numpy as np
import pytest

from photon_mosaic.core import BaseImaging
from photon_mosaic.sample_data import download_suite2p_google_drive


@pytest.mark.network
def test_suite2p_google_drive_one_file():
    imaging = download_suite2p_google_drive(num_files=1)

    assert isinstance(imaging, BaseImaging)
    assert imaging.get_num_samples() == 200


@pytest.mark.network
def test_files_are_one_lazy_movie_matching_the_eager_files():
    lazy = download_suite2p_google_drive(num_files=2)
    eager = download_suite2p_google_drive(num_files=2, lazy=False)

    assert lazy.get_num_epochs() == 1
    assert lazy.get_num_samples() == 400
    assert len(eager) == 2 and all(isinstance(imaging, BaseImaging) for imaging in eager)
    across_boundary = np.concatenate([eager[0].get_series(195, 200), eager[1].get_series(0, 5)])
    np.testing.assert_array_equal(lazy.get_series(195, 205), across_boundary)
