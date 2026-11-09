import numpy as np
import pytest

from photon_mosaic.core import BaseImaging
from photon_mosaic.extractors import read_tiff_imaging
from photon_mosaic.sample_data import download_suite2p_google_drive
from photon_mosaic.sample_data.suite2p_sample_data import _CACHE_DIR


@pytest.mark.network
def test_suite2p_google_drive_one_file():
    imaging = download_suite2p_google_drive(num_files=1)

    assert isinstance(imaging, BaseImaging)
    assert imaging.get_num_samples() == 200


@pytest.mark.network
def test_files_are_one_lazy_movie_matching_each_file():
    imaging = download_suite2p_google_drive(num_files=2)
    first, second = (
        read_tiff_imaging(file_path=str(_CACHE_DIR / f"file_00002_{index:05d}.tif"), sampling_frequency=30)
        for index in (1, 2)
    )

    assert imaging.get_num_epochs() == 1
    assert imaging.get_num_samples() == 400
    across_boundary = np.concatenate([first.get_series(195, 200), second.get_series(0, 5)])
    np.testing.assert_array_equal(imaging.get_series(195, 205), across_boundary)
