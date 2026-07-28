import pytest

from photon_mosaic.core import BaseImaging
from photon_mosaic.sample_data import download_suite2p_google_drive


@pytest.mark.network
def test_suite2p_google_drive_one_file():
    imagings = download_suite2p_google_drive(num_files=1)

    assert len(imagings) == 1
    assert isinstance(imagings[0], BaseImaging)
