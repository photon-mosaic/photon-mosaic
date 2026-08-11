"""Tests for the multifield_from_scanimage builder.

Only the argument validation is exercised here, since it runs before any file access and therefore
needs no ScanImage fixture. The file-reading path is covered by the local demo against real data.
"""

import pytest

from photon_mosaic.extractors import multifield_from_scanimage


def test_requires_a_file_argument():
    with pytest.raises(ValueError, match="provide either"):
        multifield_from_scanimage()


def test_rejects_empty_file_paths():
    with pytest.raises(ValueError, match="must not be empty"):
        multifield_from_scanimage(file_paths=[])


def test_rejects_invalid_timestamps_mode():
    # Validated before any file access, so no fixture is needed.
    with pytest.raises(ValueError, match="per_frame"):
        multifield_from_scanimage(file_path="dummy.tif", timestamps="nope")
