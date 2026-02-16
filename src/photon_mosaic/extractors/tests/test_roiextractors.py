import re
from unittest.mock import MagicMock, patch

import numpy as np

from photon_mosaic.core import BaseImaging
from photon_mosaic.extractors.roiextractors import (
    BaseROIExtractorImaging,
    BaseROIExtractorImagingEpoch,
    get_classes_and_functions_to_import,
)

# --------------- Helpers ---------------


def _make_mock_imaging_extractor(num_samples=100, shape=(64, 64), sampling_frequency=30.0):
    """Create a mock ImagingExtractor from roiextractors."""
    mock = MagicMock()
    mock.get_num_samples.return_value = num_samples
    mock.get_sample_shape.return_value = shape
    mock.get_sampling_frequency.return_value = sampling_frequency
    mock.get_series.return_value = np.random.default_rng(0).random((num_samples, *shape))
    return mock


# --------------- BaseROIExtractorImagingEpoch ---------------


def test_epoch_get_num_samples():
    mock_extractor = _make_mock_imaging_extractor(num_samples=200)
    epoch = BaseROIExtractorImagingEpoch(mock_extractor)

    assert epoch.get_num_samples() == 200
    mock_extractor.get_num_samples.assert_called_once()


def test_epoch_get_series_delegates_to_extractor():
    mock_extractor = _make_mock_imaging_extractor()
    epoch = BaseROIExtractorImagingEpoch(mock_extractor)

    epoch.get_series(start_frame=10, end_frame=50)
    mock_extractor.get_series.assert_called_once_with(10, 50)


def test_epoch_get_series_slices_planes_when_provided():
    mock_extractor = _make_mock_imaging_extractor()
    # Return 4D data (frames, H, W, planes)
    data_4d = np.random.default_rng(0).random((40, 64, 64, 3))
    mock_extractor.get_series.return_value = data_4d
    epoch = BaseROIExtractorImagingEpoch(mock_extractor)

    result = epoch.get_series(start_frame=10, end_frame=50, plane_indices=[0, 2])
    assert result.shape[-1] == 2
    np.testing.assert_array_equal(result[..., 0], data_4d[..., 0])
    np.testing.assert_array_equal(result[..., 1], data_4d[..., 2])


def test_epoch_get_series_no_plane_slicing_for_3d_data():
    mock_extractor = _make_mock_imaging_extractor()
    data_3d = np.random.default_rng(0).random((40, 64, 64))
    mock_extractor.get_series.return_value = data_3d
    epoch = BaseROIExtractorImagingEpoch(mock_extractor)

    result = epoch.get_series(start_frame=10, end_frame=50, plane_indices=[0])
    # 3D data should not be sliced (ndim <= 3)
    assert result.shape == data_3d.shape


# --------------- BaseROIExtractorImaging ---------------


@patch("photon_mosaic.extractors.roiextractors.imaging_extractor_dict")
def test_base_roi_extractor_imaging_init(mock_dict):
    mock_extractor = _make_mock_imaging_extractor()
    mock_class = MagicMock(return_value=mock_extractor)
    mock_dict.__getitem__ = MagicMock(return_value=mock_class)

    imaging = BaseROIExtractorImaging(imaging_name="TestExtractor", file_path="/fake/path.tif")

    assert isinstance(imaging, BaseImaging)
    assert imaging.sampling_frequency == 30.0
    assert imaging.shape[:2] == (64, 64)
    assert imaging.get_num_epochs() == 1
    assert "TestExtractor" in imaging.name
    assert imaging._kwargs["imaging_name"] == "TestExtractor"


# --------------- get_classes_and_functions_to_import ---------------


def test_get_classes_and_functions_returns_lists():
    classes, functions = get_classes_and_functions_to_import()
    assert isinstance(classes, list)
    assert isinstance(functions, list)
    assert len(classes) == len(functions)


def test_dynamic_classes_inherit_from_base():
    classes, _ = get_classes_and_functions_to_import()
    for cls in classes:
        assert issubclass(cls, BaseROIExtractorImaging)


def test_dynamic_read_functions_have_read_prefix():
    _, functions = get_classes_and_functions_to_import()
    for func in functions:
        assert func.__name__.startswith("read_"), f"Expected read_ prefix, got {func.__name__}"


def test_dynamic_read_function_names_are_snake_case():
    _, functions = get_classes_and_functions_to_import()
    snake_case_pattern = re.compile(r"^read_[a-z][a-z0-9_]*$")
    for func in functions:
        assert snake_case_pattern.match(func.__name__), f"'{func.__name__}' is not snake_case"


def test_dynamic_read_function_names_do_not_contain_extractor():
    _, functions = get_classes_and_functions_to_import()
    for func in functions:
        assert "_extractor" not in func.__name__, f"'{func.__name__}' should not contain '_extractor'"


def test_dynamic_class_names_match_extractor_dict():
    from roiextractors.extractorlist import imaging_extractor_dict

    classes, _ = get_classes_and_functions_to_import()
    class_names = {cls.__name__ for cls in classes}
    assert class_names == set(imaging_extractor_dict.keys())
