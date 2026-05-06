import photon_mosaic.extractors as extractors_module
from photon_mosaic.extractors.roiextractors import (
    BaseROIExtractorImaging,
    BaseROIExtractorImagingEpoch,
    get_classes_and_functions_to_import,
)


def test_base_classes_exported():
    assert hasattr(extractors_module, "BaseROIExtractorImaging")
    assert hasattr(extractors_module, "BaseROIExtractorImagingEpoch")
    assert extractors_module.BaseROIExtractorImaging is BaseROIExtractorImaging
    assert extractors_module.BaseROIExtractorImagingEpoch is BaseROIExtractorImagingEpoch


def test_all_contains_base_exports():
    assert "BaseROIExtractorImaging" in extractors_module.__all__
    assert "BaseROIExtractorImagingEpoch" in extractors_module.__all__


def test_dynamic_classes_and_functions_in_module():
    classes, functions = get_classes_and_functions_to_import()

    for cls in classes:
        assert hasattr(extractors_module, cls.__name__), f"{cls.__name__} not found in extractors module"
        assert cls.__name__ in extractors_module.__all__

    for func in functions:
        assert hasattr(extractors_module, func.__name__), f"{func.__name__} not found in extractors module"
        assert func.__name__ in extractors_module.__all__
