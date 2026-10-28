import photon_mosaic.extractors as extractors_module
from photon_mosaic.extractors.roiextractors import get_classes_and_functions_to_import


def test_dynamic_classes_and_functions_in_module():
    classes, functions = get_classes_and_functions_to_import()

    for cls in classes:
        assert hasattr(extractors_module, cls.__name__), f"{cls.__name__} not found in extractors module"
        assert cls.__name__ in extractors_module.__all__

    for func in functions:
        assert hasattr(extractors_module, func.__name__), f"{func.__name__} not found in extractors module"
        assert func.__name__ in extractors_module.__all__
