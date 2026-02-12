# Import base classes
from .roiextractors import (
    BaseROIExtractorImaging,
    BaseROIExtractorImagingEpoch,
    get_imaging_extractor,
)

from .suite2prois import read_suite2p_rois

# Build __all__ to include all exports
__all__ = [
    "BaseROIExtractorImaging",
    "BaseROIExtractorImagingEpoch",
    "get_imaging_extractor",
]


# Import dynamically created classes and read functions
def _setup_dynamic_imports():
    """Helper function to set up dynamic imports without polluting the module namespace."""
    from photon_mosaic.extractors.roiextractors import get_classes_and_functions_to_import

    _classes, _functions = get_classes_and_functions_to_import()

    # Add classes to module namespace and __all__
    for _cls in _classes:
        globals()[_cls.__name__] = _cls
        __all__.append(_cls.__name__)

    # Add functions to module namespace and __all__
    for _func in _functions:
        globals()[_func.__name__] = _func
        __all__.append(_func.__name__)


# Execute setup and clean up
_setup_dynamic_imports()
del _setup_dynamic_imports
