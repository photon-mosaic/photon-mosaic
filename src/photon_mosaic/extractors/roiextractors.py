"""Map ROI extractor implementations to BaseImaging and BaseRois"""

import inspect
import re

import numpy as np
from roiextractors.extractorlist import imaging_extractor_dict
from roiextractors.imagingextractor import ImagingExtractor

from photon_mosaic.core import BaseImaging, BaseImagingEpoch


class BaseROIExtractorImaging(BaseImaging):
    """Base class for ROI extractors that work with BaseImaging data."""

    def __init__(self, imaging_name: str, **kwargs):
        self.roiextractor_imaging_class = imaging_extractor_dict[imaging_name]

        roi_extractor = self.roiextractor_imaging_class(**kwargs)

        segment = BaseROIExtractorImagingEpoch(roi_extractor)
        BaseImaging.__init__(
            self,
            shape=roi_extractor.get_sample_shape(),
            sampling_frequency=roi_extractor.get_sampling_frequency(),
        )
        self.add_epoch(segment)
        self.name = f"{imaging_name} (ROIExtractors)"

        self._kwargs = {"imaging_name": imaging_name, **kwargs}


class BaseROIExtractorImagingEpoch(BaseImagingEpoch):
    """Base class for ROI extractors that work with BaseImaging data."""

    def __init__(self, roi_extractor_imaging: ImagingExtractor):
        BaseImagingEpoch.__init__(self, sampling_frequency=roi_extractor_imaging.get_sampling_frequency())  # type: ignore[call-arg]
        self.roiextractor_extractor = roi_extractor_imaging

    def get_num_samples(self):
        return self.roiextractor_extractor.get_num_samples()

    def get_series(self, start_frame=None, end_frame=None, plane_indices=None):
        series = self.roiextractor_extractor.get_series(start_frame, end_frame)
        if series.ndim == 3:
            series = series[..., np.newaxis]
        if plane_indices is not None:
            series = series[..., plane_indices]
        return series


# Dynamically create classes and read functions for all imaging extractors


def get_classes_and_functions_to_import():
    import_classes, import_functions = [], []
    for imaging_name, imaging_class in imaging_extractor_dict.items():
        # Dynamically create a class for each imaging extractor
        # The class inherits from BaseROIExtractorImaging and sets imaging_name automatically

        def make_class(extractor_name):
            """Factory function to create a class with proper closure over extractor_name."""

            class DynamicImagingClass(BaseROIExtractorImaging):
                """Dynamically generated imaging class for {extractor_name}."""

                def __init__(self, **kwargs):
                    super().__init__(imaging_name=extractor_name, **kwargs)

            # Set proper class name and module
            DynamicImagingClass.__name__ = extractor_name
            DynamicImagingClass.__qualname__ = extractor_name
            DynamicImagingClass.__doc__ = imaging_class.__doc__

            return DynamicImagingClass

        # Create the class
        imaging_class_photon_mosaic = make_class(imaging_name)
        import_classes.append(imaging_class_photon_mosaic)

        # Create a read_* function that instantiates the class
        def make_read_function(cls, extractor_cls):
            """Factory function to create a read function with proper closure and signature."""
            # Get the signature from the original extractor class
            try:
                sig = inspect.signature(extractor_cls.__init__)
                # Remove 'self' parameter
                params = [p for name, p in sig.parameters.items() if name != "self"]
            except Exception:
                # Fallback if we can't get the signature
                params = [inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)]

            def read_function(*args, **kwargs):
                """Convenience function to create an instance of {cls.__name__}.

                Returns
                -------
                {cls.__name__}
                    An instance of the imaging class.
                """
                return cls(*args, **kwargs)

            # Set the signature on the function
            read_function.__signature__ = inspect.Signature(parameters=params)

            return read_function

        # Create and add the read function
        # Convert CamelCase to snake_case, handling consecutive capitals like TIFF
        # Insert underscore before uppercase letters that are followed by lowercase
        snake_case_name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", imaging_name)
        snake_case_name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", snake_case_name)
        snake_case_name = snake_case_name.lower()
        # Remove _extractor suffix
        snake_case_name = snake_case_name.replace("_extractor", "")
        read_func_name = f"read_{snake_case_name}"
        read_func = make_read_function(imaging_class_photon_mosaic, imaging_class)
        read_func.__name__ = read_func_name
        read_func.__qualname__ = read_func_name
        read_func.__doc__ = imaging_class.__doc__

        import_functions.append(read_func)

    return import_classes, import_functions
