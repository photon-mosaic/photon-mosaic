from photon_mosaic.core import BaseImaging, BaseRois

from .segmentation_methods import segmentation_methods


def run_segmentation(imaging: BaseImaging, method: str = "suite2p", **method_params) -> BaseRois:
    """
    Run segmentation on the provided imaging data.

    Parameters
    ----------
    imaging : BaseImaging
        The imaging data to be segmented.
    method : str, optional
        The segmentation method to use. Default is "suite2p".
    **method_params : dict
        Additional parameters specific to the chosen segmentation method.

    Returns
    -------
    BaseRois
        The segmented regions of interest.
    """
    if method not in segmentation_methods:
        raise ValueError(f"Segmentation method '{method}' is not recognized.")
    segmentation_function = segmentation_methods[method]
    rois = segmentation_function(imaging, **method_params)
    rois.register_imaging(imaging)
    return rois
