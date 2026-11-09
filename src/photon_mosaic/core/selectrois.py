import numpy as np
from numpy.typing import ArrayLike

from .baserois import BaseRois


class SelectRois(BaseRois):
    """Class to select a subset of ROIs from an existing BaseRois object."""

    def __init__(self, rois: BaseRois, roi_ids: ArrayLike):
        self._source_rois = rois
        self._selected_roi_ids = np.array(roi_ids)

        # Validate selected ROI IDs
        source_roi_ids = rois.roi_ids.tolist()
        for roi_id in self._selected_roi_ids:
            if roi_id not in source_roi_ids:
                raise ValueError(f"ROI ID {roi_id} not found in source ROIs.")

        BaseRois.__init__(
            self,
            sampling_frequency=rois.sampling_frequency,
            shape=rois.shape,
            roi_ids=self._selected_roi_ids,
        )
        rois.copy_metadata(self, only_main=False, ids=self.roi_ids)
        self._parent = rois

        if rois._imaging is not None:
            self.register_imaging(rois._imaging)

        self._kwargs = dict(rois=rois, roi_ids=roi_ids)

    def get_roi_image_masks(self, roi_ids: list[int | str] | None = None) -> np.ndarray:
        if roi_ids is None:
            roi_ids = self.roi_ids.tolist()

        # Get masks from source rois
        source_masks = self._source_rois.get_roi_image_masks(roi_ids)
        return source_masks
