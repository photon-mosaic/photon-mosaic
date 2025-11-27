import numpy as np
from numpy.typing import ArrayLike
from spikeinterface.core.base import BaseExtractor, BaseSegment

from .baseimaging import BaseImaging


class BaseRois(BaseExtractor):
    """Base class for rois extractors."""

    def __init__(
        self,
        sampling_frequency: float,
        shape: tuple | list | np.ndarray,
        roi_ids: ArrayLike,
    ):
        BaseExtractor.__init__(self, roi_ids)
        self._sampling_frequency = float(sampling_frequency)
        assert len(shape) == 2, "Shape must be a tuple/list/array of length 2 (height, width)"
        self._image_shape = np.array(shape)
        self._roi_ids = np.array(roi_ids)
        self._imaging = None
        # no concept of segments for rois yet, since they are spatial only

    def __repr__(self):
        return self._repr_header()

    def _repr_header(self, display_name=True):
        """Generate text representation of the BaseRois object."""
        if display_name and self.name != self.__class__.__name__:
            name = f"{self.name} ({self.__class__.__name__})"
        else:
            name = self.__class__.__name__
        image_shape = self.image_shape
        # Format shape string based on whether data is volumetric or not
        image_shape_repr = f"{image_shape[0]} rows x {image_shape[1]} columns "
        return f"{name}:\n" f"{self.get_num_rois()} ROIs - " f"{image_shape_repr}"

    def __repr__(self):
        return self._repr_header()

    def _repr_html_(self, display_name=True):
        common_style = "margin-left: 10px;"
        border_style = "border:1px solid #ddd; padding:10px;"

        html_header = f"<div style='{border_style}'><strong>{self._repr_header(display_name)}</strong></div>"

        html_roi_ids = f"<details style='{common_style}'>  <summary><strong>ROI IDs</strong></summary><ul>"
        html_roi_ids += f"{list(self.roi_ids)} </details>"

        html_extra = self._get_common_repr_html(common_style)

        html_repr = html_header + html_roi_ids + html_extra
        return html_repr

    @property
    def imaging(self):
        """Get the registered imaging.

        Returns
        -------
        BaseImaging | None
            The registered imaging or None if not registered.
        """
        return self._imaging

    def has_imaging(self) -> bool:
        """Check if an imaging is registered.

        Returns
        -------
        bool
            True if an imaging is registered, False otherwise.
        """
        return self._imaging is not None

    @property
    def image_shape(self):
        """Get the shape of the images (height, width).

        Returns
        -------
        tuple
            The shape of the images as (height, width).
        """
        return self._image_shape

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def roi_ids(self) -> ArrayLike:
        """Get the ROI IDs.

        Returns
        -------
        ArrayLike
            The ROI IDs.
        """
        return self._roi_ids

    def get_num_rois(self) -> int:
        """Get the total number of ROIs.

        Returns
        -------
        int
            The total number of ROIs.
        """
        return len(self.roi_ids)

    def get_roi_image_masks(self, roi_ids: list[int | str] | None = None) -> np.ndarray:
        """Get the image mask for a specific ROI.

        Parameters
        ----------
        roi_ids : list[int | str] | None
            The IDs of the ROIs.

        Returns
        -------
        np.ndarray
            The image mask for the specified ROIs.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_roi_pixel_masks(self, roi_ids: list[int | str] | None = None) -> np.ndarray:
        """Get the pixel coordinates for a specific ROI.

        Parameters
        ----------
        roi_ids : list[int | str] | None
            The IDs of the ROIs.

        Returns
        -------
        np.ndarray
            The pixel coordinates for the specified ROIs.
        """
        if roi_ids is None:
            roi_ids = self.roi_ids

        # Get pixel masks from representations
        pixel_masks = []
        image_masks = self.get_roi_image_masks(roi_ids)
        for img_mask in image_masks:
            # 2D case
            y_coords, x_coords = np.nonzero(img_mask)
            weights = img_mask[y_coords, x_coords]
            pixel_masks.append(np.column_stack([y_coords, x_coords, weights]))

        return pixel_masks

    def select_rois(self, roi_ids: ArrayLike) -> "SelectRois":
        """Select a subset of ROIs.

        Parameters
        ----------
        roi_ids : ArrayLike
            The IDs of the ROIs to select.

        Returns
        -------
        SelectRois
            A new BaseRois object containing only the selected ROIs.
        """
        return SelectRois(self, roi_ids)

    def register_imaging(self, imaging: BaseImaging):
        """
        Register an imaging to the sorting. If the sorting and imaging both contain
        time information, the imaging's time information will be used.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging with the same number of segments as current sorting.
            Assigned to self._imaging.
        """
        assert np.isclose(
            self.sampling_frequency, imaging.sampling_frequency, atol=0.1
        ), "The imaging has a different sampling frequency than the ROIs!"
        self._imaging = imaging


class SelectRois(BaseRois):
    """Class to select a subset of ROIs from an existing BaseRois object."""

    def __init__(self, rois: BaseRois, roi_ids: ArrayLike):
        self._source_rois = rois
        self._selected_roi_ids = np.array(roi_ids)

        # Validate selected ROI IDs
        for roi_id in self._selected_roi_ids:
            if roi_id not in rois.roi_ids:
                raise ValueError(f"ROI ID {roi_id} not found in source ROIs.")

        BaseRois.__init__(
            self,
            sampling_frequency=rois.sampling_frequency,
            shape=rois.image_shape,
            roi_ids=self._selected_roi_ids,
        )
        rois.copy_metadata(self, only_main=False, ids=self.roi_ids)
        self._parent = rois

        if rois.has_imaging():
            self.register_imaging(rois._imaging)

        self._kwargs = dict(rois=rois, roi_ids=roi_ids)

    def get_roi_image_masks(self, roi_ids: list[int | str] | None = None) -> np.ndarray:
        if roi_ids is None:
            roi_ids = self.roi_ids

        # Get masks from source rois
        source_masks = self._source_rois.get_roi_image_masks(roi_ids)
        return source_masks
