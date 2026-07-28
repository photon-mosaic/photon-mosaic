import json
from pathlib import Path

import numpy as np
import sparse
from numpy.typing import ArrayLike
from spikeinterface.core.base import BaseExtractor

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
        if len(shape) == 2:
            shape = (shape[0], shape[1], 1)
        self._shape = tuple(shape)
        self._num_planes = shape[2]
        self._roi_ids = np.array(roi_ids)
        self._imaging: BaseImaging | None = None
        # no concept of epochs for rois, since they are spatial only

    def __repr__(self):
        return self._repr_header()

    def _repr_header(self, display_name=True):
        """Generate text representation of the BaseRois object."""
        if display_name and self.name != self.__class__.__name__:
            name = f"{self.name} ({self.__class__.__name__})"
        else:
            name = self.__class__.__name__
        shape = self._shape
        # Format shape string based on whether data is volumetric or not
        shape_repr = f"{shape[0]} rows x {shape[1]} columns "
        return f"{name}:\n{self.get_num_rois()} ROIs - {shape_repr}"

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
    def shape(self):
        """Get the shape of the ROIs (height, width, planes).

        Returns
        -------
        tuple
            The shape of the ROIs as (height, width, planes).
        """
        return self._shape

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def roi_ids(self) -> np.ndarray:
        """Get the ROI IDs.

        Returns
        -------
        np.ndarray
            The ROI IDs.
        """
        return self._roi_ids

    def get_num_planes(self) -> int:
        """Get the number of planes.

        Returns
        -------
        int
            The number of planes.
        """
        return self._num_planes

    @property
    def num_planes(self) -> int:
        """Number of planes for ROI masks.

        This is a convenience alias for :meth:`get_num_planes`.
        """
        return self.get_num_planes()

    def get_num_rois(self) -> int:
        """Get the total number of ROIs.

        Returns
        -------
        int
            The total number of ROIs.
        """
        return len(self.roi_ids)

    def get_roi_image_masks(
        self, roi_ids: list[int | str] | None = None
    ) -> np.ndarray | sparse.SparseArray:  # pragma: no cover
        """Get the image mask for a specific ROI. The image mask can be binary or weighted and 2D (single plane)
        or 3D (multi-plane).

        Subclasses may return either a dense ``np.ndarray`` or a sparse
        `pydata/sparse <https://sparse.pydata.org/>`_ array (e.g. ``sparse.GCXS``). Each ROI
        typically occupies only a tiny fraction of the full imaging volume, so a sparse
        representation avoids the out-of-memory errors a dense array would cause once there
        are many ROIs over a large (e.g. volumetric) field of view -- see photon-mosaic#103.
        Callers that need actual pixel values (e.g. for plotting) should call ``.todense()``
        explicitly rather than assume a dense array.

        Parameters
        ----------
        roi_ids : list[int | str] | None
            The IDs of the ROIs.

        Returns
        -------
        np.ndarray | sparse.SparseArray
            The image mask for the specified ROIs.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_roi_pixel_masks(self, roi_ids: list[int | str] | None = None) -> list[np.ndarray]:
        """Get the pixel coordinates for a specific ROI.

        Parameters
        ----------
        roi_ids : list[int | str] | None
            The IDs of the ROIs.

        Returns
        -------
        np.ndarray
            The pixel coordinates for the specified ROIs (y, x, [z,] weight).
        """
        if roi_ids is None:
            roi_ids = self.roi_ids.tolist()

        # Get pixel masks from representations
        pixel_masks = []
        image_masks = self.get_roi_image_masks(roi_ids)
        for img_mask in image_masks:
            if isinstance(img_mask, sparse.SparseArray):
                # .coords/.data give pixel coordinates and weights directly -- no need to
                # densify or scan for nonzero entries. coords has shape (ndim, nnz); unpacking
                # it gives one array per spatial dimension (y, x, [z]).
                coo = img_mask.tocoo()
                coords, weights = coo.coords, coo.data
            else:
                # Dense case (2D or 3D): np.nonzero returns one array per dimension.
                coords = np.nonzero(img_mask)
                weights = img_mask[coords]
            pixel_masks.append(np.column_stack([*coords, weights]))

        return pixel_masks

    def select_rois(self, roi_ids: ArrayLike) -> "BaseRois":
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
        from .selectrois import SelectRois

        return SelectRois(self, roi_ids)

    def register_imaging(self, imaging: BaseImaging):
        """
        Register an imaging to the ROIs. If the ROIs and imaging both contain
        time information, the imaging's time information will be used.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging with the same number of planes as the ROIs.
            Assigned to self._imaging.
        """
        assert (
            imaging.get_num_planes() == self.get_num_planes()
        ), "The imaging has a different number of planes than the ROIs!"
        assert np.isclose(
            self.sampling_frequency, imaging.sampling_frequency, atol=0.1
        ), "The imaging has a different sampling frequency than the ROIs!"
        self._imaging = imaging

    def save(self, **kwargs) -> "BaseExtractor":
        """Save a BaseRois object to disk.

        Overrides ``BaseExtractor.save()`` for ``format="zarr"``: the base
        implementation's ``save_to_zarr()`` reloads the saved data with
        spikeinterface's ``read_zarr()``, which expects a recording/sorting zarr
        layout (``channel_ids``/``unit_ids`` at the root) and doesn't understand
        the ROI zarr layout written by ``_save_zarr()``. Dispatching straight to
        ``_save_zarr()`` here mirrors how ``format="binary"`` already reaches
        ``_save_binary()``.
        """
        if kwargs.get("format") == "zarr":
            save_kwargs = {k: v for k, v in kwargs.items() if k != "format"}
            return self._save_zarr(**save_kwargs)
        return super().save(**kwargs)

    def _save(self, format="binary", **save_kwargs):
        """Save ROIs to disk as the binary format.

        Called internally by ``BaseExtractor.save_to_folder()``. Zarr saving is
        dispatched directly to ``_save_zarr()`` by ``BaseRois.save()`` instead of
        going through here (see its docstring).

        Parameters
        ----------
        format : str, default: "binary"
            Must be ``"binary"``.
        **save_kwargs
            Must include ``folder`` (str or Path).

        Returns
        -------
        BinaryFolderRois
            The on-disk representation.
        """
        if format != "binary":
            raise ValueError(
                f"format {format!r} not supported by _save(); use 'binary' ('zarr' goes through _save_zarr())"
            )
        return self._save_binary(**save_kwargs)

    def _save_binary(self, **save_kwargs):
        from .binaryrois import BinaryFolderRois, BinaryRois

        folder = Path(save_kwargs["folder"])
        folder.mkdir(parents=True, exist_ok=True)

        image_masks = self.get_roi_image_masks()
        if isinstance(image_masks, sparse.SparseArray):
            # np.save doesn't understand sparse arrays; sparse.save_npz stores the
            # data/coords/shape directly instead of densifying.
            mask_file = folder / "roi_image_masks.npz"
            sparse.save_npz(mask_file, image_masks)
        else:
            mask_file = folder / "roi_image_masks.npy"
            np.save(mask_file, image_masks)
        np.save(folder / "roi_ids.npy", np.array(self.roi_ids))

        metadata = dict(
            sampling_frequency=float(self.sampling_frequency),
            shape=list(self.shape),
        )
        with open(folder / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        binary_rois = BinaryRois(
            file_path=mask_file,
            sampling_frequency=self.sampling_frequency,
            roi_ids=self.roi_ids,
            shape=self.shape,
        )
        binary_rois.dump(folder / "binary.json", relative_to=folder)

        cached = BinaryFolderRois(folder_path=folder)

        return cached

    def _save_zarr(self, **save_kwargs):
        import zarr

        from .zarrrois import ZarrRois, save_rois_to_zarr

        zarr_path = save_kwargs["zarr_path"]
        saving_options = save_kwargs.get("saving_options", None)
        storage_options = save_kwargs.get("storage_options", None)

        zarr_root = zarr.open(str(zarr_path), mode="w", storage_options=storage_options)
        rois_group = zarr_root.create_group("rois")
        save_rois_to_zarr(self, rois_group, saving_options=saving_options)
        zarr.consolidate_metadata(zarr_root.store)

        return ZarrRois(zarr_path, storage_options=storage_options)
