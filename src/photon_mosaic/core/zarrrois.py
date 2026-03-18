"""Zarr format for ROI data.

Classes
-------
ZarrRois
    ROI extractor backed by a zarr group on disk.

Functions
---------
save_rois_to_zarr
    Save ROI masks and metadata to a zarr group.
"""

import numpy as np

from .baserois import BaseRois


class ZarrRois(BaseRois):
    """ROI extractor backed by a zarr group on disk.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the zarr store.
    zarr_group_name : str, default: "rois"
        Name of the group within the store containing ROI data.
    storage_options : dict | None, default: None
        fsspec storage options for remote zarr stores.
    """

    def __init__(self, zarr_path, zarr_group_name: str = "rois", storage_options: dict | None = None):
        from spikeinterface.core.zarrextractors import super_zarr_open

        zarr_root = super_zarr_open(str(zarr_path), mode="r", storage_options=storage_options or {})
        rois_group = zarr_root[zarr_group_name]

        roi_ids = np.array(rois_group["roi_ids"])
        sampling_frequency = rois_group.attrs["sampling_frequency"]
        shape = tuple(rois_group.attrs["shape"])

        BaseRois.__init__(
            self,
            sampling_frequency=sampling_frequency,
            shape=shape,
            roi_ids=roi_ids,
        )

        self._rois_group = rois_group

        # Load properties
        if "properties" in rois_group:
            for key in rois_group["properties"].keys():
                values = np.array(rois_group["properties"][key])
                self.set_property(key, values)

        # Load annotations
        if "annotations" in rois_group.attrs:
            self.annotate(**rois_group.attrs["annotations"])

        self._kwargs = dict(
            zarr_path=str(zarr_path),
            zarr_group_name=zarr_group_name,
            storage_options=storage_options,
        )

    def get_roi_image_masks(self, roi_ids=None):
        masks = np.array(self._rois_group["roi_image_masks"])
        if roi_ids is None:
            return masks
        roi_indices = self.ids_to_indices(roi_ids)
        return masks[roi_indices]


def save_rois_to_zarr(rois: BaseRois, zarr_group, saving_options: dict | None = None) -> None:
    """Save ROI masks and metadata to a zarr group.

    Parameters
    ----------
    rois : BaseRois
        The ROIs object to save.
    zarr_group : zarr.hierarchy.Group
        The zarr group to write to.
    saving_options : dict | None
        Additional zarr dataset creation options (e.g., compressor).
    """
    saving_options = saving_options or {}

    image_masks = rois.get_roi_image_masks()
    zarr_group.create_dataset("roi_image_masks", data=np.asarray(image_masks), **saving_options)

    roi_ids = np.array(rois.roi_ids)
    if roi_ids.dtype.kind == "U":
        import numcodecs

        zarr_group.create_dataset("roi_ids", data=roi_ids.astype(object), object_codec=numcodecs.JSON())
    else:
        zarr_group.create_dataset("roi_ids", data=roi_ids, compressor=None)

    zarr_group.attrs["sampling_frequency"] = float(rois.sampling_frequency)
    zarr_group.attrs["shape"] = list(rois.shape)

    # Save properties
    prop_group = zarr_group.create_group("properties")
    for key in rois.get_property_keys():
        values = rois.get_property(key)
        if values.dtype.kind == "O":
            continue  # skip non-serializable object-dtype properties
        prop_group.create_dataset(key, data=values, compressor=None)

    # Save annotations
    annotations = rois.get_annotation_keys()
    if annotations:
        ann_dict = {key: rois.get_annotation(key) for key in annotations}
        zarr_group.attrs["annotations"] = ann_dict
