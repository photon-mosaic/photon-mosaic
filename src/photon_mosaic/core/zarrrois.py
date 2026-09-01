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
import sparse

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
        if self._rois_group.attrs.get("roi_image_masks_sparse", False):
            return self._get_sparse_roi_image_masks(roi_ids)

        # Index the still-lazy zarr array *before* materialising, so a request for a few
        # ROIs only reads their own chunks (each ROI is its own chunk, see save_rois_to_zarr)
        # instead of loading every ROI's mask just to throw most of them away.
        roi_image_masks = self._rois_group["roi_image_masks"]
        if roi_ids is None:
            return np.array(roi_image_masks)
        roi_indices = self.ids_to_indices(roi_ids)
        return np.array(roi_image_masks[roi_indices])

    def _get_sparse_roi_image_masks(self, roi_ids):
        """Reconstruct a GCXS array from its on-disk (indptr, indices, data) components.

        These are exactly the CSR-style arrays `sparse.GCXS(compressed_axes=(0,))` already
        keeps in memory -- `indptr` marks each ROI's own slice of the flat `indices`/`data`
        arrays. Reading a subset of ROIs therefore only needs `indptr` (always loaded, tiny:
        one int per ROI) plus the requested ROIs' own slices of `indices`/`data`, read from
        the still-lazy zarr arrays -- never the full arrays, unlike reconstructing via COO.
        """
        shape = tuple(self._rois_group.attrs["roi_image_masks_shape"])
        indptr = np.array(self._rois_group["roi_image_masks_indptr"])
        indices = self._rois_group["roi_image_masks_indices"]
        data = self._rois_group["roi_image_masks_data"]

        if roi_ids is None:
            return sparse.GCXS((np.array(data), np.array(indices), indptr), shape=shape, compressed_axes=(0,))

        roi_indices = self.ids_to_indices(roi_ids)
        indices_parts = []
        data_parts = []
        counts = []
        for i in roi_indices:
            start, end = int(indptr[i]), int(indptr[i + 1])
            indices_parts.append(np.array(indices[start:end]))
            data_parts.append(np.array(data[start:end]))
            counts.append(end - start)

        new_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(indptr.dtype)
        new_indices = np.concatenate(indices_parts) if indices_parts else np.array([], dtype=indptr.dtype)
        new_data = np.concatenate(data_parts) if data_parts else np.array([], dtype=data.dtype)
        new_shape = (len(roi_indices), *shape[1:])
        return sparse.GCXS((new_data, new_indices, new_indptr), shape=new_shape, compressed_axes=(0,))


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
    if isinstance(image_masks, sparse.SparseArray):
        # zarr can't store a sparse array as a dataset value directly. Persist GCXS's own
        # (indptr, indices, data) components rather than round-tripping through COO: indptr
        # already marks each ROI's own boundary in the flat indices/data arrays (the same
        # CSR-style structure get_roi_image_masks needs to load a handful of ROIs without
        # reading every ROI's mask -- see ZarrRois._get_sparse_roi_image_masks).
        if isinstance(image_masks, sparse.GCXS):
            gcxs = image_masks if image_masks.compressed_axes == (0,) else image_masks.change_compressed_axes((0,))
        else:
            gcxs = sparse.GCXS.from_coo(image_masks.tocoo(), compressed_axes=(0,))
        zarr_group.attrs["roi_image_masks_sparse"] = True
        zarr_group.attrs["roi_image_masks_shape"] = list(gcxs.shape)
        zarr_group.create_dataset("roi_image_masks_indptr", data=gcxs.indptr, compressor=None)

        # Chunk indices/data to roughly a handful of ROIs' worth of entries, unless the caller
        # has already specified a chunk layout. zarr's own auto-chunking picks a chunk size
        # based on total array size with no notion of our per-ROI access pattern (indptr), so
        # a single-ROI request can still force decompressing a chunk sized for hundreds of
        # ROIs -- confirmed empirically to scale peak memory with total ROI count otherwise.
        sparse_saving_options = saving_options
        if "chunks" not in saving_options:
            avg_roi_nnz = max(1, gcxs.nnz // max(gcxs.shape[0], 1))
            entries_per_chunk = max(8 * avg_roi_nnz, 1024)
            sparse_saving_options = {**saving_options, "chunks": (entries_per_chunk,)}
        zarr_group.create_dataset("roi_image_masks_indices", data=gcxs.indices, **sparse_saving_options)
        zarr_group.create_dataset("roi_image_masks_data", data=gcxs.data, **sparse_saving_options)
    else:
        zarr_group.attrs["roi_image_masks_sparse"] = False
        # Chunk along the ROI axis (first dimension) for efficient per-ROI access,
        # unless the caller has already specified a chunk layout.
        if "chunks" not in saving_options:
            roi_chunks = (1,) + image_masks.shape[1:]
            saving_options = {**saving_options, "chunks": roi_chunks}
        zarr_group.create_dataset("roi_image_masks", data=image_masks, **saving_options)

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
