"""Round-trip save/load tests for BaseRois, covering both dense and sparse image masks."""

import numpy as np
import pytest
import sparse

from photon_mosaic.core.binaryrois import BinaryFolderRois
from photon_mosaic.core.generators import generate_rois
from photon_mosaic.core.zarrrois import ZarrRois


@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def rois(request):
    return generate_rois(
        num_rois=3, height=30, width=30, radius_range=(4, 6), sampling_frequency=20.0, seed=0, sparse=request.param
    )


def _assert_masks_equal(reloaded, original):
    if isinstance(original, sparse.SparseArray):
        np.testing.assert_array_equal(reloaded.todense(), original.todense())
    else:
        np.testing.assert_array_equal(reloaded, original)


def test_save_binary_roundtrip(rois, tmp_path):
    original_masks = rois.get_roi_image_masks()
    saved = rois.save(format="binary", folder=tmp_path / "rois")

    assert isinstance(saved, BinaryFolderRois)
    _assert_masks_equal(saved.get_roi_image_masks(), original_masks)
    np.testing.assert_array_equal(saved.roi_ids, rois.roi_ids)
    assert saved.sampling_frequency == rois.sampling_frequency

    ext = "npz" if isinstance(original_masks, sparse.SparseArray) else "npy"
    assert (tmp_path / "rois" / f"roi_image_masks.{ext}").is_file()


def test_save_zarr_roundtrip(rois, tmp_path):
    """folder= is resolved to a proper zarr path (.zarr suffix appended if missing) by
    spikeinterface's own save_to_zarr(), which now correctly reconstructs a ZarrRois via
    the zarr_class_info attribute _save_zarr() writes."""
    original_masks = rois.get_roi_image_masks()
    saved = rois.save(format="zarr", folder=tmp_path / "myrois")

    assert isinstance(saved, ZarrRois)
    assert (tmp_path / "myrois.zarr").is_dir()
    _assert_masks_equal(saved.get_roi_image_masks(), original_masks)
    np.testing.assert_array_equal(saved.roi_ids, rois.roi_ids)


def test_save_zarr_existing_path_raises(rois, tmp_path):
    folder = tmp_path / "rois.zarr"
    rois.save(format="zarr", folder=folder)

    with pytest.raises(AssertionError, match="already exists"):
        rois.save(format="zarr", folder=folder)


def test_save_zarr_overwrite(rois, tmp_path):
    folder = tmp_path / "rois.zarr"
    rois.save(format="zarr", folder=folder)

    original_masks = rois.get_roi_image_masks()
    saved = rois.save(format="zarr", folder=folder, overwrite=True)
    _assert_masks_equal(saved.get_roi_image_masks(), original_masks)
