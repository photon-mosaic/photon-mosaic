"""Round-trip save/load tests for BaseRois, covering both dense and sparse image masks."""

import numpy as np
import pytest
import sparse

from photon_mosaic.core.binaryrois import BinaryFolderRois
from photon_mosaic.core.generators import generate_rois
from photon_mosaic.core.zarrrois import ZarrRois


@pytest.fixture
def dense_rois():
    return generate_rois(num_rois=3, height=30, width=30, radius_range=(4, 6), sampling_frequency=20.0, seed=0)


@pytest.fixture
def sparse_rois():
    return generate_rois(
        num_rois=3, height=30, width=30, radius_range=(4, 6), sampling_frequency=20.0, seed=0, sparse=True
    )


def test_save_binary_roundtrip_dense(dense_rois, tmp_path):
    original_masks = dense_rois.get_roi_image_masks()
    saved = dense_rois.save(format="binary", folder=tmp_path / "rois")

    assert isinstance(saved, BinaryFolderRois)
    np.testing.assert_array_equal(saved.get_roi_image_masks(), original_masks)
    np.testing.assert_array_equal(saved.roi_ids, dense_rois.roi_ids)
    assert saved.sampling_frequency == dense_rois.sampling_frequency


def test_save_binary_roundtrip_sparse(sparse_rois, tmp_path):
    original_masks = sparse_rois.get_roi_image_masks()
    saved = sparse_rois.save(format="binary", folder=tmp_path / "rois")

    assert isinstance(saved, BinaryFolderRois)
    reloaded_masks = saved.get_roi_image_masks()
    assert isinstance(reloaded_masks, sparse.SparseArray)
    np.testing.assert_array_equal(reloaded_masks.todense(), original_masks.todense())
    np.testing.assert_array_equal(saved.roi_ids, sparse_rois.roi_ids)

    # Saved on disk as .npz (sparse), not .npy (dense)
    assert (tmp_path / "rois" / "roi_image_masks.npz").is_file()
    assert not (tmp_path / "rois" / "roi_image_masks.npy").is_file()


def test_save_zarr_roundtrip_dense(dense_rois, tmp_path):
    original_masks = dense_rois.get_roi_image_masks()
    saved = dense_rois.save(format="zarr", zarr_path=tmp_path / "rois.zarr")

    assert isinstance(saved, ZarrRois)
    np.testing.assert_array_equal(saved.get_roi_image_masks(), original_masks)
    np.testing.assert_array_equal(saved.roi_ids, dense_rois.roi_ids)


def test_save_zarr_roundtrip_sparse(sparse_rois, tmp_path):
    original_masks = sparse_rois.get_roi_image_masks()
    saved = sparse_rois.save(format="zarr", zarr_path=tmp_path / "rois.zarr")

    assert isinstance(saved, ZarrRois)
    reloaded_masks = saved.get_roi_image_masks()
    assert isinstance(reloaded_masks, sparse.SparseArray)
    np.testing.assert_array_equal(reloaded_masks.todense(), original_masks.todense())
    np.testing.assert_array_equal(saved.roi_ids, sparse_rois.roi_ids)
