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
    spikeinterface's own save_to_zarr(), which reconstructs a ZarrRois via the
    zarr_class_info attribute _save_zarr() writes."""
    original_masks = rois.get_roi_image_masks()
    saved = rois.save(format="zarr", folder=tmp_path / "myrois")

    assert isinstance(saved, ZarrRois)
    assert (tmp_path / "myrois.zarr").is_dir()
    _assert_masks_equal(saved.get_roi_image_masks(), original_masks)
    np.testing.assert_array_equal(saved.roi_ids, rois.roi_ids)


def test_zarr_subset_selection_matches_full(rois, tmp_path):
    """A non-contiguous, out-of-order roi_ids subset should match the same slice of the
    full array, for both dense and sparse masks (see photon-mosaic#104 review discussion --
    ZarrRois.get_roi_image_masks previously materialised the full array before indexing)."""
    saved = rois.save(format="zarr", folder=tmp_path / "myrois")

    full = saved.get_roi_image_masks()
    full_dense = full.todense() if isinstance(full, sparse.SparseArray) else full

    subset_ids = saved.roi_ids[[2, 0]]
    subset = saved.get_roi_image_masks(roi_ids=subset_ids)
    subset_dense = subset.todense() if isinstance(subset, sparse.SparseArray) else subset

    assert subset.shape[0] == 2
    np.testing.assert_array_equal(subset_dense, full_dense[[2, 0]])


def test_zarr_empty_selection(rois, tmp_path):
    saved = rois.save(format="zarr", folder=tmp_path / "myrois")
    empty = saved.get_roi_image_masks(roi_ids=[])
    assert empty.shape == (0, *rois.shape[:2])


def test_zarr_partial_load_does_not_scale_with_total_rois(tmp_path):
    """Regression test for the bug found in review: requesting one ROI's mask out of many
    should stay cheap, not cost about the same as loading everything -- which it did when
    the full array was loaded before indexing (see photon-mosaic#104). Compares single-ROI
    vs. full-array cost within one 2000-ROI file, rather than single-ROI cost across a
    small-N vs. large-N file: at small total ROI counts the whole file is already smaller
    than one storage chunk, so there's no meaningful "partial" case to contrast against, and
    fixed per-call overhead dominates."""
    import tracemalloc

    for use_sparse in (False, True):
        rois = generate_rois(
            num_rois=2000,
            height=128,
            width=128,
            radius_range=(3, 6),
            sampling_frequency=20.0,
            seed=0,
            sparse=use_sparse,
        )
        saved = rois.save(format="zarr", folder=tmp_path / f"many_rois_{use_sparse}")

        tracemalloc.start()
        saved.get_roi_image_masks(roi_ids=[saved.roi_ids[0]])
        _, peak_one = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        saved.get_roi_image_masks()
        _, peak_all = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # A regression to eager full-array loading would make peak_one roughly equal to
        # peak_all; comfortably below half of it confirms only a fraction was actually read.
        assert peak_one < peak_all / 2, (
            f"sparse={use_sparse}: single-ROI load ({peak_one} bytes) not much smaller than "
            f"full ({peak_all} bytes) out of 2000 ROIs"
        )
