"""Tests for FluorescenceNode, FluorescenceExtension, DfOverFExtension, and NeuropilExtension."""

import numpy as np
import pytest

from photon_mosaic.core import create_roi_analyzer, load_roi_analyzer
from photon_mosaic.core.generators import generate_random_imaging, generate_rois
from photon_mosaic.core.roianalyzer_core_extensions import (
    FluorescenceNode,
    _build_halo_neuropil_masks,
    _kde_mode_percentile,
    _percentile_filter_roi,
)
from photon_mosaic.extractors.suite2prois import Suite2pRois

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, W = 32, 32
NUM_FRAMES = 20
NUM_ROIS = 5
SF = 30.0
SEED = 42


@pytest.fixture
def imaging():
    return generate_random_imaging(num_frames=NUM_FRAMES, height=H, width=W, sampling_frequency=SF, seed=SEED)


@pytest.fixture
def rois():
    return generate_rois(num_rois=NUM_ROIS, height=H, width=W, sampling_frequency=SF, seed=SEED)


@pytest.fixture
def chunk(imaging):
    """Full video as a single chunk (T, H, W, P)."""
    return imaging.get_series(epoch_index=0)


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_compute_returns_tuple(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois)
    result = node.compute(chunk, 0, NUM_FRAMES, 0, 0)
    assert isinstance(result, tuple)
    assert len(result) == 1


def test_compute_output_shape(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)
    assert fluorescence.shape == (NUM_FRAMES, NUM_ROIS)


def test_compute_output_dtype(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)
    assert fluorescence.dtype == np.float32


def test_compute_matches_manual_weighted_sum(imaging, rois, chunk):
    """Verify compute() matches a simple loop over ROIs."""
    node = FluorescenceNode(imaging, rois)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    masks = rois.get_roi_image_masks()  # (N, H, W)
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = masks.reshape(NUM_ROIS, -1).astype(np.float32)

    expected = chunk_flat @ masks_flat.T
    np.testing.assert_allclose(fluorescence, expected, rtol=1e-5)


def test_zero_mask_gives_zero_fluorescence(imaging, rois, chunk):
    """ROIs with all-zero masks should produce zero traces."""
    # Create rois with zero masks
    zero_masks = np.zeros((NUM_ROIS, H, W))
    from photon_mosaic.core.numpyimaging import NumpyRois

    zero_rois = NumpyRois(
        roi_image_masks=zero_masks,
        sampling_frequency=SF,
    )
    node = FluorescenceNode(imaging, zero_rois)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)
    np.testing.assert_array_equal(fluorescence, 0.0)


def test_compute_matches_dense_result_with_sparse_masks(imaging, chunk):
    """FluorescenceNode should give the same result for sparse (e.g. Suite2p-backed) ROIs
    as for dense ones, since it operates polymorphically on whatever get_roi_image_masks
    returns (see photon-mosaic#103)."""
    sparse_rois = generate_rois(num_rois=NUM_ROIS, height=H, width=W, sampling_frequency=SF, seed=SEED, sparse=True)
    dense_rois = generate_rois(num_rois=NUM_ROIS, height=H, width=W, sampling_frequency=SF, seed=SEED)

    (fluorescence_sparse,) = FluorescenceNode(imaging, sparse_rois).compute(chunk, 0, NUM_FRAMES, 0, 0)
    (fluorescence_dense,) = FluorescenceNode(imaging, dense_rois).compute(chunk, 0, NUM_FRAMES, 0, 0)
    np.testing.assert_allclose(fluorescence_sparse, fluorescence_dense, rtol=1e-5)


# ---------------------------------------------------------------------------
# Neuropil subtraction — per-ROI (N, H, W)
# ---------------------------------------------------------------------------


def test_neuropil_per_roi_subtraction(imaging, rois, chunk):
    """Per-ROI neuropil masks should subtract per-ROI neuropil traces."""
    rng = np.random.default_rng(123)
    neuropil = rng.random((NUM_ROIS, H, W)).astype(np.float32)
    neuropil_weight = 0.7

    node = FluorescenceNode(imaging, rois, neuropil=neuropil, neuropil_weight=neuropil_weight)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    # Compute expected manually
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = rois.get_roi_image_masks().reshape(NUM_ROIS, -1).astype(np.float32)
    neuropil_flat = neuropil.reshape(NUM_ROIS, -1).astype(np.float32)
    expected = chunk_flat @ masks_flat.T - neuropil_weight * (chunk_flat @ neuropil_flat.T)

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-5)


def test_neuropil_per_roi_shape(imaging, rois, chunk):
    neuropil = np.ones((NUM_ROIS, H, W), dtype=np.float32)
    node = FluorescenceNode(imaging, rois, neuropil=neuropil)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)
    assert fluorescence.shape == (NUM_FRAMES, NUM_ROIS)


# ---------------------------------------------------------------------------
# Neuropil subtraction — global (H, W)
# ---------------------------------------------------------------------------


def test_neuropil_global_subtraction(imaging, rois, chunk):
    """A single global neuropil mask (H, W) should broadcast across ROIs."""
    rng = np.random.default_rng(456)
    neuropil = rng.random((H, W)).astype(np.float32)
    neuropil_weight = 0.7

    node = FluorescenceNode(imaging, rois, neuropil=neuropil, neuropil_weight=neuropil_weight)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = rois.get_roi_image_masks().reshape(NUM_ROIS, -1).astype(np.float32)
    neuropil_flat = neuropil.reshape(1, -1).astype(np.float32)
    expected = chunk_flat @ masks_flat.T - neuropil_weight * (chunk_flat @ neuropil_flat.T)  # (T, N) - (T, 1)

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-5)


def test_neuropil_global_broadcasts_correctly(imaging, rois, chunk):
    """Global neuropil should subtract the same weighted value from every ROI per frame."""
    # Use a uniform neuropil mask so the neuropil trace is easy to predict
    neuropil = np.ones((H, W), dtype=np.float32)
    neuropil_weight = 0.7
    node = FluorescenceNode(imaging, rois, neuropil=neuropil, neuropil_weight=neuropil_weight)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    # The global neuropil trace is the sum of each frame
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    global_trace = chunk_flat.sum(axis=1, keepdims=True)  # (T, 1)

    # Without neuropil
    node_no_np = FluorescenceNode(imaging, rois)
    (fluor_no_np,) = node_no_np.compute(chunk, 0, NUM_FRAMES, 0, 0)

    np.testing.assert_allclose(fluorescence, fluor_no_np - neuropil_weight * global_trace, rtol=1e-5)


# ---------------------------------------------------------------------------
# neuropil_weight parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("neuropil_weight", [0.0, 0.3, 0.7, 1.0, 1.5])
def test_neuropil_weight_scales_subtraction(imaging, rois, chunk, neuropil_weight):
    """Varying neuropil_weight should linearly scale the subtracted neuropil trace."""
    rng = np.random.default_rng(789)
    neuropil = rng.random((NUM_ROIS, H, W)).astype(np.float32)

    node = FluorescenceNode(imaging, rois, neuropil=neuropil, neuropil_weight=neuropil_weight)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = rois.get_roi_image_masks().reshape(NUM_ROIS, -1).astype(np.float32)
    neuropil_flat = neuropil.reshape(NUM_ROIS, -1).astype(np.float32)
    expected = chunk_flat @ masks_flat.T - neuropil_weight * (chunk_flat @ neuropil_flat.T)

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-5)


def test_neuropil_weight_zero_equals_no_subtraction(imaging, rois, chunk):
    """neuropil_weight=0 should produce identical results to passing no neuropil."""
    rng = np.random.default_rng(321)
    neuropil = rng.random((NUM_ROIS, H, W)).astype(np.float32)

    node_weighted = FluorescenceNode(imaging, rois, neuropil=neuropil, neuropil_weight=0.0)
    (fluor_weighted,) = node_weighted.compute(chunk, 0, NUM_FRAMES, 0, 0)

    node_none = FluorescenceNode(imaging, rois, neuropil=None)
    (fluor_none,) = node_none.compute(chunk, 0, NUM_FRAMES, 0, 0)

    np.testing.assert_allclose(fluor_weighted, fluor_none, rtol=1e-5)


def test_neuropil_weight_default(imaging, rois, chunk):
    """The default neuropil_weight should be 0.7."""
    rng = np.random.default_rng(654)
    neuropil = rng.random((NUM_ROIS, H, W)).astype(np.float32)

    node_default = FluorescenceNode(imaging, rois, neuropil=neuropil)
    (fluor_default,) = node_default.compute(chunk, 0, NUM_FRAMES, 0, 0)

    node_explicit = FluorescenceNode(imaging, rois, neuropil=neuropil, neuropil_weight=0.7)
    (fluor_explicit,) = node_explicit.compute(chunk, 0, NUM_FRAMES, 0, 0)

    np.testing.assert_allclose(fluor_default, fluor_explicit, rtol=1e-5)


# ---------------------------------------------------------------------------
# No neuropil
# ---------------------------------------------------------------------------


def test_no_neuropil_returns_raw_weighted_sum(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois, neuropil=None)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = rois.get_roi_image_masks().reshape(NUM_ROIS, -1).astype(np.float32)
    expected = chunk_flat @ masks_flat.T

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Partial chunk
# ---------------------------------------------------------------------------


def test_compute_partial_chunk(imaging, rois):
    """Compute on a sub-slice of the video should return matching shape."""
    chunk = imaging.get_series(epoch_index=0, start_frame=5, end_frame=10)
    node = FluorescenceNode(imaging, rois)
    (fluorescence,) = node.compute(chunk, 5, 10, 0, 0)
    assert fluorescence.shape == (5, NUM_ROIS)


# ---------------------------------------------------------------------------
# Multi-plane
# ---------------------------------------------------------------------------


def test_compute_multiplane(rois):
    """FluorescenceNode should work with multi-plane imaging and ROIs."""
    num_planes = 2
    imaging_mp = generate_random_imaging(
        num_frames=NUM_FRAMES, height=H, width=W, num_planes=num_planes, sampling_frequency=SF, seed=SEED
    )
    rois_mp = generate_rois(
        num_rois=NUM_ROIS, height=H, width=W, num_planes=num_planes, sampling_frequency=SF, seed=SEED
    )
    chunk = imaging_mp.get_series(epoch_index=0)

    node = FluorescenceNode(imaging_mp, rois_mp)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)
    assert fluorescence.shape == (NUM_FRAMES, NUM_ROIS)
    assert fluorescence.dtype == np.float32


# ---------------------------------------------------------------------------
# FluorescenceExtension._get_data() return types
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer(imaging, rois):
    return create_roi_analyzer(rois, imaging, format="memory")


def test_get_data_numpy(analyzer):
    """_get_data(outputs='numpy') should return a numpy array."""
    analyzer.compute("fluorescence")
    ext = analyzer.get_extension("fluorescence")
    result = ext.get_data(outputs="numpy")
    assert isinstance(result, np.ndarray)
    assert result.shape == (NUM_FRAMES, NUM_ROIS)
    assert result.dtype == np.float32


def test_get_data_recording(analyzer):
    """_get_data(outputs='recording') should return a NumpyRecording."""
    from spikeinterface.core import NumpyRecording

    analyzer.compute("fluorescence")
    ext = analyzer.get_extension("fluorescence")
    result = ext.get_data(outputs="recording")
    assert isinstance(result, NumpyRecording)
    assert result.get_num_channels() == NUM_ROIS
    assert result.get_num_samples() == NUM_FRAMES
    assert result.sampling_frequency == analyzer.imaging.sampling_frequency


def test_get_data_invalid_output(analyzer):
    """_get_data with an unsupported output type should raise ValueError."""
    analyzer.compute("fluorescence")
    ext = analyzer.get_extension("fluorescence")
    with pytest.raises(ValueError, match="Unsupported output type"):
        ext.get_data(outputs="pandas")


# ---------------------------------------------------------------------------
# DfOverFExtension
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer_with_fluorescence(imaging, rois):
    analyzer = create_roi_analyzer(rois, imaging, format="memory")
    analyzer.compute("fluorescence")
    return analyzer


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("maximin", {}),
        ("percentile", {"prctile_baseline": 8.0}),
        ("percentile", {"prctile_baseline": None}),
        ("running_percentile", {"prctile_baseline": 8.0}),
    ],
)
def test_df_over_f_shape_dtype_finite(analyzer_with_fluorescence, method, kwargs):
    """dF/F output should have correct shape, float32 dtype, and finite values."""
    analyzer_with_fluorescence.compute("df_over_f", method=method, **kwargs)
    result = analyzer_with_fluorescence.get_extension("df_over_f").get_data()
    assert result.shape == (NUM_FRAMES, NUM_ROIS)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_df_over_f_invalid_method(analyzer_with_fluorescence):
    """An unknown method name should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer_with_fluorescence.compute("df_over_f", method="bogus")


def test_df_over_f_parallel_matches_serial(analyzer_with_fluorescence):
    """ProcessPoolExecutor result should match serial computation exactly."""
    kw = dict(method="percentile", prctile_baseline=8.0)
    analyzer_with_fluorescence.compute("df_over_f", **kw, n_jobs=1)
    serial = analyzer_with_fluorescence.get_extension("df_over_f").get_data().copy()
    analyzer_with_fluorescence.compute("df_over_f", **kw, n_jobs=2)
    parallel = analyzer_with_fluorescence.get_extension("df_over_f").get_data()
    np.testing.assert_allclose(serial, parallel, rtol=1e-5)


def test_percentile_filter_roi_win_ge_frames_uses_global_percentile():
    """Window >= trace length: every frame gets the global percentile."""
    rng = np.random.default_rng(0)
    n_frames = 200
    col = rng.random(n_frames).astype(np.float32)
    result = _percentile_filter_roi((col, n_frames, 8.0))
    expected = np.percentile(col, 8.0)
    np.testing.assert_array_equal(result, np.full_like(col, expected))


def test_percentile_filter_roi_win_lt_frames_uses_rolling_filter():
    """Window < trace length: rolling scipy.ndimage filter matches exactly."""
    from scipy.ndimage import percentile_filter

    rng = np.random.default_rng(0)
    col = rng.random(200).astype(np.float32)
    size = 20
    result = _percentile_filter_roi((col, size, 8.0))
    expected = percentile_filter(col, 8.0, size=size)
    np.testing.assert_array_equal(result, expected)


def test_df_over_f_get_data_recording(analyzer_with_fluorescence):
    """get_data(outputs='recording') should return a NumpyRecording."""
    from spikeinterface.core import NumpyRecording

    analyzer_with_fluorescence.compute("df_over_f")
    result = analyzer_with_fluorescence.get_extension("df_over_f").get_data(outputs="recording")
    assert isinstance(result, NumpyRecording)
    assert result.get_num_channels() == NUM_ROIS


def test_df_over_f_get_data_invalid_output(analyzer_with_fluorescence):
    """get_data with an unsupported output type should raise ValueError."""
    analyzer_with_fluorescence.compute("df_over_f")
    with pytest.raises(ValueError, match="Unsupported output type"):
        analyzer_with_fluorescence.get_extension("df_over_f").get_data(outputs="pandas")


def test_df_over_f_select_extension_data(analyzer_with_fluorescence, rois):
    """_select_extension_data should return only the requested ROI columns."""
    analyzer_with_fluorescence.compute("df_over_f")
    sub = analyzer_with_fluorescence.get_extension("df_over_f")._select_extension_data(rois.roi_ids[:2])
    assert sub["df_over_f"].shape == (NUM_FRAMES, 2)


# ---------------------------------------------------------------------------
# _kde_mode_percentile unit tests
# ---------------------------------------------------------------------------


def test_kde_constant_signal_returns_50():
    """A constant signal (R==0) should short-circuit to 50.0."""
    assert _kde_mode_percentile(np.ones(500)) == 50.0


def test_kde_returns_valid_percentile():
    """KDE should return a value in [0, 100) for well-behaved data."""
    prct = _kde_mode_percentile(np.random.default_rng(0).standard_normal(1000))
    assert 0.0 <= prct < 100.0


# ---------------------------------------------------------------------------
# NeuropilExtension ("halo" / Suite2p-style ring mask)
# ---------------------------------------------------------------------------

# Suite2p's default min_neuropil_pixels (350) is a third of this file's 32x32 test frame, so
# halo-mask tests use their own larger frame and a much smaller min_neuropil_pixels.
NEUROPIL_H, NEUROPIL_W = 64, 64


def _make_halo_stats(centers, radius=2):
    """Small, well-separated square ROIs, for testing ring exclusion/weighting precisely."""
    stats = []
    for cy, cx in centers:
        yy, xx = np.meshgrid(
            np.arange(cy - radius, cy + radius + 1), np.arange(cx - radius, cx + radius + 1), indexing="ij"
        )
        ypix, xpix = yy.ravel(), xx.ravel()
        stats.append(dict(ypix=ypix, xpix=xpix, lam=np.ones(len(ypix)), radius=float(radius)))
    return stats


@pytest.fixture
def halo_stats():
    return _make_halo_stats([(15, 15), (45, 45), (15, 45)])


@pytest.fixture
def suite2p_rois(halo_stats):
    return Suite2pRois.from_stat(halo_stats, shape=(NEUROPIL_H, NEUROPIL_W, 1), sampling_frequency=SF)


@pytest.fixture
def neuropil_imaging():
    return generate_random_imaging(
        num_frames=NUM_FRAMES, height=NEUROPIL_H, width=NEUROPIL_W, sampling_frequency=SF, seed=SEED
    )


def test_halo_masks_shape_and_dtype(suite2p_rois):
    masks = _build_halo_neuropil_masks(suite2p_rois.get_roi_image_masks(), min_neuropil_pixels=30)
    assert masks.shape == (3, NEUROPIL_H, NEUROPIL_W)
    assert masks.dtype == np.float32


def test_halo_masks_ring_weights_sum_to_one(suite2p_rois):
    masks = _build_halo_neuropil_masks(suite2p_rois.get_roi_image_masks(), min_neuropil_pixels=30)
    dense = masks.todense()
    for i in range(suite2p_rois.get_num_rois()):
        assert dense[i].sum() == pytest.approx(1.0)


def test_halo_masks_exclude_own_and_other_roi_pixels(halo_stats, suite2p_rois):
    """Each ROI's ring should have zero weight at every pixel belonging to any ROI, not just itself."""
    masks = _build_halo_neuropil_masks(suite2p_rois.get_roi_image_masks(), min_neuropil_pixels=30)
    dense = masks.todense()
    for i in range(len(halo_stats)):
        for stat in halo_stats:
            assert dense[i][stat["ypix"], stat["xpix"]].sum() == 0.0


def test_halo_masks_works_with_generic_dense_masks(halo_stats):
    """The mask-based helper should work with any dense (n_rois, Ly, Lx) mask array, not just Suite2p."""
    dense_masks = np.zeros((len(halo_stats), NEUROPIL_H, NEUROPIL_W), dtype=bool)
    for i, stat in enumerate(halo_stats):
        dense_masks[i, stat["ypix"], stat["xpix"]] = True

    masks = _build_halo_neuropil_masks(dense_masks, min_neuropil_pixels=30)
    dense = masks.todense()
    for i in range(len(halo_stats)):
        assert dense[i].sum() == pytest.approx(1.0)
        for stat in halo_stats:
            assert dense[i][stat["ypix"], stat["xpix"]].sum() == 0.0


def test_halo_masks_zero_rois():
    masks = _build_halo_neuropil_masks(np.zeros((0, NEUROPIL_H, NEUROPIL_W), dtype=bool))
    assert masks.shape == (0, NEUROPIL_H, NEUROPIL_W)


def test_neuropil_extension_run_and_get_data(suite2p_rois, neuropil_imaging):
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    masks = analyzer.get_extension("neuropil").get_data()
    assert masks.shape == (3, NEUROPIL_H, NEUROPIL_W)
    dense = masks.todense()
    for i in range(3):
        assert dense[i].sum() == pytest.approx(1.0)


def test_neuropil_extension_select_extension_data(suite2p_rois, neuropil_imaging):
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    sub = analyzer.get_extension("neuropil")._select_extension_data(suite2p_rois.roi_ids[:2])
    assert sub["neuropil_masks"].shape == (2, NEUROPIL_H, NEUROPIL_W)


def test_neuropil_extension_default_params(suite2p_rois, neuropil_imaging):
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    params = analyzer.get_extension("neuropil").params
    assert params["method"] == "halo"
    assert params["inner_neuropil_radius"] == 2
    assert params["circular"] is False


def test_neuropil_extension_unknown_method_raises(suite2p_rois, neuropil_imaging):
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.compute("neuropil", method="bogus")


def test_neuropil_extension_unknown_kwarg_raises(suite2p_rois, neuropil_imaging):
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    with pytest.raises(TypeError, match="bogus_kwarg"):
        analyzer.compute("neuropil", bogus_kwarg=1)


def test_neuropil_extension_works_with_non_suite2p_rois(imaging, rois):
    """NeuropilExtension('halo') derives pixel coordinates from get_roi_image_masks(), so it
    works with any BaseRois, not only Suite2pRois -- this also matches the fact that
    create_roi_analyzer(..., format="memory") snapshots ROIs into a plain NumpyRois internally,
    so requiring a Suite2p-specific accessor would break even genuine Suite2pRois input."""
    analyzer = create_roi_analyzer(rois, imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    masks = analyzer.get_extension("neuropil").get_data()
    assert masks.shape == (NUM_ROIS, H, W)


def test_neuropil_extension_multiplane_rois_raises(halo_stats):
    multiplane_rois = Suite2pRois.from_stat(
        halo_stats,
        shape=(NEUROPIL_H, NEUROPIL_W, 2),
        sampling_frequency=SF,
        plane_assignments=np.zeros(len(halo_stats), dtype=int),
    )
    multiplane_imaging = generate_random_imaging(
        num_frames=NUM_FRAMES, height=NEUROPIL_H, width=NEUROPIL_W, num_planes=2, sampling_frequency=SF, seed=SEED
    )
    analyzer = create_roi_analyzer(multiplane_rois, multiplane_imaging, format="memory")
    with pytest.raises(NotImplementedError, match="single-plane"):
        analyzer.compute("neuropil")


def test_neuropil_extension_binary_folder_roundtrip(suite2p_rois, neuropil_imaging, tmp_path):
    folder = tmp_path / "neuropil_binary"
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="binary_folder", folder=folder)
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    original = analyzer.get_extension("neuropil").get_data()

    loaded = load_roi_analyzer(folder)
    reloaded = loaded.get_extension("neuropil").get_data()
    np.testing.assert_array_equal(reloaded.todense(), original.todense())


def test_neuropil_extension_zarr_roundtrip(suite2p_rois, neuropil_imaging, tmp_path):
    folder = tmp_path / "neuropil.zarr"
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="zarr", folder=folder)
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    original = analyzer.get_extension("neuropil").get_data()

    loaded = load_roi_analyzer(folder)
    reloaded = loaded.get_extension("neuropil").get_data()
    np.testing.assert_array_equal(reloaded.todense(), original.todense())


def test_fluorescence_extension_auto_uses_neuropil_extension(suite2p_rois, neuropil_imaging):
    """FluorescenceExtension should automatically pick up a computed NeuropilExtension."""
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    neuropil_masks = analyzer.get_extension("neuropil").get_data()

    neuropil_weight = 0.7
    analyzer.compute("fluorescence", use_neuropil=True, neuropil_weight=neuropil_weight)
    fluorescence = analyzer.get_extension("fluorescence").get_data()

    chunk = neuropil_imaging.get_series(epoch_index=0)
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    roi_masks_flat = suite2p_rois.get_roi_image_masks().reshape((3, -1)).astype(np.float32)
    neuropil_flat = neuropil_masks.reshape((3, -1)).astype(np.float32)
    expected = chunk_flat @ roi_masks_flat.T - neuropil_weight * (chunk_flat @ neuropil_flat.T)

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-4)


def test_fluorescence_extension_use_neuropil_false_ignores_computed_extension(suite2p_rois, neuropil_imaging):
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    analyzer.compute("fluorescence", use_neuropil=False)
    fluorescence = analyzer.get_extension("fluorescence").get_data()

    chunk = neuropil_imaging.get_series(epoch_index=0)
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    roi_masks_flat = suite2p_rois.get_roi_image_masks().reshape((3, -1)).astype(np.float32)
    expected = chunk_flat @ roi_masks_flat.T

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-5)
