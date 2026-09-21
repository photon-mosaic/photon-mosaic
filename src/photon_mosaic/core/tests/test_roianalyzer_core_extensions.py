"""Tests for FluorescenceNode, FluorescenceExtension, DfOverFExtension, NeuropilExtension, DeconvolutionExtension."""

import numpy as np
import pytest

from photon_mosaic.core import create_roi_analyzer, load_roi_analyzer
from photon_mosaic.core.generators import generate_fluorescence, generate_random_imaging, generate_rois
from photon_mosaic.core.numpyimaging import NumpyImaging, NumpyRois
from photon_mosaic.core.roianalyzer_core_extensions import (
    FluorescenceNode,
    _build_surround_neuropil_masks,
    _cnmf_objective,
    _kde_mode_percentile,
    _nnls_block,
    _partition_of_unity,
    _percentile_filter_roi,
    _ridge_inverse,
    _spatial_bandpass,
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
    """Verify compute() matches a simple loop over ROIs.

    FluorescenceNode normalizes each ROI's mask to sum to 1 internally (see its docstring),
    so "manual" here means normalizing rois.get_roi_image_masks() the same way before
    comparing -- for the binary masks used here, that's equivalent to dividing by pixel count.
    """
    node = FluorescenceNode(imaging, rois)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    masks = rois.get_roi_image_masks()  # (N, H, W)
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = masks.reshape(NUM_ROIS, -1).astype(np.float32)
    masks_flat = masks_flat / masks_flat.sum(axis=1, keepdims=True)

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


def test_compute_matches_least_squares_for_weighted_nonoverlapping_masks(imaging, chunk):
    """For non-overlapping weighted (non-binary) masks, FluorescenceNode's internal L1/L2
    rescaling should reproduce the least-squares reconstruction of movie ~ traces @ masks,
    which for non-overlapping ROIs is dividing by each mask's own L2 norm squared -- not L1
    (pixel count/sum), which is what a naive Suite2p-style mean convention would give instead.
    Every other FluorescenceNode test in this file uses binary masks, for which L1 == L2^2 and
    this rescaling is a no-op -- so this is the only test that actually exercises it."""
    from photon_mosaic.core.numpyimaging import NumpyRois

    weighted_masks = np.zeros((2, H, W), dtype=np.float32)
    # Two disjoint weighted (non-binary) patches -- fractional values matter here, since
    # L1 == L2^2 for any binary mask regardless of its shape.
    weighted_masks[0, 2:6, 2:6] = np.linspace(0.2, 1.0, 4)[:, None]
    weighted_masks[1, 20:25, 20:25] = np.linspace(0.1, 0.8, 5)[None, :]
    weighted_rois = NumpyRois(roi_image_masks=weighted_masks, sampling_frequency=SF)

    node = FluorescenceNode(imaging, weighted_rois)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = weighted_masks.reshape(2, -1)
    expected = (chunk_flat @ masks_flat.T) / (masks_flat**2).sum(axis=1)

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-4)


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

    # Compute expected manually (masks_flat normalized to sum to 1, see FluorescenceNode's docstring)
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = rois.get_roi_image_masks().reshape(NUM_ROIS, -1).astype(np.float32)
    masks_flat = masks_flat / masks_flat.sum(axis=1, keepdims=True)
    neuropil_flat = neuropil.reshape(NUM_ROIS, -1).astype(np.float32)
    expected = chunk_flat @ masks_flat.T - neuropil_weight * (chunk_flat @ neuropil_flat.T)

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-5)


def test_neuropil_subtraction_with_weighted_masks_matches_manual(imaging, chunk):
    """Neuropil subtraction combined with weighted (non-binary) ROI masks should match a manual
    computation of the full algorithm (L1-normalized extraction and subtraction, then rescaled
    to the L2-normalized reconstruction scale -- see FluorescenceNode's docstring). Every other
    neuropil-subtraction test in this file uses binary ROI masks, for which the L1/L2 rescaling
    is a no-op, so this is the only one that exercises rescaling and neuropil subtraction
    together."""
    from photon_mosaic.core.numpyimaging import NumpyRois

    weighted_masks = np.zeros((2, H, W), dtype=np.float32)
    weighted_masks[0, 2:6, 2:6] = np.linspace(0.2, 1.0, 4)[:, None]
    weighted_masks[1, 20:25, 20:25] = np.linspace(0.1, 0.8, 5)[None, :]
    weighted_rois = NumpyRois(roi_image_masks=weighted_masks, sampling_frequency=SF)

    rng = np.random.default_rng(789)
    neuropil = rng.random((2, H, W)).astype(np.float32)
    neuropil_weight = 0.7

    node = FluorescenceNode(imaging, weighted_rois, neuropil=neuropil, neuropil_weight=neuropil_weight)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = weighted_masks.reshape(2, -1)
    neuropil_flat = neuropil.reshape(2, -1).astype(np.float32)

    l1 = masks_flat.sum(axis=1, keepdims=True)
    l2sq = (masks_flat**2).sum(axis=1, keepdims=True)
    extracted_l1 = (chunk_flat @ (masks_flat / l1).T) - neuropil_weight * (chunk_flat @ neuropil_flat.T)
    expected = extracted_l1 * (l1 / l2sq).reshape(1, -1)

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-4)


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
    masks_flat = masks_flat / masks_flat.sum(axis=1, keepdims=True)
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
    masks_flat = masks_flat / masks_flat.sum(axis=1, keepdims=True)
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


def test_no_neuropil_returns_normalized_weighted_sum(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois, neuropil=None)
    (fluorescence,) = node.compute(chunk, 0, NUM_FRAMES, 0, 0)

    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    masks_flat = rois.get_roi_image_masks().reshape(NUM_ROIS, -1).astype(np.float32)
    masks_flat = masks_flat / masks_flat.sum(axis=1, keepdims=True)
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


def test_get_computable_extensions_lists_all_core_extensions(analyzer):
    """RoiAnalyzer.get_computable_extensions() is backed by a separate built-in registry
    from the one compute() uses to auto-import extension classes -- it must list every core
    extension defined in this module, not just the ones that happened to be registered when
    that registry was first introduced."""
    assert set(analyzer.get_computable_extensions()) == {
        "fluorescence",
        "df_over_f",
        "deconvolution",
        "neuropil",
    }


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


@pytest.mark.parametrize("method", ["maximin", "percentile"])
def test_df_over_f_f0_matches_baseline_used(analyzer_with_fluorescence, method):
    """The stored f0 should be exactly the baseline df_over_f was computed against."""
    ext = analyzer_with_fluorescence.compute("df_over_f", method=method)
    fluorescence = analyzer_with_fluorescence.get_extension("fluorescence").get_data()
    f0 = ext.data["f0"]
    assert f0.shape == (NUM_FRAMES, NUM_ROIS)
    assert f0.dtype == np.float32
    expected = (fluorescence - f0) / (f0 + np.finfo(np.float32).eps)
    np.testing.assert_allclose(ext.get_data(), expected, rtol=1e-5)


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
    assert sub["f0"].shape == (NUM_FRAMES, 2)


# ---------------------------------------------------------------------------
# DeconvolutionExtension
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer_with_df_over_f(analyzer_with_fluorescence):
    analyzer_with_fluorescence.compute("df_over_f")
    return analyzer_with_fluorescence


def test_deconvolution_shape_dtype_finite(analyzer_with_df_over_f):
    """Deconvolution output should have correct shape, float32 dtype, and finite values."""
    analyzer_with_df_over_f.compute("deconvolution")
    ext = analyzer_with_df_over_f.get_extension("deconvolution")
    deconvolved = ext.get_data()
    denoised = ext.data["denoised"]
    assert deconvolved.shape == (NUM_FRAMES, NUM_ROIS)
    assert denoised.shape == (NUM_FRAMES, NUM_ROIS)
    assert deconvolved.dtype == np.float32
    assert denoised.dtype == np.float32
    assert np.isfinite(deconvolved).all()
    assert np.isfinite(denoised).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # default: decay_time=None, rise_time=0 -> AR(1) auto-estimated
        {"rise_time": None},  # both None -> AR(2), both auto-estimated
        {"decay_time": 2.0},  # known decay, default rise=0 -> AR(1) with known decay
        {"decay_time": 2.0, "rise_time": 0.1},  # both known -> AR(2) with known kinetics
    ],
)
def test_deconvolution_kinetics_combinations_run_and_are_finite(analyzer_with_df_over_f, kwargs):
    """Every supported decay_time/rise_time combination should run without error."""
    analyzer_with_df_over_f.compute("deconvolution", **kwargs)
    ext = analyzer_with_df_over_f.get_extension("deconvolution")
    assert np.isfinite(ext.data["deconvolved"]).all()
    assert np.isfinite(ext.data["denoised"]).all()


def test_deconvolution_rise_time_without_decay_time_raises(analyzer_with_df_over_f):
    """rise_time given without decay_time is ambiguous and should raise, matching oasis's own validation."""
    with pytest.raises(ValueError, match="tau_d is required"):
        analyzer_with_df_over_f.compute("deconvolution", rise_time=0.1)


def test_deconvolution_unknown_kwarg_raises(analyzer_with_df_over_f):
    """A misspelled/unknown keyword argument should raise instead of being silently ignored."""
    with pytest.raises(TypeError, match="noisestd"):
        analyzer_with_df_over_f.compute("deconvolution", noisestd=0.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"penalty": None},  # plain deconvolution, default lam=0/s_min=0
        {"penalty": None, "lam": 0.5},  # plain deconvolution, explicit sparsity weight
        {"penalty": None, "s_min": 0.1},  # plain deconvolution, explicit minimal spike size
    ],
)
def test_deconvolution_penalty_none_runs_and_is_finite(analyzer_with_df_over_f, kwargs):
    """penalty=None (plain, non-noise-constrained deconvolution) should run without error."""
    analyzer_with_df_over_f.compute("deconvolution", **kwargs)
    ext = analyzer_with_df_over_f.get_extension("deconvolution")
    assert np.isfinite(ext.data["deconvolved"]).all()
    assert np.isfinite(ext.data["denoised"]).all()


def test_deconvolution_parallel_matches_serial(analyzer_with_df_over_f):
    """ProcessPoolExecutor result should closely match serial computation.

    Not bit-exact: OASIS's solver can diverge at floating-point-noise level between
    processes (e.g. BLAS/FFT thread-count differences) -- not something it guarantees against.
    """
    analyzer_with_df_over_f.compute("deconvolution", n_jobs=1)
    serial = analyzer_with_df_over_f.get_extension("deconvolution").get_data().copy()
    analyzer_with_df_over_f.compute("deconvolution", n_jobs=2)
    parallel = analyzer_with_df_over_f.get_extension("deconvolution").get_data()
    np.testing.assert_allclose(serial, parallel, atol=1e-2)


def test_deconvolution_get_data_recording(analyzer_with_df_over_f):
    """get_data(outputs='recording') should return a NumpyRecording with the analyzer's
    sampling_frequency, matching what _run actually used to deconvolve."""
    from spikeinterface.core import NumpyRecording

    analyzer_with_df_over_f.compute("deconvolution")
    result = analyzer_with_df_over_f.get_extension("deconvolution").get_data(outputs="recording")
    assert isinstance(result, NumpyRecording)
    assert result.get_num_channels() == NUM_ROIS
    assert result.sampling_frequency == SF


def test_deconvolution_get_data_invalid_output(analyzer_with_df_over_f):
    """get_data with an unsupported output type should raise ValueError."""
    analyzer_with_df_over_f.compute("deconvolution")
    with pytest.raises(ValueError, match="Unsupported output type"):
        analyzer_with_df_over_f.get_extension("deconvolution").get_data(outputs="pandas")


def test_deconvolution_select_extension_data(analyzer_with_df_over_f, rois):
    """_select_extension_data should return only the requested ROI columns for both fields."""
    analyzer_with_df_over_f.compute("deconvolution")
    sub = analyzer_with_df_over_f.get_extension("deconvolution")._select_extension_data(rois.roi_ids[:2])
    assert sub["deconvolved"].shape == (NUM_FRAMES, 2)
    assert sub["denoised"].shape == (NUM_FRAMES, 2)


def test_deconvolution_works_without_loaded_imaging(analyzer_with_df_over_f):
    """need_imaging=False should hold in practice: no raw imaging should be required.

    Simulates an analyzer whose raw imaging was never loaded (e.g. reloaded
    from disk), where ``roi_analyzer.imaging`` raises. sampling_frequency
    should fall back to ``roi_analyzer.sampling_frequency`` instead.
    """
    analyzer_with_df_over_f._imaging = None
    analyzer_with_df_over_f._temporary_imaging = None
    assert not analyzer_with_df_over_f.has_imaging()
    assert not analyzer_with_df_over_f.has_temporary_imaging()

    analyzer_with_df_over_f.compute("deconvolution")
    result = analyzer_with_df_over_f.get_extension("deconvolution").get_data()
    assert result.shape == (NUM_FRAMES, NUM_ROIS)


def test_deconvolution_recovers_ground_truth_spikes_and_trace(analyzer_with_df_over_f):
    """OASIS output should correlate strongly with the true spikes and clean trace.

    Bypasses the (separately tested) baseline-estimation step by injecting
    known noisy traces from :func:`generate_fluorescence` directly as the
    ``df_over_f`` extension's data, isolating the deconvolution step itself.
    """
    ground_truth_frames = 1000
    ground_truth = generate_fluorescence(
        num_frames=ground_truth_frames,
        num_rois=NUM_ROIS,
        sampling_frequency=SF,
        decay_time=2.0,
        noise_std=0.2,
        seed=SEED,
    )
    # traces == (1 + clean_traces) * bleach(t) + noise; bleach(t) == 1 since bleaching_time
    # defaults to inf, so subtracting 1 gives dF/F == clean_traces + noise.
    analyzer_with_df_over_f.get_extension("df_over_f").data["df_over_f"] = ground_truth.traces - 1.0
    analyzer_with_df_over_f.compute("deconvolution")
    ext = analyzer_with_df_over_f.get_extension("deconvolution")

    for roi_idx in range(NUM_ROIS):
        deconvolved_corr = np.corrcoef(ext.data["deconvolved"][:, roi_idx], ground_truth.spikes[:, roi_idx])[0, 1]
        denoised_corr = np.corrcoef(ext.data["denoised"][:, roi_idx], ground_truth.clean_traces[:, roi_idx])[0, 1]
        assert deconvolved_corr > 0.75
        assert denoised_corr > 0.97


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
# NeuropilExtension ("surround" / Suite2p-style ring mask)
# ---------------------------------------------------------------------------

# Suite2p's default min_neuropil_pixels (350) is a third of this file's 32x32 test frame, so
# surround-mask tests use their own larger frame and a much smaller min_neuropil_pixels.
NEUROPIL_H, NEUROPIL_W = 64, 64


def _make_surround_stats(centers, radius=2):
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
def surround_stats():
    return _make_surround_stats([(15, 15), (45, 45), (15, 45)])


@pytest.fixture
def suite2p_rois(surround_stats):
    return Suite2pRois.from_stat(surround_stats, shape=(NEUROPIL_H, NEUROPIL_W, 1), sampling_frequency=SF)


@pytest.fixture
def neuropil_imaging():
    return generate_random_imaging(
        num_frames=NUM_FRAMES, height=NEUROPIL_H, width=NEUROPIL_W, sampling_frequency=SF, seed=SEED
    )


def test_surround_masks_shape_and_dtype(suite2p_rois):
    masks = _build_surround_neuropil_masks(suite2p_rois.get_roi_image_masks(), min_neuropil_pixels=30)
    assert masks.shape == (3, NEUROPIL_H, NEUROPIL_W)
    assert masks.dtype == np.float32


def test_surround_masks_ring_weights_sum_to_one(suite2p_rois):
    masks = _build_surround_neuropil_masks(suite2p_rois.get_roi_image_masks(), min_neuropil_pixels=30)
    dense = masks.todense()
    for i in range(suite2p_rois.get_num_rois()):
        assert dense[i].sum() == pytest.approx(1.0)


def test_surround_masks_exclude_own_and_other_roi_pixels(surround_stats, suite2p_rois):
    """Each ROI's ring should have zero weight at every pixel belonging to any ROI, not just itself."""
    masks = _build_surround_neuropil_masks(suite2p_rois.get_roi_image_masks(), min_neuropil_pixels=30)
    dense = masks.todense()
    for i in range(len(surround_stats)):
        for stat in surround_stats:
            assert dense[i][stat["ypix"], stat["xpix"]].sum() == 0.0


def test_surround_masks_works_with_generic_dense_masks(surround_stats):
    """The mask-based helper should work with any dense (n_rois, Ly, Lx) mask array, not just Suite2p."""
    dense_masks = np.zeros((len(surround_stats), NEUROPIL_H, NEUROPIL_W), dtype=bool)
    for i, stat in enumerate(surround_stats):
        dense_masks[i, stat["ypix"], stat["xpix"]] = True

    masks = _build_surround_neuropil_masks(dense_masks, min_neuropil_pixels=30)
    dense = masks.todense()
    for i in range(len(surround_stats)):
        assert dense[i].sum() == pytest.approx(1.0)
        for stat in surround_stats:
            assert dense[i][stat["ypix"], stat["xpix"]].sum() == 0.0


def test_surround_masks_zero_rois():
    masks = _build_surround_neuropil_masks(np.zeros((0, NEUROPIL_H, NEUROPIL_W), dtype=bool))
    assert masks.shape == (0, NEUROPIL_H, NEUROPIL_W)


def test_surround_masks_handles_degenerate_lam_sum():
    """A mask whose nonzero values happen to sum to <= 0 (e.g. mixed positive/negative
    weights) should fall back to uniform weights internally, rather than propagating a
    degenerate lam array into suite2p's create_cell_pix/create_neuropil_masks (which use
    lam_percentile to decide which weighted pixels count as "ROI" pixels -- a non-positive
    sum would make that comparison meaningless)."""
    masks = np.zeros((1, NEUROPIL_H, NEUROPIL_W), dtype=np.float32)
    # Four nonzero pixels, but weights sum to exactly 0 -- triggers the lam.sum() <= 0 fallback.
    masks[0, 15, 15] = 1.0
    masks[0, 15, 16] = -1.0
    masks[0, 16, 15] = 0.5
    masks[0, 16, 16] = -0.5

    result = _build_surround_neuropil_masks(masks, min_neuropil_pixels=30)
    dense = result.todense()
    assert result.shape == (1, NEUROPIL_H, NEUROPIL_W)
    assert np.isfinite(dense).all()
    assert dense[0].sum() == pytest.approx(1.0)


def test_surround_masks_empty_ring_when_no_non_roi_pixels_available():
    """An ROI with no available non-ROI pixels anywhere in the frame (here: two ROIs together
    tiling the entire image) should get an all-zero ring row rather than erroring -- the
    documented behavior for a ring that ends up empty."""
    small_h, small_w = 20, 20
    masks = np.zeros((2, small_h, small_w), dtype=bool)
    masks[0, : small_h // 2, :] = True  # top half
    masks[1, small_h // 2 :, :] = True  # bottom half -- together, every pixel belongs to an ROI

    result = _build_surround_neuropil_masks(masks, min_neuropil_pixels=30)
    dense = result.todense()
    assert result.shape == (2, small_h, small_w)
    assert dense[0].sum() == 0.0
    assert dense[1].sum() == 0.0


def test_surround_masks_all_zero_roi_does_not_raise():
    """An ROI with no pixels at all (e.g. a detection that ended up empty) must not be passed
    to suite2p's create_neuropil_masks: its rectangular (default, non-circular) growth path
    calls .min()/.max() on the ROI's own pixel coordinates, which raises on an empty array.
    Such a ROI gets an all-zero ring directly instead, and doesn't affect its neighbors'
    rings."""
    masks = np.zeros((3, NEUROPIL_H, NEUROPIL_W), dtype=bool)
    masks[0, 15:20, 15:20] = True
    # masks[1] stays all-zero -- no pixels anywhere
    masks[2, 45:50, 45:50] = True

    result = _build_surround_neuropil_masks(masks, min_neuropil_pixels=30)
    dense = result.todense()
    assert result.shape == (3, NEUROPIL_H, NEUROPIL_W)
    assert dense[1].sum() == 0.0
    assert dense[0].sum() == pytest.approx(1.0)
    assert dense[2].sum() == pytest.approx(1.0)


def test_surround_masks_circular_differs_from_rectangular():
    """circular=True should actually change the ring geometry -- not covered by any other
    test, which all use the default circular=False."""
    mask = np.zeros((1, NEUROPIL_H, NEUROPIL_W), dtype=bool)
    mask[0, 28:36, 28:36] = True

    rectangular = _build_surround_neuropil_masks(mask, min_neuropil_pixels=30, circular=False)
    circular = _build_surround_neuropil_masks(mask, min_neuropil_pixels=30, circular=True)
    assert not np.allclose(rectangular.todense(), circular.todense())


def test_surround_masks_lam_percentile_affects_nearby_weighted_rois():
    """lam_percentile should actually change the ring for weighted, nearby ROIs (raising it
    excludes fewer low-weight edge pixels as "ROI territory", freeing more pixels for a
    neighboring ROI's ring) -- not covered by any other test, which all use the default 50.0."""
    yy, xx = np.meshgrid(np.arange(NEUROPIL_H), np.arange(NEUROPIL_W), indexing="ij")

    def _blob(cy, cx, radius):
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        return np.clip(1 - dist / radius, 0, 1).astype(np.float32)

    masks = np.stack([_blob(24, 32, 8), _blob(38, 32, 8)])  # close enough for fuzzy tails to interact

    default = _build_surround_neuropil_masks(masks, min_neuropil_pixels=30, inner_neuropil_radius=1)
    high_percentile = _build_surround_neuropil_masks(
        masks, min_neuropil_pixels=30, inner_neuropil_radius=1, lam_percentile=90.0
    )
    assert not np.allclose(default.todense(), high_percentile.todense())


def test_surround_masks_multiplane_confines_ring_to_same_plane():
    """Multi-plane input (Ly, Lx, n_planes) is solved per-plane -- each ROI's ring stays within
    its own plane, as long as every ROI is itself confined to a single plane (e.g. well-
    separated mesoscope planes)."""
    stats = _make_surround_stats([(15, 15), (45, 45)])
    dense_masks = np.zeros((2, NEUROPIL_H, NEUROPIL_W, 2), dtype=bool)
    for i, stat in enumerate(stats):
        dense_masks[i, stat["ypix"], stat["xpix"], i] = True  # ROI i lives entirely in plane i

    masks = _build_surround_neuropil_masks(dense_masks, min_neuropil_pixels=30)
    assert masks.shape == (2, NEUROPIL_H, NEUROPIL_W, 2)
    assert masks.dtype == np.float32
    dense = masks.todense()
    for i in range(2):
        assert dense[i].sum() == pytest.approx(1.0)
        assert dense[i, :, :, i].sum() == pytest.approx(1.0)  # entirely within its own plane


def test_surround_masks_roi_spanning_multiple_planes_raises():
    """A genuinely volumetric ROI (spanning more than one plane) isn't supported yet -- distinct
    from the well-separated-planes case above, which is."""
    masks = np.zeros((1, NEUROPIL_H, NEUROPIL_W, 2), dtype=bool)
    masks[0, 15, 15, 0] = True
    masks[0, 15, 15, 1] = True  # same ROI has pixels in both planes
    with pytest.raises(NotImplementedError, match="spans multiple planes"):
        _build_surround_neuropil_masks(masks, min_neuropil_pixels=30)


def test_surround_masks_multiplane_skips_planes_with_no_rois():
    """A declared plane that no ROI is assigned to (e.g. 3 planes, ROIs only in planes 0 and 2)
    should be skipped gracefully, not erroring or contributing any ring pixels."""
    stats = _make_surround_stats([(15, 15), (45, 45)])
    dense_masks = np.zeros((2, NEUROPIL_H, NEUROPIL_W, 3), dtype=bool)
    dense_masks[0, stats[0]["ypix"], stats[0]["xpix"], 0] = True  # ROI 0 in plane 0
    dense_masks[1, stats[1]["ypix"], stats[1]["xpix"], 2] = True  # ROI 1 in plane 2, plane 1 unused

    masks = _build_surround_neuropil_masks(dense_masks, min_neuropil_pixels=30)
    assert masks.shape == (2, NEUROPIL_H, NEUROPIL_W, 3)
    dense = masks.todense()
    assert dense[:, :, :, 1].sum() == 0.0  # unused plane gets no ring pixels from either ROI
    assert dense[0, :, :, 0].sum() == pytest.approx(1.0)
    assert dense[1, :, :, 2].sum() == pytest.approx(1.0)


def test_surround_masks_invalid_ndim_raises():
    """masks must be (n_rois, Ly, Lx) or (n_rois, Ly, Lx, n_planes) -- anything else is rejected
    explicitly rather than failing confusingly deeper in the suite2p calls."""
    with pytest.raises(ValueError, match="3 or 4 dimensions"):
        _build_surround_neuropil_masks(np.zeros((NEUROPIL_H, NEUROPIL_W)))  # missing ROI axis
    with pytest.raises(ValueError, match="3 or 4 dimensions"):
        _build_surround_neuropil_masks(np.zeros((2, NEUROPIL_H, NEUROPIL_W, 2, 1)))  # one axis too many


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
    assert params["method"] == "surround"
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
    """NeuropilExtension('surround') derives pixel coordinates from get_roi_image_masks(), so it
    works with any BaseRois, not only Suite2pRois -- this also matches the fact that
    create_roi_analyzer(..., format="memory") snapshots ROIs into a plain NumpyRois internally,
    so requiring a Suite2p-specific accessor would break even genuine Suite2pRois input."""
    analyzer = create_roi_analyzer(rois, imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    masks = analyzer.get_extension("neuropil").get_data()
    assert masks.shape == (NUM_ROIS, H, W)


def test_neuropil_extension_multiplane_well_separated_rois(surround_stats):
    """ROIs confined to a single plane each (e.g. well-separated mesoscope planes) are
    supported: each plane is solved as an independent 2D problem, end-to-end through the
    extension (not raising, unlike a genuinely volumetric ROI -- see
    test_surround_masks_roi_spanning_multiple_planes_raises)."""
    plane_assignments = np.array([0, 1, 0])  # centers are [(15, 15), (45, 45), (15, 45)]
    multiplane_rois = Suite2pRois.from_stat(
        surround_stats,
        shape=(NEUROPIL_H, NEUROPIL_W, 2),
        sampling_frequency=SF,
        plane_assignments=plane_assignments,
    )
    multiplane_imaging = generate_random_imaging(
        num_frames=NUM_FRAMES, height=NEUROPIL_H, width=NEUROPIL_W, num_planes=2, sampling_frequency=SF, seed=SEED
    )
    analyzer = create_roi_analyzer(multiplane_rois, multiplane_imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    masks = analyzer.get_extension("neuropil").get_data()
    assert masks.shape == (3, NEUROPIL_H, NEUROPIL_W, 2)
    dense = masks.todense()
    for i, plane in enumerate(plane_assignments):
        assert dense[i].sum() == pytest.approx(1.0)
        assert dense[i, :, :, plane].sum() == pytest.approx(1.0)


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


@pytest.mark.parametrize("neuropil_weight", [0.3, 1.0])
def test_fluorescence_extension_auto_uses_neuropil_extension(suite2p_rois, neuropil_imaging, neuropil_weight):
    """FluorescenceExtension should automatically pick up a computed NeuropilExtension.

    Uses weights other than FluorescenceNode's own default (0.7) so that a regression where
    the extension's ``neuropil_weight`` param stops being forwarded to the node (silently
    falling back to that unrelated default instead) would actually be caught.
    """
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    analyzer.compute("neuropil", min_neuropil_pixels=30)
    neuropil_masks = analyzer.get_extension("neuropil").get_data()

    analyzer.compute("fluorescence", use_neuropil=True, neuropil_weight=neuropil_weight)
    fluorescence = analyzer.get_extension("fluorescence").get_data()

    chunk = neuropil_imaging.get_series(epoch_index=0)
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    roi_masks_flat = suite2p_rois.get_roi_image_masks().todense().reshape(3, -1).astype(np.float32)
    roi_masks_flat = roi_masks_flat / roi_masks_flat.sum(axis=1, keepdims=True)
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
    roi_masks_flat = suite2p_rois.get_roi_image_masks().todense().reshape(3, -1).astype(np.float32)
    roi_masks_flat = roi_masks_flat / roi_masks_flat.sum(axis=1, keepdims=True)
    expected = chunk_flat @ roi_masks_flat.T

    np.testing.assert_allclose(fluorescence, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# NeuropilExtension ("cnmf" / CNMF-style low-rank background)
# ---------------------------------------------------------------------------

# A hand-built ground truth rather than generate_imaging_with_rois: that generator's `background`
# is a uniform, non-fluctuating scalar, so a low-rank fit on it is degenerate (f constant, b flat)
# and could not validate anything. Here Y is literally C A.T + f b.T + noise.
CNMF_H = CNMF_W = 24
CNMF_FRAMES = 200
CNMF_ROIS = 4
CNMF_CENTERS = [(3, 3), (3, 15), (15, 3), (15, 15)]
CNMF_SIDE = 4
# 8 rather than the 20px default: the synthetic field of view is only 24px across.
CNMF_HIGHPASS = 8.0


def _cnmf_masks(weighted=False):
    masks = np.zeros((CNMF_ROIS, CNMF_H, CNMF_W), dtype=np.float32)
    for i, (y, x) in enumerate(CNMF_CENTERS):
        block = np.ones((CNMF_SIDE, CNMF_SIDE), dtype=np.float32)
        if weighted:
            ramp = np.linspace(0.4, 1.0, CNMF_SIDE, dtype=np.float32)
            block = np.outer(ramp, ramp)
        masks[i, y : y + CNMF_SIDE, x : x + CNMF_SIDE] = block
    return masks


def _cnmf_ground_truth(num_frames=CNMF_FRAMES, weighted=False, background_scale=1.0, seed=0):
    """Return ``(masks, movie, traces_true, background_spatial_true, background_temporal_true)``.

    ``movie`` is ``(num_frames, H, W, 1)``; the two background factors are ``(n_pixels, 1)`` and
    ``(num_frames, 1)``.
    """
    from scipy.ndimage import gaussian_filter1d

    rng = np.random.default_rng(seed)
    masks = _cnmf_masks(weighted=weighted)
    masks_flat = masks.reshape(CNMF_ROIS, -1)

    traces = 100.0 + 50.0 * np.abs(gaussian_filter1d(rng.standard_normal((num_frames, CNMF_ROIS)), 3, axis=0))
    traces = traces.astype(np.float32)

    yy, xx = np.mgrid[0:CNMF_H, 0:CNMF_W]
    spatial = background_scale * (50.0 + 30.0 * np.exp(-((yy - 8) ** 2 + (xx - 16) ** 2) / 50.0) + 0.5 * yy)
    spatial = spatial.reshape(-1, 1).astype(np.float32)
    temporal = (
        (1.0 + 0.3 * np.sin(2 * np.pi * np.arange(num_frames) / num_frames) + 0.2 * np.arange(num_frames) / num_frames)
        .reshape(num_frames, 1)
        .astype(np.float32)
    )

    movie = traces @ masks_flat + temporal @ spatial.T
    movie = (movie + rng.normal(0.0, 0.5, movie.shape)).astype(np.float32)
    return masks, movie.reshape(num_frames, CNMF_H, CNMF_W, 1), traces, spatial, temporal


def _remove_direction(matrix, direction):
    """Project ``direction``'s column space out of every column of ``matrix``."""
    return matrix - direction @ (np.linalg.pinv(direction) @ matrix)


def _corr(a, b):
    return float(np.corrcoef(np.asarray(a).ravel(), np.asarray(b).ravel())[0, 1])


@pytest.fixture(scope="module")
def cnmf_truth():
    return _cnmf_ground_truth()


@pytest.fixture(scope="module")
def cnmf_analyzer(cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    imaging = NumpyImaging(movie, sampling_frequency=SF)
    rois = NumpyRois(roi_image_masks=masks, sampling_frequency=SF)
    analyzer = create_roi_analyzer(rois, imaging, format="memory")
    analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=40, highpass_sigma=CNMF_HIGHPASS)
    return analyzer


@pytest.fixture(scope="module")
def cnmf_extension(cnmf_analyzer):
    return cnmf_analyzer.get_extension("neuropil")


# --- numeric helpers -------------------------------------------------------


def test_spatial_bandpass_removes_a_constant_frame():
    chunk = np.full((3, 16, 16, 1), 7.0, dtype=np.float32)
    out = _spatial_bandpass(chunk, highpass_sigma=4.0, lowpass_sigma=0.0)
    # mode="nearest" keeps a constant frame constant under both blurs, so it cancels exactly.
    np.testing.assert_allclose(out, 0.0, atol=1e-4)


def test_spatial_bandpass_keeps_small_structure():
    chunk = np.zeros((1, 32, 32, 1), dtype=np.float32)
    chunk[0, 16, 16, 0] = 1.0
    chunk += 5.0  # a large constant pedestal the high-pass should remove
    out = _spatial_bandpass(chunk, highpass_sigma=8.0, lowpass_sigma=0.0)
    assert out[0, 16, 16, 0] > 0.5
    assert abs(out[0, 0, 0, 0]) < 1e-3


def test_spatial_bandpass_treats_planes_independently():
    chunk = np.zeros((2, 16, 16, 2), dtype=np.float32)
    chunk[..., 0] = 3.0
    chunk[0, 8, 8, 0] = 10.0
    out = _spatial_bandpass(chunk, highpass_sigma=4.0, lowpass_sigma=1.0)
    np.testing.assert_allclose(out[..., 1], 0.0, atol=1e-5)


def test_spatial_bandpass_disabled_is_identity():
    chunk = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4, 1)
    np.testing.assert_allclose(_spatial_bandpass(chunk, None, None), chunk)


def test_cnmf_objective_matches_brute_force():
    rng = np.random.default_rng(3)
    n_frames, n_pixels, n_rois, gnb = 12, 20, 3, 2
    movie = rng.standard_normal((n_frames, n_pixels))
    masks = rng.standard_normal((n_pixels, n_rois))
    traces = rng.standard_normal((n_frames, n_rois))
    background = rng.standard_normal((n_pixels, gnb))
    temporal = rng.standard_normal((n_frames, gnb))

    expected = float(((movie - traces @ masks.T - temporal @ background.T) ** 2).sum())
    got = _cnmf_objective(
        float((movie**2).sum()),
        traces,
        temporal,
        movie @ masks,
        movie @ background,
        masks.T @ masks,
        masks.T @ background,
        background.T @ background,
    )
    assert got == pytest.approx(expected, rel=1e-9)


def test_nnls_block_is_nonnegative_and_exact_for_scalar_gram():
    rng = np.random.default_rng(5)
    gram = np.array([[2.0]])
    rhs = rng.standard_normal((7, 1)) * 3.0
    got = _nnls_block(gram, rhs, np.zeros((7, 1)), n_iter=200)
    assert (got >= 0).all()
    # A 1x1 Gram makes the rows independent, so clipping the unconstrained solution is exact.
    np.testing.assert_allclose(got, np.maximum(rhs / 2.0, 0.0), atol=1e-9)


def test_nnls_block_does_not_increase_the_quadratic():
    rng = np.random.default_rng(6)
    gram = rng.standard_normal((3, 3))
    gram = gram @ gram.T + np.eye(3)
    rhs = rng.standard_normal((10, 3))
    x0 = np.abs(rng.standard_normal((10, 3)))

    def quad(x):
        return float(0.5 * np.sum((x @ gram) * x) - np.sum(x * rhs))

    assert quad(_nnls_block(gram, rhs, x0, n_iter=100)) <= quad(x0) + 1e-9


def test_ridge_inverse_is_scale_equivariant_per_block():
    # The joint [A, b] Gram mixes blocks whose diagonals differ by orders of magnitude; a ridge
    # scaled by the *mean* diagonal would badly over-penalise the small block.
    gram = np.diag([16.0, 16.0, 2.3e6])
    inverse = _ridge_inverse(gram, ridge=1e-6)
    np.testing.assert_allclose(np.diag(inverse), 1.0 / (np.diag(gram) * (1 + 1e-6)), rtol=1e-9)


def test_partition_of_unity_sums_to_one_per_pixel():
    weights = _partition_of_unity((5, 9, 1), gnb=3)
    assert weights.shape == (3, 45)
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=0), 1.0, rtol=1e-9)


# --- ground-truth recovery -------------------------------------------------


def test_cnmf_data_keys_shapes_and_dtypes(cnmf_extension):
    data = cnmf_extension.data
    assert set(data) == {
        "background_spatial",
        "background_temporal",
        "neuropil_traces",
        "demixed_fluorescence",
        "epoch_frame_offsets",
        "fit_info",
    }
    assert data["background_spatial"].shape == (1, CNMF_H, CNMF_W, 1)
    assert data["background_temporal"].shape == (1, CNMF_FRAMES)
    assert data["neuropil_traces"].shape == (CNMF_FRAMES, CNMF_ROIS)
    assert data["demixed_fluorescence"].shape == (CNMF_FRAMES, CNMF_ROIS)
    np.testing.assert_array_equal(data["epoch_frame_offsets"], [0, CNMF_FRAMES])
    for key in ("background_spatial", "background_temporal", "neuropil_traces", "demixed_fluorescence"):
        assert data[key].dtype == np.float32
        assert np.isfinite(data[key]).all()


def test_cnmf_background_factors_are_nonnegative_and_l2_normalised(cnmf_extension):
    spatial, temporal = cnmf_extension.get_background()
    assert (spatial >= 0).all()
    assert (temporal >= 0).all()
    assert np.linalg.norm(spatial.reshape(1, -1)[0]) == pytest.approx(1.0, rel=1e-5)


def test_cnmf_recovers_the_background_movie(cnmf_extension, cnmf_truth):
    _, _, _, spatial_true, temporal_true = cnmf_truth
    spatial, temporal = cnmf_extension.get_background()
    # Compare the reconstructed background *movie*: b and f are individually only defined up to a
    # per-component positive scale, but their product is not.
    got = temporal.T @ spatial.reshape(1, -1)
    expected = temporal_true @ spatial_true.T
    assert np.linalg.norm(got - expected) / np.linalg.norm(expected) < 0.05


def test_cnmf_recovers_the_background_timecourse(cnmf_extension, cnmf_truth):
    _, _, _, _, temporal_true = cnmf_truth
    _, temporal = cnmf_extension.get_background()
    assert _corr(temporal[0], temporal_true[:, 0]) > 0.99


def test_cnmf_recovers_the_background_away_from_rois(cnmf_extension, cnmf_truth):
    masks, _, _, spatial_true, _ = cnmf_truth
    spatial, _ = cnmf_extension.get_background()
    # Only off-ROI pixels are identifiable: b -> b + A alpha with C -> C - f alpha.T leaves the
    # model bit-for-bit unchanged, and that gauge lives exactly on the ROI support.
    off_roi = masks.reshape(CNMF_ROIS, -1).sum(axis=0) == 0
    assert _corr(spatial.reshape(-1)[off_roi], spatial_true[:, 0][off_roi]) > 0.99
    # On ROI pixels the background is only pinned by the initialisation's smoothness assumption,
    # so the whole-image agreement is real but looser than the off-ROI agreement.
    assert _corr(spatial.reshape(-1), spatial_true[:, 0]) > 0.95


def test_cnmf_recovers_traces_up_to_the_background_gauge(cnmf_extension, cnmf_truth):
    _, _, traces_true, _, _ = cnmf_truth
    _, temporal = cnmf_extension.get_background()
    traces = cnmf_extension.get_data("demixed_fluorescence")

    # The initialisation fixes the gauge sensibly, so the traces land close to ground truth outright.
    assert np.linalg.norm(traces - traces_true) / np.linalg.norm(traces_true) < 0.1

    # What residual error there is must lie in the single non-identifiable f direction...
    error = traces - traces_true
    residual = _remove_direction(error, temporal.T)
    assert np.linalg.norm(residual) / np.linalg.norm(error) < 0.2

    # ...so once that direction is projected out, the agreement is much tighter.
    projected = _remove_direction(traces, temporal.T)
    projected_true = _remove_direction(traces_true, temporal.T)
    assert np.linalg.norm(projected - projected_true) / np.linalg.norm(projected_true) < 0.02
    for i in range(CNMF_ROIS):
        assert _corr(projected[:, i], projected_true[:, i]) > 0.99


def test_cnmf_objective_is_monotone_and_matches_the_final_residual(cnmf_extension, cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    info = cnmf_extension.get_data("fit_info")
    objective = np.asarray(info["objective"])
    assert len(objective) >= 2
    # Every block update is a descent step, so the objective can only move backwards by float noise.
    assert np.all(np.diff(objective) <= 1e-9 * info["ynorm_sq"])

    flat = movie.reshape(CNMF_FRAMES, -1)
    spatial, temporal = cnmf_extension.get_background()
    traces = cnmf_extension.get_data("demixed_fluorescence")
    residual = flat - (traces @ masks.reshape(CNMF_ROIS, -1) + temporal.T @ spatial.reshape(1, -1))
    assert info["objective_full"] == pytest.approx(float((residual**2).sum()), rel=1e-3)


def test_cnmf_fit_info_holds_plain_python_scalars(cnmf_extension):
    import json

    info = cnmf_extension.get_data("fit_info")
    # The zarr backend serialises this with numcodecs.JSON() and binary_folder with check_json;
    # numpy scalars do not survive either.
    json.dumps(info)
    assert isinstance(info["gnb"], int)
    assert isinstance(info["converged"], bool)
    assert all(isinstance(v, float) for v in info["objective"])


def test_cnmf_neuropil_traces_match_the_background_projected_through_each_mask(cnmf_extension, cnmf_truth):
    masks, _, _, _, _ = cnmf_truth
    spatial, temporal = cnmf_extension.get_background()
    masks_flat = masks.reshape(CNMF_ROIS, -1)
    masks_l1 = masks_flat / masks_flat.sum(axis=1, keepdims=True)
    expected = temporal.T @ (masks_l1 @ spatial.reshape(-1, 1)).T
    np.testing.assert_allclose(cnmf_extension.get_data("neuropil_traces"), expected, rtol=1e-4)


def test_cnmf_runs_with_weighted_masks():
    masks, movie, _, _, temporal_true = _cnmf_ground_truth(weighted=True)
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    ext = analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=30, highpass_sigma=CNMF_HIGHPASS)
    _, temporal = ext.get_background()
    assert _corr(temporal[0], temporal_true[:, 0]) > 0.99


@pytest.mark.parametrize("init_method", ["ramp", "svd"])
def test_cnmf_init_methods_converge_to_a_similar_background(cnmf_truth, init_method):
    masks, movie, _, _, temporal_true = cnmf_truth
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    ext = analyzer.compute(
        "neuropil", method="cnmf", gnb=1, max_iter=40, highpass_sigma=CNMF_HIGHPASS, init_method=init_method
    )
    _, temporal = ext.get_background()
    assert _corr(temporal[0], temporal_true[:, 0]) > 0.99


def test_cnmf_with_several_background_components_runs(cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    ext = analyzer.compute("neuropil", method="cnmf", gnb=3, max_iter=15, highpass_sigma=CNMF_HIGHPASS)
    spatial, temporal = ext.get_background()
    assert spatial.shape == (3, CNMF_H, CNMF_W, 1)
    assert temporal.shape == (3, CNMF_FRAMES)
    assert (spatial >= 0).all() and (temporal >= 0).all()
    # Components are ordered by descending temporal energy.
    energies = np.linalg.norm(temporal, axis=1)
    assert np.all(np.diff(energies) <= 1e-6 * max(energies[0], 1.0))
    objective = np.asarray(ext.get_data("fit_info")["objective"])
    assert np.all(np.diff(objective) <= 1e-9 * ext.get_data("fit_info")["ynorm_sq"])


def test_cnmf_is_invariant_to_chunk_size(cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    results = []
    for chunk_duration in (None, "1s"):
        analyzer = create_roi_analyzer(
            NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
            NumpyImaging(movie, sampling_frequency=SF),
            format="memory",
        )
        kwargs = {} if chunk_duration is None else {"chunk_duration": chunk_duration}
        results.append(
            analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=20, highpass_sigma=CNMF_HIGHPASS, **kwargs)
        )
    for key in ("background_spatial", "background_temporal", "demixed_fluorescence", "neuropil_traces"):
        # rtol=1e-4, not 1e-7: float32 movie chunks are summed in a different order.
        np.testing.assert_allclose(results[0].get_data(key), results[1].get_data(key), rtol=1e-4, atol=1e-5)


def test_cnmf_concatenates_epochs_in_order(cnmf_truth):
    masks, movie, traces_true, spatial_true, temporal_true = cnmf_truth
    first = 120
    # Deliberately asymmetric, and only the *background* is brightened in the second epoch -- so
    # the recovered f, not C, has to carry the step. A symmetric test would pass even with the
    # epoch offsets wired backwards.
    boost = 3.0
    scaled_temporal = temporal_true.copy()
    scaled_temporal[first:] *= boost
    rebuilt = traces_true @ masks.reshape(CNMF_ROIS, -1) + scaled_temporal @ spatial_true.T
    rebuilt = rebuilt.astype(np.float32).reshape(CNMF_FRAMES, CNMF_H, CNMF_W, 1)
    epochs = [rebuilt[:first].copy(), rebuilt[first:].copy()]

    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(epochs, sampling_frequency=SF),
        format="memory",
    )
    ext = analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=30, highpass_sigma=CNMF_HIGHPASS)

    np.testing.assert_array_equal(ext.get_data("epoch_frame_offsets"), [0, first, CNMF_FRAMES])
    _, temporal = ext.get_background()
    assert temporal.shape == (1, CNMF_FRAMES)
    assert ext.get_data("demixed_fluorescence").shape == (CNMF_FRAMES, CNMF_ROIS)
    # The recovered timecourse must track the ground truth across the epoch boundary, which pins
    # both the ordering and the offsets.
    assert _corr(temporal[0], scaled_temporal[:, 0]) > 0.99
    # Against the ground truth's own between-epoch ratio, not `boost`: f_true is not flat, so the
    # two epochs' means differ for reasons other than the boost.
    expected_ratio = scaled_temporal[first:, 0].mean() / scaled_temporal[:first, 0].mean()
    ratio = temporal[0, first:].mean() / temporal[0, :first].mean()
    assert ratio == pytest.approx(expected_ratio, rel=0.05)
    assert expected_ratio > 2.0  # the boost really is visible in the second epoch
    assert movie.shape[0] == CNMF_FRAMES


def test_cnmf_handles_multiplane_masks():
    masks_2d, movie, _, _, _ = _cnmf_ground_truth(num_frames=60)
    masks = np.zeros((CNMF_ROIS, CNMF_H, CNMF_W, 2), dtype=np.float32)
    masks[..., 0] = masks_2d
    volume = np.concatenate([movie, movie * np.float32(0.5)], axis=3)
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(volume, sampling_frequency=SF),
        format="memory",
    )
    ext = analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=10, highpass_sigma=CNMF_HIGHPASS)
    spatial, _ = ext.get_background()
    assert spatial.shape == (1, CNMF_H, CNMF_W, 2)
    assert np.isfinite(ext.get_data("neuropil_traces")).all()


def test_cnmf_with_sparse_masks_matches_dense(cnmf_truth):
    import sparse as sparse_lib

    masks, movie, _, _, _ = cnmf_truth
    dense = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    ).compute("neuropil", method="cnmf", gnb=1, max_iter=10, highpass_sigma=CNMF_HIGHPASS)
    sparse_masks = sparse_lib.GCXS.from_numpy(masks, compressed_axes=(0,))
    spare = create_roi_analyzer(
        NumpyRois(roi_image_masks=sparse_masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    ).compute("neuropil", method="cnmf", gnb=1, max_iter=10, highpass_sigma=CNMF_HIGHPASS)
    np.testing.assert_allclose(
        dense.get_data("demixed_fluorescence"), spare.get_data("demixed_fluorescence"), rtol=1e-5, atol=1e-4
    )


# --- params, dispatch, error paths ----------------------------------------


def test_cnmf_default_params(analyzer):
    defaults = analyzer.get_default_extension_params("neuropil")
    assert defaults["gnb"] == 1
    assert defaults["max_iter"] == 20
    assert defaults["tol"] == 1e-4
    assert defaults["highpass_sigma"] == 20.0
    assert defaults["lowpass_sigma"] == 1.0
    assert defaults["init_method"] == "ramp"
    assert defaults["nonneg_background"] is True
    assert defaults["nonneg_traces"] is False
    assert defaults["subsample_frames"] == 1000


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"gnb": 0}, "gnb must be >= 1"),
        ({"init_method": "bogus"}, "Unknown init_method"),
        ({"subsample_frames": 1}, "subsample_frames must be None or >= 2"),
        ({"max_iter": 0}, "max_iter must be >= 1"),
    ],
)
def test_cnmf_invalid_params_raise(analyzer, kwargs, match):
    with pytest.raises(ValueError, match=match):
        analyzer.compute("neuropil", method="cnmf", **kwargs)


def test_cnmf_params_are_not_validated_for_surround(analyzer):
    # gnb is a cnmf-only knob, so an absurd value must not block the surround method.
    analyzer.compute("neuropil", method="surround", min_neuropil_pixels=30, gnb=0)


def test_neuropil_unknown_method_message_lists_both(analyzer):
    with pytest.raises(ValueError, match="Supported: 'surround', 'cnmf'"):
        analyzer.compute("neuropil", method="nope")


# --- data accessors --------------------------------------------------------


def test_cnmf_get_data_defaults_to_neuropil_traces(cnmf_extension):
    np.testing.assert_array_equal(cnmf_extension.get_data(), cnmf_extension.get_data("neuropil_traces"))


def test_cnmf_get_data_unknown_key_lists_the_available_ones(cnmf_extension):
    with pytest.raises(KeyError, match="available keys"):
        cnmf_extension.get_data("not_a_key")


def test_surround_get_data_still_returns_masks(imaging, rois):
    # Backward-compatibility lock on the _get_data signature change.
    analyzer = create_roi_analyzer(rois, imaging, format="memory")
    ext = analyzer.compute("neuropil", method="surround", min_neuropil_pixels=30)
    masks = ext.get_data()
    assert masks.shape == (NUM_ROIS, H, W)
    np.testing.assert_array_equal(masks.todense(), ext.get_data("neuropil_masks").todense())


def test_get_background_rejects_the_surround_method(imaging, rois):
    analyzer = create_roi_analyzer(rois, imaging, format="memory")
    ext = analyzer.compute("neuropil", method="surround", min_neuropil_pixels=30)
    with pytest.raises(ValueError, match="only available for method='cnmf'"):
        ext.get_background()


def test_cnmf_select_extension_data_keeps_every_key(cnmf_extension, cnmf_analyzer):
    keep = cnmf_analyzer.rois.roi_ids[:2]
    selected = cnmf_extension._select_extension_data(keep)
    # copy() assigns this dict straight over `data`, so a missing key is silently lost.
    assert set(selected) == set(cnmf_extension.data)
    assert selected["neuropil_traces"].shape == (CNMF_FRAMES, 2)
    assert selected["demixed_fluorescence"].shape == (CNMF_FRAMES, 2)
    assert selected["background_spatial"].shape == cnmf_extension.data["background_spatial"].shape
    np.testing.assert_array_equal(selected["demixed_fluorescence"], cnmf_extension.data["demixed_fluorescence"][:, :2])


def test_cnmf_survives_select_rois(cnmf_analyzer):
    keep = cnmf_analyzer.rois.roi_ids[:2]
    sub = cnmf_analyzer.select_rois(keep)
    ext = sub.get_extension("neuropil")
    assert ext.get_data("neuropil_traces").shape == (CNMF_FRAMES, 2)
    assert ext.get_data("background_temporal").shape == (1, CNMF_FRAMES)


@pytest.mark.parametrize("fmt", ["binary_folder", "zarr"])
def test_cnmf_roundtrips_through_disk(cnmf_truth, tmp_path, fmt):
    masks, movie, _, _, _ = cnmf_truth
    folder = tmp_path / ("cnmf_binary" if fmt == "binary_folder" else "cnmf.zarr")
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format=fmt,
        folder=folder,
    )
    ext = analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=10, highpass_sigma=CNMF_HIGHPASS)
    reloaded = load_roi_analyzer(folder).get_extension("neuropil")

    for key in (
        "background_spatial",
        "background_temporal",
        "neuropil_traces",
        "demixed_fluorescence",
        "epoch_frame_offsets",
    ):
        np.testing.assert_array_equal(reloaded.get_data(key), ext.get_data(key))
    assert reloaded.params["method"] == "cnmf"
    assert reloaded.get_data("fit_info")["gnb"] == 1
    assert reloaded.get_data("fit_info")["converged"] in (True, False)


# --- integration with FluorescenceExtension --------------------------------


def test_fluorescence_uses_cnmf_neuropil_traces(cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    neuropil = analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=20, highpass_sigma=CNMF_HIGHPASS)
    weight = 0.3
    corrected = analyzer.compute("fluorescence", neuropil_weight=weight).get_data()

    masks_flat = masks.reshape(CNMF_ROIS, -1).astype(np.float32)
    l1 = masks_flat.sum(axis=1, keepdims=True)
    rescale = (l1 / (masks_flat**2).sum(axis=1, keepdims=True)).T
    raw = movie.reshape(CNMF_FRAMES, -1) @ (masks_flat / l1).T
    expected = (raw - weight * neuropil.get_data("neuropil_traces")) * rescale
    np.testing.assert_allclose(corrected, expected, rtol=1e-4, atol=1e-3)


def test_fluorescence_cnmf_and_surround_subtract_on_the_same_scale(cnmf_truth):
    # Both methods must land in the same units, so that neuropil_weight means the same thing.
    masks, movie, _, _, _ = cnmf_truth
    imaging = NumpyImaging(movie, sampling_frequency=SF)
    rois = NumpyRois(roi_image_masks=masks, sampling_frequency=SF)

    plain = create_roi_analyzer(rois, imaging, format="memory")
    uncorrected = plain.compute("fluorescence", use_neuropil=False).get_data()

    cnmf = create_roi_analyzer(rois, imaging, format="memory")
    neuropil = cnmf.compute("neuropil", method="cnmf", gnb=1, max_iter=20, highpass_sigma=CNMF_HIGHPASS)
    corrected = cnmf.compute("fluorescence", neuropil_weight=1.0).get_data()

    masks_flat = masks.reshape(CNMF_ROIS, -1).astype(np.float32)
    rescale = (masks_flat.sum(axis=1) / (masks_flat**2).sum(axis=1)).reshape(1, -1)
    difference = (uncorrected - corrected) / rescale
    np.testing.assert_allclose(difference, neuropil.get_data("neuropil_traces"), rtol=1e-4, atol=1e-3)


def test_fluorescence_use_neuropil_false_ignores_a_cnmf_extension(cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=5, highpass_sigma=CNMF_HIGHPASS)
    traces = analyzer.compute("fluorescence", use_neuropil=False).get_data()

    masks_flat = masks.reshape(CNMF_ROIS, -1).astype(np.float32)
    l1 = masks_flat.sum(axis=1, keepdims=True)
    rescale = (l1 / (masks_flat**2).sum(axis=1, keepdims=True)).T
    expected = (movie.reshape(CNMF_FRAMES, -1) @ (masks_flat / l1).T) * rescale
    np.testing.assert_allclose(traces, expected, rtol=1e-4, atol=1e-3)


def test_cnmf_neuropil_correction_beats_no_correction(cnmf_truth):
    masks, movie, traces_true, _, temporal_true = cnmf_truth
    imaging = NumpyImaging(movie, sampling_frequency=SF)
    rois = NumpyRois(roi_image_masks=masks, sampling_frequency=SF)

    plain = create_roi_analyzer(rois, imaging, format="memory")
    uncorrected = plain.compute("fluorescence", use_neuropil=False).get_data()

    cnmf = create_roi_analyzer(rois, imaging, format="memory")
    cnmf.compute("neuropil", method="cnmf", gnb=1, max_iter=20, highpass_sigma=CNMF_HIGHPASS)
    corrected = cnmf.compute("fluorescence", neuropil_weight=1.0).get_data()

    # The whole point of the correction is to remove the background's f-shaped contamination, so
    # this is compared directly, without projecting that direction out.
    error_without = np.linalg.norm(uncorrected - traces_true)
    error_with = np.linalg.norm(corrected - traces_true)
    assert error_with < error_without / 5
    assert temporal_true.shape[0] == CNMF_FRAMES


def test_demixed_fluorescence_matches_mask_projection_without_background():
    # With binary non-overlapping masks, M M.T is diagonal, so the least-squares traces coincide
    # exactly with FluorescenceNode's L1-then-L2 rescaled mask projection. No fit convergence is
    # involved, which makes this a tight check on the units of demixed_fluorescence.
    masks, movie, _, _, _ = _cnmf_ground_truth(num_frames=80, background_scale=0.0, seed=11)
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    ext = analyzer.compute("neuropil", method="cnmf", gnb=1, max_iter=30, highpass_sigma=CNMF_HIGHPASS)
    projection = analyzer.compute("fluorescence", use_neuropil=False).get_data()
    demixed = ext.get_data("demixed_fluorescence")
    background = ext.get_data("neuropil_traces")
    # demixed = projection - (the background the fit attributed to each ROI), in the same units.
    np.testing.assert_allclose(demixed, projection - background, rtol=1e-3, atol=1e-2)


# --- FluorescenceNode-level tests ------------------------------------------


def test_fluorescence_node_rejects_both_neuropil_arguments(imaging, rois):
    with pytest.raises(ValueError, match="not both"):
        FluorescenceNode(
            imaging,
            rois,
            neuropil=np.zeros((NUM_ROIS, H, W), dtype=np.float32),
            neuropil_traces=np.zeros((NUM_FRAMES, NUM_ROIS), dtype=np.float32),
        )


def test_fluorescence_node_checks_the_neuropil_traces_shape(imaging, rois):
    with pytest.raises(ValueError, match="expected"):
        FluorescenceNode(imaging, rois, neuropil_traces=np.zeros((NUM_FRAMES, NUM_ROIS + 1), dtype=np.float32))


def test_fluorescence_node_rejects_a_nonzero_margin(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois, neuropil_traces=np.zeros((NUM_FRAMES, NUM_ROIS), dtype=np.float32))
    with pytest.raises(NotImplementedError, match="zero-margin"):
        node.compute(chunk, 0, NUM_FRAMES, 0, 3)


def test_fluorescence_node_slices_neuropil_traces_by_epoch(rois):
    first, second = 7, 5
    imaging = generate_random_imaging(num_frames=(first, second), height=H, width=W, sampling_frequency=SF, seed=SEED)
    traces = np.arange((first + second) * NUM_ROIS, dtype=np.float32).reshape(first + second, NUM_ROIS)
    node = FluorescenceNode(imaging, rois, neuropil_traces=traces, neuropil_weight=1.0)

    chunk = imaging.get_series(start_frame=2, end_frame=5, epoch_index=1)
    (got,) = node.compute(chunk, 2, 5, 1, 0)

    chunk_flat = chunk.reshape(3, -1).astype(np.float32)
    expected = (chunk_flat @ node._masks_flat.T - traces[first + 2 : first + 5]) * node._rescale_to_l2
    np.testing.assert_allclose(got, expected, rtol=1e-5)


@pytest.mark.parametrize("method_params", [{"method": "surround", "min_neuropil_pixels": 30}, None])
def test_multi_extension_compute_matches_sequential(imaging, rois, method_params, cnmf_truth):
    """The shared node-pipeline path must agree with computing one extension at a time.

    It used to raise ``TypeError: 'NoneType' is not iterable`` because ``FluorescenceExtension``
    never declared ``nodepipeline_variables``, so this whole path was dead and any post-gather
    neuropil subtraction would have silently been skipped on it.
    """
    if method_params is None:
        masks, movie, _, _, _ = cnmf_truth
        imaging = NumpyImaging(movie, sampling_frequency=SF)
        rois = NumpyRois(roi_image_masks=masks, sampling_frequency=SF)
        method_params = {"method": "cnmf", "gnb": 1, "max_iter": 5, "highpass_sigma": CNMF_HIGHPASS}

    together = create_roi_analyzer(rois, imaging, format="memory")
    together.compute({"neuropil": method_params, "fluorescence": {"neuropil_weight": 0.5}})

    one_by_one = create_roi_analyzer(rois, imaging, format="memory")
    one_by_one.compute("neuropil", **method_params)
    one_by_one.compute("fluorescence", neuropil_weight=0.5)

    np.testing.assert_allclose(
        together.get_extension("fluorescence").get_data(),
        one_by_one.get_extension("fluorescence").get_data(),
        rtol=1e-6,
    )
