"""Tests for FluorescenceNode, FluorescenceExtension, DfOverFExtension, NeuropilExtension, BackgroundExtension,
DeconvolutionExtension."""

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
    result = node.compute(chunk)
    assert isinstance(result, tuple)
    assert len(result) == 2  # (fluorescence, background)


def test_compute_output_shape(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois)
    fluorescence, _ = node.compute(chunk)
    assert fluorescence.shape == (NUM_FRAMES, NUM_ROIS)


def test_compute_output_dtype(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois)
    fluorescence, _ = node.compute(chunk)
    assert fluorescence.dtype == np.float32


def test_compute_matches_manual_weighted_sum(imaging, rois, chunk):
    """Verify compute() matches a simple loop over ROIs.

    FluorescenceNode normalizes each ROI's mask to sum to 1 internally (see its docstring),
    so "manual" here means normalizing rois.get_roi_image_masks() the same way before
    comparing -- for the binary masks used here, that's equivalent to dividing by pixel count.
    """
    node = FluorescenceNode(imaging, rois)
    fluorescence, _ = node.compute(chunk)

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
    fluorescence, _ = node.compute(chunk)
    np.testing.assert_array_equal(fluorescence, 0.0)


def test_compute_matches_dense_result_with_sparse_masks(imaging, chunk):
    """FluorescenceNode should give the same result for sparse (e.g. Suite2p-backed) ROIs
    as for dense ones, since it operates polymorphically on whatever get_roi_image_masks
    returns (see photon-mosaic#103)."""
    sparse_rois = generate_rois(num_rois=NUM_ROIS, height=H, width=W, sampling_frequency=SF, seed=SEED, sparse=True)
    dense_rois = generate_rois(num_rois=NUM_ROIS, height=H, width=W, sampling_frequency=SF, seed=SEED)

    fluorescence_sparse, _ = FluorescenceNode(imaging, sparse_rois).compute(chunk)
    fluorescence_dense, _ = FluorescenceNode(imaging, dense_rois).compute(chunk)
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
    fluorescence, _ = node.compute(chunk)

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
    fluorescence, _ = node.compute(chunk)

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
    fluorescence, _ = node.compute(chunk)

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
    fluorescence, _ = node.compute(chunk)
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
    fluorescence, _ = node.compute(chunk)

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
    fluorescence, _ = node.compute(chunk)

    # The global neuropil trace is the sum of each frame
    chunk_flat = chunk.reshape(NUM_FRAMES, -1).astype(np.float32)
    global_trace = chunk_flat.sum(axis=1, keepdims=True)  # (T, 1)

    # Without neuropil
    node_no_np = FluorescenceNode(imaging, rois)
    fluor_no_np, _ = node_no_np.compute(chunk)

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
    fluorescence, _ = node.compute(chunk)

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
    fluor_weighted, _ = node_weighted.compute(chunk)

    node_none = FluorescenceNode(imaging, rois, neuropil=None)
    fluor_none, _ = node_none.compute(chunk)

    np.testing.assert_allclose(fluor_weighted, fluor_none, rtol=1e-5)


def test_neuropil_weight_default(imaging, rois, chunk):
    """The default neuropil_weight should be 0.7."""
    rng = np.random.default_rng(654)
    neuropil = rng.random((NUM_ROIS, H, W)).astype(np.float32)

    node_default = FluorescenceNode(imaging, rois, neuropil=neuropil)
    fluor_default, _ = node_default.compute(chunk)

    node_explicit = FluorescenceNode(imaging, rois, neuropil=neuropil, neuropil_weight=0.7)
    fluor_explicit, _ = node_explicit.compute(chunk)

    np.testing.assert_allclose(fluor_default, fluor_explicit, rtol=1e-5)


# ---------------------------------------------------------------------------
# No neuropil
# ---------------------------------------------------------------------------


def test_no_neuropil_returns_normalized_weighted_sum(imaging, rois, chunk):
    node = FluorescenceNode(imaging, rois, neuropil=None)
    fluorescence, _ = node.compute(chunk)

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
    fluorescence, _ = node.compute(chunk)
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
    fluorescence, _ = node.compute(chunk)
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
# BackgroundExtension ("cnmf" / CNMF-style low-rank background)
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
    analyzer.compute("background", gnb=1, max_iter=40, highpass_sigma=CNMF_HIGHPASS)
    return analyzer


@pytest.fixture(scope="module")
def cnmf_extension(cnmf_analyzer):
    return cnmf_analyzer.get_extension("background")


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


def _cnmf_fluorescence(masks, movie, method="regression", background_kwargs=None, **fluorescence_kwargs):
    """Compute a CNMF background then the joint extraction on it; return the fluorescence extension.

    The extension holds its analyzer only weakly, so callers that need ``ext.roi_analyzer`` must
    keep the analyzer alive themselves: use :func:`_cnmf_analyzer_and_fluorescence`.
    """
    return _cnmf_analyzer_and_fluorescence(masks, movie, method, background_kwargs, **fluorescence_kwargs)[1]


def _cnmf_analyzer_and_fluorescence(masks, movie, method="regression", background_kwargs=None, **fluorescence_kwargs):
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    params = dict(gnb=1, max_iter=40, highpass_sigma=CNMF_HIGHPASS)
    params.update(background_kwargs or {})
    analyzer.compute("background", **params)
    return analyzer, analyzer.compute("fluorescence", method=method, **fluorescence_kwargs)


@pytest.fixture(scope="module")
def cnmf_fluorescence_analyzer(cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    return _cnmf_analyzer_and_fluorescence(masks, movie)[0]


@pytest.fixture(scope="module")
def cnmf_fluorescence(cnmf_fluorescence_analyzer):
    return cnmf_fluorescence_analyzer.get_extension("fluorescence")


def test_cnmf_data_keys_shapes_and_dtypes(cnmf_extension):
    data = cnmf_extension.data
    # Only b: C and f are solved by FluorescenceExtension, which reads the whole movie anyway.
    assert set(data) == {"background_spatial"}
    assert data["background_spatial"].shape == (1, CNMF_H, CNMF_W, 1)
    assert data["background_spatial"].dtype == np.float32
    assert np.isfinite(data["background_spatial"]).all()


def test_cnmf_background_is_nonnegative_and_l2_normalised(cnmf_extension):
    spatial = cnmf_extension.get_data()
    assert (spatial >= 0).all()
    assert np.linalg.norm(spatial.reshape(-1)) == pytest.approx(1.0, rel=1e-5)


def test_cnmf_recovers_the_background_away_from_rois(cnmf_extension, cnmf_truth):
    masks, _, _, spatial_true, _ = cnmf_truth
    spatial = cnmf_extension.get_data()
    # Only off-ROI pixels are identifiable: b -> b + A alpha with C -> C - f alpha.T leaves the
    # model bit-for-bit unchanged, and that gauge lives exactly on the ROI support.
    off_roi = masks.reshape(CNMF_ROIS, -1).sum(axis=0) == 0
    assert _corr(spatial.reshape(-1)[off_roi], spatial_true[:, 0][off_roi]) > 0.99


def test_cnmf_joint_extraction_recovers_the_background_timecourse(cnmf_fluorescence, cnmf_truth):
    _, _, _, _, temporal_true = cnmf_truth
    # Each ROI's background is f times a per-ROI constant, so every column must track f.
    background = cnmf_fluorescence.get_data(key="background")
    for i in range(CNMF_ROIS):
        assert _corr(background[:, i], temporal_true[:, 0]) > 0.99


def test_cnmf_joint_extraction_recovers_traces_up_to_the_background_gauge(cnmf_fluorescence, cnmf_truth):
    _, _, traces_true, _, temporal_true = cnmf_truth
    traces = cnmf_fluorescence.get_data()
    # The initialisation fixes the gauge sensibly, so the traces land close to ground truth outright...
    assert np.linalg.norm(traces - traces_true) / np.linalg.norm(traces_true) < 0.1
    # ...and once the non-identifiable f direction is projected out, the agreement is much tighter.
    projected = _remove_direction(traces, temporal_true)
    projected_true = _remove_direction(traces_true, temporal_true)
    assert np.linalg.norm(projected - projected_true) / np.linalg.norm(projected_true) < 0.02


def test_cnmf_concatenates_epochs_in_order(cnmf_truth):
    masks, _, traces_true, spatial_true, temporal_true = cnmf_truth
    first = 120
    # Deliberately asymmetric, and only the *background* is brightened in the second epoch -- so
    # the recovered f, not C, has to carry the step. A symmetric test would pass even with the
    # epochs wired in the wrong order.
    boost = 3.0
    scaled_temporal = temporal_true.copy()
    scaled_temporal[first:] *= boost
    rebuilt = traces_true @ masks.reshape(CNMF_ROIS, -1) + scaled_temporal @ spatial_true.T
    rebuilt = rebuilt.astype(np.float32).reshape(CNMF_FRAMES, CNMF_H, CNMF_W, 1)
    epochs = [rebuilt[:first].copy(), rebuilt[first:].copy()]

    ext = _cnmf_fluorescence(masks, epochs, background_kwargs={"max_iter": 30})
    background = ext.get_data(key="background")
    assert background.shape == (CNMF_FRAMES, CNMF_ROIS)
    assert ext.get_data().shape == (CNMF_FRAMES, CNMF_ROIS)
    assert _corr(background[:, 0], scaled_temporal[:, 0]) > 0.99
    # Against the ground truth's own between-epoch ratio, not `boost`: f_true is not flat, so the
    # two epochs' means differ for reasons other than the boost.
    expected_ratio = scaled_temporal[first:, 0].mean() / scaled_temporal[:first, 0].mean()
    ratio = background[first:, 0].mean() / background[:first, 0].mean()
    assert ratio == pytest.approx(expected_ratio, rel=0.05)
    assert expected_ratio > 2.0  # the boost really is visible in the second epoch


def test_cnmf_handles_multiplane_masks():
    masks_2d, movie, _, _, _ = _cnmf_ground_truth(num_frames=60)
    masks = np.zeros((CNMF_ROIS, CNMF_H, CNMF_W, 2), dtype=np.float32)
    masks[..., 0] = masks_2d
    volume = np.concatenate([movie, movie * np.float32(0.5)], axis=3)
    analyzer, ext = _cnmf_analyzer_and_fluorescence(masks, volume, background_kwargs={"max_iter": 10})
    spatial = analyzer.get_extension("background").get_data()
    assert spatial.shape == (1, CNMF_H, CNMF_W, 2)
    assert np.isfinite(ext.get_data()).all()
    assert np.isfinite(ext.get_data(key="background")).all()


def test_cnmf_with_sparse_masks_matches_dense(cnmf_truth):
    import sparse as sparse_lib

    masks, movie, _, _, _ = cnmf_truth
    dense = _cnmf_fluorescence(masks, movie, background_kwargs={"max_iter": 10})
    spare = _cnmf_fluorescence(
        sparse_lib.GCXS.from_numpy(masks, compressed_axes=(0,)), movie, background_kwargs={"max_iter": 10}
    )
    np.testing.assert_allclose(dense.get_data(), spare.get_data(), rtol=1e-5, atol=1e-4)


# --- params, dispatch, error paths ----------------------------------------


def test_cnmf_default_params(analyzer):
    defaults = analyzer.get_default_extension_params("background")
    assert defaults["method"] == "cnmf"
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
        ({"method": "surround"}, "Supported: 'cnmf'"),
    ],
)
def test_cnmf_invalid_params_raise(analyzer, kwargs, match):
    with pytest.raises(ValueError, match=match):
        analyzer.compute("background", **kwargs)


def test_neuropil_rejects_cnmf_params(analyzer):
    # The CNMF knobs now live on BackgroundExtension only.
    with pytest.raises(TypeError, match="gnb"):
        analyzer.compute("neuropil", method="surround", min_neuropil_pixels=30, gnb=1)


def test_neuropil_rejects_cnmf_method(analyzer):
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.compute("neuropil", method="cnmf")


# --- data accessors --------------------------------------------------------


def test_cnmf_select_extension_data_keeps_every_key(cnmf_extension, cnmf_analyzer):
    keep = cnmf_analyzer.rois.roi_ids[:2]
    selected = cnmf_extension._select_extension_data(keep)
    # copy() assigns this dict straight over `data`, so a missing key is silently lost.
    assert set(selected) == set(cnmf_extension.data)
    # b belongs to the whole field of view, so it passes through unsliced.
    np.testing.assert_array_equal(selected["background_spatial"], cnmf_extension.data["background_spatial"])


def test_cnmf_fluorescence_survives_select_rois(cnmf_fluorescence_analyzer, cnmf_fluorescence):
    analyzer = cnmf_fluorescence_analyzer
    keep = analyzer.rois.roi_ids[:2]
    sub = analyzer.select_rois(keep)
    ext = sub.get_extension("fluorescence")
    np.testing.assert_array_equal(ext.get_data(), cnmf_fluorescence.get_data()[:, :2])
    np.testing.assert_array_equal(ext.get_data(key="background"), cnmf_fluorescence.get_data(key="background")[:, :2])
    assert sub.get_extension("background").get_data().shape == (1, CNMF_H, CNMF_W, 1)


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
    background = analyzer.compute("background", gnb=1, max_iter=10, highpass_sigma=CNMF_HIGHPASS)
    fluorescence = analyzer.compute("fluorescence", method="regression")
    reloaded = load_roi_analyzer(folder)

    assert reloaded.get_extension("background").params["method"] == "cnmf"
    np.testing.assert_array_equal(reloaded.get_extension("background").get_data(), background.get_data())
    assert reloaded.get_extension("fluorescence").params["method"] == "regression"
    for key in ("fluorescence", "background"):
        np.testing.assert_array_equal(
            reloaded.get_extension("fluorescence").get_data(key=key), fluorescence.get_data(key=key)
        )


# --- integration with FluorescenceExtension --------------------------------


def test_fluorescence_default_params(analyzer):
    defaults = analyzer.get_default_extension_params("fluorescence")
    assert defaults["method"] == "projection"
    assert defaults["ridge"] == 1e-6


def test_fluorescence_unknown_method_raises(analyzer):
    with pytest.raises(ValueError, match="Unknown method"):
        analyzer.compute("fluorescence", method="nope")


def test_fluorescence_cnmf_background_requires_a_joint_method(cnmf_analyzer):
    with pytest.raises(ValueError, match="method='regression'"):
        cnmf_analyzer.compute("fluorescence", method="projection")


def test_fluorescence_background_key_needs_a_neuropil(analyzer):
    ext = analyzer.compute("fluorescence")
    assert ext.data["background"].shape == (NUM_FRAMES, 0)
    with pytest.raises(ValueError, match="No background"):
        ext.get_data(key="background")


def test_fluorescence_use_neuropil_false_ignores_a_cnmf_extension(cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    analyzer.compute("background", gnb=1, max_iter=5, highpass_sigma=CNMF_HIGHPASS)
    traces = analyzer.compute("fluorescence", use_neuropil=False).get_data()

    masks_flat = masks.reshape(CNMF_ROIS, -1).astype(np.float32)
    l1 = masks_flat.sum(axis=1, keepdims=True)
    rescale = (l1 / (masks_flat**2).sum(axis=1, keepdims=True)).T
    expected = (movie.reshape(CNMF_FRAMES, -1) @ (masks_flat / l1).T) * rescale
    np.testing.assert_allclose(traces, expected, rtol=1e-4, atol=1e-3)


# --- neuropil_source: choosing between NeuropilExtension and BackgroundExtension --------------


def _analyzer_with(cnmf_truth, *corrections):
    masks, movie, _, _, _ = cnmf_truth
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    if "neuropil" in corrections:
        analyzer.compute("neuropil", min_neuropil_pixels=30)
    if "background" in corrections:
        analyzer.compute("background", gnb=1, max_iter=5, highpass_sigma=CNMF_HIGHPASS)
    return analyzer


@pytest.fixture(scope="module")
def both_corrections_analyzer(cnmf_truth):
    return _analyzer_with(cnmf_truth, "neuropil", "background")


def test_fluorescence_default_neuropil_source_is_none(analyzer):
    assert analyzer.get_default_extension_params("fluorescence")["neuropil_source"] is None


def test_fluorescence_unknown_neuropil_source_raises(analyzer):
    with pytest.raises(ValueError, match="Unknown neuropil_source"):
        analyzer.compute("fluorescence", neuropil_source="bogus")


def test_fluorescence_with_both_corrections_needs_a_neuropil_source(both_corrections_analyzer):
    with pytest.raises(ValueError, match="Both 'neuropil' and 'background' are computed"):
        both_corrections_analyzer.compute("fluorescence", method="regression")


@pytest.mark.parametrize("source", ["neuropil", "background"])
def test_fluorescence_neuropil_source_must_be_computed(cnmf_truth, source):
    other = "background" if source == "neuropil" else "neuropil"
    analyzer = _analyzer_with(cnmf_truth, other)
    with pytest.raises(ValueError, match=f"the '{source}' extension has not been computed"):
        analyzer.compute("fluorescence", method="regression", neuropil_source=source)


@pytest.mark.parametrize("source", ["neuropil", "background"])
def test_fluorescence_neuropil_source_picks_that_correction(cnmf_truth, both_corrections_analyzer, source):
    # With both computed, naming one must give exactly what having only that one gives.
    only = _analyzer_with(cnmf_truth, source).compute("fluorescence", method="regression")
    picked = both_corrections_analyzer.compute("fluorescence", method="regression", neuropil_source=source)
    for key in ("fluorescence", "background"):
        np.testing.assert_array_equal(picked.get_data(key=key), only.get_data(key=key))


@pytest.mark.parametrize("source", ["neuropil", "background"])
def test_fluorescence_auto_picks_the_only_computed_correction(cnmf_truth, source):
    analyzer = _analyzer_with(cnmf_truth, source)
    auto = analyzer.compute("fluorescence", method="regression").get_data().copy()
    named = analyzer.compute("fluorescence", method="regression", neuropil_source=source).get_data()
    np.testing.assert_array_equal(auto, named)


def test_df_over_f_ignores_the_background_when_neuropil_source_is_neuropil(both_corrections_analyzer):
    # A computed BackgroundExtension must not leak into dF/F when the traces were not fitted with it.
    both_corrections_analyzer.compute("fluorescence", method="regression", neuropil_source="neuropil")
    kwargs = dict(method="percentile", prctile_baseline=8.0)
    with_background = both_corrections_analyzer.compute("df_over_f", **kwargs).get_data().copy()
    without = both_corrections_analyzer.compute("df_over_f", use_background=False, **kwargs).get_data()
    np.testing.assert_array_equal(with_background, without)


@pytest.mark.parametrize("method", ["regression", "nnls"])
def test_cnmf_joint_extraction_beats_no_correction(cnmf_truth, method):
    masks, movie, traces_true, _, _ = cnmf_truth
    imaging = NumpyImaging(movie, sampling_frequency=SF)
    rois = NumpyRois(roi_image_masks=masks, sampling_frequency=SF)
    uncorrected = create_roi_analyzer(rois, imaging, format="memory").compute("fluorescence").get_data()
    corrected = _cnmf_fluorescence(masks, movie, method=method).get_data()
    # The whole point is to remove the background's f-shaped contamination, so this is compared
    # directly, without projecting that direction out.
    assert np.linalg.norm(corrected - traces_true) < np.linalg.norm(uncorrected - traces_true) / 5


@pytest.mark.parametrize("neuropil_weight", [0.0, 0.7, 1.0])
def test_cnmf_traces_are_the_joint_solve_with_no_neuropil_subtraction(cnmf_truth, neuropil_weight):
    # With a CNMF background, C is the ROI block of the joint [C, f] least-squares solve, returned
    # as is: the background is modelled, not subtracted, so neuropil_weight must have no effect.
    masks, movie, _, _, _ = cnmf_truth
    analyzer, ext = _cnmf_analyzer_and_fluorescence(masks, movie, neuropil_weight=neuropil_weight)
    b = analyzer.get_extension("background").get_data().reshape(1, -1)
    design = np.concatenate([masks.reshape(CNMF_ROIS, -1), b], axis=0).T.astype(np.float64)
    expected, *_ = np.linalg.lstsq(design, movie.reshape(CNMF_FRAMES, -1).T.astype(np.float64), rcond=None)
    np.testing.assert_allclose(ext.get_data(), expected[:CNMF_ROIS].T, rtol=1e-4, atol=1e-2)


@pytest.mark.parametrize("weighted", [False, True])
def test_cnmf_traces_plus_background_equal_the_regression_without_background(weighted):
    # The first block of the normal equations reads Y A.T = C A A.T + f (A b).T. For non-overlapping
    # masks A A.T is diagonal, so C + B (B = f (A b).T / ||a_i||^2) is exactly the regression without
    # b. That pins the units of `background` to those of C.
    masks, movie, _, _, _ = _cnmf_ground_truth(weighted=weighted)
    ext = _cnmf_fluorescence(masks, movie)
    plain = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    ).compute("fluorescence", method="regression", use_neuropil=False)
    np.testing.assert_allclose(ext.get_data() + ext.get_data(key="background"), plain.get_data(), rtol=1e-4, atol=1e-2)


def test_regression_matches_projection_for_nonoverlapping_masks(cnmf_truth):
    # With non-overlapping masks A A.T is diagonal, so the least-squares traces are exactly the
    # L2-rescaled projection -- for weighted masks too.
    masks = _cnmf_masks(weighted=True)
    _, movie, _, _, _ = cnmf_truth
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    projection = analyzer.compute("fluorescence").get_data()
    regression = analyzer.compute("fluorescence", method="regression").get_data()
    np.testing.assert_allclose(regression, projection, rtol=1e-4, atol=1e-3)


def _overlapping_problem(num_frames=40, seed=5):
    """Two ROIs sharing a quarter of their pixels, with an exactly known movie ``C A.T``."""
    rng = np.random.default_rng(seed)
    masks = np.zeros((2, 12, 12), dtype=np.float32)
    masks[0, 2:8, 2:8] = 1.0
    masks[1, 5:11, 5:11] = 1.0
    traces = rng.uniform(0.0, 10.0, (num_frames, 2)).astype(np.float32)
    traces[::4, 1] = 0.0  # frames where the second ROI is silent
    movie = (traces @ masks.reshape(2, -1)).reshape(num_frames, 12, 12, 1)
    return masks, movie.astype(np.float32), traces


@pytest.mark.parametrize("method", ["regression", "nnls"])
def test_joint_methods_separate_overlapping_rois(method):
    masks, movie, traces = _overlapping_problem()
    analyzer = create_roi_analyzer(
        NumpyRois(roi_image_masks=masks, sampling_frequency=SF),
        NumpyImaging(movie, sampling_frequency=SF),
        format="memory",
    )
    projection = analyzer.compute("fluorescence").get_data()
    joint = analyzer.compute("fluorescence", method=method).get_data()
    np.testing.assert_allclose(joint, traces, rtol=1e-4, atol=1e-3)
    assert np.abs(projection - traces).max() > 1.0  # the crosstalk the joint solve removes


def test_nnls_matches_a_bounded_least_squares_solve_per_frame(cnmf_truth):
    from scipy.optimize import lsq_linear

    masks, movie, _, _, _ = cnmf_truth
    # Take away more than the ~100 trace offset from the ROI pixels, more than a non-negative b can
    # hand back there, so some traces have to hit the C >= 0 bound.
    movie = movie - np.float32(150.0) * masks.sum(axis=0)[None, :, :, None]
    analyzer, ext = _cnmf_analyzer_and_fluorescence(masks, movie, method="nnls", background_kwargs={"max_iter": 10})
    traces = ext.get_data()
    assert (traces >= 0).all()
    assert (traces == 0).any()

    b = analyzer.get_extension("background").get_data().reshape(1, -1)
    design = np.concatenate([masks.reshape(CNMF_ROIS, -1), b], axis=0).T.astype(np.float64)
    lower = np.r_[np.zeros(CNMF_ROIS), -np.inf]
    for t in (0, 57, 199):
        expected = lsq_linear(design, movie[t].reshape(-1).astype(np.float64), bounds=(lower, np.inf), tol=1e-12).x
        np.testing.assert_allclose(traces[t], expected[:CNMF_ROIS], rtol=1e-3, atol=1e-2)


def test_regression_with_surround_neuropil_matches_projection(suite2p_rois, neuropil_imaging):
    # Surround is a two-step correction either way; for binary non-overlapping masks the regression
    # traces equal the projection, so the subtracted results must too.
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    analyzer.compute("neuropil", method="surround")
    projection = analyzer.compute("fluorescence", neuropil_weight=0.7).get_data()
    regression = analyzer.compute("fluorescence", neuropil_weight=0.7, method="regression").get_data()
    np.testing.assert_allclose(regression, projection, rtol=1e-4, atol=1e-2)


@pytest.mark.parametrize("method", ["projection", "regression"])
def test_surround_persists_its_ring_mean(suite2p_rois, neuropil_imaging, method):
    # The ring mean Fneu used to be discarded per chunk; it is now persisted, before neuropil_weight.
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    ring = analyzer.compute("neuropil", method="surround").get_data()
    ext = analyzer.compute("fluorescence", neuropil_weight=0.7, method=method)
    uncorrected = analyzer.compute("fluorescence", use_neuropil=False, method=method).get_data()

    movie = neuropil_imaging.get_series(epoch_index=0)
    expected_fneu = movie.reshape(movie.shape[0], -1) @ ring.reshape((ring.shape[0], -1)).todense().T
    np.testing.assert_allclose(ext.get_data(key="background"), expected_fneu, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(ext.get_data() + 0.7 * ext.get_data(key="background"), uncorrected, rtol=1e-4, atol=1e-2)


def test_cnmf_background_is_projected_onto_each_rois_own_footprint():
    # B = f b A.T / ||a_i||^2: each ROI's own footprint only, even where ROIs overlap -- not mixed
    # through inv(A A.T). Checked against an independent least-squares solve for f.
    masks, movie, _ = _overlapping_problem()
    yy, xx = np.mgrid[0:12, 0:12]
    b = (1.0 + 0.1 * yy + 0.05 * xx).astype(np.float32)
    f = np.linspace(20.0, 40.0, movie.shape[0], dtype=np.float32)
    movie = movie + (f[:, None, None] * b)[..., None]
    imaging = NumpyImaging(movie, sampling_frequency=SF)
    rois = NumpyRois(roi_image_masks=masks, sampling_frequency=SF)
    node = FluorescenceNode(imaging, rois, background_spatial=b[None, ..., None], method="regression")
    traces, background = node.compute(movie)

    masks_flat = masks.reshape(2, -1).astype(np.float64)
    design = np.concatenate([masks_flat, b.reshape(1, -1)], axis=0).T
    solution, *_ = np.linalg.lstsq(design, movie.reshape(movie.shape[0], -1).T.astype(np.float64), rcond=None)
    f_solved = solution[2:].T  # (T, 1)
    expected = f_solved @ (masks_flat @ b.reshape(-1, 1)).T / (masks_flat**2).sum(axis=1)
    np.testing.assert_allclose(traces, solution[:2].T, rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(background, expected, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("method", ["regression", "nnls"])
def test_joint_methods_keep_an_all_zero_mask_at_zero(method):
    masks, movie, _ = _overlapping_problem()
    masks = np.concatenate([masks, np.zeros((1, 12, 12), dtype=np.float32)])
    imaging = NumpyImaging(movie, sampling_frequency=SF)
    rois = NumpyRois(roi_image_masks=masks, sampling_frequency=SF)
    b = np.ones((1, 12, 12, 1), dtype=np.float32)
    traces, background = FluorescenceNode(imaging, rois, background_spatial=b, method=method).compute(movie)
    assert np.isfinite(traces).all() and np.isfinite(background).all()
    np.testing.assert_allclose(traces[:, 2], 0.0, atol=1e-6)
    np.testing.assert_allclose(background[:, 2], 0.0, atol=1e-6)


@pytest.mark.parametrize("method", ["regression", "nnls"])
def test_cnmf_joint_extraction_is_invariant_to_chunk_size(cnmf_truth, method):
    masks, movie, _, _, _ = cnmf_truth
    whole = _cnmf_fluorescence(masks, movie, method=method, background_kwargs={"max_iter": 10})
    chunked = _cnmf_fluorescence(masks, movie, method=method, background_kwargs={"max_iter": 10}, chunk_duration="1s")
    for key in ("fluorescence", "background"):
        np.testing.assert_allclose(whole.get_data(key=key), chunked.get_data(key=key), rtol=1e-4, atol=1e-3)


# --- FluorescenceNode-level tests ------------------------------------------


def test_fluorescence_node_rejects_both_neuropil_arguments(imaging, rois):
    with pytest.raises(ValueError, match="not both"):
        FluorescenceNode(
            imaging,
            rois,
            neuropil=np.zeros((NUM_ROIS, H, W), dtype=np.float32),
            background_spatial=np.zeros((1, H, W, 1), dtype=np.float32),
            method="regression",
        )


def test_fluorescence_node_checks_the_background_pixel_count(imaging, rois):
    with pytest.raises(ValueError, match="pixels per component"):
        FluorescenceNode(imaging, rois, background_spatial=np.ones((1, H + 1, W, 1)), method="regression")


@pytest.mark.parametrize("method_params", [{"method": "surround", "min_neuropil_pixels": 30}, None])
def test_multi_extension_compute_matches_sequential(imaging, rois, method_params, cnmf_truth):
    """The shared node-pipeline path must agree with computing one extension at a time.

    It used to raise ``TypeError: 'NoneType' is not iterable`` because ``FluorescenceExtension``
    never declared ``nodepipeline_variables``, so this whole path was dead and any post-gather
    neuropil subtraction would have silently been skipped on it.
    """
    fluorescence_params = {"neuropil_weight": 0.5}
    correction = "neuropil"
    if method_params is None:
        masks, movie, _, _, _ = cnmf_truth
        imaging = NumpyImaging(movie, sampling_frequency=SF)
        rois = NumpyRois(roi_image_masks=masks, sampling_frequency=SF)
        correction = "background"
        method_params = {"gnb": 1, "max_iter": 5, "highpass_sigma": CNMF_HIGHPASS}
        fluorescence_params["method"] = "regression"

    together = create_roi_analyzer(rois, imaging, format="memory")
    together.compute({correction: method_params, "fluorescence": fluorescence_params})

    one_by_one = create_roi_analyzer(rois, imaging, format="memory")
    one_by_one.compute(correction, **method_params)
    one_by_one.compute("fluorescence", **fluorescence_params)

    for key in ("fluorescence", "background"):
        np.testing.assert_array_equal(
            together.get_extension("fluorescence").data[key], one_by_one.get_extension("fluorescence").data[key]
        )


# --- DfOverFExtension with a CNMF background (item 3 of the #111 "Proposed split") -------------


def _percentile_baselines(traces, win, prct):
    from scipy.ndimage import percentile_filter

    return np.stack([percentile_filter(traces[:, i], prct, size=win) for i in range(traces.shape[1])], axis=1)


@pytest.fixture(scope="module")
def cnmf_df_over_f_analyzer(cnmf_truth):
    masks, movie, _, _, _ = cnmf_truth
    return _cnmf_analyzer_and_fluorescence(masks, movie)[0]


def test_df_over_f_default_uses_the_background(analyzer):
    assert analyzer.get_default_extension_params("df_over_f")["use_background"] is True


@pytest.mark.parametrize("prctile_baseline", [8.0, 50.0])
def test_df_over_f_adds_the_cnmf_background_baseline_to_the_denominator(cnmf_df_over_f_analyzer, prctile_baseline):
    # CaImAn's detrend_df_f: (C - C_baseline) / (B_baseline + C_baseline).
    win_s = 2.0
    ext = cnmf_df_over_f_analyzer.compute(
        "df_over_f", method="percentile", win_baseline=win_s, prctile_baseline=prctile_baseline
    )
    fluorescence = cnmf_df_over_f_analyzer.get_extension("fluorescence")
    C, B = fluorescence.get_data(), fluorescence.get_data(key="background")
    win = int(win_s * SF)
    c0, b0 = _percentile_baselines(C, win, prctile_baseline), _percentile_baselines(B, win, prctile_baseline)
    np.testing.assert_allclose(ext.get_data(), (C - c0) / (b0 + c0), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(ext.data["f0"], b0 + c0, rtol=1e-5)


def test_df_over_f_uses_the_percentile_chosen_from_c_for_b_too(cnmf_df_over_f_analyzer):
    # With prctile_baseline=None the KDE picks a percentile per ROI from C alone, and that same
    # percentile is applied to B -- never one estimated from B.
    win_s = 2.0
    ext = cnmf_df_over_f_analyzer.compute("df_over_f", method="percentile", win_baseline=win_s, prctile_baseline=None)
    fluorescence = cnmf_df_over_f_analyzer.get_extension("fluorescence")
    C, B = fluorescence.get_data(), fluorescence.get_data(key="background")
    win = int(win_s * SF)
    for i in range(CNMF_ROIS):
        prct = _kde_mode_percentile(C[:win, i].astype(np.float64))
        c0 = _percentile_baselines(C[:, i : i + 1], win, prct)[:, 0]
        b0 = _percentile_baselines(B[:, i : i + 1], win, prct)[:, 0]
        np.testing.assert_allclose(ext.data["f0"][:, i], b0 + c0, rtol=1e-5)


@pytest.mark.parametrize("method", ["percentile", "maximin"])
def test_df_over_f_is_unchanged_when_baseline_moves_between_c_and_b(cnmf_truth, method):
    # The fit can't tell how much of a cell's baseline belongs in C and how much in the background.
    # Moving a constant per ROI from C into B must leave dF/F as it was: C_baseline drops and
    # B_baseline rises by the same amount, so the denominator does not move.
    masks, movie, _, _, _ = cnmf_truth
    analyzer, fluorescence = _cnmf_analyzer_and_fluorescence(masks, movie, background_kwargs={"max_iter": 10})
    kwargs = dict(method=method, win_baseline=2.0, prctile_baseline=20.0)
    before = analyzer.compute("df_over_f", **kwargs).get_data().copy()

    shift = np.linspace(10.0, 40.0, CNMF_ROIS, dtype=np.float32)[None, :]
    fluorescence.data["fluorescence"] = fluorescence.data["fluorescence"] - shift
    fluorescence.data["background"] = fluorescence.data["background"] + shift
    after = analyzer.compute("df_over_f", **kwargs).get_data()
    np.testing.assert_allclose(after, before, rtol=1e-4, atol=1e-5)

    # Using C's baseline alone is thrown off by the same shift.
    c_only = analyzer.compute("df_over_f", use_background=False, **kwargs).get_data()
    assert np.abs(c_only - before).max() > 1e-2


def test_df_over_f_use_background_false_ignores_it(cnmf_df_over_f_analyzer):
    kwargs = dict(method="percentile", win_baseline=2.0, prctile_baseline=8.0)
    ext = cnmf_df_over_f_analyzer.compute("df_over_f", use_background=False, **kwargs)
    C = cnmf_df_over_f_analyzer.get_extension("fluorescence").get_data()
    c0 = _percentile_baselines(C, int(2.0 * SF), 8.0)
    np.testing.assert_allclose(ext.get_data(), (C - c0) / c0, rtol=1e-4, atol=1e-6)


def test_df_over_f_ignores_the_surround_ring_mean(suite2p_rois, neuropil_imaging):
    # For surround, `background` is Fneu, already subtracted from F; it is not part of F's baseline.
    analyzer = create_roi_analyzer(suite2p_rois, neuropil_imaging, format="memory")
    analyzer.compute("neuropil", method="surround")
    analyzer.compute("fluorescence")
    kwargs = dict(method="percentile", prctile_baseline=8.0)
    with_background = analyzer.compute("df_over_f", **kwargs).get_data().copy()
    without = analyzer.compute("df_over_f", use_background=False, **kwargs).get_data()
    np.testing.assert_array_equal(with_background, without)


def test_df_over_f_with_background_parallel_matches_serial(cnmf_df_over_f_analyzer):
    kwargs = dict(method="percentile", win_baseline=2.0, prctile_baseline=None)
    serial = cnmf_df_over_f_analyzer.compute("df_over_f", n_jobs=1, **kwargs).get_data().copy()
    parallel = cnmf_df_over_f_analyzer.compute("df_over_f", n_jobs=2, **kwargs).get_data()
    np.testing.assert_allclose(serial, parallel, rtol=1e-6)
