"""Tests for FluorescenceNode, FluorescenceExtension, DfOverFExtension, and DeconvolutionExtension."""

import numpy as np
import pytest

from photon_mosaic.core import create_roi_analyzer
from photon_mosaic.core.generators import generate_fluorescence, generate_random_imaging, generate_rois
from photon_mosaic.core.roianalyzer_core_extensions import (
    FluorescenceNode,
    _kde_mode_percentile,
)

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
    """ProcessPoolExecutor result should match serial computation exactly."""
    analyzer_with_df_over_f.compute("deconvolution", n_jobs=1)
    serial = analyzer_with_df_over_f.get_extension("deconvolution").get_data().copy()
    analyzer_with_df_over_f.compute("deconvolution", n_jobs=2)
    parallel = analyzer_with_df_over_f.get_extension("deconvolution").get_data()
    np.testing.assert_array_equal(serial, parallel)


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
