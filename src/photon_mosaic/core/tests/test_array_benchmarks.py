import shutil

import numpy as np
import pytest

import dask.array as da

from photon_mosaic.core.arrayimaging import ArrayImaging
from photon_mosaic.core.binaryimaging import BinaryImaging
from photon_mosaic.core.zarrimaging import ZarrImaging
from photon_mosaic.core.generators import (
    write_random_binary,
    write_random_npy,
    write_random_zarr,
)

# ---------------------------------------------------------------------------
# Shape & dtype constants
# ---------------------------------------------------------------------------

DTYPE = np.uint16
SAMPLING_FREQUENCY = 30.0

# Small – fits in 16 GB RAM
# (1000, 256, 256, 50) × 2 bytes ≈ 6.1 GB
SMALL_SHAPE = (1000, 256, 256, 50)

# Large – does NOT fit in 16 GB RAM
# (4000, 256, 256, 50) × 2 bytes ≈ 24.4 GB
LARGE_SHAPE = (4000, 256, 256, 50)

QUERY = dict(start_frame=0, end_frame=100, plane_ids=[3])
CHUNKS = (100, 256, 256, 1)


# ===================================================================
# Fixtures – small data
# ===================================================================


@pytest.fixture(scope="session")
def small_npy_path(tmp_path_factory):
    """Write a small .npy file once per session."""
    path = tmp_path_factory.mktemp("small_npy") / "video.npy"
    write_random_npy(path, SMALL_SHAPE, dtype=DTYPE, seed=0)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def small_binary_path(tmp_path_factory):
    """Write a small binary file once per session."""
    path = tmp_path_factory.mktemp("small_bin") / "video.bin"
    write_random_binary(path, SMALL_SHAPE, dtype=DTYPE, seed=0)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def small_zarr_path(tmp_path_factory):
    """Small zarr store chunked by time + plane."""
    path = tmp_path_factory.mktemp("small_zarr") / "video.zarr"
    write_random_zarr(path, SMALL_SHAPE, dtype=DTYPE, chunks=CHUNKS, seed=0)
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ===================================================================
# Fixtures – large data (> 16 GB)
# ===================================================================


@pytest.fixture(scope="session")
def large_binary_path(tmp_path_factory):
    """Write a large binary file once per session (~24.4 GB)."""
    path = tmp_path_factory.mktemp("large_bin") / "video.bin"
    write_random_binary(path, LARGE_SHAPE, dtype=DTYPE, seed=0)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def large_zarr_path(tmp_path_factory):
    """Large zarr store chunked by time + plane (~24.4 GB)."""
    chunks = (100, LARGE_SHAPE[1], LARGE_SHAPE[2], 1)
    path = tmp_path_factory.mktemp("large_zarr") / "video.zarr"
    write_random_zarr(path, LARGE_SHAPE, dtype=DTYPE, chunks=chunks, seed=0)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def large_npy_path(tmp_path_factory):
    """Write a large .npy file once per session (~24.4 GB)."""
    path = tmp_path_factory.mktemp("large_npy") / "video.npy"
    write_random_npy(path, LARGE_SHAPE, dtype=DTYPE, seed=0)
    yield path
    path.unlink(missing_ok=True)


# ===================================================================
# Helper
# ===================================================================


def _expected_shape(dataset_shape: tuple) -> tuple:
    """Expected output shape for the QUERY constants."""
    num_frames = QUERY["end_frame"] - QUERY["start_frame"]
    _, h, w, _ = dataset_shape
    num_planes = len(QUERY["plane_ids"])
    return (num_frames, h, w, num_planes)


# ###################################################################
#  SMALL DATA BENCHMARKS
# ###################################################################


# --- 1. numpy .npy file → full load into RAM + slice ------------------
@pytest.mark.small
def test_small_npy_full_load(benchmark, small_npy_path):
    """Load entire .npy from disk, wrap in ArrayImaging, then slice.

    This is the baseline: numpy *must* load the whole file into RAM
    before any slicing can happen.
    """

    def _load_and_slice():
        video = np.load(str(small_npy_path))
        imaging = ArrayImaging(video, sampling_frequency=SAMPLING_FREQUENCY)
        return imaging.get_series(**QUERY)

    result = benchmark(_load_and_slice)
    assert result.shape == _expected_shape(SMALL_SHAPE)


# --- 2. Small binary → BinaryImaging (memmap, slice only) -------------
@pytest.mark.small
def test_small_binary_memmap(benchmark, small_binary_path):
    imaging = BinaryImaging(
        file_paths=str(small_binary_path),
        sampling_frequency=SAMPLING_FREQUENCY,
        shape=SMALL_SHAPE[1:],  # (H, W, P)
        dtype=DTYPE,
    )
    result = benchmark(imaging.get_series, **QUERY)
    assert result.shape == _expected_shape(SMALL_SHAPE)


# --- 3. Small zarr → ZarrImaging (dask, slice only) -------------------
@pytest.mark.small
def test_small_zarr_dask(benchmark, small_zarr_path):
    imaging = ZarrImaging(
        zarr_paths=str(small_zarr_path),
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    def _read():
        return imaging.get_series(**QUERY).compute()

    result = benchmark(_read)
    assert result.shape == _expected_shape(SMALL_SHAPE)


# ###################################################################
#  LARGE DATA BENCHMARKS  (> 16 GB – out-of-core only)
# ###################################################################


# --- 4. Large binary → BinaryImaging (memmap) -------------------------
@pytest.mark.large
def test_large_binary_memmap(benchmark, large_binary_path):
    imaging = BinaryImaging(
        file_paths=str(large_binary_path),
        sampling_frequency=SAMPLING_FREQUENCY,
        shape=LARGE_SHAPE[1:],
        dtype=DTYPE,
    )
    result = benchmark(imaging.get_series, **QUERY)
    assert result.shape == _expected_shape(LARGE_SHAPE)


# --- 5. Large zarr → ZarrImaging (dask) -------------------------------
@pytest.mark.large
def test_large_zarr_dask(benchmark, large_zarr_path):
    imaging = ZarrImaging(
        zarr_paths=str(large_zarr_path),
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    def _read():
        return imaging.get_series(**QUERY).compute()

    result = benchmark(_read)
    assert result.shape == _expected_shape(LARGE_SHAPE)


# --- 6. Large .npy → memmap (slice only, never full-load) -------------
@pytest.mark.large
def test_large_npy_memmap(benchmark, large_npy_path):
    """Open the .npy as a memory-mapped array and slice via ArrayImaging.

    Like a TIFF-style file: the data sits on disk and we only read the
    pages we need.
    """
    video = np.load(str(large_npy_path), mmap_mode="r")
    imaging = ArrayImaging(video, sampling_frequency=SAMPLING_FREQUENCY)
    result = benchmark(imaging.get_series, **QUERY)
    assert result.shape == _expected_shape(LARGE_SHAPE)


# --- 7. Large .npy → dask over memmap (slice only) --------------------
@pytest.mark.large
def test_large_npy_dask(benchmark, large_npy_path):
    """Open the .npy as memmap, wrap in dask, slice via ArrayImaging.

    Combines numpy-on-disk with dask's lazy/chunked execution.
    """
    mmap = np.load(str(large_npy_path), mmap_mode="r")
    da_video = da.from_array(mmap, chunks=CHUNKS)
    imaging = ArrayImaging(da_video, sampling_frequency=SAMPLING_FREQUENCY)

    def _read():
        return imaging.get_series(**QUERY).compute()

    result = benchmark(_read)
    assert result.shape == _expected_shape(LARGE_SHAPE)