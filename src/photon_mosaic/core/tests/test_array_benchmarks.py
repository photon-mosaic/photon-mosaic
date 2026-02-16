"""Parametrized benchmarks for ArrayImaging / BinaryImaging / ZarrImaging.

The benchmark grid is designed to find where each backend wins or loses.
The main sweep axes are:

* **num_planes**       – total planes on disk (1, 10, 50)
* **queried_planes**   – 1 plane vs all planes
* **start_frame**      – beginning (0), mid-file, or near-end (random-seek test)
* **num_frames_read**  – how many frames per call (10, 100, 500)
* **zarr time-chunk**  – aligned vs misaligned with the query window

For zarr specifically, each dataset is written with a *large* temporal
chunk (``T_CHUNK_DISK = 256``) so that small reads and mid-chunk reads
force partial-chunk decompression.  At read time we also test an
*overridden* dask chunk size (``T_CHUNK_READ = 64``) which is deliberately
discordant with the on-disk layout.

Binary's strength is pure byte-offset seek — no decompression, no chunk
boundaries.  The grid is constructed so that many queries land *between*
zarr chunk boundaries, which is the scenario most likely to show a
binary advantage.

Run the full grid::

    pytest -m grid --benchmark-json=benchmark_results.json

Run only the small grid::

    pytest -m "grid and small" --benchmark-json=benchmark_results.json

Run only the large grid::

    pytest -m "grid and large" --benchmark-json=benchmark_results.json
"""

import shutil

import dask.array as da
import numpy as np
import pytest

from photon_mosaic.core.arrayimaging import ArrayImaging
from photon_mosaic.core.binaryimaging import BinaryImaging
from photon_mosaic.core.generators import (
    write_random_binary,
    write_random_npy,
    write_random_zarr,
)
from photon_mosaic.core.zarrimaging import ZarrImaging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DTYPE = np.uint16
SAMPLING_FREQUENCY = 30.0
HEIGHT = 128
WIDTH = 128

# Plane options: single-plane vs many-plane
NUM_PLANES_OPTIONS = [1, 50]

# Frame-read sizes: small slice vs large slice
NUM_FRAMES_READ_OPTIONS = [10, 500]

# Total frames per dataset
SMALL_NUM_FRAMES = 1000
LARGE_NUM_FRAMES = 4000

# Zarr on-disk temporal chunk – deliberately large so small reads land
# *inside* a chunk and mid-file reads straddle chunk boundaries.
T_CHUNK_DISK = 256

# An alternative dask read-chunk that is intentionally discordant with
# T_CHUNK_DISK (not a divisor/multiple).
T_CHUNK_READ = 64


def _zarr_disk_chunks(num_planes):
    """Chunk layout used when *writing* the zarr store."""
    return (T_CHUNK_DISK, HEIGHT, WIDTH, 1)


def _zarr_read_chunks(num_planes):
    """Discordant chunk layout used when *reading* with dask."""
    return (T_CHUNK_READ, HEIGHT, WIDTH, 1)


# ---------------------------------------------------------------------------
# Start-frame positions
# ---------------------------------------------------------------------------


def _start_positions(total_frames: int) -> list[tuple[str, int]]:
    """Return (label, start_frame) pairs.

    Positions are chosen so that the mid query lands between zarr chunk
    boundaries (T_CHUNK_DISK = 256):
      - "start"  → 0        (aligned with chunk 0)
      - "mid"    → 370      (inside chunk [256..512), offset 114 frames in)
    """
    mid = min(370, total_frames // 2)
    positions = [
        ("start", 0),
        ("mid", mid),
    ]
    return positions


# ---------------------------------------------------------------------------
# Queried-plane strategies (simplified: 1 plane or all)
# ---------------------------------------------------------------------------


def _queried_planes(num_planes: int) -> list[tuple[str, list[int]]]:
    """Return (label, plane_ids) pairs.

    Two strategies:
    - "1plane" : read only plane 0
    - "all"    : read every plane
    """
    strategies = [
        ("1plane", [0]),
        ("all", list(range(num_planes))),
    ]
    # Deduplicate when num_planes == 1
    seen = set()
    unique = []
    for label, ids in strategies:
        key = tuple(ids)
        if key not in seen:
            seen.add(key)
            unique.append((label, ids))
    return unique


# ---------------------------------------------------------------------------
# Build the parameter grid
# ---------------------------------------------------------------------------


def _build_param_grid(total_frames: int):
    """Yield pytest.param(num_planes, plane_ids, start_frame, nf_read, id=...)."""
    for num_planes in NUM_PLANES_OPTIONS:
        for plabel, plane_ids in _queried_planes(num_planes):
            for slabel, start in _start_positions(total_frames):
                for nf_read in NUM_FRAMES_READ_OPTIONS:
                    if start + nf_read > total_frames:
                        continue
                    test_id = f"P{num_planes}-{plabel}-{slabel}-F{nf_read}"
                    yield pytest.param(
                        num_planes,
                        plane_ids,
                        start,
                        nf_read,
                        id=test_id,
                    )


SMALL_GRID = list(_build_param_grid(SMALL_NUM_FRAMES))
LARGE_GRID = list(_build_param_grid(LARGE_NUM_FRAMES))


# ===================================================================
# Fixture caches
# ===================================================================


@pytest.fixture(scope="session")
def _file_cache(tmp_path_factory):
    """Session-wide dict mapping (backend, total_frames, num_planes, ...) → path."""
    cache = {}
    yield cache
    for path in cache.values():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)


def _get_or_create_npy(file_cache, tmp_path_factory, total_frames, num_planes):
    key = ("npy", total_frames, num_planes)
    if key not in file_cache:
        shape = (total_frames, HEIGHT, WIDTH, num_planes)
        path = tmp_path_factory.mktemp(f"npy_T{total_frames}_P{num_planes}") / "video.npy"
        write_random_npy(path, shape, dtype=DTYPE, seed=0)
        file_cache[key] = path
    return file_cache[key]


def _get_or_create_binary(file_cache, tmp_path_factory, total_frames, num_planes):
    key = ("bin", total_frames, num_planes)
    if key not in file_cache:
        shape = (total_frames, HEIGHT, WIDTH, num_planes)
        path = tmp_path_factory.mktemp(f"bin_T{total_frames}_P{num_planes}") / "video.bin"
        write_random_binary(path, shape, dtype=DTYPE, seed=0)
        file_cache[key] = path
    return file_cache[key]


def _get_or_create_zarr(file_cache, tmp_path_factory, total_frames, num_planes):
    key = ("zarr", total_frames, num_planes)
    if key not in file_cache:
        shape = (total_frames, HEIGHT, WIDTH, num_planes)
        chunks = _zarr_disk_chunks(num_planes)
        path = tmp_path_factory.mktemp(f"zarr_T{total_frames}_P{num_planes}") / "video.zarr"
        write_random_zarr(path, shape, dtype=DTYPE, chunks=chunks, seed=0)
        file_cache[key] = path
    return file_cache[key]


# ===================================================================
# Expected-shape helper
# ===================================================================


def _expected_shape(num_frames_read, num_queried_planes):
    return (num_frames_read, HEIGHT, WIDTH, num_queried_planes)


# ###################################################################
#  SMALL-DATA GRID
# ###################################################################


# --- npy full-load (baseline) ------------------------------------
@pytest.mark.small
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", SMALL_GRID)
def test_small_npy_full_load(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """Load entire .npy into RAM, wrap in ArrayImaging, then slice."""
    path = _get_or_create_npy(_file_cache, tmp_path_factory, SMALL_NUM_FRAMES, num_planes)

    def _load_and_slice():
        video = np.load(str(path))
        imaging = ArrayImaging(video, sampling_frequency=SAMPLING_FREQUENCY)
        return imaging.get_series(start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids)

    result = benchmark(_load_and_slice)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# --- npy memmap ---------------------------------------------------
@pytest.mark.small
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", SMALL_GRID)
def test_small_npy_memmap(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """Open .npy as memmap and slice via ArrayImaging."""
    path = _get_or_create_npy(_file_cache, tmp_path_factory, SMALL_NUM_FRAMES, num_planes)
    video = np.load(str(path), mmap_mode="r")
    imaging = ArrayImaging(video, sampling_frequency=SAMPLING_FREQUENCY)

    result = benchmark(imaging.get_series, start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# --- binary memmap ------------------------------------------------
@pytest.mark.small
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", SMALL_GRID)
def test_small_binary_memmap(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """BinaryImaging memmap – direct byte-offset seek."""
    path = _get_or_create_binary(_file_cache, tmp_path_factory, SMALL_NUM_FRAMES, num_planes)
    shape = (HEIGHT, WIDTH, num_planes)
    imaging = BinaryImaging(
        file_paths=str(path),
        sampling_frequency=SAMPLING_FREQUENCY,
        shape=shape,
        dtype=DTYPE,
    )
    result = benchmark(imaging.get_series, start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# --- zarr (native chunks) ----------------------------------------
@pytest.mark.small
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", SMALL_GRID)
def test_small_zarr_native(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """ZarrImaging using the on-disk chunk layout (T_CHUNK_DISK=256)."""
    path = _get_or_create_zarr(_file_cache, tmp_path_factory, SMALL_NUM_FRAMES, num_planes)
    imaging = ZarrImaging(
        zarr_paths=str(path),
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    def _read():
        return imaging.get_series(start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids).compute()

    result = benchmark(_read)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# --- zarr (discordant read-chunks) --------------------------------
@pytest.mark.small
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", SMALL_GRID)
def test_small_zarr_rechunked(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """ZarrImaging with dask read-chunks (T_CHUNK_READ=64) misaligned
    with the on-disk layout (T_CHUNK_DISK=256)."""
    path = _get_or_create_zarr(_file_cache, tmp_path_factory, SMALL_NUM_FRAMES, num_planes)
    imaging = ZarrImaging(
        zarr_paths=str(path),
        sampling_frequency=SAMPLING_FREQUENCY,
        chunks=_zarr_read_chunks(num_planes),
    )

    def _read():
        return imaging.get_series(start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids).compute()

    result = benchmark(_read)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# ###################################################################
#  LARGE-DATA GRID  (out-of-core only — skip full npy load)
# ###################################################################


# --- npy memmap ---------------------------------------------------
@pytest.mark.large
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", LARGE_GRID)
def test_large_npy_memmap(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """Open large .npy as memmap and slice via ArrayImaging."""
    path = _get_or_create_npy(_file_cache, tmp_path_factory, LARGE_NUM_FRAMES, num_planes)
    video = np.load(str(path), mmap_mode="r")
    imaging = ArrayImaging(video, sampling_frequency=SAMPLING_FREQUENCY)

    result = benchmark(imaging.get_series, start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids)
    # check the datatype... is it a numpy array or a mem meap view?
    assert isinstance(result, np.ndarray), "Expected a numpy array, got {}".format(type(result))
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# --- npy dask-over-memmap -----------------------------------------
@pytest.mark.large
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", LARGE_GRID)
def test_large_npy_dask(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """Open large .npy as memmap, wrap in dask, slice via ArrayImaging."""
    path = _get_or_create_npy(_file_cache, tmp_path_factory, LARGE_NUM_FRAMES, num_planes)
    mmap = np.load(str(path), mmap_mode="r")
    da_video = da.from_array(mmap, chunks=_zarr_read_chunks(num_planes))
    imaging = ArrayImaging(da_video, sampling_frequency=SAMPLING_FREQUENCY)

    def _read():
        return imaging.get_series(start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids).compute()

    result = benchmark(_read)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# --- binary memmap ------------------------------------------------
@pytest.mark.large
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", LARGE_GRID)
def test_large_binary_memmap(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """BinaryImaging memmap on large dataset."""
    path = _get_or_create_binary(_file_cache, tmp_path_factory, LARGE_NUM_FRAMES, num_planes)
    shape = (HEIGHT, WIDTH, num_planes)
    imaging = BinaryImaging(
        file_paths=str(path),
        sampling_frequency=SAMPLING_FREQUENCY,
        shape=shape,
        dtype=DTYPE,
    )
    result = benchmark(imaging.get_series, start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# --- zarr native chunks -------------------------------------------
@pytest.mark.large
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", LARGE_GRID)
def test_large_zarr_native(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """ZarrImaging using on-disk chunk layout on large dataset."""
    path = _get_or_create_zarr(_file_cache, tmp_path_factory, LARGE_NUM_FRAMES, num_planes)
    imaging = ZarrImaging(
        zarr_paths=str(path),
        sampling_frequency=SAMPLING_FREQUENCY,
    )

    def _read():
        return imaging.get_series(start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids).compute()

    result = benchmark(_read)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))


# --- zarr discordant read-chunks ----------------------------------
@pytest.mark.large
@pytest.mark.grid
@pytest.mark.parametrize("num_planes,plane_ids,start,nf_read", LARGE_GRID)
def test_large_zarr_rechunked(
    benchmark,
    _file_cache,
    tmp_path_factory,
    num_planes,
    plane_ids,
    start,
    nf_read,
):
    """ZarrImaging with discordant dask read-chunks on large dataset."""
    path = _get_or_create_zarr(_file_cache, tmp_path_factory, LARGE_NUM_FRAMES, num_planes)
    imaging = ZarrImaging(
        zarr_paths=str(path),
        sampling_frequency=SAMPLING_FREQUENCY,
        chunks=_zarr_read_chunks(num_planes),
    )

    def _read():
        return imaging.get_series(start_frame=start, end_frame=start + nf_read, plane_ids=plane_ids).compute()

    result = benchmark(_read)
    assert result.shape == _expected_shape(nf_read, len(plane_ids))
