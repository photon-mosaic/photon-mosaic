"""Spike-inference benchmark on real data: photon-mosaic vs. CaImAn on paired imaging + loose-seal ephys.

Dataset: DANDI 000168 (Zhang et al., jGCaMP8; simultaneous cell-attached recordings and two-photon imaging of
GCaMP-expressing mouse V1 neurons). Each file is one patched cell: a 122 Hz movie, the ROIs of the field of view and the
patched cell's spike times. The movie is downloaded from DANDI (nothing from the dataset is stored in the repository),
binned 2x2 in space and 4x in time, and run through photon-mosaic (several neuropil corrections) and through CaImAn
(seeded with the same ROIs, run by ``run_caiman.py``). The score is the correlation between each method's inferred
spikes for the patched cell and its true spikes, both Gaussian-smoothed.

Which cells: by default the ones in ``CELLS``, chosen by one rule for every indicator: each animal's smallest file with
a movie of at least ``MIN_SPIKES`` spikes, and the 4 animals whose chosen files are smallest
(``--list --per-indicator 4`` prints the result). A file is one cell (its extra movies are further recordings of it):
its first movie with enough spikes is run.
``--per-indicator N`` applies the rule with N animals, ``--all`` runs every file in the dandiset.

    python benchmark_dandi_000168.py --data-dir ~/data/dandi-000168            # the default cells
    python benchmark_dandi_000168.py --data-dir ~/data/dandi-000168 --summary  # print the table
    python benchmark_dandi_000168.py --data-dir ~/data/dandi-000168 --all --delete-nwb   # everything: hours to days
    python benchmark_dandi_000168.py --data-dir /scratch --list --all                    # the paths, one per line
    python benchmark_dandi_000168.py --data-dir /scratch --cells <path> --results-dir /results   # one file, e.g. a task

Needs the photon-mosaic environment (pynwb, h5py, requests, oasis, and suite2p for the surround neuropil: ``pip
install "photon-mosaic[suite2p-registration]"``) and ``CAIMAN_PYTHON`` pointing at an environment with CaImAn. A cell
takes a few minutes plus its download (the files are 1 to 23 GB; ``--delete-nwb`` removes each one once it is
processed). Finished cells are skipped, so a run can be interrupted and resumed. A file that fails is logged to
``<results-dir>/<file>.failed`` and the run continues (failed files are retried next time); the exit status is 1 if
any failed.
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests  # type: ignore[import-untyped]
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.optimize import linear_sum_assignment

DANDISET, VERSION = "000168", "draft"
API = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}"
HERE = Path(__file__).parent

INDICATORS = ("jGCaMP8f", "jGCaMP8m", "jGCaMP8s", "jGCaMP7f")
PER_INDICATOR, MIN_SPIKES = 4, 40
# The result of that rule (``--list --per-indicator 4``, 4 animals per indicator, as of the dandiset's current state).
CELLS = [
    "jGCaMP8f/jGCaMP8f_ANM471994_cell06.nwb",
    "jGCaMP8f/jGCaMP8f_ANM478407_cell03.nwb",
    "jGCaMP8f/jGCaMP8f_ANM478349_cell06.nwb",
    "jGCaMP8f/jGCaMP8f_ANM478342_cell03.nwb",
    "jGCaMP8m/jGCaMP8m_ANM479115_cell03.nwb",
    "jGCaMP8m/jGCaMP8m_ANM479569_cell05.nwb",
    "jGCaMP8m/jGCaMP8m_ANM472179_cell05.nwb",
    "jGCaMP8m/jGCaMP8m_ANM479570_cell02.nwb",
    "jGCaMP8s/jGCaMP8s_ANM472182_cell09.nwb",
    "jGCaMP8s/jGCaMP8s_ANM479116_cell01.nwb",
    "jGCaMP8s/jGCaMP8s_ANM479120_cell03.nwb",
    "jGCaMP8s/jGCaMP8s_ANM479572_cell06.nwb",
    "jGCaMP7f/jGCaMP7f_ANM471991_cell02.nwb",
    "jGCaMP7f/jGCaMP7f_ANM478404_cell02.nwb",
    "jGCaMP7f/jGCaMP7f_ANM478406_cell03.nwb",
    "jGCaMP7f/jGCaMP7f_ANM478347_cell04.nwb",
]
WEIGHTS = (0.5, 0.7, 1.0)  # neuropil_weight values compared
SIGMAS = (1, 2, 4)  # Gaussian width (binned frames) used to smooth both spike trains before correlating
SPATIAL_BIN, TEMPORAL_BIN = 2, 4
MAIN_SIGMA = "2"  # the width reported in the summary
QUALITY_SPLIT = 8.0  # exploratory: cells with spike-triggered peak / noise at or above this count as high quality
NOT_A_METHOD = ("cell", "fs", "n_rois", "n_spikes", "footprint corr", "decay time", "signal quality")


# ---------------------------------------------------------------------------------------------------- choosing cells
def list_assets():
    """All files of the dandiset (dicts with ``path``, ``asset_id`` and ``size``)."""
    url, params, assets = f"{API}/assets/", {"page_size": 200}, []
    while url:
        page = requests.get(url, params=params).json()
        assets += page["results"]
        url, params = page.get("next"), None
    return assets


class _RangeFile(io.RawIOBase):
    """Read-only file object over HTTP range requests (so a file's metadata can be read without downloading it)."""

    def __init__(self, url, size, block=1 << 20):
        self.url, self.size, self.block, self.pos, self.cache = url, size, block, 0, {}
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            start = i * self.block
            end = min(start + self.block, self.size) - 1
            r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"})
            r.raise_for_status()
            self.cache[i] = r.content
        return self.cache[i]

    def readinto(self, buffer):
        n = min(len(buffer), self.size - self.pos)
        data = bytearray()
        while len(data) < n:
            i, offset = divmod(self.pos + len(data), self.block)
            data += self._block(i)[offset : offset + n - len(data)]
        buffer[:n] = data
        self.pos += n
        return n


def _count_spikes(asset, retries=3):
    """Spikes inside each movie's time window (in movie order), read from the file's metadata only."""
    import h5py

    for attempt in range(retries):
        try:
            url = requests.head(f"{API}/assets/{asset['asset_id']}/download/", allow_redirects=True).url
            with h5py.File(_RangeFile(url, asset["size"]), "r") as f:
                ap = f["analysis/responses/ap_time"][:]
                names = [k for k in f["acquisition"] if k.startswith("Registered movie")]
                names.sort(key=lambda k: int(k.split()[-1]))
                windows = [
                    (float(f["acquisition"][k]["timestamps"][0]), float(f["acquisition"][k]["timestamps"][-1]))
                    for k in names
                ]
            return {"bytes": asset["size"], "spikes_per_movie": [int(((ap > a) & (ap < b)).sum()) for a, b in windows]}
        except Exception:
            if attempt == retries - 1:
                return None


def spike_counts(data_dir, indicators=INDICATORS):
    """``{path: {bytes, spikes_per_movie}}`` for the files of ``indicators`` (cached in ``spike_counts.json``)."""
    cache_file = Path(data_dir) / "spike_counts.json"
    counts = json.load(open(cache_file)) if cache_file.exists() else {}
    todo = [a for a in list_assets() if a["path"].split("/")[0] in indicators and a["path"] not in counts]
    if todo:
        print(
            f"reading spike counts of {len(todo)} files (metadata only, no movies are downloaded) ...",
            file=sys.stderr,  # stdout is the path list of --list
            flush=True,
        )
        with ThreadPoolExecutor(8) as pool:
            for asset, count in zip(todo, pool.map(_count_spikes, todo)):
                if count is not None:
                    counts[asset["path"]] = count
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        json.dump(counts, open(cache_file, "w"), indent=1)
    return {p: c for p, c in counts.items() if p.split("/")[0] in indicators}


def select_cells(counts, per_indicator=PER_INDICATOR, min_spikes=MIN_SPIKES, indicators=INDICATORS):
    """The rule: per indicator, each animal's smallest file with a movie of at least ``min_spikes`` spikes, then the
    ``per_indicator`` animals whose chosen files are smallest (ties by name). A file is one cell: the extra movies of a
    file are further recordings of the same patched cell."""
    selected = []
    for indicator in indicators:
        smallest = {}  # animal -> (bytes, path)
        for path, c in counts.items():
            if path.split("/")[0] != indicator or max(c["spikes_per_movie"]) < min_spikes:
                continue
            animal = Path(path).name.split("_")[1]
            if animal not in smallest or (c["bytes"], path) < smallest[animal]:
                smallest[animal] = (c["bytes"], path)
        selected += [path for _, path in sorted(smallest.values())[:per_indicator]]
    return selected


# ------------------------------------------------------------------------------------------------------ the data
def _wait_for_space(directory, nbytes, max_wait_hours):
    """Wait until the filesystem of ``directory`` has room for ``nbytes`` plus a margin. Parallel runs can fill a small
    volume with downloads; waiting lets the tasks that are already processing free theirs."""
    need = int(nbytes * 1.05) + 2 * 10**9
    deadline = time.time() + max_wait_hours * 3600
    while shutil.disk_usage(directory).free < need:
        if time.time() > deadline:
            raise OSError(f"no room for {nbytes / 1e9:.1f} GB in {directory} after {max_wait_hours} h")
        time.sleep(60)


def download(path, data_dir, wait_hours=3.0, attempts=3):
    """Fetch ``path`` (e.g. ``jGCaMP8f/jGCaMP8f_ANM471994_cell06.nwb``) from DANDI into ``data_dir`` unless present.

    Waits for free space first (up to ``wait_hours``) and retries a broken connection ``attempts`` times."""
    target = Path(data_dir) / Path(path).name
    assets = requests.get(f"{API}/assets/", params={"path": path}).json()["results"]
    asset = next(a for a in assets if a["path"] == path)
    if target.exists() and target.stat().st_size == asset["size"]:
        return target
    _wait_for_space(target.parent, asset["size"], wait_hours)
    tmp = target.with_suffix(".part")
    for attempt in range(attempts):
        try:
            with requests.get(f"{API}/assets/{asset['asset_id']}/download/", stream=True) as r, open(tmp, "wb") as f:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=1 << 22):
                    f.write(chunk)
            break
        except requests.exceptions.RequestException:
            if attempt == attempts - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(30)
    os.replace(tmp, target)
    return target


def movie_numbers(nwb_path):
    """The numbers of the movies in a file (most have one; each movie has its own ROIs)."""
    import pynwb

    with pynwb.NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io_:
        return sorted(int(k.rsplit(" ", 1)[1]) for k in io_.read().acquisition if k.startswith("Registered movie"))


def load_binned(nwb_path, movie_number=0, spatial=SPATIAL_BIN, temporal=TEMPORAL_BIN, chunk=1220):
    """Movie ``movie_number`` binned ``spatial`` x ``spatial`` and ``temporal``-fold in time, with masks and spikes.

    Returns a dict: ``movie`` (T, H, W) float32, ``fs`` (Hz after binning), ``masks`` (K, H, W) (binned by mean; the NWB
    masks are weights that sum to 1), ``ephys_roi`` (indices of the ROIs recorded from; one for a patched cell) and
    ``spikes`` (T,) spike counts per binned frame.
    """
    import pynwb

    with pynwb.NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io_:
        nwb = io_.read()
        series = nwb.acquisition[f"Registered movie {movie_number}"]
        data, fs, t0 = series.data, series.imaging_plane.imaging_rate, float(series.timestamps[0])
        n, h, w = data.shape
        n_bins, hh, ww = n // temporal, h // spatial, w // spatial
        movie = np.empty((n_bins, hh, ww), np.float32)
        assert chunk % temporal == 0
        for start in range(0, n_bins * temporal, chunk):  # streamed: the full movie is never in memory
            stop = min(start + chunk, n_bins * temporal)
            x = np.asarray(data[start:stop], dtype=np.float32)[:, : hh * spatial, : ww * spatial]
            movie[start // temporal : stop // temporal] = x.reshape(-1, temporal, hh, spatial, ww, spatial).mean(
                axis=(1, 3, 5)
            )
        seg = nwb.processing[f"ophys of movie {movie_number}"]["ImageSegmentation"]["MovieSegmentation"]
        masks = np.asarray(seg["image_mask"].data[:], dtype=np.float32)[:, : hh * spatial, : ww * spatial]
        masks = masks.reshape(len(masks), hh, spatial, ww, spatial).mean(axis=(2, 4))
        ephys = np.flatnonzero(np.asarray(seg["recorded_with_ephys"].data[:], dtype=bool))
        ap_times = np.asarray(nwb.analysis["responses"]["ap_time"].data[:])
    fs_binned = fs / temporal
    idx = np.floor((ap_times - t0) * fs_binned).astype(int)
    spikes = np.bincount(idx[(idx >= 0) & (idx < n_bins)], minlength=n_bins)
    return dict(movie=movie, fs=fs_binned, masks=masks, ephys_roi=ephys, spikes=spikes)


# ----------------------------------------------------------------------------------------------------- the scores
def spike_corr(inferred, truth):
    """Correlation of two spike trains after Gaussian smoothing, per width in ``SIGMAS`` (string keys, as in JSON)."""

    def smooth(x, s):
        return gaussian_filter1d(np.asarray(x, dtype=float), s)

    return {str(s): float(np.corrcoef(smooth(inferred, s), smooth(truth, s))[0, 1]) for s in SIGMAS}


def signal_quality(movie, weights, spikes, fs):
    """How visible the patched cell's spikes are, independent of any method: its mean trace over the ROI (``weights``),
    as ΔF/F with a 30 s running 20th-percentile baseline, averaged around the true spikes. Returns the peak of that
    spike-triggered average and the peak over the frame-to-frame noise of the ΔF/F."""
    trace = (movie * (weights / weights.sum())).sum(axis=(1, 2))
    f0 = percentile_filter(trace, 20, size=int(30 * fs))
    dff = (trace - f0) / f0
    pre, post = int(0.5 * fs), int(2.5 * fs)
    idx = np.flatnonzero(spikes)
    idx = idx[(idx >= pre) & (idx < len(dff) - post)]
    sta = np.mean([dff[i - pre : i + post] for i in idx], axis=0)
    sta -= sta[:pre].mean()
    noise = np.std(np.diff(dff)) / np.sqrt(2)
    return {"sta_peak": float(sta.max()), "sta_over_noise": float(sta.max() / noise)}


def run_cell(nwb, movie_number, data_dir, caiman_python, min_spikes=MIN_SPIKES):
    """Score every method on the patched cell of one movie: ``{method: {sigma: correlation}}`` plus metadata.

    Returns ``None`` (with a message) for a movie without a patched ROI or with fewer than ``min_spikes`` spikes."""
    import photon_mosaic as pm
    from photon_mosaic.core.numpyimaging import NumpyImaging, NumpyRois
    from photon_mosaic.core.roianalyzer_core_extensions import _deconvolve_roi

    tag = nwb.stem if movie_number == 0 else f"{nwb.stem}_movie{movie_number}"
    d = load_binned(nwb, movie_number)
    movie, fs, truth = d["movie"], d["fs"], d["spikes"]
    if len(d["ephys_roi"]) != 1 or truth.sum() < min_spikes:
        print(f"{tag}: skipped ({len(d['ephys_roi'])} patched ROIs, {int(truth.sum())} spikes)", flush=True)
        return None
    masks = d["masks"] / d["masks"].max(axis=(1, 2), keepdims=True)  # photon-mosaic gets the weighted masks ...
    seed = (d["masks"] > 0).astype(np.float32)  # ... CaImAn is seeded with their support
    roi = int(d["ephys_roi"][0])
    print(
        f"{tag}: movie {movie.shape} @ {fs:.1f} Hz, {len(masks)} ROIs, patched ROI {roi}, {int(truth.sum())} spikes",
        flush=True,
    )
    imaging = NumpyImaging(movie, sampling_frequency=fs)
    out = {"cell": tag, "fs": fs, "n_rois": len(masks), "n_spikes": int(truth.sum())}
    out["signal quality"] = signal_quality(movie, masks[roi], truth, fs)
    kept = {}  # dF/F and deconvolution parameters of the CNMF w=0.7 run, reused below

    def chain(analyzer):
        analyzer.compute("df_over_f", n_jobs=4, method="percentile")
        return analyzer.compute("deconvolution", n_jobs=4)

    def photon_mosaic_methods(prefix, rois, patched):
        # One analyzer per neuropil source; only `fluorescence` (and what depends on it) is recomputed for each weight.
        an = pm.create_roi_analyzer(rois, imaging)
        an.compute("fluorescence", n_jobs=4, use_neuropil=False)
        out[f"{prefix} none"] = spike_corr(chain(an).get_data()[:, patched], truth)
        an = pm.create_roi_analyzer(rois, imaging)
        an.compute("neuropil", method="surround")
        for w in WEIGHTS:
            an.compute("fluorescence", n_jobs=4, use_neuropil=True, neuropil_weight=w)
            out[f"{prefix} surround w={w}"] = spike_corr(chain(an).get_data()[:, patched], truth)
        an = pm.create_roi_analyzer(rois, imaging)
        an.compute("background", gnb=1, n_jobs=4)
        for w in WEIGHTS:
            an.compute(
                "fluorescence",
                n_jobs=4,
                method="regression",
                use_neuropil=True,
                neuropil_source="background",
                neuropil_weight=w,
            )
            deconvolution = chain(an)
            out[f"{prefix} cnmf w={w}"] = spike_corr(deconvolution.get_data()[:, patched], truth)
            if prefix == "pm" and w == 0.7:
                kept["dff"] = an.get_extension("df_over_f").get_data()[:, patched]
                kept["params"] = dict(deconvolution.params)

    photon_mosaic_methods("pm", NumpyRois(masks, sampling_frequency=fs), roi)

    work = Path(data_dir) / "work"
    work.mkdir(exist_ok=True)
    movie_path, seed_path, result_path = work / f"{tag}_movie.npy", work / f"{tag}_seed.npy", work / f"{tag}_caiman.npz"
    if not result_path.exists():
        np.save(movie_path, movie)
        np.save(seed_path, seed)
        run = subprocess.run(
            [caiman_python, str(HERE / "run_caiman.py"), "--movie", str(movie_path), "--masks", str(seed_path),
             "--out", str(result_path), "--fs", str(fs), "--p", "1", "--gnb", "1"],
            capture_output=True, text=True, cwd=work,
            env={**os.environ, "CAIMAN_TEMP": str(work)},  # else CaImAn writes its memmap to ~/caiman_data/temp
        )  # fmt: skip
        # CaImAn's memmap of the movie (about 330 MB), named after the result file; not needed again
        for leftover in work.glob(f"{result_path.stem}_memmap_*"):
            leftover.unlink(missing_ok=True)
        if run.returncode:
            raise RuntimeError(run.stdout[-2000:] + run.stderr[-2000:])
    movie_path.unlink(missing_ok=True)  # cheap to rebuild
    seed_path.unlink(missing_ok=True)
    c = np.load(result_path)

    # CaImAn can drop seeded components, so match its components to the ROIs by footprint correlation.
    height, width = c["dims"]
    maps = np.stack([c["A"][:, k].reshape(height, width, order="F") for k in range(c["A"].shape[1])])
    corr = np.corrcoef(maps.reshape(len(maps), -1), masks.reshape(len(masks), -1))[: len(maps), len(maps) :]
    rows, cols = linear_sum_assignment(-corr)
    component_of = {int(k): int(j) for j, k in zip(rows, cols) if corr[j, k] > 0.5}
    out["footprint corr"] = {
        "seeded": len(masks), "kept": len(maps), "matched": len(component_of),
        "median": float(np.median([corr[j, k] for k, j in component_of.items()])) if component_of else float("nan"),
    }  # fmt: skip
    patched = component_of.get(roi)
    if patched is None:
        print(f"  CaImAn dropped the patched ROI of {tag}: no CaImAn rows", flush=True)
        return out
    out["footprint corr"]["patched"] = float(corr[patched, roi])
    out["caiman own (p=1)"] = spike_corr(c["S"][patched], truth)
    g = float(c["g"][patched])
    decay_time = -1 / (fs * np.log(g)) if 0 < g < 1 else float("nan")
    out["decay time"] = {"caiman_s": decay_time}
    if np.isfinite(decay_time):  # photon-mosaic's deconvolution with CaImAn's decay time instead of its own estimate
        params = dict(kept["params"], decay_time=decay_time)
        out["pm cnmf w=0.7, CaImAn's decay time"] = spike_corr(_deconvolve_roi((kept["dff"], fs, params))[1], truth)
    out["caiman dF/F + pm OASIS"] = spike_corr(
        _deconvolve_roi((c["F_dff"][patched].astype(float), fs, kept["params"]))[1], truth
    )
    photon_mosaic_methods(
        "pm[CaImAn A]", NumpyRois(maps, sampling_frequency=fs), patched
    )  # photon-mosaic on CaImAn's footprints
    return out


# ------------------------------------------------------------------------------------------------- running, summary
def resolve_cells(data_dir, cells=None, per_indicator=None, all_files=False, min_spikes=MIN_SPIKES):
    """The DANDI paths to run: ``cells`` if given; every file with ``all_files``; the selection rule with
    ``per_indicator`` animals per indicator; otherwise the default ``CELLS``."""
    if cells:
        return list(cells)
    if all_files:
        return sorted(a["path"] for a in list_assets())
    if per_indicator is not None:
        return select_cells(spike_counts(data_dir), per_indicator, min_spikes)
    return CELLS


def run_benchmark(
    data_dir,
    cells=None,
    per_indicator=None,
    all_files=False,
    min_spikes=MIN_SPIKES,
    delete_nwb=False,
    force=False,
    all_movies=False,
    results_dir=None,
    caiman_python=None,
):
    """Download and score the cells, writing ``<results_dir>/<cell>.json`` for each; finished cells are skipped.
    Returns the names of the files that failed.
    ``results_dir`` defaults to ``<data_dir>/results``. A file that raises is logged to ``<file>.failed`` and skipped.

    ``cells``: DANDI paths; otherwise ``per_indicator`` applies the selection rule with that many animals per indicator,
    ``all_files`` takes every file of the dandiset, and by default it is the ``CELLS`` list. A file is one cell: only
    its first movie with a patched ROI and at least ``min_spikes`` spikes is run, unless ``all_movies``, which runs
    every such movie and marks the file ``<file>.complete`` once all are done.
    ``caiman_python`` is the interpreter that has CaImAn; by default ``$CAIMAN_PYTHON``.
    """
    data_dir = Path(data_dir).expanduser().resolve()  # CaImAn runs with cwd=<data_dir>/work, so paths must be absolute
    results_dir = Path(results_dir).expanduser().resolve() if results_dir else data_dir / "results"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    caiman_python = caiman_python or os.environ.get(
        "CAIMAN_PYTHON", str(Path.home() / "miniconda3/envs/caiman/bin/python")
    )
    for path in resolve_cells(data_dir, cells, per_indicator, all_files, min_spikes):
        stem = Path(path).stem
        if all_movies:
            done = (results_dir / f"{stem}.complete").exists()
        else:
            done = (results_dir / f"{stem}.json").exists() or bool(list(results_dir.glob(f"{stem}_movie*.json")))
        if done and not force:
            print("skip (done):", stem, flush=True)
            continue
        nwb = None
        try:
            nwb = download(path, data_dir)
            for movie_number in movie_numbers(nwb):
                cell = stem if movie_number == 0 else f"{stem}_movie{movie_number}"
                if not force and (results_dir / f"{cell}.json").exists():
                    continue
                result = run_cell(nwb, movie_number, data_dir, caiman_python, min_spikes)
                if result is not None:
                    json.dump(result, open(results_dir / f"{result['cell']}.json", "w"), indent=1)
                    if not all_movies:
                        break
            (results_dir / f"{stem}.failed").unlink(missing_ok=True)
            if all_movies:
                (results_dir / f"{stem}.complete").touch()
        except Exception:
            (results_dir / f"{stem}.failed").write_text(traceback.format_exc())
            failed.append(stem)
            print(f"FAILED: {stem} (see {stem}.failed)", flush=True)
        finally:
            if delete_nwb and nwb is not None:
                nwb.unlink(missing_ok=True)
    if failed:
        print(f"{len(failed)} files failed: {', '.join(failed)}", flush=True)
    return failed


def summary_table(results_dir, cells=None, sigma=MAIN_SIGMA):
    """Mean spike correlation per method, by indicator and by signal quality, as text, over the results in
    ``results_dir`` (only those of the DANDI paths ``cells``, if given)."""
    results = [json.load(open(f)) for f in sorted(Path(results_dir).glob("*.json"))]
    if cells is not None:
        stems = {Path(c).stem for c in cells}
        results = [r for r in results if r["cell"].split("_movie")[0] in stems]
    methods = list(dict.fromkeys(k for r in results for k in r if k not in NOT_A_METHOD))

    def table(groups, title):
        lines = [title, f"{'':<38}" + "".join(f"{f'{name} (n={len(rs)})':>20}" for name, rs in groups.items())]
        spikes = [f"{np.median([r['n_spikes'] for r in rs]):.0f}" if rs else "-" for rs in groups.values()]
        lines.append(f"{'median spikes per cell':<38}" + "".join(f"{s:>20}" for s in spikes))
        for m in methods:
            values = [[r[m][sigma] for r in rs if m in r] for rs in groups.values()]
            lines.append(f"{m:<38}" + "".join(f"{np.mean(v):>20.3f}" if v else f"{'-':>20}" for v in values))
        return "\n".join(lines)

    by_indicator = {}
    for r in results:
        by_indicator.setdefault(r["cell"].split("_")[0], []).append(r)
    text = table(
        dict(sorted(by_indicator.items())),
        f"mean spike correlation over cells (Gaussian sigma = {sigma} binned frames)",
    )
    if results and all("signal quality" in r for r in results):
        high = [r for r in results if r["signal quality"]["sta_over_noise"] >= QUALITY_SPLIT]
        low = [r for r in results if r["signal quality"]["sta_over_noise"] < QUALITY_SPLIT]
        text += "\n\n" + table(
            {f"peak/noise >= {QUALITY_SPLIT:g}": high, f"peak/noise < {QUALITY_SPLIT:g}": low},
            "the same by signal quality (spike-triggered dF/F peak over noise; exploratory split)",
        )
        text += "\n\nsignal quality per cell: " + ", ".join(
            f"{r['cell'].split('_', 1)[1]} {r['signal quality']['sta_over_noise']:.1f}" for r in results
        )
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="where the NWB files and results go (outside the repository)")
    ap.add_argument("--cells", nargs="*", help="DANDI paths to run instead of the default cells")
    ap.add_argument("--per-indicator", type=int, help="apply the selection rule with this many animals per indicator")
    ap.add_argument("--all", action="store_true", help="run every file of the dandiset")
    ap.add_argument(
        "--min-spikes", type=int, default=MIN_SPIKES, help="skip movies with fewer spikes (default: %(default)s)"
    )
    ap.add_argument("--all-movies", action="store_true", help="run every movie of a file, not just the first valid one")
    ap.add_argument("--list", action="store_true", help="print the DANDI paths that would be run and exit")
    ap.add_argument("--force", action="store_true", help="recompute cells that already have a result")
    ap.add_argument("--delete-nwb", action="store_true", help="delete each NWB file once its cell is processed")
    ap.add_argument("--summary", action="store_true", help="only print the summary table")
    ap.add_argument("--results-dir", help="where the result files go (default: <data-dir>/results)")
    a = ap.parse_args()
    data_dir = Path(a.data_dir).expanduser()
    results_dir = Path(a.results_dir).expanduser() if a.results_dir else data_dir / "results"
    if a.list:
        print("\n".join(resolve_cells(data_dir, a.cells, a.per_indicator, a.all, a.min_spikes)))
        return
    failed = []
    if not a.summary:
        failed = run_benchmark(
            data_dir, a.cells, a.per_indicator, a.all, a.min_spikes, a.delete_nwb, a.force, a.all_movies, results_dir
        )
    cells = None if a.all else resolve_cells(data_dir, a.cells, a.per_indicator, False, a.min_spikes)
    print(summary_table(results_dir, cells))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":  # the guard matters: photon-mosaic's workers are spawned processes
    main()
