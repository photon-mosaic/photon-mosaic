"""Spike-inference benchmark on real data: photon-mosaic vs. CaImAn on paired imaging + loose-seal ephys.

Dataset: DANDI 000168 (Zhang et al., jGCaMP8; simultaneous cell-attached recordings and two-photon imaging of
GCaMP-expressing mouse V1 neurons). Each file is one patched cell: a 122 Hz movie, the ROIs of the field of view and the
patched cell's spike times. The movie is downloaded from DANDI (nothing from the dataset is stored in the repository),
binned 2x2 in space and 4x in time, and run through photon-mosaic (several neuropil corrections) and through CaImAn
(seeded with the same ROIs, run by ``run_caiman.py``). The score is the correlation between each method's inferred
spikes for the patched cell and its true spikes, both Gaussian-smoothed.

    python benchmark_dandi_000168.py --data-dir ~/data/dandi-000168            # download, run, write <data-dir>/results
    python benchmark_dandi_000168.py --data-dir ~/data/dandi-000168 --summary  # print the table

Needs the photon-mosaic environment (pynwb, h5py, requests, oasis) and ``CAIMAN_PYTHON`` pointing at an environment with
CaImAn. A cell takes a few minutes; its NWB file is about 2 GB (``--delete-nwb`` removes it once the cell is processed).
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import requests  # type: ignore[import-untyped]
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

DANDISET, VERSION = "000168", "draft"
API = f"https://api.dandiarchive.org/api/dandisets/{DANDISET}/versions/{VERSION}"
HERE = Path(__file__).parent

# Smallest files with the most spikes, a few per indicator, from different animals where possible.
CELLS = [
    "jGCaMP8f/jGCaMP8f_ANM471994_cell06.nwb",
    "jGCaMP8f/jGCaMP8f_ANM471994_cell08.nwb",
    "jGCaMP8f/jGCaMP8f_ANM471994_cell03.nwb",
    "jGCaMP8f/jGCaMP8f_ANM478407_cell05.nwb",
    "jGCaMP8m/jGCaMP8m_ANM479115_cell08.nwb",
    "jGCaMP8m/jGCaMP8m_ANM479115_cell02.nwb",
    "jGCaMP8m/jGCaMP8m_ANM479569_cell05.nwb",
    "jGCaMP8m/jGCaMP8m_ANM479115_cell03.nwb",
    "jGCaMP8s/jGCaMP8s_ANM472182_cell09.nwb",
    "jGCaMP8s/jGCaMP8s_ANM479572_cell06.nwb",
    "jGCaMP8s/jGCaMP8s_ANM479116_cell01.nwb",
    "jGCaMP8s/jGCaMP8s_ANM479571_cell02.nwb",
]
WEIGHTS = (0.5, 0.7, 1.0)  # neuropil_weight values compared
SIGMAS = (1, 2, 4)  # Gaussian width (binned frames) used to smooth both spike trains before correlating
SPATIAL_BIN, TEMPORAL_BIN = 2, 4
MAIN_SIGMA = "2"  # the width reported in the summary
NOT_A_METHOD = ("cell", "fs", "n_rois", "n_spikes", "footprint corr", "decay time")


def download(path, data_dir):
    """Fetch ``path`` (e.g. ``jGCaMP8f/jGCaMP8f_ANM471994_cell06.nwb``) from DANDI into ``data_dir`` unless present."""
    target = Path(data_dir) / Path(path).name
    assets = requests.get(f"{API}/assets/", params={"path": path}).json()["results"]
    asset = next(a for a in assets if a["path"] == path)
    if target.exists() and target.stat().st_size == asset["size"]:
        return target
    tmp = target.with_suffix(".part")
    with requests.get(f"{API}/assets/{asset['asset_id']}/download/", stream=True) as r, open(tmp, "wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=1 << 22):
            f.write(chunk)
    os.replace(tmp, target)
    return target


def load_binned(nwb_path, spatial=SPATIAL_BIN, temporal=TEMPORAL_BIN, chunk=1220):
    """The movie binned ``spatial`` x ``spatial`` and ``temporal``-fold in time (mean), with ROI masks and spikes.

    Returns a dict: ``movie`` (T, H, W) float32, ``fs`` (Hz after binning), ``masks`` (K, H, W) (binned by mean; the NWB
    masks are weights that sum to 1), ``ephys_roi`` (index of the patched cell's ROI) and ``spikes`` (T,) spike counts
    per binned frame.
    """
    import pynwb

    with pynwb.NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        (name,) = [k for k in nwb.acquisition if k.startswith("Registered movie")]
        series = nwb.acquisition[name]
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
        (seg,) = [v for k, v in nwb.processing.items() if k.startswith("ophys")]
        seg = seg["ImageSegmentation"]["MovieSegmentation"]
        masks = np.asarray(seg["image_mask"].data[:], dtype=np.float32)[:, : hh * spatial, : ww * spatial]
        masks = masks.reshape(len(masks), hh, spatial, ww, spatial).mean(axis=(2, 4))
        ephys = np.flatnonzero(np.asarray(seg["recorded_with_ephys"].data[:], dtype=bool))
        ap_times = np.asarray(nwb.analysis["responses"]["ap_time"].data[:])
    fs_binned = fs / temporal
    idx = np.floor((ap_times - t0) * fs_binned).astype(int)
    spikes = np.bincount(idx[(idx >= 0) & (idx < n_bins)], minlength=n_bins)
    return dict(movie=movie, fs=fs_binned, masks=masks, ephys_roi=ephys, spikes=spikes)


def spike_corr(inferred, truth):
    """Correlation of two spike trains after Gaussian smoothing, per width in ``SIGMAS`` (string keys, as in JSON)."""

    def smooth(x, s):
        return gaussian_filter1d(np.asarray(x, dtype=float), s)

    return {str(s): float(np.corrcoef(smooth(inferred, s), smooth(truth, s))[0, 1]) for s in SIGMAS}


def run_cell(path, data_dir, caiman_python):
    """Score every method on the patched cell of one file: ``{method: {sigma: correlation}}`` plus metadata."""
    import photon_mosaic as pm
    from photon_mosaic.core.numpyimaging import NumpyImaging, NumpyRois
    from photon_mosaic.core.roianalyzer_core_extensions import _deconvolve_roi

    nwb = download(path, data_dir)
    tag = nwb.stem
    d = load_binned(nwb)
    movie, fs, truth = d["movie"], d["fs"], d["spikes"]
    masks = d["masks"] / d["masks"].max(axis=(1, 2), keepdims=True)  # photon-mosaic gets the weighted masks ...
    seed = (d["masks"] > 0).astype(np.float32)  # ... CaImAn is seeded with their support
    roi = int(d["ephys_roi"][0])
    print(
        f"{tag}: movie {movie.shape} @ {fs:.1f} Hz, {len(masks)} ROIs, patched ROI {roi}, {int(truth.sum())} spikes",
        flush=True,
    )
    imaging = NumpyImaging(movie, sampling_frequency=fs)
    out = {"cell": tag, "fs": fs, "n_rois": len(masks), "n_spikes": int(truth.sum())}
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
        )  # fmt: skip
        if run.returncode:
            raise RuntimeError(run.stdout[-2000:] + run.stderr[-2000:])
    movie_path.unlink(missing_ok=True)  # cheap to rebuild
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


def summary_table(results_dir, sigma=MAIN_SIGMA):
    """Mean spike correlation per method and indicator over the cells in ``results_dir``, as text."""
    results = [json.load(open(f)) for f in sorted(Path(results_dir).glob("*.json"))]
    by_indicator = {}
    for r in results:
        by_indicator.setdefault(r["cell"].split("_")[0], []).append(r)
    indicators = sorted(by_indicator)
    methods = [k for r in results for k in r if k not in NOT_A_METHOD]
    methods = list(dict.fromkeys(methods))
    lines = [f"mean spike correlation over cells (Gaussian sigma = {sigma} binned frames)"]
    lines.append(f"{'':<38}" + "".join(f"{f'{i} (n={len(by_indicator[i])})':>16}" for i in indicators))
    lines.append(
        f"{'spikes per cell':<38}"
        + "".join(f"{'/'.join(str(r['n_spikes']) for r in by_indicator[i]):>16}" for i in indicators)
    )
    for m in methods:
        cells = [[r[m][sigma] for r in by_indicator[i] if m in r] for i in indicators]
        lines.append(f"{m:<38}" + "".join(f"{np.mean(v):>16.3f}" if v else f"{'-':>16}" for v in cells))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="where the NWB files and results go (outside the repository)")
    ap.add_argument("--cells", nargs="*", default=CELLS, help="DANDI paths; default: the 12 cells in CELLS")
    ap.add_argument("--force", action="store_true", help="recompute cells that already have a result")
    ap.add_argument("--delete-nwb", action="store_true", help="delete each NWB file once its cell is processed")
    ap.add_argument("--summary", action="store_true", help="only print the summary table")
    a = ap.parse_args()
    data_dir = Path(a.data_dir).expanduser()
    results_dir = data_dir / "results"
    if not a.summary:
        data_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        caiman_python = os.environ.get("CAIMAN_PYTHON", str(Path.home() / "miniconda3/envs/caiman/bin/python"))
        for path in a.cells:
            result_file = results_dir / (Path(path).stem + ".json")
            if result_file.exists() and not a.force:
                print("skip (done):", result_file.name, flush=True)
                continue
            result = run_cell(path, data_dir, caiman_python)
            json.dump(result, open(result_file, "w"), indent=1)
            if a.delete_nwb:
                (data_dir / Path(path).name).unlink(missing_ok=True)
    print(summary_table(results_dir))


if __name__ == "__main__":  # the guard matters: photon-mosaic's workers are spawned processes
    main()
