"""Run CaImAn's CNMF on a movie, for compare_with_caiman.ipynb.

``--mode seeded`` (default) starts from given ROI masks (no detection). ``--mode corr_pnr`` detects the ROIs
with CaImAn's ``corr_pnr`` initialisation (the one meant for a strong shared background) and keeps the
low-rank ``f bᵀ`` background, so the two modes differ only in where the footprints come from.

Run inside an environment that has CaImAn installed (the notebook calls this via subprocess).
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movie", required=True)  # (T, H, W) .npy
    ap.add_argument("--mode", choices=("seeded", "corr_pnr"), default="seeded")
    ap.add_argument("--masks")  # (K, H, W) .npy, for --mode seeded
    ap.add_argument("--out", required=True)  # .npz
    ap.add_argument("--fs", type=float, default=30.0)
    ap.add_argument("--p", type=int, default=0)
    ap.add_argument("--gnb", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--n-processes", type=int, default=8)
    ap.add_argument("--gSig", type=int, default=5)  # expected half-size of a neuron, px (corr_pnr)
    ap.add_argument("--min-corr", type=float, default=0.8)  # corr_pnr seed thresholds
    ap.add_argument("--min-pnr", type=float, default=4.5)
    ap.add_argument("--max-components", type=int, default=100)  # upper bound for corr_pnr
    a = ap.parse_args()

    logging.basicConfig(format="%(relativeCreated)d %(levelname)s %(message)s", level=logging.INFO, stream=sys.stdout)
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf
    from caiman.source_extraction.cnmf import params as cparams

    npy = a.movie
    tag = os.path.splitext(os.path.basename(a.out))[0]
    if a.max_frames:  # quick test runs: CaImAn memory-maps whole files, so write the subset out
        npy = os.path.join(os.path.dirname(os.path.abspath(a.out)), f"{tag}_subset.npy")
        np.save(npy, np.load(a.movie, mmap_mode="r")[: a.max_frames])

    if a.mode == "seeded":
        masks = np.load(a.masks)
        # CaImAn only derives its own initial C/b/f from a *boolean* seed (see computing_indicator in
        # spatial.py); a float Ain raises "You need to provide estimate of C and f".
        Ain = np.stack([m.reshape(-1, order="F") > 0 for m in masks], axis=1)  # (H*W, K), F-order pixels
        # (dense on purpose: a sparse seed makes computing_indicator's pixel mask a 2-D np.matrix and fail)
        init_params = {"method_init": "greedy_roi", "K": None, "gSig": (5, 5)}
    else:
        Ain = None
        init_params = {
            "method_init": "corr_pnr",
            "K": a.max_components,
            "gSig": (a.gSig, a.gSig),
            "gSiz": (4 * a.gSig + 1, 4 * a.gSig + 1),
            "center_psf": True,
            "min_corr": a.min_corr,
            "min_pnr": a.min_pnr,
        }

    _, dview, n_processes = cm.cluster.setup_cluster(
        backend="multiprocessing", n_processes=a.n_processes, single_thread=False
    )
    try:
        fname = cm.save_memmap([npy], base_name=f"{tag}_memmap_", order="C", dview=dview)
        Yr, dims, T = cm.load_memmap(fname)
        images = np.reshape(Yr.T, [T] + list(dims), order="F")
        opts = cparams.CNMFParams(
            params_dict={
                "fnames": [npy],
                "fr": a.fs,
                "decay_time": 0.5,
                "p": a.p,
                "nb": a.gnb,
                "rf": None,
                "only_init": False,
                "ssub": 1,
                "tsub": 1,
                "do_merge": False,
                "merge_thr": 1.0,
                **init_params,
            }
        )
        t0 = time.time()
        cnm = cnmf.CNMF(n_processes, params=opts, dview=dview, Ain=Ain)
        cnm.fit(images)
        print(f"fit took {time.time() - t0:.0f}s")
        e = cnm.estimates
        # Same baseline window as photon-mosaic's df_over_f default (win_baseline=60 s), with the
        # quantile chosen automatically from each trace (KDE), as photon-mosaic's percentile method does.
        e.detrend_df_f(frames_window=int(60 * a.fs), flag_auto=True)
        A = e.A.toarray() if sp.issparse(e.A) else np.asarray(e.A)
        print(
            "A",
            A.shape,
            "C",
            e.C.shape,
            "b",
            None if e.b is None else e.b.shape,
            "f",
            None if e.f is None else e.f.shape,
        )
        S = np.zeros_like(e.C) if e.S is None else np.asarray(e.S)  # spikes (p > 0), zeros for p=0
        # AR(1) coefficient per component (p=1), for the decay time
        g = (
            np.array([np.atleast_1d(x)[0] if np.size(x) else np.nan for x in e.g])
            if getattr(e, "g", None) is not None
            else np.array([])
        )
        np.savez(a.out, A=A, C=e.C, YrA=e.YrA, S=S, g=g, b=e.b, f=e.f, F_dff=e.F_dff, dims=np.array(dims))
    finally:
        cm.stop_server(dview=dview)


if __name__ == "__main__":
    main()
