#!/usr/bin/env python
"""Summarise parametrized benchmark results into compact heatmap tables.

Reads the JSON produced by ``pytest --benchmark-json=<file>`` and prints
pivot tables that make it easy to see where each backend wins.

Usage
-----
::

    # After running the benchmarks:
    pytest -m "grid and small" --benchmark-json=benchmark_results.json

    # Summarise:
    python src/photon_mosaic/core/tests/summarise_benchmarks.py benchmark_results.json

    # Or with plots (requires matplotlib):
    python src/photon_mosaic/core/tests/summarise_benchmarks.py benchmark_results.json --plot

Output
------
For each (dataset_size, plane_strategy) combination a table like this is
printed::

    small | half-planes
    ─────────────────────────────────────────────────────
    Median time (ms)  │  P=1   P=5   P=10   P=25   P=50
    ──────────────────┼──────────────────────────────────
    F=10              │
      npy_full_load   │  1.2   3.4    6.7   15.1   30.2
      npy_memmap      │  0.1   0.3    0.5    1.1    2.3
      binary_memmap   │  0.1   0.2    0.4    0.9    1.8
      zarr_dask       │  2.1   2.3    2.5    3.0    3.5
    F=100             │
      ...

This lets you immediately spot the cross-over points.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Test name pattern:
#   test_{size}_{backend}[P{planes}-{strategy}-{position}-F{frames}]
# e.g. test_small_zarr_native[P10-1plane-mid-F100]
_NAME_RE = re.compile(
    r"test_(?P<size>small|large)_(?P<backend>[a-z_]+)\[P(?P<planes>\d+)-(?P<strategy>\w+)-(?P<position>start|mid|end)-F(?P<frames>\d+)\]"
)


def _parse_benchmark(bench: dict) -> dict | None:
    """Extract structured fields from a single benchmark entry."""
    name = bench["name"]
    m = _NAME_RE.search(name)
    if m is None:
        return None
    return {
        "size": m.group("size"),
        "backend": m.group("backend"),
        "num_planes": int(m.group("planes")),
        "strategy": m.group("strategy"),
        "position": m.group("position"),
        "num_frames": int(m.group("frames")),
        "median_s": bench["stats"]["median"],
        "mean_s": bench["stats"]["mean"],
        "min_s": bench["stats"]["min"],
        "max_s": bench["stats"]["max"],
        "stddev_s": bench["stats"]["stddev"],
        "rounds": bench["stats"]["rounds"],
    }


def load_results(path: str | Path) -> list[dict]:
    """Load and parse a pytest-benchmark JSON file."""
    with open(path) as f:
        data = json.load(f)

    parsed = []
    for bench in data.get("benchmarks", []):
        entry = _parse_benchmark(bench)
        if entry is not None:
            parsed.append(entry)
    return parsed


# ---------------------------------------------------------------------------
# Pivot-table builder
# ---------------------------------------------------------------------------


def build_pivot_tables(results: list[dict]) -> dict:
    """Group results into nested dicts for tabular display.

    Returns
    -------
    tables : dict
        ``tables[(size, strategy, position)][num_frames][backend][num_planes] = min_ms``
    """
    tables: dict[tuple, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in results:
        key = (r["size"], r["strategy"], r["position"])
        tables[key][r["num_frames"]][r["backend"]][r["num_planes"]] = r["min_s"] * 1000  # → ms
    return dict(tables)


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

# ANSI colours for the "winner" highlight
_GREEN = "\033[92m"
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _format_cell(value: float, is_best: bool) -> str:
    """Format a single cell value, highlighting the best."""
    if is_best:
        return f"{_GREEN}{value:>8.2f}{_RESET}"
    return f"{value:>8.2f}"


def print_tables(tables: dict, use_colour: bool = True):
    """Print the pivot tables to stdout."""
    if not use_colour:
        global _GREEN, _RESET, _BOLD, _DIM
        _GREEN = _RESET = _BOLD = _DIM = ""

    for (size, strategy, position), by_frames in sorted(tables.items()):
        print()
        print(f"{_BOLD}{'═' * 70}{_RESET}")
        print(f"{_BOLD}  {size.upper()} data  |  planes: {strategy}  |  seek: {position}{_RESET}")
        print(f"{_BOLD}{'═' * 70}{_RESET}")

        # Collect all plane counts across the table
        all_planes = sorted({p for by_be in by_frames.values() for vals in by_be.values() for p in vals})

        # Header row
        header = f"  {'Backend':<20s}"
        for p in all_planes:
            header += f"{'P=' + str(p):>8s}"
        print(f"\n  {_DIM}Min time (ms){_RESET}")
        print(f"  {'─' * (20 + 8 * len(all_planes))}")
        print(header)
        print(f"  {'─' * (20 + 8 * len(all_planes))}")

        for nf in sorted(by_frames):
            print(f"\n  {_BOLD}F={nf}{_RESET}")
            backends = by_frames[nf]

            # Find the best (lowest) backend for each plane count
            best_per_plane = {}
            for p in all_planes:
                vals = {be: backends[be].get(p) for be in backends if backends[be].get(p) is not None}
                if vals:
                    best_per_plane[p] = min(vals, key=vals.get)

            for backend in sorted(backends):
                row = f"    {backend:<18s}"
                for p in all_planes:
                    val = backends[backend].get(p)
                    if val is not None:
                        is_best = best_per_plane.get(p) == backend
                        row += _format_cell(val, is_best)
                    else:
                        row += f"{'—':>8s}"
                print(row)

        print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(results: list[dict], path: str | Path):
    """Export flat results to CSV for further analysis in pandas/Excel."""
    import csv

    fieldnames = [
        "size",
        "backend",
        "num_planes",
        "strategy",
        "position",
        "num_frames",
        "median_ms",
        "mean_ms",
        "min_ms",
        "max_ms",
        "stddev_ms",
        "rounds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "size": r["size"],
                    "backend": r["backend"],
                    "num_planes": r["num_planes"],
                    "strategy": r["strategy"],
                    "position": r["position"],
                    "num_frames": r["num_frames"],
                    "median_ms": r["median_s"] * 1000,
                    "mean_ms": r["mean_s"] * 1000,
                    "min_ms": r["min_s"] * 1000,
                    "max_ms": r["max_s"] * 1000,
                    "stddev_ms": r["stddev_s"] * 1000,
                    "rounds": r["rounds"],
                }
            )
    print(f"\nCSV exported → {path}")


# ---------------------------------------------------------------------------
# Matplotlib heatmap (optional)
# ---------------------------------------------------------------------------


def plot_heatmaps(results: list[dict], output_dir: str | Path = "."):
    """Generate a single figure with two subplot heatmaps (small + large).

    Layout (per subplot)
    --------------------
    - **Rows** = backends (sorted alphabetically)
    - **Columns** = every unique (seek, P, strategy, F) combination,
      ordered as: seek → num_planes/strategy → num_frames.
      Columns are visually grouped by seek position with a divider.

    Colour scale is logarithmic so both fast and slow backends are
    distinguishable.  Each cell shows the min time in ms.
    Produces ``bench_combined.png``.
    """
    try:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plots. Install with: pip install matplotlib")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a flat lookup: (size, backend, position, strategy, num_planes, num_frames) → min_ms
    lookup: dict[tuple, float] = {}
    for r in results:
        key = (r["size"], r["backend"], r["position"], r["strategy"], r["num_planes"], r["num_frames"])
        lookup[key] = r["min_s"] * 1000

    # Discover sizes
    sizes = sorted({r["size"] for r in results})

    # Pre-compute per-size data so we can determine subplot dimensions
    panels: list[dict] = []
    for size in sizes:
        size_results = [r for r in results if r["size"] == size]
        backends = sorted({r["backend"] for r in size_results})
        col_keys = sorted(
            {(r["position"], r["num_planes"], r["strategy"], r["num_frames"]) for r in size_results},
            key=lambda x: (x[0], x[1], x[2], x[3]),
        )
        if not col_keys or not backends:
            continue

        data = []
        for be in backends:
            row = []
            for pos, np_, strat, nf in col_keys:
                val = lookup.get((size, be, pos, strat, np_, nf), float("nan"))
                row.append(val)
            data.append(row)

        panels.append(dict(size=size, backends=backends, col_keys=col_keys, data=data))

    if not panels:
        print("  No data to plot.")
        return

    n_panels = len(panels)
    # Use height ratios proportional to the number of backends in each panel
    height_ratios = [len(p["backends"]) for p in panels]

    max_cols = max(len(p["col_keys"]) for p in panels)
    fig_w = max(10, max_cols * 1.1 + 5)  # extra room for colorbar
    fig_h = sum(max(3, h * 0.7 + 1.8) for h in height_ratios) + 1.2

    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )
    # Leave generous right margin for the colour-bar
    fig.subplots_adjust(right=0.88, left=0.10, hspace=0.45, top=0.90, bottom=0.08)

    # Global min/max across all panels for a shared log colour scale
    all_vals = [v for p in panels for row in p["data"] for v in row if v == v]
    vmin = max(min(all_vals), 0.01)  # avoid log(0)
    vmax = max(all_vals)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    for idx, panel in enumerate(panels):
        ax = axes[idx, 0]
        size = panel["size"]
        backends = panel["backends"]
        col_keys = panel["col_keys"]
        data = panel["data"]
        n_rows = len(backends)
        n_cols = len(col_keys)

        data_arr = np.array(data, dtype=float)
        im = ax.imshow(data_arr, aspect="auto", cmap="RdYlGn_r", norm=norm)

        # Column labels
        col_labels = []
        for pos, np_, strat, nf in col_keys:
            if np_ == 1:
                col_labels.append(f"{pos}\nP={np_}\nF={nf}")
            else:
                col_labels.append(f"{pos}\nP={np_} ({strat})\nF={nf}")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, fontsize=7, ha="center")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(backends, fontsize=9)

        # Find column-wise minima for highlighting
        col_mins = []
        for j in range(n_cols):
            col_vals = [data[i][j] for i in range(n_rows) if data[i][j] == data[i][j]]
            col_mins.append(min(col_vals) if col_vals else float("nan"))

        # Annotate cells with values
        for i in range(n_rows):
            for j in range(n_cols):
                val = data[i][j]
                if val == val:  # not NaN
                    is_best = val == col_mins[j]
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=9 if is_best else 7,
                        fontweight="bold" if is_best else "normal",
                        color="white" if val > (vmax * 0.5) else "black",
                    )

        # Draw vertical dividers between seek-position groups
        prev_pos = col_keys[0][0]
        for j, (pos, *_) in enumerate(col_keys):
            if pos != prev_pos:
                ax.axvline(x=j - 0.5, color="white", linewidth=2)
                prev_pos = pos

        ax.set_title(f"{size.upper()} data", fontsize=11, fontweight="bold", pad=6)

    # Shared colour-bar on the right, outside the subplots
    cbar_ax = fig.add_axes([0.91, 0.08, 0.02, 0.82])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label="min time (ms)")

    fig.suptitle(
        "Backend benchmark — min time (ms, log scale)\n" "P = number of planes on disk  |  F = number of frames read",
        fontsize=13,
        fontweight="bold",
    )

    # Backend descriptions as a footnote at the bottom of the figure
    backend_notes = (
        "npy_full_load: numpy .npy loaded entirely into RAM  •  "
        "npy_memmap: numpy memory-mapped (lazy)  •  "
        "npy_dask: dask array backed by numpy memmap\n"
        "binary_memmap: raw binary with fresh mmap per read  •  "
        "zarr_native: zarr with on-disk chunks (256 frames)  •  "
        "zarr_rechunked: zarr read with smaller dask chunks (64 frames), "
        "misaligned with the on-disk layout"
    )
    fig.text(
        0.50,
        -0.01,
        backend_notes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontstyle="italic",
        color="0.35",
        wrap=True,
    )

    fname = output_dir / "bench_combined.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {fname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Summarise pytest-benchmark JSON into pivot tables & heatmaps.",
    )
    parser.add_argument(
        "json_file",
        help="Path to the pytest-benchmark JSON file (from --benchmark-json=<file>)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Also export flat results to this CSV path",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate heatmap PNGs (requires matplotlib)",
    )
    parser.add_argument(
        "--plot-dir",
        default="benchmark_plots",
        help="Directory for heatmap PNGs (default: benchmark_plots/)",
    )
    parser.add_argument(
        "--no-colour",
        "--no-color",
        action="store_true",
        help="Disable ANSI colour in terminal output",
    )
    args = parser.parse_args()

    results = load_results(args.json_file)
    if not results:
        print("No benchmark results found — did the tests run?", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoaded {len(results)} benchmark results from {args.json_file}")

    tables = build_pivot_tables(results)
    print_tables(tables, use_colour=not args.no_colour)

    if args.csv:
        export_csv(results, args.csv)

    if args.plot:
        plot_heatmaps(results, args.plot_dir)


if __name__ == "__main__":
    main()
