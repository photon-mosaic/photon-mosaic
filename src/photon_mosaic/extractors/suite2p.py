"""Load suite2p registered binary movies as photon-mosaic imaging objects.

Suite2p stores the registered functional movie as one ``data.bin`` per plane
under ``<suite2p_root>/plane{i}/`` together with an ``ops.npy`` describing
the geometry (``Ly``, ``Lx``, ``nframes``, ``fs``, ``nplanes``, ``nchannels``)
and the per-source-file frame counts (``frames_per_file``,
``frames_per_folder``). When two functional channels are present the second
channel lives next to ``data.bin`` as ``data_chan2.bin``.

This module exposes :class:`Suite2pImaging`, which folds all planes of a
single suite2p run into one multi-plane :class:`BinaryImaging` (one epoch),
or concatenates multiple suite2p runs of the same FOV as additional epochs,
and :func:`read_suite2p`, which returns one object per functional channel.
"""

from pathlib import Path
from typing import Any

import numpy as np

from photon_mosaic.core.binaryimaging import BinaryImaging

SUITE2P_BIN_DTYPE = np.int16  # suite2p writes registered movies as int16
SUITE2P_PLANE_DIR_PREFIX = "plane"


def _list_plane_dirs(root: Path) -> list[Path]:
    """Return ``plane{i}/`` subdirectories of a suite2p root, sorted by plane index."""
    candidates = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(SUITE2P_PLANE_DIR_PREFIX)]

    def _plane_index(p: Path) -> int:
        try:
            return int(p.name[len(SUITE2P_PLANE_DIR_PREFIX) :])
        except ValueError:
            return -1

    valid = sorted([p for p in candidates if _plane_index(p) >= 0], key=_plane_index)
    if not valid:
        raise FileNotFoundError(f"No plane*/ directories found under {root}")
    return valid


def _load_ops(plane_dir: Path) -> dict[str, Any]:
    ops_path = plane_dir / "ops.npy"
    if not ops_path.exists():
        raise FileNotFoundError(f"Missing ops.npy in {plane_dir}")
    return np.load(ops_path, allow_pickle=True).item()


def _data_bin_name(chan: int) -> str:
    if chan == 1:
        return "data.bin"
    if chan == 2:
        return "data_chan2.bin"
    raise ValueError(f"chan must be 1 or 2, got {chan}")


def _collect_run_metadata(root: Path, chan: int) -> dict[str, Any]:
    """Read every plane's ops.npy and return the consistent geometry for one suite2p run.

    Validates that ``Ly``, ``Lx``, ``fs``, ``nplanes``, ``nchannels`` and ``nframes``
    agree across all planes. ``nchannels`` must be >= ``chan`` (no data.bin if not).
    """
    plane_dirs = _list_plane_dirs(root)
    per_plane_ops = [_load_ops(p) for p in plane_dirs]

    declared_nplanes = int(per_plane_ops[0]["nplanes"])
    if declared_nplanes != len(plane_dirs):
        raise ValueError(
            f"ops['nplanes']={declared_nplanes} disagrees with on-disk plane count " f"{len(plane_dirs)} for {root}"
        )

    Ly = int(per_plane_ops[0]["Ly"])
    Lx = int(per_plane_ops[0]["Lx"])
    fs = float(per_plane_ops[0]["fs"])
    nframes = int(per_plane_ops[0]["nframes"])
    nchannels = int(per_plane_ops[0]["nchannels"])

    for i, (pd, ops) in enumerate(zip(plane_dirs, per_plane_ops)):
        for key, expected in (("Ly", Ly), ("Lx", Lx), ("nframes", nframes)):
            if int(ops[key]) != expected:
                raise ValueError(
                    f"Plane {i} ({pd.name}) ops['{key}']={int(ops[key])} disagrees with plane 0 ({expected})"
                )
        if float(ops["fs"]) != fs:
            raise ValueError(f"Plane {i} ({pd.name}) ops['fs']={float(ops['fs'])} disagrees with plane 0 ({fs})")
        if int(ops["nchannels"]) != nchannels:
            raise ValueError(
                f"Plane {i} ({pd.name}) ops['nchannels']={int(ops['nchannels'])} disagrees with plane 0 ({nchannels})"
            )

    if chan > nchannels:
        raise FileNotFoundError(
            f"Requested chan={chan} but ops['nchannels']={nchannels}; no per-plane binary exists for this channel"
        )

    bin_name = _data_bin_name(chan)
    plane_files: list[Path] = []
    for pd in plane_dirs:
        bin_path = pd / bin_name
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing {bin_name} in {pd}")
        plane_files.append(bin_path)

    # frames_per_file may not be present for all input formats; treat as optional
    fpf_raw = per_plane_ops[0].get("frames_per_file")
    if fpf_raw is None:
        frames_per_file = None
    else:
        frames_per_file = np.asarray(fpf_raw, dtype=int).tolist()

    return {
        "plane_files": plane_files,
        "Ly": Ly,
        "Lx": Lx,
        "nframes": nframes,
        "fs": fs,
        "nplanes": declared_nplanes,
        "nchannels": nchannels,
        "frames_per_file": frames_per_file,
    }


class Suite2pImaging(BinaryImaging):
    """A photon-mosaic imaging object backed by suite2p's per-plane registered binaries.

    One :class:`Suite2pImaging` represents a single brain volume covered by one or
    more suite2p runs of the same field of view. Planes within a run are stitched
    on read into the ``(T, H, W, n_planes)`` volume; multiple runs (passed as a
    list of folders) become successive epochs of the same imaging object.

    Parameters
    ----------
    folder_path : str | Path | list[str | Path]
        Path to a suite2p output root (the parent of ``plane0/``, ``plane1/``,
        ...). A list of paths is interpreted as multiple acquisitions of the
        same FOV; each path becomes one epoch.
    chan : int, default: 1
        Functional channel to load. 1 -> ``data.bin``, 2 -> ``data_chan2.bin``.

    Notes
    -----
    The per-source-file frame counts reported by suite2p (``ops['frames_per_file']``)
    are kept on the returned object as :attr:`frames_per_file_per_epoch`, so the
    user can subdivide an epoch back into per-file segments via
    :func:`photon_mosaic.core.split.split_epoch_at_frames`.
    """

    def __init__(self, folder_path, chan: int = 1):
        folders = folder_path if isinstance(folder_path, list) else [folder_path]
        folder_paths = [Path(f) for f in folders]
        if not folder_paths:
            raise ValueError("folder_path must be a path or a non-empty list of paths")

        per_run = [_collect_run_metadata(root, chan) for root in folder_paths]

        # Cross-run consistency: same shape, sampling frequency, plane count
        Ly = per_run[0]["Ly"]
        Lx = per_run[0]["Lx"]
        fs = per_run[0]["fs"]
        nplanes = per_run[0]["nplanes"]
        nchannels = per_run[0]["nchannels"]
        for i, meta in enumerate(per_run[1:], start=1):
            if (meta["Ly"], meta["Lx"]) != (Ly, Lx):
                raise ValueError(
                    f"Run {i} ({folder_paths[i]}) has shape ({meta['Ly']}, {meta['Lx']}) " f"but run 0 has ({Ly}, {Lx})"
                )
            if meta["fs"] != fs:
                raise ValueError(f"Run {i} sampling frequency {meta['fs']} disagrees with run 0 ({fs})")
            if meta["nplanes"] != nplanes:
                raise ValueError(f"Run {i} nplanes {meta['nplanes']} disagrees with run 0 ({nplanes})")

        per_epoch_files: list[list[str]] = [[str(p) for p in meta["plane_files"]] for meta in per_run]

        BinaryImaging.__init__(
            self,
            file_paths=per_epoch_files,
            sampling_frequency=fs,
            shape=(Ly, Lx, nplanes),
            dtype=SUITE2P_BIN_DTYPE,
            file_offset=0,
        )
        # BinaryImaging.__init__ populated _kwargs with the binary spec; keep it
        # for get_binary_description and then overwrite _kwargs with the
        # Suite2p-level reconstruction args so dump/load reinstantiates by folder.
        self._binary_kwargs = self._kwargs

        # suite2p binaries are the registered (motion-corrected) movie
        self.is_registered = True

        self.frames_per_file_per_epoch: list[list[int] | None] = [meta["frames_per_file"] for meta in per_run]
        self.suite2p_roots: list[str] = [str(p.absolute()) for p in folder_paths]
        self.chan: int = chan
        self.nchannels: int = nchannels
        self.name = f"Suite2p chan{chan} ({len(folder_paths)} run{'s' if len(folder_paths) > 1 else ''})"

        self._kwargs = {
            "folder_path": self.suite2p_roots if len(self.suite2p_roots) > 1 else self.suite2p_roots[0],
            "chan": chan,
        }

    def get_binary_description(self):
        bk = self._binary_kwargs
        return dict(
            file_paths=bk["file_paths"],
            dtype=np.dtype(bk["dtype"]),
            shape=bk["shape"],
            file_offset=bk["file_offset"],
        )


def read_suite2p(
    folder_path,
) -> "Suite2pImaging | tuple[Suite2pImaging, Suite2pImaging]":
    """Load suite2p output, returning one object per functional channel.

    Parameters
    ----------
    folder_path : str | Path | list[str | Path]
        Path(s) to suite2p output root(s); see :class:`Suite2pImaging`.

    Returns
    -------
    Suite2pImaging or (Suite2pImaging, Suite2pImaging)
        A single imaging object when the suite2p run was acquired with one
        functional channel; a ``(chan1, chan2)`` tuple when ``nchannels == 2``.
    """
    folders = folder_path if isinstance(folder_path, list) else [folder_path]
    first_root = Path(folders[0])
    first_meta = _collect_run_metadata(first_root, chan=1)
    nchannels = first_meta["nchannels"]

    chan1 = Suite2pImaging(folder_path, chan=1)
    if nchannels == 1:
        return chan1
    return chan1, Suite2pImaging(folder_path, chan=2)
