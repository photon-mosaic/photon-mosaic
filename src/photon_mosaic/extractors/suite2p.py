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

from photon_mosaic.core.baseimaging import BaseImaging
from photon_mosaic.core.binaryimaging import BinaryImaging
from photon_mosaic.core.split import split_epoch_at_frames

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


def _require_consistent(
    items: list[dict[str, Any]],
    fields: list[tuple[str, Any]],
    *,
    noun: str,
    labels: list[str],
) -> dict[str, Any]:
    """Require every mapping in ``items`` to agree with ``items[0]`` on each ``(key, cast)``
    field; return ``items[0]``'s cast values, or raise ValueError naming the first item that
    disagrees. ``cast`` (int/float) normalises values first; ``noun``/``labels`` label the
    item kind and each item in that message.
    """
    reference = {key: cast(items[0][key]) for key, cast in fields}
    for i, item in enumerate(items):
        for key, cast in fields:
            value = cast(item[key])
            if value != reference[key]:
                raise ValueError(
                    f"{noun} {i} ({labels[i]}) {key}={value} disagrees with {noun} 0 ({reference[key]})"
                )
    return reference


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

    geometry = _require_consistent(
        per_plane_ops,
        [("Ly", int), ("Lx", int), ("nframes", int), ("fs", float), ("nchannels", int)],
        noun="Plane",
        labels=[pd.name for pd in plane_dirs],
    )
    Ly = geometry["Ly"]
    Lx = geometry["Lx"]
    fs = geometry["fs"]
    nframes = geometry["nframes"]
    nchannels = geometry["nchannels"]

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

        # Cross-run consistency: same shape, sampling frequency, plane count.
        # (nframes legitimately differs between runs; nchannels is taken from run 0.)
        geometry = _require_consistent(
            per_run,
            [("Ly", int), ("Lx", int), ("fs", float), ("nplanes", int)],
            noun="Run",
            labels=[str(p) for p in folder_paths],
        )
        Ly = geometry["Ly"]
        Lx = geometry["Lx"]
        fs = geometry["fs"]
        nplanes = geometry["nplanes"]
        nchannels = per_run[0]["nchannels"]

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


class SplitSuite2pIntoFilesImaging(BaseImaging):
    """Imaging proxy that subdivides a suite2p object at its source-file boundaries.

    Suite2p concatenates the source acquisition files into one registered movie
    per run and records the per-file frame counts in ``ops['frames_per_file']``.
    A :class:`Suite2pImaging` keeps these as ``frames_per_file_per_epoch`` (one
    list per run/epoch). This proxy flattens **every** epoch of the input into
    its constituent per-file sub-epochs, in order, so each original acquisition
    file becomes its own epoch. Pixels are pulled lazily from the parent.

    Parameters
    ----------
    imaging : BaseImaging
        A suite2p-derived imaging object carrying ``frames_per_file_per_epoch``
        (e.g. :class:`Suite2pImaging`).
    """

    def __init__(self, imaging: BaseImaging):
        fpf_per_epoch = getattr(imaging, "frames_per_file_per_epoch", None)
        if fpf_per_epoch is None:
            raise TypeError("imaging has no 'frames_per_file_per_epoch'; autosplit needs a suite2p-derived object")
        if any(fpf is None for fpf in fpf_per_epoch):
            raise ValueError("ops['frames_per_file'] is missing for at least one epoch; cannot autosplit this object")

        BaseImaging.__init__(self, sampling_frequency=imaging.sampling_frequency, shape=imaging.shape)
        # Carries is_registered (and other metadata) from the suite2p parent.
        imaging.copy_metadata(self)

        for epoch_index, frames_per_file in enumerate(fpf_per_epoch):
            n_samples = imaging.epochs[epoch_index].get_num_samples()
            if sum(frames_per_file) != n_samples:
                raise ValueError(
                    f"frames_per_file for epoch {epoch_index} sum to {sum(frames_per_file)} "
                    f"but the epoch has {n_samples} frames"
                )
            boundaries = np.cumsum(frames_per_file)[:-1].tolist()
            # Reuse the generic splitter; [] boundaries (single-file epoch) yields the whole epoch.
            per_file = split_epoch_at_frames(imaging, epoch_index, boundaries)
            for epoch in per_file.epochs:
                self.add_epoch(epoch)

        self._parent = imaging
        self._kwargs = {"imaging": imaging}
        self.name = f"Suite2p per-file split ({self.get_num_epochs()} files)"


def split_suite2p_into_files(imaging: BaseImaging) -> SplitSuite2pIntoFilesImaging:
    """Split a suite2p imaging object into one epoch per source acquisition file.

    Reads ``frames_per_file_per_epoch`` and flattens every run/epoch of
    ``imaging`` into its per-file sub-epochs (use case 3 of issue #77).

    Parameters
    ----------
    imaging : BaseImaging
        A suite2p-derived imaging object (see :class:`Suite2pImaging`).

    Returns
    -------
    SplitSuite2pIntoFilesImaging
        A lazy view with one epoch per original acquisition file.
    """
    return SplitSuite2pIntoFilesImaging(imaging)


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
