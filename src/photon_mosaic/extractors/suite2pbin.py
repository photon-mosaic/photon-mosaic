"""Load suite2p registered binary movies as photon-mosaic imaging objects.

Suite2p stores the registered functional movie as one ``data.bin`` per plane
under ``<suite2p_root>/plane{i}/`` together with an ``ops.npy`` describing the
geometry (``Ly``, ``Lx``, ``fs``, ``nchannels``) and the per-source-file frame
counts (``frames_per_file``). When two functional channels are present the
second channel lives next to ``data.bin`` as ``data_chan2.bin``.

Following issue #77, each :class:`Suite2pImaging` loads **one plane** as a
:class:`~photon_mosaic.core.binaryimaging.BinaryImaging` (use cases 1-2).
Multi-plane volumes are built by stitching single-plane objects with
:func:`~photon_mosaic.core.concatenate.concatenate_planes`; the convenience
:func:`read_suite2p_full` does this for a whole suite2p folder. Per-source-file
epochs are recovered with :meth:`Suite2pImaging.into_epochs` (use case 3).

The module-level aliases mirror the ROI extractor's convention:
``read_suite2p_binary`` is :class:`Suite2pImaging` (one plane) and
``read_suite2p_full`` loads and stitches a whole folder.
"""

from pathlib import Path
from typing import Any

import numpy as np

from photon_mosaic.core.baseimaging import BaseImaging
from photon_mosaic.core.binaryimaging import BinaryImaging
from photon_mosaic.core.concatenate import concatenate_planes
from photon_mosaic.core.split import SplitEpochAtFramesImaging, split_epoch_at_frames

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


def _resolve_plane_dir(path: Path) -> Path:
    """Resolve ``path`` to a single suite2p plane directory.

    Accepts either a plane directory (one that contains ``ops.npy``) or a
    suite2p run root that holds exactly one ``plane*/``. A root with several
    planes is rejected with a pointer to the multi-plane entry points.
    """
    if (path / "ops.npy").exists():
        return path
    plane_dirs = _list_plane_dirs(path)
    if len(plane_dirs) != 1:
        raise ValueError(
            f"{path} contains {len(plane_dirs)} planes; Suite2pImaging loads a single plane. "
            f"Use read_suite2p_full({str(path)!r}) or concatenate_planes(...) to build a multi-plane volume."
        )
    return plane_dirs[0]


class Suite2pImaging(BinaryImaging):
    """A single plane of a suite2p registered movie, exposed as a :class:`BinaryImaging`.

    suite2p writes one registered ``data.bin`` per plane; this class memory-maps
    that binary lazily (no pixels are copied) and marks the movie as registered
    (:attr:`~photon_mosaic.core.baseimaging.BaseImaging.is_registered` is ``True``).
    One object holds exactly one plane: following issue #77, a multi-plane volume is
    assembled from several single-plane objects with
    :func:`~photon_mosaic.core.concatenate.concatenate_planes`, or in one step for a
    whole suite2p folder via :func:`read_suite2p_full`.

    Parameters
    ----------
    folder_path : str | Path
        A suite2p plane directory (one that contains ``data.bin`` and ``ops.npy``),
        or a suite2p run root holding a single ``plane0/``. A multi-plane root is
        rejected with a pointer to :func:`read_suite2p_full` / ``concatenate_planes``.
    chan : int, default: 1
        Functional channel to load (``1`` -> ``data.bin``, ``2`` -> ``data_chan2.bin``).

    Attributes
    ----------
    frames_per_file_per_epoch : list
        The suite2p per-source-file frame counts (``ops['frames_per_file']``) as a
        one-element list, so the single epoch can be subdivided into per-file
        sub-epochs with :meth:`into_epochs`. ``[None]`` when the run did not
        record ``frames_per_file``.
    chan, nchannels : int
        The functional channel loaded, and the number of channels in the run.
    suite2p_root : str
        Absolute path to the plane directory this object was loaded from.
    """

    def __init__(self, folder_path, chan: int = 1):
        plane_dir = _resolve_plane_dir(Path(folder_path))
        ops = _load_ops(plane_dir)
        Ly, Lx = int(ops["Ly"]), int(ops["Lx"])
        fs = float(ops["fs"])
        nchannels = int(ops["nchannels"])

        if chan > nchannels:
            raise FileNotFoundError(
                f"Requested chan={chan} but ops['nchannels']={nchannels}; "
                f"no per-plane binary exists for this channel"
            )
        bin_path = plane_dir / _data_bin_name(chan)
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing {bin_path.name} in {plane_dir}")

        # A single plane is one interleaved binary -> a plain BinaryImaging. We
        # keep BinaryImaging's binary spec in _kwargs untouched so this object
        # stays binary-compatible (get_binary_description works); multi-plane
        # volumes are assembled separately by concatenate_planes.
        BinaryImaging.__init__(
            self,
            file_paths=str(bin_path),
            sampling_frequency=fs,
            shape=(Ly, Lx, 1),
            dtype=SUITE2P_BIN_DTYPE,
            file_offset=0,
        )

        # suite2p binaries are the registered (motion-corrected) movie
        self.is_registered = True

        fpf_raw = ops.get("frames_per_file")
        frames_per_file = None if fpf_raw is None else np.asarray(fpf_raw, dtype=int).tolist()
        # Stored as a one-epoch list so into_epochs can subdivide it.
        self.frames_per_file_per_epoch: list[list[int] | None] = [frames_per_file]

        self.chan: int = chan
        self.nchannels: int = nchannels
        self.suite2p_root: str = str(plane_dir.absolute())
        self.name = f"Suite2p plane chan{chan}"

    def into_epochs(self) -> SplitEpochAtFramesImaging:
        """Split this movie into one epoch per source acquisition file.

        suite2p concatenates the source acquisition files into a single registered
        movie and records the per-file frame counts in ``ops['frames_per_file']``.
        This recovers one epoch per original file (use case 3 of issue #77) by
        recycling :func:`~photon_mosaic.core.split.split_epoch_at_frames` at those
        boundaries — pixels stay lazy, nothing is re-read.

        Returns
        -------
        SplitEpochAtFramesImaging
            A lazy view with one epoch per source acquisition file.

        Raises
        ------
        ValueError
            If the run did not record ``ops['frames_per_file']``, or those counts
            do not sum to the movie's frame count (a malformed ``ops``).
        """
        frames_per_file = self.frames_per_file_per_epoch[0]
        if frames_per_file is None:
            raise ValueError("ops['frames_per_file'] was not recorded; cannot split this run into files")
        n_samples = self.epochs[0].get_num_samples()
        if sum(frames_per_file) != n_samples:
            raise ValueError(
                f"ops['frames_per_file'] sums to {sum(frames_per_file)} but the movie has "
                f"{n_samples} frames; refusing to split a malformed run"
            )
        boundaries = np.cumsum(frames_per_file)[:-1].tolist()
        return split_epoch_at_frames(self, 0, boundaries)


def read_suite2p_full(
    folder_path,
) -> "BaseImaging | tuple[BaseImaging, BaseImaging]":
    """Load a suite2p output folder, one imaging object per functional channel.

    Each plane is loaded as a single-plane :class:`Suite2pImaging` and the planes
    are stitched into one volume with
    :func:`~photon_mosaic.core.concatenate.concatenate_planes` (use cases 1-2 of
    issue #77).

    Parameters
    ----------
    folder_path : str | Path
        Path to a suite2p output root (the parent of ``plane0/``, ``plane1/``, ...).

    Returns
    -------
    BaseImaging or (BaseImaging, BaseImaging)
        A single imaging object when the run has one functional channel; a
        ``(chan1, chan2)`` tuple when ``nchannels == 2``. Single-plane runs
        return the :class:`Suite2pImaging` directly; multi-plane runs return the
        stitched volume.
    """
    root = Path(folder_path)
    plane_dirs = _list_plane_dirs(root)
    nchannels = int(_load_ops(plane_dirs[0])["nchannels"])

    def _volume(chan: int) -> BaseImaging:
        planes = [Suite2pImaging(plane_dir, chan=chan) for plane_dir in plane_dirs]
        # concatenate_planes already propagates is_registered from the planes; the
        # stitched volume stays a plain multi-plane imaging (no suite2p specifics).
        return planes[0] if len(planes) == 1 else concatenate_planes(*planes)

    chan1 = _volume(1)
    if nchannels == 1:
        return chan1
    return chan1, _volume(2)


# Aliases mirroring the ROI extractor's read_* convention (cf. read_suite2p_rois).
# read_suite2p_binary loads one plane's binary; read_suite2p_full loads a folder.
read_suite2p_binary = Suite2pImaging
