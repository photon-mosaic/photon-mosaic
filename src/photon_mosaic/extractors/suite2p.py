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
:func:`read_suite2p` does this for a whole suite2p folder. Per-source-file
epochs are recovered with :func:`split_suite2p_into_files` (use case 3).
"""

from pathlib import Path
from typing import Any

import numpy as np

from photon_mosaic.core.baseimaging import BaseImaging
from photon_mosaic.core.binaryimaging import BinaryImaging
from photon_mosaic.core.concatenate import concatenate_planes
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
            f"Use read_suite2p({str(path)!r}) or concatenate_planes(...) to build a multi-plane volume."
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
    whole suite2p folder via :func:`read_suite2p`.

    Parameters
    ----------
    folder_path : str | Path
        A suite2p plane directory (one that contains ``data.bin`` and ``ops.npy``),
        or a suite2p run root holding a single ``plane0/``. A multi-plane root is
        rejected with a pointer to :func:`read_suite2p` / ``concatenate_planes``.
    chan : int, default: 1
        Functional channel to load (``1`` -> ``data.bin``, ``2`` -> ``data_chan2.bin``).

    Attributes
    ----------
    frames_per_file_per_epoch : list
        The suite2p per-source-file frame counts (``ops['frames_per_file']``) as a
        one-element list, so the single epoch can be subdivided into per-file
        sub-epochs with :func:`split_suite2p_into_files`. ``[None]`` when the run
        did not record ``frames_per_file``.
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
        # Stored as a one-epoch list so split_suite2p_into_files can subdivide it.
        self.frames_per_file_per_epoch: list[list[int] | None] = [frames_per_file]

        self.chan: int = chan
        self.nchannels: int = nchannels
        self.suite2p_root: str = str(plane_dir.absolute())
        self.name = f"Suite2p plane chan{chan}"


class SplitSuite2pIntoFilesImaging(BaseImaging):
    """Imaging proxy that subdivides a suite2p object at its source-file boundaries.

    Suite2p concatenates the source acquisition files into one registered movie
    and records the per-file frame counts in ``ops['frames_per_file']``. A
    suite2p-derived object keeps these as ``frames_per_file_per_epoch`` (one list
    per epoch). This proxy flattens every epoch into its constituent per-file
    sub-epochs, in order, so each original acquisition file becomes its own
    epoch. Pixels are pulled lazily from the parent.

    Parameters
    ----------
    imaging : BaseImaging
        A suite2p-derived imaging object carrying ``frames_per_file_per_epoch``
        (e.g. :class:`Suite2pImaging`, or a multi-plane volume from
        :func:`read_suite2p`).
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

    Reads ``frames_per_file_per_epoch`` and flattens every epoch of ``imaging``
    into its per-file sub-epochs (use case 3 of issue #77).

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
        if len(planes) == 1:
            return planes[0]
        volume = concatenate_planes(planes)
        # Carry suite2p-level metadata onto the stitched view so downstream
        # helpers (e.g. split_suite2p_into_files) keep working.
        volume.is_registered = True
        volume.frames_per_file_per_epoch = planes[0].frames_per_file_per_epoch
        volume.chan = chan
        volume.nchannels = nchannels
        return volume

    chan1 = _volume(1)
    if nchannels == 1:
        return chan1
    return chan1, _volume(2)
