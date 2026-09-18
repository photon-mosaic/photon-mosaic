"""Operations that derive one imaging object from another.

Every public callable here takes imaging object(s) and hands back a new lazy
proxy: selecting or subdividing epochs, cutting out a frame range, or joining
objects along the plane axis. Nothing copies pixels — the proxies pull from
their parents on read.
"""

from typing import Sequence

import numpy as np

from .baseimaging import BaseImaging, BaseImagingEpoch


def _normalize_epoch_indices(epoch_indices: int | list[int], num_epochs: int) -> list[int]:
    """Normalize and validate epoch indices requested by the user."""
    if isinstance(epoch_indices, int):
        normalized_epoch_indices = [epoch_indices]
    elif isinstance(epoch_indices, list):
        normalized_epoch_indices = epoch_indices
    else:
        raise TypeError("epoch_indices must be an int or a list of ints")

    if len(normalized_epoch_indices) == 0:
        raise ValueError("epoch_indices must contain at least one index")

    for epoch_index in normalized_epoch_indices:
        # isinstance(True, int) returns True, so we need the second check here
        if not isinstance(epoch_index, int) or isinstance(epoch_index, bool):
            raise TypeError("All epoch indices must be ints")
        if not 0 <= epoch_index < num_epochs:
            raise IndexError(f"Epoch index {epoch_index} out of range for imaging with {num_epochs} epochs")

    return normalized_epoch_indices


class SelectEpochImaging(BaseImaging):
    """Proxy imaging object exposing only selected epochs from a parent imaging."""

    def __init__(self, imaging: BaseImaging, epoch_indices: int | list[int]):
        normalized_epoch_indices = _normalize_epoch_indices(epoch_indices, imaging.get_num_epochs())

        BaseImaging.__init__(self, sampling_frequency=imaging.sampling_frequency, shape=imaging.shape)
        imaging.copy_metadata(self)

        for epoch_index in normalized_epoch_indices:
            imaging_epoch = imaging.epochs[epoch_index]
            self.add_epoch(imaging_epoch)

        self._parent = imaging
        self._kwargs = {
            "imaging": imaging,
            "epoch_indices": normalized_epoch_indices,
        }


def split_epochs(imaging: BaseImaging, epoch_indices: int | list[int]) -> SelectEpochImaging:
    """Return a proxy imaging object with only the requested epochs."""
    return SelectEpochImaging(imaging=imaging, epoch_indices=epoch_indices)


class _FrameRangeEpoch(BaseImagingEpoch):
    """Lazy view over a contiguous frame range of a parent epoch."""

    def __init__(self, parent_epoch: BaseImagingEpoch, start_frame: int, end_frame: int):
        t_start = parent_epoch.t_start if getattr(parent_epoch, "t_start", None) is not None else 0.0
        sampling_frequency = parent_epoch.sampling_frequency
        BaseImagingEpoch.__init__(  # type: ignore[call-arg]
            self,
            sampling_frequency=sampling_frequency,
            t_start=t_start + start_frame / sampling_frequency,
        )
        self._parent_epoch = parent_epoch
        self._start = int(start_frame)
        self._end = int(end_frame)

    def get_num_samples(self) -> int:
        return self._end - self._start

    def get_series(self, start_frame, end_frame, plane_indices=None):
        return self._parent_epoch.get_series(
            self._start + start_frame,
            self._start + end_frame,
            plane_indices,
        )


class SplitEpochAtFramesImaging(BaseImaging):
    """Imaging proxy that subdivides one epoch of a parent into several sub-epochs.

    The frame boundaries are interpreted in the same way as ``np.split``: the
    boundaries split the parent epoch into ``len(boundaries) + 1`` contiguous
    pieces, each exposed as its own epoch on the returned object. Pixels are
    pulled lazily from the parent — no data is copied.
    """

    def __init__(self, imaging: BaseImaging, epoch_index: int, frame_boundaries: Sequence[int]):
        num_epochs = imaging.get_num_epochs()
        if not isinstance(epoch_index, int) or isinstance(epoch_index, bool):
            raise TypeError("epoch_index must be an int")
        if not 0 <= epoch_index < num_epochs:
            raise IndexError(f"Epoch index {epoch_index} out of range for imaging with {num_epochs} epochs")

        parent_epoch = imaging.epochs[epoch_index]
        n_samples = parent_epoch.get_num_samples()

        boundaries = list(map(int, frame_boundaries))
        if any(b <= 0 or b >= n_samples for b in boundaries):
            raise ValueError(f"frame_boundaries must be strictly between 0 and {n_samples}; got {boundaries}")
        if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
            raise ValueError(f"frame_boundaries must be strictly increasing; got {boundaries}")

        BaseImaging.__init__(self, sampling_frequency=imaging.sampling_frequency, shape=imaging.shape)
        imaging.copy_metadata(self)

        edges = [0, *boundaries, n_samples]
        for lo, hi in zip(edges[:-1], edges[1:]):
            self.add_epoch(_FrameRangeEpoch(parent_epoch, lo, hi))

        self._parent = imaging
        self._kwargs = {
            "imaging": imaging,
            "epoch_index": epoch_index,
            "frame_boundaries": np.asarray(boundaries, dtype=int).tolist(),
        }


def split_epoch_at_frames(
    imaging: BaseImaging,
    epoch_index: int,
    frame_boundaries: Sequence[int],
) -> SplitEpochAtFramesImaging:
    """Split one epoch of ``imaging`` into contiguous sub-epochs at the given frame boundaries.

    Convenience wrapper for :class:`SplitEpochAtFramesImaging`. Useful when an
    extractor exposes recorded files as a single concatenated epoch (e.g.
    suite2p's per-plane ``data.bin`` plus ``ops['frames_per_file']``) and you
    want each underlying file as its own epoch.
    """
    return SplitEpochAtFramesImaging(imaging=imaging, epoch_index=epoch_index, frame_boundaries=frame_boundaries)


class FrameSliceImaging(BaseImaging):
    """Imaging proxy exposing one contiguous frame range of a parent epoch.

    The imaging counterpart of SpikeInterface's ``FrameSliceRecording``: the way
    to cut a short stub out of a long video, whether for a test fixture, a quick
    look, or to hand something small to code that would choke on the whole
    recording. Pixels are pulled lazily from the parent.

    The result has exactly one epoch. ``t_start`` stays on the parent's
    timeline rather than being re-referenced to zero, so the stub keeps its
    place in the original recording.

    Parameters
    ----------
    parent_imaging : BaseImaging
        The imaging object to slice.
    start_frame : int | None, default: None
        First included frame, or 0 if None.
    end_frame : int | None, default: None
        End frame, excluded as in ordinary python slicing; the parent epoch's
        frame count if None.
    epoch_index : int | None, default: None
        Which epoch of the parent to slice. If None and the parent has a single
        epoch it defaults to that one; a multi-epoch parent must name its epoch,
        as it must for :meth:`BaseImaging.get_series`. SpikeInterface refuses a
        multi-segment parent outright; naming the epoch keeps that capability
        without letting an unnamed one be picked silently (see issue #129).
    """

    def __init__(
        self,
        parent_imaging: BaseImaging,
        start_frame: int | None = None,
        end_frame: int | None = None,
        epoch_index: int | None = None,
    ):
        num_epochs = parent_imaging.get_num_epochs()
        if epoch_index is None:
            if num_epochs > 1:
                raise ValueError("epoch_index must be provided for multi-epoch imaging data.")
            epoch_index = 0
        if not isinstance(epoch_index, int) or isinstance(epoch_index, bool):
            raise TypeError("epoch_index must be an int")
        if not 0 <= epoch_index < num_epochs:
            raise IndexError(f"Epoch index {epoch_index} out of range for imaging with {num_epochs} epochs")

        parent_epoch = parent_imaging.epochs[epoch_index]
        parent_size = parent_epoch.get_num_samples()

        start_frame = 0 if start_frame is None else int(start_frame)
        end_frame = parent_size if end_frame is None else int(end_frame)
        if not 0 <= start_frame < parent_size:
            raise ValueError(f"start_frame must be within [0, {parent_size}); got {start_frame}")
        if not 0 < end_frame <= parent_size:
            raise ValueError(f"end_frame must be within (0, {parent_size}]; got {end_frame}")
        if end_frame <= start_frame:
            raise ValueError(f"start_frame must be smaller than end_frame; got {start_frame} and {end_frame}")

        BaseImaging.__init__(self, sampling_frequency=parent_imaging.sampling_frequency, shape=parent_imaging.shape)
        parent_imaging.copy_metadata(self)
        self.add_epoch(_FrameRangeEpoch(parent_epoch, start_frame, end_frame))

        self._parent = parent_imaging
        self._kwargs = {
            "parent_imaging": parent_imaging,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "epoch_index": epoch_index,
        }


def frame_slice(
    parent_imaging: BaseImaging,
    start_frame: int | None = None,
    end_frame: int | None = None,
    epoch_index: int | None = None,
) -> FrameSliceImaging:
    """Return a single-epoch proxy over ``[start_frame, end_frame)`` of one parent epoch.

    Convenience wrapper for :class:`FrameSliceImaging`.
    """
    return FrameSliceImaging(
        parent_imaging=parent_imaging,
        start_frame=start_frame,
        end_frame=end_frame,
        epoch_index=epoch_index,
    )


class _StackedPlanesEpoch(BaseImagingEpoch):
    """Lazy epoch whose planes are gathered from several parent epochs.

    Each parent epoch contributes a contiguous block of planes; on read the
    blocks are joined along the plane axis into
    ``(num_samples, H, W, sum_of_planes)``. Only the parents holding requested
    planes are read.
    """

    def __init__(
        self,
        parent_epochs: Sequence[BaseImagingEpoch],
        height: int,
        width: int,
        planes_per_parent: Sequence[int],
        dtype: np.dtype,
    ):
        first = parent_epochs[0]
        BaseImagingEpoch.__init__(  # type: ignore[call-arg]
            self,
            sampling_frequency=first.sampling_frequency,
            t_start=getattr(first, "t_start", None),
        )
        self._parent_epochs = list(parent_epochs)
        self._height = int(height)
        self._width = int(width)
        self._dtype = np.dtype(dtype)
        self._planes_per_parent = [int(n) for n in planes_per_parent]
        self._num_planes = int(sum(self._planes_per_parent))
        # Global index of each parent's first plane, so a requested global plane
        # index resolves to (parent, local index) without reading anything.
        self._parent_offsets = np.cumsum([0, *self._planes_per_parent[:-1]]).tolist()

    def get_num_samples(self) -> int:
        return self._parent_epochs[0].get_num_samples()

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: slice | np.ndarray | None = None,
    ) -> np.ndarray:
        requested = self._resolve_plane_indices(plane_indices)
        if not requested:
            return np.empty((end_frame - start_frame, self._height, self._width, 0), dtype=self._dtype)

        # Group the requested planes by the parent holding them, so each parent
        # is read at most once and parents holding none are not read at all.
        local_by_parent: dict[int, list[int]] = {}
        for global_index in requested:
            parent = int(np.searchsorted(self._parent_offsets, global_index, side="right") - 1)
            local_by_parent.setdefault(parent, []).append(global_index - self._parent_offsets[parent])

        by_parent = sorted(local_by_parent.items())
        parts = [
            self._parent_epochs[parent].get_series(start_frame, end_frame, self._parent_selection(parent, local))
            for parent, local in by_parent
        ]
        stacked = np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]

        # ``stacked`` is in parent order; restore the order the caller asked for.
        gathered = [self._parent_offsets[parent] + local for parent, locals_ in by_parent for local in locals_]
        if gathered == requested:
            return stacked
        position = {global_index: i for i, global_index in enumerate(gathered)}
        return stacked[..., [position[global_index] for global_index in requested]]

    def _parent_selection(self, parent: int, local: list[int]) -> slice | np.ndarray:
        """Pick how to ask ``parent`` for ``local``.

        A parent asked for all of its planes in order gets a slice, so it can
        hand back a view instead of an advanced-indexing copy — the whole-volume
        read is the common case and should stay as cheap as it was.
        """
        if local == list(range(self._planes_per_parent[parent])):
            return slice(None)
        return np.asarray(local, dtype=int)

    def _resolve_plane_indices(self, plane_indices: slice | np.ndarray | None) -> list[int]:
        """Normalise ``plane_indices`` to an explicit list of global plane indices.

        Negative indices count from the last plane, as they do on the sibling
        epochs that index a numpy array directly.
        """
        if plane_indices is None:
            return list(range(self._num_planes))
        if isinstance(plane_indices, slice):
            return list(range(*plane_indices.indices(self._num_planes)))
        candidates = np.atleast_1d(plane_indices)
        if candidates.dtype == bool:
            # A boolean mask selects planes, it does not name them by index.
            candidates = np.flatnonzero(candidates)
        indices = []
        for raw in candidates:
            index = int(raw)
            if index < 0:
                index += self._num_planes
            if not 0 <= index < self._num_planes:
                raise IndexError(f"Plane index {int(raw)} out of range for {self._num_planes} planes")
            indices.append(index)
        return indices


class StackPlanesImaging(BaseImaging):
    """Imaging proxy that stacks several imaging objects along the plane axis.

    All inputs must share the same epoch structure (number of epochs and the
    per-epoch frame counts), the same ``(height, width)``, sampling frequency
    and dtype. The resulting object has ``sum(num_planes)`` planes and pulls
    pixels lazily from the inputs — no data is copied.

    Parameters
    ----------
    imagings : sequence of BaseImaging
        Two or more imaging objects covering the same field of view at the
        same timestamps, each providing one or more planes.
    """

    def __init__(self, imagings: Sequence[BaseImaging]):
        imagings = list(imagings)
        if len(imagings) < 2:
            raise ValueError("stack_planes requires at least two imaging objects")
        for i, im in enumerate(imagings):
            if not isinstance(im, BaseImaging):
                raise TypeError(f"Input {i} is not a BaseImaging (got {type(im).__name__})")

        ref = imagings[0]
        num_epochs = ref.get_num_epochs()
        height, width = int(ref.shape[0]), int(ref.shape[1])
        fs = ref.sampling_frequency
        dtype = ref.get_dtype()
        ref_samples = [ref.epochs[e].get_num_samples() for e in range(num_epochs)]

        for i, im in enumerate(imagings[1:], start=1):
            if im.get_num_epochs() != num_epochs:
                raise ValueError(f"Input {i} has {im.get_num_epochs()} epochs but input 0 has {num_epochs}")
            if (int(im.shape[0]), int(im.shape[1])) != (height, width):
                raise ValueError(
                    f"Input {i} has frame shape ({im.shape[0]}, {im.shape[1]}) " f"but input 0 has ({height}, {width})"
                )
            if im.sampling_frequency != fs:
                raise ValueError(f"Input {i} sampling frequency {im.sampling_frequency} disagrees with input 0 ({fs})")
            if np.dtype(im.get_dtype()) != np.dtype(dtype):
                raise ValueError(f"Input {i} dtype {im.get_dtype()} disagrees with input 0 ({dtype})")
            samples = [im.epochs[e].get_num_samples() for e in range(num_epochs)]
            if samples != ref_samples:
                raise ValueError(f"Input {i} per-epoch frame counts {samples} disagree with input 0 ({ref_samples})")

        planes_per_parent = [int(im.num_planes) for im in imagings]
        total_planes = sum(planes_per_parent)
        BaseImaging.__init__(self, sampling_frequency=fs, shape=(height, width, total_planes))

        for epoch_index in range(num_epochs):
            parent_epochs = [im.epochs[epoch_index] for im in imagings]
            self.add_epoch(_StackedPlanesEpoch(parent_epochs, height, width, planes_per_parent, dtype))

        self._parents = imagings
        self._kwargs = {"imagings": imagings}
        self.name = f"Stacked planes ({total_planes} planes from {len(imagings)} objects)"


def stack_planes(*imagings: BaseImaging) -> StackPlanesImaging:
    """Stack imaging objects along the plane axis into one multi-plane volume.

    Accepts the objects either as separate arguments (``stack_planes(a, b)``)
    or as a single sequence (``stack_planes([a, b])``).

    Parameters
    ----------
    *imagings : BaseImaging
        Two or more imaging objects of the same FOV / timestamps; see
        :class:`StackPlanesImaging`.

    Returns
    -------
    StackPlanesImaging
        A lazy multi-plane view over the inputs.
    """
    if len(imagings) == 1 and isinstance(imagings[0], (list, tuple)):
        items: Sequence[BaseImaging] = imagings[0]
    else:
        items = imagings
    return StackPlanesImaging(items)
