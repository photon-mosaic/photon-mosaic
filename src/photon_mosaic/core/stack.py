"""Stack imaging objects along the plane axis.

Several imaging objects covering the same field of view at the same timestamps
(each providing one or more planes) are joined into a single multi-plane volume
without copying the underlying pixels — planes are pulled lazily from the parent
objects on read, and only from the parents that actually hold the requested
planes. A general core primitive; see the suite2p extractor for one caller.
"""

from typing import Sequence

import numpy as np

from .baseimaging import BaseImaging, BaseImagingEpoch


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

        # The stacked volume is registered only if every input is.
        self.is_registered = all(im.is_registered for im in imagings)

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
