"""Concatenate imaging objects along the plane axis.

Several imaging objects covering the same field of view at the same timestamps
(each providing one or more planes) are joined into a single multi-plane volume
without copying the underlying pixels — planes are pulled lazily from the parent
objects on read. A general core primitive; see the suite2p extractor for one
caller.
"""

from typing import Sequence

import numpy as np

from .baseimaging import BaseImaging, BaseImagingEpoch


class _ConcatenatedPlanesEpoch(BaseImagingEpoch):
    """Lazy epoch whose planes are gathered from several parent epochs.

    Each parent epoch contributes its planes; on read they are concatenated
    along the plane axis into ``(num_samples, H, W, sum_of_planes)``.
    """

    def __init__(self, parent_epochs: Sequence[BaseImagingEpoch], height: int, width: int):
        first = parent_epochs[0]
        BaseImagingEpoch.__init__(  # type: ignore[call-arg]
            self,
            sampling_frequency=first.sampling_frequency,
            t_start=getattr(first, "t_start", None),
        )
        self._parent_epochs = list(parent_epochs)
        self._height = int(height)
        self._width = int(width)
        # Number of planes each parent contributes, derived from a tiny probe read
        # so we don't assume a particular epoch attribute name.
        self._planes_per_parent = [pe.get_series(0, 1).shape[-1] for pe in self._parent_epochs]
        self._num_planes = int(sum(self._planes_per_parent))

    def get_num_samples(self) -> int:
        return self._parent_epochs[0].get_num_samples()

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: slice | np.ndarray | None = None,
    ) -> np.ndarray:
        parts = [pe.get_series(start_frame, end_frame, None) for pe in self._parent_epochs]
        full = np.concatenate(parts, axis=-1)
        if plane_indices is None:
            return full
        if isinstance(plane_indices, slice):
            return full[..., plane_indices]
        return full[..., list(plane_indices)]


class ConcatenatePlanesImaging(BaseImaging):
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
            raise ValueError("concatenate_planes requires at least two imaging objects")
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

        total_planes = sum(im.num_planes for im in imagings)
        BaseImaging.__init__(self, sampling_frequency=fs, shape=(height, width, total_planes))

        # The concatenated volume is registered only if every input is.
        self.is_registered = all(im.is_registered for im in imagings)

        for epoch_index in range(num_epochs):
            parent_epochs = [im.epochs[epoch_index] for im in imagings]
            self.add_epoch(_ConcatenatedPlanesEpoch(parent_epochs, height, width))

        self._parents = imagings
        self._kwargs = {"imagings": imagings}
        self.name = f"Concatenated planes ({total_planes} planes from {len(imagings)} objects)"


def concatenate_planes(*imagings: BaseImaging) -> ConcatenatePlanesImaging:
    """Stack imaging objects along the plane axis into one multi-plane volume.

    Accepts the objects either as separate arguments
    (``concatenate_planes(a, b)``) or as a single sequence
    (``concatenate_planes([a, b])``).

    Parameters
    ----------
    *imagings : BaseImaging
        Two or more imaging objects of the same FOV / timestamps; see
        :class:`ConcatenatePlanesImaging`.

    Returns
    -------
    ConcatenatePlanesImaging
        A lazy multi-plane view over the inputs.
    """
    if len(imagings) == 1 and isinstance(imagings[0], (list, tuple)):
        items: Sequence[BaseImaging] = imagings[0]
    else:
        items = imagings
    return ConcatenatePlanesImaging(items)
