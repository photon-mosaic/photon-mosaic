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


def select_epochs(imaging: BaseImaging, epoch_indices: int | list[int]) -> SelectEpochImaging:
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
