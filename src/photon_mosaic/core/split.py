import numpy as np

from .baseimaging import BaseImaging


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

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        plane_ids: list | np.ndarray | None = None,
        epoch_index: int | None = None,
    ) -> np.ndarray:
        # parent may want to do further operations to the series before returning
        # so we explicitly delegate to them rather than relying on BaseImaging.
        return self._parent.get_series(
            start_frame=start_frame,
            end_frame=end_frame,
            plane_ids=plane_ids,
            epoch_index=epoch_index,
        )


def split_epochs(imaging: BaseImaging, epoch_indices: int | list[int]) -> SelectEpochImaging:
    """Return a proxy imaging object with only the requested epochs."""
    return SelectEpochImaging(imaging=imaging, epoch_indices=epoch_indices)
