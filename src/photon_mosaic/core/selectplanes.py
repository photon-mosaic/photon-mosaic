from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .baseimaging import BaseImaging, BaseImagingEpoch


class _SelectPlanesEpoch(BaseImagingEpoch):
    """Lazy view over the selected planes of a parent epoch."""

    def __init__(self, parent_epoch: BaseImagingEpoch, plane_indices: np.ndarray):
        BaseImagingEpoch.__init__(  # type: ignore[call-arg]
            self,
            sampling_frequency=parent_epoch._sampling_frequency,
            t_start=getattr(parent_epoch, "_t_start", None),
        )
        self._parent_epoch = parent_epoch
        self._plane_indices = np.asarray(plane_indices, dtype=int)

    def get_num_samples(self) -> int:
        return self._parent_epoch.get_num_samples()

    def get_series(self, start_frame, end_frame, plane_indices=None):
        selected_indices = np.arange(len(self._plane_indices))
        if plane_indices is not None:
            selected_indices = selected_indices[plane_indices]
        parent_indices = self._plane_indices[selected_indices]
        return self._parent_epoch.get_series(start_frame, end_frame, parent_indices)


class SelectPlanesImaging(BaseImaging):
    """Lazy proxy imaging exposing only a selected subset of source plane IDs."""

    @staticmethod
    def _validate_plane_ids(plane_ids: np.ndarray, available_plane_ids: list | np.ndarray, error_suffix: str) -> None:
        available_plane_ids = set(np.asarray(available_plane_ids).tolist())
        for plane_id in plane_ids:
            if plane_id not in available_plane_ids:
                raise ValueError(f"Plane ID {plane_id} {error_suffix}")

    def __init__(self, imaging: BaseImaging, plane_ids: Sequence):
        plane_ids = np.asarray(plane_ids)
        if len(plane_ids) == 0:
            raise ValueError("plane_ids cannot be empty.")

        parent_plane_ids = imaging.plane_ids
        self._validate_plane_ids(
            plane_ids=plane_ids,
            available_plane_ids=parent_plane_ids,
            error_suffix="not found in parent imaging.",
        )

        shape = (imaging.shape[0], imaging.shape[1], len(plane_ids))
        BaseImaging.__init__(
            self, sampling_frequency=imaging.sampling_frequency, shape=shape, dtype=imaging.get_dtype()
        )
        imaging.copy_metadata(self)

        parent_plane_ids = list(imaging.plane_ids)
        selected_plane_indices = np.asarray([parent_plane_ids.index(plane_id) for plane_id in plane_ids], dtype=int)
        for epoch in imaging.epochs:
            self.add_epoch(_SelectPlanesEpoch(epoch, selected_plane_indices))

        self._selected_plane_ids = plane_ids
        self._parent = imaging
        self._kwargs = {
            "plane_ids": plane_ids,
        }

    @property
    def plane_ids(self):
        return self._selected_plane_ids

    def get_num_planes(self) -> int:
        return len(self._selected_plane_ids)

    def get_series(
        self,
        start_frame: int | None = None,
        end_frame: int | None = None,
        plane_ids: list | np.ndarray | None = None,
        epoch_index: int | None = None,
    ) -> np.ndarray:
        if plane_ids is None:
            requested_plane_ids = self._selected_plane_ids
        else:
            requested_plane_ids = plane_ids
            self._validate_plane_ids(
                plane_ids=requested_plane_ids,
                available_plane_ids=self._selected_plane_ids,
                error_suffix="not available in selected imaging.",
            )

        return self._parent.get_series(
            start_frame=start_frame,
            end_frame=end_frame,
            plane_ids=requested_plane_ids,
            epoch_index=epoch_index,
        )


def select_planes(imaging: BaseImaging, plane_ids: ArrayLike) -> SelectPlanesImaging:
    """Return a lazy proxy imaging object exposing only selected planes."""
    return SelectPlanesImaging(imaging=imaging, plane_ids=plane_ids)
