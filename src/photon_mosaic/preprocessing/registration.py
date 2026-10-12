from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

from photon_mosaic.core import BaseImaging, BaseImagingEpoch, KnownMotionMethod, Motion, get_registration_class

from .basepreprocessor import BasePreprocessor, BasePreprocessorEpoch


class RegistrationSettings(BaseSettings):
    """Settings shared by every motion correction backend.

    Only options that are meaningful for any backend live here; algorithm
    specific parameters belong on the subclass (e.g.
    ``Suite2pRegistrationSettings``), which is also expected to set its own
    ``env_prefix``. The prefix here is deliberately not empty: without it these
    fields would read bare ``DEVICE`` / ``BATCH_SIZE`` from the environment.

    Shared fields carry one meaningful name; each field's description records what the
    underlying library calls it (suite2p ``maxregshiftNR``, CaImAn ``pw_rigid``, ...).
    """

    debug: bool = Field(default=False, description="Run with partial dataset")
    tmp_dir: str | Path = Field(
        default=Path("/scratch"),
        description="Directory into which to write temporary files produced by the registration backend",
    )
    batch_size: int = Field(default=500, description="Number of frames per batch")
    device: str = Field(
        default="cpu",
        description="Torch device for registration: 'cpu', 'cuda', or 'mps'.",
    )
    nonrigid: bool = Field(
        default=True,
        description="Correct each block/patch of the field of view separately on top of the rigid shift "
        "(suite2p 'nonrigid', CaImAn 'pw_rigid').",
    )
    max_nonrigid_shift: int = Field(
        default=5,
        description="Maximum pixels a block/patch shift may deviate from the frame's rigid shift "
        "(suite2p 'maxregshiftNR', CaImAn 'max_deviation_rigid').",
    )
    bounds: tuple[tuple[int | None, int | None], tuple[int | None, int | None]] | None = Field(
        default=None,
        description="Region of the field of view used to *estimate* motion, as ((y0, y1), (x0, x1)) slice "
        "bounds; None uses the full frame. Shifts are applied to the full frame regardless "
        "(CaImAn 'indices').",
    )

    model_config = ConfigDict(env_prefix="REGISTRATION_", case_sensitive=False, env_file=".env")


class RegisterImaging(BasePreprocessor):
    """Apply pre-computed motion correction on-the-fly, whatever the backend.

    Dispatch is by ``method`` (like :meth:`Motion.compute`), defaulting to
    ``"suite2p"``, and resolved through
    :func:`photon_mosaic.core.get_registration_class`, so this class imports
    no backend itself.
    """

    def __init__(
        self,
        imaging: BaseImaging,
        motion: Motion,
        method: KnownMotionMethod | str = "suite2p",
        **kwargs: Any,
    ) -> None:
        """Build an imaging view that applies stored motion fields lazily."""
        BasePreprocessor.__init__(self, imaging)

        if motion.num_epochs != len(imaging.epochs):
            raise ValueError(
                f"Number of epochs in motion ({motion.num_epochs}) does not match imaging ({len(imaging.epochs)})"
            )

        registration_class = get_registration_class(method)

        for epoch_idx, parent_epoch in enumerate(imaging.epochs):
            self.add_epoch(registration_class(parent_epoch, motion, epoch_idx, **kwargs))

        self._kwargs = dict(imaging=imaging, motion=motion, method=method, **kwargs)


class RegisterImagingEpoch(BasePreprocessorEpoch):
    """Epoch view that applies stored motion on read; backends implement :meth:`_correct_plane`.

    Holds the bookkeeping every backend shares -- frame-bound checks, plane selection,
    output allocation -- so a backend's epoch class is only the per-plane correction.
    """

    def __init__(
        self,
        parent_imaging_epoch: BaseImagingEpoch,
        motion: Motion,
        epoch_index: int,
        **kwargs: Any,
    ) -> None:
        """Create an epoch preprocessor for a specific epoch and stored motion."""
        BasePreprocessorEpoch.__init__(self, parent_imaging_epoch)
        self.motion = motion
        self.epoch_index = epoch_index
        self.kwargs = kwargs

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: int | slice | Sequence[int] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return motion-corrected frames ``(n_frames, H, W, n_planes)`` for the requested interval and planes."""

        num_samples = self.parent_imaging_epoch.get_num_samples()
        if end_frame > num_samples:
            logging.warning(
                "end_frame %d exceeds recording length %d; clamping. "
                "This usually indicates a miscalculation upstream.",
                end_frame,
                num_samples,
            )
            end_frame = num_samples
        if start_frame > end_frame:
            raise ValueError(
                f"start_frame ({start_frame}) is past end_frame ({end_frame}); " f"recording length is {num_samples}."
            )

        video = self.parent_imaging_epoch.get_series(start_frame, end_frame)
        num_planes = video.shape[3] if video.ndim == 4 else 1

        if plane_indices is None:
            planes_to_process = list(range(num_planes))
        elif isinstance(plane_indices, int):
            planes_to_process = [plane_indices]
        elif isinstance(plane_indices, slice):
            planes_to_process = list(range(*plane_indices.indices(num_planes)))
        else:
            planes_to_process = list(plane_indices)

        n_frames = end_frame - start_frame
        H, W = video.shape[1], video.shape[2]
        output = np.empty((n_frames, H, W, len(planes_to_process)), dtype=np.float32)
        if n_frames == 0:
            return output

        for i, p in enumerate(planes_to_process):
            plane_video = video[:, :, :, p] if video.ndim == 4 else video
            plane_video = plane_video.astype("float32", copy=True)
            if plane_video.ndim == 2:
                plane_video = plane_video[np.newaxis, :, :]

            registered = np.asarray(self._correct_plane(plane_video, p, start_frame, end_frame))
            if registered.ndim == 2:
                registered = registered[np.newaxis, :, :]
            output[..., i] = registered

        return output

    def _correct_plane(
        self,
        plane_video: NDArray[np.floating[Any]],
        plane_index: int,
        start_frame: int,
        end_frame: int,
    ) -> NDArray[np.floating[Any]]:
        """Apply the stored motion of ``plane_index`` to ``plane_video`` ``(n_frames, H, W)`` float32."""

        raise NotImplementedError


register_motion = RegisterImaging
