from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from photon_mosaic.core import BaseImaging


class Motion:
    """Container for Suite2P motion correction artifacts."""

    def __init__(
        self,
        imaging: BaseImaging,
        displacements: Sequence[NDArray[np.floating[Any]]],
        refAndMasks: Any,
        ops: dict[str, Any],
        nonrigid_offsets: Sequence[Sequence[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]] | None]
        | Sequence[Sequence[tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]] | None]]
        | None = None,
        blocks: Sequence[Any] | None = None,
        yranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        xranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        corrected_badframes: Sequence[Sequence[NDArray[np.bool_]]] | None = None,
        registration_outputs: dict | None = None,
    ) -> None:
        """Store displacement fields and metadata produced by Suite2P.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging object associated with the computed motion.
        displacements : Sequence[NDArray]
            Per-epoch displacement arrays shaped ``(frames, planes, 2)`` or ``(frames, 2)``.
        refAndMasks : Any
            Suite2P reference images returned by ``_compute_reference_wrapper``.
        ops : dict[str, Any]
            Suite2P options used during registration.
        nonrigid_offsets : Sequence | None, optional
            Per-epoch, per-plane non-rigid offsets.
            Structure: ``[epoch][plane] -> (yoff1, xoff1)`` each ``(n_frames, n_blocks)``.
        blocks : Sequence | None, optional
            Suite2P block definitions when non-rigid registration is enabled.
        yranges : Sequence | None, optional
            Per-epoch, per-plane valid pixel row range ``[epoch][plane] -> (ymin, ymax)``.
        xranges : Sequence | None, optional
            Per-epoch, per-plane valid pixel column range ``[epoch][plane] -> (xmin, xmax)``.
        corrected_badframes : Sequence | None, optional
            Per-epoch, per-plane boolean bad-frame mask ``[epoch][plane] -> (n_frames,)``.
            Combines the input ``badframes`` with frames flagged by large registration shifts.
        registration_outputs : dict | None, optional
            Additional outputs from the registration pipeline.
        """

        self.imaging = imaging
        self.displacements = displacements
        self.refAndMasks = refAndMasks
        self.ops = ops
        self.nonrigid_offsets = nonrigid_offsets
        self.blocks = blocks
        self.yranges = yranges
        self.xranges = xranges
        self.corrected_badframes = corrected_badframes
        self.registration_outputs = registration_outputs

    @property
    def num_epochs(self) -> int:
        """Number of epochs represented in the stored displacements."""

        return len(self.displacements)

    def get_displacement_at_frames(
        self,
        frame_indices: int | NDArray[np.integer],
        plane_index: int | None = None,
        epoch_index: int = 0,
    ) -> NDArray[np.floating[Any]]:
        """Return displacement vectors for the requested frames.

        With ``plane_index=None`` the planes axis is preserved
        (``(..., n_planes, 2)``); with an integer ``plane_index`` it is dropped
        (``(..., 2)``).
        """

        disps = self.displacements[epoch_index]
        if plane_index is None:
            return disps[frame_indices]
        return disps[frame_indices, plane_index]
