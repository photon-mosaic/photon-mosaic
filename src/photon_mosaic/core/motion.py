from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from photon_mosaic.core import BaseImaging


class Motion:
    """Algorithm-agnostic container for motion correction artifacts.

    Holds outputs that any motion correction backend (Suite2P, CaImAn, ...) is
    expected to produce. Backend-specific fields (e.g. Suite2P ``ops`` or
    block-wise non-rigid offsets) belong on dedicated subclasses such as
    ``Suite2PMotion``.
    """

    def __init__(
        self,
        imaging: BaseImaging,
        displacements: Sequence[NDArray[np.floating[Any]]],
        reference: Any = None,
        yranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        xranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        corrected_badframes: Sequence[Sequence[NDArray[np.bool_]]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store displacement fields and shared metadata.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging object associated with the computed motion.
        displacements : Sequence[NDArray]
            Per-epoch rigid displacement arrays shaped ``(frames, planes, 2)``
            or ``(frames, 2)`` in ``(y, x)`` order.
        reference : Any, optional
            Per-plane reference image(s) used by the registration algorithm.
        yranges : Sequence | None, optional
            Per-epoch, per-plane valid pixel row range
            ``[epoch][plane] -> (ymin, ymax)``.
        xranges : Sequence | None, optional
            Per-epoch, per-plane valid pixel column range
            ``[epoch][plane] -> (xmin, xmax)``.
        corrected_badframes : Sequence | None, optional
            Per-epoch, per-plane boolean bad-frame mask
            ``[epoch][plane] -> (n_frames,)``. Combines any input bad-frame
            mask with frames flagged by large registration shifts.
        metadata : dict | None, optional
            Free-form algorithm-agnostic metadata.
        """

        self.imaging = imaging
        self.displacements = displacements
        self.reference = reference
        self.yranges = yranges
        self.xranges = xranges
        self.corrected_badframes = corrected_badframes
        self.metadata = metadata if metadata is not None else {}

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
