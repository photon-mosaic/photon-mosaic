"""Suite2p ROI extractor.

:class:`Suite2pRois` exposes suite2p detection output as a
:class:`~photon_mosaic.core.baserois.BaseRois`. It can be constructed two ways:

* :class:`Suite2pRois(folder_path) <Suite2pRois>` — load ``stat.npy`` (and
  ``iscell.npy`` if present) from a saved suite2p output folder.
* :meth:`Suite2pRois.from_stat` — wrap the in-memory ``stat`` list returned by
  suite2p detection, without touching disk.

To reload the registered movie from the same folder, pair with
:class:`~photon_mosaic.extractors.Suite2pImaging`.
"""

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import sparse
from numpy.typing import NDArray

from photon_mosaic.core import BaseRois

# Stat fields that are per-pixel or otherwise not exposable as scalar ROI properties.
_SKIP_STAT_PROPERTIES = frozenset({"xpix", "ypix", "lam", "soma_crop", "overlap", "neuropil_mask"})


class Suite2pRois(BaseRois):
    """Suite2p ROIs loaded from a folder or wrapped from in-memory stats."""

    def __init__(
        self,
        folder_path: str | Path,
        sparse: bool = True,
    ) -> None:
        """Load suite2p ROIs from a saved output folder.

        Parameters
        ----------
        folder_path : str or Path
            Folder containing at least ``ops.npy`` and ``stat.npy``. May also
            contain ``iscell.npy`` (classifier output).
        sparse : bool, default: True
            Return image masks as a sparse :class:`sparse.GCXS` array (see
            :meth:`get_roi_image_masks`). Suite2p detections can number in the tens of
            thousands over a large field of view, where a dense array would be far larger
            than the raw movie itself -- but for a small recording with few ROIs, the dense
            array may be smaller and faster to work with than the sparse overhead. Pass
            ``False`` to get a plain ``np.ndarray`` instead.
        """
        folder_path = Path(folder_path)
        ops_file = folder_path / "ops.npy"
        stat_file = folder_path / "stat.npy"
        if not ops_file.is_file():
            raise FileNotFoundError(f"No ops.npy file found in {folder_path}")
        if not stat_file.is_file():
            raise FileNotFoundError(f"No stat.npy file found in {folder_path}")

        ops = np.load(ops_file, allow_pickle=True).item()
        # stat ypix/xpix are in full-frame coordinates, so use Ly/Lx (not the
        # cropped Lyc/Lxc) as the mask shape.
        height = int(ops.get("Ly", ops.get("Lyc")))
        width = int(ops.get("Lx", ops.get("Lxc")))
        sampling_frequency = float(ops["fs"])
        stats = list(np.load(stat_file, allow_pickle=True))

        self._init_from_stats(
            stats=stats,
            shape=(height, width, 1),
            sampling_frequency=sampling_frequency,
            plane_assignments=None,
            sparse=sparse,
        )

        # Optional classifier output.
        iscell_file = folder_path / "iscell.npy"
        if iscell_file.is_file():
            iscell = np.load(iscell_file)
            self.set_property("iscell", iscell[:, 0] == 1)
            self.set_property("iscell_probability", iscell[:, 1])

        self._kwargs = {"folder_path": str(folder_path.absolute()), "sparse": sparse}

    @classmethod
    def from_stat(
        cls,
        stats: Sequence[dict[str, Any]],
        shape: tuple[int, int, int],
        sampling_frequency: float,
        plane_assignments: NDArray[np.intp] | None = None,
        sparse: bool = True,
    ) -> "Suite2pRois":
        """Build ROIs directly from an in-memory suite2p ``stat`` list.

        Parameters
        ----------
        stats : sequence of dict
            Per-ROI stat dicts from suite2p detection. Each must contain
            ``ypix``, ``xpix``, and ``lam``.
        shape : tuple[int, int, int]
            Spatial shape ``(height, width, n_planes)``.
        sampling_frequency : float
            Imaging sampling rate in Hz.
        plane_assignments : NDArray | None, optional
            Integer plane index for each ROI. Required when ``n_planes > 1``.
        sparse : bool, default: True
            Return image masks as a sparse :class:`sparse.GCXS` array instead of a dense
            ``np.ndarray``. See :meth:`__init__`.
        """
        instance = cls.__new__(cls)
        stat_list = list(stats)
        instance._init_from_stats(
            stats=stat_list,
            shape=shape,
            sampling_frequency=float(sampling_frequency),
            plane_assignments=plane_assignments,
            sparse=sparse,
        )
        kwargs: dict[str, Any] = {
            "stats": stat_list,
            "shape": shape,
            "sampling_frequency": float(sampling_frequency),
            "plane_assignments": plane_assignments,
            "sparse": sparse,
        }
        instance._kwargs = kwargs
        return instance

    def _init_from_stats(
        self,
        stats: list[dict[str, Any]],
        shape: tuple[int, int, int],
        sampling_frequency: float,
        plane_assignments: NDArray[np.intp] | None,
        sparse: bool = True,
    ) -> None:
        """Shared initialisation for both constructors."""
        roi_ids = np.arange(len(stats))
        BaseRois.__init__(
            self,
            sampling_frequency=sampling_frequency,
            shape=shape,
            roi_ids=roi_ids,
        )
        self._stats = stats
        self._sparse = sparse
        self._plane_assignments = (
            np.asarray(plane_assignments, dtype=int)
            if plane_assignments is not None
            else np.zeros(len(stats), dtype=int)
        )
        self._set_stat_properties(stats)

    def _set_stat_properties(self, stats: Sequence[dict[str, Any]]) -> None:
        """Expose scalar stat fields as per-ROI properties."""
        if not len(stats):
            return
        for key in stats[0]:
            if key in _SKIP_STAT_PROPERTIES:
                continue
            values = [s[key] for s in stats]
            try:
                self.set_property(key, values)
            except Exception:
                logging.debug("Could not set property %r from stat", key)

    def get_roi_image_masks(self, roi_ids: list[int | str] | None = None) -> sparse.GCXS | NDArray:
        """Return binary image masks shaped ``(n_rois, H, W)`` or ``(n_rois, H, W, n_planes)``.

        Returned as a sparse :class:`sparse.GCXS` array by default, built directly from each
        ROI's ``ypix``/``xpix`` pixel coordinates -- suite2p detections can number in the tens
        of thousands over large (e.g. volumetric) fields of view, where a dense array would be
        far larger than the raw movie itself (see photon-mosaic#103). Pass ``sparse=False`` to
        the constructor for a plain ``np.ndarray`` instead.
        """
        if roi_ids is None:
            roi_ids = self.roi_ids.tolist()

        H, W, n_planes = self.shape
        mask_shape = (0, H, W) if n_planes == 1 else (0, H, W, n_planes)
        if len(roi_ids) == 0:
            if self._sparse:
                # sparse.stack() requires at least one array; build an empty result directly
                # instead, matching the dense case's (0, H, W[, P]) shape.
                return sparse.GCXS.from_numpy(np.zeros(mask_shape, dtype=bool), compressed_axes=(0,))
            return np.zeros(mask_shape, dtype=bool)

        if not self._sparse:
            full_shape = (len(roi_ids), H, W) if n_planes == 1 else (len(roi_ids), H, W, n_planes)
            dense_masks = np.zeros(full_shape, dtype=bool)
            for i, roi_id in enumerate(roi_ids):
                roi_index = int(roi_id)
                stat = self._stats[roi_index]
                ypix = np.asarray(stat["ypix"])
                xpix = np.asarray(stat["xpix"])
                if n_planes == 1:
                    dense_masks[i, ypix, xpix] = True
                else:
                    p = int(self._plane_assignments[roi_index])
                    dense_masks[i, ypix, xpix, p] = True
            return dense_masks

        masks = []
        for roi_id in roi_ids:
            roi_index = int(roi_id)
            stat = self._stats[roi_index]
            ypix = np.asarray(stat["ypix"])
            xpix = np.asarray(stat["xpix"])
            data = np.ones(len(ypix), dtype=bool)
            shape: tuple[int, ...]
            if n_planes == 1:
                coords = np.stack([ypix, xpix])
                shape = (H, W)
            else:
                p = int(self._plane_assignments[roi_index])
                coords = np.stack([ypix, xpix, np.full(len(ypix), p, dtype=ypix.dtype)])
                shape = (H, W, n_planes)
            masks.append(sparse.COO(coords, data, shape=shape))
        # Compress along the ROI axis so per-ROI indexing (e.g. select_rois) stays fast --
        # the default heuristic often picks a different axis, making it ~40x slower.
        return sparse.GCXS.from_coo(sparse.stack(masks, axis=0), compressed_axes=(0,))

    def get_stats(self, roi_ids: list[int | str] | None = None) -> list[dict[str, Any]]:
        """Return the raw suite2p ``stat`` dicts for the given ROIs.

        Unlike :meth:`get_roi_image_masks`, which keeps only ``ypix``/``xpix`` (as a binary
        mask), this exposes the full per-ROI stat dict -- including ``lam`` and ``radius`` --
        needed by consumers that call suite2p's own mask-construction functions directly (e.g.
        ``suite2p.extraction.masks.create_cell_pix``/``create_neuropil_masks``).

        Parameters
        ----------
        roi_ids : list[int | str] | None
            The IDs of the ROIs. If None, all ROIs are returned.

        Returns
        -------
        list[dict]
            The stat dicts, in the order of ``roi_ids`` (or ``self.roi_ids`` if None). Returned
            by reference, not copied -- callers must not mutate the result.
        """
        if roi_ids is None:
            roi_ids = self.roi_ids.tolist()
        return [self._stats[int(roi_id)] for roi_id in roi_ids]


read_suite2p_rois = Suite2pRois
