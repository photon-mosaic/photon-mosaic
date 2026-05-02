import logging
import time
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from photon_mosaic.core import BaseImaging, BaseRois


class Suite2pSegmentationSettings(BaseSettings):
    """Settings for Suite2P ROI detection."""

    diameter: list[float] = Field(
        default=[12.0, 12.0],
        description="Expected cell diameter in pixels [y, x]. 0 lets CellPose estimate it.",
    )
    tau: float = Field(
        default=1.0,
        description="Timescale of the indicator (seconds). GCaMP6s ~1.5, GCaMP6f ~0.7.",
    )
    fs: float = Field(
        default=30.0,
        description="Sampling frequency of the imaging data in Hz.",
    )
    preclassify: float = Field(
        default=0.0,
        description="Apply classifier before refinement with probability threshold. 0 disables.",
    )
    device: str = Field(
        default="cpu",
        description="Torch device for detection: 'cpu', 'cuda', or 'mps'.",
    )

    algorithm: str = Field(
        default="sparsery",
        description=(
            "ROI detection algorithm. Options: 'sparsery' (suite2p sparse mode), "
            "'sourcery' (suite2p dense mode), 'cellpose' (CellPose)."
        ),
    )
    denoise: bool = Field(
        default=False,
        description="Apply denoising to binned movie before cell detection.",
    )
    threshold_scaling: float = Field(
        default=1.0,
        description="Scale the automatically determined detection threshold.",
    )
    max_overlap: float = Field(
        default=0.75,
        description="ROIs with more overlap than this are removed during triage.",
    )
    soma_crop: bool = Field(
        default=True,
        description="Crop dendrites for cell classification stats like compactness.",
    )
    allow_overlap: bool = Field(
        default=False,
        description="If False, overlapping pixels are discarded; if True, added to both ROIs.",
    )
    nbins: int = Field(
        default=5000,
        description="Number of bins for the activity histogram.",
    )
    highpass_time: int = Field(
        default=100,
        description="Window size (frames) for temporal high-pass filter before detection.",
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SUITE2P_SEGMENTATION_", case_sensitive=False, extra="ignore"
    )

    @field_validator("algorithm")
    @classmethod
    def lowercase_algorithm(cls, v: str) -> str:
        return v.lower()

    def merged_with_suite2p_defaults(self) -> dict[str, Any]:
        """Return suite2p's installed detection defaults overlaid with our overrides.

        Used to source per-algorithm sub-dicts (``sparsery_settings``,
        ``sourcery_settings``, ``cellpose_settings``) without hard-coding them
        and to forward ``denoise``/``highpass_time`` etc. to the inner calls.
        """
        from suite2p import default_settings  # noqa: PLC0415

        base: dict[str, Any] = dict(default_settings()["detection"])
        base.update(
            {
                "algorithm": self.algorithm,
                "denoise": self.denoise,
                "threshold_scaling": self.threshold_scaling,
                "max_overlap": self.max_overlap,
                "soma_crop": self.soma_crop,
                "allow_overlap": self.allow_overlap,
                "nbins": self.nbins,
                "highpass_time": self.highpass_time,
            }
        )
        return base


class Suite2pDetectedRois(BaseRois):
    """ROIs produced by running suite2p detection on a registered imaging object.

    Unlike :class:`~photon_mosaic.extractors.Suite2pRois`, this class stores the
    ``stat`` array in memory rather than loading it from a saved folder.
    """

    def __init__(
        self,
        stats: list[dict[str, Any]],
        shape: tuple[int, int, int],
        sampling_frequency: float,
        plane_assignments: NDArray[np.intp] | None = None,
    ) -> None:
        """Create ROIs from an in-memory suite2p ``stat`` list.

        Parameters
        ----------
        stats : list[dict]
            List of per-ROI stat dicts from suite2p detection. Each dict must
            contain ``ypix``, ``xpix``, and ``lam``.
        shape : tuple[int, int, int]
            Spatial shape ``(height, width, n_planes)``.
        sampling_frequency : float
            Imaging sampling rate in Hz.
        plane_assignments : NDArray | None, optional
            Integer plane index for each ROI. Required when ``n_planes > 1``.
        """
        roi_ids = np.arange(len(stats))
        BaseRois.__init__(self, sampling_frequency=sampling_frequency, shape=shape, roi_ids=roi_ids)

        self._stats = stats
        self._plane_assignments = (
            plane_assignments if plane_assignments is not None else np.zeros(len(stats), dtype=int)
        )

        # Expose scalar stat fields as properties
        skip = {"xpix", "ypix", "lam", "soma_crop", "overlap", "neuropil_mask"}
        if stats:
            for key in stats[0]:
                if key in skip:
                    continue
                values = [s[key] for s in stats]
                try:
                    self.set_property(key, values)
                except Exception:
                    logging.debug("Could not set property %r from stat: %s", key, values[:3])

        self._kwargs = dict(
            stats=stats,
            shape=shape,
            sampling_frequency=sampling_frequency,
            plane_assignments=plane_assignments,
        )

    def get_roi_image_masks(self, roi_ids: list[int] | None = None) -> NDArray:
        """Return binary image masks shaped ``(n_rois, H, W)`` or ``(n_rois, H, W, n_planes)``."""
        if roi_ids is None:
            roi_ids = self.roi_ids.tolist()

        H, W, n_planes = self.shape
        masks = []
        for roi_id in roi_ids:
            stat = self._stats[roi_id]
            if n_planes == 1:
                mask = np.zeros((H, W), dtype=bool)
                mask[stat["ypix"], stat["xpix"]] = True
            else:
                mask = np.zeros((H, W, n_planes), dtype=bool)
                p = int(self._plane_assignments[roi_id])
                mask[stat["ypix"], stat["xpix"], p] = True
            masks.append(mask)
        return np.array(masks)


class Suite2pEpochSegmentations:
    """Container of per-epoch :class:`Suite2pDetectedRois` results.

    Returned by :func:`detect_rois_suite2p` when ``scope='per_epoch'`` so that
    each epoch keeps its own ROI set, mirroring the per-epoch structure used by
    motion correction.
    """

    def __init__(self, segmentations: dict[int, Suite2pDetectedRois]) -> None:
        """Wrap an ``{epoch_index: rois}`` mapping.

        Parameters
        ----------
        segmentations : dict[int, Suite2pDetectedRois]
            One entry per processed epoch.
        """
        self._segmentations = dict(segmentations)
        self.epoch_indices: list[int] = sorted(self._segmentations)

    def __getitem__(self, epoch_index: int) -> Suite2pDetectedRois:
        """Return the ROIs detected for ``epoch_index``."""
        return self._segmentations[epoch_index]

    def __iter__(self):
        return iter(self.epoch_indices)

    def __len__(self) -> int:
        return len(self._segmentations)

    def items(self):
        """Iterate ``(epoch_index, rois)`` pairs in epoch order."""
        return ((i, self._segmentations[i]) for i in self.epoch_indices)


def _coerce_segmentation_settings(
    settings: Suite2pSegmentationSettings | dict[str, Any] | None,
) -> Suite2pSegmentationSettings:
    """Normalize user-provided settings into a validated settings object."""
    if settings is None:
        return Suite2pSegmentationSettings()
    if isinstance(settings, dict):
        return Suite2pSegmentationSettings(**settings)
    return settings


def _resolve_ranges(
    value: Sequence[int] | Sequence[Sequence[int]] | None,
    n_epochs: int,
    *,
    default: Sequence[int],
    label: str,
) -> list[list[int]]:
    """Expand a y/x range argument into one ``[lo, hi]`` per epoch.

    Accepts ``None`` (use ``default`` for every epoch), a flat ``[lo, hi]``
    pair (broadcast to every epoch), or a ``(n_epochs, 2)`` sequence with one
    range per epoch.
    """
    if value is None:
        return [list(default) for _ in range(n_epochs)]
    arr = np.asarray(value)
    if arr.ndim == 1:
        if arr.shape[0] != 2:
            raise ValueError(f"{label} must have length 2, got shape {arr.shape}")
        return [arr.tolist() for _ in range(n_epochs)]
    if arr.ndim == 2:
        if arr.shape != (n_epochs, 2):
            raise ValueError(f"{label} must have shape ({n_epochs}, 2), got {arr.shape}")
        return [list(row) for row in arr.tolist()]
    raise ValueError(f"{label} must be 1D or 2D, got ndim={arr.ndim}")


def _resolve_badframes(
    value: NDArray[np.bool_] | Sequence[NDArray[np.bool_] | None] | None,
    n_epochs: int,
    frame_counts: Sequence[int],
) -> list[NDArray[np.bool_] | None]:
    """Normalize a badframes argument into one boolean array (or None) per epoch.

    Accepts ``None``, a single flat ``(total_frames,)`` boolean array (split
    along ``frame_counts``), or a per-epoch sequence of per-epoch arrays.
    """
    if value is None:
        return [None] * n_epochs

    if isinstance(value, np.ndarray):
        total = int(sum(frame_counts))
        if value.shape[0] != total:
            raise ValueError(
                f"badframes length {value.shape[0]} does not match total frame count {total}"
            )
        out: list[NDArray[np.bool_] | None] = []
        offset = 0
        for n in frame_counts:
            out.append(np.asarray(value[offset : offset + n], dtype=bool))
            offset += n
        return out

    if len(value) != n_epochs:
        raise ValueError(
            f"badframes list length {len(value)} does not match number of epochs {n_epochs}"
        )
    out = []
    for i, (b, n) in enumerate(zip(value, frame_counts)):
        if b is None:
            out.append(None)
            continue
        b_arr = np.asarray(b, dtype=bool)
        if b_arr.shape[0] != n:
            raise ValueError(
                f"badframes[{i}] length {b_arr.shape[0]} does not match epoch frame count {n}"
            )
        out.append(b_arr)
    return out


def _read_plane_range(
    imaging: BaseImaging,
    plane_index: int,
    epoch_indices: Sequence[int],
    epoch_offsets: NDArray[np.int64],
    start: int,
    stop: int,
) -> NDArray:
    """Read frames ``[start, stop)`` of one plane, stitched across epochs.

    Pulls only the epoch chunks that overlap the requested global frame
    range via :func:`BaseImagingEpoch.get_series`. Returns whatever dtype
    the underlying epochs produce (no upcast).
    """
    chunks: list[NDArray] = []
    for i, epoch_idx in enumerate(epoch_indices):
        ep_lo = int(epoch_offsets[i])
        ep_hi = int(epoch_offsets[i + 1])
        if ep_hi <= start or ep_lo >= stop:
            continue
        local_start = max(0, start - ep_lo)
        local_end = min(ep_hi - ep_lo, stop - ep_lo)
        frames = imaging.epochs[epoch_idx].get_series(local_start, local_end)
        plane = frames[:, :, :, plane_index] if frames.ndim == 4 else frames
        chunks.append(plane)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


def _stream_bin_movie(
    imaging: BaseImaging,
    plane_index: int,
    epoch_indices: Sequence[int],
    *,
    bin_size: int,
    yrange: Sequence[int],
    xrange: Sequence[int],
    badframes: NDArray[np.bool_] | None,
    nbins: int,
) -> NDArray[np.float32]:
    """Stream-bin one plane across the selected epochs.

    Mirrors :func:`suite2p.detection.detect.bin_movie` but pulls each batch
    via :func:`BaseImagingEpoch.get_series` instead of holding the full
    movie in memory. The only large array kept is the binned ``mov``.
    """
    sizes = [int(imaging.get_num_samples(segment_index=i)) for i in epoch_indices]
    epoch_offsets = np.cumsum([0, *sizes]).astype(np.int64)
    n_frames = int(epoch_offsets[-1])

    good_frames = ~badframes if badframes is not None else np.ones(n_frames, dtype=bool)
    n_good = int(good_frames.sum())
    if n_good == 0:
        raise ValueError("no good frames available for binning")
    batch_size = min(n_good, 500)

    Lyc = int(yrange[1] - yrange[0])
    Lxc = int(xrange[1] - xrange[0])

    num_binned_frames = min(nbins, n_frames // bin_size)
    mov = np.zeros((num_binned_frames, Lyc, Lxc), np.float32)
    curr = 0

    tstarts = np.arange(0, n_frames, batch_size)
    bins_per_batch = max(1, batch_size // bin_size)
    n_batches = min(nbins // bins_per_batch, len(tstarts))
    if n_batches < 1:
        n_batches = min(1, len(tstarts))
    tstarts = tstarts[np.linspace(0, len(tstarts) - 1, n_batches, dtype=int)]

    for raw_tstart in tstarts:
        tstart = int(raw_tstart)
        tend = min(tstart + batch_size, n_frames)
        data = _read_plane_range(
            imaging, plane_index, epoch_indices, epoch_offsets, tstart, tend
        )

        good = good_frames[tstart:tend]
        if good.mean() > 0.5:
            data = data[good]

        data = data[:, slice(*yrange), slice(*xrange)]

        if data.shape[0] > bin_size:
            n_d = data.shape[0]
            data = data[: (n_d // bin_size) * bin_size]
            data = data.reshape(-1, bin_size, Lyc, Lxc).astype(np.float32).mean(axis=1)
        else:
            data = data.mean(axis=0).astype(np.float32)[np.newaxis, :, :]

        if mov.shape[0] > curr:
            nb = data.shape[0]
            mov[curr : curr + nb] = data
            curr += nb

    return mov[:curr]


def _detect_rois_from_mov(
    mov: NDArray[np.float32],
    *,
    cfg: Suite2pSegmentationSettings,
    Ly: int,
    Lx: int,
    yrange: Sequence[int],
    xrange: Sequence[int],
) -> list[dict[str, Any]]:
    """Run the configured suite2p detection algorithm on a binned movie.

    Replaces :func:`suite2p.detection.detect.detection_wrapper` with a
    direct dispatch onto the algorithm-specific entry points
    (``sparsedetect.sparsery``, ``sourcery.sourcery``, or
    ``anatomical.select_rois``) so each backend's settings are wired
    explicitly. Returns the post-:func:`roi_stats` stat list with pixel
    coordinates already shifted back to the full image.
    """
    import torch  # noqa: PLC0415
    from suite2p.detection import anatomical, sourcery, sparsedetect, utils  # noqa: PLC0415
    from suite2p.detection.stats import roi_stats  # noqa: PLC0415

    s = cfg.merged_with_suite2p_defaults()
    device = torch.device(cfg.device)
    diameter = np.array(cfg.diameter, dtype=float)

    if s.get("denoise", False):
        from suite2p.detection.denoise import pca_denoise  # noqa: PLC0415

        mov = pca_denoise(mov, block_size=s["block_size"], n_comps_frac=0.5)

    meanImg = mov.mean(axis=0)
    mov = utils.temporal_high_pass_filter(mov=mov, width=s["highpass_time"])
    max_proj = mov.max(axis=0)

    use_cellpose = cfg.algorithm == "cellpose" and anatomical.CELLPOSE_INSTALLED
    if cfg.algorithm == "cellpose" and not anatomical.CELLPOSE_INSTALLED:
        logging.warning(
            "cellpose requested but not installed (%s); falling back to sparsery",
            anatomical.cellpose_error,
        )

    if use_cellpose:
        _, raw_stat = anatomical.select_rois(
            meanImg,
            max_proj,
            settings=s["cellpose_settings"],
            diameter=diameter,
            device=device,
        )
    else:
        algo = "sparsery" if cfg.algorithm == "cellpose" else cfg.algorithm
        sdmov = utils.standard_deviation_over_time(mov, batch_size=1000)
        if algo == "sparsery":
            _, raw_stat = sparsedetect.sparsery(
                mov=mov,
                sdmov=sdmov,
                threshold_scaling=cfg.threshold_scaling,
                **s["sparsery_settings"],
            )
        elif algo == "sourcery":
            _, raw_stat = sourcery.sourcery(
                mov=mov,
                sdmov=sdmov,
                diameter=diameter,
                threshold_scaling=cfg.threshold_scaling,
                **s["sourcery_settings"],
            )
        else:
            raise ValueError(f"unknown algorithm {cfg.algorithm!r}")

    stat = list(raw_stat)
    if not stat:
        return []

    # shift ROI pixel coordinates from cropped frame back to full image
    ymin, xmin = int(yrange[0]), int(xrange[0])
    for st in stat:
        st["ypix"] += ymin
        st["xpix"] += xmin
        if "med" in st:
            st["med"][0] += ymin
            st["med"][1] += xmin

    # roi_stats indexes its first arg with a boolean mask, so it must be a
    # numpy object array — not a plain list.
    stat_array = np.array(stat, dtype=object)
    return list(
        roi_stats(
            stat_array,
            Ly,
            Lx,
            diameter=diameter,
            max_overlap=cfg.max_overlap,
            do_soma_crop=cfg.soma_crop,
            median=use_cellpose,
        )
    )


def _detect_segmentation(
    imaging: BaseImaging,
    epoch_indices: Sequence[int],
    yrange: Sequence[int],
    xrange: Sequence[int],
    badframes: NDArray[np.bool_] | None,
    cfg: Suite2pSegmentationSettings,
) -> Suite2pDetectedRois:
    """Bin and detect ROIs over a single contiguous span of epochs."""
    H, W, n_planes = imaging.shape

    n_frames = int(
        sum(imaging.get_num_samples(segment_index=i) for i in epoch_indices)
    )
    bin_size = int(max(1, n_frames // cfg.nbins, np.round(cfg.tau * cfg.fs)))

    all_stats: list[dict[str, Any]] = []
    plane_assignments: list[int] = []

    for plane_index in range(n_planes):
        logging.info(
            "Suite2p detection — plane %d/%d (epochs %s)",
            plane_index + 1,
            n_planes,
            list(epoch_indices),
        )
        t0 = time.time()

        mov = _stream_bin_movie(
            imaging,
            plane_index=plane_index,
            epoch_indices=epoch_indices,
            bin_size=bin_size,
            yrange=yrange,
            xrange=xrange,
            badframes=badframes,
            nbins=cfg.nbins,
        )

        stat = _detect_rois_from_mov(
            mov,
            cfg=cfg,
            Ly=H,
            Lx=W,
            yrange=yrange,
            xrange=xrange,
        )

        logging.info("  found %d ROIs in %.1fs", len(stat), time.time() - t0)
        all_stats.extend(stat)
        plane_assignments.extend([plane_index] * len(stat))

    rois = Suite2pDetectedRois(
        stats=all_stats,
        shape=(H, W, n_planes),
        sampling_frequency=imaging.sampling_frequency,
        plane_assignments=np.array(plane_assignments, dtype=int),
    )
    rois.register_imaging(imaging)
    return rois


def detect_rois_suite2p(
    imaging: BaseImaging,
    yrange: Sequence[int] | Sequence[Sequence[int]] | None = None,
    xrange: Sequence[int] | Sequence[Sequence[int]] | None = None,
    badframes: NDArray[np.bool_] | Sequence[NDArray[np.bool_] | None] | None = None,
    *,
    scope: Literal["all_epochs", "per_epoch"] = "all_epochs",
    epoch_indices: Sequence[int] | None = None,
    settings: Suite2pSegmentationSettings | dict[str, Any] | None = None,
) -> Suite2pDetectedRois | Suite2pEpochSegmentations:
    """Detect ROIs using suite2p detection on a registered imaging object.

    Honours the imaging Epoch model: ``scope='per_epoch'`` runs detection
    independently per epoch and returns a :class:`Suite2pEpochSegmentations`,
    while ``scope='all_epochs'`` (default) concatenates the selected epochs and
    returns a single :class:`Suite2pDetectedRois`. Frames are pulled lazily via
    :func:`BaseImagingEpoch.get_series` and stream-binned in place, so the full
    raw movie is never materialised.

    Parameters
    ----------
    imaging : BaseImaging
        Motion-corrected imaging object.
    yrange, xrange : sequence of int | sequence of sequence of int | None, optional
        Valid pixel range ``[lo, hi]``. May be a single 1D pair (broadcast to
        every epoch) or a 2D ``(n_epochs, 2)`` sequence with one pair per
        epoch. ``None`` defaults to the full image extent. With
        ``scope='all_epochs'`` per-epoch ranges must be uniform.
    badframes : NDArray[bool] | sequence of NDArray[bool] | None, optional
        Frames to exclude. May be a single flat ``(total_frames,)`` boolean
        array or a per-epoch sequence (each shaped ``(n_frames_in_epoch,)``).
    scope : {"all_epochs", "per_epoch"}, optional
        ``"all_epochs"`` concatenates the selected epochs and returns one
        :class:`Suite2pDetectedRois`. ``"per_epoch"`` runs detection per epoch
        and returns a :class:`Suite2pEpochSegmentations`.
    epoch_indices : sequence of int | None, optional
        Epochs to include (in order). Defaults to all epochs of ``imaging``.
    settings : Suite2pSegmentationSettings | dict | None, optional
        Detection settings. Dicts are coerced into ``Suite2pSegmentationSettings``.

    Returns
    -------
    Suite2pDetectedRois | Suite2pEpochSegmentations
        Single ROI set when ``scope='all_epochs'``; one ROI set per epoch
        otherwise.
    """
    cfg = _coerce_segmentation_settings(settings)
    H, W, _ = imaging.shape

    selected = (
        list(range(imaging.get_num_epochs())) if epoch_indices is None else list(epoch_indices)
    )
    n_eps = len(selected)
    frame_counts = [int(imaging.get_num_samples(segment_index=i)) for i in selected]

    yranges = _resolve_ranges(yrange, n_eps, default=[0, H], label="yrange")
    xranges = _resolve_ranges(xrange, n_eps, default=[0, W], label="xrange")
    bf_per_ep = _resolve_badframes(badframes, n_eps, frame_counts)

    if scope == "all_epochs":
        if not all(yr == yranges[0] for yr in yranges):
            raise ValueError(
                "yrange differs across selected epochs; use scope='per_epoch' "
                "or pass a single uniform range."
            )
        if not all(xr == xranges[0] for xr in xranges):
            raise ValueError(
                "xrange differs across selected epochs; use scope='per_epoch' "
                "or pass a single uniform range."
            )

        if any(b is not None for b in bf_per_ep):
            merged_bf: NDArray[np.bool_] | None = np.concatenate(
                [
                    b if b is not None else np.zeros(n, dtype=bool)
                    for b, n in zip(bf_per_ep, frame_counts)
                ],
                axis=0,
            )
        else:
            merged_bf = None

        return _detect_segmentation(
            imaging,
            epoch_indices=selected,
            yrange=yranges[0],
            xrange=xranges[0],
            badframes=merged_bf,
            cfg=cfg,
        )

    if scope == "per_epoch":
        results: dict[int, Suite2pDetectedRois] = {}
        for i, ep_idx in enumerate(selected):
            results[ep_idx] = _detect_segmentation(
                imaging,
                epoch_indices=[ep_idx],
                yrange=yranges[i],
                xrange=xranges[i],
                badframes=bf_per_ep[i],
                cfg=cfg,
            )
        return Suite2pEpochSegmentations(results)

    raise ValueError(f"scope must be 'all_epochs' or 'per_epoch', got {scope!r}")
