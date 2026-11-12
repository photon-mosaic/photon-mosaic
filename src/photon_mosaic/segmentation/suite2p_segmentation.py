import importlib.metadata
import logging
import time
from typing import Annotated, Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from photon_mosaic.core import BaseImaging
from photon_mosaic.extractors.suite2prois import Suite2pRois

from .basesegmentation import BaseSegmentation, register_segmentation_class


class Suite2pSegmentationSettings(BaseSettings):
    """Settings for Suite2P ROI detection."""

    diameter: list[Annotated[float, Field(ge=1)]] = Field(
        default=[12.0, 12.0],
        description=(
            "Expected cell diameter in pixels [y, x], each at least 1. Cellpose rescales the image by "
            "30 / diameter, so [30, 30] runs Cellpose-SAM at its native scale, as when no diameter is "
            "given; use the same value for y and x. Sourcery uses it to set its smoothing and neuropil scales."
        ),
    )
    tau: float = Field(
        default=1.0,
        description="Timescale of the indicator (seconds). GCaMP6s ~1.5, GCaMP6f ~0.7.",
    )
    preclassify: float = Field(
        default=0.0,
        description="Apply classifier before refinement with probability threshold. 0 disables. Not used yet.",
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
        description="If False, overlapping pixels are discarded; if True, added to both ROIs. Not used yet.",
    )
    nbins: int = Field(
        default=5000,
        description="Number of bins for the activity histogram.",
    )
    bin_size: int | None = Field(
        default=None,
        description=(
            "Frames per bin. None uses max(1, n_frames // nbins, round(tau * fs)), where fs is the "
            "imaging object's sampling frequency."
        ),
    )
    highpass_time: int = Field(
        default=100,
        description="Window size (frames) for temporal high-pass filter before detection.",
    )
    block_size: list[int] = Field(default=[64, 64], description="Block size [y, x] for PCA denoising.")
    npix_norm_min: float = Field(default=0.0, description="Remove ROIs with a soma smaller than this x the median.")
    npix_norm_max: float = Field(default=100.0, description="Remove ROIs with a soma larger than this x the median.")
    highpass_neuropil: int = Field(default=25, description="Sparsery: high-pass filter (pixels) removing neuropil.")
    max_ROIs: int = Field(default=5000, description="Sparsery: maximum number of ROIs to detect.")
    spatial_scale: int = Field(default=0, description="Sparsery: 0 estimates it; 1, 2, 3, 4 = 6, 12, 24, 48 pixels.")
    active_percentile: float = Field(default=0.0, description="Sparsery: if > 0, this percentile is the threshold.")
    connected: bool = Field(default=True, description="Sourcery: keep ROIs connected (False for dendrites).")
    max_iterations: int = Field(default=20, description="Sourcery: maximum number of detection iterations.")
    smooth_masks: bool = Field(default=False, description="Sourcery: smooth the ROI masks.")
    cellpose_img: str = Field(
        default="max_proj / meanImg", description="Cellpose: image to segment; also 'meanImg' or 'max_proj'."
    )
    highpass_spatial: float = Field(default=0.0, description="Cellpose: high-pass sigma, x diameter; 0 disables it.")
    flow_threshold: float = Field(default=0.4, description="Cellpose: flow error threshold.")
    cellprob_threshold: float = Field(default=0.0, description="Cellpose: cell probability threshold.")
    cellpose_params: dict[str, Any] | None = Field(default=None, description="Cellpose: extra arguments to model.eval.")
    cellpose_model: str = Field(
        default="cpsam", description="Must be 'cpsam': suite2p 1.1.0 ignores it and always loads cpsam."
    )
    chan2_threshold: float = Field(default=0.25, description="Must be 0.25: chan2 detection is not supported.")
    cellpose_chan2: bool = Field(default=False, description="Must be False: chan2 detection is not supported.")
    cellpose_params_chan2: dict[str, Any] | None = Field(default=None, description="Must be None: as cellpose_chan2.")

    model_config = SettingsConfigDict(env_file=".env", env_prefix="SUITE2P_SEGMENTATION_", case_sensitive=False)

    @field_validator("algorithm")
    @classmethod
    def lowercase_algorithm(cls, v: str) -> str:
        return v.lower()

    @classmethod
    def settings_customise_sources(
        cls, settings_cls: Any, init_settings: Any, env_settings: Any, dotenv_settings: Any, file_secret_settings: Any
    ) -> tuple[Any, ...]:
        """Skip the keys of a shared ``.env`` that are not ours; a misspelt ``SUITE2P_SEGMENTATION_*`` still raises."""
        prefix = str(cls.model_config.get("env_prefix", "")).lower()

        def own_dotenv() -> dict[str, Any]:
            return {k: v for k, v in dotenv_settings().items() if k in cls.model_fields or k.startswith(prefix)}

        return init_settings, env_settings, own_dotenv, file_secret_settings


_UNSUPPORTED = ("cellpose_model", "chan2_threshold", "cellpose_chan2", "cellpose_params_chan2")


def _read_plane_range(
    imaging: BaseImaging,
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
    for i in range(imaging.get_num_epochs()):
        ep_lo = int(epoch_offsets[i])
        ep_hi = int(epoch_offsets[i + 1])
        if ep_hi <= start or ep_lo >= stop:
            continue
        local_start = max(0, start - ep_lo)
        local_end = min(ep_hi - ep_lo, stop - ep_lo)
        frames = imaging.get_series(start_frame=local_start, end_frame=local_end, epoch_index=i)
        chunks.append(frames[:, :, :, 0])
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks, axis=0)


def _stream_bin_movie(
    imaging: BaseImaging,
    *,
    bin_size: int,
    yrange: Sequence[int],
    xrange: Sequence[int],
    badframes: NDArray[np.bool_] | None,
    nbins: int,
) -> NDArray[np.float32]:
    """Stream-bin one plane across all epochs of ``imaging``.

    Mirrors :func:`suite2p.detection.detect.bin_movie` but pulls each batch
    via :func:`BaseImagingEpoch.get_series` instead of holding the full
    movie in memory. The only large array kept is the binned ``mov``.
    """
    n_epochs = imaging.get_num_epochs()
    sizes = [int(imaging.get_num_frames(epoch_index=i)) for i in range(n_epochs)]
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
        data = _read_plane_range(imaging, epoch_offsets, tstart, tend)

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

    device = torch.device(cfg.device)
    diameter = np.array(cfg.diameter, dtype=float)

    if cfg.denoise:
        from suite2p.detection.denoise import pca_denoise  # noqa: PLC0415

        mov = pca_denoise(mov, block_size=cfg.block_size, n_comps_frac=0.5)

    meanImg = mov.mean(axis=0)
    mov = utils.temporal_high_pass_filter(mov=mov, width=cfg.highpass_time)
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
            settings={
                "img": cfg.cellpose_img,
                "params": cfg.cellpose_params,
                **cfg.model_dump(include={"highpass_spatial", "flow_threshold", "cellprob_threshold"}),
            },
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
                **cfg.model_dump(include={"highpass_neuropil", "spatial_scale", "max_ROIs", "active_percentile"}),
            )
        elif algo == "sourcery":
            _, raw_stat = sourcery.sourcery(
                mov=mov,
                sdmov=sdmov,
                diameter=diameter,
                threshold_scaling=cfg.threshold_scaling,
                **cfg.model_dump(include={"connected", "max_iterations", "smooth_masks"}),
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
            npix_norm_min=cfg.npix_norm_min,
            npix_norm_max=cfg.npix_norm_max,
            median=use_cellpose,
        )
    )


@register_segmentation_class
class Suite2pSegmentation(BaseSegmentation):
    """Suite2p ROI detection, streamed from the imaging object.

    Frames are pulled lazily via :func:`BaseImagingEpoch.get_series` and
    stream-binned in place, so the full raw movie is never materialised.
    Returns a :class:`Suite2pRois` holding every plane's suite2p ``stat``.
    """

    method_name = "suite2p"
    settings_class = Suite2pSegmentationSettings

    @classmethod
    def is_installed(cls) -> bool:
        try:
            import suite2p  # noqa: F401, PLC0415
        except ImportError:
            return False
        return True

    @classmethod
    def get_method_version(cls) -> str:
        return importlib.metadata.version("suite2p")

    @classmethod
    def run(
        cls,
        imaging: BaseImaging,
        yrange: Sequence[int],
        xrange: Sequence[int],
        badframes: NDArray[np.bool_] | None,
        settings: Suite2pSegmentationSettings,
    ) -> Suite2pRois:
        from suite2p.classification import builtin_classfile, classify  # noqa: PLC0415

        if changed := [k for k in _UNSUPPORTED if getattr(settings, k) != cls.settings_class.model_fields[k].default]:
            raise ValueError(f"Not supported, keep the default: {changed} (see each setting's description).")
        H, W, n_planes = imaging.shape

        n_epochs = imaging.get_num_epochs()
        n_frames = int(imaging.get_total_frames())
        bin_size = settings.bin_size or int(
            max(1, n_frames // settings.nbins, np.round(settings.tau * imaging.sampling_frequency))
        )

        logging.info("Suite2p detection — %d epochs", n_epochs)
        t0 = time.time()

        mov = _stream_bin_movie(
            imaging,
            bin_size=bin_size,
            yrange=yrange,
            xrange=xrange,
            badframes=badframes,
            nbins=settings.nbins,
        )

        stat = _detect_rois_from_mov(
            mov,
            cfg=settings,
            Ly=H,
            Lx=W,
            yrange=yrange,
            xrange=xrange,
        )
        logging.info("  found %d ROIs in %.1fs", len(stat), time.time() - t0)

        rois = Suite2pRois.from_stat(
            stats=stat,
            shape=(H, W, n_planes),
            sampling_frequency=imaging.sampling_frequency,
        )
        if stat:  # classify reads stat[0]
            iscell = classify(stat=np.array(stat, dtype=object), classfile=builtin_classfile)
            # same properties Suite2pRois sets when loading iscell.npy from a suite2p folder
            rois.set_property("iscell", iscell[:, 0] == 1)
            rois.set_property("iscell_probability", iscell[:, 1])

        return rois
