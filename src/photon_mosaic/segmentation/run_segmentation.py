from __future__ import annotations

import importlib.metadata
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from spikeinterface.core.core_tools import SIJsonEncoder

from photon_mosaic.core import BaseImaging
from photon_mosaic.core.baserois import BaseRois
from photon_mosaic.core.binaryrois import BinaryFolderRois
from photon_mosaic.core.core_tools import _coerce_settings
from photon_mosaic.preprocessing.registration import RegisterImaging

from .basesegmentation import (
    EpochSegmentations,
    KnownSegmentationMethod,
    _resolve_badframes,
    _resolve_ranges,
    get_segmentation_class,
)

_output_subfolder = "segmentation_output"


def _intersect_plane_ranges(plane_ranges: Sequence[Sequence[int]]) -> list[int]:
    """Collapse per-plane ``(lo, hi)`` ranges into the one range valid on every plane."""
    return [max(int(r[0]) for r in plane_ranges), min(int(r[1]) for r in plane_ranges)]


def run_segmentation(
    imaging: BaseImaging,
    method: KnownSegmentationMethod | str = "suite2p",
    folder: str | Path | None = "segmentation_folder",
    *,
    settings: Any = None,
    scope: Literal["all_epochs", "per_epoch"] = "all_epochs",
    epoch_indices: Sequence[int] | None = None,
    yrange: Sequence[int] | Sequence[Sequence[int]] | None = None,
    xrange: Sequence[int] | Sequence[Sequence[int]] | None = None,
    badframes: NDArray[np.bool_] | Sequence[NDArray[np.bool_] | None] | None = None,
    remove_existing_folder: bool = False,
    raise_error: bool = True,
    **params: Any,
) -> BaseRois | EpochSegmentations | None:
    """Detect ROIs on ``imaging`` with the requested segmentation ``method``.

    Mirrors SpikeInterface's ``run_sorter``: the backend returns the ROIs and
    the input imaging is registered to them. If ``folder`` is given, it
    receives ``segmentation_params.json`` (validated settings and run
    arguments), ``segmentation_log.json`` (versions, run time, error), the
    imaging provenance (``imaging.json``, else ``imaging.pickle``) and the ROIs
    saved as binary under ``segmentation_output/``, which
    :func:`read_segmentation_output` loads back.

    ``scope='all_epochs'`` (default) concatenates the selected epochs and
    returns a single ROI set; ``scope='per_epoch'`` runs the backend once per
    epoch and returns an :class:`EpochSegmentations`.

    Parameters
    ----------
    imaging : BaseImaging
        Motion-corrected imaging object.
    method : str, default: "suite2p"
        Segmentation backend; see
        :func:`photon_mosaic.segmentation.get_available_segmentation_methods`.
    folder : str | Path | None, default: "segmentation_folder"
        Output folder. Must not exist unless ``remove_existing_folder=True``.
        ``None`` runs in memory only and writes nothing.
    settings : BaseSettings | dict | None, optional
        Backend settings. Dicts and ``None`` are coerced into the backend's
        ``settings_class``.
    scope : {"all_epochs", "per_epoch"}, optional
        Run once over the concatenated epochs, or once per epoch.
    epoch_indices : sequence of int | None, optional
        Epochs to include (in order). Defaults to all epochs of ``imaging``.
    yrange, xrange : sequence of int | sequence of sequence of int | None, optional
        Valid pixel range ``[lo, hi]``. May be a single 1D pair (broadcast to
        every epoch) or a 2D ``(n_epochs, 2)`` sequence with one pair per
        epoch. ``None`` defaults to the full image extent. With
        ``scope='all_epochs'`` per-epoch ranges must be uniform.
    badframes : NDArray[bool] | sequence of NDArray[bool] | None, optional
        Frames to exclude. May be a single flat ``(total_frames,)`` boolean
        array or a per-epoch sequence (each shaped ``(n_frames_in_epoch,)``).
    remove_existing_folder : bool, default: False
        Delete ``folder`` first if it already exists.
    raise_error : bool, default: True
        Re-raise a backend error. If False, the error is only logged (and
        recorded in ``segmentation_log.json`` when ``folder`` is given) and
        ``None`` is returned.
    **params : Any
        Backend settings, overriding ``settings``.

    Returns
    -------
    BaseRois | EpochSegmentations | None
        Single ROI set when ``scope='all_epochs'``; one ROI set per epoch
        otherwise; ``None`` if the backend failed and ``raise_error=False``.
    """
    segmentation_class = get_segmentation_class(method)
    if not segmentation_class.is_installed():
        raise ImportError(f"Segmentation method '{method}' is not installed.")

    settings_class = segmentation_class.settings_class
    settings = _coerce_settings(settings, settings_class)
    if params:
        if settings_class is None:
            raise TypeError(f"Segmentation method '{method}' takes no parameters, got {sorted(params)}.")
        # settings classes may ignore extra fields, so catch typos here rather than drop them silently
        unknown = sorted(set(params) - set(settings_class.model_fields))
        if unknown:
            raise ValueError(
                f"Unknown parameters {unknown} for segmentation method '{method}'. "
                f"Valid parameters: {sorted(settings_class.model_fields)}."
            )
        settings = settings_class(**{**settings.model_dump(), **params})

    if scope not in ("all_epochs", "per_epoch"):
        raise ValueError(f"scope must be 'all_epochs' or 'per_epoch', got {scope!r}")

    H, W, _ = imaging.shape
    selected = list(range(imaging.get_num_epochs())) if epoch_indices is None else list(epoch_indices)
    n_eps = len(selected)
    frame_counts = [int(imaging.get_num_samples(segment_index=i)) for i in selected]

    # Explicit arguments win; otherwise fall back to what registration computed for the selected epochs.
    if isinstance(imaging, RegisterImaging):
        motion_y, motion_x = imaging.motion.yranges, imaging.motion.xranges
        motion_bf = imaging.motion.corrected_badframes
        if yrange is None and motion_y is not None:
            yrange = [_intersect_plane_ranges(motion_y[i]) for i in selected]
        if xrange is None and motion_x is not None:
            xrange = [_intersect_plane_ranges(motion_x[i]) for i in selected]
        if badframes is None and motion_bf is not None:
            badframes = [motion_bf[i] for i in selected]

    yranges = _resolve_ranges(yrange, n_eps, default=[0, H], label="yrange")
    xranges = _resolve_ranges(xrange, n_eps, default=[0, W], label="xrange")
    bf_per_ep = _resolve_badframes(badframes, n_eps, frame_counts)

    # One span per call of run: (epoch_indices, yrange, xrange, badframes)
    if scope == "all_epochs":
        if not all(yr == yranges[0] for yr in yranges):
            raise ValueError(
                "yrange differs across selected epochs; use scope='per_epoch' or pass a single uniform range."
            )
        if not all(xr == xranges[0] for xr in xranges):
            raise ValueError(
                "xrange differs across selected epochs; use scope='per_epoch' or pass a single uniform range."
            )

        if any(b is not None for b in bf_per_ep):
            merged_bf: NDArray[np.bool_] | None = np.concatenate(
                [b if b is not None else np.zeros(n, dtype=bool) for b, n in zip(bf_per_ep, frame_counts)],
                axis=0,
            )
        else:
            merged_bf = None
        spans = [(selected, yranges[0], xranges[0], merged_bf)]
    else:
        spans = [([ep_idx], yranges[i], xranges[i], bf_per_ep[i]) for i, ep_idx in enumerate(selected)]

    run_kwargs = {
        "scope": scope,
        "epoch_indices": selected,
        "yrange": yranges,
        "xrange": xranges,
        "badframes": any(b is not None for b in bf_per_ep),
    }

    if folder is not None:
        folder = segmentation_class.initialize_folder(folder, remove_existing_folder=remove_existing_folder)
        segmentation_class.set_params_to_folder(folder, settings, run_kwargs)
        if run_kwargs["badframes"]:
            badframes_per_epoch: dict[str, Any] = {
                f"epoch{ep}": b for ep, b in zip(selected, bf_per_ep) if b is not None
            }
            np.savez(folder / "badframes.npz", **badframes_per_epoch)
        _dump_imaging_provenance(imaging, folder)

    log: dict[str, Any] = {
        "method": method,
        "method_version": segmentation_class.get_method_version(),
        "photon_mosaic_version": importlib.metadata.version("photon-mosaic"),
        "datetime": datetime.now().isoformat(),
        "run_time": None,
        "error": False,
        "error_trace": None,
    }
    t0 = time.perf_counter()
    try:
        span_rois = [
            segmentation_class.run(
                imaging,
                epoch_indices=span_epochs,
                yrange=span_yrange,
                xrange=span_xrange,
                badframes=span_badframes,
                settings=settings,
            )
            for span_epochs, span_yrange, span_xrange, span_badframes in spans
        ]
    except Exception:
        log["error"] = True
        log["error_trace"] = traceback.format_exc()
        if folder is not None:
            _write_log(folder, log)
        if raise_error:
            raise
        logging.warning("Segmentation method '%s' failed:\n%s", method, log["error_trace"])
        return None

    log["run_time"] = time.perf_counter() - t0
    # annotations are saved with plain json.dumps, so store only json-native values
    imaging_dict = json.loads(json.dumps(imaging.to_dict(recursive=True), cls=SIJsonEncoder))
    params_dict = settings.model_dump(mode="json") if settings is not None else None
    for rois in span_rois:
        rois.register_imaging(imaging)
        rois.set_segmentation_info(imaging_dict, params_dict=params_dict, log_dict=log)
    if folder is not None:
        for subfolder, rois in zip(_rois_subfolders(scope, selected), span_rois):
            rois.save(format="binary", folder=folder / subfolder)
        _write_log(folder, log)

    if scope == "all_epochs":
        return span_rois[0]
    return EpochSegmentations(dict(zip(selected, span_rois)))


def read_segmentation_output(
    folder: str | Path,
    imaging: BaseImaging | None = None,
) -> BaseRois | EpochSegmentations:
    """Load the result of a :func:`run_segmentation` call from its ``folder``.

    Parameters
    ----------
    folder : str | Path
        Folder previously passed to :func:`run_segmentation`.
    imaging : BaseImaging | None, optional
        Imaging to register to the loaded ROIs.

    Returns
    -------
    BaseRois | EpochSegmentations
        Same shape of result as :func:`run_segmentation` returned.
    """
    folder = Path(folder)
    with open(folder / "segmentation_params.json") as f:
        run_kwargs = json.load(f)["run_kwargs"]
    scope, epoch_indices = run_kwargs["scope"], run_kwargs["epoch_indices"]

    span_rois = [BinaryFolderRois(folder / subfolder) for subfolder in _rois_subfolders(scope, epoch_indices)]
    if imaging is not None:
        for rois in span_rois:
            rois.register_imaging(imaging)

    if scope == "all_epochs":
        return span_rois[0]
    return EpochSegmentations(dict(zip(epoch_indices, span_rois)))


def _rois_subfolders(scope: str, epoch_indices: Sequence[int]) -> list[Path]:
    """Where :func:`run_segmentation` saves each span's ROIs, relative to its folder."""
    if scope == "all_epochs":
        return [Path(_output_subfolder)]
    return [Path(_output_subfolder) / f"epoch{ep_idx}" for ep_idx in epoch_indices]


def _dump_imaging_provenance(imaging: BaseImaging, folder: Path) -> None:
    """Write ``imaging.json``, else ``imaging.pickle``; provenance never fails the run.

    ``check_serializability`` only inspects nested extractors, so a json dump
    can still fail on some other kwarg; fall back to pickle in that case.
    """
    if imaging.check_serializability("json"):
        try:
            imaging.dump_to_json(folder / "imaging.json")
            return
        except (TypeError, ValueError) as exc:
            logging.warning("Could not write imaging.json (%s); trying pickle instead.", exc)
    if imaging.check_serializability("pickle"):
        imaging.dump_to_pickle(folder / "imaging.pickle")
    else:
        logging.warning("Imaging is not serializable; no imaging provenance written to %s", folder)


def _write_log(folder: Path, log: dict[str, Any]) -> None:
    with open(folder / "segmentation_log.json", "w") as f:
        json.dump(log, f, indent=4)
