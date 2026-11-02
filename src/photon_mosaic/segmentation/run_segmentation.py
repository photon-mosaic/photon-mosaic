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

from photon_mosaic.core import BaseImaging
from photon_mosaic.core.baserois import BaseRois
from photon_mosaic.core.core_tools import _coerce_settings

from .basesegmentation import (
    EpochSegmentations,
    KnownSegmentationMethod,
    _resolve_badframes,
    _resolve_ranges,
    get_segmentation_class,
)

_output_subfolder = "segmentation_output"


def run_segmentation(
    imaging: BaseImaging,
    method: KnownSegmentationMethod | str = "suite2p",
    folder: str | Path = "segmentation_folder",
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

    Mirrors SpikeInterface's ``run_sorter``: the backend runs inside
    ``folder``, which also receives ``segmentation_params.json`` (validated
    settings and run arguments), ``segmentation_log.json`` (versions, run
    time, error) and, when the imaging is serializable, ``imaging.json``. The
    result is read back from the folder with :func:`read_segmentation_output`
    and the input imaging is registered to it.

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
    folder : str | Path, default: "segmentation_folder"
        Output folder. Must not exist unless ``remove_existing_folder=True``.
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
        Re-raise a backend error. If False, the error is only recorded in
        ``segmentation_log.json`` and ``None`` is returned.
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

    yranges = _resolve_ranges(yrange, n_eps, default=[0, H], label="yrange")
    xranges = _resolve_ranges(xrange, n_eps, default=[0, W], label="xrange")
    bf_per_ep = _resolve_badframes(badframes, n_eps, frame_counts)

    # One span per call of run_from_folder: (output_folder, epoch_indices, yrange, xrange, badframes)
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
        spans = [(Path(_output_subfolder), selected, yranges[0], xranges[0], merged_bf)]
    else:
        spans = [
            (Path(_output_subfolder) / f"epoch{ep_idx}", [ep_idx], yranges[i], xranges[i], bf_per_ep[i])
            for i, ep_idx in enumerate(selected)
        ]

    folder = segmentation_class.initialize_folder(folder, remove_existing_folder=remove_existing_folder)
    run_kwargs = {
        "scope": scope,
        "epoch_indices": selected,
        "yrange": yranges,
        "xrange": xranges,
        "badframes": any(b is not None for b in bf_per_ep),
    }
    segmentation_class.set_params_to_folder(folder, settings, run_kwargs)
    if run_kwargs["badframes"]:
        badframes_per_epoch: dict[str, Any] = {f"epoch{ep}": b for ep, b in zip(selected, bf_per_ep) if b is not None}
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
        for subfolder, span_epochs, span_yrange, span_xrange, span_badframes in spans:
            output_folder = folder / subfolder
            output_folder.mkdir(parents=True)
            segmentation_class.run_from_folder(
                output_folder,
                imaging,
                epoch_indices=span_epochs,
                yrange=span_yrange,
                xrange=span_xrange,
                badframes=span_badframes,
                settings=settings,
            )
    except Exception:
        log["error"] = True
        log["error_trace"] = traceback.format_exc()
        _write_log(folder, log)
        if raise_error:
            raise
        logging.warning("Segmentation method '%s' failed; see %s", method, folder / "segmentation_log.json")
        return None

    log["run_time"] = time.perf_counter() - t0
    _write_log(folder, log)

    return read_segmentation_output(folder, imaging=imaging)


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
        params = json.load(f)
    segmentation_class = get_segmentation_class(params["method"])
    run_kwargs = params["run_kwargs"]

    if run_kwargs["scope"] == "all_epochs":
        rois = segmentation_class.read_rois_from_folder(folder / _output_subfolder)
        if imaging is not None:
            rois.register_imaging(imaging)
        return rois

    results: dict[int, BaseRois] = {}
    for ep_idx in run_kwargs["epoch_indices"]:
        rois = segmentation_class.read_rois_from_folder(folder / _output_subfolder / f"epoch{ep_idx}")
        if imaging is not None:
            rois.register_imaging(imaging)
        results[ep_idx] = rois
    return EpochSegmentations(results)


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
