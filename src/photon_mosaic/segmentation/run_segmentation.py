from __future__ import annotations

import importlib.metadata
import json
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from spikeinterface.core.core_tools import SIJsonEncoder

from photon_mosaic.core import BaseImaging, Motion, SelectEpochImaging
from photon_mosaic.core.baserois import BaseRois
from photon_mosaic.core.binaryrois import BinaryFolderRois
from photon_mosaic.core.core_tools import _coerce_settings
from photon_mosaic.preprocessing.registration import RegisterImaging

from .basesegmentation import (
    BaseSegmentation,
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
    imaging_or_list_or_dict: BaseImaging | list[BaseImaging] | dict[Any, BaseImaging],
    method: KnownSegmentationMethod | str = "suite2p",
    folder: str | Path | None = "segmentation_folder",
    *,
    settings: Any = None,
    yrange: Sequence[int] | Sequence[Sequence[int]] | None = None,
    xrange: Sequence[int] | Sequence[Sequence[int]] | None = None,
    badframes: NDArray[np.bool_] | Sequence[NDArray[np.bool_] | None] | None = None,
    remove_existing_folder: bool = False,
    raise_error: bool = True,
    **params: Any,
) -> BaseRois | list[BaseRois | None] | dict[Any, BaseRois | None] | None:
    """Detect ROIs on one imaging, or on each of a list or dict of imagings.

    Each imaging is segmented once over all of its epochs, concatenated. To
    get one ROI set per epoch, split the imaging first with
    :func:`photon_mosaic.core.split_imaging_by_epochs`.

    Mirrors SpikeInterface's ``run_sorter``: the backend returns the ROIs and
    the input imaging is registered to them. If ``folder`` is given, it
    receives ``segmentation_params.json`` (validated settings and run
    arguments), ``segmentation_log.json`` (versions, run time, error), the
    imaging provenance (``imaging.json``, else ``imaging.pickle``) and the ROIs
    saved as binary under ``segmentation_output/``, which
    :func:`read_segmentation_output` loads back. For a list or dict, each
    imaging gets that layout in its own ``folder/<key>`` subfolder, and
    ``segmentation_inputs.json`` records the input type and keys.

    Parameters
    ----------
    imaging_or_list_or_dict : BaseImaging | list[BaseImaging] | dict[Any, BaseImaging]
        Motion-corrected imaging object, list of imaging objects, or dict of
        imaging objects. With a ``folder``, dict keys must be distinct str or int.
    method : str, default: "suite2p"
        Segmentation backend; see
        :func:`photon_mosaic.segmentation.get_available_segmentation_methods`.
    folder : str | Path | None, default: "segmentation_folder"
        Output folder. Must not exist unless ``remove_existing_folder=True``.
        ``None`` runs in memory only and writes nothing.
    settings : BaseSettings | dict | None, optional
        Backend settings. Dicts and ``None`` are coerced into the backend's
        ``settings_class``.
    yrange, xrange : sequence of int | sequence of sequence of int | None, optional
        Valid pixel range ``[lo, hi]``, applied to every imaging. May be a
        single 1D pair (broadcast to every epoch) or a 2D ``(n_epochs, 2)``
        sequence with one pair per epoch, which must then be uniform. ``None``
        uses the registration's ranges for a registered imaging, else the full
        image extent.
    badframes : NDArray[bool] | sequence of NDArray[bool] | None, optional
        Frames to exclude, applied to every imaging. May be a single flat
        ``(total_frames,)`` boolean array or a per-epoch sequence (each shaped
        ``(n_frames_in_epoch,)``). ``None`` uses the registration's bad frames
        for a registered imaging.
    remove_existing_folder : bool, default: False
        Delete ``folder`` first if it already exists.
    raise_error : bool, default: True
        Re-raise a backend error. If False, the error is only logged (and
        recorded in ``segmentation_log.json`` when ``folder`` is given) and
        ``None`` is returned for that imaging.
    **params : Any
        Backend settings, overriding ``settings``.

    Returns
    -------
    BaseRois | list[BaseRois | None] | dict[Any, BaseRois | None] | None
        One ROI set per imaging, in the same form as the input.
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

    keys: list[Any]
    if isinstance(imaging_or_list_or_dict, BaseImaging):
        input_type, keys, imagings = "single", [None], [imaging_or_list_or_dict]
    elif isinstance(imaging_or_list_or_dict, list):
        input_type, keys, imagings = "list", list(range(len(imaging_or_list_or_dict))), list(imaging_or_list_or_dict)
    elif isinstance(imaging_or_list_or_dict, dict):
        input_type, keys = "dict", list(imaging_or_list_or_dict)
        imagings = list(imaging_or_list_or_dict.values())
    else:
        raise TypeError("imaging_or_list_or_dict must be a BaseImaging, list of BaseImaging, or dict of BaseImaging.")

    if folder is not None:
        if input_type == "dict":
            # keys name the subfolders and are stored in json, so they must survive both
            if not all(isinstance(k, (str, int)) for k in keys) or len({str(k) for k in keys}) != len(keys):
                raise ValueError(f"dict keys must be distinct str or int to be saved to a folder, got {keys}")
        folder = segmentation_class.initialize_folder(folder, remove_existing_folder=remove_existing_folder)
        if input_type != "single":
            inputs = {"type": input_type, "keys": keys}
            (folder / "segmentation_inputs.json").write_text(json.dumps(inputs, indent=4))

    rois_results: list[BaseRois | None] = []
    for key, imaging in zip(keys, imagings):
        output_folder = None
        if folder is not None:
            output_folder = folder if input_type == "single" else folder / str(key)
            output_folder.mkdir(exist_ok=True)
        rois_results.append(
            _segment_one(
                segmentation_class,
                method,
                imaging,
                settings,
                yrange=yrange,
                xrange=xrange,
                badframes=badframes,
                output_folder=output_folder,
                raise_error=raise_error,
            )
        )

    if input_type == "single":
        return rois_results[0]
    if input_type == "list":
        return rois_results
    return dict(zip(keys, rois_results))


def _segment_one(
    segmentation_class: type[BaseSegmentation],
    method: str,
    imaging: BaseImaging,
    settings: Any,
    *,
    yrange: Sequence[int] | Sequence[Sequence[int]] | None,
    xrange: Sequence[int] | Sequence[Sequence[int]] | None,
    badframes: NDArray[np.bool_] | Sequence[NDArray[np.bool_] | None] | None,
    output_folder: Path | None,
    raise_error: bool,
) -> BaseRois | None:
    """Run the backend once over all epochs of ``imaging``; write to ``output_folder`` if given."""
    H, W, _ = imaging.shape
    n_eps = imaging.get_num_epochs()
    frame_counts = [int(imaging.get_num_frames(epoch_index=i)) for i in range(n_eps)]

    # Explicit arguments win; otherwise fall back to what registration computed for these epochs.
    registered = _registered_motion(imaging)
    if registered is not None:
        motion, motion_epochs = registered
        if yrange is None and motion.yranges is not None:
            yrange = [_intersect_plane_ranges(motion.yranges[i]) for i in motion_epochs]
        if xrange is None and motion.xranges is not None:
            xrange = [_intersect_plane_ranges(motion.xranges[i]) for i in motion_epochs]
        if badframes is None and motion.corrected_badframes is not None:
            badframes = [motion.corrected_badframes[i] for i in motion_epochs]

    yranges = _resolve_ranges(yrange, n_eps, default=[0, H], label="yrange")
    xranges = _resolve_ranges(xrange, n_eps, default=[0, W], label="xrange")
    bf_per_ep = _resolve_badframes(badframes, n_eps, frame_counts)

    if not all(yr == yranges[0] for yr in yranges):
        raise ValueError("yrange differs across epochs; split the imaging by epoch or pass a single uniform range.")
    if not all(xr == xranges[0] for xr in xranges):
        raise ValueError("xrange differs across epochs; split the imaging by epoch or pass a single uniform range.")

    if any(b is not None for b in bf_per_ep):
        merged_bf: NDArray[np.bool_] | None = np.concatenate(
            [b if b is not None else np.zeros(n, dtype=bool) for b, n in zip(bf_per_ep, frame_counts)],
            axis=0,
        )
    else:
        merged_bf = None

    if output_folder is not None:
        run_kwargs = {"yrange": yranges, "xrange": xranges, "badframes": merged_bf is not None}
        segmentation_class.set_params_to_folder(output_folder, settings, run_kwargs)
        if merged_bf is not None:
            badframes_per_epoch: dict[str, Any] = {f"epoch{ep}": b for ep, b in enumerate(bf_per_ep) if b is not None}
            np.savez(output_folder / "badframes.npz", **badframes_per_epoch)
        _dump_imaging_provenance(imaging, output_folder)

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
        rois = segmentation_class.run(
            imaging, yrange=yranges[0], xrange=xranges[0], badframes=merged_bf, settings=settings
        )
    except Exception:
        log["error"] = True
        log["error_trace"] = traceback.format_exc()
        if output_folder is not None:
            _write_log(output_folder, log)
        if raise_error:
            raise
        logging.warning("Segmentation method '%s' failed:\n%s", method, log["error_trace"])
        return None

    log["run_time"] = time.perf_counter() - t0
    # annotations are saved with plain json.dumps, so store only json-native values
    imaging_dict = json.loads(json.dumps(imaging.to_dict(recursive=True), cls=SIJsonEncoder))
    params_dict = settings.model_dump(mode="json") if settings is not None else None
    rois.register_imaging(imaging)
    rois.set_segmentation_info(imaging_dict, params_dict=params_dict, log_dict=log)
    if output_folder is not None:
        rois.save(format="binary", folder=output_folder / _output_subfolder)
        _write_log(output_folder, log)
    return rois


def _registered_motion(imaging: BaseImaging) -> tuple[Motion, list[int]] | None:
    """Return ``(motion, motion epoch of each imaging epoch)`` if ``imaging`` is registered.

    Covers a :class:`RegisterImaging` and one split from it with
    :func:`photon_mosaic.core.split_epochs`.
    """
    if isinstance(imaging, RegisterImaging):
        return imaging.motion, list(range(imaging.get_num_epochs()))
    if isinstance(imaging, SelectEpochImaging) and isinstance(imaging._parent, RegisterImaging):
        selected = imaging._selected_epoch_indices
        return imaging._parent.motion, [selected[i] for i in range(imaging.get_num_epochs())]
    return None


def read_segmentation_output(
    folder: str | Path,
    imaging: BaseImaging | list[BaseImaging] | dict[Any, BaseImaging] | None = None,
) -> BaseRois | list[BaseRois | None] | dict[Any, BaseRois | None]:
    """Load the result of a :func:`run_segmentation` call from its ``folder``.

    Parameters
    ----------
    folder : str | Path
        Folder previously passed to :func:`run_segmentation`.
    imaging : BaseImaging | list[BaseImaging] | dict[Any, BaseImaging] | None, optional
        Imaging to register to the loaded ROIs, in the same form (single, list
        or dict) that was passed to :func:`run_segmentation`.

    Returns
    -------
    BaseRois | list[BaseRois | None] | dict[Any, BaseRois | None]
        Same form as :func:`run_segmentation` returned; ``None`` for an
        imaging whose segmentation failed.
    """
    folder = Path(folder)
    inputs_file = folder / "segmentation_inputs.json"
    if not inputs_file.is_file():
        rois = BinaryFolderRois(folder / _output_subfolder)
        if isinstance(imaging, BaseImaging):
            rois.register_imaging(imaging)
        return rois

    inputs = json.loads(inputs_file.read_text())
    keys = inputs["keys"]
    if imaging is None:
        imagings: list[BaseImaging | None] = [None] * len(keys)
    elif isinstance(imaging, dict):
        imagings = [imaging[k] for k in keys]
    else:
        imagings = list(imaging)

    rois_results: list[BaseRois | None] = []
    for key, im in zip(keys, imagings):
        rois_folder = folder / str(key) / _output_subfolder
        loaded = BinaryFolderRois(rois_folder) if rois_folder.is_dir() else None
        if loaded is not None and im is not None:
            loaded.register_imaging(im)
        rois_results.append(loaded)

    if inputs["type"] == "list":
        return rois_results
    return dict(zip(keys, rois_results))


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
