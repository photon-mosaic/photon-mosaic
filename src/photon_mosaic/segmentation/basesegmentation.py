from __future__ import annotations

import importlib
import json
import shutil
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic_settings import BaseSettings

from photon_mosaic.core import BaseImaging
from photon_mosaic.core.baserois import BaseRois

# Editor autocomplete/typo-checking for the built-in methods. Not exhaustive:
# any string registered via ``register_segmentation_class`` (including
# third-party ones) is still accepted at runtime.
KnownSegmentationMethod = Literal["suite2p"]

# Method name -> module that defines (and registers) the backend's
# BaseSegmentation subclass. Only module *paths* live here, so this module
# imports no backend: the module is imported lazily the first time the method
# is requested.
_builtin_segmentation_modules: dict[str, str] = {
    "suite2p": "photon_mosaic.segmentation.suite2p_segmentation",
}

# Method name -> BaseSegmentation subclass, filled in by ``register_segmentation_class``.
_registered_segmentation_classes: dict[str, type["BaseSegmentation"]] = {}


def register_segmentation_class(segmentation_class: type["BaseSegmentation"]) -> type["BaseSegmentation"]:
    """Register a :class:`BaseSegmentation` subclass under its ``method_name``.

    Backend modules call this at import time so that
    ``run_segmentation(..., method=...)`` can find them. Third-party backends
    can call it too, which is what makes the set of methods extensible.

    Parameters
    ----------
    segmentation_class : type[BaseSegmentation]
        Subclass defining ``method_name`` (and usually ``settings_class``).

    Returns
    -------
    type[BaseSegmentation]
        The class, unchanged, so this can be used as a decorator.
    """

    name = segmentation_class.method_name
    if not name:
        raise ValueError(f"{segmentation_class.__name__} must define a non-empty 'method_name' to be registered.")
    _registered_segmentation_classes[name] = segmentation_class
    return segmentation_class


def get_segmentation_class(method: str) -> type["BaseSegmentation"]:
    """Return the :class:`BaseSegmentation` subclass implementing ``method``.

    Built-in backends are imported lazily on first use, so an optional
    dependency is only required when its method is actually requested.
    """

    if method in _registered_segmentation_classes:
        return _registered_segmentation_classes[method]

    module_path = _builtin_segmentation_modules.get(method)
    if module_path is None:
        known = sorted(set(_registered_segmentation_classes) | set(_builtin_segmentation_modules))
        raise ValueError(f"Unknown segmentation method '{method}'. Available methods: {known}.")

    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Segmentation method '{method}' requires optional dependencies that are not installed "
            f"(failed to import '{module_path}': {exc})."
        ) from exc

    if method not in _registered_segmentation_classes:
        raise ImportError(
            f"Module '{module_path}' was imported but did not register a segmentation class for method '{method}'."
        )
    return _registered_segmentation_classes[method]


def get_available_segmentation_methods() -> list[str]:
    """Return the names of all built-in and registered segmentation methods."""
    return sorted(set(_registered_segmentation_classes) | set(_builtin_segmentation_modules))


class BaseSegmentation:
    """Base class for ROI segmentation backends.

    Modelled on SpikeInterface's ``BaseSorter``: subclasses are never
    instantiated, they only provide classmethods that
    :func:`photon_mosaic.segmentation.run_segmentation` calls in order:

    1. :meth:`initialize_folder` / :meth:`set_params_to_folder` (provided here)
    2. :meth:`run` -- detect ROIs on one contiguous span of epochs and return
       them as a :class:`BaseRois`

    Subclasses declare ``method_name`` (the string users pass as ``method``)
    and ``settings_class`` (their pydantic settings model), and register
    themselves with :func:`register_segmentation_class`.
    """

    method_name: str | None = None
    settings_class: type[BaseSettings] | None = None

    @classmethod
    def is_installed(cls) -> bool:
        """Return whether the backend's dependencies are importable."""
        raise NotImplementedError

    @classmethod
    def get_method_version(cls) -> str:
        """Return the version of the underlying segmentation library."""
        raise NotImplementedError

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        """Return the default value of every setting of ``settings_class``."""
        if cls.settings_class is None:
            return {}
        return cls.settings_class().model_dump()

    @classmethod
    def params_description(cls) -> dict[str, str | None]:
        """Return the description of every setting of ``settings_class``."""
        if cls.settings_class is None:
            return {}
        return {name: field.description for name, field in cls.settings_class.model_fields.items()}

    @classmethod
    def initialize_folder(cls, folder: str | Path, remove_existing_folder: bool = False) -> Path:
        """Create the output ``folder``, refusing to overwrite unless asked."""
        folder = Path(folder)
        if folder.exists():
            if not remove_existing_folder:
                raise FileExistsError(f"Folder {folder} already exists; pass remove_existing_folder=True to overwrite.")
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        return folder

    @classmethod
    def set_params_to_folder(cls, folder: str | Path, settings: Any, run_kwargs: dict[str, Any]) -> None:
        """Write the validated settings and the run arguments to ``segmentation_params.json``."""
        params = {
            "method": cls.method_name,
            "settings": settings.model_dump(mode="json") if settings is not None else None,
            "run_kwargs": run_kwargs,
        }
        with open(Path(folder) / "segmentation_params.json", "w") as f:
            json.dump(params, f, indent=4)

    @classmethod
    def run(
        cls,
        imaging: BaseImaging,
        yrange: Sequence[int],
        xrange: Sequence[int],
        badframes: NDArray[np.bool_] | None,
        settings: Any,
    ) -> BaseRois:
        """Detect ROIs on the concatenation of all epochs of ``imaging`` and return them.

        Parameters
        ----------
        imaging : BaseImaging
            Motion-corrected imaging object.
        yrange, xrange : sequence of int
            Valid pixel range ``[lo, hi]``.
        badframes : NDArray[bool] | None
            Frames to exclude, shaped ``(total_frames_in_epochs,)``.
        settings : BaseSettings
            Validated ``settings_class`` instance.

        Returns
        -------
        rois : BaseRois
            Detected ROIs for the concatenated epochs.
        """
        raise NotImplementedError


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
            raise ValueError(f"badframes length {value.shape[0]} does not match total frame count {total}")
        out: list[NDArray[np.bool_] | None] = []
        offset = 0
        for n in frame_counts:
            out.append(np.asarray(value[offset : offset + n], dtype=bool))
            offset += n
        return out

    if len(value) != n_epochs:
        raise ValueError(f"badframes list length {len(value)} does not match number of epochs {n_epochs}")
    out = []
    for i, (b, n) in enumerate(zip(value, frame_counts)):
        if b is None:
            out.append(None)
            continue
        b_arr = np.asarray(b, dtype=bool)
        if b_arr.shape[0] != n:
            raise ValueError(f"badframes[{i}] length {b_arr.shape[0]} does not match epoch frame count {n}")
        out.append(b_arr)
    return out
