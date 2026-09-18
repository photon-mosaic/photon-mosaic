from __future__ import annotations

import importlib
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from photon_mosaic.core import BaseImaging

# Method name -> module that defines (and registers) the backend's Motion
# subclass. Only module *paths* live here, so core imports no backend: the
# module is imported lazily the first time the method is requested.
_builtin_motion_modules: dict[str, str] = {
    "suite2p": "photon_mosaic.preprocessing.suite2p_registration",
    "jnormcorre": "photon_mosaic.preprocessing.jnormcorre_registration",
}

# Method name -> Motion subclass, filled in by ``register_motion_class``.
_registered_motion_classes: dict[str, type["Motion"]] = {}


def register_motion_class(motion_class: type["Motion"]) -> type["Motion"]:
    """Register a :class:`Motion` subclass under its ``method_name``.

    Backend modules call this at import time so that
    ``Motion.compute(..., method=...)`` can find them. Third-party backends can
    call it too, which is what makes the set of methods extensible.

    Parameters
    ----------
    motion_class : type[Motion]
        Subclass defining ``method_name`` (and usually ``settings_class`` and
        ``epoch_class``).

    Returns
    -------
    type[Motion]
        The class, unchanged, so this can be used as a decorator.
    """

    name = motion_class.method_name
    if not name:
        raise ValueError(f"{motion_class.__name__} must define a non-empty 'method_name' to be registered.")
    _registered_motion_classes[name] = motion_class
    return motion_class


def _get_motion_class(method: str) -> type["Motion"]:
    """Return the :class:`Motion` subclass implementing ``method``.

    Built-in backends are imported lazily on first use, so an optional
    dependency is only required when its method is actually requested.
    """

    if method in _registered_motion_classes:
        return _registered_motion_classes[method]

    module_path = _builtin_motion_modules.get(method)
    if module_path is None:
        known = sorted(set(_registered_motion_classes) | set(_builtin_motion_modules))
        raise ValueError(f"Unknown motion correction method '{method}'. Available methods: {known}.")

    try:
        importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Motion correction method '{method}' requires optional dependencies that are not installed "
            f"(failed to import '{module_path}': {exc})."
        ) from exc

    if method not in _registered_motion_classes:
        raise ImportError(
            f"Module '{module_path}' was imported but did not register a Motion class for method '{method}'."
        )
    return _registered_motion_classes[method]


def coerce_settings(settings: Any, settings_class: type | None) -> Any:
    """Turn ``None`` / a dict / a settings instance into ``settings_class``.

    Returns ``settings`` unchanged when ``settings_class`` is ``None`` (a
    backend without a settings model).
    """

    if settings_class is None:
        return settings
    if settings is None:
        return settings_class()
    if isinstance(settings, settings_class):
        return settings
    if isinstance(settings, dict):
        return settings_class(**settings)
    raise TypeError(f"'settings' must be None, a dict or a {settings_class.__name__}, got {type(settings).__name__}.")


class Motion:
    """Algorithm-agnostic container for motion correction artifacts.

    Holds outputs that any motion correction backend (Suite2P, CaImAn, ...) is
    expected to produce. Backend-specific fields (e.g. Suite2P ``ops`` or
    block-wise non-rigid offsets) belong on dedicated subclasses such as
    ``Suite2PMotion``.

    Subclasses declare three class attributes so that they can be reached
    through :meth:`compute` and applied by
    :class:`photon_mosaic.preprocessing.registration.RegisterImaging`:
    ``method_name`` (the string users pass as ``method``), ``settings_class``
    (their pydantic settings model) and ``epoch_class`` (the
    ``BasePreprocessorEpoch`` that applies the stored motion).
    """

    method_name: str | None = None
    settings_class: type | None = None
    epoch_class: type | None = None

    def __init__(
        self,
        imaging: BaseImaging,
        displacements: Sequence[NDArray[np.floating[Any]]],
        reference: Any = None,
        yranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        xranges: Sequence[Sequence[tuple[int, int]]] | None = None,
        corrected_badframes: Sequence[NDArray[np.bool_]] | None = None,
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
            Per-epoch boolean mask of frames to exclude from downstream
            analysis, shaped ``[epoch] -> (n_frames,)``. Bad frames are treated
            as a property of the time axis (a corrupted volume frame is bad on
            all planes), so a single mask per epoch is stored. How the mask is
            populated depends on the backend.
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

    @classmethod
    def compute(
        cls,
        imaging: BaseImaging,
        method: str | None = None,
        settings: Any = None,
        badframes: NDArray | None = None,
        **params: Any,
    ) -> "Motion":
        """Estimate motion for ``imaging`` with the requested backend.

        Called on :class:`Motion` itself, ``method`` selects the backend
        (default ``"suite2p"``); called on a subclass, that subclass is used and
        ``method``, if given, must match its ``method_name``.

        Parameters
        ----------
        imaging : BaseImaging
            Imaging object containing one or more epochs/planes to register.
        method : str | None, optional
            Registration backend, e.g. ``"suite2p"``.
        settings : BaseSettings | dict | None, optional
            Backend settings. Dicts and ``None`` are coerced into the backend's
            ``settings_class``.
        badframes : NDArray | None, optional
            Boolean array of shape ``(n_frames,)`` marking frames to exclude
            from reference image computation.
        **params : Any
            Extra backend options, overriding ``settings``.

        Returns
        -------
        Motion
            A backend-specific :class:`Motion` subclass instance.
        """

        if method is not None and not isinstance(method, str):
            raise TypeError(
                f"'method' must be a string naming a registration backend, got {type(method).__name__}. "
                "Settings must be passed as a keyword argument: compute_motion(imaging, settings=...)."
            )

        if cls is Motion:
            target = _get_motion_class(method or "suite2p")
        else:
            if method is not None and method != cls.method_name:
                raise ValueError(
                    f"{cls.__name__} implements method '{cls.method_name}', but method='{method}' was requested. "
                    "Call Motion.compute(...) to dispatch on 'method'."
                )
            target = cls

        resolved_settings = coerce_settings(settings, target.settings_class)
        return target._compute(imaging, settings=resolved_settings, badframes=badframes, **params)

    @classmethod
    def _compute(
        cls,
        imaging: BaseImaging,
        *,
        settings: Any = None,
        badframes: NDArray | None = None,
        **params: Any,
    ) -> "Motion":
        """Backend-specific estimation. Implemented by each subclass."""

        raise NotImplementedError(f"{cls.__name__} does not implement motion estimation.")

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


compute_motion = Motion.compute
