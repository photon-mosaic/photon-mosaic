from __future__ import annotations

import importlib
from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from photon_mosaic.core import BaseImaging
from photon_mosaic.core.core_tools import _coerce_settings

# Editor autocomplete/typo-checking for the built-in methods. Not exhaustive:
# any string registered via ``register_motion_class``/``register_registration_class``
# (including third-party ones) is still accepted at runtime.
KnownMotionMethod = Literal["suite2p", "normcorre"]

# Method name -> module that defines (and registers) the backend's Motion
# subclass. Only module *paths* live here, so core imports no backend: the
# module is imported lazily the first time the method is requested.
_builtin_motion_modules: dict[str, str] = {
    "suite2p": "photon_mosaic.preprocessing.suite2p_registration",
    "normcorre": "photon_mosaic.preprocessing.normcorre_registration",
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
        Subclass defining ``method_name`` (and usually ``settings_class``).

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


# Method name -> registration/epoch class, filled in by
# ``register_registration_class``. Kept independent of
# ``_registered_motion_classes``: a backend may register a ``Motion``
# subclass without a registration class (e.g. it only produces diagnostics),
# or register a registration class for motion produced elsewhere.
_registered_registration_classes: dict[str, type] = {}


def register_registration_class(method: str, registration_class: type) -> type:
    """Register the class that applies motion computed by ``method``.

    Backend modules call this at import time, independently of
    :func:`register_motion_class`, so that
    :class:`photon_mosaic.preprocessing.registration.RegisterImaging` can find
    the class that knows how to apply their motion.

    Returns
    -------
    type
        ``registration_class``, unchanged, so this can be used as a decorator.
    """

    if not method:
        raise ValueError("'method' must be a non-empty string to register a registration class.")
    _registered_registration_classes[method] = registration_class
    return registration_class


def get_registration_class(method: str) -> type:
    """Return the class that applies motion computed by ``method``.

    Built-in backends are imported lazily on first use, mirroring
    :func:`_get_motion_class`.
    """

    if method in _registered_registration_classes:
        return _registered_registration_classes[method]

    module_path = _builtin_motion_modules.get(method)
    if module_path is not None:
        try:
            importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Motion correction method '{method}' requires optional dependencies that are not installed "
                f"(failed to import '{module_path}': {exc})."
            ) from exc

    if method not in _registered_registration_classes:
        raise TypeError(
            f"Motion correction method '{method}' did not register a 'registration_class', "
            "so its motion cannot be applied."
        )
    return _registered_registration_classes[method]


def _encode_motion_value(value: Any) -> Any:
    """Tag ndarrays, torch tensors and tuples so :func:`_decode_motion_value` restores them exactly.

    JSON turns all three into plain lists, which would lose dtypes (an int16
    reference image would come back as int64), shapes of empty arrays, and
    tuple-ness (suite2p ``blocks``). Arrays are kept as arrays here:
    ``SIJsonEncoder`` only converts them to lists when dumping to json, and
    pickling keeps them as they are.
    """

    if isinstance(value, np.ndarray):
        return {"__ndarray__": value, "dtype": value.dtype.name, "shape": list(value.shape)}
    if type(value).__module__.split(".")[0] == "torch":
        array = value.detach().cpu().numpy()
        return {"__tensor__": array, "dtype": array.dtype.name, "shape": list(array.shape)}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_motion_value(v) for v in value]}
    if isinstance(value, list):
        return [_encode_motion_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode_motion_value(v) for k, v in value.items()}
    return value


def _decode_motion_value(value: Any) -> Any:
    """Inverse of :func:`_encode_motion_value`, for dicts fresh from json or still holding arrays."""

    if isinstance(value, dict):
        if "__ndarray__" in value:
            return np.asarray(value["__ndarray__"], dtype=value["dtype"]).reshape(value["shape"])
        if "__tensor__" in value:
            import torch  # noqa: PLC0415

            array = np.asarray(value["__tensor__"], dtype=value["dtype"]).reshape(value["shape"])
            return torch.from_numpy(array.copy())
        if "__tuple__" in value:
            return tuple(_decode_motion_value(v) for v in value["__tuple__"])
        return {k: _decode_motion_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_motion_value(v) for v in value]
    return value


class Motion:
    """Algorithm-agnostic container for motion correction artifacts.

    Holds outputs that any motion correction backend (Suite2P, CaImAn, ...) is
    expected to produce. Backend-specific fields (e.g. Suite2P ``ops`` or
    block-wise non-rigid offsets) belong on dedicated subclasses such as
    ``Suite2PMotion``.

    Subclasses declare two class attributes so that they can be reached
    through :meth:`compute`: ``method_name`` (the string users pass as
    ``method``) and ``settings_class`` (their pydantic settings model).

    The class that knows how to apply the stored motion (a
    ``BasePreprocessorEpoch``) is unrelated to ``Motion`` itself: a backend
    registers it under the same ``method_name`` via
    :func:`register_registration_class`, independently of whether or how it
    registers its ``Motion`` subclass, and
    :class:`photon_mosaic.preprocessing.registration.RegisterImaging` resolves
    it via :func:`get_registration_class`.
    """

    method_name: str | None = None
    settings_class: type | None = None

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
        method: KnownMotionMethod | str | None = None,
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

        resolved_settings = _coerce_settings(settings, target.settings_class)
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

    def _to_dict_kwargs(self) -> dict[str, Any]:
        """Constructor kwargs, minus ``imaging``, that rebuild this motion.

        Subclasses with extra constructor arguments extend this via ``super()``.
        """

        return dict(
            displacements=self.displacements,
            reference=self.reference,
            yranges=self.yranges,
            xranges=self.xranges,
            corrected_badframes=self.corrected_badframes,
            metadata=self.metadata,
        )

    @classmethod
    def _from_dict_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Inverse of :meth:`_to_dict_kwargs`, for subclasses that store a compacted form."""

        return kwargs

    def to_dict(self) -> dict[str, Any]:
        """Return a json-serializable description of this motion (with ``SIJsonEncoder``).

        ``imaging`` is not included: the owner of the motion (e.g.
        :class:`photon_mosaic.preprocessing.registration.RegisterImaging`)
        serializes it already and passes it back to :meth:`from_dict`.
        """

        return {
            "object": "Motion",
            "method": self.method_name,
            "kwargs": _encode_motion_value(self._to_dict_kwargs()),
        }

    @classmethod
    def from_dict(cls, motion_dict: dict[str, Any], imaging: BaseImaging) -> "Motion":
        """Rebuild a motion from :meth:`to_dict` output, as is or reloaded from json.

        Called on :class:`Motion` itself, the subclass is resolved from the
        stored ``method``; called on a subclass, that subclass is used.
        """

        method = motion_dict.get("method")
        target = _get_motion_class(method) if cls is Motion and method is not None else cls
        return target(imaging=imaging, **target._from_dict_kwargs(_decode_motion_value(motion_dict["kwargs"])))

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
