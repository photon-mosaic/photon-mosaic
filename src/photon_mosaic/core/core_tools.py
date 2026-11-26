import warnings
from typing import Any

# a torch device string: "cpu", "cuda", "mps", optionally with an index ("cuda:1")
_DEVICE_PATTERN = r"^(cpu|cuda|mps)(:\d+)?$"


def _warn_and_keep_default(settings_class: Any, name: str, value: Any, backend: str) -> Any:
    """Return the default of a setting the backend does not use yet, warning if ``value`` differs from it."""

    default = settings_class.model_fields[name].default
    if value != default and str(value) != str(default):  # str: tmp_dir="/scratch" is its Path default
        warnings.warn(
            f"{name}={value!r} is not used by the {backend} backend yet: using the default {default!r}.",
            UserWarning,
            stacklevel=3,
        )
        return default
    return value


def _coerce_settings(settings: Any, settings_class: type | None) -> Any:
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
