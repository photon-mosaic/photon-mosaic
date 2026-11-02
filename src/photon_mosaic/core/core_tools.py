from typing import Any


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
