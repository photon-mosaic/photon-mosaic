from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

from photon_mosaic.core import BaseImaging, Motion, get_registration_class

from .basepreprocessor import BasePreprocessor


class RegistrationSettings(BaseSettings):
    """Settings shared by every motion correction backend.

    Only options that are meaningful for any backend live here; algorithm
    specific parameters belong on the subclass (e.g.
    ``Suite2pRegistrationSettings``), which is also expected to set its own
    ``env_prefix``. The prefix here is deliberately not empty: without it these
    fields would read bare ``DEVICE`` / ``BATCH_SIZE`` from the environment.
    """

    debug: bool = Field(default=False, description="Run with partial dataset")
    tmp_dir: str | Path = Field(
        default=Path("/scratch"),
        description="Directory into which to write temporary files produced by the registration backend",
    )
    batch_size: int = Field(default=500, description="Number of frames per batch")
    device: str = Field(
        default="cpu",
        description="Torch device for registration: 'cpu', 'cuda', or 'mps'.",
    )

    model_config = ConfigDict(env_prefix="REGISTRATION_", case_sensitive=False, env_file=".env")


class RegisterImaging(BasePreprocessor):
    """Apply pre-computed motion correction on-the-fly, whatever the backend.

    Dispatch is by ``method`` (like :meth:`Motion.compute`), defaulting to
    ``motion.method_name``, and resolved through
    :func:`photon_mosaic.core.get_registration_class`, so this class imports
    no backend itself.
    """

    def __init__(self, imaging: BaseImaging, motion: Motion, method: str | None = None, **kwargs: Any) -> None:
        """Build an imaging view that applies stored motion fields lazily."""
        BasePreprocessor.__init__(self, imaging)

        if motion.num_epochs != len(imaging.epochs):
            raise ValueError(
                f"Number of epochs in motion ({motion.num_epochs}) does not match imaging ({len(imaging.epochs)})"
            )

        if method is not None and motion.method_name is not None and method != motion.method_name:
            raise ValueError(
                f"motion was computed with method '{motion.method_name}', but method='{method}' was requested."
            )

        resolved_method = method or motion.method_name
        if resolved_method is None:
            raise TypeError(
                f"{type(motion).__name__} does not declare a 'method_name', so no registration_class can be "
                "resolved for it."
            )

        registration_class = get_registration_class(resolved_method)

        for epoch_idx, parent_epoch in enumerate(imaging.epochs):
            self.add_epoch(registration_class(parent_epoch, motion, epoch_idx, **kwargs))

        self._kwargs = dict(imaging=imaging, motion=motion, **kwargs)


register_motion = RegisterImaging
