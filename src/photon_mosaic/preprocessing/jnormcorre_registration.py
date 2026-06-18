"""jnormcorre (JAX NoRMCorre) motion-correction backend.

This mirrors the Suite2P backend in :mod:`photon_mosaic.preprocessing.suite2p_registration`:
a *compute* phase (``compute_motion_jnormcorre``) that estimates a registration
template + per-frame shifts without touching the data, and an *apply* phase
(``RegisterJNormcorreImaging`` / ``RegisterJNormcorreImagingEpoch``) that applies
the correction lazily, frame-chunk by frame-chunk, inside ``get_series``.

jnormcorre is the JAX-accelerated reimplementation of NoRMCorre — the
motion-correction core of CaImAn (issue #79). Its API maps onto ours as:

================================  ==========================================
photon-mosaic (this module)       jnormcorre
================================  ==========================================
``compute_motion_jnormcorre``     ``MotionCorrect(loader, ...).motion_correct``
``JNormcorreMotion``              the returned ``FrameCorrector`` (+ template)
``register_frames`` in get_series ``FrameCorrector.register_frames``
``JNormcorreRegistrationSettings``the ``MotionCorrect`` constructor kwargs
================================  ==========================================

Two facts about jnormcorre shaped the design:

1. ``MotionCorrect`` consumes a ``lazy_data_loader`` (abstract ``dtype`` /
   ``shape`` / ``_compute_at_indices``), not an in-memory array. We adapt a
   :class:`~photon_mosaic.core.BaseImagingEpoch` (per plane — jnormcorre is 2-D)
   with a private loader built by :func:`_make_plane_loader`.
2. The returned ``FrameCorrector`` holds the final *template* and registers any
   frames handed to it against that template. So the lazy apply only needs the
   retained ``FrameCorrector``; the per-frame shift arrays
   (``MotionCorrect.shifts_rig`` / ``x_shifts_els`` / ``y_shifts_els``) are read
   afterwards purely to populate the algorithm-agnostic
   :class:`~photon_mosaic.core.motion.Motion` ``displacements`` contract and are
   *best-effort diagnostics*, not the source of truth for the applied shift.

jnormcorre is an optional dependency (``pip install
"photon-mosaic[jnormcorre-registration]"``); it is imported lazily inside the
functions that need it so the rest of the package works without JAX installed.
"""

import logging
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

from photon_mosaic.core import BaseImaging, BaseImagingEpoch, Motion

from .basepreprocessor import BasePreprocessor, BasePreprocessorEpoch


class JNormcorreRegistrationSettings(BaseSettings):
    """Settings for jnormcorre (JAX NoRMCorre) motion correction.

    Field names and defaults follow ``jnormcorre.motion_correction.MotionCorrect``
    so that ``model_dump()`` can be forwarded to it directly. Values can be
    provided via constructor, environment variables (prefix
    ``JNORMCORRE_REGISTRATION_``), or a ``.env`` file.
    """

    max_shifts: tuple[int, int] = Field(
        default=(6, 6),
        description="Maximum allowed rigid shift in pixels, as (y, x).",
    )
    frames_per_split: int = Field(
        default=1000,
        description="Number of frames per processing chunk during template estimation.",
    )
    num_splits_to_process_rig: int | None = Field(
        default=None,
        description="Number of chunks to use for the rigid template estimate " "(None uses all chunks).",
    )
    niter_rig: int = Field(
        default=1,
        description="Number of iterations of rigid template refinement.",
    )
    pw_rigid: bool = Field(
        default=False,
        description="Whether to use piecewise-rigid (non-rigid) registration in "
        "addition to the rigid pass. Mirrors Suite2P's `nonrigid`.",
    )
    strides: tuple[int, int] = Field(
        default=(96, 96),
        description="Distance between the start of overlapping patches for "
        "piecewise-rigid registration, in pixels (y, x).",
    )
    overlaps: tuple[int, int] = Field(
        default=(32, 32),
        description="Overlap between adjacent piecewise-rigid patches, in pixels (y, x).",
    )
    max_deviation_rigid: int = Field(
        default=3,
        description="Maximum allowed deviation (pixels) of a patch shift from the "
        "rigid shift, for piecewise-rigid registration.",
    )
    num_splits_to_process_els: int | None = Field(
        default=None,
        description="Number of chunks to use for the piecewise-rigid template " "estimate (None uses all chunks).",
    )
    niter_els: int = Field(
        default=1,
        description="Number of iterations of piecewise-rigid template refinement.",
    )
    min_mov: float | None = Field(
        default=None,
        description="Minimum value of the movie, subtracted before registration. " "If None, jnormcorre estimates it.",
    )
    upsample_factor_grid: int = Field(
        default=4,
        description="Upsampling factor of the patch grid for piecewise-rigid " "shift interpolation.",
    )
    batching: int = Field(
        default=100,
        description="Number of frames the FrameCorrector registers per batch when "
        "applying the correction (apply phase only — not a MotionCorrect argument).",
    )

    model_config = ConfigDict(env_prefix="JNORMCORRE_REGISTRATION_", case_sensitive=False, env_file=".env")


def _make_plane_loader(
    epoch: BaseImagingEpoch,
    plane_index: int,
    n_frames: int,
    Ly: int,
    Lx: int,
) -> Any:
    """Build a jnormcorre ``lazy_data_loader`` view over a single plane of an epoch.

    jnormcorre operates on 2-D ``(n_frames, Ly, Lx)`` movies, so one loader is
    created per plane. Frames are pulled lazily from the parent epoch's
    ``get_series`` and the requested plane is sliced out — mirroring how
    ``compute_motion_suite2p`` slices ``all_frames[:, :, :, p]``.

    Parameters
    ----------
    epoch : BaseImagingEpoch
        Parent imaging epoch providing frames.
    plane_index : int
        Plane to expose.
    n_frames, Ly, Lx : int
        Movie dimensions for the loader's ``shape``.

    Returns
    -------
    lazy_data_loader
        A jnormcorre loader exposing this plane as ``(n_frames, Ly, Lx)``.
    """
    from jnormcorre.utils.lazy_array import lazy_data_loader

    def _slice_plane(frames: NDArray) -> NDArray:
        arr = np.asarray(frames, dtype=np.float32)
        # get_series returns (n, H, W, n_planes) for volumetric data and
        # (n, H, W) for a single plane.
        if arr.ndim == 4:
            return arr[:, :, :, plane_index]
        return arr

    class _EpochPlaneLoader(lazy_data_loader):
        """Adapter: a single plane of a ``BaseImagingEpoch`` as a jnormcorre loader."""

        @property
        def dtype(self) -> str:
            return "float32"

        @property
        def shape(self) -> tuple[int, int, int]:
            return (n_frames, Ly, Lx)

        def _compute_at_indices(self, indices: list | int | slice) -> NDArray:
            if isinstance(indices, slice):
                start, stop, step = indices.indices(n_frames)
                block = _slice_plane(epoch.get_series(start, stop))
                if step != 1:
                    block = block[::step]
                return block
            if isinstance(indices, (int, np.integer)):
                i = int(indices)
                return _slice_plane(epoch.get_series(i, i + 1))[0]
            # Arbitrary list of frame indices: load the enclosing contiguous
            # range once and gather, since get_series only takes a range.
            idx = list(indices)
            lo, hi = min(idx), max(idx)
            block = _slice_plane(epoch.get_series(lo, hi + 1))
            return block[[i - lo for i in idx]]

    return _EpochPlaneLoader()


def _extract_shifts(mc: Any, n_frames: int, pw_rigid: bool) -> tuple[NDArray, tuple[NDArray, NDArray] | None]:
    """Read per-frame shifts off a completed ``MotionCorrect`` (best-effort).

    These populate the algorithm-agnostic :class:`Motion` ``displacements``
    contract and the piecewise-rigid patch offsets. They are *diagnostics*: the
    correction actually applied at read time is recomputed by the retained
    ``FrameCorrector`` against the template, so this extraction never gates
    correctness. Attributes are read defensively (``getattr``) because their
    presence depends on the rigid/piecewise path taken.

    Returns
    -------
    displacements : NDArray
        ``(n_frames, 2)`` rigid displacement in ``(y, x)`` order. Zeros if the
        rigid shifts are unavailable.
    patch_shifts : tuple[NDArray, NDArray] | None
        ``(y_shifts, x_shifts)`` each ``(n_frames, n_patches)`` when
        ``pw_rigid`` is set and available, else ``None``.
    """
    displacements = np.zeros((n_frames, 2), dtype=np.float32)

    shifts_rig = getattr(mc, "shifts_rig", None)
    if shifts_rig is not None and len(shifts_rig) > 0:
        arr = np.asarray(shifts_rig, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == n_frames and arr.shape[1] == 2:
            displacements = arr

    patch_shifts: tuple[NDArray, NDArray] | None = None
    if pw_rigid:
        y_els = getattr(mc, "y_shifts_els", None)
        x_els = getattr(mc, "x_shifts_els", None)
        if y_els is not None and x_els is not None and len(y_els) > 0:
            y_arr = np.asarray(y_els, dtype=np.float32)
            x_arr = np.asarray(x_els, dtype=np.float32)
            patch_shifts = (y_arr, x_arr)
            # If a rigid summary wasn't available, use the mean patch shift as a
            # representative rigid displacement so the contract stays populated.
            if not (shifts_rig is not None and len(shifts_rig) > 0) and y_arr.ndim == 2:
                displacements = np.stack([y_arr.mean(axis=1), x_arr.mean(axis=1)], axis=1).astype(np.float32)

    return displacements, patch_shifts


class JNormcorreMotion(Motion):
    """Motion artifacts produced by jnormcorre registration.

    Adds jnormcorre-specific fields on top of the algorithm-agnostic
    :class:`Motion` container: the per-epoch / per-plane ``FrameCorrector``
    objects (which hold the registration template and apply the correction at
    read time), the templates, and — for piecewise-rigid — the per-patch shifts.
    """

    def __init__(
        self,
        imaging: BaseImaging,
        displacements: Sequence[NDArray[np.floating[Any]]],
        frame_correctors: Sequence[Sequence[Any]],
        templates: Sequence[NDArray | None],
        pw_rigid: bool,
        settings: dict[str, Any] | None = None,
        patch_shifts: Sequence[Sequence[tuple[NDArray, NDArray] | None] | None] | None = None,
        reference: Any = None,
    ) -> None:
        """Store jnormcorre registration outputs.

        Parameters
        ----------
        imaging, displacements, reference
            See :class:`photon_mosaic.core.motion.Motion`.
        frame_correctors : Sequence[Sequence]
            ``[epoch][plane] -> jnormcorre.FrameCorrector``. Each corrector holds
            the final template and applies the correction in ``get_series``.
        templates : Sequence
            Per-plane registration templates (also stored as ``reference``).
        pw_rigid : bool
            Whether piecewise-rigid registration was used (selects the
            ``register_frames`` mode at apply time).
        settings : dict | None, optional
            The resolved settings used for registration.
        patch_shifts : Sequence | None, optional
            ``[epoch][plane] -> (y_shifts, x_shifts)`` per-patch offsets for
            piecewise-rigid registration (``None`` per plane/epoch when
            unavailable, or ``None`` overall for rigid-only).
        """
        super().__init__(
            imaging=imaging,
            displacements=displacements,
            reference=reference if reference is not None else templates,
        )
        self.frame_correctors = frame_correctors
        self.templates = templates
        self.pw_rigid = pw_rigid
        self.settings = settings if settings is not None else {}
        self.patch_shifts = patch_shifts


def compute_motion_jnormcorre(
    imaging: BaseImaging,
    settings: JNormcorreRegistrationSettings | dict[str, Any] | None = None,
    badframes: NDArray | None = None,
    **kwargs: Any,
) -> "JNormcorreMotion":
    """Estimate jnormcorre registration templates + shifts for all planes/epochs.

    Builds a per-plane lazy loader over each epoch, runs jnormcorre's
    ``MotionCorrect.motion_correct`` (which estimates the template and per-frame
    shifts without writing corrected frames, ``save_movie=False``), and stores the
    resulting ``FrameCorrector`` objects for lazy application by
    ``RegisterJNormcorreImaging``. The plane templates are computed on the first
    epoch and reused for later epochs, mirroring ``compute_motion_suite2p``.

    Parameters
    ----------
    imaging : BaseImaging
        Imaging object containing one or more epochs/planes to register.
    settings : JNormcorreRegistrationSettings | dict | None, optional
        Registration settings. Dicts and None are coerced into
        ``JNormcorreRegistrationSettings``. Extra keyword arguments override.
    badframes : NDArray | None, optional
        Accepted for interface parity with ``compute_motion_suite2p``. jnormcorre
        has no bad-frame mechanism, so this is currently ignored (a warning is
        logged if provided).
    **kwargs : Any
        Extra options forwarded to ``JNormcorreRegistrationSettings``.

    Returns
    -------
    JNormcorreMotion
        Motion container with per-epoch displacements and the per-epoch/per-plane
        ``FrameCorrector`` objects.
    """
    from jnormcorre.motion_correction import MotionCorrect

    if settings is None:
        user_settings = JNormcorreRegistrationSettings()
    elif isinstance(settings, dict):
        user_settings = JNormcorreRegistrationSettings(**settings)
    else:
        user_settings = settings
    cfg = user_settings.model_dump()
    cfg.update(kwargs)

    if badframes is not None and badframes.any():
        logging.warning(
            "compute_motion_jnormcorre received `badframes` but jnormcorre has no bad-frame mechanism; ignoring."
        )

    pw_rigid = bool(cfg["pw_rigid"])
    cfg.pop("batching")  # apply-phase only; not a MotionCorrect argument

    # Whatever remains in cfg is exactly the MotionCorrect constructor kwargs.
    mc_kwargs = dict(cfg)

    n_planes = imaging.num_planes
    n_epochs = imaging.get_num_epochs()
    Ly, Lx = imaging.shape[0], imaging.shape[1]

    templates: list[NDArray | None] = [None] * n_planes

    all_displacements: list[NDArray] = []
    all_frame_correctors: list[list[Any]] = []
    all_patch_shifts: list[list[tuple[NDArray, NDArray] | None] | None] = []

    for epoch_idx in range(n_epochs):
        epoch = imaging.epochs[epoch_idx]
        n_frames = imaging.get_num_samples(segment_index=epoch_idx)

        epoch_disps: list[NDArray] = []
        epoch_fcs: list[Any] = []
        epoch_patch: list[tuple[NDArray, NDArray] | None] = []

        for p in range(n_planes):
            loader = _make_plane_loader(epoch, p, n_frames, Ly, Lx)
            mc = MotionCorrect(loader, **mc_kwargs)
            # Reuse the epoch-0 template for later epochs (cross-epoch consistency).
            frame_corrector, _ = mc.motion_correct(template=templates[p], save_movie=False)

            if epoch_idx == 0:
                templates[p] = getattr(frame_corrector, "_template", None)

            disp, patch = _extract_shifts(mc, n_frames, pw_rigid)
            epoch_disps.append(disp)
            epoch_fcs.append(frame_corrector)
            epoch_patch.append(patch)

        # Stack planes → (n_frames, n_planes, 2)
        all_displacements.append(np.stack(epoch_disps, axis=1))
        all_frame_correctors.append(epoch_fcs)
        has_patch = any(o is not None for o in epoch_patch)
        all_patch_shifts.append(epoch_patch if has_patch else None)

    patch_shifts = all_patch_shifts if any(o is not None for o in all_patch_shifts) else None

    return JNormcorreMotion(
        imaging=imaging,
        displacements=all_displacements,
        frame_correctors=all_frame_correctors,
        templates=templates,
        pw_rigid=pw_rigid,
        settings=cfg,
        patch_shifts=patch_shifts,
        reference=templates,
    )


class RegisterJNormcorreImaging(BasePreprocessor):
    """Apply pre-computed jnormcorre motion correction on-the-fly."""

    def __init__(self, imaging: BaseImaging, motion: JNormcorreMotion, **kwargs: Any) -> None:
        """Build an imaging view that applies stored motion fields lazily."""
        BasePreprocessor.__init__(self, imaging)

        if motion.num_epochs != len(imaging.epochs):
            raise ValueError(
                f"Number of epochs in motion ({motion.num_epochs}) does not match imaging ({len(imaging.epochs)})"
            )

        for epoch_idx, parent_epoch in enumerate(imaging.epochs):
            epoch = RegisterJNormcorreImagingEpoch(parent_epoch, motion, epoch_idx, **kwargs)
            self.add_epoch(epoch)

        self._kwargs = dict(imaging=imaging, motion=motion, **kwargs)


class RegisterJNormcorreImagingEpoch(BasePreprocessorEpoch):
    """Epoch-level preprocessor that applies stored jnormcorre correction."""

    def __init__(
        self,
        parent_imaging_epoch: BaseImagingEpoch,
        motion: JNormcorreMotion,
        epoch_index: int,
        **kwargs: Any,
    ) -> None:
        """Create an epoch preprocessor for a specific epoch and FrameCorrector set."""
        BasePreprocessorEpoch.__init__(self, parent_imaging_epoch)
        self.motion = motion
        self.epoch_index = epoch_index
        self.kwargs = kwargs

    def get_series(
        self,
        start_frame: int,
        end_frame: int,
        plane_indices: int | slice | Sequence[int] | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return motion-corrected frames for the requested interval and planes.

        Each plane's frames are aligned to that plane's template by the stored
        ``FrameCorrector`` (jnormcorre re-registers the requested frames against
        the template), so no per-frame shift replay is needed here.
        """
        video = self.parent_imaging_epoch.get_series(start_frame, end_frame)
        num_planes = video.shape[3] if video.ndim == 4 else 1

        if plane_indices is None:
            planes_to_process = list(range(num_planes))
        elif isinstance(plane_indices, int):
            planes_to_process = [plane_indices]
        elif isinstance(plane_indices, slice):
            planes_to_process = list(range(*plane_indices.indices(num_planes)))
        else:
            planes_to_process = list(plane_indices)

        # Match the actual frame count the parent returned (it may truncate when
        # end_frame exceeds the available samples).
        n_frames = video.shape[0]
        H, W = video.shape[1], video.shape[2]
        output = np.empty((n_frames, H, W, len(planes_to_process)), dtype=np.float32)
        if n_frames == 0:
            return output

        frame_correctors = self.motion.frame_correctors[self.epoch_index]

        for i, p in enumerate(planes_to_process):
            plane_video = video[:, :, :, p] if video.ndim == 4 else video
            plane_video = plane_video.astype("float32", copy=True)

            # Ensure plane_video is always 3D (frames, height, width)
            if plane_video.ndim == 2:
                plane_video = plane_video[np.newaxis, :, :]

            registered_plane = frame_correctors[p].register_frames(plane_video, pw_rigid=self.motion.pw_rigid)
            registered_plane = np.asarray(registered_plane)

            # Ensure registered_plane is always 3D (frames, height, width)
            if registered_plane.ndim == 2:
                registered_plane = registered_plane[np.newaxis, :, :]

            output[..., i] = registered_plane

        return output


register_jnormcorre = RegisterJNormcorreImaging
