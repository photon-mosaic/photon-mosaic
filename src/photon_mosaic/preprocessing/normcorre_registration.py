"""NoRMCorre motion correction following CaImAn v1.13.2 (``240e1f2``); line numbers cite that version."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from photon_mosaic.core import BaseImaging, BaseImagingEpoch, Motion, register_motion_class

from .registration import RegisterImagingEpoch, RegistrationSettings

BorderNan = bool | Literal["copy", "min"]

# Hard-coded in CaImAn's ``tile_and_correct_wrapper`` (``upsample_factor_fft=10``).
_UPSAMPLE_FACTOR_FFT = 10
# Frames CaImAn reads to estimate ``min_mov`` (``MotionCorrect.motion_correct``).
_MIN_MOV_FRAMES = 400


class NormcorreRegistrationSettings(RegistrationSettings):
    """CaImAn's ``motion`` parameters, with CaImAn's names and defaults.

    The shared ``nonrigid`` (CaImAn ``pw_rigid``), ``max_nonrigid_shift`` (CaImAn
    ``max_deviation_rigid``) and ``indices`` come from
    :class:`~photon_mosaic.preprocessing.registration.RegistrationSettings`.
    """

    max_shifts: tuple[int, int] = Field(default=(6, 6), description="Maximum rigid shift in pixels, (y, x).")
    niter_rig: int = Field(default=1, description="Rigid template-refinement iterations.")
    num_frames_split: int = Field(
        default=80,
        description="Sets the number of temporal splits, max(T // max(num_frames_split, 10), 1), from the first "
        "epoch's length (CaImAn params.py L935).",
    )
    strides: tuple[int, int] = Field(default=(96, 96), description="Distance between patch starts, (y, x).")
    overlaps: tuple[int, int] = Field(
        default=(32, 32), description="Overlap between patches, (y, x); patch size is strides + overlaps."
    )
    upsample_factor_grid: int = Field(
        default=4, description="Patch-grid upsampling of the FFT piecewise path (shifts_opencv=False)."
    )
    shifts_opencv: bool = Field(
        default=True,
        description="Apply shifts by interpolation (cubic warp / remap); False applies them in the Fourier "
        "domain, with piecewise patches upsampled and blended.",
    )
    shifts_interpolate: bool = Field(
        default=False,
        description="Build the piecewise field by interpolating from the patch centres instead of resizing the "
        "patch grid corner to corner.",
    )
    border_nan: BorderNan = Field(
        default="copy",
        description="Fill for the strip a shift uncovers: 'copy' the edge, the frame 'min', True for NaN, "
        "False for zero.",
    )
    gSig_filt: tuple[int, int] | None = Field(
        default=None,
        description="Std of the zero-mean Gaussian high-pass applied to the frames used for estimation "
        "(one-photon mode; CaImAn documents it as a size). Only the first value is used, for both axes. "
        "None disables.",
    )
    min_mov: float | None = Field(
        default=None, description="Movie minimum; None takes it from the first 400 frames of the first epoch."
    )

    model_config = ConfigDict(env_prefix="NORMCORRE_REGISTRATION_", case_sensitive=False, env_file=".env")


# --------------------------------------------------------------------------- #
# Leaf kernels -- stubs. Frames are float ``(T, H, W)``.
# --------------------------------------------------------------------------- #


def _high_pass_filter_space(frames: NDArray, gSig_filt: tuple[int, int]) -> NDArray:
    """Zero-mean Gaussian high-pass, frame by frame.

    Kernel from ``g = gSig_filt[0]`` alone: a 2-D Gaussian of std ``g`` and size ``(3*g)//2*2+1``;
    its values at or above the first column's maximum are made zero-mean, the rest set to
    zero; convolved with
    reflect padding. CaImAn ``high_pass_filter_space`` L1965 (``cv2.filter2D``);
    jnormcorre ``onephotonmethods.high_pass_filter_cv`` L29; masknmf
    ``spatial_filters.compute_highpass_filter_kernel`` L8.
    """
    raise NotImplementedError


def _bin_median(frames: NDArray, window: int = 10) -> NDArray:
    """``(H, W)`` median of the means of ``window``-frame bins.

    The first ``window * (T // window)`` frames are reshaped to ``(window, T // window, H, W)``
    and averaged over axis 0, so a bin holds frames ``T // window`` apart, not consecutive ones;
    then the median over bins. ``window`` is clamped to ``T`` for short inputs, and both
    reductions ignore NaNs (``nanmean``/``nanmedian``), as in CaImAn.

    CaImAn ``bin_median`` L1008; jnormcorre ``bin_median`` L690.
    """
    raise NotImplementedError


def _register_sample(frames: NDArray, max_shifts: tuple[int, int]) -> NDArray:
    """CaImAn's template-free registration of the template sample; returns the registered frames.

    ``movie.motion_correct(max_shifts[1], max_shifts[0], template=None)``: take every
    ``max(1, T / (1e8 / (H*W)))``-th frame, template = bin median; register by normalised
    cross-correlation (``cv2.matchTemplate`` ``TM_CCORR_NORMED`` on the template cropped by
    ``max_shifts``, sub-pixel by a log-parabola through the peak's neighbours); shift with
    cubic interpolation; new template = bin median; register all frames to it, take the bin
    median, register all frames to that and return them. CaImAn ``caiman/base/movies.py``
    ``motion_correct`` L97, ``extract_shifts`` L216, ``apply_shifts`` L308.
    """
    raise NotImplementedError


def _register_translation(
    frames: NDArray,
    template: NDArray,
    max_shifts: tuple[int, int],
    upsample_factor: int,
    shifts_lb: NDArray | None = None,
    shifts_ub: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Per frame, the ``(y, x)`` shift of ``frames`` relative to ``template`` and the global phase difference.

    Cross-power spectrum, integer peak within ``max_shifts``, or within ``[shifts_lb, shifts_ub]``
    instead when bounds are given (CaImAn L1586-1600),
    upsampled-DFT refinement to ``1/upsample_factor`` px. Returns CaImAn's ``rigid_shts``
    (the correction is its negative) ``(T, 2)`` and ``diffphase`` ``(T,)``.
    CaImAn ``register_translation`` L1454; jnormcorre ``register_translation_jax_simple`` L903;
    masknmf ``estimate_rigid_shifts`` L99.
    """
    raise NotImplementedError


def _register_patches(
    frames: NDArray,
    template: NDArray,
    strides: tuple[int, int],
    overlaps: tuple[int, int],
    max_shifts: tuple[int, int],
    max_nonrigid_shift: int,
    upsample_factor: int,
    rigid_shifts: NDArray,
) -> tuple[NDArray, NDArray, tuple[NDArray, NDArray]]:
    """:func:`_register_translation` per patch, bounded to ``ceil(rigid - d)..floor(rigid + d)``.

    Patches of ``strides + overlaps`` (``sliding_window`` L1839). Returns raw patch shifts
    ``(T, ny, nx, 2)``, their phase differences ``(T, ny, nx)`` and the patch centres
    ``(cy, cx)`` (``get_patch_centers`` L1793). CaImAn ``tile_and_correct`` L2013, piecewise
    branch; jnormcorre ``_register_to_template_pwrigid`` L1556; masknmf
    ``_estimate_patchwise_rigid_shifts`` L614.
    """
    raise NotImplementedError


def _apply_shift_iteration(frames: NDArray, shifts: NDArray, border_nan: BorderNan) -> NDArray:
    """Translate each frame by its correction shift with cubic interpolation (``shifts_opencv`` rigid path).

    ``cv2.warpAffine`` ``INTER_CUBIC``, reflect border, clipped to the frame's min/max, then the
    uncovered strip filled per ``border_nan``. CaImAn ``apply_shift_iteration`` L512.
    """
    raise NotImplementedError


def _apply_shifts_dft(frames: NDArray, shifts: NDArray, diffphase: NDArray | None, border_nan: BorderNan) -> NDArray:
    """Translate each frame by its correction shift through a Fourier phase ramp, then fill per ``border_nan``.

    CaImAn ``apply_shifts_dft`` L1654; jnormcorre ``apply_shifts_dft_fast_1`` L1219;
    masknmf ``apply_rigid_shifts`` L45.
    """
    raise NotImplementedError


def _apply_pw_shifts_remap(
    frames: NDArray,
    patch_shifts: NDArray,
    patch_centers: tuple[NDArray, NDArray],
    border_nan: BorderNan,
    shifts_interpolate: bool,
) -> NDArray:
    """Warp each frame by the dense field built from its correction patch shifts.

    Field from the patch grid: interpolated from ``patch_centers`` (``shifts_interpolate``) or
    resized corner to corner; frame sampled at ``grid - correction`` with cubic remap,
    ``border_nan`` as the border mode. CaImAn ``apply_pw_shifts_remap_2d`` L2533 and
    ``interpolate_shifts`` L1899; masknmf ``apply_displacement_vector_field`` L465.
    """
    raise NotImplementedError


def _apply_patch_shifts_fft(
    frames: NDArray,
    patch_shifts: NDArray,
    patch_diffphase: NDArray,
    patch_centers: tuple[NDArray, NDArray],
    strides: tuple[int, int],
    overlaps: tuple[int, int],
    upsample_factor_grid: int,
    shifts_interpolate: bool,
    border_nan: BorderNan,
) -> tuple[NDArray, NDArray, tuple[NDArray, NDArray]]:
    """CaImAn's FFT piecewise path: upsample the patch grid, Fourier-shift each patch, blend.

    New strides ``round(strides / upsample_factor_grid)``; raw shifts and phases brought onto
    the finer grid (from the centres if ``shifts_interpolate``, else resized); each finer patch
    shifted with :func:`_apply_shifts_dft`; overlaps blended with linear weights, or cut at the
    middle when neighbouring shifts differ by >= 0.5 px. Returns the corrected frames, the
    *correction* shifts on the finer grid and its centres. CaImAn ``tile_and_correct`` L2013
    (FFT branch) and ``create_weight_matrix_for_blending`` L1929.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------- #
# The CaImAn driver -- real code over the stubs.
# --------------------------------------------------------------------------- #


def _read_plane(epoch: BaseImagingEpoch, plane_index: int, start: int, end: int, step: int = 1) -> NDArray:
    """``(n, H, W)`` float32 frames of one plane."""

    video = epoch.get_series(start, end)[::step]
    plane = video[:, :, :, plane_index] if video.ndim == 4 else video
    return np.asarray(plane, dtype=np.float32)


def _crop(frames: NDArray, settings: NormcorreRegistrationSettings) -> NDArray:
    """The ``indices`` region of the frames (CaImAn ``m[:, indices[0], indices[1]]``)."""

    if settings.indices is None:
        return frames
    (y0, y1), (x0, x1) = settings.indices
    return frames[:, y0:y1, x0:x1]


def _tile_and_correct(
    frames: NDArray, template: NDArray, add_to_movie: float, settings: NormcorreRegistrationSettings
) -> tuple[NDArray, NDArray, NDArray | None, tuple[NDArray, NDArray] | None]:
    """One split through CaImAn's ``tile_and_correct`` (L2013), vectorised over frames.

    Returns the corrected frames and the *correction* shifts: rigid ``(T, 2)`` and, when
    ``settings.nonrigid``, per patch ``(T, ny, nx, 2)`` with the patch centres.
    """

    raw = frames.astype(np.float64)
    estimation = _high_pass_filter_space(raw, settings.gSig_filt) if settings.gSig_filt is not None else raw
    estimation = estimation + add_to_movie
    target = template.astype(np.float64) + add_to_movie
    # CaImAn shifts the unfiltered, un-offset image on the OpenCV paths when gSig_filt is set
    shift_input = raw if settings.gSig_filt is not None else estimation

    rigid, diffphase = _register_translation(estimation, target, settings.max_shifts, _UPSAMPLE_FACTOR_FFT)
    if not settings.nonrigid:
        if settings.shifts_opencv:
            corrected = _apply_shift_iteration(shift_input, -rigid, settings.border_nan)
        else:
            corrected = _apply_shifts_dft(estimation, -rigid, diffphase, settings.border_nan)
        return corrected - add_to_movie, -rigid, None, None

    patches, patch_phase, centers = _register_patches(
        estimation,
        target,
        settings.strides,
        settings.overlaps,
        settings.max_shifts,
        settings.max_nonrigid_shift,
        _UPSAMPLE_FACTOR_FFT,
        rigid,
    )
    if settings.shifts_opencv:
        corrected = _apply_pw_shifts_remap(
            shift_input, -patches, centers, settings.border_nan, settings.shifts_interpolate
        )
        return corrected - add_to_movie, -rigid, -patches, centers
    corrected, fine_shifts, fine_centers = _apply_patch_shifts_fft(
        estimation,
        patches,
        patch_phase,
        centers,
        settings.strides,
        settings.overlaps,
        settings.upsample_factor_grid,
        settings.shifts_interpolate,
        settings.border_nan,
    )
    return corrected - add_to_movie, -rigid, fine_shifts, fine_centers


def _split_template(corrected: NDArray) -> NDArray:
    """A split's template: ``nanmean`` over frames, NaN filled with the minimum (``tile_and_correct_wrapper``)."""

    template = np.nanmean(corrected.astype(np.float32), axis=0)  # CaImAn writes mc as float32
    template[np.isnan(template)] = np.nanmin(template)
    return template


def _correct_epoch(
    epoch: BaseImagingEpoch,
    plane_index: int,
    template: NDArray,
    add_to_movie: float,
    n_splits: int,
    settings: NormcorreRegistrationSettings,
) -> tuple[NDArray, NDArray, NDArray | None, tuple[NDArray, NDArray] | None]:
    """One pass of ``motion_correction_piecewise`` (L3144) over an epoch's contiguous splits.

    Returns the new template (median of the split templates, high-passed if ``gSig_filt``)
    and the shifts of every frame against the *input* template, as CaImAn's are.
    """

    n_frames = epoch.get_num_samples()
    rigid, patches, split_templates = [], [], []
    centers = None
    for idx in np.array_split(np.arange(n_frames), n_splits):
        if idx.size == 0:
            continue
        frames = _crop(_read_plane(epoch, plane_index, int(idx[0]), int(idx[-1]) + 1), settings)
        corrected, rigid_split, patch_split, centers = _tile_and_correct(frames, template, add_to_movie, settings)
        split_templates.append(_split_template(np.asarray(corrected)))
        rigid.append(rigid_split)
        if patch_split is not None:
            patches.append(patch_split)

    new_template = np.nanmedian(np.stack(split_templates), axis=0)
    if settings.gSig_filt is not None:
        new_template = _high_pass_filter_space(new_template[None], settings.gSig_filt)[0]
    return new_template, np.concatenate(rigid), np.concatenate(patches) if patches else None, centers


class NormcorreMotion(Motion):
    """Motion estimated by the NoRMCorre backend: arrays and a settings dump only, so it pickles.

    ``displacements`` is the rigid correction per frame ``(frames, planes, 2)`` (CaImAn
    ``shifts_rig``); ``reference`` the final per-plane template. For piecewise runs
    ``patch_shifts`` is ``[epoch][plane] -> (frames, ny, nx, 2)`` (CaImAn ``-x_shifts_els``,
    ``-y_shifts_els``) and ``patch_centers`` ``[plane] -> (cy, cx)`` in full-frame
    coordinates. ``metadata["border_to_0"]`` is CaImAn's ``border_to_0``.
    """

    method_name = "normcorre"
    settings_class = NormcorreRegistrationSettings

    def __init__(
        self,
        imaging: BaseImaging,
        displacements: Sequence[NDArray[np.floating[Any]]],
        reference: Any = None,
        patch_shifts: Sequence[Sequence[NDArray[np.floating[Any]] | None] | None] | None = None,
        patch_centers: Sequence[tuple[NDArray, NDArray] | None] | None = None,
        settings: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(imaging=imaging, displacements=displacements, reference=reference, metadata=metadata)
        self.patch_shifts = patch_shifts
        self.patch_centers = patch_centers
        self.settings = settings if settings is not None else {}

    @classmethod
    def _compute(
        cls,
        imaging: BaseImaging,
        *,
        settings: NormcorreRegistrationSettings | None = None,
        badframes: NDArray | None = None,
        **params: Any,
    ) -> "NormcorreMotion":
        """CaImAn's ``MotionCorrect.motion_correct`` (L222), per plane, with epochs as files.

        ``min_mov`` from the first 400 frames of the first epoch (high-passed if ``gSig_filt``).
        Rigid (``motion_correct_rigid`` L270 / ``motion_correct_batch_rigid`` L2782): for the
        first epoch the template is the bin median of the registered sample of every
        ``T // 50 + 1``-th frame; each epoch then runs ``niter_rig`` passes, each pass
        registering every split to the current template and replacing it by the median of the
        split means; the epoch's shifts are those of its last pass, and its template carries to
        the next epoch. Piecewise (``motion_correct_pwrigid`` L326): after the rigid run, one
        pass per epoch from the rigid template, carried across epochs the same way.
        """

        if badframes is not None:
            raise ValueError("The normcorre backend follows CaImAn, which has no bad-frame mechanism.")
        base = cls.settings_class() if settings is None else settings
        settings = cls.settings_class(**{**base.model_dump(), **params})
        if settings.gSig_filt is not None and not settings.shifts_opencv:
            # CaImAn raises the same inside tile_and_correct (L2107, L2209)
            raise ValueError("FFT shifts with gSig_filt are untested in CaImAn: set shifts_opencv=True.")
        if settings.nonrigid and settings.max_nonrigid_shift == 0:
            # CaImAn then takes the rigid branch of tile_and_correct and fails building x_shifts_els
            raise ValueError("nonrigid registration needs max_nonrigid_shift > 0.")
        epochs = imaging.epochs
        n_splits = max(epochs[0].get_num_samples() // max(settings.num_frames_split, 10), 1)
        y0 = settings.indices[0][0] or 0 if settings.indices is not None else 0
        x0 = settings.indices[1][0] or 0 if settings.indices is not None else 0
        rigid_settings = settings.model_copy(update={"nonrigid": False})

        rigid: list[list[NDArray]] = [[] for _ in epochs]
        patches: list[list[NDArray | None]] = [[] for _ in epochs]
        templates: list[NDArray] = []
        centers: list[tuple[NDArray, NDArray] | None] = []
        for p in range(imaging.num_planes):
            if settings.min_mov is None:
                first = _read_plane(epochs[0], p, 0, min(_MIN_MOV_FRAMES, epochs[0].get_num_samples()))
                if settings.gSig_filt is not None:
                    first = _high_pass_filter_space(first, settings.gSig_filt)
                min_mov = float(np.min(first))
            else:
                min_mov = settings.min_mov
            add_to_movie = float(np.float32(-min_mov))  # CaImAn passes it as float32 (L3202)
            if np.isnan(add_to_movie):
                raise ValueError("The movie contains NaNs. NaNs are not allowed!")

            template = None
            for e, epoch in enumerate(epochs):
                if template is None:
                    n_frames = epoch.get_num_samples()
                    picks = range(0, n_frames, n_frames // 50 + 1)
                    sample = _crop(np.concatenate([_read_plane(epoch, p, i, i + 1) for i in picks]), settings)
                    if settings.gSig_filt is not None:
                        sample = _high_pass_filter_space(sample, settings.gSig_filt)
                    template = _bin_median(_register_sample(sample, settings.max_shifts))
                for _ in range(settings.niter_rig):
                    template, shifts, _, _ = _correct_epoch(epoch, p, template, add_to_movie, n_splits, rigid_settings)
                rigid[e].append(shifts)

            plane_centers = None
            if settings.nonrigid:
                for e, epoch in enumerate(epochs):
                    template, _, plane_patches, plane_centers = _correct_epoch(
                        epoch, p, template, add_to_movie, n_splits, settings
                    )
                    if np.isnan(np.sum(template)):
                        raise ValueError("Template contains NaNs, something went wrong. Reconsider the parameters")
                    patches[e].append(plane_patches)
                assert plane_centers is not None
                plane_centers = (np.asarray(plane_centers[0]) + y0, np.asarray(plane_centers[1]) + x0)
            templates.append(template)
            centers.append(plane_centers)

        displacements = [np.stack(epoch_shifts, axis=1) for epoch_shifts in rigid]
        moved = [np.abs(d).max() for d in displacements]
        if settings.nonrigid:
            moved = [max(np.abs(x).max() for x in epoch_patches if x is not None) for epoch_patches in patches]
        return cls(
            imaging=imaging,
            displacements=displacements,
            reference=templates,
            patch_shifts=patches if settings.nonrigid else None,
            patch_centers=centers if settings.nonrigid else None,
            settings=settings.model_dump(),
            metadata={"border_to_0": int(np.ceil(max(moved)))},
        )


class RegisterNormcorreImagingEpoch(RegisterImagingEpoch):
    """Replays stored NoRMCorre shifts on read, as CaImAn's ``apply_shifts_movie`` (L399).

    Frame bounds, plane selection and output allocation come from
    :class:`~photon_mosaic.preprocessing.registration.RegisterImagingEpoch`. Registered with
    :func:`photon_mosaic.core.register_registration_class` only once the apply kernels exist.
    """

    motion: NormcorreMotion

    def _correct_plane(
        self,
        plane_video: NDArray[np.floating[Any]],
        plane_index: int,
        start_frame: int,
        end_frame: int,
    ) -> NDArray[np.floating[Any]]:
        """Rigid: cubic warp or DFT per ``shifts_opencv``. Piecewise: always the remap, on the full frame.

        As CaImAn, the piecewise replay interpolates from the patch centres whenever
        ``indices`` crops the estimation region.
        """

        settings = NormcorreRegistrationSettings.model_validate(self.motion.settings)
        p = plane_index
        epoch_patches = self.motion.patch_shifts[self.epoch_index] if self.motion.patch_shifts is not None else None
        if epoch_patches is None:
            shifts = self.motion.displacements[self.epoch_index][start_frame:end_frame, p]
            if settings.shifts_opencv:
                return _apply_shift_iteration(plane_video, shifts, settings.border_nan)
            return _apply_shifts_dft(plane_video, shifts, None, settings.border_nan)

        centers = self.motion.patch_centers[p] if self.motion.patch_centers is not None else None
        assert centers is not None, "piecewise motion without patch centres"
        cropped = settings.indices is not None and settings.indices != ((None, None), (None, None))
        interpolate = settings.shifts_interpolate or cropped
        patches = epoch_patches[p]
        assert patches is not None
        return _apply_pw_shifts_remap(
            plane_video, patches[start_frame:end_frame], centers, settings.border_nan, interpolate
        )


def compute_motion_metrics(
    imaging: BaseImaging, motion: NormcorreMotion | None = None, template: NDArray | None = None
) -> dict[str, Any]:
    """Registration quality: per-frame correlation to the template, optical-flow residual, crispness.

    CaImAn ``compute_metrics_motion_correction`` L2650.
    """
    raise NotImplementedError


# ``register_registration_class("normcorre", RegisterNormcorreImagingEpoch)`` is added
# by the PR that implements the apply kernels.
register_motion_class(NormcorreMotion)
