"""NoRMCorre motion correction following CaImAn v1.13.2 (``240e1f2``); line numbers cite that version."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from photon_mosaic.core import (
    BaseImaging,
    BaseImagingEpoch,
    Motion,
    register_motion_class,
)

from .registration import RegisterImagingEpoch, RegistrationSettings

BorderNan = bool | Literal["copy", "min"]

# Hard-coded in CaImAn's ``tile_and_correct_wrapper`` (``upsample_factor_fft=10``).
_UPSAMPLE_FACTOR_FFT = 10
# Frames CaImAn reads to estimate ``min_mov`` (``MotionCorrect.motion_correct``).
_MIN_MOV_FRAMES = 400


class NormcorreRegistrationSettings(RegistrationSettings):
    """Settings of the NoRMCorre backend, with CaImAn's parameter names and defaults.

    Pass any of them as keyword arguments to ``compute_motion(..., method="normcorre")``,
    or set them as ``NORMCORRE_REGISTRATION_<NAME>`` environment variables. Three settings
    are shared with the other backends and documented on
    :class:`~photon_mosaic.preprocessing.registration.RegistrationSettings`:
    ``nonrigid`` (CaImAn ``pw_rigid``), ``max_nonrigid_shift`` (CaImAn
    ``max_deviation_rigid``) and ``bounds`` (CaImAn ``indices``).
    """

    max_shifts: tuple[int, int] = Field(
        default=(6, 6),
        description="Largest rigid shift searched for, in pixels, as (y, x). Raise it if the sample moves more.",
    )
    niter_rig: int = Field(
        default=1,
        description="How many times the rigid pass is repeated, each against the template the previous one made.",
    )
    num_frames_split: int = Field(
        default=80,
        description="Approximate frames per chunk. Each pass cuts an epoch into max(T // max(num_frames_split, "
        "10), 1) chunks (T: frames in the first epoch), corrects each against the current template, and makes "
        "the next template from the median of the chunk averages.",
    )
    strides: tuple[int, int] = Field(
        default=(96, 96),
        description="Non-rigid only: distance in pixels between the starts of neighbouring patches, (y, x).",
    )
    overlaps: tuple[int, int] = Field(
        default=(32, 32),
        description="Non-rigid only: overlap in pixels between neighbouring patches, (y, x). Each patch is "
        "strides + overlaps pixels wide.",
    )
    upsample_factor_grid: int = Field(
        default=4,
        description="Non-rigid with shifts_opencv=False only: how much finer than the estimation grid the grid "
        "of patches that are shifted and blended back together is.",
    )
    shifts_opencv: bool = Field(
        default=True,
        description="True applies shifts by cubic interpolation (OpenCV warp/remap); False applies them in the "
        "Fourier domain. Stored non-rigid shifts are always replayed by interpolation, as in CaImAn.",
    )
    shifts_interpolate: bool = Field(
        default=False,
        description="Non-rigid only: build the per-pixel shift field by interpolating between patch centres, "
        "instead of stretching the patch grid over the whole frame.",
    )
    border_nan: BorderNan = Field(
        default="copy",
        description="What fills the strip a shift exposes at the frame edge: 'copy' repeats the edge pixels, "
        "'min' uses the frame minimum, True leaves NaN. False fills nothing: the strip keeps what the shift "
        "produced (reflected pixels for the rigid warp, wrapped-around ones in the Fourier domain, zero for "
        "the non-rigid remap).",
    )
    gSig_filt: tuple[int, int] | None = Field(
        default=None,
        description="For one-photon data: width (std, in pixels) of a high-pass filter applied to the frames "
        "before shifts are estimated, to remove the diffuse background; the shifts are still applied to the "
        "unfiltered data. Only the first value is used. None (default) disables it.",
    )
    min_mov: float | None = Field(
        default=None,
        description="Offset that makes the movie non-negative during estimation; None takes the minimum of the "
        "first 400 frames of the first epoch.",
    )

    model_config = ConfigDict(env_prefix="NORMCORRE_REGISTRATION_", case_sensitive=False, env_file=".env")


# --------------------------------------------------------------------------- #
# Kernels. Each runs every frame through CaImAn's per-image code with the same OpenCV
# calls, so results, dtypes and rounding match CaImAn's.
# --------------------------------------------------------------------------- #


def _high_pass_filter_space(frames: NDArray, gSig_filt: tuple[int, int]) -> NDArray:
    """Remove the slowly varying background of each frame (one-photon mode).

    Each frame is convolved with a Gaussian of std ``g = gSig_filt[0]`` (size
    ``(3*g)//2*2+1``) that is cut to its central lobe and made zero-mean, which keeps
    cell-sized structure and removes anything broader. Borders are reflected.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to filter.
    gSig_filt : tuple of int
        Only the first value is used, for both axes.

    Returns
    -------
    array, ``(T, H, W)`` float32
        The filtered frames.

    Notes
    -----
    CaImAn ``high_pass_filter_space`` L1965; jnormcorre ``onephotonmethods.high_pass_filter_cv``
    L29; masknmf ``spatial_filters.compute_highpass_filter_kernel`` L8.
    """
    ksize = tuple((3 * i) // 2 * 2 + 1 for i in gSig_filt)
    ker = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    ker2D = ker.dot(ker.T)
    nz = np.nonzero(ker2D >= ker2D[:, 0].max())
    zz = np.nonzero(ker2D < ker2D[:, 0].max())
    ker2D[nz] -= ker2D[nz].mean()
    ker2D[zz] = 0
    return np.stack(
        [cv2.filter2D(np.array(img, dtype=np.float32), -1, ker2D, borderType=cv2.BORDER_REFLECT) for img in frames]
    )


def _bin_median(frames: NDArray, window: int = 10, clamp_window: bool = True) -> NDArray:
    """A robust average image: the median of ``window`` bin averages.

    The first ``window * (T // window)`` frames are split into ``window`` bins of frames
    spaced ``T // window`` apart (not consecutive ones), each bin is averaged, and the
    median of those averages is returned. NaNs are ignored.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        The movie.
    window : int
        Number of bins.
    clamp_window : bool
        Use ``min(window, T)`` bins, as CaImAn's module function does; False is CaImAn's
        movie method, which does not clamp.

    Returns
    -------
    array, ``(H, W)``
        The median image.

    Notes
    -----
    CaImAn ``bin_median`` L1008 and ``caiman/base/movies.py`` ``bin_median`` L175;
    jnormcorre ``bin_median`` L690.
    """
    T, d1, d2 = frames.shape
    if clamp_window and T < window:
        window = T
    num_windows = int(T // window)
    num_frames = num_windows * window
    binned = np.reshape(frames[:num_frames], (window, num_windows, d1, d2))
    return np.nanmedian(np.nanmean(binned, axis=0), axis=0)


def _extract_shifts(frames: NDArray, max_shift_w: int, max_shift_h: int, template: NDArray) -> NDArray:
    """Find each frame's rigid correction by template matching, for the template sample only.

    The template, cropped by the maximum shift on every side, is slid over each frame
    (normalised cross-correlation); the best position, refined to sub-pixel by a
    log-parabola through its neighbours, gives the shift.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to register.
    max_shift_w, max_shift_h : int
        Largest shift searched for along x and y.
    template : array, ``(H, W)``
        The image to register to.

    Returns
    -------
    array, ``(T, 2)``
        Per frame, the ``(y, x)`` correction.

    Notes
    -----
    CaImAn ``caiman/base/movies.py`` ``extract_shifts`` L216.
    """
    min_val = np.percentile(frames, 1)
    if min_val < -0.1:
        frames = frames - min_val
    frames = np.asarray(frames, dtype=np.float32)
    _, h_i, w_i = frames.shape
    ms_w, ms_h = max_shift_w, max_shift_h
    if np.percentile(template, 8) < -0.1:
        template = template - np.percentile(template, 1)
    template = template[ms_h : h_i - ms_h, ms_w : w_i - ms_w].astype(np.float32)

    shifts = np.empty((len(frames), 2))
    for i, frame in enumerate(frames):
        res = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
        top_left = cv2.minMaxLoc(res)[3]
        sh_y, sh_x = top_left
        if (0 < top_left[1] < 2 * ms_h - 1) & (0 < top_left[0] < 2 * ms_w - 1):
            log_xm1_y = np.log(res[sh_x - 1, sh_y])
            log_xp1_y = np.log(res[sh_x + 1, sh_y])
            log_x_ym1 = np.log(res[sh_x, sh_y - 1])
            log_x_yp1 = np.log(res[sh_x, sh_y + 1])
            four_log_xy = 4 * np.log(res[sh_x, sh_y])
            sh_x_n = -(sh_x - ms_h + ((log_xm1_y - log_xp1_y) / (2 * log_xm1_y - four_log_xy + 2 * log_xp1_y)))
            sh_y_n = -(sh_y - ms_w + ((log_x_ym1 - log_x_yp1) / (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1)))
        else:
            sh_x_n = -(sh_x - ms_h)
            sh_y_n = -(sh_y - ms_w)
        shifts[i] = sh_x_n, sh_y_n
    return shifts


def _apply_shifts_cubic(frames: NDArray, shifts: NDArray) -> NDArray:
    """Move each frame by its correction with cubic interpolation, for the template sample only.

    Edges are reflected and each frame is clipped to its own original range.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to shift.
    shifts : array, ``(T, 2)``
        Per frame, the ``(y, x)`` correction.

    Returns
    -------
    array, ``(T, H, W)`` float32
        The shifted frames (a copy).

    Notes
    -----
    CaImAn ``caiman/base/movies.py`` ``apply_shifts`` L308, ``interpolation='cubic'``.
    """
    out = np.array(frames, dtype=np.float32)
    _, h, w = out.shape
    for i, frame in enumerate(out):
        sh_x_n, sh_y_n = shifts[i]
        M = np.float32([[1, 0, sh_y_n], [0, 1, sh_x_n]])
        min_, max_ = np.min(frame), np.max(frame)
        warped = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        out[i] = np.clip(warped, min_, max_)
    return out


def _register_sample(frames: NDArray, max_shifts: tuple[int, int]) -> NDArray:
    """Register a sample of frames without a template, to build the first template.

    Three rounds: register the sample to its own median image, take the median of the
    result as a better template, register to that, take the median again, and register a
    last time. Shifts come from template matching (:func:`_extract_shifts`) and are
    applied with cubic interpolation (:func:`_apply_shifts_cubic`). The caller takes the
    median of the returned frames as the initial template.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        The sample (every ``T // 50 + 1``-th frame of the first epoch).
    max_shifts : tuple of int
        Largest shift searched for, ``(y, x)``.

    Returns
    -------
    array, ``(T, H, W)`` float32
        The registered sample.

    Notes
    -----
    CaImAn ``movie.motion_correct(max_shifts[1], max_shifts[0], template=None)``,
    ``caiman/base/movies.py`` L97. Very large frames are subsampled further, to at most
    ``1e8 / (H * W)`` frames, as there.
    """
    max_shift_h, max_shift_w = max_shifts
    T, H, W = frames.shape
    frames_to_skip = int(np.maximum(1, T / (10e7 / (H * W))))
    submov = frames[::frames_to_skip].copy()
    template = _bin_median(submov, clamp_window=False)
    submov = _apply_shifts_cubic(submov, _extract_shifts(submov, max_shift_w, max_shift_h, template))
    template = _bin_median(submov, clamp_window=False)
    registered = _apply_shifts_cubic(frames, _extract_shifts(frames, max_shift_w, max_shift_h, template))
    template = _bin_median(registered, clamp_window=False)
    return _apply_shifts_cubic(frames, _extract_shifts(frames, max_shift_w, max_shift_h, template))


def _dft(img: NDArray, real_input: bool) -> NDArray:
    """The scaled 2-D Fourier transform of an image, as CaImAn computes it with OpenCV.

    Parameters
    ----------
    img : array, ``(H, W)``
        The image.
    real_input : bool
        True transforms the real image directly (``register_translation``); False
        transforms it as a two-channel complex image (``apply_shifts_dft`` with
        ``is_freq=False``, L1704). The two agree to rounding; the choice keeps CaImAn's.

    Returns
    -------
    array, ``(H, W)`` complex128
        The spectrum, divided by ``H * W``.
    """
    flags = cv2.DFT_COMPLEX_OUTPUT + cv2.DFT_SCALE
    freq = cv2.dft(img if real_input else np.dstack([np.real(img), np.imag(img)]), flags=flags)
    return np.asarray(freq[:, :, 0] + 1j * freq[:, :, 1], dtype=np.complex128)


def _upsampled_dft(data: NDArray, upsampled_region_size: Any, upsample_factor: Any, axis_offsets: Any) -> NDArray:
    """The inverse DFT of ``data`` on a small, finely sampled region, by matrix multiplication.

    Used to refine a cross-correlation peak to sub-pixel precision without zero-padding the
    whole spectrum.

    Parameters
    ----------
    data : array, ``(H, W)`` complex
        The spectrum.
    upsampled_region_size : int or sequence of int
        Size of the output region, in upsampled pixels.
    upsample_factor : float
        How many output samples per original pixel.
    axis_offsets : sequence of float or None
        Where the region starts, in upsampled pixels.

    Returns
    -------
    array, complex
        The region, ``upsampled_region_size`` in each dimension.

    Notes
    -----
    CaImAn ``_upsampled_dft`` L1168, from scikit-image.
    """
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size] * data.ndim
    if axis_offsets is None:
        axis_offsets = [0] * data.ndim
    col_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[1] * upsample_factor))
        * (np.fft.ifftshift(np.arange(data.shape[1]))[:, None] - np.floor(data.shape[1] // 2)).dot(
            np.arange(upsampled_region_size[1])[None, :] - axis_offsets[1]
        )
    )
    row_kernel = np.exp(
        (-1j * 2 * np.pi / (data.shape[0] * upsample_factor))
        * (np.arange(upsampled_region_size[0])[:, None] - axis_offsets[0]).dot(
            np.fft.ifftshift(np.arange(data.shape[0]))[None, :] - np.floor(data.shape[0] // 2)
        )
    )
    output = np.tensordot(row_kernel, data, axes=[1, 0])
    return np.tensordot(output, col_kernel, axes=[1, 0])


def _register_translation_image(
    src_image: NDArray,
    target_image: NDArray,
    upsample_factor: int,
    max_shifts: tuple[int, int],
    shifts_lb: NDArray | None,
    shifts_ub: NDArray | None,
) -> tuple[NDArray, float]:
    """How far one image is translated relative to another, by phase correlation.

    The integer peak of the cross-correlation is found within the allowed range, then
    refined to ``1 / upsample_factor`` pixel with :func:`_upsampled_dft`.

    Parameters
    ----------
    src_image : array, ``(H, W)``
        The image to register (a frame or a patch of it).
    target_image : array, ``(H, W)``
        The template (or the same patch of it).
    upsample_factor : int
        Sub-pixel precision, in fractions of a pixel.
    max_shifts : tuple of int
        Largest shift searched for, ``(y, x)``, when no bounds are given.
    shifts_lb, shifts_ub : array of int, ``(2,)``, or None
        If given, the shift is searched for in ``[shifts_lb, shifts_ub]`` instead
        (CaImAn L1586-1600); used to keep patch shifts near the rigid one.

    Returns
    -------
    shift : array, ``(2,)``
        The ``(y, x)`` translation of ``src_image``; the correction is its negative.
    diffphase : float
        The global phase difference at the peak, used by the Fourier-domain shift.

    Notes
    -----
    CaImAn ``register_translation`` L1454; jnormcorre ``register_translation_jax_simple``
    L903; masknmf ``estimate_rigid_shifts`` L99.
    """
    src_freq = _dft(src_image, real_input=True)
    target_freq = _dft(target_image, real_input=True)

    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    image_product_cv = np.dstack([np.real(image_product), np.imag(image_product)])
    cross_correlation = cv2.dft(image_product_cv, flags=cv2.DFT_INVERSE + cv2.DFT_SCALE)
    cross_correlation = cross_correlation[:, :, 0] + 1j * cross_correlation[:, :, 1]

    new_cross_corr = np.abs(cross_correlation)
    if shifts_lb is not None and shifts_ub is not None:
        if (shifts_lb[0] < 0) and (shifts_ub[0] >= 0):
            new_cross_corr[shifts_ub[0] : shifts_lb[0], :] = 0
        else:
            new_cross_corr[: shifts_lb[0], :] = 0
            new_cross_corr[shifts_ub[0] :, :] = 0
        if (shifts_lb[1] < 0) and (shifts_ub[1] >= 0):
            new_cross_corr[:, shifts_ub[1] : shifts_lb[1]] = 0
        else:
            new_cross_corr[:, : shifts_lb[1]] = 0
            new_cross_corr[:, shifts_ub[1] :] = 0
    else:
        new_cross_corr[max_shifts[0] : -max_shifts[0], :] = 0
        new_cross_corr[:, max_shifts[1] : -max_shifts[1]] = 0

    maxima = np.unravel_index(np.argmax(new_cross_corr), cross_correlation.shape)
    midpoints = np.array([np.trunc(axis_size // 2) for axis_size in shape])
    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        CCmax = cross_correlation.max()
    else:
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        dftshift = np.trunc(upsampled_region_size / 2.0)
        factor = np.array(upsample_factor, dtype=np.float64)
        normalization = src_freq.size * factor**2
        sample_region_offset = dftshift - shifts * factor
        cross_correlation = _upsampled_dft(
            image_product.conj(), upsampled_region_size, factor, sample_region_offset
        ).conj()
        cross_correlation /= normalization
        maxima = np.array(
            np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape), dtype=np.float64
        )
        maxima -= dftshift
        shifts = shifts + (maxima / factor)
        CCmax = cross_correlation.max()

    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0
    return shifts, float(np.arctan2(CCmax.imag, CCmax.real))


def _register_translation(
    frames: NDArray,
    template: NDArray,
    max_shifts: tuple[int, int],
    upsample_factor: int,
    shifts_lb: NDArray | None = None,
    shifts_ub: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """The rigid translation of every frame relative to the template.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to register.
    template : array, ``(H, W)``
        The template.
    max_shifts : tuple of int
        Largest shift searched for, ``(y, x)``.
    upsample_factor : int
        Sub-pixel precision, in fractions of a pixel.
    shifts_lb, shifts_ub : array of int, ``(2,)``, optional
        Search range replacing ``max_shifts``; see :func:`_register_translation_image`.

    Returns
    -------
    shifts : array, ``(T, 2)``
        Per frame, its ``(y, x)`` translation (CaImAn ``rigid_shts``); the correction is the
        negative.
    diffphase : array, ``(T,)``
        Per frame, the phase difference at the correlation peak.

    Notes
    -----
    CaImAn ``register_translation`` L1454, frame by frame.
    """
    shifts = np.empty((len(frames), 2))
    diffphase = np.empty(len(frames))
    for t, img in enumerate(frames):
        shifts[t], diffphase[t] = _register_translation_image(
            img, template, upsample_factor, max_shifts, shifts_lb, shifts_ub
        )
    return shifts, diffphase


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
    """The ``bounds`` region of the frames (CaImAn ``m[:, indices[0], indices[1]]``)."""

    if settings.bounds is None:
        return frames
    (y0, y1), (x0, x1) = settings.bounds
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
        y0 = settings.bounds[0][0] or 0 if settings.bounds is not None else 0
        x0 = settings.bounds[1][0] or 0 if settings.bounds is not None else 0
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
        ``bounds`` crops the estimation region.
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
        cropped = settings.bounds is not None and settings.bounds != ((None, None), (None, None))
        interpolate = settings.shifts_interpolate or cropped
        patches = epoch_patches[p]
        assert patches is not None
        return _apply_pw_shifts_remap(
            plane_video, patches[start_frame:end_frame], centers, settings.border_nan, interpolate
        )


# ``register_registration_class("normcorre", RegisterNormcorreImagingEpoch)`` is added
# by the PR that implements the apply kernels.
register_motion_class(NormcorreMotion)
