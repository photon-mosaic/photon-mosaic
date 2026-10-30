"""NoRMCorre motion correction, reimplemented to follow CaImAn step by step.

NoRMCorre (Pnevmatikakis & Giovannucci, 2017) first corrects each frame with one rigid
translation, estimated against a template that is refined as it goes. With
``nonrigid=True`` it then splits the field of view into overlapping patches, finds a
shift for each patch (allowed to deviate from the rigid shift by at most
``max_nonrigid_shift`` pixels) and warps the frame with the smooth field those shifts
define. Beyond numpy and scipy it needs only OpenCV, which photon-mosaic already installs.

Typical use::

    from photon_mosaic.preprocessing import compute_motion, register_motion

    motion = compute_motion(imaging, method="normcorre", nonrigid=True)
    registered = register_motion(imaging, motion, method="normcorre")

``compute_motion`` estimates the shifts and keeps the final template as
``motion.reference``; ``register_motion`` applies them lazily, as frames are read.

Every step reproduces CaImAn v1.13.2 (``240e1f2``); the line numbers in the docstrings
below cite that version, so each function can be checked against its original.
Internally, as in CaImAn, shifts are *corrections* (how far to move a frame back onto the
template); what :class:`NormcorreMotion` stores is the opposite, the frame's motion, which
is suite2p's convention for ``displacements``.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal, Sequence

import cv2
import numpy as np
import scipy.interpolate
from numpy.typing import NDArray
from pydantic import ConfigDict, Field

from photon_mosaic.core import (
    BaseImaging,
    BaseImagingEpoch,
    Motion,
    register_motion_class,
    register_registration_class,
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


def _patch_edges(dims: Sequence[int], overlaps: Sequence[int], strides: Sequence[int]) -> tuple[list[int], ...]:
    """Where patches start along each dimension.

    Patches of ``strides + overlaps`` pixels start every ``strides`` pixels; the last one
    is placed flush with the far edge, so it may overlap its neighbour more.

    Parameters
    ----------
    dims : sequence of int
        Image shape.
    overlaps, strides : sequence of int
        Patch overlap and stride per dimension.

    Returns
    -------
    tuple of lists of int
        Per dimension, the first pixel of each patch.

    Notes
    -----
    CaImAn ``get_patch_edges`` L1783.
    """
    window_sizes = np.add(overlaps, strides)
    return tuple(
        list(range(0, dim - size, stride)) + [dim - size] for dim, size, stride in zip(dims, window_sizes, strides)
    )


def _patch_centers(dims: Sequence[int], overlaps: Sequence[int], strides: Sequence[int]) -> tuple[NDArray, NDArray]:
    """The centre of each patch along each dimension, in pixels.

    Parameters
    ----------
    dims : sequence of int
        Image shape.
    overlaps, strides : sequence of int
        Patch overlap and stride per dimension.

    Returns
    -------
    cy, cx : array
        Row centres of the patch rows and column centres of the patch columns.

    Notes
    -----
    CaImAn ``get_patch_centers`` L1793.
    """
    window_sizes = np.add(overlaps, strides)
    edges = _patch_edges(dims, overlaps, strides)
    cy, cx = (np.array([edge + (size - 1) / 2 for edge in dim_edges]) for dim_edges, size in zip(edges, window_sizes))
    return cy, cx


def _sliding_window(
    image: NDArray, overlaps: Sequence[int], strides: Sequence[int]
) -> list[tuple[int, int, int, int, NDArray]]:
    """Cut an image into overlapping patches.

    Parameters
    ----------
    image : array, ``(H, W)``
        The image.
    overlaps, strides : sequence of int
        Patch overlap and stride, ``(y, x)``.

    Returns
    -------
    list of tuples
        Per patch, in row-major order: its grid row and column, its top-left pixel
        ``(y0, x0)`` and the patch itself (a view).

    Notes
    -----
    CaImAn ``sliding_window`` L1839.
    """
    size = np.add(overlaps, strides)
    edges = _patch_edges(image.shape, overlaps, strides)
    return [
        (iy, ix, y0, x0, image[y0 : y0 + size[0], x0 : x0 + size[1]])
        for (iy, y0), (ix, x0) in itertools.product(enumerate(edges[0]), enumerate(edges[1]))
    ]


def _interpolate_shifts(
    shifts: NDArray, coords_orig: tuple[NDArray, NDArray], coords_new: tuple[NDArray, NDArray], method: str = "linear"
) -> NDArray:
    """Interpolate values given on the patch grid at new coordinates.

    Coordinates outside the grid are clipped to it, so the outermost values extend to the
    image edge.

    Parameters
    ----------
    shifts : array
        One value per patch, reshapeable to ``(len(cy), len(cx))``.
    coords_orig : tuple of arrays
        The patch centres ``(cy, cx)`` the values belong to.
    coords_new : tuple of arrays
        Row and column coordinates to interpolate at.
    method : str
        Any ``scipy.interpolate.interpn`` method.

    Returns
    -------
    array, ``(len(coords_new[0]), len(coords_new[1]))``
        The interpolated values.

    Notes
    -----
    CaImAn ``interpolate_shifts`` L1899.
    """
    clipped = [np.clip(new, min(orig), max(orig)) for new, orig in zip(coords_new, coords_orig)]
    stacked = np.stack(np.meshgrid(*clipped, indexing="ij"), axis=-1)
    grid = np.reshape(shifts, tuple(len(orig) for orig in coords_orig))
    return scipy.interpolate.interpn(coords_orig, grid, stacked, method=method)


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
    """The translation of every patch of every frame, kept close to the frame's rigid translation.

    Each patch is registered to the same patch of the template by phase correlation, with
    the search limited to ``ceil(rigid - max_nonrigid_shift)`` to
    ``floor(rigid + max_nonrigid_shift)``.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to register.
    template : array, ``(H, W)``
        The template.
    strides, overlaps : tuple of int
        Patch layout, ``(y, x)``.
    max_shifts : tuple of int
        Passed on; unused when bounds are set, as here.
    max_nonrigid_shift : int
        Largest deviation of a patch from the frame's rigid translation, in pixels.
    upsample_factor : int
        Sub-pixel precision, in fractions of a pixel.
    rigid_shifts : array, ``(T, 2)``
        The frames' rigid translations, from :func:`_register_translation`.

    Returns
    -------
    shifts : array, ``(T, ny, nx, 2)``
        Per frame and patch, its ``(y, x)`` translation; the correction is the negative.
    diffphase : array, ``(T, ny, nx)``
        Per frame and patch, the phase difference at the correlation peak.
    centers : tuple of arrays
        The patch centres ``(cy, cx)``.

    Notes
    -----
    CaImAn ``tile_and_correct`` L2013, piecewise branch; jnormcorre
    ``_register_to_template_pwrigid`` L1556; masknmf ``_estimate_patchwise_rigid_shifts`` L614.
    """
    templates = [patch for *_, patch in _sliding_window(template, overlaps, strides)]
    grid = tuple(len(edges) for edges in _patch_edges(template.shape, overlaps, strides))
    shifts = np.empty((len(frames), *grid, 2))
    diffphase = np.empty((len(frames), *grid))
    for t, img in enumerate(frames):
        lb = np.ceil(np.subtract(rigid_shifts[t], max_nonrigid_shift)).astype(int)
        ub = np.floor(np.add(rigid_shifts[t], max_nonrigid_shift)).astype(int)
        for (iy, ix, _, _, patch), tpatch in zip(_sliding_window(img, overlaps, strides), templates):
            shifts[t, iy, ix], diffphase[t, iy, ix] = _register_translation_image(
                patch, tpatch, upsample_factor, max_shifts, lb, ub
            )
    return shifts, diffphase, _patch_centers(template.shape, overlaps, strides)


def _fill_border(img: NDArray, shift: Sequence[float], border_nan: BorderNan, min_value: float) -> NDArray:
    """Fill, in place, the strip at the frame edge that a shift left without data.

    Parameters
    ----------
    img : array, ``(H, W)``
        The shifted image; modified in place.
    shift : sequence of float
        The ``(y, x)`` correction that was applied; its rounded-up size sets the strip width.
    border_nan : bool or {'copy', 'min'}
        True fills NaN, 'min' fills ``min_value``, 'copy' repeats the nearest valid row or
        column, False leaves the strip as it is.
    min_value : float
        The fill for 'min'.

    Returns
    -------
    array, ``(H, W)``
        ``img``.

    Notes
    -----
    CaImAn ``apply_shift_iteration`` L522 and ``apply_shifts_dft`` L1733.
    """
    if border_nan is False:
        return img
    max_h, max_w = np.ceil(np.maximum((0, 0), shift[:2])).astype(int)
    min_h, min_w = np.floor(np.minimum((0, 0), shift[:2])).astype(int)
    if border_nan == "copy":
        if max_h > 0:
            img[:max_h] = img[max_h]
        if min_h < 0:
            img[min_h:] = img[min_h - 1]
        if max_w > 0:
            img[:, :max_w] = img[:, max_w, np.newaxis]
        if min_w < 0:
            img[:, min_w:] = img[:, min_w - 1, np.newaxis]
        return img
    fill = np.nan if border_nan is True else min_value
    img[:max_h, :] = fill
    if min_h < 0:
        img[min_h:, :] = fill
    img[:, :max_w] = fill
    if min_w < 0:
        img[:, min_w:] = fill
    return img


def _apply_shift_iteration(frames: NDArray, shifts: NDArray, border_nan: BorderNan) -> NDArray:
    """Move each frame by its rigid correction, with cubic interpolation.

    The default way rigid shifts are applied (``shifts_opencv=True``). Edges are
    reflected, each frame is clipped to its original range, and the exposed strip is then
    filled as ``border_nan`` says.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to shift; float32 or float64, kept.
    shifts : array, ``(T, 2)``
        Per frame, the ``(y, x)`` correction.
    border_nan : bool or {'copy', 'min'}
        See :func:`_fill_border`.

    Returns
    -------
    array, ``(T, H, W)``
        The shifted frames.

    Notes
    -----
    CaImAn ``apply_shift_iteration`` L512.
    """
    out = []
    for img, shift in zip(frames, shifts):
        h_i, w_i = img.shape
        M = np.array([[1, 0, shift[1]], [0, 1, shift[0]]], dtype=np.float32)
        min_, max_ = np.nanmin(img), np.nanmax(img)
        warped = cv2.warpAffine(img, M, (w_i, h_i), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        out.append(_fill_border(np.clip(warped, min_, max_), shift, border_nan, min_))
    return np.stack(out)


def _shift_dft_image(src_freq: NDArray, shift: Sequence[float], diffphase: float, border_nan: BorderNan) -> NDArray:
    """Move one image by a correction in the Fourier domain.

    Multiplies the spectrum by a phase ramp, transforms back and fills the exposed strip.
    Edges wrap around before the fill.

    Parameters
    ----------
    src_freq : array, ``(H, W)`` complex
        The image's spectrum, from :func:`_dft`.
    shift : sequence of float
        The ``(y, x)`` correction.
    diffphase : float
        Global phase from :func:`_register_translation_image`; 0 when replaying.
    border_nan : bool or {'copy', 'min'}
        See :func:`_fill_border`; 'min' uses the shifted image's minimum.

    Returns
    -------
    array, ``(H, W)`` float
        The shifted image.

    Notes
    -----
    CaImAn ``apply_shifts_dft`` L1654, 2-D branch.
    """
    nr, nc = src_freq.shape
    Nr = np.fft.ifftshift(np.arange(-np.trunc(nr / 2.0), np.ceil(nr / 2.0)))
    Nc = np.fft.ifftshift(np.arange(-np.trunc(nc / 2.0), np.ceil(nc / 2.0)))
    Nc, Nr = np.meshgrid(Nc, Nr)
    Greg = src_freq * np.exp(1j * 2 * np.pi * (-shift[0] * Nr / nr - shift[1] * Nc / nc))
    Greg = Greg.dot(np.exp(1j * diffphase))
    new_img = cv2.idft(np.dstack([np.real(Greg), np.imag(Greg)]))[:, :, 0]
    min_value = np.nanmin(new_img) if border_nan == "min" else 0.0
    return _fill_border(new_img, shift, border_nan, min_value)


def _apply_shifts_dft(frames: NDArray, shifts: NDArray, diffphase: NDArray | None, border_nan: BorderNan) -> NDArray:
    """Move each frame by its rigid correction in the Fourier domain (``shifts_opencv=False``).

    Exact for whole and fractional pixels alike, but edges wrap around before the
    ``border_nan`` fill.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to shift.
    shifts : array, ``(T, 2)``
        Per frame, the ``(y, x)`` correction.
    diffphase : array, ``(T,)``, or None
        Per-frame phases from estimation, which then uses the spectrum as
        ``register_translation`` computes it; None when replaying stored shifts, which
        CaImAn does with ``diffphase=0, is_freq=False``.
    border_nan : bool or {'copy', 'min'}
        See :func:`_fill_border`.

    Returns
    -------
    array, ``(T, H, W)`` float64
        The shifted frames.

    Notes
    -----
    CaImAn ``apply_shifts_dft`` L1654; jnormcorre ``apply_shifts_dft_fast_1`` L1219;
    masknmf ``apply_rigid_shifts`` L45.
    """
    if diffphase is None:
        out = [_shift_dft_image(_dft(img, real_input=False), sh, 0.0, border_nan) for img, sh in zip(frames, shifts)]
    else:
        out = [
            _shift_dft_image(_dft(img, real_input=True), sh, ph, border_nan)
            for img, sh, ph in zip(frames, shifts, diffphase)
        ]
    return np.stack(out)


def _apply_pw_shifts_remap(
    frames: NDArray,
    patch_shifts: NDArray,
    patch_centers: tuple[NDArray, NDArray],
    border_nan: BorderNan,
    shifts_interpolate: bool,
) -> NDArray:
    """Warp each frame by a smooth per-pixel field built from its patch corrections.

    The patch grid of corrections is expanded to one correction per pixel, either by
    interpolating between patch centres (``shifts_interpolate``) or by stretching the
    grid over the whole frame; each pixel is then sampled, with cubic interpolation, from
    where its correction says it came from. Used for non-rigid shifts with
    ``shifts_opencv=True``, and always when replaying non-rigid shifts.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to warp.
    patch_shifts : array, ``(T, ny, nx, 2)``
        Per frame and patch, the ``(y, x)`` correction.
    patch_centers : tuple of arrays
        The patch centres ``(cy, cx)``, in the frames' pixels.
    border_nan : bool or {'copy', 'min'}
        Value for pixels sampled from outside the frame: 'copy' repeats the edge, 'min' the
        frame minimum, True NaN, False zero.
    shifts_interpolate : bool
        Interpolate between patch centres instead of stretching the grid.

    Returns
    -------
    array, ``(T, H, W)``
        The warped frames.

    Notes
    -----
    CaImAn ``apply_pw_shifts_remap_2d`` L2533 and ``interpolate_shifts`` L1899; masknmf
    ``apply_displacement_vector_field`` L465. For float64 frames, OpenCV rounds a constant
    border value next to the frame edge; replay uses float32, which it does not.
    """
    dims = frames.shape[1:]
    x_coords, y_coords = (np.arange(0.0, dims[dim]).astype(np.float32) for dim in (1, 0))
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    out = []
    for img, correction in zip(frames, patch_shifts):
        shift_img_y, shift_img_x = -correction[..., 0], -correction[..., 1]
        if shifts_interpolate:
            shifts_y = _interpolate_shifts(shift_img_y, patch_centers, (y_coords, x_coords)).astype(np.float32)
            shifts_x = _interpolate_shifts(shift_img_x, patch_centers, (y_coords, x_coords)).astype(np.float32)
        else:
            shifts_y = cv2.resize(shift_img_y.astype(np.float32), dims[::-1])
            shifts_x = cv2.resize(shift_img_x.astype(np.float32), dims[::-1])
        if border_nan == "copy":
            mode, value = cv2.BORDER_REPLICATE, 0.0
        else:
            # float(): CaImAn passes the numpy scalar, which OpenCV >= 5 rejects as a border value
            mode, value = cv2.BORDER_CONSTANT, {False: 0.0, True: np.nan, "min": float(np.min(img))}[border_nan]
        out.append(
            cv2.remap(img, shifts_x + x_grid, shifts_y + y_grid, cv2.INTER_CUBIC, borderMode=mode, borderValue=value)
        )
    return np.stack(out)


def _blending_weights(image: NDArray, overlaps: Sequence[int], strides: Sequence[int]) -> list[NDArray]:
    """Per patch, weights that fade linearly to zero across the overlaps it shares.

    Parameters
    ----------
    image : array, ``(H, W)``
        Any image of the frame shape.
    overlaps, strides : sequence of int
        Patch overlap and stride, ``(y, x)``.

    Returns
    -------
    list of arrays
        One ``(strides + overlaps)`` weight map per patch, in :func:`_sliding_window` order.

    Notes
    -----
    CaImAn ``create_weight_matrix_for_blending`` L1929.
    """
    shapes = np.add(strides, overlaps)
    windows = _sliding_window(image, overlaps, strides)
    max_grid_1, max_grid_2 = np.max(np.array([w[:2] for w in windows]), 0)
    weights = []
    for grid_1, grid_2, *_ in windows:
        weight_mat = np.ones(shapes)
        if grid_1 > 0:
            weight_mat[: overlaps[0], :] = np.linspace(0, 1, overlaps[0])[:, None]
        if grid_1 < max_grid_1:
            weight_mat[-overlaps[0] :, :] = np.linspace(1, 0, overlaps[0])[:, None]
        if grid_2 > 0:
            weight_mat[:, : overlaps[1]] = weight_mat[:, : overlaps[1]] * np.linspace(0, 1, overlaps[1])[None, :]
        if grid_2 < max_grid_2:
            weight_mat[:, -overlaps[1] :] = weight_mat[:, -overlaps[1] :] * np.linspace(1, 0, overlaps[1])[None, :]
        weights.append(weight_mat)
    return weights


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
    """Apply non-rigid shifts in the Fourier domain (``shifts_opencv=False``), patch by patch.

    The patch grid is made ``upsample_factor_grid`` times finer (strides
    ``round(strides / upsample_factor_grid)``, same overlaps), the shifts and phases are
    brought onto it (interpolated from the centres if ``shifts_interpolate``, else
    resized), each fine patch is shifted in the Fourier domain, and the patches are put
    back together: blended with linear weights where they overlap, or cut at the middle of
    each overlap when neighbouring shifts differ by 0.5 px or more.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        Frames to correct.
    patch_shifts : array, ``(T, ny, nx, 2)``
        Per frame and patch, the ``(y, x)`` translation from :func:`_register_patches`
        (not yet negated).
    patch_diffphase : array, ``(T, ny, nx)``
        The matching phase differences.
    patch_centers : tuple of arrays
        Centres ``(cy, cx)`` of the estimation patches.
    strides, overlaps : tuple of int
        Layout of the estimation patches, ``(y, x)``.
    upsample_factor_grid : int
        How much finer the grid of applied patches is.
    shifts_interpolate : bool
        Interpolate from the centres instead of resizing the grid.
    border_nan : bool or {'copy', 'min'}
        See :func:`_fill_border`, applied per fine patch.

    Returns
    -------
    corrected : array, ``(T, H, W)``
        The corrected frames.
    corrections : array, ``(T, nfy, nfx, 2)``
        Per frame, the correction of every fine patch.
    centers : tuple of arrays
        The fine patches' centres ``(cy, cx)``.

    Notes
    -----
    CaImAn ``tile_and_correct`` L2013, FFT branch, and ``create_weight_matrix_for_blending``
    L1929.
    """
    dims = frames.shape[1:]
    newoverlaps = overlaps
    newstrides = tuple(int(s) for s in np.round(np.divide(strides, upsample_factor_grid)).astype(int))
    newshapes = np.add(newstrides, newoverlaps)
    fine_centers = _patch_centers(dims, newoverlaps, newstrides)
    weights = _blending_weights(frames[0], newoverlaps, newstrides)
    half_overlap_x, half_overlap_y = int(newoverlaps[0] / 2), int(newoverlaps[1] / 2)

    corrected, corrections = [], []
    for img, raw, phase in zip(frames, patch_shifts, patch_diffphase):
        windows = _sliding_window(img, newoverlaps, newstrides)
        dim_new_grid = (windows[-1][0] + 1, windows[-1][1] + 1)
        num_tiles = len(windows)
        # CaImAn's shift_img_x is the first (row) axis
        shift_img_x, shift_img_y = raw[..., 0], raw[..., 1]
        if shifts_interpolate:
            shift_img_x = _interpolate_shifts(shift_img_x, patch_centers, fine_centers, method="cubic")
            shift_img_y = _interpolate_shifts(shift_img_y, patch_centers, fine_centers, method="cubic")
            phase_us = _interpolate_shifts(phase, patch_centers, fine_centers, method="cubic")
        else:
            shift_img_x = cv2.resize(shift_img_x, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
            shift_img_y = cv2.resize(shift_img_y, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)
            phase_us = cv2.resize(phase, dim_new_grid[::-1], interpolation=cv2.INTER_CUBIC)

        max_shear = np.percentile(
            [np.max(np.abs(np.diff(s, axis=ax))) for s, ax in itertools.product([shift_img_x, shift_img_y], [0, 1])],
            75,
        )
        total_shifts = np.stack([-shift_img_x.reshape(num_tiles), -shift_img_y.reshape(num_tiles)], axis=-1)
        imgs = [
            _shift_dft_image(_dft(patch, real_input=False), sh, ph, border_nan)
            for (*_, patch), sh, ph in zip(windows, total_shifts, phase_us.reshape(num_tiles))
        ]

        normalizer = np.zeros_like(img) * np.nan
        new_img = np.zeros_like(img) * np.nan
        if max_shear < 0.5:
            for (_, _, x, y, _), im, weight_mat in zip(windows, imgs, weights):
                prev_val_1 = normalizer[x : x + newshapes[0], y : y + newshapes[1]]
                normalizer[x : x + newshapes[0], y : y + newshapes[1]] = np.nansum(
                    np.dstack([~np.isnan(im) * 1 * weight_mat, prev_val_1]), -1
                )
                prev_val = new_img[x : x + newshapes[0], y : y + newshapes[1]]
                new_img[x : x + newshapes[0], y : y + newshapes[1]] = np.nansum(
                    np.dstack([im * weight_mat, prev_val]), -1
                )
            new_img = new_img / normalizer
        else:
            # neighbouring shifts differ by >= 0.5 px: cut the overlaps at the middle instead of blending
            for (idx_0, idx_1, x, y, _), im in zip(windows, imgs):
                x_start = x if idx_0 == 0 else x + half_overlap_x
                y_start = y if idx_1 == 0 else y + half_overlap_y
                new_img[x_start : x + newshapes[0], y_start : y + newshapes[1]] = im[x_start - x :, y_start - y :]
        corrected.append(new_img)
        corrections.append(total_shifts.reshape(*dim_new_grid, 2))
    return np.stack(corrected), np.stack(corrections), fine_centers


# --------------------------------------------------------------------------- #
# The driver: CaImAn's MotionCorrect over photon-mosaic epochs and planes.
# --------------------------------------------------------------------------- #


def _read_plane(epoch: BaseImagingEpoch, plane_index: int, start: int, end: int, step: int = 1) -> NDArray:
    """Frames ``start:end:step`` of one plane of an epoch, as ``(n, H, W)`` float32."""

    video = epoch.get_series(start, end)[::step]
    plane = video[:, :, :, plane_index] if video.ndim == 4 else video
    return np.asarray(plane, dtype=np.float32)


def _crop(frames: NDArray, settings: NormcorreRegistrationSettings) -> NDArray:
    """The part of ``(T, H, W)`` frames used for estimation: ``settings.bounds``, or all of it."""

    if settings.bounds is None:
        return frames
    (y0, y1), (x0, x1) = settings.bounds
    return frames[:, y0:y1, x0:x1]


def _tile_and_correct(
    frames: NDArray, template: NDArray, add_to_movie: float, settings: NormcorreRegistrationSettings
) -> tuple[NDArray, NDArray, NDArray | None, tuple[NDArray, NDArray] | None]:
    """Register a chunk of frames to the template and correct them.

    Estimation runs on the frames plus ``add_to_movie`` (and high-passed if ``gSig_filt``);
    the rigid translation is always estimated, the patch translations only when
    ``settings.nonrigid``. The corrections are then applied as the settings say.

    Parameters
    ----------
    frames : array, ``(T, H, W)``
        The chunk, already cropped to ``bounds``.
    template : array, ``(H, W)``
        The current template.
    add_to_movie : float
        Offset that makes the movie non-negative (``-min_mov``).
    settings : NormcorreRegistrationSettings
        The backend settings.

    Returns
    -------
    corrected : array, ``(T, H, W)``
        The corrected frames, used to build the next template.
    rigid : array, ``(T, 2)``
        Per frame, the rigid ``(y, x)`` correction.
    patches : array, ``(T, ny, nx, 2)``, or None
        Non-rigid only: per frame, the correction of every patch (on the finer grid for
        the Fourier-domain path).
    centers : tuple of arrays or None
        Non-rigid only: the centres of those patches.

    Notes
    -----
    CaImAn ``tile_and_correct`` L2013, vectorised over frames.
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
    """A chunk's template: the mean of its corrected frames, with NaN pixels set to the minimum.

    Notes
    -----
    CaImAn ``tile_and_correct_wrapper`` L3066, which accumulates in float32.
    """

    template = np.nanmean(corrected.astype(np.float32), axis=0)
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
    """One registration pass over an epoch, and the template it produces.

    The epoch is cut into ``n_splits`` contiguous chunks; each is registered to
    ``template`` and corrected, and the new template is the median of the chunks' mean
    images (high-passed if ``gSig_filt``). The shifts returned are measured against the
    template the pass started from, as CaImAn's are.

    Parameters
    ----------
    epoch : BaseImagingEpoch
        The epoch.
    plane_index : int
        The plane to register.
    template : array, ``(H, W)``
        Template to register to.
    add_to_movie : float
        Offset that makes the movie non-negative.
    n_splits : int
        Number of chunks.
    settings : NormcorreRegistrationSettings
        The backend settings.

    Returns
    -------
    template : array, ``(H, W)``
        The new template.
    rigid : array, ``(frames, 2)``
        Per frame, the rigid correction.
    patches : array, ``(frames, ny, nx, 2)``, or None
        Non-rigid only: per frame, the patch corrections.
    centers : tuple of arrays or None
        Non-rigid only: the patch centres, in cropped-frame pixels.

    Notes
    -----
    CaImAn ``motion_correction_piecewise`` L3144.
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
    """The motion NoRMCorre estimated, returned by ``compute_motion(..., method="normcorre")``.

    Holds only arrays and a dump of the settings used, so it pickles. Shifts are stored in
    pixels, ``(y, x)``, as the *motion* of each frame (or patch) relative to the template,
    suite2p's convention for ``displacements``: the correction applied on registration is
    the negative. CaImAn stores the corrections, so CaImAn's ``shifts_rig`` is
    ``-displacements`` and its ``x_shifts_els``/``y_shifts_els`` are ``-patch_shifts``.

    Attributes
    ----------
    displacements : list of arrays
        Per epoch, the rigid motion of every frame, ``(frames, planes, 2)``.
    reference : list of arrays
        Per plane, the final template every frame was registered to (cropped to ``bounds``).
    patch_shifts : list or None
        Non-rigid runs only: ``patch_shifts[epoch][plane]`` is the motion of every patch,
        ``(frames, ny, nx, 2)``.
    patch_centers : list or None
        Non-rigid runs only: per plane, the patch centres ``(cy, cx)`` in full-frame pixels.
    settings : dict
        The :class:`NormcorreRegistrationSettings` used, as a dict.
    metadata : dict
        ``"border_to_0"``: the largest shift, rounded up, i.e. how many pixels at each edge
        may hold filled-in values (CaImAn ``border_to_0``).
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
        """Estimate the motion of every plane, treating epochs as CaImAn treats a list of files.

        For each plane: the offset ``min_mov`` comes from the first 400 frames of the first
        epoch. The first template is the median of a registered sample of every
        ``T // 50 + 1``-th frame of the first epoch. Each epoch then gets ``niter_rig`` rigid
        passes (:func:`_correct_epoch`), and its rigid shifts are those of the last pass; the
        template carries over to the next epoch. With ``nonrigid``, one more pass per epoch
        estimates patch shifts, starting from the final rigid template.

        Parameters
        ----------
        imaging : BaseImaging
            The movie.
        settings : NormcorreRegistrationSettings, optional
            Base settings; defaults, plus environment variables, if omitted.
        badframes : None
            Must be None: CaImAn has no bad-frame mechanism.
        **params
            Settings overriding ``settings``.

        Returns
        -------
        NormcorreMotion

        Raises
        ------
        ValueError
            Where CaImAn raises: bad frames given, ``gSig_filt`` with ``shifts_opencv=False``,
            ``nonrigid`` with ``max_nonrigid_shift=0``, NaNs in the movie or the template.

        Notes
        -----
        CaImAn ``MotionCorrect.motion_correct`` L222, ``motion_correct_rigid`` L270,
        ``motion_correct_batch_rigid`` L2782 and ``motion_correct_pwrigid`` L326.
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

        # Everything above is CaImAn's corrections; store motion (their negative), suite2p's convention
        # for displacements, so that every backend's displacements mean the same thing.
        displacements = [-np.stack(epoch_shifts, axis=1) for epoch_shifts in rigid]
        patch_motion = [[-x if x is not None else None for x in epoch_patches] for epoch_patches in patches]
        moved = [np.abs(d).max() for d in displacements]
        if settings.nonrigid:
            moved = [max(np.abs(x).max() for x in epoch_patches if x is not None) for epoch_patches in patches]
        return cls(
            imaging=imaging,
            displacements=displacements,
            reference=templates,
            patch_shifts=patch_motion if settings.nonrigid else None,
            patch_centers=centers if settings.nonrigid else None,
            settings=settings.model_dump(),
            metadata={"border_to_0": int(np.ceil(max(moved)))},
        )


class RegisterNormcorreImagingEpoch(RegisterImagingEpoch):
    """One epoch of a registered movie: applies stored NoRMCorre shifts as frames are read.

    Not built directly: ``register_motion(imaging, motion, method="normcorre")`` makes one
    per epoch. Rigid shifts are applied as they were estimated (``shifts_opencv``);
    non-rigid shifts always by cubic interpolation of the per-pixel field, as CaImAn's
    ``apply_shifts_movie`` (L399). Frame bounds, plane selection and output allocation come
    from :class:`~photon_mosaic.preprocessing.registration.RegisterImagingEpoch`.
    """

    motion: NormcorreMotion

    def _correct_plane(
        self,
        plane_video: NDArray[np.floating[Any]],
        plane_index: int,
        start_frame: int,
        end_frame: int,
    ) -> NDArray[np.floating[Any]]:
        """Correct frames ``start_frame:end_frame`` of one plane.

        Rigid motion is undone with the cubic warp or the Fourier shift, as estimated.
        Non-rigid motion is undone with the per-pixel remap on the full frame, interpolated
        from the patch centres whenever ``shifts_interpolate`` is set or ``bounds`` cropped
        the estimation region, as CaImAn does.

        Parameters
        ----------
        plane_video : array, ``(n_frames, H, W)`` float32
            The raw frames.
        plane_index : int
            Which plane they belong to.
        start_frame, end_frame : int
            Their range within the epoch.

        Returns
        -------
        array, ``(n_frames, H, W)``
            The registered frames.
        """

        settings = NormcorreRegistrationSettings.model_validate(self.motion.settings)
        p = plane_index
        epoch_patches = self.motion.patch_shifts[self.epoch_index] if self.motion.patch_shifts is not None else None
        # stored shifts are motion; the kernels apply corrections, their negative
        if epoch_patches is None:
            correction = -self.motion.displacements[self.epoch_index][start_frame:end_frame, p]
            if settings.shifts_opencv:
                return _apply_shift_iteration(plane_video, correction, settings.border_nan)
            return _apply_shifts_dft(plane_video, correction, None, settings.border_nan)

        centers = self.motion.patch_centers[p] if self.motion.patch_centers is not None else None
        assert centers is not None, "piecewise motion without patch centres"
        cropped = settings.bounds is not None and settings.bounds != ((None, None), (None, None))
        interpolate = settings.shifts_interpolate or cropped
        patches = epoch_patches[p]
        assert patches is not None
        return _apply_pw_shifts_remap(
            plane_video, -patches[start_frame:end_frame], centers, settings.border_nan, interpolate
        )


register_motion_class(NormcorreMotion)
register_registration_class("normcorre", RegisterNormcorreImagingEpoch)
