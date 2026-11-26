import numpy as np
import functools
import logging
import warnings
from skimage.exposure import rescale_intensity
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
from scipy.fft import dctn, idctn
import skimage.transform as transform

from ._conventions import robust_range, warn_deprecated

logger = logging.getLogger(__name__)


def _float_to_int(arr, dtype):
    """Round floats and cast them to an integer dtype, clipping to its range (NaN -> 0).

    float(np.iinfo(int64).max) rounds up to 2**63, so a plain clip-then-cast can
    still overflow and wrap to the minimum; the upper bound stays strictly inside.
    """
    info = np.iinfo(dtype)
    hi = np.float64(info.max)
    if int(hi) > info.max:
        hi = np.nextafter(hi, -np.inf)
    out = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0)
    return np.clip(np.round(out), np.float64(info.min), hi).astype(dtype)


def ensure_float_and_restore_dtype(func):
    """
    Decorator to safely convert integer arrays to normalized floats for processing,
    and restore the original integer dtype upon completion.

    - 8/16-bit integers are divided by the dtype maximum (unsigned -> [0, 1]; signed
      -> [-1, 1] with 0 kept at 0, as skimage's img_as_float) and processed in float32.
    - Wider integers (32/64-bit) are divided by the data's own maximum magnitude and
      processed in float64: dividing by e.g. 2**63 - 1 would leave ~1e-17 values that
      float32 cannot resolve and that make scale-dependent operations meaningless.
    - float32/float64 inputs pass through unchanged (other floats become float32).
    """
    @functools.wraps(func)
    def wrapper(image, *args, **kwargs):
        if not isinstance(image, np.ndarray):
            return func(image, *args, **kwargs)

        orig_dtype = image.dtype
        is_integer = np.issubdtype(orig_dtype, np.integer)

        if is_integer:
            info = np.iinfo(orig_dtype)
            valid_min, valid_max = info.min, info.max
            if orig_dtype.itemsize <= 2:
                work_dtype = np.float32
                scale = float(valid_max)
            else:
                work_dtype = np.float64
                scale = float(max(np.abs(image.max(initial=0)), np.abs(image.min(initial=0)), 1))
            image_float = image.astype(work_dtype) / work_dtype(scale)
        elif orig_dtype in (np.float32, np.float64):
            image_float = image
        else:
            image_float = image.astype(np.float32)

        result = func(image_float, *args, **kwargs)

        def convert_output(arr):
            if not isinstance(arr, np.ndarray):
                return arr

            if not is_integer:
                return arr.astype(orig_dtype)

            arr_scaled = arr.astype(np.float64) * scale

            if np.nanmax(arr_scaled) > valid_max or np.nanmin(arr_scaled) < valid_min:
                warnings.warn(
                    f"Values outside {orig_dtype} range were clipped to [{valid_min}, {valid_max}].",
                    RuntimeWarning,
                    stacklevel=3
                )

            return _float_to_int(arr_scaled, orig_dtype)

        if isinstance(result, tuple):
            return tuple(convert_output(res) if i == 0 else res for i, res in enumerate(result))
        elif isinstance(result, dict):
            res_dict = result.copy()
            for key in ['image', 'corrected']:
                if key in res_dict:
                    res_dict[key] = convert_output(res_dict[key])
            return res_dict
        else:
            return convert_output(result)

    return wrapper

# Default (lower, upper) percentiles of the input range for processing (D02 (i),
# D27): contrast_stretching, normalize_image and clahe_u8 share it. Processing
# for segmentation tolerates clipping the top 0.1%; the viewers display up to
# p99.99 instead, so sparse bright signal is not saturated on screen.
PROCESSING_PERCENTILES = (0.1, 99.9)


def _resolve_percentiles(func, percentiles, lo_name, lo, hi_name, hi, stacklevel):
    """(lo, hi) from ``percentiles`` or from the deprecated per-bound names (R6-21).

    A scalar in the ``percentiles`` slot is the legacy positional lower bound.
    ``stacklevel`` is passed to warn_deprecated (frames up to the user's call).
    """
    if percentiles is not None and np.ndim(percentiles) == 0:
        if lo is not None:
            raise TypeError(f"{func}() got multiple values for {lo_name!r}")
        lo, percentiles = percentiles, None
    if lo is None and hi is None:
        pct = PROCESSING_PERCENTILES if percentiles is None else tuple(percentiles)
        if len(pct) != 2:
            raise ValueError(f"percentiles must be a (lower, upper) pair, got {percentiles!r}")
        return pct
    if percentiles is not None:
        raise TypeError(f"{func}() got both 'percentiles' and its deprecated aliases "
                        f"{lo_name!r}/{hi_name!r}")
    warn_deprecated(f"{func}({lo_name}, {hi_name})", f"{func}(image, percentiles=(lo, hi))",
                    stacklevel=stacklevel)
    d_lo, d_hi = PROCESSING_PERCENTILES
    return (d_lo if lo is None else lo, d_hi if hi is None else hi)


@ensure_float_and_restore_dtype
def contrast_stretching(image, percentiles=None, p_max=None, *, p_min=None):
    """
    Stretch the intensity range between two percentiles to [0, 1].

    Parameters
    ----------
    image : ndarray
    percentiles : (float, float), optional
        Lower and upper percentiles of the input range, default
        ``PROCESSING_PERCENTILES`` = (0.1, 99.9).
    p_min, p_max : float, deprecated
        Old names of the two bounds (also accepted positionally,
        ``contrast_stretching(image, p_min, p_max)``); FutureWarning, removed
        in eigenp_utils 0.1.0.

    Returns
    -------
    ndarray of the input dtype: floats in [0, 1]; 8/16-bit integers in
    [0, dtype max]; 32/64-bit integers in [0, data max] (the dtype decorator
    scales them by the data's own maximum magnitude, see
    `ensure_float_and_restore_dtype`). Use `normalize_image` for
    [0, dtype max] with wide integers and for a choice of output dtype.

    The percentiles are exact, over the finite values (NaN and inf ignored).
    When they are equal although the image is not constant (signal on fewer than
    (100 - upper)% of the pixels), the range falls back to the data maximum
    (and minimum) with a UserWarning. A constant image gives 0 (NaN stays NaN
    in float output).

    Changed (D02, 2026-11): the degenerate case used to return the input
    unchanged (raw units, not mapped to [0, 1]).
    Changed (D27, 2026-11): the default lower percentile is 0.1 (was 0.0,
    so one dead pixel set the floor); a constant image gives 0 (it was
    returned unchanged, while normalize_image gave 0); the bounds are passed
    as ``percentiles=(lo, hi)`` (R6-21).
    """
    lo_pct, hi_pct = _resolve_percentiles("contrast_stretching", percentiles,
                                          "p_min", p_min, "p_max", p_max, stacklevel=5)
    v_min, v_max = robust_range(image, lo_pct, hi_pct, n=max(np.size(image), 1), stacklevel=3)
    if v_min == v_max:
        image = np.asarray(image)
        return np.where(np.isnan(image), image, 0).astype(image.dtype)
    out = rescale_intensity(image, in_range=(v_min, v_max), out_range=(0.0, 1.0))
    # float32 rounding of the bounds can leave the top at 0.9999999: pin both ends
    out[image >= v_max] = 1.0
    out[image <= v_min] = 0.0
    return out


def normalize_image(image, percentiles=None, upper_percentile=None, dtype=None, *,
                    output_dtype=None, lower_percentile=None):
    """
    Normalize intensities between two percentiles to [0, 1] (float) or [0, dtype max] (int).

    Parameters
    ----------
    image : ndarray
    percentiles : (float, float), optional
        Lower and upper percentiles of the input range, default
        ``PROCESSING_PERCENTILES`` = (0.1, 99.9).
    output_dtype : dtype, optional
        Output dtype, default the input dtype. Integer outputs span
        [0, dtype max] for every integer width (int64 included).
    lower_percentile, upper_percentile, dtype : deprecated
        Old names (also accepted positionally,
        ``normalize_image(image, lower, upper, dtype)``); FutureWarning,
        removed in eigenp_utils 0.1.0.

    NaN pixels are ignored when computing the percentiles; they stay NaN in float
    outputs and become 0 in integer outputs (inf is ignored too). The
    percentiles are exact. When they are equal although the image is not constant (signal on fewer
    than (100 - upper)% of the pixels), the range falls back to the
    data maximum (and minimum) with a UserWarning. A constant image gives 0.

    Unlike `contrast_stretching` (same mapping), the computation is in float32
    for float32 input and float64 otherwise, without the dtype decorator.

    Changed (D02, 2026-11): the degenerate case used to give an all-zero image.
    Changed (D27, 2026-11): the default lower percentile is 0.1 (was 0.5),
    shared with contrast_stretching and clahe_u8; the bounds are passed as
    ``percentiles=(lo, hi)`` and the dtype as ``output_dtype`` (R6-21).
    """
    lo_pct, hi_pct = _resolve_percentiles("normalize_image", percentiles, "lower_percentile",
                                          lower_percentile, "upper_percentile",
                                          upper_percentile, stacklevel=4)
    if dtype is not None:
        if output_dtype is not None:
            raise TypeError("normalize_image() got both 'output_dtype' and its deprecated alias 'dtype'")
        warn_deprecated("normalize_image(dtype=...)", "output_dtype=...")
        output_dtype = dtype

    image = np.asarray(image)
    lower_bound, upper_bound = robust_range(image, lo_pct, hi_pct, n=max(image.size, 1))

    work = image if image.dtype == np.float32 else image.astype(np.float64)
    denom = upper_bound - lower_bound
    if denom == 0:
        denom = 1.0

    normalized_image = (np.clip(work, lower_bound, upper_bound) - lower_bound) / denom

    if output_dtype is None:
        output_dtype = image.dtype

    if np.issubdtype(output_dtype, np.integer):
        return _float_to_int(normalized_image.astype(np.float64) * np.iinfo(output_dtype).max, output_dtype)
    return normalized_image.astype(output_dtype)

def _fit_linear(x_data, y_data):
    """Exact OLS for linear decay (additive noise)."""
    m, c = np.polyfit(x_data, y_data, 1)
    return m * x_data + c

def _fit_exponential(x_data, y_data):
    y_safe = np.clip(y_data, 1e-9, None)
    b_guess, log_a_guess = np.polyfit(x_data, np.log(y_safe), 1)
    a_guess = np.exp(log_a_guess)
    b_guess = min(b_guess, 0.0)

    def model(x, a, b):
        return a * np.exp(b * x)

    params, _ = curve_fit(
        model,
        x_data,
        y_data,
        p0=[a_guess, b_guess],
        bounds=([0, -np.inf], [np.inf, 0.0]),
        maxfev=10000
    )
    return model(x_data, *params)

# Registry of supported parametric models
FIT_MODELS = {
    'linear': _fit_linear,
    'exponential': _fit_exponential
}

def _legacy_z_decay_kwargs(kwargs):
    """Map the deprecated wrappers' keywords onto correct_z_intensity_decay's.

    gamma_fit_func -> fit_model. FLIP_Z_AXIS was dropped silently; it is still
    ignored (the wrappers never flipped anything), but True now warns (R6-14).
    """
    if 'gamma_fit_func' in kwargs:
        kwargs['fit_model'] = kwargs.pop('gamma_fit_func')
    if kwargs.pop('FLIP_Z_AXIS', False):
        warnings.warn(
            "FLIP_Z_AXIS=True is ignored: correct_z_intensity_decay has no flip option. "
            "Flip the stack yourself (image[::-1]; check_z_axis_orientation tells "
            "whether it is needed) and flip the result back.",
            UserWarning, stacklevel=3)
    return kwargs


# Removed @ensure_float_and_restore_dtype to avoid double-decoration
def adjust_gamma_per_slice(*args, **kwargs):
    """Deprecated: use ``correct_z_intensity_decay(image, method='gamma', fit_model=...)``.

    Emits a FutureWarning; removed in eigenp_utils 0.1.0 (D08).
    ``gamma_fit_func`` maps to ``fit_model``; ``FLIP_Z_AXIS=True`` is ignored
    with a UserWarning.
    """
    warn_deprecated("adjust_gamma_per_slice", "correct_z_intensity_decay(image, method='gamma')")
    kwargs = _legacy_z_decay_kwargs(kwargs)
    return correct_z_intensity_decay(*args, method='gamma', **kwargs)


# Removed @ensure_float_and_restore_dtype to avoid double-decoration
def adjust_brightness_per_slice(*args, **kwargs):
    """Deprecated: use ``correct_z_intensity_decay`` (``method='gamma'`` or ``'gain'``).

    Emits a FutureWarning; removed in eigenp_utils 0.1.0 (D08). Its default
    method is 'gamma', as correct_z_intensity_decay's. ``gamma_fit_func``
    maps to ``fit_model``; ``FLIP_Z_AXIS=True`` is ignored with a UserWarning.
    """
    warn_deprecated("adjust_brightness_per_slice", "correct_z_intensity_decay")
    kwargs = _legacy_z_decay_kwargs(kwargs)
    return correct_z_intensity_decay(*args, **kwargs)

@ensure_float_and_restore_dtype
def correct_z_intensity_decay(image, method='gamma', fit_model='exponential', final_gamma=0.8,
                              return_diagnostic=False, *, exclude_background=True):
    """
    Corrects Z-axis intensity attenuation using statistical decay modeling.

    The per-slice 99th percentile (p99, NaN ignored) is fitted with
    `fit_model` and every slice is corrected so that its fitted percentile maps
    onto the brightest fitted one (y_ref):

    - method='gain': multiply slice i by y_ref / y_fit[i] (clipped to [0.1, 100]).
    - method='gamma': power law on a rescaled intensity, I -> s * (I / s) ** g_i,
      with t = 0.5, s = 2 * y_ref and g_i = log(t) / log(t * y_fit[i] / y_ref)
      (clipped to [0.1, 10]). This maps y_fit[i] exactly onto y_ref and only
      depends on ratios, so the same counts give the same result whatever the
      dtype container (uint8, uint16 or float in raw units).

    Both methods clip the output to [0, 1] for data in [0, 1] (which includes
    every integer image, via the dtype decorator); float data in raw units are
    only clipped at 0 when non-negative. A UserWarning lists the slices whose
    factor hit its clip bound.

    Parameters
    ----------
    image : (Z, Y, X) ndarray
    method : {'gamma', 'gain'}
    fit_model : {'exponential', 'linear'}, callable or None
        'exponential' fits a * exp(b z) with b <= 0 (non-linear least squares
        seeded by OLS on log p99); 'linear' is exact OLS. A callable takes
        (z, p99) of the fitted slices and returns the fitted p99 at those z.
        None applies the legacy linear ramp of gamma exponents from 1 to
        `final_gamma` (method='gamma' only).
    final_gamma : float
        Last exponent of the fit_model=None ramp; ignored otherwise.
    return_diagnostic : bool
        Also return {'x_data', 'y_data_norm', 'y_fit_norm', 'gamma_fit_func'}
        (every slice; y_data_norm is NaN for all-NaN slices).
    exclude_background : bool
        Leave background-only slices out of the fit (default True). The
        background level B and its spread are the median and MAD of the slice
        with the lowest p99; a slice is background-only when its p99 is within
        10 MAD of B **and** less than 5% of the brightest slice's p99 above B.
        In sparse samples, p99 of a slice beyond the specimen measures camera
        background, and fitting it drags the curve down and amplifies that
        background 7-285x.

    Slices left out of the fit (background-only or all-NaN) take the fitted
    curve interpolated between their fitted neighbours, held constant beyond
    the first/last fitted slice (no extrapolation), so empty slices get the
    factor of the nearest slice with signal. A UserWarning lists them.

    The exponential model only decays: a stack that gets brighter with Z (by
    more than 10% over the stack, e.g. acquired from the other side) is left
    unchanged with a UserWarning; flip it (``image[::-1]``, see
    `check_z_axis_orientation`) or use fit_model='linear'.

    Changed (D21, 2026-11):
    (a) background-only and all-NaN slices are left out of the fit (an
        all-NaN slice used to abort with RuntimeError), and clipped factors
        are reported;
    (b) gamma is always anchored at t = 0.5, s = 2 * y_ref, and clipped like
        gain. Before, [0, 1] data with y_ref <= 0.5 used the classic
        I ** g (t = y_ref), so uint8 and uint16 holding the same counts
        differed by up to 61 levels, and bright pixels in dim slices
        overshot the range;
    (c) method='gain' with fit_model=None raises ValueError (it multiplied
        the image by the gamma ramp, darkening deep slices);
    (d) brightening stacks warn instead of being silently left unchanged.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Image must be a 3D numpy array.")
    if method not in ('gamma', 'gain'):
        raise ValueError(f"Unknown correction method: {method}")
    if fit_model is None and method == 'gain':
        raise ValueError(
            "fit_model=None (the final_gamma ramp of gamma exponents) applies to "
            "method='gamma' only; use fit_model='exponential' or 'linear' with method='gain'.")

    num_slices = image.shape[0]
    adjusted_image = np.empty_like(image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN input
        data_min, data_max = float(np.nanmin(image)), float(np.nanmax(image))
    if not np.isfinite(data_min):
        data_min, data_max = 0.0, 0.0
    unit_range = data_min >= 0.0 and data_max <= 1.0
    if method == 'gamma' and data_min < 0:
        raise ValueError("method='gamma' needs non-negative intensities.")

    diagnostic_data = None
    scale = 1.0
    upper = 1.0 if unit_range else np.inf
    lower = 0.0 if data_min >= 0 else -np.inf

    if fit_model is not None:
        x_data = np.arange(num_slices)
        y_data = _slice_p99(image)
        fit_idx = np.flatnonzero(np.isfinite(y_data))
        left_out = {}
        if fit_idx.size < num_slices:
            left_out['all-NaN'] = np.flatnonzero(~np.isfinite(y_data))
        if exclude_background and fit_idx.size > 2:
            background = _background_only_slices(image, y_data, fit_idx)
            if background.size and fit_idx.size - background.size >= 2:
                left_out['background only'] = background
                fit_idx = np.setdiff1d(fit_idx, background)
        if fit_idx.size == 0:
            fit_idx = x_data  # nothing finite: the fit fails with a clear error below

        try:
            if isinstance(fit_model, str):
                if fit_model not in FIT_MODELS:
                    raise ValueError(f"Unknown fit_model: {fit_model}. Available: {list(FIT_MODELS.keys())}")
                y_fit_sub = FIT_MODELS[fit_model](x_data[fit_idx], y_data[fit_idx])
            elif callable(fit_model):
                y_fit_sub = fit_model(x_data[fit_idx], y_data[fit_idx])
            else:
                raise ValueError("fit_model must be a string or callable.")
        except Exception as e:
            raise RuntimeError(f"Z-decay fitting failed: {e}. Pipeline aborted.")
        y_fit_sub = np.asarray(y_fit_sub, dtype=np.float64)
        if fit_idx.size == num_slices:
            y_fit = y_fit_sub
        else:
            # no extrapolation: hold the end values, interpolate inner gaps
            y_fit = np.interp(x_data, x_data[fit_idx], y_fit_sub)

        if left_out:
            warnings.warn(
                "correct_z_intensity_decay: slices left out of the fit ("
                + "; ".join(f"{why}: {idx.tolist()}" for why, idx in left_out.items())
                + "). They take the fitted value of the nearest fitted slices (no extrapolation).",
                UserWarning, stacklevel=3)

        if fit_model == 'exponential' and fit_idx.size >= 2:
            y_pos = np.clip(y_data[fit_idx], 1e-9, None)
            slope = np.polyfit(x_data[fit_idx], np.log(y_pos), 1)[0]
            if np.exp(slope * (fit_idx[-1] - fit_idx[0])) > 1.1:
                warnings.warn(
                    "correct_z_intensity_decay: intensity increases with Z, but the "
                    "exponential model only decays, so the stack is left uncorrected. "
                    "The stack is probably upside down: flip it (image[::-1], see "
                    "check_z_axis_orientation), or use fit_model='linear'.",
                    UserWarning, stacklevel=3)

        if return_diagnostic:
            diagnostic_data = {
                "x_data": x_data,
                "y_data_norm": y_data,
                "y_fit_norm": y_fit,
                "gamma_fit_func": fit_model if isinstance(fit_model, str) else "custom_callable"
            }

        y_ref = float(np.max(y_fit))
        if not y_ref > 0:
            factors = np.ones(num_slices)
        else:
            # Ratio of each slice's fitted percentile to the reference (scale-free)
            ratio = np.clip(y_fit / y_ref, 1e-9, 1.0)
            if method == 'gamma':
                t = 0.5
                scale = y_ref / t
                raw, bounds = np.log(t) / np.log(t * ratio), (0.1, 10.0)
            else:
                raw, bounds = 1.0 / ratio, (0.1, 100.0)
            factors = np.clip(raw, *bounds)
            clipped = np.flatnonzero(factors != raw)
            if clipped.size:
                warnings.warn(
                    f"correct_z_intensity_decay: the {method} factor of slices "
                    f"{clipped.tolist()} was clipped to [{bounds[0]}, {bounds[1]}]; the "
                    "fitted curve falls too low there (empty slices or a poor fit).",
                    UserWarning, stacklevel=3)
    else:
        factors = np.linspace(1.0, final_gamma, num_slices)
        if not unit_range and data_max > 0:
            scale = data_max

    for i in range(num_slices):
        if method == 'gamma':
            res = scale * np.power(image[i, :, :] / scale, factors[i])
            if fit_model is not None:
                res = np.clip(res, lower, upper)
            adjusted_image[i, :, :] = res
        else:
            adjusted_image[i, :, :] = np.clip(image[i, :, :] * factors[i], lower, upper)

    if return_diagnostic:
        return {"image": adjusted_image, "diagnostic_data": diagnostic_data}
    return adjusted_image


def _slice_p99(image):
    """Per-slice nanpercentile 99 (NaN for an all-NaN slice, without a RuntimeWarning)."""
    out = np.full(image.shape[0], np.nan)
    for i in range(image.shape[0]):
        if np.isfinite(image[i]).any():
            out[i] = np.nanpercentile(image[i], 99)
    return out


# A slice is background-only when its p99 is within _BG_MADS MADs of the
# background level AND less than _BG_FRACTION of the brightest p99 above it.
_BG_MADS = 10.0
_BG_FRACTION = 0.05


def _background_only_slices(image, p99, idx):
    """Indices (within idx) of slices whose p99 measures background only (D21a)."""
    j = idx[np.argmin(p99[idx])]
    ref = image[j][np.isfinite(image[j])]
    b = float(np.median(ref))
    mad = float(np.median(np.abs(ref - b)))
    top = float(np.max(p99[idx])) - b
    if not top > 0:
        return np.array([], dtype=int)
    sel = (p99[idx] <= b + _BG_MADS * mad) & (p99[idx] - b <= _BG_FRACTION * top)
    return idx[sel]

def contrast_stretch_per_slice(image, in_min=None, in_max=None, flip_z=False, *,
                               p_min_array=None, p_max_array=None, FLIP_Z_AXIS=None):
    """
    Adjusts the intensity of each slice in a 3D image along the Z-axis using per-slice min/max intensity values.

    Args:
        image (numpy.ndarray): The 3D image array in Z, Y, X order.
        in_min (numpy.ndarray): Per-slice lower bound of the input range, in
            intensity units (not a percentile). None: each slice's own minimum.
        in_max (numpy.ndarray): Per-slice upper bound, in intensity units.
            None: each slice's own maximum.
        flip_z (bool): Reverse in_min / in_max along Z before applying them.
        p_min_array, p_max_array, FLIP_Z_AXIS: deprecated names of in_min,
            in_max and flip_z (FutureWarning, removed in eigenp_utils 0.1.0).
            They were renamed because p_min elsewhere means a percentile.

    Returns:
        numpy.ndarray: A new 3D image array with adjusted intensities per slice,
        in [0, 1] for floats and [0, dtype max] for unsigned integers.

    Changed (R6-21, 2026-11): parameters renamed (in_min, in_max, flip_z).
    """
    renamed = (("in_min", in_min, "p_min_array", p_min_array),
               ("in_max", in_max, "p_max_array", p_max_array),
               ("flip_z", flip_z or None, "FLIP_Z_AXIS", FLIP_Z_AXIS))
    values = []
    for new, new_value, old, old_value in renamed:
        if old_value is not None:
            if new_value is not None:
                raise TypeError(f"contrast_stretch_per_slice() got both {new!r} and its "
                                f"deprecated alias {old!r}")
            warn_deprecated(f"contrast_stretch_per_slice({old}=...)", f"{new}=...")
            new_value = old_value
        values.append(new_value)
    p_min_array, p_max_array, flip = values

    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Image must be a 3D numpy array.")
    if p_min_array is None and p_max_array is None:
        raise ValueError("in_min and in_max are both None, no difference will be produced in the image")

    # Floats always map to [0, 1]: out_range='dtype' gives [-1, 1] for a float
    # slice whose lower bound is negative (as in contrast_stretching)
    out_range = (0.0, 1.0) if np.issubdtype(image.dtype, np.floating) else 'dtype'

    # Get the number of slices
    num_slices = image.shape[0]

    # Create an output image array
    adjusted_image = np.empty_like(image)

    if flip:
        if p_min_array is not None:
            p_min_array = p_min_array[::-1]
        if p_max_array is not None:
            p_max_array = p_max_array[::-1]

    # Apply contrast stretching slice by slice
    for i in range(num_slices):
        # Use provided min/max or defaults (min/max of slice or image logic could be applied,
        # but rescale_intensity handles "image" range if not specified.
        # However, here we assume arrays are meant to define the range).

        # A missing bound defaults to the slice's own min / max.
        in_range_min = p_min_array[i] if p_min_array is not None else np.nanmin(image[i])
        in_range_max = p_max_array[i] if p_max_array is not None else np.nanmax(image[i])

        # Construct in_range tuple
        current_in_range = (in_range_min, in_range_max)

        adjusted_image[i, :, :] = rescale_intensity(image[i, :, :], in_range=current_in_range, out_range=out_range)

    return adjusted_image

def check_z_axis_orientation(img, channel=None, return_image=True):
    """
    Check whether the Z axis should be flipped so that the brighter half comes first.

    Compares the 99.99th percentile of the top half (first Z slices) with the
    bottom half; flag is True when the bottom half is brighter. Z-decay
    correction (`correct_z_intensity_decay`) assumes intensity falls with Z,
    so a True flag means the stack is probably upside down for it.

    NaN voxels are ignored. If either half has no finite voxel the halves
    cannot be compared: a UserWarning is emitted and the flag is False. The
    two metrics are logged at DEBUG level (module logger).

    Args:
        img (numpy.ndarray): Input image (nD). Expected 6D (S,T,C,Z,Y,X) or 3D (Z,Y,X).
        channel (int, optional): Channel index to use for metric calculation if input is high-dimensional.
        return_image (bool): If True, returns (image, flag). If False, returns flag.

    Returns:
        tuple or bool: (flipped_image, flag) if return_image is True, else flag.

    Changed (D07/R6-16, 2026-11): renamed from ``test_z_axis_orientation``
    (pytest collected it as a test; the old name is a deprecated alias); it no
    longer prints the two metrics; NaN voxels are ignored (one NaN made the
    comparison silently False).
    """
    # Extract Z-stack for metric calculation
    if img.ndim == 3:
        img_Z = img
    elif channel is not None:
        if img.ndim < 4:
            raise ValueError(f"channel given but image has {img.ndim} dims; expected (..., C, Z, Y, X).")
        # C is the 4th axis from the end in (S, T, C, Z, Y, X), (T, C, Z, Y, X) and (C, Z, Y, X)
        img_Z = np.take(img, channel, axis=-4)
    else:
        img_Z = np.squeeze(img)

    # Ensure img_Z is 3D (Z, Y, X) for the split
    if img_Z.ndim > 3:
        # If squeeze didn't reduce enough, take the first element of extra dims
        while img_Z.ndim > 3:
            img_Z = img_Z[0]

    FLAG = False

    if img_Z.ndim != 3 or img_Z.shape[0] < 2:
        raise ValueError(f"Need a (Z, Y, X) stack with Z >= 2, got shape {img_Z.shape}.")

    # Compare 99.99th percentile of top vs bottom half of Z
    z_mid = img_Z.shape[0] // 2
    halves = (img_Z[0:z_mid, :, :], img_Z[z_mid:, :, :])
    if not all(np.isfinite(h).any() for h in halves):
        warnings.warn(
            "check_z_axis_orientation: one half of the stack has no finite voxel "
            "(all NaN), so the halves cannot be compared; not flipping.",
            UserWarning, stacklevel=2)
    else:
        top_image_metric, bot_image_metric = (float(np.nanpercentile(h, 99.99)) for h in halves)
        logger.debug("Z orientation: p99.99 top half %g, bottom half %g",
                     top_image_metric, bot_image_metric)
        if top_image_metric < bot_image_metric:
            FLAG = True

    if return_image:
        if FLAG:
            # Flip along Z axis.
            # Assuming Z is the 3rd to last dimension (..., Z, Y, X)
            img = img[..., ::-1, :, :]
        return img, FLAG
    else:
        return FLAG


def test_z_axis_orientation(img, channel=None, return_image=True):
    """Deprecated alias of `check_z_axis_orientation` (FutureWarning; removed in eigenp_utils 0.1.0).

    The ``test_`` prefix made pytest collect this library function as a test
    whenever a test module imported it under its own name.
    """
    warn_deprecated("test_z_axis_orientation", "check_z_axis_orientation")
    return check_z_axis_orientation(img, channel=channel, return_image=return_image)


# Do not let pytest collect the deprecated alias in modules that import it
test_z_axis_orientation.__test__ = False


def _tshrinkage(x, thresh):
    return np.sign(x) * np.clip(np.abs(x) - thresh, a_min=0, a_max=None)

def _prepare_data_for_basic(images, is_3d=False):
    """
    Normalizes input `images` into a numpy array of shape (N, ...)
    where `...` is the spatial dimensions matching the flatfield.
    """
    if isinstance(images, list):
        images = np.stack(images, axis=0)

    images = np.asarray(images)
    original_shape = images.shape

    if is_3d:
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        elif images.ndim >= 4:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    else:
        if images.ndim == 2:
            images = images[np.newaxis, ...]
        elif images.ndim >= 3:
            images = images.reshape(-1, images.shape[-2], images.shape[-1])

    return images, original_shape

def _resize_spatial(images, target_spatial_shape):
    N = images.shape[0]
    out = np.empty((N,) + target_spatial_shape, dtype=np.float32)
    for i in range(N):
        out[i] = transform.resize(
            images[i], target_spatial_shape,
            order=1, mode='reflect', anti_aliasing=False, preserve_range=True
        )
    return out

def _reweight_score(S, last_S, D, last_D):
    """Reweighting convergence score: max relative change of flatfield and darkfield.

    As MATLAB BaSiC: sum|S - last_S| / sum|last_S| and sum|D - last_D| /
    max(sum|last_D|, 1e-6), the latter 0 when the darkfield moved by < 1e-7
    in total (always, without darkfield estimation). BaSiCPy computes the
    darkfield term from the flatfield change instead (R6-17).
    """
    mad_flatfield = np.sum(np.abs(S - last_S)) / max(float(np.sum(np.abs(last_S))), 1e-9)
    temp_diff = float(np.sum(np.abs(D - last_D)))
    if temp_diff < 1e-7:
        mad_darkfield = 0.0
    else:
        mad_darkfield = temp_diff / max(float(np.sum(np.abs(last_D))), 1e-6)
    return float(max(mad_flatfield, mad_darkfield))


def fit_basic_shading(
    images,
    is_3d=False,
    working_size=128,
    max_reweight_iterations=10,
    max_iterations=500,
    optimization_tol=1e-3,
    optimization_tol_diff=1e-2,
    reweighting_tol=1e-2,
    smoothness_flatfield=None,
    smoothness_darkfield=None,
    sparse_cost_darkfield=None,
    get_darkfield=False,
    fitting_mode='approximate',
    epsilon=0.1,
    rho=1.5,
    mu_coef=12.5,
    max_mu_coef=1e7
):
    """
    Fits the BaSiC shading model to a collection of images.

    Args:
        images: Input images. Can be a 3D array (Z, Y, X), 4D array (T, Z, Y, X),
                or list of 3D/2D arrays.
        is_3d (bool): If True, computes a 3D flatfield (Z, Y, X).
                      If False (default), computes a 2D flatfield (Y, X).
        working_size (int): Spatial dimension size to downscale to for faster fitting.
        max_reweight_iterations (int): Maximum number of reweighting iterations.
        max_iterations (int): Maximum number of ADMM iterations per reweighting loop.
        optimization_tol (float): Tolerance for ADMM convergence (relative fit residual).
        optimization_tol_diff (float): 'ladmap' only: tolerance on the relative size of the
            last update. Both tolerances must be met to stop (as in BaSiCPy).
        reweighting_tol (float): Tolerance for reweighting loop convergence.
        smoothness_flatfield (float): Smoothness regularization weight. If None, it is estimated.
        smoothness_darkfield (float): Smoothness regularization weight for darkfield. If None, it is estimated.
        sparse_cost_darkfield (float): 'ladmap' only: weight of the darkfield sparse term.
            If None, it is estimated. The 'approximate' mode (as MATLAB BaSiC)
            shrinks the darkfield with smoothness_darkfield only, so a value
            given there is ignored, with a UserWarning.
        get_darkfield (bool): Whether to estimate darkfield. Use
            fitting_mode='approximate' for it: with 'ladmap' the estimate is
            unreliable (a UserWarning says so; see below).
        fitting_mode (str): Fitting mode, 'approximate' or 'ladmap'.
        epsilon (float): Small value for weight calculation.
        rho (float): ADMM penalty parameter update factor.
        mu_coef (float): Initial penalty parameter coefficient.
        max_mu_coef (float): Maximum penalty parameter coefficient.

    Returns:
        dict: Containing 'flatfield', 'darkfield', and 'baseline'.

    Raises:
        ValueError: for NaN or inf in `images` (the ADMM has no mask support,
            and one NaN pixel made the SVD fail with LinAlgError). Fill or crop
            them first.

    Darkfield with 'ladmap': on synthetic images (20 frames, baselines
    0.5-1.5) with known darkfields of 10, 40 and 100, the 'ladmap' darkfield
    came out at 0.04, 117 and 268 (the last two above the dimmest pixel,
    which is impossible), while 'approximate' gives 4.5, 36 and 82. The ladmap
    estimate is returned as before, with a UserWarning, until it is fixed
    and validated.

    Changed (R6-17, 2026-11): reweighting convergence uses the darkfield
    change (as MATLAB BaSiC) instead of repeating the flatfield change (a
    BaSiCPy quirk); this only affects get_darkfield=True. NaN/inf input
    raises ValueError (R6-15).
    """
    if fitting_mode not in ('approximate', 'ladmap'):
        raise ValueError(f"fitting_mode '{fitting_mode}' is not valid. Use 'approximate' or 'ladmap'.")
    if max_reweight_iterations < 1:
        raise ValueError("max_reweight_iterations must be >= 1.")
    if fitting_mode == 'approximate' and sparse_cost_darkfield is not None:
        warnings.warn(
            "sparse_cost_darkfield is ignored with fitting_mode='approximate' (which, as "
            "MATLAB BaSiC, regularises the darkfield with smoothness_darkfield only); "
            "it applies to fitting_mode='ladmap'.", UserWarning, stacklevel=2)

    images, original_shape = _prepare_data_for_basic(images, is_3d)
    n_bad = int(images.size - np.count_nonzero(np.isfinite(images)))
    if n_bad:
        raise ValueError(
            f"fit_basic_shading needs finite input: the images hold {n_bad} non-finite "
            "(NaN or inf) values. Fill or crop them first (the fit has no mask support).")

    s_s = images.shape[0]
    original_spatial_shape = images.shape[1:]

    if get_darkfield and s_s < 2:
        # BaSiC models each sample as I_i = B_i * S + D and separates the darkfield D
        # from the flatfield S only through variation of the baselines B_i across
        # samples. With a single image/volume any D can be absorbed into S, so the
        # darkfield is not identifiable (in 'approximate' mode its estimator has no
        # samples to average and the fit degenerated to NaN after max iterations).
        warnings.warn(
            "get_darkfield=True needs at least 2 images (or volumes with is_3d=True): the "
            "darkfield is not identifiable from a single sample. Fitting the flatfield only; "
            "the returned darkfield is zero.",
            UserWarning,
            stacklevel=2,
        )
        get_darkfield = False
    if get_darkfield and fitting_mode == 'ladmap':
        warnings.warn(
            "get_darkfield=True with fitting_mode='ladmap' gives an unreliable darkfield "
            "(anywhere from ~0 to ~3x the true offset on synthetic data); use "
            "fitting_mode='approximate' to estimate the darkfield.", UserWarning, stacklevel=2)

    target_spatial_shape = tuple([min(d, working_size) if working_size is not None else d for d in original_spatial_shape])

    if target_spatial_shape != original_spatial_shape:
        Im = _resize_spatial(images, target_spatial_shape)
    else:
        Im = images.astype(np.float32)

    s_spatial = Im.shape[1:]
    Im_flat = Im.reshape(s_s, -1)

    image_norm = np.linalg.norm(Im_flat)
    if image_norm == 0:
        image_norm = 1.0

    if smoothness_flatfield is None:
        meanD = Im.mean(axis=0)
        mean_meanD = meanD.mean()
        if mean_meanD == 0:
            mean_meanD = 1.0
        meanD = meanD / mean_meanD
        W_meanD = dctn(meanD, norm='ortho')
        smoothness_flatfield = np.sum(np.abs(W_meanD)) / 400.0 * 0.5

    if smoothness_darkfield is None:
        smoothness_darkfield = smoothness_flatfield * 0.1

    if sparse_cost_darkfield is None:
        sparse_cost_darkfield = smoothness_darkfield * 0.01 * 100 # Default is 0.01 * 100 in original BaSiC

    _, S_svd, _ = np.linalg.svd(Im_flat, full_matrices=False)
    spectral_norm = S_svd[0]
    if spectral_norm == 0:
        spectral_norm = 1.0

    if fitting_mode == 'approximate':
        init_mu = mu_coef / spectral_norm
    else:
        init_mu = mu_coef / spectral_norm / np.prod(Im_flat.shape)

    max_mu = init_mu * max_mu_coef

    ent1 = 1.0
    ent2 = 10.0
    D_Z_max = np.min(Im_flat)

    W = np.ones_like(Im_flat)
    W_D = np.ones(s_spatial, dtype=np.float32)

    last_S = None
    last_D = None

    S = np.ones(s_spatial, dtype=np.float32)
    B = np.ones(s_s, dtype=np.float32)
    D_R = np.zeros(s_spatial, dtype=np.float32)
    D_Z = 0.0

    for reweight_iter in range(max_reweight_iterations):
        S_hat = dctn(S, norm='ortho')
        D_R = np.zeros(s_spatial, dtype=np.float32)
        D_Z = 0.0

        if fitting_mode == 'approximate':
            mean_Im_flat = np.nanmean(Im_flat)
            if mean_Im_flat == 0:
                mean_Im_flat = 1.0
            B = np.nanmean(Im_flat, axis=1) / mean_Im_flat
        else:
            B = np.ones(s_s, dtype=np.float32)
            S = np.median(Im, axis=0)
            S_hat = dctn(S, norm='ortho')

        I_R_flat = np.zeros_like(Im_flat)
        I_B_flat = (S[np.newaxis, ...] * B[(...,) + (np.newaxis,) * S.ndim] + D_R[np.newaxis, ...] + D_Z).reshape(s_s, -1)

        Y_flat = np.zeros_like(Im_flat)
        mu = init_mu

        for k in range(max_iterations):
            if fitting_mode == 'approximate':
                temp_W = (Im_flat - I_R_flat - I_B_flat + Y_flat / mu) / ent1
                temp_W = np.mean(temp_W, axis=0)

                S_hat = S_hat + dctn(temp_W.reshape(s_spatial), norm="ortho")
                S_hat = _tshrinkage(S_hat, smoothness_flatfield / (ent1 * mu))
                S = idctn(S_hat, norm="ortho")

                I_B_flat = (S[np.newaxis, ...] * B[(...,) + (np.newaxis,) * S.ndim] + D_R[np.newaxis, ...] + D_Z).reshape(s_s, -1)

                I_R_flat = I_R_flat + (Im_flat - I_B_flat - I_R_flat + (1 / mu) * Y_flat) / ent1
                I_R_flat = _tshrinkage(I_R_flat, W / (ent1 * mu))

                R_flat = Im_flat - I_R_flat
                mean_R_flat = np.mean(R_flat)
                if mean_R_flat == 0:
                    mean_R_flat = 1.0
                B = np.mean(R_flat, axis=1) / mean_R_flat
                B = np.clip(B, 0, None)

                I_B_flat = (S[np.newaxis, ...] * B[(...,) + (np.newaxis,) * S.ndim] + D_R[np.newaxis, ...] + D_Z).reshape(s_s, -1)

                # The darkfield estimator below averages over samples with B < 1; if
                # there are none (e.g. identical samples) this iteration carries no
                # darkfield information, so keep the current D_R / D_Z.
                if get_darkfield and np.any(B < 1):
                    validA1coeff_idx = B < 1
                    S_flat = S.reshape(-1)
                    mean_S = np.mean(S)
                    S_inmask = S_flat >= mean_S
                    S_outmask = S_flat < mean_S

                    R_0 = np.where(S_inmask[np.newaxis, :] & validA1coeff_idx[:, np.newaxis], R_flat, np.nan)
                    R_1 = np.where(S_outmask[np.newaxis, :] & validA1coeff_idx[:, np.newaxis], R_flat, np.nan)

                    mean_R = np.mean(R_flat)

                    # Avoid Mean of empty slice warnings by checking for all-NaN rows
                    def safe_nanmean(arr, axis):
                        # Returns NaN for rows that are all NaN, without triggering a RuntimeWarning
                        mask = np.isnan(arr).all(axis=axis)
                        res = np.zeros(arr.shape[0])
                        res[mask] = np.nan
                        res[~mask] = np.nanmean(arr[~mask], axis=axis)
                        return res

                    mean_R_0 = safe_nanmean(R_0, axis=1)
                    mean_R_1 = safe_nanmean(R_1, axis=1)
                    B1_coeff = (mean_R_0 - mean_R_1) / (mean_R + 1e-6)

                    num_valid = np.sum(validA1coeff_idx)
                    B_nan = np.where(validA1coeff_idx, B, np.nan)

                    temp1 = np.nan_to_num(np.nansum(B_nan**2))
                    temp2 = np.nan_to_num(np.nansum(B_nan))
                    temp3 = np.nan_to_num(np.nansum(B1_coeff))
                    temp4 = np.nan_to_num(np.nansum(B_nan * B1_coeff))
                    temp5 = temp2 * temp3 - num_valid * temp4

                    if temp5 == 0:
                        D_Z = 0.0
                    else:
                        D_Z = (temp1 * temp3 - temp2 * temp4) / temp5
                    D_Z = max(D_Z, 0.0)
                    if mean_S > 1e-9:
                        D_Z = min(D_Z, D_Z_max / mean_S)

                    Z = D_Z * mean_S - D_Z * S_flat

                    R_nan = np.where(validA1coeff_idx[:, np.newaxis], R_flat, np.nan)
                    A1_offset = np.nanmean(R_nan, axis=0) - np.nanmean(B_nan) * S_flat
                    A1_offset = A1_offset.flatten()
                    A1_offset = A1_offset - np.nanmean(A1_offset)

                    D_R = A1_offset - np.mean(A1_offset) - Z
                    D_R = dctn(D_R.reshape(s_spatial), norm="ortho")
                    D_R = _tshrinkage(D_R, smoothness_darkfield / (ent2 * mu))
                    D_R = idctn(D_R, norm="ortho")
                    D_R = _tshrinkage(D_R, smoothness_darkfield / (ent2 * mu))
                    D_R = D_R + Z.reshape(s_spatial)

                fit_residual_flat = Im_flat - I_B_flat - I_R_flat
                Y_flat = Y_flat + mu * fit_residual_flat
                mu = min(mu * rho, max_mu)

                norm_ratio = np.linalg.norm(fit_residual_flat) / image_norm
                if norm_ratio <= optimization_tol:
                    break

            elif fitting_mode == 'ladmap':
                I_B = (S[np.newaxis, ...] * B[(...,) + (np.newaxis,) * S.ndim] + D_R[np.newaxis, ...] + D_Z)
                I_B_flat = I_B.reshape(s_s, -1)
                eta_S = np.sum(B**2) * 1.02 + 0.01

                S_new = S + np.sum(B[(...,) + (np.newaxis,) * S.ndim] * (Im - I_B - I_R_flat.reshape(s_s, *s_spatial) + Y_flat.reshape(s_s, *s_spatial) / mu), axis=0) / eta_S
                S_new = idctn(_tshrinkage(dctn(S_new, norm="ortho"), smoothness_flatfield / (eta_S * mu)), norm="ortho")

                if np.min(S_new) < 0:
                    S_new = S_new - np.min(S_new)
                dS = S_new - S
                S = S_new

                I_B = (S[np.newaxis, ...] * B[(...,) + (np.newaxis,) * S.ndim] + D_R[np.newaxis, ...] + D_Z)
                I_B_flat = I_B.reshape(s_s, -1)

                I_R_new_flat = _tshrinkage(Im_flat - I_B_flat + Y_flat / mu, W / (mu * s_s))
                dI_R_flat = I_R_new_flat - I_R_flat
                I_R_flat = I_R_new_flat

                R_flat = Im_flat - I_R_flat
                S_sq = np.sum(S**2)

                R_spatial = R_flat.reshape(s_s, *s_spatial)
                Y_spatial = Y_flat.reshape(s_s, *s_spatial)

                # While the shrinkage still zeroes S (mu starts tiny), B carries no
                # information: keep it instead of collapsing it to 0 for good.
                if S_sq > 0:
                    B_new = np.sum(S[np.newaxis, ...] * (R_spatial + Y_spatial / mu), axis=tuple(range(1, Im.ndim))) / S_sq
                else:
                    B_new = B.copy()
                B_new = np.clip(B_new, 0, None)

                mean_B = np.mean(B_new)
                if mean_B > 0:
                    B_new = B_new / mean_B
                    S = S * mean_B

                dB = B_new - B
                B = B_new

                BS = S[np.newaxis, ...] * B[(...,) + (np.newaxis,) * S.ndim]

                if get_darkfield:
                    D_Z_new = np.mean(Im - BS - D_R[np.newaxis, ...] - I_R_flat.reshape(s_s, *s_spatial) + Y_spatial / 2.0 / mu)
                    D_Z_new = np.clip(D_Z_new, 0, D_Z_max)
                    dD_Z = D_Z_new - D_Z
                    D_Z = D_Z_new

                    eta_D = s_s * 1.02
                    D_R_new = D_R + 1.0 / eta_D * np.sum(Im - BS - D_R[np.newaxis, ...] - D_Z - I_R_flat.reshape(s_s, *s_spatial) + Y_spatial / mu, axis=0)
                    D_R_new = idctn(_tshrinkage(dctn(D_R_new), smoothness_darkfield / eta_D / mu))
                    D_R_new = _tshrinkage(D_R_new, sparse_cost_darkfield * W_D / eta_D / mu)
                    dD_R = D_R_new - D_R
                    D_R = D_R_new

                I_B = BS + D_R[np.newaxis, ...] + D_Z
                I_B_flat = I_B.reshape(s_s, -1)

                fit_residual_flat = R_flat - I_B_flat
                Y_flat = Y_flat + mu * fit_residual_flat

                value_diff = max([
                    np.linalg.norm(dS.ravel()) * np.sqrt(eta_S),
                    np.linalg.norm(dI_R_flat.ravel()) * 1.0,
                    np.linalg.norm(dB.ravel())
                ])

                if get_darkfield:
                    value_diff = max([
                        value_diff,
                        np.linalg.norm(dD_R.ravel()) * np.sqrt(eta_D),
                        dD_Z**2
                    ])

                norm_ratio = value_diff / image_norm
                mu = min(mu * rho, max_mu)

                # Stop only when the fit residual AND the update are both small. A
                # small update alone happens in the first iterations, while the
                # shrinkage still zeroes S, and stopping there returned S == 0.
                residual_ratio = np.linalg.norm(fit_residual_flat) / image_norm
                if residual_ratio <= optimization_tol and norm_ratio <= optimization_tol_diff:
                    break

        D_R = D_R + D_Z * S
        I_B_flat = (S[np.newaxis, ...] * B[(...,) + (np.newaxis,) * S.ndim] + D_R[np.newaxis, ...]).reshape(s_s, -1)

        S = np.mean(I_B_flat.reshape(s_s, *s_spatial), axis=0) - D_R
        mean_S = np.mean(S)
        if mean_S > 1e-9:
            S = S / mean_S

        if fitting_mode == 'approximate':
            XE_norm = I_R_flat / (np.mean(I_B_flat, axis=1, keepdims=True) + 1e-6)
            W = 1.0 / (np.abs(XE_norm) + epsilon)
            W = W * W.size / np.sum(W)

            W_D = np.ones_like(D_R)
        else:
            Ws = np.ones_like(I_R_flat) / (np.abs(I_R_flat / (I_B_flat + epsilon)) + epsilon)
            W = Ws / np.mean(Ws)

            Ws_D = np.ones_like(D_R) / (np.abs(D_R) + epsilon)
            W_D = Ws_D / np.mean(Ws_D)

        if last_S is not None:
            if _reweight_score(S, last_S, D_R, last_D) <= reweighting_tol:
                break
        last_S = S
        last_D = D_R

    if target_spatial_shape != original_spatial_shape:
        flatfield = transform.resize(S, original_spatial_shape, order=1, mode='reflect', anti_aliasing=False, preserve_range=True)
        darkfield = transform.resize(D_R, original_spatial_shape, order=1, mode='reflect', anti_aliasing=False, preserve_range=True)
    else:
        flatfield = S
        darkfield = D_R

    baseline = B * mean_S

    return {
        'flatfield': flatfield,
        'darkfield': darkfield,
        'baseline': baseline
    }

def apply_basic_shading(
    images,
    flatfield,
    darkfield=None,
    baseline=None,
    baseline_smooth_method=None,
    baseline_smooth_sigma=2.0,
    output_dtype=None,
    offset=None,
    return_offset=False,
):
    """
    Applies the basic shading correction to images.

    The result is ``(images - darkfield) / flatfield - baseline + offset``.

    Args:
        images: Input images.
        flatfield: Estimated flatfield.
        darkfield: Estimated darkfield.
        baseline: Estimated baseline for timelapse correction.
        baseline_smooth_method: Method to smooth baseline ('gaussian').
        baseline_smooth_sigma: Sigma for gaussian smoothing.
        output_dtype: dtype of the result. Defaults to the input dtype, except when
            `baseline` is given without an `offset`: baseline-corrected images are
            centred on zero (as in BaSiCPy), so the default is then float32 rather
            than clipping the negative half of the background to 0. Integer outputs
            are rounded and clipped to the dtype range; a UserWarning reports the
            fraction of pixels clipped when it exceeds 0.1% at either end.
        offset: Constant added after the correction, typically to bring a
            zero-centred (baseline-subtracted) result back into the range of an
            unsigned integer dtype:

            - None (default): no offset.
            - 'baseline_mean': the mean of the (smoothed) baseline that was
              subtracted, so the background returns to its average level and
              intensities stay in roughly the raw units, with only the
              frame-to-frame background changes removed. Requires `baseline`.
            - a number: added as is.

            When an offset is given and `output_dtype` is None, the result has the
            input dtype (e.g. uint16). Intensities are then shifted by the offset:
            subtract it (see `return_offset`) to get back to the zero-centred
            values, e.g. before measuring absolute background-subtracted signal.
            Clipping at 0 is not neutral: for zero-mean background noise of std s
            it raises the mean to about 0.4 s, so pick an offset that keeps the
            low end in range (the clipping warning tells you when it does not).
        return_offset: If True, return ``(corrected, offset_value)`` where
            ``offset_value`` is the float actually added (0.0 when `offset` is None),
            so it can be stored (e.g. in xarray attrs) and subtracted later.

    Returns:
        Corrected images, or ``(corrected, offset_value)`` if `return_offset`.
    """
    images = np.asarray(images)

    if darkfield is None:
        darkfield = np.zeros_like(flatfield)

    diff_dims = images.ndim - flatfield.ndim
    if diff_dims < 0:
        raise ValueError("Images have fewer dimensions than flatfield.")

    expand_tuple = (np.newaxis,) * diff_dims + (...,)

    ff = flatfield[expand_tuple]
    df = darkfield[expand_tuple]

    ff_safe = np.where(ff == 0, 1e-9, ff)
    corrected = (images - df) / ff_safe

    if baseline is not None:
        if baseline_smooth_method == 'gaussian':
            baseline = ndimage.gaussian_filter1d(baseline, sigma=baseline_smooth_sigma)

        leading_shape = images.shape[:diff_dims]
        if np.prod(leading_shape) != baseline.size:
            raise ValueError(f"Baseline size {baseline.size} does not match leading dimensions of images {leading_shape}.")

        b = baseline.reshape(leading_shape)
        b = b.reshape(leading_shape + (1,) * flatfield.ndim)

        corrected = corrected - b

    if offset is None:
        offset_value = 0.0
    elif isinstance(offset, str):
        if offset != 'baseline_mean':
            raise ValueError(f"offset must be None, 'baseline_mean' or a number, got {offset!r}.")
        if baseline is None:
            raise ValueError("offset='baseline_mean' requires a baseline.")
        offset_value = float(np.mean(baseline))
    else:
        offset_value = float(offset)
    if offset_value != 0.0:
        corrected = corrected + offset_value

    if output_dtype is None:
        output_dtype = np.float32 if (baseline is not None and offset is None) else images.dtype

    if np.issubdtype(output_dtype, np.integer):
        info = np.iinfo(output_dtype)
        below = float(np.mean(corrected < info.min - 0.5))
        above = float(np.mean(corrected > info.max + 0.5))
        if below > 1e-3 or above > 1e-3:
            warnings.warn(
                f"Casting the shading-corrected images to {np.dtype(output_dtype).name} clips "
                f"{below:.2%} of pixels at {info.min} and {above:.2%} at {info.max}. "
                "Use a (larger) `offset`, or a float `output_dtype`, to keep them.",
                UserWarning, stacklevel=2,
            )
        result = _float_to_int(corrected, output_dtype)
    else:
        result = corrected.astype(output_dtype)

    return (result, offset_value) if return_offset else result
