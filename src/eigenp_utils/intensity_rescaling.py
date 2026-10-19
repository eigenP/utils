import numpy as np
import functools
import warnings
from skimage.exposure import rescale_intensity
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
from scipy.fft import dctn, idctn
import skimage.transform as transform


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

@ensure_float_and_restore_dtype
def contrast_stretching(image, p_min=0.0, p_max=99.9):
    """
    Stretch the intensity range of the image based on percentiles, to [0, 1]
    (floats) or [0, dtype max] (integers).
    """
    v_min, v_max = np.nanpercentile(image, (p_min, p_max))
    if v_min == v_max:
        return image
    return rescale_intensity(image, in_range=(v_min, v_max), out_range=(0.0, 1.0))

def normalize_image(image, lower_percentile=0.5, upper_percentile=99.9, dtype=None):
    """
    Normalize image intensities to lie between 0 and 1 (for float) or 0 and MAX (for int).

    NaN pixels are ignored when computing the percentiles; they stay NaN in float
    outputs and become 0 in integer outputs.
    """
    image = np.asarray(image)
    lower_bound, upper_bound = np.nanpercentile(image, (lower_percentile, upper_percentile))

    work = image if image.dtype == np.float32 else image.astype(np.float64)
    denom = upper_bound - lower_bound
    if denom == 0:
        denom = 1.0

    normalized_image = (np.clip(work, lower_bound, upper_bound) - lower_bound) / denom

    if dtype is None:
        dtype = image.dtype

    if np.issubdtype(dtype, np.integer):
        return _float_to_int(normalized_image.astype(np.float64) * np.iinfo(dtype).max, dtype)
    return normalized_image.astype(dtype)

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

# Removed @ensure_float_and_restore_dtype to avoid double-decoration
def adjust_gamma_per_slice(*args, **kwargs):
    """Deprecated. Use correct_z_intensity_decay with method='gamma' instead."""
    warnings.warn(
        "adjust_gamma_per_slice is deprecated. Use correct_z_intensity_decay with method='gamma' instead.",
        DeprecationWarning, stacklevel=2
    )
    if 'gamma_fit_func' in kwargs:
        kwargs['fit_model'] = kwargs.pop('gamma_fit_func')
    if 'FLIP_Z_AXIS' in kwargs:
        kwargs.pop('FLIP_Z_AXIS')
    return correct_z_intensity_decay(*args, method='gamma', **kwargs)


# Removed @ensure_float_and_restore_dtype to avoid double-decoration
def adjust_brightness_per_slice(*args, **kwargs):
    """Deprecated. Use correct_z_intensity_decay instead."""
    warnings.warn(
        "adjust_brightness_per_slice is deprecated. Use correct_z_intensity_decay instead.",
        DeprecationWarning, stacklevel=2
    )
    if 'gamma_fit_func' in kwargs:
        kwargs['fit_model'] = kwargs.pop('gamma_fit_func')
    if 'FLIP_Z_AXIS' in kwargs:
        kwargs.pop('FLIP_Z_AXIS')
    return correct_z_intensity_decay(*args, **kwargs)

@ensure_float_and_restore_dtype
def correct_z_intensity_decay(image, method='gamma', fit_model='exponential', final_gamma=0.8, return_diagnostic=False):
    """
    Corrects Z-axis intensity attenuation using statistical decay modeling.

    The per-slice 99th percentile is fitted with `fit_model` and every slice is
    corrected so that its fitted percentile maps onto the brightest fitted one
    (y_ref):

    - method='gain': multiply slice i by y_ref / y_fit[i].
    - method='gamma': power law on a rescaled intensity, I -> s * (I / s) ** g_i,
      with g_i = log(t) / log(t * y_fit[i] / y_ref), t = min(y_ref, 0.5) for data in
      [0, 1] (t = 0.5 otherwise) and s = y_ref / t. This still maps y_fit[i] exactly
      onto y_ref; for [0, 1] data with y_ref <= 0.5 it is the classic
      g_i = log(y_ref) / log(y_fit[i]). Anchoring at t <= 0.5 keeps g_i away from 0
      when y_ref approaches full scale (log(y_ref) -> 0), which used to collapse every
      factor to the 0.1 clip and flood deep slices.

    Works for integer images (via the dtype decorator), floats in [0, 1] and floats in
    raw intensity units. Gain output is clipped to [0, 1] only for [0, 1] data.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Image must be a 3D numpy array.")
    if method not in ('gamma', 'gain'):
        raise ValueError(f"Unknown correction method: {method}")

    num_slices = image.shape[0]
    adjusted_image = np.empty_like(image)

    data_min, data_max = float(np.nanmin(image)), float(np.nanmax(image))
    unit_range = data_min >= 0.0 and data_max <= 1.0
    if method == 'gamma' and data_min < 0:
        raise ValueError("method='gamma' needs non-negative intensities.")

    diagnostic_data = None
    scale = 1.0

    if fit_model is not None:
        x_data = np.arange(num_slices)
        y_data = np.array([np.nanpercentile(image[i], 99) for i in range(num_slices)])

        y_data_norm = y_data

        try:
            if isinstance(fit_model, str):
                if fit_model not in FIT_MODELS:
                    raise ValueError(f"Unknown fit_model: {fit_model}. Available: {list(FIT_MODELS.keys())}")
                y_fit_norm = FIT_MODELS[fit_model](x_data, y_data_norm)
            elif callable(fit_model):
                y_fit_norm = fit_model(x_data, y_data_norm)
            else:
                raise ValueError("fit_model must be a string or callable.")
        except Exception as e:
            raise RuntimeError(f"Z-decay fitting failed: {e}. Pipeline aborted.")

        if return_diagnostic:
            diagnostic_data = {
                "x_data": x_data,
                "y_data_norm": y_data_norm,
                "y_fit_norm": y_fit_norm,
                "gamma_fit_func": fit_model if isinstance(fit_model, str) else "custom_callable"
            }

        y_ref = float(np.max(y_fit_norm))
        if not y_ref > 0:
            factors = np.ones(num_slices)
        else:
            # Ratio of each slice's fitted percentile to the reference (scale-free)
            ratio = np.clip(np.asarray(y_fit_norm, dtype=np.float64) / y_ref, 1e-9, 1.0)
            if method == 'gamma':
                t = min(y_ref, 0.5) if unit_range else 0.5
                scale = y_ref / t
                factors = np.clip(np.log(t) / np.log(t * ratio), 0.1, 10.0)
            else:
                factors = np.clip(1.0 / ratio, 0.1, 100.0)
    else:
        factors = np.linspace(1.0, final_gamma, num_slices)
        if method == 'gamma' and not unit_range and data_max > 0:
            scale = data_max

    upper = 1.0 if unit_range else np.inf
    lower = 0.0 if data_min >= 0 else -np.inf
    for i in range(num_slices):
        if method == 'gamma':
            adjusted_image[i, :, :] = scale * np.power(image[i, :, :] / scale, factors[i])
        else:
            adjusted_image[i, :, :] = np.clip(image[i, :, :] * factors[i], lower, upper)

    if return_diagnostic:
        return {"image": adjusted_image, "diagnostic_data": diagnostic_data}
    return adjusted_image

def contrast_stretch_per_slice(image, p_min_array=None, p_max_array=None, FLIP_Z_AXIS=False):
    """
    Adjusts the intensity of each slice in a 3D image along the Z-axis using per-slice min/max intensity values.

    Args:
        image (numpy.ndarray): The 3D image array in Z, Y, X order.
        p_min_array (numpy.ndarray): Array of min intensity values for each slice.
        p_max_array (numpy.ndarray): Array of max intensity values for each slice.
        FLIP_Z_AXIS (bool): Whether to flip the order of min/max arrays.

    Returns:
        numpy.ndarray: A new 3D image array with adjusted intensities per slice.
    """
    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Image must be a 3D numpy array.")
    if p_min_array is None and p_max_array is None:
        raise ValueError("p_min_array & p_max_array are None, no difference will be produced in the image")

    # Get the number of slices
    num_slices = image.shape[0]

    # Create an output image array
    adjusted_image = np.empty_like(image)

    if FLIP_Z_AXIS:
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

        adjusted_image[i, :, :] = rescale_intensity(image[i, :, :], in_range=current_in_range)

    return adjusted_image

def test_z_axis_orientation(img, channel=None, return_image=True):
    """
    Check if the Z-axis needs to be flipped based on brightness metrics of top vs bottom halves.
    Assumes standard orientation where the ventral side (brighter) should be processed specifically.

    Args:
        img (numpy.ndarray): Input image (nD). Expected 6D (S,T,C,Z,Y,X) or 3D (Z,Y,X).
        channel (int, optional): Channel index to use for metric calculation if input is high-dimensional.
        return_image (bool): If True, returns (image, flag). If False, returns flag.

    Returns:
        tuple or bool: (flipped_image, flag) if return_image is True, else flag.
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
    top_image_metric = np.percentile(img_Z[0:z_mid, :, :], 99.99)
    bot_image_metric = np.percentile(img_Z[z_mid:, :, :], 99.99)
    print(top_image_metric, bot_image_metric)

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
        sparse_cost_darkfield (float): Weight of the darkfield sparse term. If None, it is estimated.
        get_darkfield (bool): Whether to estimate darkfield.
        fitting_mode (str): Fitting mode, 'approximate' or 'ladmap'.
        epsilon (float): Small value for weight calculation.
        rho (float): ADMM penalty parameter update factor.
        mu_coef (float): Initial penalty parameter coefficient.
        max_mu_coef (float): Maximum penalty parameter coefficient.

    Returns:
        dict: Containing 'flatfield', 'darkfield', and 'baseline'.
    """
    if fitting_mode not in ('approximate', 'ladmap'):
        raise ValueError(f"fitting_mode '{fitting_mode}' is not valid. Use 'approximate' or 'ladmap'.")
    if max_reweight_iterations < 1:
        raise ValueError("max_reweight_iterations must be >= 1.")

    images, original_shape = _prepare_data_for_basic(images, is_3d)

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
            sum_last_S = np.sum(np.abs(last_S))
            if sum_last_S < 1e-9:
                sum_last_S = 1e-9
            mad_flatfield = np.sum(np.abs(S - last_S)) / sum_last_S

            temp_diff = np.sum(np.abs(S - last_S))
            if temp_diff < 1e-7:
                mad_darkfield = 0
            else:
                mad_darkfield = temp_diff / max(np.sum(np.abs(last_S)), 1e-6)

            reweight_score = max(mad_flatfield, mad_darkfield)
            if reweight_score <= reweighting_tol:
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
    output_dtype=None
):
    """
    Applies the basic shading correction to images.

    Args:
        images: Input images.
        flatfield: Estimated flatfield.
        darkfield: Estimated darkfield.
        baseline: Estimated baseline for timelapse correction.
        baseline_smooth_method: Method to smooth baseline ('gaussian').
        baseline_smooth_sigma: Sigma for gaussian smoothing.
        output_dtype: dtype of the result. Defaults to the input dtype, except when
            `baseline` is given: baseline-corrected images are centred on zero (as in
            BaSiCPy), so the default is then float32 rather than clipping the negative
            half of the background to 0. Integer outputs are rounded and clipped.

    Returns:
        Corrected images.
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

    if output_dtype is None:
        output_dtype = np.float32 if baseline is not None else images.dtype

    if np.issubdtype(output_dtype, np.integer):
        return _float_to_int(corrected, output_dtype)

    return corrected.astype(output_dtype)
