import numpy as np
import warnings
from skimage.exposure import adjust_gamma, rescale_intensity
from scipy.optimize import curve_fit
import scipy.ndimage as ndimage
from scipy.fft import dctn, idctn
import skimage.transform as transform


def ensure_float_and_restore_dtype(func):
    """
    Decorator to safely convert input to float32 for processing,
    and restore the original data type upon return without forcefully
    stretching the intensity range.
    """
    def wrapper(image, *args, **kwargs):
        orig_dtype = image.dtype
        is_integer = np.issubdtype(orig_dtype, np.integer)

        # Determine valid bounds for clipping if original is integer
        if is_integer:
            info = np.iinfo(orig_dtype)
            valid_min, valid_max = info.min, info.max

        # Convert to float32 only if it isn't already a float
        if not np.issubdtype(orig_dtype, np.floating):
            image_float = image.astype(np.float32)
        else:
            image_float = image

        # Execute the core function
        result = func(image_float, *args, **kwargs)

        def convert_output(arr):
            # If the original was float, just cast back to original float precision
            if not is_integer:
                return arr.astype(orig_dtype)

            # If original was integer, we must clip to prevent underflow/overflow wrap-around
            # We do NOT rescale, we only clip out-of-bounds values.
            if arr.max() > valid_max or arr.min() < valid_min:
                warnings.warn(f"Values outside {orig_dtype} range were clipped to [{valid_min}, {valid_max}].")
                arr = np.clip(arr, valid_min, valid_max)

            # Round before casting to integer to avoid truncation errors
            # (e.g., 254.9 becoming 254 instead of 255)
            return np.round(arr).astype(orig_dtype)

        # Handle tuple returns (rescale primary output only)
        if isinstance(result, tuple):
            primary = convert_output(result[0])
            return (primary,) + result[1:]
        elif isinstance(result, dict):
            # Try to convert primary output if it exists in expected keys
            if 'image' in result:
                result['image'] = convert_output(result['image'])
            elif 'corrected' in result:
                result['corrected'] = convert_output(result['corrected'])
            return result
        else:
            return convert_output(result)

    return wrapper

@ensure_float_and_restore_dtype
def contrast_stretching(image, p_min=0.0, p_max=99.9):
    """
    Stretch the intensity range of the image based on percentiles.

    Args:
        image (numpy.ndarray): Input image.
        p_min (float): Lower percentile (0-100). Default 0.0.
        p_max (float): Upper percentile (0-100). Default 99.9.

    Returns:
        numpy.ndarray: Rescaled image.
    """
    v_min, v_max = np.percentile(image, (p_min, p_max))
    print(v_min, v_max)
    image = rescale_intensity(image, in_range=(v_min, v_max))
    return image

def normalize_image(image, lower_percentile=0.5, upper_percentile=99.9, dtype=None):
    """
    Normalize image intensities to lie between 0 and 1 (for float) or 0 and 255/MAX (for int).

    Args:
        image (numpy.ndarray): Input image.
        lower_percentile (float): Percentile for lower bound.
        upper_percentile (float): Percentile for upper bound.
        dtype (type, optional): Output data type. If None, matches input dtype.

    Returns:
        numpy.ndarray: Normalized image.
    """
    lower_bound = np.percentile(image, lower_percentile)
    upper_bound = np.percentile(image, upper_percentile)

    # Clip image to bounds
    image_clipped = np.clip(image, lower_bound, upper_bound)

    # Avoid division by zero
    denom = upper_bound - lower_bound
    if denom == 0:
        denom = 1.0

    # Normalize to 0-1 float
    normalized_image = (image_clipped - lower_bound) / denom

    if dtype is None:
        dtype = image.dtype

    if np.issubdtype(dtype, np.floating):
        return normalized_image.astype(dtype)
    elif np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        normalized_image *= max_val
        return normalized_image.astype(dtype)
    else:
        return normalized_image.astype(dtype)

@ensure_float_and_restore_dtype
def adjust_gamma_per_slice(image, final_gamma=0.8, gamma_fit_func=None, FLIP_Z_AXIS=False):
    """
    Adjusts the gamma of each slice in a 3D image along the Z-axis.

    If gamma_fit_func is None, it uses a linear ramp from 1.0 to final_gamma.
    If gamma_fit_func is provided, it fits the slice intensity decay and calculates
    gamma values to restore intensity to the reference (max fitted) level.

    Args:
        image (numpy.ndarray): The 3D image array in Z, Y, X order.
        final_gamma (float, optional): The gamma value for the last slice (used only if gamma_fit_func is None). Defaults to 0.8.
        gamma_fit_func (str or callable, optional):
            Method to fit intensity decay.
            'exponential' fits y = a * exp(b * x).
            'linear' fits y = m * x + c.
            Can also be a callable model function to be passed to curve_fit.
            If None (default), manual linear ramp is used.
        FLIP_Z_AXIS (bool): Whether to flip the Z-axis for manual gamma ramp application.
                            Ignored or handled implicitly if gamma_fit_func is used.

    Returns:
        numpy.ndarray: A new 3D image array with adjusted gamma values per slice.
    """
    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Image must be a 3D numpy array.")

    # Get the number of slices
    num_slices = image.shape[0]

    # Create an output image array
    adjusted_image = np.empty_like(image)

    if gamma_fit_func is not None:
        # Automatic gamma finding logic

        # 1. Calculate stats (99th percentile) for each slice
        x_data = np.arange(num_slices)
        y_data = np.array([np.percentile(image[i], 99) for i in range(num_slices)])

        # 2. Normalize y_data to [0, 1] based on dtype
        dtype = image.dtype
        if np.issubdtype(dtype, np.integer):
            max_val = np.iinfo(dtype).max
        elif np.issubdtype(dtype, np.floating):
            max_val = 1.0 # Assuming standard float images 0-1. Could use max(image) if unnormalized.
        else:
            max_val = np.max(image) # Fallback

        # Avoid division by zero
        if max_val == 0:
            max_val = 1.0

        y_data_norm = y_data / max_val

        # 3. Fit the model
        if isinstance(gamma_fit_func, str):
            if gamma_fit_func == 'exponential':
                def model(x, a, b):
                    return a * np.exp(b * x)
                # Initial guess: a=start value, b=small decay
                p0 = [y_data_norm[0], -0.1]
            elif gamma_fit_func == 'linear':
                def model(x, m, c):
                    return m * x + c
                p0 = [-0.01, y_data_norm[0]]
            else:
                raise ValueError(f"Unknown gamma_fit_func string: {gamma_fit_func}")
        elif callable(gamma_fit_func):
            model = gamma_fit_func
            p0 = None # Let curve_fit estimate or user should have provided partial?
                      # curve_fit doesn't take p0 via wrapper easily unless we inspect.
                      # We'll assume curve_fit can handle it or fail.
        else:
             raise ValueError("gamma_fit_func must be a string or callable")

        try:
            # For exponential fit on potentially noisy data, we might want to ensure positive inputs if taking logs,
            # but curve_fit works on raw data.
            params, _ = curve_fit(model, x_data, y_data_norm, p0=p0, maxfev=10000)
            y_fit_norm = model(x_data, *params)
        except Exception as e:
            # Fallback or re-raise?
            print(f"Warning: Curve fit failed: {e}. Returning original image.")
            return image

        # 4. Calculate gamma values
        # We want: y_fit_norm[i] ** gamma[i] = y_ref_norm
        # y_ref_norm is the "target" intensity (e.g., the max of the fitted curve)
        y_ref_norm = np.max(y_fit_norm)

        # Avoid mathematical errors
        y_fit_norm = np.clip(y_fit_norm, 1e-9, 1.0) # Clip low values
        y_ref_norm = np.clip(y_ref_norm, 1e-9, 1.0)

        # gamma = log(target) / log(current)
        # Note: log(x) < 0 for x < 1.
        # If current < target, we expect gamma < 1 (brightening).
        # Example: current=0.5, target=1.0 -> log(1)=0 -> gamma=0.
        # Wait, if target=1.0, 0.5^0 = 1. Correct.

        # If y_ref_norm is 1.0 (log is 0), and y_fit_norm < 1.0 (log is neg), gamma is 0.
        # If y_ref_norm < 1.0, say 0.8. y_fit_norm 0.4.
        # log(0.8)/log(0.4) = -0.22 / -0.91 = 0.24.

        gamma_values = np.zeros_like(y_fit_norm)

        log_y_fit = np.log(y_fit_norm)
        log_y_ref = np.log(y_ref_norm)

        # Handle cases where y_fit is 1.0 (log 0)
        mask_valid = np.abs(log_y_fit) > 1e-9
        gamma_values[mask_valid] = log_y_ref / log_y_fit[mask_valid]
        gamma_values[~mask_valid] = 1.0 # If fit is 1.0, gamma 1.0

        # Clip gammas to avoid extreme values
        gamma_values = np.clip(gamma_values, 0.1, 10.0)

    else:
        # Manual linear ramp mode
        gamma_values = np.linspace(1.0, final_gamma, num_slices)

        if FLIP_Z_AXIS:
            gamma_values = gamma_values[::-1]

    # Apply gamma adjustment slice by slice
    for i in range(num_slices):
        adjusted_image[i, :, :] = adjust_gamma(image[i, :, :], gamma=gamma_values[i])

    return adjusted_image

@ensure_float_and_restore_dtype
def adjust_brightness_per_slice(image, final_gamma=0.8, gamma_fit_func=None, FLIP_Z_AXIS=False, method='gamma', return_diagnostic=False):
    """
    Adjusts the brightness of each slice in a 3D image along the Z-axis.

    Args:
        image (numpy.ndarray): The 3D image array in Z, Y, X order.
        final_gamma (float, optional): The gamma value for the last slice (used only if gamma_fit_func is None). Defaults to 0.8.
        gamma_fit_func (str or callable, optional):
            Method to fit intensity decay.
            'exponential' fits y = a * exp(b * x).
            'linear' fits y = m * x + c.
            Can also be a callable model function to be passed to curve_fit.
            If None (default), manual linear ramp is used.
        FLIP_Z_AXIS (bool): Whether to flip the Z-axis for manual gamma ramp application.
                            Ignored or handled implicitly if gamma_fit_func is used.
        method (str): Adjustment method. 'gamma' (default) or 'gain' (linear multiplication).
        return_diagnostic (bool, optional): If True, returns a dictionary containing the adjusted image and the diagnostic figure showing raw Z-axis intensity vs the fitted correction curve. Defaults to False.

    Returns:
        numpy.ndarray or dict: A new 3D image array with adjusted brightness values per slice. If return_diagnostic is True, returns `{"image": adjusted_image, "figure": matplotlib.figure.Figure}`.
    """
    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Image must be a 3D numpy array.")

    print(f"Adjusting brightness using method: {method} (default: gamma)")

    # Get the number of slices
    num_slices = image.shape[0]

    # Create an output image array
    adjusted_image = np.empty_like(image)

    # Pre-calculate global max for clipping logic if needed
    is_integer = np.issubdtype(image.dtype, np.integer)
    max_dtype_val = None
    if is_integer:
        max_dtype_val = np.iinfo(image.dtype).max

    # For float images, check range once to decide clipping behavior
    should_clip_float = False
    if not is_integer:
        # If the image is normalized (max <= 1.0), we clip to 1.0 after gain.
        # If it's arbitrary float, we might not want to clip, or clip to max?
        # Standard skimage practice: if float and range is [0,1], keep it [0,1].
        if np.max(image) <= 1.0:
            should_clip_float = True

    if gamma_fit_func is not None:
        # Automatic brightness finding logic

        # 1. Calculate stats (99th percentile) for each slice
        x_data = np.arange(num_slices)
        y_data = np.array([np.percentile(image[i], 99) for i in range(num_slices)])

        # 2. Normalize y_data to [0, 1] based on dtype
        if is_integer:
            max_val = max_dtype_val
        elif np.issubdtype(image.dtype, np.floating):
            max_val = 1.0 # Assuming standard float images 0-1.
        else:
            max_val = np.max(image) # Fallback

        # Avoid division by zero
        if max_val == 0:
            max_val = 1.0

        y_data_norm = y_data / max_val

        # 3. Fit the model
        if isinstance(gamma_fit_func, str):
            if gamma_fit_func == 'exponential':
                def model(x, a, b):
                    return a * np.exp(b * x)
                # Initial guess: a=start value, b=small decay
                p0 = [y_data_norm[0], -0.1]
            elif gamma_fit_func == 'linear':
                def model(x, m, c):
                    return m * x + c
                p0 = [-0.01, y_data_norm[0]]
            else:
                raise ValueError(f"Unknown gamma_fit_func string: {gamma_fit_func}")
        elif callable(gamma_fit_func):
            model = gamma_fit_func
            p0 = None
        else:
             raise ValueError("gamma_fit_func must be a string or callable")

        try:
            params, _ = curve_fit(model, x_data, y_data_norm, p0=p0, maxfev=10000)
            y_fit_norm = model(x_data, *params)
        except Exception as e:
            print(f"Warning: Curve fit failed: {e}. Returning original image.")
            if return_diagnostic:
                return {"image": image, "diagnostic_data": None}
            return image

        if return_diagnostic:
            diagnostic_data = {
                "x_data": x_data,
                "y_data_norm": y_data_norm,
                "y_fit_norm": y_fit_norm,
                "gamma_fit_func": gamma_fit_func
            }

        # 4. Calculate correction factors (gamma or gain)
        y_ref_norm = np.max(y_fit_norm)

        # Avoid mathematical errors
        y_fit_norm = np.clip(y_fit_norm, 1e-9, 1.0)
        y_ref_norm = np.clip(y_ref_norm, 1e-9, 1.0)

        if method == 'gamma':
            # gamma = log(target) / log(current)
            gamma_values = np.zeros_like(y_fit_norm)
            log_y_fit = np.log(y_fit_norm)
            log_y_ref = np.log(y_ref_norm)

            mask_valid = np.abs(log_y_fit) > 1e-9
            gamma_values[mask_valid] = log_y_ref / log_y_fit[mask_valid]
            gamma_values[~mask_valid] = 1.0

            # Clip gammas
            factors = np.clip(gamma_values, 0.1, 10.0)

        elif method == 'gain':
            # gain = target / current
            gain_values = y_ref_norm / y_fit_norm
            # Clip gain to avoid extreme noise amplification (e.g., max 100x gain)
            factors = np.clip(gain_values, 0.1, 100.0)

        else:
            raise ValueError(f"Unknown method: {method}")

    else:
        # Manual linear ramp mode
        factors = np.linspace(1.0, final_gamma, num_slices)

        if FLIP_Z_AXIS:
            factors = factors[::-1]

    # Apply adjustment slice by slice
    warned_about_clipping = False

    for i in range(num_slices):
        if method == 'gamma':
            adjusted_image[i, :, :] = adjust_gamma(image[i, :, :], gamma=factors[i])
        elif method == 'gain':
            # Linear multiplication
            # Note: image is already typically loaded as is.
            img_slice = image[i, :, :].astype(np.float32)
            img_slice = img_slice * factors[i]

            if is_integer:
                img_slice = np.clip(img_slice, 0, max_dtype_val)
                adjusted_image[i, :, :] = img_slice.astype(image.dtype)
            else:
                # Float image clipping
                if should_clip_float:
                    if not warned_about_clipping and np.any(img_slice > 1.0):
                        warnings.warn("Intensity values were clipped to 1.0 during brightness adjustment.")
                        warned_about_clipping = True

                    img_slice = np.clip(img_slice, 0, 1.0)
                adjusted_image[i, :, :] = img_slice.astype(image.dtype)

    if return_diagnostic:
        return {"image": adjusted_image, "diagnostic_data": diagnostic_data if gamma_fit_func is not None else None}
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

        # We need to handle cases where one array might be None if the user only supplied one?
        # The user code assumed both exist or passed them directly.
        # We'll assume if they are passed, they are valid for indexing.

        in_range_min = p_min_array[i] if p_min_array is not None else "image"
        in_range_max = p_max_array[i] if p_max_array is not None else "image"

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
    else:
        # Expecting high-dimensional input, e.g., 6D (S,T,C,Z,Y,X)
        # We try to extract using the user's pattern: S=0, T=0
        if channel is not None:
            try:
                # Use provided channel.
                # User pattern: img[0, 0, channel, ...]
                img_Z = img[0, 0, channel, ...]
            except IndexError:
                # Fallback: try to just squeeze or look at first channel
                print("Warning: Could not index [0,0,channel,...]. Checking dimensions.")
                # If 4D (C, Z, Y, X)
                if img.ndim == 4:
                    img_Z = img[channel, ...]
                else:
                    img_Z = np.squeeze(img)
        else:
            img_Z = np.squeeze(img)

    # Ensure img_Z is 3D (Z, Y, X) for the split
    if img_Z.ndim > 3:
        # If squeeze didn't reduce enough, take the first element of extra dims
        while img_Z.ndim > 3:
            img_Z = img_Z[0]

    FLAG = False

    # Compare 99.99th percentile of top vs bottom half of Z
    z_mid = img_Z.shape[0] // 2
    top_image_metric = np.percentile(img_Z[0:z_mid, :, :], 99.99)
    bot_image_metric = np.percentile(img_Z[z_mid:-1, :, :], 99.99)
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
        optimization_tol (float): Tolerance for ADMM convergence.
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

    images, original_shape = _prepare_data_for_basic(images, is_3d)

    s_s = images.shape[0]
    original_spatial_shape = images.shape[1:]

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

                if get_darkfield:
                    validA1coeff_idx = B < 1
                    S_flat = S.reshape(-1)
                    mean_S = np.mean(S)
                    S_inmask = S_flat >= mean_S
                    S_outmask = S_flat < mean_S

                    R_0 = np.where(S_inmask[np.newaxis, :] & validA1coeff_idx[:, np.newaxis], R_flat, np.nan)
                    R_1 = np.where(S_outmask[np.newaxis, :] & validA1coeff_idx[:, np.newaxis], R_flat, np.nan)

                    mean_R = np.mean(R_flat)
                    B1_coeff = (np.nanmean(R_0, axis=1) - np.nanmean(R_1, axis=1)) / (mean_R + 1e-6)

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
                if S_sq < 1e-9:
                    S_sq = 1e-9

                R_spatial = R_flat.reshape(s_s, *s_spatial)
                Y_spatial = Y_flat.reshape(s_s, *s_spatial)

                B_new = np.sum(S[np.newaxis, ...] * (R_spatial + Y_spatial / mu), axis=tuple(range(1, Im.ndim))) / S_sq
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

                if norm_ratio <= optimization_tol:
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
        b = b[..., *((np.newaxis,) * flatfield.ndim)]

        corrected = corrected - b

    if output_dtype is None:
        output_dtype = images.dtype

    if np.issubdtype(output_dtype, np.integer):
        # Clip to valid integer range before casting to prevent wrap-around
        min_val = np.iinfo(output_dtype).min
        max_val = np.iinfo(output_dtype).max
        corrected = np.clip(corrected, min_val, max_val)

    return corrected.astype(output_dtype)
