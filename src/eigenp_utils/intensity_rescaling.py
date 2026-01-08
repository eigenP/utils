import numpy as np
import warnings
from skimage.exposure import adjust_gamma, rescale_intensity
from scipy.optimize import curve_fit

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
        # Scale to max value of the integer type (e.g., 255 for uint8)
        max_val = np.iinfo(dtype).max
        normalized_image *= max_val
        return normalized_image.astype(dtype)
    else:
        # Fallback for other types
        return normalized_image.astype(dtype)

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

def adjust_brightness_per_slice(image, final_gamma=0.8, gamma_fit_func=None, FLIP_Z_AXIS=False, method='gamma'):
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

    Returns:
        numpy.ndarray: A new 3D image array with adjusted brightness values per slice.
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
            return image

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
