import numpy as np
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

def adjust_gamma_per_slice(image, final_gamma=0.8, find_gamma_corr=False, FLIP_Z_AXIS=False):
    """
    Adjusts the gamma of each slice in a 3D image along the Z-axis.
    Gamma starts at 1.0 for the first slice and smoothly changes to 'final_gamma' for the last slice.

    Args:
        image (numpy.ndarray): The 3D image array in Z, Y, X order.
        final_gamma (float, optional): The gamma value for the last slice. Defaults to 0.8.
        find_gamma_corr (bool): Placeholder for auto-gamma finding.
        FLIP_Z_AXIS (bool): Whether to flip the Z-axis for gamma ramp application.

    Returns:
        numpy.ndarray: A new 3D image array with adjusted gamma values per slice.
    """
    # Validate inputs
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Image must be a 3D numpy array.")

    if find_gamma_corr:
        # TODO: Implement gamma correction finding logic
        # Original code reference:
        # def model(x, a, b):
        #     return a * np.exp(b * x)
        # params, covariance = curve_fit(model, x_data, np.log(y_data))
        # a = np.exp(params[0])
        # b = params[1]
        pass

    # Get the number of slices
    num_slices = image.shape[0]

    # Create an output image array
    adjusted_image = np.empty_like(image)

    # Calculate the gamma values for each slice
    gamma_values = np.linspace(1.0, final_gamma, num_slices)

    if FLIP_Z_AXIS:
        gamma_values = gamma_values[::-1]

    # Apply gamma adjustment slice by slice
    for i in range(num_slices):
        adjusted_image[i, :, :] = adjust_gamma(image[i, :, :], gamma=gamma_values[i])

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
