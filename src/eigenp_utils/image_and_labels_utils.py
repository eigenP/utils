import warnings
import numpy as np
from skimage.segmentation import expand_labels
from scipy.ndimage import uniform_filter


def windowed_slice_projection(nuclei_image, window_size=11, axis=0, operation='max'):
    """
    Apply a thick z-slice operation on a given image array, with the option to either average or take the maximum across the window.
    Assumes the input volume is in (Z, Y, X) order.

    Parameters:
        nuclei_image (np.ndarray): The input 3D image array, expected in (Z, Y, X) order.
        window_size (int): The size of the window for the thick slice operation.
        axis (int): The axis along which to perform the operation. Default is 0 (Z-axis).
        operation (str): Operation to perform on the slices. Options are 'average' or 'max'.

    Returns:
        np.ndarray: An image array with the thick z-slice operation applied.
    """
    if window_size % 2 == 0:
        int(window_size + 1)
        # raise ValueError("Window size must be an odd number.")
        print(f'Using window_size: {window_size} (need odd number)')

    # Calculate the margin to adjust the slicing
    margin = window_size // 2

    # Pad the image along the specified axis
    pad_width = [(0, 0)] * nuclei_image.ndim
    pad_width[axis] = (margin, margin)
    padded_image = np.pad(nuclei_image, pad_width=pad_width, mode='constant', constant_values=0)

    # Initialize output image with zeros of the appropriate data type for the operation
    if operation == 'max':
        nuclei_image_thick = np.full_like(nuclei_image, np.min(nuclei_image), dtype=nuclei_image.dtype)
    else:
        nuclei_image_thick = np.zeros_like(nuclei_image, dtype=np.float64)

    # Apply the sliding window and perform the specified operation
    indices = [slice(None)] * nuclei_image.ndim
    for i in range(window_size):
        indices[axis] = slice(i, i + nuclei_image.shape[axis])
        current_slice = padded_image[tuple(indices)]
        if operation == 'max':
            nuclei_image_thick = np.maximum(nuclei_image_thick, current_slice)
        else:
            nuclei_image_thick += current_slice

    # Finalize the operation
    if operation == 'average':
        nuclei_image_thick /= window_size

    return nuclei_image_thick


def optimized_entire_labels_touching_mask(labels_data, mask):
    """
    Optimized function to mask labels that touch the provided mask.
    Assumes `labels_data` and `mask` are provided in (Z, Y, X) order.

    Parameters:
        labels_data (np.ndarray): The input label image array, expected in (Z, Y, X) order.
        mask (np.ndarray): The mask array indicating regions of interest, expected in (Z, Y, X) order.

    Returns:
        np.ndarray: A filtered label array where only entire labels touching the mask are retained.
    """
    # Expand labels
    dilated_labels_data = expand_labels(labels_data, distance=10)

    # Find all labels that touch the dilated mask
    touching_labels_mask = np.isin(dilated_labels_data, labels_data) & (mask > 0)
    touching_labels = np.unique(dilated_labels_data[touching_labels_mask])

    # Create a mask for the entire extent of touching labels
    entire_touching_labels_mask = np.isin(labels_data, touching_labels)

    # Apply the mask to include entire labels
    optimized_entire_touching_labels = labels_data * entire_touching_labels_mask

    return optimized_entire_touching_labels


def sample_intensity_around_points_optimized(image_3d, points_3d, diameter=5):
    """
    Sample intensities around given points in a 3D image using an optimized mean filter.
    Assumes `image_3d` and `points_3d` are provided in (Z, Y, X) order.

    Parameters:
        image_3d (np.ndarray): The input 3D image array, expected in (Z, Y, X) order.
        points_3d (np.ndarray or list): The coordinates of points to sample, expected in (Z, Y, X) order.
        diameter (int): The diameter of the cube around each point to compute the mean intensity.

    Returns:
        list: A list of sampled mean intensities for each point. Out-of-bounds points will have NaN.
    """
    points_3d = np.asarray(points_3d)

    # Heuristic warning for likely XYZ vs ZYX mismatch
    if len(points_3d) > 0 and image_3d.ndim == 3 and points_3d.ndim == 2 and points_3d.shape[1] == 3:
        max_coords = np.max(points_3d, axis=0)
        shape = image_3d.shape

        # Check if Z-coord is out of bounds but would fit if it were X, and X-coord fits but would fit in Z if swapped
        out_of_bounds_zyx = (max_coords[0] >= shape[0])
        fits_if_xyz = (max_coords[2] < shape[0]) and (max_coords[0] < shape[2])

        if out_of_bounds_zyx and fits_if_xyz:
            warnings.warn(
                "Points appear to be in (X, Y, Z) order. "
                "The maximum Z coordinate exceeds the image Z dimension, but would fit if swapped with X. "
                "Please ensure points are in (Z, Y, X) order to match the image.",
                UserWarning,
                stacklevel=2
            )

    # Compute the radius and ensure the diameter is an integer and odd
    radius = diameter // 2
    diameter = 2 * radius + 1

    # Apply mean filtering using uniform_filter, which computes the mean over a cube
    filtered_image = uniform_filter(image_3d, size=diameter, mode='constant', cval=0.0)

    # Vectorize point sampling:
    # Round points and ensure they are within valid image bounds
    points_3d = np.round(points_3d).astype(int)
    valid_mask = (points_3d >= 0) & (points_3d < np.array(image_3d.shape)[np.newaxis, :])
    valid_mask = np.all(valid_mask, axis=1)

    valid_points = points_3d[valid_mask]
    # Extract valid results using a single operation
    valid_results = filtered_image[tuple(valid_points.T)]  # Transpose to index correctly

    # Assign NaN for out-of-bounds points
    results = np.full(points_3d.shape[0], np.nan)
    results[valid_mask] = valid_results

    return results.tolist()
