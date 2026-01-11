# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scikit-image",
#     "scipy",
# ]
# ///
### extended depth of focus with patch blending

import numpy as np
import skimage.io

from scipy.ndimage import generic_filter, zoom

from skimage.filters import median
from skimage.morphology import disk
from skimage.util import view_as_windows
# # Define the custom function
# def custom_filter(values):
#     center_value = values[values.size // 2]
#     counts = np.bincount(values.astype(int))
#     mode = np.argmax(counts)
#     if counts[mode] >= 3 and abs(mode - center_value) >= 2:
#         return mode
#     else:
#         return center_value

def apply_median_filter(height_map):
    """
    Apply a 3x3 median filter to a 2D array.

    Parameters:
    height_map (ndarray): A 2D array representing the height map.

    Returns:
    ndarray: The filtered 2D array.
    """
    # Define a 2*2+1 diameter (i.e. radius 2) disk structuring element for the median filter
    selem = disk(3)

    # Apply the median filter with the defined structuring element
    filtered_map = median(height_map, selem, mode = 'reflect')

    return filtered_map

def _2D_weight(patch_size, overlap):
    # 1D weight function based on cubic spline
    def weight_1d(x):
        return 3 * x**2 - 2 * x**3

    # Generate the 1D taper profile
    x = np.linspace(0, 1, overlap)
    taper = weight_1d(x)

    # Construct 1D profiles for Y and X axes
    # The profile is 1.0 in the center and tapers to 0.0 at the edges
    # We use multiplicative updates to handle cases where overlap regions intersect (overlap > patch_size/2)
    profile_y = np.ones(patch_size, dtype=np.float64)
    profile_y[:overlap] *= taper
    profile_y[-overlap:] *= taper[::-1]

    profile_x = np.ones(patch_size, dtype=np.float64)
    profile_x[:overlap] *= taper
    profile_x[-overlap:] *= taper[::-1]

    # Apply weights using broadcasting
    # (H, W) * (H, 1) * (1, W)
    # This avoids allocating a full (H, W) weight matrix initialised with ones and then modified in a loop
    weight_2d = profile_y[:, None] * profile_x[None, :]

    return weight_2d

def best_focus_image(image_or_path, patch_size=None, return_heightmap=False, test = None):
    '''
    Expecting an image with dimension order ZYX
    If you have a timelapse, please pass in each individual frame
    e.g. you can slice as frame_img = time_lapse_img[t, ...]
    '''
    # 1. Load the image
    if isinstance(image_or_path, str):
        img = skimage.io.imread(image_or_path)
    else:
        img = image_or_path
    
    # 1.1 Validate ndim
    if img.ndim != 3:
        raise ValueError(f'Image not 3D, instead received {img.ndim} dims')

    original_shape = img.shape[1:]

    # 2. Determine the patch size and pad the image to fit
    if patch_size is None:
        patch_size = min(original_shape) // 10
    # overlap = patch_size // 4  # 25% overlap
    overlap = patch_size // 3  # 33% overlap

    pad_y = (patch_size - img.shape[0] % patch_size) + overlap
    pad_x = (patch_size - img.shape[1] % patch_size) + overlap
    img = np.pad(img, ((0,0), (0, pad_y), (0, pad_x)), mode='reflect')

    # 3. Prepare the empty height map and the storage for final image and counts
    height_map_small = np.zeros((img.shape[1] // (patch_size - overlap), img.shape[2] // (patch_size - overlap)), dtype=int)
    final_img = np.zeros_like(img[0, :, :], dtype=np.float64)
    counts = np.zeros_like(img[0, :, :], dtype=np.float64)

    # 4. Partition into smaller patches and select the best focus level for each patch
    step = patch_size - overlap

    # Use view_as_windows for vectorized patch extraction
    # We want patches of size (Z, patch_size, patch_size)
    # The image is (Z, Y, X). We want to slide along Y and X.
    # So window shape should handle the full Z dimension: (img.shape[0], patch_size, patch_size)
    # And we step only in Y and X: (img.shape[0], step, step)

    # Note: view_as_windows returns a 6D array: (1, n_rows, n_cols, Z, patch_size, patch_size)
    # due to the stride in Z being equal to the size in Z.

    window_shape = (img.shape[0], patch_size, patch_size)
    step_shape = (img.shape[0], step, step)

    patches = view_as_windows(img, window_shape, step=step_shape)

    # Remove singleton dimensions (the first dimension which corresponds to Z steps)
    # Patches shape becomes (n_rows, n_cols, Z, patch_size, patch_size)
    patches = patches.squeeze(axis=0)

    # Handle edge case where view_as_windows returns slightly different number of windows
    # than the manual loop logic if padding wasn't perfect.
    # The original logic iterates range(height_map_small.shape[0/1]).
    # We slice our result to match height_map_small.

    n_rows, n_cols = height_map_small.shape
    patches = patches[:n_rows, :n_cols]

    # Compute standard deviation across the spatial dimensions of the patch (last two axes)
    # Result shape: (n_rows, n_cols, Z)
    sdoL_values = np.std(patches, axis=(-2, -1))

    # Find the Z-index with maximum standard deviation
    # Result shape: (n_rows, n_cols)
    height_map_small[:patches.shape[0], :patches.shape[1]] = np.argmax(sdoL_values, axis=2)


    # Now we apply this custom median filter function to the height_map
    # height_map_small = generic_filter(height_map_small, custom_filter, size=3)
    height_map_small = apply_median_filter(height_map_small)

    # 5. Combine patches to create the final image
    _2D_window = _2D_weight(patch_size, overlap)

    for i in range(height_map_small.shape[0]):
        for j in range(height_map_small.shape[1]):
            y_start = i * (patch_size - overlap)
            x_start = j * (patch_size - overlap)
            best_z = height_map_small[i, j]
            patch = img[best_z, y_start:y_start+patch_size, x_start:x_start+patch_size]
            # print(patch.shape)


            # Create weighted patch
            try:
                weighted_patch = patch * _2D_window
            except ValueError:
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(patch.shape, _2D_window.shape))
                weighted_patch = patch * _2D_window[:min_shape[0], :min_shape[1]]

            # Add the weighted patch to final_img and counts
            final_img[y_start:y_start+patch_size, x_start:x_start+patch_size] += weighted_patch


    # 6. Recrop the image to its original size
    final_img = final_img[:original_shape[0], :original_shape[1]]

    if return_heightmap:
        # 7. Generate a full resolution height map by interpolating height_map_small
        zoom_y = original_shape[0] / height_map_small.shape[0]
        zoom_x = original_shape[1] / height_map_small.shape[1]
        height_map_full = zoom(height_map_small, (zoom_y, zoom_x), order=0)

        return final_img, height_map_full

    return final_img
