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

from scipy.ndimage import generic_filter, zoom, laplace, uniform_filter

from skimage.filters import median
from skimage.morphology import disk
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
    # Using float32 for weights to match image processing dtype and save memory
    profile_y = np.ones(patch_size, dtype=np.float32)
    profile_y[:overlap] *= taper
    profile_y[-overlap:] *= taper[::-1]

    profile_x = np.ones(patch_size, dtype=np.float32)
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

    # Fix: padding should be based on Y and X dimensions (shape[1] and shape[2]), not Z (shape[0])
    pad_y = (patch_size - img.shape[1] % patch_size) + overlap
    pad_x = (patch_size - img.shape[2] % patch_size) + overlap

    # matth: Padding keeps original dtype to save memory (avoiding premature cast to float)
    img_padded = np.pad(img, ((0,0), (0, pad_y), (0, pad_x)), mode='reflect')

    # 3. Calculate Focus Metric Vectorized
    # Metric: Laplacian Energy (Sum of Squared Laplacian)
    # Optimization: Use float32 to reduce memory usage by 50% compared to float64.

    # Grid dimensions
    n_patches_y = img_padded.shape[1] // (patch_size - overlap)
    n_patches_x = img_padded.shape[2] // (patch_size - overlap)

    # Initialize score matrix: (Z, rows, cols)
    score_matrix = np.zeros((img.shape[0], n_patches_y, n_patches_x), dtype=np.float32)

    # Grid coordinates (centers of patches) for sampling
    y_starts = np.arange(n_patches_y) * (patch_size - overlap)
    x_starts = np.arange(n_patches_x) * (patch_size - overlap)

    y_centers = y_starts + patch_size // 2
    x_centers = x_starts + patch_size // 2

    # Iterate over Z-slices one by one to keep memory usage low
    for z in range(img.shape[0]):
        # 1. Compute Laplacian directly into float32 buffer
        # 'output=np.float32' automatically handles casting from input dtype (e.g. uint16) to float32
        # This AVOIDS allocating a separate float copy of the input slice!
        lap = laplace(img_padded[z], output=np.float32)

        # 2. Compute Energy (Squared) in-place
        np.square(lap, out=lap)

        # 3. Local Average Energy (proxy for sum over patch)
        # We reuse 'lap' buffer for output if filter supports it?
        # uniform_filter(input, output=input) is generally safe for non-recursive filters,
        # but to be safe and avoid boundary artifacts influencing the center during partial writes,
        # we let it allocate or use a temp.
        # Actually uniform_filter(input, output=input) IS safe for 2D.
        # But let's allocate just to be 100% correct about boundaries, it's just one slice.
        mean_energy = uniform_filter(lap, size=patch_size, mode='reflect')

        # 4. Sample at patch centers
        score_matrix[z] = mean_energy[np.ix_(y_centers, x_centers)]

        # Explicitly delete large temps to help GC (though loop scope handles it)
        del lap, mean_energy

    # 4. Select best Z
    height_map_small = np.argmax(score_matrix, axis=0)

    # Apply median filter
    height_map_small = apply_median_filter(height_map_small)

    # 5. Combine patches to create the final image
    # Use float32 for accumulation to save memory
    final_img = np.zeros_like(img_padded[0, :, :], dtype=np.float32)
    counts = np.zeros_like(img_padded[0, :, :], dtype=np.float32) # matth: Restored counts

    _2D_window = _2D_weight(patch_size, overlap)

    for i in range(height_map_small.shape[0]):
        for j in range(height_map_small.shape[1]):
            y_start = i * (patch_size - overlap)
            x_start = j * (patch_size - overlap)
            best_z = height_map_small[i, j]

            # Extract patch and cast to float32 on the fly
            patch = img_padded[best_z, y_start:y_start+patch_size, x_start:x_start+patch_size].astype(np.float32)

            # Create weighted patch
            try:
                # weighted_patch = patch * _2D_window
                # In-place multiplication to save a buffer:
                # patch *= _2D_window
                # (but patch is needed? No, we just add it. 'patch' is a temp copy from astype)
                np.multiply(patch, _2D_window, out=patch)
                weight_matrix = _2D_window
            except ValueError:
                # Boundary case
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(patch.shape, _2D_window.shape))
                # Slicing creates copies/views.
                # Just multiply carefully.
                patch = patch * _2D_window[:min_shape[0], :min_shape[1]]
                weight_matrix = _2D_window[:min_shape[0], :min_shape[1]]

            # Add to accumulators
            final_img[y_start:y_start+patch_size, x_start:x_start+patch_size] += patch
            counts[y_start:y_start+patch_size, x_start:x_start+patch_size] += weight_matrix

    # Normalize by the weight counts
    # Avoid division by zero
    counts[counts < 1e-9] = 1.0
    # In-place division
    np.divide(final_img, counts, out=final_img)

    # 6. Recrop
    final_img = final_img[:original_shape[0], :original_shape[1]]

    # Optional: cast back to original input dtype?
    # Usually focus stacking keeps float result to preserve dynamic range after blending.
    # We will return float32.

    if return_heightmap:
        zoom_y = original_shape[0] / height_map_small.shape[0]
        zoom_x = original_shape[1] / height_map_small.shape[1]
        height_map_full = zoom(height_map_small, (zoom_y, zoom_x), order=0)

        return final_img, height_map_full

    return final_img
