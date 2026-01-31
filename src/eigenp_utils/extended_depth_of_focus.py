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
from scipy.interpolate import RegularGridInterpolator

from skimage.filters import median
from skimage.morphology import disk


def apply_median_filter(height_map):
    """
    Apply a 3x3 median filter to a 2D array.

    Parameters:
    height_map (ndarray): A 2D array representing the height map.

    Returns:
    ndarray: The filtered 2D array.
    """
    # Define a 3x3 (radius 1) disk structuring element for the median filter
    selem = disk(1)

    # Apply the median filter with the defined structuring element
    filtered_map = median(height_map, selem, mode='reflect')

    return filtered_map


def _get_fractional_peak(score_matrix):
    """
    Refines the discrete argmax peak using parabolic interpolation.

    score_matrix: (Z, H, W)

    Returns:
    peak_z: (H, W) float32
    """
    Z, H, W = score_matrix.shape
    idx = np.argmax(score_matrix, axis=0) # (H, W)

    # We need values at idx-1, idx, idx+1
    # Clamp to ensure we don't access out of bounds
    z_c = idx
    z_l = np.maximum(z_c - 1, 0)
    z_r = np.minimum(z_c + 1, Z - 1)

    # Extract values
    # Advanced indexing
    grid_y, grid_x = np.indices((H, W))

    v_c = score_matrix[z_c, grid_y, grid_x]
    v_l = score_matrix[z_l, grid_y, grid_x]
    v_r = score_matrix[z_r, grid_y, grid_x]

    # Parabolic fit
    # Delta = (v_l - v_r) / (2 * (v_l - 2*v_c + v_r))
    denom = v_l - 2*v_c + v_r

    # Handle denominator close to zero (flat or linear)
    # v_c is max, so denom is <= 0.
    delta = np.zeros_like(v_c, dtype=np.float32)
    mask = np.abs(denom) > 1e-9

    # We expect negative denominator for a maximum
    delta[mask] = (v_l[mask] - v_r[mask]) / (2 * denom[mask])

    # Clamp delta to [-0.5, 0.5] to prevent instability
    # Also clamp to 0 if we are at the boundaries of the stack
    boundary_mask = (idx == 0) | (idx == Z - 1)
    delta[boundary_mask] = 0

    delta = np.clip(delta, -0.5, 0.5)

    return idx.astype(np.float32) + delta


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

    # bolt: Removed full 3D padding (img_padded) to save O(Z*H*W) memory.
    # Instead, we apply padding on the fly per slice (scoring) or per patch (reconstruction).
    # This reduces peak memory significantly (e.g. from 1.6GB to ~200MB for typical stacks).

    # Virtual dimensions of the padded space
    padded_H = img.shape[1] + pad_y
    padded_W = img.shape[2] + pad_x

    # 3. Calculate Focus Metric Vectorized
    # Metric: Laplacian Energy (Sum of Squared Laplacian)
    # Optimization: Use float32 to reduce memory usage by 50% compared to float64.

    # Grid dimensions
    n_patches_y = padded_H // (patch_size - overlap)
    n_patches_x = padded_W // (patch_size - overlap)

    # Initialize score matrix: (Z, rows, cols)
    score_matrix = np.zeros((img.shape[0], n_patches_y, n_patches_x), dtype=np.float32)

    # Grid coordinates (centers of patches) for sampling
    y_starts = np.arange(n_patches_y) * (patch_size - overlap)
    x_starts = np.arange(n_patches_x) * (patch_size - overlap)

    y_centers = y_starts + patch_size // 2
    x_centers = x_starts + patch_size // 2

    # Pre-allocate buffers to reuse memory across Z-slices
    # Reduces allocation churn and runtime overhead significantly
    # padded_buffer stores the padded input slice
    # lap_buffer stores the laplacian (float32)
    # energy_buffer stores the uniform filter result (float32)

    # Check if manual padding is safe (pad < dim - 1)
    # If pad is too large, manual slice logic fails, so fallback to np.pad
    H, W = img.shape[1], img.shape[2]
    use_fast_pad = (pad_y < H - 1) and (pad_x < W - 1)

    padded_buffer = np.zeros((padded_H, padded_W), dtype=img.dtype)
    lap_buffer = np.zeros((padded_H, padded_W), dtype=np.float32)
    energy_buffer = np.zeros((padded_H, padded_W), dtype=np.float32)

    # Iterate over Z-slices one by one to keep memory usage low
    for z in range(img.shape[0]):
        # 1. Pad only the current slice (2D)
        # This keeps memory overhead to O(H*W) instead of O(Z*H*W)

        if use_fast_pad:
            # Copy core
            padded_buffer[:H, :W] = img[z]

            # Reflect Right (approx, ensuring dims match)
            # Reflect H rows, pad_x cols
            # input: img[:H, W-pad_x-1:W-1][:, ::-1] -> shape (H, pad_x)
            padded_buffer[:H, W:] = img[z][:, -pad_x-1:-1][:, ::-1]

            # Reflect Bottom (approx)
            # Reflect pad_y rows, all cols (including already padded right side)
            # input: padded_buffer[H-pad_y-1:H-1, :][::-1, :] -> shape (pad_y, W+pad_x)
            padded_buffer[H:, :] = padded_buffer[H-pad_y-1:H-1, :][::-1, :]

            slice_padded = padded_buffer # Reference, no copy
        else:
            # Fallback for large pads
            slice_padded = np.pad(img[z], ((0, pad_y), (0, pad_x)), mode='reflect')

        # 2. Compute Laplacian directly into reusable float32 buffer
        # 'output=lap_buffer' reuses memory
        laplace(slice_padded, output=lap_buffer)

        # 3. Compute Energy (Squared) in-place
        np.square(lap_buffer, out=lap_buffer)

        # 4. Local Average Energy (proxy for sum over patch)
        # Reuses energy_buffer
        uniform_filter(lap_buffer, size=patch_size, output=energy_buffer, mode='reflect')

        # 5. Sample at patch centers
        score_matrix[z] = energy_buffer[np.ix_(y_centers, x_centers)]

        # Explicit delete not needed as buffers are reused, but slice_padded might be a new array in fallback
        if not use_fast_pad:
             del slice_padded

    # 4. Select best Z with Subpixel Precision
    # matth: Use parabolic interpolation to find fractional peak
    height_map_small = _get_fractional_peak(score_matrix)

    # Apply median filter (works on floats, preserves edges while removing outliers)
    height_map_small = apply_median_filter(height_map_small)

    # 5. Per-Pixel Reconstruction (Matth Algorithm)
    # Upscale the height map to pixel resolution and sample the stack directly.
    # This avoids blocking artifacts from patch-based fusion and handles slanted surfaces correctly.

    # Coordinates of the centers where height_map_small is defined
    # These coordinates are in the PADDED space, which aligns with the original image at (0,0).
    y_c = y_centers
    x_c = x_centers

    # Create interpolator
    # bounds_error=False, fill_value=None -> Linear extrapolation
    # height_map_small is (Ny, Nx). RegularGridInterpolator expects (y, x).
    interp = RegularGridInterpolator((y_c, x_c), height_map_small, bounds_error=False, fill_value=None)

    # Target grid coordinates (Original Image Size)
    gy = np.arange(original_shape[0])
    gx = np.arange(original_shape[1])

    # We use meshgrid to generate query points
    GY, GX = np.meshgrid(gy, gx, indexing='ij')

    # Interpolate height map to full resolution
    # Use flattening approach for robustness across SciPy versions
    pts = np.array([GY.ravel(), GX.ravel()]).T
    height_map_full = interp(pts).reshape(original_shape)

    height_map_full = height_map_full.astype(np.float32)

    # 6. Reconstruct Image from Stack using Full Resolution Height Map
    # Z_map = height_map_full
    # Sample Z_floor and Z_ceil

    Z_dim = img.shape[0]

    z_floor = np.floor(height_map_full).astype(int)
    z_ceil = np.ceil(height_map_full).astype(int)
    alpha = height_map_full - z_floor

    # Clamp indices to valid range
    z_floor = np.clip(z_floor, 0, Z_dim - 1)
    z_ceil = np.clip(z_ceil, 0, Z_dim - 1)

    # Vectorized sampling using take_along_axis
    # img is (Z, H, W). We want to sample at (z_floor[y,x], y, x)
    # z_floor[None, :, :] makes it (1, H, W) to broadcast against Z axis?
    # No, take_along_axis requires indices to match shape of arr (except axis).
    # If axis=0, indices should be (1, H, W) or (N, H, W).

    val_floor = np.take_along_axis(img, z_floor[None, :, :], axis=0)[0]
    val_ceil = np.take_along_axis(img, z_ceil[None, :, :], axis=0)[0]

    # Linear interpolation between slices
    final_img = (1.0 - alpha) * val_floor + alpha * val_ceil

    # Cast to float32 (or original dtype?)
    final_img = final_img.astype(np.float32)

    if return_heightmap:
        return final_img, height_map_full

    return final_img
