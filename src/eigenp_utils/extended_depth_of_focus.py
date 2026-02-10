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

    # matth: Use Log-Parabolic Interpolation
    # Focus metrics (squared Laplacian) often follow a Gaussian-like decay: E ~ exp(-(z-z0)^2).
    # Fitting a parabola to the raw Gaussian yields biased peak estimates.
    # Fitting a parabola to log(E) ~ -(z-z0)^2 recovers the peak exactly.
    eps = 1e-12
    v_c = np.log(score_matrix[z_c, grid_y, grid_x] + eps)
    v_l = np.log(score_matrix[z_l, grid_y, grid_x] + eps)
    v_r = np.log(score_matrix[z_r, grid_y, grid_x] + eps)

    # Parabolic fit on Log scores
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

    # 5. Upscale to full resolution using RegularGridInterpolator
    # This provides a smooth, continuous manifold Z(x, y)

    n_patches_y = height_map_small.shape[0]
    n_patches_x = height_map_small.shape[1]

    # Coordinates of the centers where height_map_small is defined
    # Note: In scoring, y_centers = y_starts + patch_size // 2
    # y_starts = i * (patch_size - overlap)
    y_starts = np.arange(n_patches_y) * (patch_size - overlap)
    x_starts = np.arange(n_patches_x) * (patch_size - overlap)

    y_c = y_starts + patch_size // 2
    x_c = x_starts + patch_size // 2

    # matth: Filter out grid points that are primarily in the padded region (reflection).
    # Reflection padding distorts the surface derivative at boundaries.
    # Extrapolation (linear) is safer for continuous surfaces.

    valid_y = (y_c < original_shape[0])
    valid_x = (x_c < original_shape[1])

    # Ensure we keep at least 2 points for interpolation if possible, or 1 if very small
    if np.sum(valid_y) < 2 and n_patches_y >= 2:
        # If we aggressively filtered everything, keep the last ones even if boundary
        # But usually we have points. If image is tiny, we might have 1 point.
        # RegularGridInterpolator handles 1 point (constant).
        pass
    if np.sum(valid_y) == 0: valid_y[:] = True # Fallback
    if np.sum(valid_x) == 0: valid_x[:] = True

    y_c_valid = y_c[valid_y]
    x_c_valid = x_c[valid_x]

    # Slice height_map_small
    # height_map_small is (Ny, Nx)
    h_small_valid = height_map_small[valid_y][:, valid_x]

    # Create interpolator
    # bounds_error=False, fill_value=None -> Linear extrapolation
    interp = RegularGridInterpolator((y_c_valid, x_c_valid), h_small_valid, bounds_error=False, fill_value=None)

    # Target grid coordinates (original image space)
    gy = np.arange(original_shape[0])
    gx = np.arange(original_shape[1])

    # Meshgrid for interpolation (indexing='ij')
    GY, GX = np.meshgrid(gy, gx, indexing='ij')

    # Interpolate height map
    # Passing tuple (GY, GX) is supported in SciPy 1.9+
    # We'll assume a reasonably modern SciPy.
    try:
        height_map_full = interp((GY, GX))
    except (TypeError, ValueError):
        # Fallback for older SciPy
        pts = np.array([GY.ravel(), GX.ravel()]).T
        height_map_full = interp(pts).reshape(original_shape)

    height_map_full = height_map_full.astype(np.float32)

    # 6. Reconstruct Image using Vectorized Per-Pixel Sampling
    # Instead of stitching patches, we sample the volume at V(x, y, Z(x,y))

    Z_dim = img.shape[0]

    # Clip height map to valid Z range [0, Z-1]
    # We clip slightly below Z-1 to ensure floor/ceil stay within bounds
    # However, since we interpolate between floor and floor+1, we need floor+1 < Z
    # If floor = Z-1, then ceil = Z (out of bounds).
    # So max floor must be Z-2.
    # But wait, if z = Z-1, we want value at Z-1.
    # We handle this by clipping indices after floor/ceil calculation.

    # Clip Z to [0, Z-1]
    h_map = np.clip(height_map_full, 0, Z_dim - 1)

    z_floor = np.floor(h_map).astype(int)
    z_ceil = np.ceil(h_map).astype(int)
    alpha = h_map - z_floor

    # Clamp indices to [0, Z-1]
    z_floor = np.clip(z_floor, 0, Z_dim - 1)
    z_ceil = np.clip(z_ceil, 0, Z_dim - 1)

    # Expand dims for broadcasting with take_along_axis: (1, H, W)
    z_floor_exp = z_floor[None, :, :]
    z_ceil_exp = z_ceil[None, :, :]

    # Efficiently gather pixels from the Z-stack
    # img is (Z, H, W)
    v_floor = np.take_along_axis(img, z_floor_exp, axis=0).squeeze(0)
    v_ceil = np.take_along_axis(img, z_ceil_exp, axis=0).squeeze(0)

    # Linear interpolation
    final_img = (1.0 - alpha) * v_floor + alpha * v_ceil

    final_img = final_img.astype(np.float32)

    if return_heightmap:
        return final_img, height_map_full

    return final_img
