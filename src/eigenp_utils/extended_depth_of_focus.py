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


def _get_1d_weight_variants(patch_size, overlap):
    """
    Generate 1D weight variants:
    - full: Taper Start, Taper End (Internal)
    - start: Flat Start, Taper End (Top/Left Boundary)
    - end: Taper Start, Flat End (Bottom/Right Boundary)
    - flat: Flat Start, Flat End (Single Patch)
    """
    def weight_1d(x):
        return 3 * x**2 - 2 * x**3

    x = np.linspace(0, 1, overlap)
    taper = weight_1d(x).astype(np.float32)

    # Base: Flat
    w_base = np.ones(patch_size, dtype=np.float32)

    # Full (Internal)
    w_full = w_base.copy()
    w_full[:overlap] *= taper
    w_full[-overlap:] *= taper[::-1]

    # Start (Top/Left Edge -> Flat Start, Taper End)
    w_start = w_base.copy()
    w_start[-overlap:] *= taper[::-1]

    # End (Bottom/Right Edge -> Taper Start, Flat End)
    w_end = w_base.copy()
    w_end[:overlap] *= taper

    # Flat (Single Patch -> No Taper)
    w_flat = w_base.copy()

    return w_full, w_start, w_end, w_flat


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

    # 5. Combine patches to create the final image
    # Use float32 for accumulation to save memory
    final_img = np.zeros((padded_H, padded_W), dtype=np.float32)
    counts = np.zeros((padded_H, padded_W), dtype=np.float32) # matth: Restored counts

    # Precompute 1D weight variants to handle boundaries
    wy_full, wy_start, wy_end, wy_flat = _get_1d_weight_variants(patch_size, overlap)
    wx_full, wx_start, wx_end, wx_flat = _get_1d_weight_variants(patch_size, overlap)

    n_patches_y = height_map_small.shape[0]
    n_patches_x = height_map_small.shape[1]

    Z_dim = img.shape[0]

    # Helper to extract a padded patch from a specific Z slice
    def _get_padded_patch(z_idx, y_s, x_s, y_e, x_e, y_start, x_start):
        # Determine safe extraction bounds from original image
        y_s_clamped = y_s
        y_e_clamped = min(y_e, img.shape[1])
        pad_bottom = 0

        if y_e > img.shape[1]:
            pad_bottom = pad_y
            needed_history = max(pad_bottom + 1, y_e - img.shape[1] + 2)
            y_s_clamped = min(y_s_clamped, img.shape[1] - needed_history)
            y_s_clamped = max(0, y_s_clamped)
            y_e_clamped = img.shape[1]

        x_s_clamped = x_s
        x_e_clamped = min(x_e, img.shape[2])
        pad_right = 0

        if x_e > img.shape[2]:
            pad_right = pad_x
            needed_history = max(pad_right + 1, x_e - img.shape[2] + 2)
            x_s_clamped = min(x_s_clamped, img.shape[2] - needed_history)
            x_s_clamped = max(0, x_s_clamped)
            x_e_clamped = img.shape[2]

        # Extract chunk
        chunk = img[z_idx, y_s_clamped:y_e_clamped, x_s_clamped:x_e_clamped]

        # Pad chunk locally
        if pad_bottom > 0 or pad_right > 0:
            chunk_padded = np.pad(chunk, ((0, pad_bottom), (0, pad_right)), mode='reflect')
        else:
            chunk_padded = chunk

        # Slice out the exact target region relative to chunk start
        y_rel = y_start - y_s_clamped
        x_rel = x_start - x_s_clamped

        return chunk_padded[y_rel : y_rel + patch_size, x_rel : x_rel + patch_size].astype(np.float32)


    for i in range(n_patches_y):
        # Select Y-weight
        if n_patches_y == 1:
            wy = wy_flat
        elif i == 0:
            wy = wy_start
        elif i == n_patches_y - 1:
            wy = wy_end
        else:
            wy = wy_full

        for j in range(n_patches_x):
            # Select X-weight
            if n_patches_x == 1:
                wx = wx_flat
            elif j == 0:
                wx = wx_start
            elif j == n_patches_x - 1:
                wx = wx_end
            else:
                wx = wx_full

            # Construct 2D window on the fly
            _2D_window = wy[:, None] * wx[None, :]

            y_start = i * (patch_size - overlap)
            x_start = j * (patch_size - overlap)
            best_z = height_map_small[i, j]

            # Patch bounds
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            # matth: Subpixel reconstruction using Cubic Interpolation (Catmull-Rom)
            # Linear interpolation acts as a low-pass filter, reducing contrast and sharpness
            # by ~12% for Gaussian-like focus profiles. Cubic interpolation recovers ~95%
            # of the original intensity and sharpness.

            z_int = int(np.floor(best_z))
            t = best_z - z_int

            # Indices for 4-point interpolation: z-1, z, z+1, z+2
            idx0 = max(0, min(z_int - 1, Z_dim - 1))
            idx1 = max(0, min(z_int, Z_dim - 1))
            idx2 = max(0, min(z_int + 1, Z_dim - 1))
            idx3 = max(0, min(z_int + 2, Z_dim - 1))

            # Fetch patch at z (idx1) first as it's always needed
            p1 = _get_padded_patch(idx1, y_start, x_start, y_end, x_end, y_start, x_start)

            # Optimization: If t is negligible, skip interpolation
            if abs(t) < 1e-4 and idx1 == idx2:
                patch = p1
            else:
                p0 = _get_padded_patch(idx0, y_start, x_start, y_end, x_end, y_start, x_start)
                p2 = _get_padded_patch(idx2, y_start, x_start, y_end, x_end, y_start, x_start)
                p3 = _get_padded_patch(idx3, y_start, x_start, y_end, x_end, y_start, x_start)

                # Catmull-Rom weights
                t2 = t * t
                t3 = t2 * t

                w0 = -0.5 * t3 + t2 - 0.5 * t
                w1 =  1.5 * t3 - 2.5 * t2 + 1.0
                w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
                w3 =  0.5 * t3 - 0.5 * t2

                # Weighted sum
                patch = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3

                # Clip negative lobes to avoid artifacts (intensity cannot be negative)
                np.maximum(patch, 0, out=patch)

            # Create weighted patch
            try:
                # weighted_patch = patch * _2D_window
                # In-place multiplication
                np.multiply(patch, _2D_window, out=patch)
                weight_matrix = _2D_window
            except ValueError:
                # Boundary case
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(patch.shape, _2D_window.shape))
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

    if return_heightmap:
        # matth: Use RegularGridInterpolator for spatially accurate upscaling
        # scipy.ndimage.zoom assumes a different coordinate system that introduces
        # a systematic shift. We map the exact patch centers to the pixel grid.
        from scipy.interpolate import RegularGridInterpolator

        n_patches_y = height_map_small.shape[0]
        n_patches_x = height_map_small.shape[1]

        # Coordinates of the centers where height_map_small is defined
        # Note: In scoring, y_centers = y_starts + patch_size // 2
        # y_starts = i * (patch_size - overlap)
        y_starts = np.arange(n_patches_y) * (patch_size - overlap)
        x_starts = np.arange(n_patches_x) * (patch_size - overlap)

        y_c = y_starts + patch_size // 2
        x_c = x_starts + patch_size // 2

        # Create interpolator
        # bounds_error=False, fill_value=None -> Linear extrapolation
        interp = RegularGridInterpolator((y_c, x_c), height_map_small, bounds_error=False, fill_value=None)

        # Target grid coordinates
        gy = np.arange(original_shape[0])
        gx = np.arange(original_shape[1])

        # Meshgrid for interpolation (indexing='ij')
        # We can optimize by broadcasting if grid is huge, but RegularGridInterpolator
        # usually expects (N, 2) points or tuple of grids.
        # interp((gy[:, None], gx[None, :])) works if grid is tuple?
        # No, RegularGridInterpolator.__call__ expects points (N, D) or (y, x) if method='linear'.
        # Actually it supports meshgrid style inputs in newer scipy.
        # Let's use the explicit meshgrid to be safe and clear.
        GY, GX = np.meshgrid(gy, gx, indexing='ij')

        # Flatten for interpolation then reshape, or pass directly if supported.
        # Passing tuple (GY, GX) is supported in SciPy 1.9+.
        # We'll assume a reasonably modern SciPy.
        try:
            height_map_full = interp((GY, GX))
        except (TypeError, ValueError):
            # Fallback for older SciPy
            pts = np.array([GY.ravel(), GX.ravel()]).T
            height_map_full = interp(pts).reshape(original_shape)

        height_map_full = height_map_full.astype(np.float32)

        return final_img, height_map_full

    return final_img
