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
    taper = np.linspace(0, 1, overlap, dtype=np.float32)

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


def _cubic_interp_1d(p0, p1, p2, p3, t):
    """
    Cubic Hermite spline (Catmull-Rom) interpolation.
    p0, p1, p2, p3: values at t=-1, 0, 1, 2
    t: fractional position between p1 and p2 (0 <= t <= 1)
    """
    return 0.5 * (
        (2 * p1) +
        (-p0 + p2) * t +
        (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t ** 2) +
        (-p0 + 3 * p1 - 3 * p2 + p3) * (t ** 3)
    )


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
    overlap = patch_size // 2  # 50% overlap for partition of unity with linear windows

    # padding should be based on Y and X dimensions (shape[1] and shape[2]), not Z (shape[0])
    pad_y = (patch_size - img.shape[1] % patch_size) % patch_size + overlap
    pad_x = (patch_size - img.shape[2] % patch_size) % patch_size + overlap

    # Virtual dimensions of the padded space
    padded_H = img.shape[1] + pad_y
    padded_W = img.shape[2] + pad_x

    # Pre-pad the full 3D volume on Y and X dimensions
    img_padded = np.pad(img, ((0, 0), (0, pad_y), (0, pad_x)), mode='reflect')

    # 3. Calculate Focus Metric Vectorized
    # Metric: Laplacian Energy (Sum of Squared Laplacian)
    # Optimization: Use float32 to reduce memory usage by 50% compared to float64.

    # Grid dimensions
    step = patch_size - overlap
    n_patches_y = (padded_H - overlap) // step
    n_patches_x = (padded_W - overlap) // step

    # Initialize score matrix: (Z, rows, cols)
    score_matrix = np.zeros((img.shape[0], n_patches_y, n_patches_x), dtype=np.float32)

    # Grid coordinates (centers of patches) for sampling
    y_starts = np.arange(n_patches_y) * step
    x_starts = np.arange(n_patches_x) * step

    y_centers = y_starts + patch_size // 2
    x_centers = x_starts + patch_size // 2

    # Pre-allocate buffers to reuse memory across Z-slices
    # Reduces allocation churn and runtime overhead significantly
    # lap_buffer stores the laplacian (float32)
    # energy_buffer stores the uniform filter result (float32)

    lap_buffer = np.zeros((padded_H, padded_W), dtype=np.float32)
    energy_buffer = np.zeros((padded_H, padded_W), dtype=np.float32)

    # Iterate over Z-slices one by one to keep memory usage low
    for z in range(img.shape[0]):
        slice_padded = img_padded[z]

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

    # 4. Select best Z with Subpixel Precision
    # matth: Use parabolic interpolation to find fractional peak
    height_map_small = _get_fractional_peak(score_matrix)

    # Apply median filter (works on floats, preserves edges while removing outliers)
    height_map_small = apply_median_filter(height_map_small)

    # 5. Combine patches to create the final image
    # Use float32 for accumulation to save memory
    final_img = np.zeros((padded_H, padded_W), dtype=np.float32)

    # Precompute 1D weight variants to handle boundaries
    wy_full, wy_start, wy_end, wy_flat = _get_1d_weight_variants(patch_size, overlap)
    wx_full, wx_start, wx_end, wx_flat = _get_1d_weight_variants(patch_size, overlap)

    n_patches_y = height_map_small.shape[0]
    n_patches_x = height_map_small.shape[1]

    Z_dim = img.shape[0]

    # Pre-calculate z parameters vectorized mapping for patches
    z_floor_map = np.floor(height_map_small).astype(np.int32)
    alpha_map = height_map_small - z_floor_map

    # Clamp indices for 4-point stencil globally for the heightmap
    z0_map = np.clip(z_floor_map - 1, 0, Z_dim - 1)
    z1_map = np.clip(z_floor_map, 0, Z_dim - 1)
    z2_map = np.clip(z_floor_map + 1, 0, Z_dim - 1)
    z3_map = np.clip(z_floor_map + 2, 0, Z_dim - 1)

    # Calculate 2D weights for each grid position to avoid allocating per-patch
    # Construct 2D window on the fly
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

        y_start = i * step
        y_end = y_start + patch_size

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

            _2D_window = wy[:, None] * wx[None, :]

            x_start = j * step
            x_end = x_start + patch_size

            # Get spline properties for this patch directly from mapped global structures
            alpha = alpha_map[i, j]
            z0 = z0_map[i, j]
            z1 = z1_map[i, j]
            z2 = z2_map[i, j]
            z3 = z3_map[i, j]

            # matth: Subpixel reconstruction
            # Use Cubic (Catmull-Rom) interpolation to preserve high-frequency content (contrast)
            # Linear interpolation acts as a low-pass filter, degrading the sharpness gained by subpixel depth estimation.

            # Fetch patches directly from the globally pre-padded volume as memory views
            # Optimization: If integer coordinates, skip interpolation
            if z1 == z2:  # z_floor == z_ceil implies alpha=0
                patch = img_padded[z1, y_start:y_end, x_start:x_end].astype(np.float32)
            else:
                p1 = img_padded[z1, y_start:y_end, x_start:x_end]
                p2 = img_padded[z2, y_start:y_end, x_start:x_end]

                # Fetch extra points for cubic spline
                # If at boundaries, clamp (duplicate nearest neighbor)
                # This corresponds to "natural" or "clamped" spline behavior at edges
                p0 = p1 if z0 == z1 else img_padded[z0, y_start:y_end, x_start:x_end]
                p3 = p2 if z3 == z2 else img_padded[z3, y_start:y_end, x_start:x_end]

                patch = _cubic_interp_1d(p0, p1, p2, p3, alpha).astype(np.float32)
                # matth: Clamp negative values that may arise from cubic undershoot (ringing)
                np.maximum(patch, 0, out=patch)

            # Apply strict partition of unity window weights and add directly to target accumulator
            np.multiply(patch, _2D_window, out=patch)
            final_img[y_start:y_end, x_start:x_end] += patch

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
