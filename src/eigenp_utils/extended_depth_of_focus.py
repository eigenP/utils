# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scikit-image",
#     "scipy",
# ]
# ///
"""extended depth of focus with patch blending"""

import numpy as np
import skimage.io

from scipy.ndimage import laplace, uniform_filter
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
    selem = disk(1)
    return median(height_map, selem, mode='reflect')


def _get_1d_weight_variants(patch_size, overlap):
    """
    Generate 1D weight variants:
    - full: Taper Start, Taper End (Internal)
    - start: Flat Start, Taper End (Top/Left Boundary)
    - end: Taper Start, Flat End (Bottom/Right Boundary)
    - flat: Flat Start, Flat End (Single Patch)
    """
    x = np.linspace(0, 1, overlap, dtype=np.float32)
    x2 = x * x
    x3 = x2 * x
    taper = (3.0 * x2 - 2.0 * x3).astype(np.float32)

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
    w_flat = w_base

    return w_full, w_start, w_end, w_flat


def _cubic_interp_1d(p0, p1, p2, p3, t):
    """
    Cubic Hermite spline (Catmull-Rom) interpolation evaluated via scalar basis weights.
    p0, p1, p2, p3: ndarray values at t=-1, 0, 1, 2
    t: scalar fractional position between p1 and p2 (0 <= t <= 1)
    """
    t2 = t * t
    t3 = t2 * t
    c0 = 0.5 * (-t + 2.0 * t2 - t3)
    c1 = 0.5 * (2.0 - 5.0 * t2 + 3.0 * t3)
    c2 = 0.5 * (t + 4.0 * t2 - 3.0 * t3)
    c3 = 0.5 * (-t2 + t3)

    return c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3


def _get_fractional_peak(score_matrix):
    """
    Refines the discrete argmax peak using parabolic interpolation on log-scores.

    score_matrix: (Z, H, W)

    Returns:
    peak_z: (H, W) float32
    """
    Z, H, W = score_matrix.shape
    idx = np.argmax(score_matrix, axis=0)  # (H, W)

    z_c = idx
    z_l = np.maximum(z_c - 1, 0)
    z_r = np.minimum(z_c + 1, Z - 1)

    eps = 1e-12
    # Extract values along Z using take_along_axis to avoid dense coordinate meshes
    v_c = np.log(np.take_along_axis(score_matrix, z_c[None, ...], axis=0)[0] + eps)
    v_l = np.log(np.take_along_axis(score_matrix, z_l[None, ...], axis=0)[0] + eps)
    v_r = np.log(np.take_along_axis(score_matrix, z_r[None, ...], axis=0)[0] + eps)

    denom = v_l - 2.0 * v_c + v_r

    delta = np.zeros_like(v_c, dtype=np.float32)
    # Only refine valid non-flat interiors away from Z boundaries
    valid = (np.abs(denom) > 1e-9) & (idx > 0) & (idx < Z - 1)
    delta[valid] = (v_l[valid] - v_r[valid]) / (2.0 * denom[valid])

    np.clip(delta, -0.5, 0.5, out=delta)

    return idx.astype(np.float32) + delta


def best_focus_image(image_or_path, patch_size=None, return_heightmap=False, test=None):
    """
    Extended depth of focus via local focus scoring and weighted subpixel patch blending.
    Dimension order expected: ZYX
    """
    # 1. Load the image
    if isinstance(image_or_path, str):
        img = skimage.io.imread(image_or_path)
    else:
        img = image_or_path

    if img.ndim != 3:
        raise ValueError(f'Image not 3D, instead received {img.ndim} dims')

    original_shape = img.shape[1:]
    H, W = original_shape

    # 2. Determine patch size and padding
    if patch_size is None:
        patch_size = min(original_shape) // 10
    overlap = patch_size // 3  # 33% overlap

    pad_y = (patch_size - H % patch_size) + overlap
    pad_x = (patch_size - W % patch_size) + overlap

    padded_H = H + pad_y
    padded_W = W + pad_x

    n_patches_y = padded_H // (patch_size - overlap)
    n_patches_x = padded_W // (patch_size - overlap)

    max_y_end = (n_patches_y - 1) * (patch_size - overlap) + patch_size
    max_x_end = (n_patches_x - 1) * (patch_size - overlap) + patch_size

    pad_y_needed = max(pad_y, max_y_end - H)
    pad_x_needed = max(pad_x, max_x_end - W)

    padded_H = H + pad_y_needed
    padded_W = W + pad_x_needed

    # 3. Calculate Focus Metric Vectorized
    score_matrix = np.zeros((img.shape[0], n_patches_y, n_patches_x), dtype=np.float32)

    y_starts = np.arange(n_patches_y) * (patch_size - overlap)
    x_starts = np.arange(n_patches_x) * (patch_size - overlap)

    y_centers = y_starts + patch_size // 2
    x_centers = x_starts + patch_size // 2

    # Pre-index mesh outside the slice loop to eliminate redundant allocations
    sample_idx = np.ix_(y_centers, x_centers)

    use_fast_pad = (pad_y_needed < H - 1) and (pad_x_needed < W - 1)

    padded_buffer = np.zeros((padded_H, padded_W), dtype=img.dtype)
    lap_buffer = np.zeros((padded_H, padded_W), dtype=np.float32)
    energy_buffer = np.zeros((padded_H, padded_W), dtype=np.float32)

    for z in range(img.shape[0]):
        if use_fast_pad:
            padded_buffer[:H, :W] = img[z]
            padded_buffer[:H, W:] = img[z][:, -pad_x_needed-1:-1][:, ::-1]
            padded_buffer[H:, :] = padded_buffer[H-pad_y_needed-1:H-1, :][::-1, :]
            slice_padded = padded_buffer
        else:
            slice_padded = np.pad(img[z], ((0, pad_y_needed), (0, pad_x_needed)), mode='reflect')

        laplace(slice_padded, output=lap_buffer)
        np.square(lap_buffer, out=lap_buffer)
        uniform_filter(lap_buffer, size=patch_size, output=energy_buffer, mode='reflect')

        score_matrix[z] = energy_buffer[sample_idx]

    # 4. Select best Z with Subpixel Precision
    height_map_small = _get_fractional_peak(score_matrix)
    height_map_small = apply_median_filter(height_map_small)

    # 5. Combine patches to create the final image
    final_img = np.zeros((padded_H, padded_W), dtype=np.float32)
    counts = np.zeros((padded_H, padded_W), dtype=np.float32)

    # Precompute unique 1D weight variants and unique 2D blending windows
    w_variants = _get_1d_weight_variants(patch_size, overlap)  # (full, start, end, flat)

    def _get_weight_type(idx, total):
        if total == 1:
            return 3  # flat
        if idx == 0:
            return 1  # start
        if idx == total - 1:
            return 2  # end
        return 0      # full

    # Cache unique 2D window arrays (max 9 combinations)
    unique_windows = {}
    for wy_t in range(4):
        for wx_t in range(4):
            unique_windows[(wy_t, wx_t)] = w_variants[wy_t][:, None] * w_variants[wx_t][None, :]

    window_grid = [
        [unique_windows[(_get_weight_type(i, n_patches_y), _get_weight_type(j, n_patches_x))]
         for j in range(n_patches_x)]
        for i in range(n_patches_y)
    ]

    Z_dim = img.shape[0]

    # Fast patch extraction
    def _get_padded_patch(z_idx, y_start, x_start):
        y_end = y_start + patch_size
        x_end = x_start + patch_size

        # Fast path: strictly interior patch requiring no boundary reflection
        if y_end <= H and x_end <= W:
            return img[z_idx, y_start:y_end, x_start:x_end].astype(np.float32)

        # Boundary path
        return np.pad(img[z_idx], ((0, pad_y_needed), (0, pad_x_needed)), mode='reflect')[y_start:y_end, x_start:x_end].astype(np.float32)

    # Main patch reconstruction loop
    for i in range(n_patches_y):
        y_start = i * (patch_size - overlap)
        y_end = y_start + patch_size

        for j in range(n_patches_x):
            x_start = j * (patch_size - overlap)
            x_end = x_start + patch_size

            _2D_window = window_grid[i][j]
            best_z = height_map_small[i, j]

            z_floor = int(np.floor(best_z))
            alpha = float(best_z - z_floor)

            z0 = max(0, min(z_floor - 1, Z_dim - 1))
            z1 = max(0, min(z_floor, Z_dim - 1))
            z2 = max(0, min(z_floor + 1, Z_dim - 1))
            z3 = max(0, min(z_floor + 2, Z_dim - 1))

            if z1 == z2:
                patch = _get_padded_patch(z1, y_start, x_start)
            else:
                p1 = _get_padded_patch(z1, y_start, x_start)
                p2 = _get_padded_patch(z2, y_start, x_start)
                p0 = p1 if z0 == z1 else _get_padded_patch(z0, y_start, x_start)
                p3 = p2 if z3 == z2 else _get_padded_patch(z3, y_start, x_start)

                patch = _cubic_interp_1d(p0, p1, p2, p3, alpha)
                np.maximum(patch, 0.0, out=patch)

            # In-place patch weighting
            patch *= _2D_window

            final_img[y_start:y_end, x_start:x_end] += patch
            counts[y_start:y_end, x_start:x_end] += _2D_window

    # Normalize by accumulated weights
    counts[counts < 1e-9] = 1.0
    final_img /= counts

    # 6. Recrop to original shape
    final_img = final_img[:H, :W]

    if return_heightmap:
        y_c = (y_starts + patch_size // 2).astype(np.float32)
        x_c = (x_starts + patch_size // 2).astype(np.float32)

        interp = RegularGridInterpolator((y_c, x_c), height_map_small, bounds_error=False, fill_value=None)

        gy = np.arange(H, dtype=np.float32)
        gx = np.arange(W, dtype=np.float32)

        try:
            # SciPy 1.9+ broadcastable open grid evaluation
            height_map_full = interp((gy[:, None], gx[None, :])).astype(np.float32)
        except (TypeError, ValueError):
            GY, GX = np.meshgrid(gy, gx, indexing='ij')
            pts = np.array([GY.ravel(), GX.ravel()]).T
            height_map_full = interp(pts).reshape(original_shape).astype(np.float32)

        return final_img, height_map_full

    return final_img
