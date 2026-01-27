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

from scipy.ndimage import generic_filter, zoom, laplace, uniform_filter, map_coordinates

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

def _get_fractional_peak(scores_stack):
    """
    Given a stack of scores (Z, H, W), find the sub-pixel peak for each pixel (H, W)
    using parabolic interpolation.
    Returns height_map (H, W) in float coordinates.
    """
    Z, H, W = scores_stack.shape
    best_z = np.argmax(scores_stack, axis=0) # (H, W)

    # Indices grid
    rows, cols = np.indices((H, W))

    # Mask for interior peaks (0 < z < Z-1) to allow neighbor access
    mask = (best_z > 0) & (best_z < Z - 1)

    # Initialize delta with zeros
    delta = np.zeros((H, W), dtype=np.float32)

    # Extract y-1, y0, y+1 only where valid
    # Using advanced indexing
    z_int = best_z[mask]
    r_int = rows[mask]
    c_int = cols[mask]

    # y0 is the peak
    y0 = scores_stack[z_int, r_int, c_int]
    y_minus = scores_stack[z_int - 1, r_int, c_int]
    y_plus = scores_stack[z_int + 1, r_int, c_int]

    # Quadratic interpolation
    # Shift = (y_minus - y_plus) / (2 * (y_minus - 2*y0 + y_plus))
    # Note: Curvature (y_minus - 2*y0 + y_plus) should be negative for a max.
    denom = 2 * (y_minus - 2 * y0 + y_plus)

    # Handle small denominator (flat peak or singularity)
    valid_denom = np.abs(denom) > 1e-6

    # Compute delta only where denom is safe
    d = np.zeros_like(y0) # Default 0

    # We only compute where curvature is valid
    # Note: If curvature is positive (minima), the formula finds a minimum.
    # But since y0 >= y_plus and y0 >= y_minus, y_minus - 2y0 + y_plus <= 0 is guaranteed.
    # Just need to check for zero.

    d[valid_denom] = (y_minus[valid_denom] - y_plus[valid_denom]) / denom[valid_denom]

    # Clamp delta to [-0.5, 0.5] to prevent wild extrapolation
    d = np.clip(d, -0.5, 0.5)

    delta[mask] = d

    return best_z.astype(np.float32) + delta


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

    # 4. Select best Z (Sub-pixel refinement)
    # Replaced simple argmax with parabolic interpolation
    height_map_small = _get_fractional_peak(score_matrix)

    # Apply median filter (works on floats too)
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

            # --- Sub-pixel Chunk Extraction ---
            best_z = height_map_small[i, j]

            # Ensure index safety
            best_z = np.clip(best_z, 0, img.shape[0] - 1.00001)
            z_int = int(best_z)
            alpha = best_z - z_int

            # Determine safe extraction bounds from original image
            # If the patch extends beyond original image (into padded region),
            # we need to extract enough "history" for reflection.
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            y_s_clamped = y_start
            y_e_clamped = min(y_end, img.shape[1])
            pad_bottom = 0

            if y_end > img.shape[1]:
                pad_bottom = pad_y
                # Ensure we capture enough context for reflection.
                # np.pad 'reflect' needs at least 'pad_width' elements if mirroring from edge.
                # We need to ensure extraction size >= needed context.
                needed_history = max(pad_bottom + 1, y_end - img.shape[1] + 2)
                y_s_clamped = min(y_s_clamped, img.shape[1] - needed_history)
                y_s_clamped = max(0, y_s_clamped)
                y_e_clamped = img.shape[1]

            x_s_clamped = x_start
            x_e_clamped = min(x_end, img.shape[2])
            pad_right = 0

            if x_end > img.shape[2]:
                pad_right = pad_x
                needed_history = max(pad_right + 1, x_end - img.shape[2] + 2)
                x_s_clamped = min(x_s_clamped, img.shape[2] - needed_history)
                x_s_clamped = max(0, x_s_clamped)
                x_e_clamped = img.shape[2]

            # Extract chunk from floor slice
            chunk0 = img[z_int, y_s_clamped:y_e_clamped, x_s_clamped:x_e_clamped]

            # If sub-pixel and next slice exists, interpolate
            if alpha > 0.001 and z_int < img.shape[0] - 1:
                chunk1 = img[z_int + 1, y_s_clamped:y_e_clamped, x_s_clamped:x_e_clamped]
                # Linear interpolation
                chunk = (1.0 - alpha) * chunk0 + alpha * chunk1
            else:
                chunk = chunk0

            # Pad chunk locally
            if pad_bottom > 0 or pad_right > 0:
                chunk_padded = np.pad(chunk, ((0, pad_bottom), (0, pad_right)), mode='reflect')
            else:
                chunk_padded = chunk

            # Slice out the exact target region relative to chunk start
            y_rel = y_start - y_s_clamped
            x_rel = x_start - x_s_clamped

            patch = chunk_padded[y_rel : y_rel + patch_size, x_rel : x_rel + patch_size].astype(np.float32)

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
        # Correctly resample the height map to align with the original image coordinates.
        # height_map_small contains values at patch centers.
        # We map output pixels (y, x) to the index space of height_map_small.

        H, W = original_shape
        stride = patch_size - overlap
        # Center of the first patch (index 0)
        c0_y = patch_size // 2
        c0_x = patch_size // 2

        # Grid of output coordinates
        # k = (y - c0) / stride
        grid_y, grid_x = np.indices((H, W))
        coord_y = (grid_y - c0_y) / stride
        coord_x = (grid_x - c0_x) / stride

        # Map coordinates
        # mode='nearest' extends the edge values to the boundary (reasonable for padding)
        height_map_full = map_coordinates(
            height_map_small,
            [coord_y, coord_x],
            order=1,
            mode='nearest'
        )

        return final_img, height_map_full

    return final_img
