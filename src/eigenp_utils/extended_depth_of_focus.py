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
    Generate 1D weight profiles for Start, Mid, and End patches.
    Start: Taper only right.
    Mid: Taper both.
    End: Taper only left.
    All: No taper (for single patch).
    """
    def weight_1d(x):
        return 3 * x**2 - 2 * x**3

    x = np.linspace(0, 1, overlap)
    taper = weight_1d(x)

    # Mid: Taper both ends
    w_mid = np.ones(patch_size, dtype=np.float32)
    w_mid[:overlap] *= taper
    w_mid[-overlap:] *= taper[::-1]

    # Start: Taper only end (right)
    w_start = np.ones(patch_size, dtype=np.float32)
    w_start[-overlap:] *= taper[::-1]

    # End: Taper only start (left)
    w_end = np.ones(patch_size, dtype=np.float32)
    w_end[:overlap] *= taper

    # All: No taper
    w_all = np.ones(patch_size, dtype=np.float32)

    return w_start, w_mid, w_end, w_all

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

    # 4. Select best Z (Sub-pixel Refinement)
    # Start with integer max
    z_int = np.argmax(score_matrix, axis=0)
    height_map_small = z_int.astype(np.float32)

    # Refine peaks using Parabolic Interpolation on Log-Scores (Gaussian assumption)
    # Only possible for indices 1 to Z-2 (need neighbors on both sides)

    # Create valid mask
    valid_mask = (z_int > 0) & (z_int < (score_matrix.shape[0] - 1))

    if np.any(valid_mask):
        # Gather neighbors using advanced indexing
        rows, cols = np.indices(z_int.shape)

        # Clip indices for safety (though valid_mask prevents OOB access for the mask)
        z_safe = np.clip(z_int, 1, score_matrix.shape[0] - 2)

        # Log scores (add eps for numerical stability)
        eps = 1e-9

        # Extract 3 points
        s_m1 = score_matrix[z_safe - 1, rows, cols]
        s_0  = score_matrix[z_safe,     rows, cols]
        s_p1 = score_matrix[z_safe + 1, rows, cols]

        # Parabolic fit requires log scores
        # We process all valid interior points. Since z_int is argmax, it is a local max (or flat).

        # Filter for significant signal to avoid fitting noise?
        # (Optional, but here we just fit the peak)

        mask = valid_mask # We trust argmax found a peak

        if np.any(mask):
            y_m1 = np.log(s_m1[mask] + eps)
            y_0  = np.log(s_0[mask]  + eps)
            y_p1 = np.log(s_p1[mask] + eps)

            # Parabolic vertex offset: (y-1 - y+1) / (2 * (y-1 - 2y0 + y+1))
            # Denominator D = y-1 - 2y0 + y+1. Since y0 is max, D <= 0.
            denom = y_m1 - 2*y_0 + y_p1

            # Handle flat peaks (denom ~ 0).
            # If denom is 0, it means y-1 - y0 = y0 - y+1.
            # Since y0 >= y-1 and y0 >= y+1, this implies y-1 = y0 = y+1 (flat).
            # Delta should be 0.

            # If denom is very small, delta can blow up.
            # We treat denom=0 as no curvature -> no update (or undefined).
            # But earlier we showed for s0=s1, denom is non-zero (negative).
            # If y0 = y1 and y-1 < y0. D = y-1 - y0 < 0.

            # Avoid div by zero
            safe_denom = denom.copy()
            safe_denom[np.abs(safe_denom) < 1e-9] = -1e-9

            delta = (y_m1 - y_p1) / (2 * safe_denom)

            # Clamp delta to [-0.5, 0.5] to ensure we stay within the bin
            delta = np.clip(delta, -0.5, 0.5)

            # If original denom was effectively zero (flat 3 points), delta might be unstable?
            # If y-1 = y0 = y+1, num=0, den=0. 0/0.
            # If flat, delta should be 0.
            delta[np.abs(denom) < 1e-9] = 0.0

            height_map_small[mask] += delta

    # Apply median filter (works on floats too)
    height_map_small = apply_median_filter(height_map_small)

    # 5. Combine patches to create the final image
    # Use float32 for accumulation to save memory
    final_img = np.zeros((padded_H, padded_W), dtype=np.float32)
    counts = np.zeros((padded_H, padded_W), dtype=np.float32) # matth: Restored counts

    w_y_start, w_y_mid, w_y_end, w_y_all = _get_1d_weight_variants(patch_size, overlap)
    w_x_start, w_x_mid, w_x_end, w_x_all = _get_1d_weight_variants(patch_size, overlap)

    n_py = height_map_small.shape[0]
    n_px = height_map_small.shape[1]

    for i in range(n_py):
        # Select Y weight profile
        if n_py == 1:
            wy = w_y_all
        elif i == 0:
            wy = w_y_start
        elif i == n_py - 1:
            wy = w_y_end
        else:
            wy = w_y_mid

        for j in range(n_px):
            # Select X weight profile
            if n_px == 1:
                wx = w_x_all
            elif j == 0:
                wx = w_x_start
            elif j == n_px - 1:
                wx = w_x_end
            else:
                wx = w_x_mid

            # Construct 2D window on the fly (cheap O(patch_size))
            weight_matrix = wy[:, None] * wx[None, :]

            y_start = i * (patch_size - overlap)
            x_start = j * (patch_size - overlap)
            best_z = height_map_small[i, j]

            # Sub-pixel interpolation setup
            floor_z = int(np.floor(best_z))
            ceil_z = floor_z + 1
            alpha = best_z - floor_z # Weight for ceil slice

            # Clamp indices
            floor_z = np.clip(floor_z, 0, img.shape[0] - 1)
            ceil_z = np.clip(ceil_z, 0, img.shape[0] - 1)

            # Extract patch with on-demand padding/cropping logic to avoid full padded copy
            y_end = y_start + patch_size
            x_end = x_start + patch_size

            # Determine safe extraction bounds from original image
            # If the patch extends beyond original image (into padded region),
            # we need to extract enough "history" for reflection.
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

            # Slice out the exact target region relative to chunk start
            y_rel = y_start - y_s_clamped
            x_rel = x_start - x_s_clamped
            slice_y = slice(y_s_clamped, y_e_clamped)
            slice_x = slice(x_s_clamped, x_e_clamped)

            # Extract and Pad Function
            def get_padded_patch(z_idx):
                c = img[z_idx, slice_y, slice_x]
                if pad_bottom > 0 or pad_right > 0:
                    c = np.pad(c, ((0, pad_bottom), (0, pad_right)), mode='reflect')
                return c[y_rel : y_rel + patch_size, x_rel : x_rel + patch_size].astype(np.float32)

            # Linear Interpolation between slices
            patch0 = get_padded_patch(floor_z)
            if floor_z != ceil_z and alpha > 1e-4:
                patch1 = get_padded_patch(ceil_z)
                patch = patch0 + alpha * (patch1 - patch0)
            else:
                patch = patch0

            # Create weighted patch
            try:
                np.multiply(patch, weight_matrix, out=patch)
            except ValueError:
                # Boundary case (should be rare with correct padding)
                min_shape = tuple(min(s1, s2) for s1, s2 in zip(patch.shape, weight_matrix.shape))
                patch = patch * weight_matrix[:min_shape[0], :min_shape[1]]
                weight_matrix = weight_matrix[:min_shape[0], :min_shape[1]]

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
