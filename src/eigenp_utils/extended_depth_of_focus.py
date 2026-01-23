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


def _2D_weight(patch_size, overlap):
    """
    Generate a 2D weight matrix for blending patches, using a cubic spline (smoothstep) taper.
    This ensures C1 continuity across patch boundaries.
    """
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

def _refine_subpixel_parabolic(scores, max_indices):
    """
    Refine integer peak indices using Gaussian (log-parabolic) interpolation.

    Formula:
    delta = (ln(y1) - ln(y3)) / (2 * (ln(y1) - 2*ln(y2) + ln(y3)))
    peak = z + delta

    Where y2 is the peak score, y1 is score at z-1, y3 is score at z+1.
    """
    Z, H, W = scores.shape

    # Initialize with integer indices (casted to float)
    refined_map = max_indices.astype(np.float32)

    # Identify valid pixels for refinement (not at boundaries)
    # mask: 0 < z < Z-1
    mask = (max_indices > 0) & (max_indices < Z - 1)

    # Coordinates of valid pixels
    rows, cols = np.where(mask)
    z_vals = max_indices[rows, cols]

    # Extract scores: center (y2), left (y1), right (y3)
    # Using advanced indexing
    y2 = scores[z_vals, rows, cols]
    y1 = scores[z_vals - 1, rows, cols]
    y3 = scores[z_vals + 1, rows, cols]

    # Avoid log of zero or negative (though energy is non-negative)
    epsilon = 1e-10
    y1 = np.maximum(y1, epsilon)
    y2 = np.maximum(y2, epsilon)
    y3 = np.maximum(y3, epsilon)

    # Log domain
    ly1 = np.log(y1)
    ly2 = np.log(y2)
    ly3 = np.log(y3)

    # Denominator: (ly1 - 2*ly2 + ly3)
    # This is effectively the curvature (2nd derivative). Should be negative for a peak.
    denom = ly1 - 2 * ly2 + ly3

    # Filter out cases where curvature is positive or zero (not a peak, or flat)
    valid_curvature = denom < -1e-6

    # Compute delta only for valid curvature
    # delta = (ly1 - ly3) / (2 * denom)
    # Note: formula sign depends on definition.
    # Parabola: a*x^2 + b*x + c.
    # y(-1)=ly1, y(0)=ly2, y(1)=ly3.
    # b = (ly3 - ly1) / 2
    # a = (ly1 - 2*ly2 + ly3) / 2
    # peak x* = -b / (2a) = - (ly3 - ly1) / (2 * (ly1 - 2*ly2 + ly3))
    # = (ly1 - ly3) / (2 * denom)

    delta = np.zeros_like(ly2)
    delta[valid_curvature] = (ly1[valid_curvature] - ly3[valid_curvature]) / (2 * denom[valid_curvature])

    # Clip delta to [-0.5, 0.5] to ensure we stay within the bin
    delta = np.clip(delta, -0.5, 0.5)

    # Update map
    refined_map[rows, cols] += delta

    return refined_map

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

    # 4. Select best Z
    max_indices = np.argmax(score_matrix, axis=0)

    # 4b. Sub-pixel refinement
    height_map_small = _refine_subpixel_parabolic(score_matrix, max_indices)

    # Apply median filter (works on floats)
    height_map_small = apply_median_filter(height_map_small)

    # 5. Combine patches to create the final image
    # Use float32 for accumulation to save memory
    final_img = np.zeros((padded_H, padded_W), dtype=np.float32)
    counts = np.zeros((padded_H, padded_W), dtype=np.float32) # matth: Restored counts

    _2D_window = _2D_weight(patch_size, overlap)

    def extract_patch(z_idx, y_s, x_s, y_e, x_e, y_rel, x_rel, pb, pr):
        """Helper to extract a patch from a specific Z slice."""
        chunk = img[z_idx, y_s:y_e, x_s:x_e]
        if pb > 0 or pr > 0:
            chunk = np.pad(chunk, ((0, pb), (0, pr)), mode='reflect')
        return chunk[y_rel : y_rel + patch_size, x_rel : x_rel + patch_size].astype(np.float32)

    for i in range(height_map_small.shape[0]):
        for j in range(height_map_small.shape[1]):
            y_start = i * (patch_size - overlap)
            x_start = j * (patch_size - overlap)

            best_z = height_map_small[i, j]

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

            # Slice relative coords
            y_rel = y_start - y_s_clamped
            x_rel = x_start - x_s_clamped

            # Linear interpolation for sub-pixel Z
            floor_z = int(np.floor(best_z))
            ceil_z = min(floor_z + 1, img.shape[0] - 1)
            alpha = best_z - floor_z

            # Clamp bounds
            floor_z = max(0, min(floor_z, img.shape[0] - 1))

            patch_low = extract_patch(floor_z, y_s_clamped, x_s_clamped, y_e_clamped, x_e_clamped, y_rel, x_rel, pad_bottom, pad_right)

            if alpha > 1e-4 and floor_z != ceil_z:
                patch_high = extract_patch(ceil_z, y_s_clamped, x_s_clamped, y_e_clamped, x_e_clamped, y_rel, x_rel, pad_bottom, pad_right)
                patch = (1.0 - alpha) * patch_low + alpha * patch_high
            else:
                patch = patch_low

            # Create weighted patch
            try:
                # weighted_patch = patch * _2D_window
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

    # Optional: cast back to original input dtype?
    # Usually focus stacking keeps float result to preserve dynamic range after blending.
    # We will return float32.

    if return_heightmap:
        # Proper geometric projection of patch scores to pixel grid
        stride = patch_size - overlap
        offset = patch_size // 2

        out_h, out_w = original_shape

        # Map output pixel coordinates to input indices in height_map_small
        # physical_y = index * stride + offset
        # index = (physical_y - offset) / stride

        y_coords = (np.arange(out_h) - offset) / stride
        x_coords = (np.arange(out_w) - offset) / stride

        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        coords = np.stack([yy, xx])

        # Use order=1 (bilinear) for smooth height map, mode='nearest' to extrapolate edges
        height_map_full = map_coordinates(height_map_small, coords, order=1, mode='nearest')

        return final_img, height_map_full

    return final_img
