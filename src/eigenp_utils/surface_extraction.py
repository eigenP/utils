# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scikit-image",
#     "scipy",
# ]
# ///



### References ###
# https://forum.image.sc/t/surface-peeler-fiji-macro-for-fast-near-real-time-3d-surface-identification-and-extraction/61966
##################

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import exposure, filters
from typing import Union, Tuple

def extract_surface(
    image: np.ndarray,
    downscale_factor: Union[int, Tuple[int, int, int]] = 4,
    gaussian_sigma: float = 4.0,
    clahe_clip: float = 0.00,
    return_heightmap: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:

    # --- 1. HANDLE ANISOTROPIC FACTORS ---
    if isinstance(downscale_factor, int):
        s_tup = (downscale_factor, downscale_factor, downscale_factor)
    else:
        s_tup = tuple(downscale_factor)

    sz, sy, sx = s_tup

    # --- 2. BINNING & DOWNSAMPLING ---
    # Optimized to avoid large intermediate allocation and redundant computation
    # Uses block averaging (binning) instead of dense convolution + subsampling

    pad_z = (sz - image.shape[0] % sz) % sz
    pad_y = (sy - image.shape[1] % sy) % sy
    pad_x = (sx - image.shape[2] % sx) % sx

    if pad_z or pad_y or pad_x:
        image_padded = np.pad(image, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='reflect')
    else:
        image_padded = image

    new_d = image_padded.shape[0] // sz
    new_h = image_padded.shape[1] // sy
    new_w = image_padded.shape[2] // sx

    img_reduced = image_padded.reshape(new_d, sz, new_h, sy, new_w, sx).mean(
        axis=(1, 3, 5), dtype=np.float32
    )

    # --- 3. SMOOTHING ---
    img_blurred = ndimage.gaussian_filter(
        img_reduced, sigma=gaussian_sigma, output=np.float32
    )

    # --- 4. NORMALIZE TO UINT8 ---
    min_val = img_blurred.min()
    max_val = img_blurred.max()

    if max_val > min_val:
        img_norm = (img_blurred - min_val) * (255.0 / (max_val - min_val))
    else:
        img_norm = np.zeros_like(img_blurred)

    img_u8 = img_norm.astype(np.uint8)

    # --- 5. CLAHE ---
    if clahe_clip > 0.0:
        img_proc = exposure.equalize_adapthist(img_u8, clip_limit=clahe_clip)
        img_proc = (img_proc * 255).astype(np.uint8)
    else:
        img_proc = img_u8

    # --- 6. THRESHOLD ---
    thresh = filters.threshold_otsu(img_proc)
    img_mask = img_proc > thresh

    # Identify columns with valid surface (at least one foreground pixel)
    has_surface = np.any(img_mask, axis=0)

    # First True along Z (safe because masked later)
    top_surface_indices = np.argmax(img_mask, axis=0)

    Z_red, Y_red, X_red = img_mask.shape

    # --- 7. SUBPIXEL REFINEMENT ---
    # Convert integer indices to float indices using linear interpolation of intensity
    # around the threshold crossing.
    # z_int is the index where mask becomes True (img_proc[z] > thresh).
    # Since we scan 0->Z, and use argmax, z_int is the first voxel > thresh.
    # The crossing happened between z_int-1 and z_int.

    # Extract intensities at z_int and z_int-1
    # We need to handle z_int=0 carefully (no z_int-1).
    # Use advanced indexing.

    # Filter for columns that actually have a surface
    z_int = top_surface_indices[has_surface]
    y_idx, x_idx = np.where(has_surface)

    # Values at z_int
    val_at = img_proc[z_int, y_idx, x_idx].astype(np.float32)

    # Values at z_int - 1
    # Clip indices to be at least 0. If z_int=0, we just use index 0 twice (delta=0).
    z_prev = np.maximum(z_int - 1, 0)
    val_prev = img_proc[z_prev, y_idx, x_idx].astype(np.float32)

    # Linear interpolation:
    # We want z where I(z) = thresh.
    # I(z) ~ val_prev + (val_at - val_prev) * delta
    # thresh = val_prev + (val_at - val_prev) * delta
    # delta = (thresh - val_prev) / (val_at - val_prev)
    # Refined z = (z_int - 1) + delta = z_int - (1 - delta)
    # Or: z = z_prev + delta.

    denom = val_at - val_prev
    # Avoid division by zero
    denom[denom == 0] = 1.0

    delta = (thresh - val_prev) / denom

    # If z_int was 0, z_prev is 0, val_at == val_prev, delta=0. Refined z=0. Correct.
    z_refined = z_prev.astype(np.float32) + delta

    # --- 8. SURFACE UPSCALING VIA HEIGHT MAP ---
    # Use float32 for the height map to preserve subpixel precision during zoom
    surface_z_red = np.full((Y_red, X_red), -1.0, dtype=np.float32)
    surface_z_red[has_surface] = z_refined

    # Upscale in Y/X only
    from eigenp_utils.upscaling_utils import interpolate_heightmap, get_block_centers

    # Calculate coordinates of block centers in the padded coordinate system
    y_coords = get_block_centers(Y_red, sy)
    x_coords = get_block_centers(X_red, sx)

    # Upscale to the padded image dimensions to match coordinate system
    padded_H = image_padded.shape[1]
    padded_W = image_padded.shape[2]

    surface_z_full = interpolate_heightmap(
        surface_z_red,
        (padded_H, padded_W),
        y_coords,
        x_coords
    )

    # Scale Z back to full resolution
    # Shift to center of the blocks
    surface_z_full = surface_z_full * sz + (sz - 1) / 2.0

    # Ensure dimensions match original image (handle padding/scaling mismatch)
    # The padding added during binning can cause surface_z_full to be larger than image.
    Z, Y, X = image.shape
    surface_z_full = surface_z_full[:Y, :X]

    # FIX: mask-aware smoothing
    valid_mask = surface_z_full >= 0
    surface_z_full[~valid_mask] = 0

    surface_z_full = ndimage.gaussian_filter(
        surface_z_full, sigma=(0.75, 0.75)
    )

    surface_z_full[~valid_mask] = -1

    # Keep float map for return if requested
    surface_z_float = surface_z_full.copy()

    # Convert to integer indices for mask generation
    Z = image.shape[0]
    z_dtype = np.int16 if Z <= 32767 else np.int32
    surface_z_int = np.round(surface_z_full).astype(z_dtype)

    # --- 8. RASTERIZE TO BOOLEAN MASK ---
    Z, Y, X = image.shape
    surface_upscaled = np.zeros((Z, Y, X), dtype=bool)

    # Optimization: Use nonzero indices instead of creating full 2D meshgrids (np.indices)
    # This avoids allocating two (Y, X) arrays, saving O(Y*X) memory (e.g., ~64MB for 2k x 2k)
    valid = (surface_z_int >= 0) & (surface_z_int < Z)
    y_idxs, x_idxs = np.nonzero(valid)

    surface_upscaled[
        surface_z_int[valid],
        y_idxs,
        x_idxs
    ] = True

    if return_heightmap:
        return surface_upscaled, surface_z_float
    return surface_upscaled


# if needed, add padding on the z axis


def pad_z_slices(arr,
                 num_slices: float | int = 0.1,
                 fill_value=None,
                 axis: int = 0,
                 pad_before: bool = True,
                 pad_after: bool = False) -> np.ndarray:
    """
    Pad a NumPy array along a given axis by adding constant-filled slices.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    num_slices : int or float, optional
        If float in [0,1), fraction of the axis length to add (default 0.1 → 10%).
        Otherwise interpreted as an integer count of slices.
    fill_value : scalar, optional
        Fill value for the new slices. If None, uses arr.min().
    axis : int, optional
        Axis index along which to pad (default 0).
    pad_before : bool, optional
        If True, pad before the existing data (default True).
    pad_after : bool, optional
        If True, pad after the existing data (default True).

    Returns
    -------
    np.ndarray
        The padded array.

    Raises
    ------
    ValueError
        If axis is out of range or num_slices < 0.
    """
    arr = np.asanyarray(arr)
    orig_dtype = arr.dtype
    nd = arr.ndim
    if axis < 0 or axis >= nd:
        raise ValueError(f"Axis {axis} is out of bounds for array with {nd} dims")

    size = arr.shape[axis]
    # Determine integer slice count
    if isinstance(num_slices, float) and 0 <= num_slices < 1:
        n = int(round(num_slices * size))
    else:
        n = int(num_slices)
    if n < 0:
        raise ValueError("num_slices must be non-negative")

    # Default fill_value → minimum of array
    if fill_value is None:
        fill_value = arr.min()

    # Build pad_width tuple for each axis
    pad_width = [(0, 0)] * nd
    pad_width[axis] = (n if pad_before else 0,
                       n if pad_after  else 0)

    # Perform constant padding
    padded = np.pad(arr, pad_width, mode='constant', constant_values=fill_value)
    # Ensure dtype matches original
    if padded.dtype != orig_dtype:
        padded = padded.astype(orig_dtype)
    return padded


### Remove 1 % off the dges of the mask to avoid boundary artifacts

def crop_edges(image, crop_percentage_z,crop_percentage_y, crop_percentage_x):
    '''

    Crop a percentage of the image from the x and y edges and return a new image with the same dimension

    Args:
    -- image: 3D numpy array to perform the cropping on
    -- crop_percentage_y: float, percentage of the image to crop from the y edges
    -- crop_percentage_x: float, percentage of the image to crop from the x edges

    Returns:
    image_cropped = 3D numpy array, should have the same dimension as image

    '''

    y_cropped = int(image.shape[1] * crop_percentage_y)
    x_cropped = int(image.shape[2] * crop_percentage_x)

    # Initialize the new array of the same shape with zeros, and set the central region to the cropped portion of the original image
    image_cropped = np.zeros_like(image)

    y_slice = slice(y_cropped, -y_cropped) if y_cropped > 0 else slice(None)
    x_slice = slice(x_cropped, -x_cropped) if x_cropped > 0 else slice(None)

    image_cropped[:, y_slice, x_slice] = image[:, y_slice, x_slice]

    return image_cropped


def expand_surface_z(surface: np.ndarray, thickness: int) -> np.ndarray:
    """
    Fast Z-only expansion using a small Python loop over thickness.
    This is usually faster and more memory-efficient than full vectorization.
    """
    if thickness <= 0:
        return surface

    Z, _, _ = surface.shape
    z0, y0, x0 = np.nonzero(surface)

    expanded = surface.copy()

    for dz in range(1, thickness + 1):
        z = z0 + dz
        valid = z < Z
        expanded[z[valid], y0[valid], x0[valid]] = True

        z = z0 - dz
        valid = z >= 0
        expanded[z[valid], y0[valid], x0[valid]] = True

    return expanded
### Translate if necessary
def adjust_mask_location(mask, translation=(0, 0, 0), morph = None, iterations=1):
    """
      -- when you roll to the end of the image and set off the boundary, set a warning ✅ and the rolled over value from true to false❌
    ==================================
    Adjust the location or morphology of a binary mask by translation or erosion/dilation

    Args:
    -- mask: ndarray; the original binary mask to adjust
    -- translation: a tuple of three integers (z, y, x); the translation in each axis (z, y, x)
    -- morph: string; the morphological operation to perform: 'erode' or 'dilate'; default is None.
    -- iterations: int; the number of iterations for the morphological operation; Default is 1.

    Returns:
    mask_translated: ndarray; the adjusted binary mask
    """
    # Translation
    dz, dy, dx = translation
    Z, Y, X = mask.shape

    # Initialize with zeros to handle "rolled over value from true to false"
    mask_translated = np.zeros_like(mask)

    # 1. Define source (mask) and destination (mask_translated) slices
    # Z-axis
    if dz > 0:
        src_z, dst_z = slice(0, max(0, Z - dz)), slice(dz, Z)
        lost_z = slice(Z - dz, Z)
    elif dz < 0:
        src_z, dst_z = slice(-dz, Z), slice(0, max(0, Z + dz))
        lost_z = slice(0, -dz)
    else:
        src_z, dst_z = slice(None), slice(None)
        lost_z = slice(0, 0)

    # Y-axis
    if dy > 0:
        src_y, dst_y = slice(0, max(0, Y - dy)), slice(dy, Y)
        lost_y = slice(Y - dy, Y)
    elif dy < 0:
        src_y, dst_y = slice(-dy, Y), slice(0, max(0, Y + dy))
        lost_y = slice(0, -dy)
    else:
        src_y, dst_y = slice(None), slice(None)
        lost_y = slice(0, 0)

    # X-axis
    if dx > 0:
        src_x, dst_x = slice(0, max(0, X - dx)), slice(dx, X)
        lost_x = slice(X - dx, X)
    elif dx < 0:
        src_x, dst_x = slice(-dx, X), slice(0, max(0, X + dx))
        lost_x = slice(0, -dx)
    else:
        src_x, dst_x = slice(None), slice(None)
        lost_x = slice(0, 0)

    # 2. Copy valid data
    # Only copy if the shift is within bounds (otherwise result is all zeros)
    if (abs(dz) < Z) and (abs(dy) < Y) and (abs(dx) < X):
        mask_translated[dst_z, dst_y, dst_x] = mask[src_z, src_y, src_x]

    # 3. Check for lost data (Warning)
    # Warn if any part of the mask that contains data is shifted out of the frame
    lost_data = False

    # Check Z loss
    if abs(dz) >= Z:
        if np.any(mask): lost_data = True
    elif dz != 0:
        if np.any(mask[lost_z, :, :]): lost_data = True

    # Check Y loss
    if not lost_data:
        if abs(dy) >= Y:
            if np.any(mask): lost_data = True
        elif dy != 0:
            if np.any(mask[:, lost_y, :]): lost_data = True

    # Check X loss
    if not lost_data:
        if abs(dx) >= X:
            if np.any(mask): lost_data = True
        elif dx != 0:
            if np.any(mask[:, :, lost_x]): lost_data = True

    if lost_data:
        print("Warning: The mask has rolled over the boundaries. Please adjust the value.")

    # Morphological operation
    if morph == 'erode':
        mask_translated = binary_erosion(mask_translated, iterations=iterations)
    elif morph == 'dilate':
        mask_translated = binary_dilation(mask_translated, iterations=iterations)
    elif morph is not None:
        raise ValueError(f"Invalid morphological operation '{morph}'. Options are 'erode' or 'dilate'.")

    return mask_translated
