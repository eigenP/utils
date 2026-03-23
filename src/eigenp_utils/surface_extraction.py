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
from scipy.ndimage import binary_erosion, binary_dilation, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
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
    # Uses chunk-based block averaging (binning) instead of padding the full volume.

    pad_z = (sz - image.shape[0] % sz) % sz
    pad_y = (sy - image.shape[1] % sy) % sy
    pad_x = (sx - image.shape[2] % sx) % sx

    new_d = (image.shape[0] + pad_z) // sz
    new_h = (image.shape[1] + pad_y) // sy
    new_w = (image.shape[2] + pad_x) // sx

    img_reduced = np.empty((new_d, new_h, new_w), dtype=np.float32)

    for i in range(new_d):
        z_start = i * sz
        z_end = min(z_start + sz, image.shape[0])
        z_chunk = image[z_start:z_end]

        pad_z_chunk = sz - z_chunk.shape[0]
        if pad_z_chunk > 0 or pad_y > 0 or pad_x > 0:
            z_chunk = np.pad(z_chunk, ((0, pad_z_chunk), (0, pad_y), (0, pad_x)), mode='reflect')

        img_reduced[i] = z_chunk.reshape(sz, new_h, sy, new_w, sx).mean(axis=(0, 2, 4))

    # --- 3. SMOOTHING ---
    img_blurred = ndimage.gaussian_filter(
        img_reduced, sigma=gaussian_sigma, output=np.float32
    )
    # Free memory
    del img_reduced

    # --- 4. NORMALIZE TO UINT8 ---
    min_val = img_blurred.min()
    max_val = img_blurred.max()

    if max_val > min_val:
        # In-place normalization to avoid new array allocation
        img_blurred -= min_val
        img_blurred *= (255.0 / (max_val - min_val))
    else:
        img_blurred.fill(0)

    img_u8 = img_blurred.astype(np.uint8)
    del img_blurred

    # --- 5. CLAHE ---
    if clahe_clip > 0.0:
        img_proc = exposure.equalize_adapthist(img_u8, clip_limit=clahe_clip)
        img_proc = (img_proc * 255).astype(np.uint8)
    else:
        img_proc = img_u8

    if img_u8 is not img_proc:
        del img_u8

    # --- 6. THRESHOLD ---
    thresh = filters.threshold_otsu(img_proc)
    img_mask = img_proc > thresh

    # --- 6.5 TOPOLOGICAL FILTERING ---
    # Remove floating debris (outliers) that are topologically disconnected
    # and significantly smaller than the main surface structure.
    # This prevents the argmax from picking up noise above the actual surface.
    labeled_array, num_features = ndimage.label(img_mask)

    if num_features > 1:
        # Calculate component sizes (bincount is O(N) and very fast)
        # distinct labels are 1..num_features. 0 is background.
        component_sizes = np.bincount(labeled_array.ravel())

        # component_sizes[0] is background, ignore it.
        if len(component_sizes) > 1:
            # Get sizes of foreground components
            foreground_sizes = component_sizes[1:]
            max_size = foreground_sizes.max()

            # Threshold: Keep components at least 10% of the largest one
            size_threshold = 0.1 * max_size

            # Valid labels (indices in foreground_sizes are label-1)
            # We want actual labels, so add 1 to indices
            valid_labels = np.where(foreground_sizes >= size_threshold)[0] + 1

            # Update mask
            # Only keep pixels belonging to valid components
            img_mask = np.isin(labeled_array, valid_labels)

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

    del img_proc, img_mask

    # --- 8. SURFACE UPSCALING VIA HEIGHT MAP ---
    # Use float32 for the height map to preserve subpixel precision during zoom
    surface_z_red = np.full((Y_red, X_red), -1.0, dtype=np.float32)
    surface_z_red[has_surface] = z_refined

    # matth: Robust Upscaling
    # 1. Inpaint invalid regions (-1.0) with nearest valid value to prevent
    #    interpolation artifacts (ringing/undershoot) at boundaries.
    # 2. Use RegularGridInterpolator with correct block centers to eliminate
    #    spatial shift bias caused by ndimage.zoom assuming top-left alignment.

    # Inpainting using Distance Transform
    if np.any(has_surface):
        # Calculate indices of nearest valid pixel for every pixel
        # distance_transform_edt computes distance to background (0).
        # We want distance to valid pixels (has_surface=1).
        # So input is ~has_surface (0 at valid, 1 at invalid).
        _, inds = distance_transform_edt(~has_surface, return_indices=True)
        surface_z_inpainted = surface_z_red[tuple(inds)]
    else:
        # Fallback if no surface found at all
        surface_z_inpainted = np.zeros_like(surface_z_red)

    # Define coordinate systems
    # Reduced grid: centers of blocks. Block i spans [i*S, (i+1)*S). Center: (i+0.5)*S - 0.5.
    y_red_coords = (np.arange(Y_red) + 0.5) * sy - 0.5
    x_red_coords = (np.arange(X_red) + 0.5) * sx - 0.5

    # Handle single pixel dimensions (avoid RegularGridInterpolator error if length < 2)
    # If length is 1, coordinates are just [center]. Interpolation is constant.
    # RegularGridInterpolator requires points to be strictly ascending.
    # It handles length=1 if we are careful, or we can use fill_value.

    # Create Interpolator for Height
    # We use 'cubic' for smooth surface reconstruction, or 'linear' if very small.
    interp_method = 'cubic' if (Y_red > 3 and X_red > 3) else 'linear'

    # Check bounds to avoid errors with empty/small arrays
    if Y_red > 0 and X_red > 0:
        # Handle dimensions with size 1 by padding to size 2 (replicating)
        # RegularGridInterpolator requires at least 2 points for interpolation

        # Copy to avoid modifying original
        grid_y = y_red_coords
        grid_x = x_red_coords
        data_z = surface_z_inpainted
        data_mask = has_surface.astype(np.float32)

        if Y_red == 1:
            grid_y = np.array([y_red_coords[0], y_red_coords[0] + 1.0])
            data_z = np.stack([data_z[0], data_z[0]], axis=0)
            data_mask = np.stack([data_mask[0], data_mask[0]], axis=0)

        if X_red == 1:
            grid_x = np.array([x_red_coords[0], x_red_coords[0] + 1.0])
            # If we padded Y, data is (2, 1). If not, (Y, 1).
            data_z = np.stack([data_z[..., 0], data_z[..., 0]], axis=-1)
            data_mask = np.stack([data_mask[..., 0], data_mask[..., 0]], axis=-1)

        interp_z = RegularGridInterpolator(
            (grid_y, grid_x),
            data_z,
            method=interp_method,
            bounds_error=False,
            fill_value=None # Extrapolate using spline
        )

        # Create Interpolator for Mask (Nearest Neighbor)
        # We interpolate the boolean mask to find valid regions at full resolution
        interp_mask = RegularGridInterpolator(
            (grid_y, grid_x),
            data_mask,
            method='nearest',
            bounds_error=False,
            fill_value=0.0
        )

        # Generate Full Resolution Grid
        Z, Y, X = image.shape
        # We only need to interpolate up to the original image size
        # Generating grid only for relevant area saves time if padding was huge (unlikely here)
        gy = np.arange(Y)
        gx = np.arange(X)

        # Optimization: Pass tuple of coordinates to save memory (requires scipy >= 1.9)
        # Use try-except to fallback for older scipy
        try:
            surface_z_full = interp_z((gy, gx))
            mask_full = interp_mask((gy, gx)) > 0.5

            # Check if output is 1D (older scipy treating tuple as points)
            if surface_z_full.ndim == 1 and Y > 1 and X > 1:
                 raise ValueError("Scipy returned 1D array for grid interpolation")

        except (TypeError, ValueError):
            # Fallback for older scipy (or if tuple is not supported)
            GY, GX = np.meshgrid(gy, gx, indexing='ij')
            # Stack to create (Y, X, 2) array for interpolator
            points = np.stack([GY, GX], axis=-1)
            surface_z_full = interp_z(points)
            mask_full = interp_mask(points) > 0.5
    else:
        Z, Y, X = image.shape
        surface_z_full = np.zeros((Y, X), dtype=np.float32)
        mask_full = np.zeros((Y, X), dtype=bool)

    # Scale Z back to full resolution
    # matth: Correct mapping from block centers to pixels
    # Z_full = (Z_red + 0.5) * sz - 0.5
    surface_z_full = (surface_z_full + 0.5) * sz - 0.5

    # Smoothing (Post-Upscale)
    # We smooth the fully populated (inpainted) surface.
    # This ensures the surface is smooth even at the valid/invalid boundary.
    # Since we mask *after* smoothing, the valid region will be smooth up to the cut.
    surface_z_full = ndimage.gaussian_filter(
        surface_z_full, sigma=(0.75, 0.75)
    )

    # Apply Mask and Sentinel
    surface_z_full[~mask_full] = -1.0

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
