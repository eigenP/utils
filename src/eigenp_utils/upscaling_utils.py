import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Union, Tuple, Optional

def upscale_from_grid(
    values: np.ndarray,
    grid_y: np.ndarray,
    grid_x: np.ndarray,
    target_shape: Tuple[int, int],
    kind: str = 'linear'
) -> np.ndarray:
    """
    Upscales a 2D array defined on an irregular (but rectilinear) grid
    to a dense target pixel grid using RegularGridInterpolator.

    Parameters:
    - values: (Ny, Nx) array of values.
    - grid_y: (Ny,) array of Y-coordinates for the rows of 'values'.
    - grid_x: (Nx,) array of X-coordinates for the columns of 'values'.
    - target_shape: (H, W) tuple of the target grid dimensions.
    - kind: Interpolation method ('linear', 'nearest', 'slinear', 'cubic', 'quintic'). Default 'linear'.

    Returns:
    - result: (H, W) float32 array interpolated at pixel centers.
    """
    H, W = target_shape

    # Create interpolator
    # bounds_error=False, fill_value=None -> Linear extrapolation
    try:
        interp = RegularGridInterpolator((grid_y, grid_x), values, method=kind, bounds_error=False, fill_value=None)
    except TypeError:
        # Fallback for older scipy versions that don't support 'method' (default is 'linear')
        interp = RegularGridInterpolator((grid_y, grid_x), values, bounds_error=False, fill_value=None)

    # Target grid coordinates (pixel centers)
    gy = np.arange(H)
    gx = np.arange(W)

    GY, GX = np.meshgrid(gy, gx, indexing='ij')

    # Interpolate
    # Stack coordinates to (N, 2) array for compatibility
    pts = np.stack([GY.ravel(), GX.ravel()], axis=-1)

    result = interp(pts).reshape(H, W)

    return result.astype(np.float32)

def upscale_heightmap(
    height_map_small: np.ndarray,
    original_shape: Tuple[int, int],
    block_size: Union[int, Tuple[int, int]]
) -> np.ndarray:
    """
    Upscales a height map (defined on block centers) to the original pixel grid
    using RegularGridInterpolator to ensure correct spatial alignment.

    This eliminates the systematic shift (~0.5 * block_size) introduced by
    scipy.ndimage.zoom, which assumes corner alignment.

    Parameters:
    - height_map_small: (H_s, W_s) array, values at block centers.
    - original_shape: (H_full, W_full) tuple of the target grid.
    - block_size: (sy, sx) tuple or int block size used for downsampling.

    Returns:
    - height_map_full: (H_full, W_full) float32 array.
    """

    H, W = original_shape
    H_s, W_s = height_map_small.shape

    if isinstance(block_size, int):
        sy, sx = block_size, block_size
    else:
        sy, sx = block_size

    # Coordinates of the centers where height_map_small is defined
    # For block averaging, the center of block k (0-indexed) of size s is:
    # center = k * s + (s - 1) / 2.0
    y_starts = np.arange(H_s) * sy
    x_starts = np.arange(W_s) * sx

    y_c = y_starts + (sy - 1) / 2.0
    x_c = x_starts + (sx - 1) / 2.0

    return upscale_from_grid(height_map_small, y_c, x_c, (H, W))
