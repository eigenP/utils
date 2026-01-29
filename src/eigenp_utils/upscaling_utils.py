import numpy as np
from scipy.interpolate import RegularGridInterpolator

def interpolate_heightmap(height_map, target_shape, y_coords, x_coords):
    """
    Upscales a height map using RegularGridInterpolator to avoid systematic shifts.

    Parameters:
    -----------
    height_map : np.ndarray
        The small height map to upscale (2D).
    target_shape : tuple
        The shape of the target output (H, W).
    y_coords : np.ndarray
        The Y coordinates of the rows in height_map relative to target_shape.
    x_coords : np.ndarray
        The X coordinates of the cols in height_map relative to target_shape.

    Returns:
    --------
    np.ndarray
        The upscaled height map with shape target_shape.
    """
    # Create interpolator
    # bounds_error=False, fill_value=None -> Linear extrapolation for edges
    # RegularGridInterpolator expects (points_y, points_x, values)
    interp = RegularGridInterpolator((y_coords, x_coords), height_map, bounds_error=False, fill_value=None)

    # Target grid coordinates
    gy = np.arange(target_shape[0])
    gx = np.arange(target_shape[1])

    # Meshgrid for interpolation (indexing='ij')
    GY, GX = np.meshgrid(gy, gx, indexing='ij')

    try:
        height_map_full = interp((GY, GX))
    except (TypeError, ValueError):
        # Fallback for older SciPy versions that don't support tuple args or meshgrids directly
        pts = np.array([GY.ravel(), GX.ravel()]).T
        height_map_full = interp(pts).reshape(target_shape)

    return height_map_full.astype(np.float32)

def get_block_centers(n_blocks, block_size):
    """
    Calculates the center coordinates of blocks for block-averaged data.

    Parameters:
    -----------
    n_blocks : int
        Number of blocks along the dimension.
    block_size : float or int
        Size of each block.

    Returns:
    --------
    np.ndarray
        Array of center coordinates.
    """
    return np.arange(n_blocks) * block_size + (block_size - 1) / 2.0
