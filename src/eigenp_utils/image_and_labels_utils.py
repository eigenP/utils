import warnings
import numpy as np
from skimage.segmentation import expand_labels
from scipy.ndimage import uniform_filter, map_coordinates
from skimage import filters, feature, segmentation
import scipy.ndimage as ndi
from itertools import product
from typing import Optional, Tuple, Dict, List, Union


def voronoi_otsu_labeling(image, spot_sigma=2, outline_sigma=2, spacing=None, pixel_sizes=None):
    """
    Segment spots using a marker-based watershed algorithm (Voronoi-Otsu-Labeling).

    Parameters:
    - image: Input image array (2D or 3D, (Z)YX order).
    - spot_sigma: Sigma for the Gaussian blur used to detect local maxima (spots). Can be scalar or tuple.
            If `pixel_sizes` is provided, this is assumed to be in physical units.
    - outline_sigma: Sigma for the Gaussian blur used to create the watershed outline. Can be scalar or tuple.
            If `pixel_sizes` is provided, this is assumed to be in physical units.
    - spacing: Deprecated. Use `pixel_sizes` instead.
    - pixel_sizes: Optional dict mapping dimension to pixel size, e.g., {'Z': 1.0, 'Y': 0.5, 'X': 0.5}.

    Returns:
    - labels: Segmented labeled mask.
    """
    if spacing is not None:
        warnings.warn("The 'spacing' parameter is deprecated. Use 'pixel_sizes' instead.", DeprecationWarning, stacklevel=2)
        if pixel_sizes is None:
            pixel_sizes = spacing

    dim_keys = ['Z', 'Y', 'X'] if image.ndim == 3 else ['Y', 'X']

    # Convert sigmas into tuples per axis, scaling by pixel size if provided
    def _process_sigma(sigma_val):
        if not isinstance(sigma_val, (list, tuple)):
            sigma_val = [sigma_val] * image.ndim

        if pixel_sizes is not None:
            sigma_val = [s / pixel_sizes.get(dim, 1.0) for s, dim in zip(sigma_val, dim_keys)]

        return tuple(sigma_val)

    spot_sigma_tuple = _process_sigma(spot_sigma)
    outline_sigma_tuple = _process_sigma(outline_sigma)

    # Step 1: Apply Gaussian blur for spot detection directly into a new variable
    blurred_spot = filters.gaussian(image, sigma=spot_sigma_tuple)

    # Step 2: Detect local maxima (spots)
    # TODO: consider scaling min_distance based on pixel sizes
    local_maxi = feature.peak_local_max(blurred_spot, min_distance=3, labels=None)

    # local_maxi contains the coordinates of the peaks
    peaks = np.zeros_like(blurred_spot, dtype=bool)
    if len(local_maxi) > 0:
        peaks[tuple(local_maxi.T)] = True  # Transpose and index to set True at peak locations

    # Step 3: Reuse blurred_spot for outline blurring to save memory
    blurred_outline = filters.gaussian(image, sigma=outline_sigma_tuple, out=blurred_spot)

    # Step 4: Perform Otsu thresholding directly
    thresh = filters.threshold_otsu(blurred_outline)
    binary_mask = blurred_outline > thresh  # In-place comparison, no need for extra space

    # Step 5: Label the markers (local maxima)
    markers, _ = ndi.label(peaks, output=np.int32)  # Use int32 if default int64 is not necessary

    # Step 6: Perform marker-based watershed segmentation
    labels = segmentation.watershed(-blurred_outline, markers, mask=binary_mask, compactness=0.5)

    return labels


def windowed_slice_projection(nuclei_image, window_size=11, axis=0, operation='max', pixel_sizes=None):
    """
    Apply a thick z-slice operation on a given image array, with the option to either average or take the maximum across the window.
    Assumes the input volume is in (Z, Y, X) order.

    Parameters:
        nuclei_image (np.ndarray): The input 3D image array, expected in (Z, Y, X) order.
        window_size (int or float): The size of the window for the thick slice operation.
            If `pixel_sizes` is provided, this is assumed to be in physical units.
        axis (int): The axis along which to perform the operation. Default is 0 (Z-axis).
        operation (str): Operation to perform on the slices. Options are 'average' or 'max'.
        pixel_sizes (dict, optional): Dictionary specifying physical pixel dimensions, e.g., {'Z': 0.79, 'Y': 0.468, 'X': 0.468}.

    Returns:
        np.ndarray: An image array with the thick z-slice operation applied.
    """
    if pixel_sizes is None:
        warnings.warn("pixel_sizes not provided; defaulting to isotropic pixel size of 1.0.", UserWarning, stacklevel=2)

    if pixel_sizes is not None:
        dim_keys = ['Z', 'Y', 'X']
        pixel_size_axis = pixel_sizes.get(dim_keys[axis], 1.0)
        window_size = int(round(window_size / pixel_size_axis))
    else:
        window_size = int(window_size)

    if window_size < 1:
        window_size = 1

    if window_size % 2 == 0:
        window_size += 1
        print(f'Using window_size: {window_size} (need odd number)')

    # Calculate the margin to adjust the slicing
    margin = window_size // 2

    # Pad the image along the specified axis
    pad_width = [(0, 0)] * nuclei_image.ndim
    pad_width[axis] = (margin, margin)
    padded_image = np.pad(nuclei_image, pad_width=pad_width, mode='constant', constant_values=0)

    # Initialize output image with zeros of the appropriate data type for the operation
    if operation == 'max':
        nuclei_image_thick = np.full_like(nuclei_image, np.min(nuclei_image), dtype=nuclei_image.dtype)
    else:
        nuclei_image_thick = np.zeros_like(nuclei_image, dtype=np.float64)

    # Apply the sliding window and perform the specified operation
    indices = [slice(None)] * nuclei_image.ndim
    for i in range(window_size):
        indices[axis] = slice(i, i + nuclei_image.shape[axis])
        current_slice = padded_image[tuple(indices)]
        if operation == 'max':
            nuclei_image_thick = np.maximum(nuclei_image_thick, current_slice)
        else:
            nuclei_image_thick += current_slice

    # Finalize the operation
    if operation == 'average':
        nuclei_image_thick /= window_size

    return nuclei_image_thick


def optimized_entire_labels_touching_mask(labels_data, mask, distance=10, pixel_sizes=None):
    """
    Optimized function to mask labels that touch the provided mask.
    Assumes `labels_data` and `mask` are provided in (Z, Y, X) order.

    Parameters:
        labels_data (np.ndarray): The input label image array, expected in (Z, Y, X) order.
        mask (np.ndarray): The mask array indicating regions of interest, expected in (Z, Y, X) order.
        distance (int or float): The distance to dilate labels before finding touching ones.
            If `pixel_sizes` is provided, this is assumed to be in physical units.
        pixel_sizes (dict, optional): Dictionary specifying physical pixel dimensions, e.g., {'Z': 0.79, 'Y': 0.468, 'X': 0.468}.

    Returns:
        np.ndarray: A filtered label array where only entire labels touching the mask are retained.
    """
    if pixel_sizes is None:
        warnings.warn("pixel_sizes not provided; defaulting to isotropic pixel size of 1.0.", UserWarning, stacklevel=2)

    # Expand labels
    if pixel_sizes is not None:
        dim_keys = ['Z', 'Y', 'X'] if labels_data.ndim == 3 else ['Y', 'X']
        spacing = tuple(pixel_sizes.get(dim, 1.0) for dim in dim_keys)
        try:
            dilated_labels_data = expand_labels(labels_data, distance=distance, spacing=spacing)
        except TypeError:
            raise NotImplementedError(
                "Anisotropic expansion requires scikit-image >= 0.21.0 for the `spacing` parameter. "
                "Uniform pixel fallback causes severe structural distortion."
            )
    else:
        dilated_labels_data = expand_labels(labels_data, distance=int(distance))

    # Find all labels that touch the dilated mask
    touching_labels_mask = np.isin(dilated_labels_data, labels_data) & (mask > 0)
    touching_labels = np.unique(dilated_labels_data[touching_labels_mask])

    # Create a mask for the entire extent of touching labels
    entire_touching_labels_mask = np.isin(labels_data, touching_labels)

    # Apply the mask to include entire labels
    optimized_entire_touching_labels = labels_data * entire_touching_labels_mask

    return optimized_entire_touching_labels


def sample_intensity_around_points(image_3d, points_3d, diameter=5, pixel_sizes=None):
    """
    Sample intensities around given points in a 3D image using an optimized mean filter.
    Assumes `image_3d` and `points_3d` are provided in (Z, Y, X) order.

    Parameters:
        image_3d (np.ndarray): The input 3D image array, expected in (Z, Y, X) order.
        points_3d (np.ndarray or list): The coordinates of points to sample, expected in (Z, Y, X) order.
        diameter (int or float): The diameter of the cube around each point to compute the mean intensity.
            If `pixel_sizes` is provided, this is assumed to be in physical units.
        pixel_sizes (dict, optional): Dictionary mapping dimension to pixel size, e.g., {'Z': 1.0, 'Y': 0.5, 'X': 0.5}.

    Returns:
        list: A list of sampled mean intensities for each point. Out-of-bounds points will have NaN.
    """
    points_3d = np.asarray(points_3d)

    # Heuristic warning for likely XYZ vs ZYX mismatch
    if len(points_3d) > 0 and image_3d.ndim == 3 and points_3d.ndim == 2 and points_3d.shape[1] == 3:
        max_coords = np.max(points_3d, axis=0)
        shape = image_3d.shape

        # Check if Z-coord is out of bounds but would fit if it were X, and X-coord fits but would fit in Z if swapped
        out_of_bounds_zyx = (max_coords[0] >= shape[0])
        fits_if_xyz = (max_coords[2] < shape[0]) and (max_coords[0] < shape[2])

        if out_of_bounds_zyx and fits_if_xyz:
            warnings.warn(
                "Points appear to be in (X, Y, Z) order. "
                "The maximum Z coordinate exceeds the image Z dimension, but would fit if swapped with X. "
                "Please ensure points are in (Z, Y, X) order to match the image.",
                UserWarning,
                stacklevel=2
            )

    if pixel_sizes is None:
        warnings.warn("pixel_sizes not provided; defaulting to isotropic pixel size of 1.0.", UserWarning, stacklevel=2)

    # Compute filter dimensions in pixel units cleanly
    if pixel_sizes is not None:
        dim_keys = ['Z', 'Y', 'X'] if image_3d.ndim == 3 else ['Y', 'X']
        spacing = [pixel_sizes.get(dim, 1.0) for dim in dim_keys]
    else:
        spacing = [1.0] * image_3d.ndim

    filter_size = []
    for s in spacing:
        # Calculate radius in pixels from physical radius (diameter / 2)
        physical_radius = diameter / 2.0
        pixel_radius = int(round(physical_radius / s))
        # Window size must be odd to center perfectly on the coordinate point
        filter_size.append(2 * pixel_radius + 1)

    filter_size = tuple(filter_size)

    # Use 'reflect' or 'nearest' to ensure boundary points are not diluted by zero-padding
    filtered_image = uniform_filter(image_3d, size=filter_size, mode='reflect')

    # Vectorize point sampling:
    # Round points and ensure they are within valid image bounds
    points_3d = np.round(points_3d).astype(int)
    valid_mask = (points_3d >= 0) & (points_3d < np.array(image_3d.shape)[np.newaxis, :])
    valid_mask = np.all(valid_mask, axis=1)

    valid_points = points_3d[valid_mask]
    # Extract valid results using a single operation
    valid_results = filtered_image[tuple(valid_points.T)]  # Transpose to index correctly

    # Assign NaN for out-of-bounds points
    results = np.full(points_3d.shape[0], np.nan)
    results[valid_mask] = valid_results

    return results.tolist()


def sample_intensity_along_surface_normals(image, surface_points_grid, thickness=3, num_steps=5, interpolation='nearest', pixel_sizes=None):
    """
    Samples an image along the normals of a precomputed surface grid, accounting for physical pixel sizes.

    Parameters:
    -----------
    image : ndarray
        The 3D image volume to sample from. Assumed to be in (Z, Y, X) order.
    surface_points_grid : ndarray
        3D array of shape (U, V, 3) representing the spatial coordinates of the mesh in pixel space.
        The last dimension must correspond to (Z, Y, X) coordinates.
    thickness : float, optional
        The total sampling distance along the normal vector, centered at the surface point.
        Measured in physical units (if `pixel_sizes` is provided) or pixel units. Default is 3.
    num_steps : int, optional
        The number of discrete points to sample along the specified `thickness`.
        If `num_steps` is greater than `thickness`, this results in subpixel sampling. Default is 5.
    interpolation : str, optional
        Interpolation method to use. Options are 'nearest' (order 0) or 'bicubic' (order 3).
        Default is 'nearest'.
    pixel_sizes : dict, optional
        Dictionary specifying physical pixel dimensions, e.g., {'Z': 0.79, 'Y': 0.468, 'X': 0.468}.
        If provided, normals and offsets are computed in physical space. Assumes Z, Y, X order.

    Returns:
    --------
    sampled_3d : ndarray
        A 3D array of shape (U, V, num_steps) containing the sampled intensities.
    """
    if pixel_sizes is None:
        warnings.warn("pixel_sizes not provided; defaulting to isotropic pixel size of 1.0.", UserWarning, stacklevel=2)

    # Determine spacing, default to isotropic 1.0 if not provided
    if pixel_sizes is None:
        spacing = np.array([1.0, 1.0, 1.0])
    else:
        spacing = np.array([pixel_sizes.get('Z', 1.0), pixel_sizes.get('Y', 1.0), pixel_sizes.get('X', 1.0)])

    # 1. Convert grid to physical space for accurate normal computation
    physical_grid = surface_points_grid * spacing

    # 2. Estimate Normals numerically using gradients over the physical grid structure
    t_u = np.gradient(physical_grid, axis=0)
    t_v = np.gradient(physical_grid, axis=1)

    physical_normals = np.cross(t_u, t_v)
    norms = np.linalg.norm(physical_normals, axis=2, keepdims=True)
    physical_normals = np.divide(physical_normals, norms, out=np.zeros_like(physical_normals), where=norms!=0)

    # 3. Reshape for sampling
    physical_pts = physical_grid.reshape(-1, 3)
    normals_flat = physical_normals.reshape(-1, 3)

    # 4. Create sampling coordinates along the normal in physical space
    offsets = np.linspace(-thickness/2, thickness/2, num_steps)
    physical_sampling_coords = physical_pts[:, np.newaxis, :] + (normals_flat[:, np.newaxis, :] * offsets[np.newaxis, :, np.newaxis])

    # 5. Convert back to pixel space for interpolation
    pixel_sampling_coords = physical_sampling_coords / spacing
    coords_reshaped = pixel_sampling_coords.reshape(-1, 3).T

    # 6. Perform Interpolation
    order = 3 if interpolation == 'bicubic' else 0
    sampled_flat = map_coordinates(image, coords_reshaped, order=order, mode='constant', cval=0)

    # 7. Reshape back to (U, V, steps)
    return sampled_flat.reshape(surface_points_grid.shape[0], surface_points_grid.shape[1], num_steps)
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from itertools import product
from scipy.ndimage import map_coordinates

def _ensure_pixel_size_array(pixel_sizes: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None) -> np.ndarray:
    """Helper to convert dict or list to [Z, Y, X] numpy array."""
    if pixel_sizes is None:
        warnings.warn("pixel_sizes not provided; defaulting to isotropic pixel size of 1.0.", UserWarning, stacklevel=2)
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)
    if isinstance(pixel_sizes, dict):
        return np.array([pixel_sizes.get('Z', 1.0), pixel_sizes.get('Y', 1.0), pixel_sizes.get('X', 1.0)], dtype=np.float64)
    return np.array(pixel_sizes, dtype=np.float64)

def fit_plane_ransac(
    points_zyx: np.ndarray,
    pixel_sizes: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
    inlier_threshold_um: float = 1.0,
    max_iters: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts pixel points to physical space, then fits a plane using RANSAC.

    Parameters:
        points_zyx: (N, 3) array of point coordinates in voxel indices.
        pixel_sizes: Optional dict mapping dimension to pixel size, e.g., {'Z': 1.0, 'Y': 0.5, 'X': 0.5}.
        inlier_threshold_um: Distance threshold for inliers in physical units (microns).
    """
    pixel_size = _ensure_pixel_size_array(pixel_sizes)
    points_phys = points_zyx * pixel_size

    num_points = points_phys.shape[0]
    if num_points < 3:
        raise ValueError("At least 3 points are required.")

    best_inliers = []
    best_normal_phys = None
    best_p0_phys = None
    rng = np.random.default_rng()

    for _ in range(max_iters):
        idx = rng.choice(num_points, 3, replace=False)
        p1, p2, p3 = points_phys[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)

        area = 0.5 * np.linalg.norm(normal)
        if area < 1e-6:
            continue

        normal /= np.linalg.norm(normal)
        distances = np.abs(np.dot(points_phys - p1, normal))
        inliers = np.where(distances < inlier_threshold_um)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal_phys = normal
            best_p0_phys = p1

    if len(best_inliers) < 3:
        raise RuntimeError("RANSAC failed to find a valid plane.")

    inlier_points = points_phys[best_inliers]
    best_p0_phys = np.mean(inlier_points, axis=0)
    _, _, vv = np.linalg.svd(inlier_points - best_p0_phys)
    best_normal_phys = vv[2, :]

    if best_normal_phys[0] < 0:
        best_normal_phys = -best_normal_phys

    return best_p0_phys, best_normal_phys

def generate_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generates an orthonormal basis (u, v) perpendicular to the normal vector."""
    min_axis = np.argmin(np.abs(normal))
    ortho_choice = np.zeros(3)
    ortho_choice[min_axis] = 1.0
    u = np.cross(normal, ortho_choice)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v

def sample_volume_plane(
    volume: np.ndarray,
    pixel_sizes: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
    p0_phys: Optional[np.ndarray] = None,
    normal_phys: Optional[np.ndarray] = None,
    u_range_um: Optional[Tuple[float, float]] = None,
    v_range_um: Optional[Tuple[float, float]] = None,
    u_res: Optional[int] = None,
    v_res: Optional[int] = None,
    method: str = 'linear',
    num_offsets: int = 0,
    offset_step_um: float = 1.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Samples an anisotropic 3D (Z, Y, X) volume.
    """
    if p0_phys is None or normal_phys is None:
        raise ValueError("p0_phys and normal_phys must be provided.")

    pixel_size = _ensure_pixel_size_array(pixel_sizes)
    z_dim, y_dim, x_dim = volume.shape
    sz, sy, sx = pixel_size

    u_dir, v_dir = generate_plane_basis(normal_phys)

    # 1. Dynamic Extent & Resolution Calculation
    if None in (u_range_um, v_range_um, u_res, v_res):
        z_phys_max = (z_dim - 1) * sz
        y_phys_max = (y_dim - 1) * sy
        x_phys_max = (x_dim - 1) * sx
        corners = np.array(list(product([0, z_phys_max], [0, y_phys_max], [0, x_phys_max])))
        diffs = corners - p0_phys
        u_projections = np.dot(diffs, u_dir)
        v_projections = np.dot(diffs, v_dir)

        if u_range_um is None: u_range_um = (np.min(u_projections), np.max(u_projections))
        if v_range_um is None: v_range_um = (np.min(v_projections), np.max(v_projections))
        optimal_res = min(sz, sy, sx)
        if u_res is None: u_res = int(np.ceil((u_range_um[1] - u_range_um[0]) / optimal_res)) + 1
        if v_res is None: v_res = int(np.ceil((v_range_um[1] - v_range_um[0]) / optimal_res)) + 1

    # 2. Pre-allocate the final output array using the original memory-efficient dtype
    total_slices = 2 * num_offsets + 1
    sampled_volume = np.zeros((total_slices, u_res, v_res), dtype=volume.dtype)

    # 3. Generate 2D coordinate matrices in float32 to halve query RAM
    u_vals = np.linspace(u_range_um[0], u_range_um[1], u_res, dtype=np.float32)
    v_vals = np.linspace(v_range_um[0], v_range_um[1], v_res, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u_vals, v_vals, indexing='ij')

    offsets = np.arange(-num_offsets, num_offsets + 1, dtype=np.float32) * offset_step_um

    # map_coordinates requires an integer order (1 for linear, 3 for cubic)
    spline_order = 3 if method == 'cubic' else 1

    # 4. Extract slices directly to the pre-allocated array
    for i, d in enumerate(offsets):
        # Calculate physical coordinates
        phys_z = p0_phys[0] + u_grid * u_dir[0] + v_grid * v_dir[0] + d * normal_phys[0]
        phys_y = p0_phys[1] + u_grid * u_dir[1] + v_grid * v_dir[1] + d * normal_phys[1]
        phys_x = p0_phys[2] + u_grid * u_dir[2] + v_grid * v_dir[2] + d * normal_phys[2]

        # Convert physical metrics to fractional voxel indices
        # map_coordinates operates on dimensions: (3, N)
        coords = np.stack([
            (phys_z / sz).ravel(),
            (phys_y / sy).ravel(),
            (phys_x / sx).ravel()
        ], axis=0)

        # Sample directly into the assigned matrix slice
        # The 'output' kwarg forces memory mapping straight to the integer dtype
        sampled_volume[i] = map_coordinates(
            volume,
            coords,
            order=spline_order,
            mode='constant',
            cval=0,
            output=volume.dtype,
            prefilter=False # Avoids calculating spline coefficients across the entire input volume for order=3
        ).reshape(u_res, v_res)

        # Force clear intermediates for massive planes
        del phys_z, phys_y, phys_x, coords

    if num_offsets == 0:
        sampled_volume = sampled_volume[0]

    # 5. Calculate new pixel spacings
    spacing_u = (u_range_um[1] - u_range_um[0]) / max(1, u_res - 1)
    spacing_v = (v_range_um[1] - v_range_um[0]) / max(1, v_res - 1)
    spacing_n = offset_step_um if num_offsets > 0 else 0.0

    output_pixel_sizes = {'X': spacing_u, 'Y': spacing_v, 'Z': spacing_n}
    return sampled_volume, output_pixel_sizes
