import inspect
import warnings
from itertools import product
from typing import Optional, Tuple, Dict, List, Union

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import uniform_filter, map_coordinates
from scipy import ndimage
from scipy.spatial import KDTree
from collections.abc import Iterable
from scipy.interpolate import RegularGridInterpolator
from skimage import filters, feature, segmentation
from skimage.segmentation import expand_labels


def _pixel_sizes_to_spacing(pixel_sizes, ndim: int, stacklevel: int = 2) -> tuple[float, ...]:
    """
    Per-axis spacing (the last `ndim` of Z, Y, X) from a dict or a sequence.

    Readers return None for axes without calibration (e.g. {'Z': None} for an
    ImageJ stack without spacing); those fall back to 1.0 with a UserWarning
    naming the axis. Non-finite or non-positive sizes raise ValueError.
    `stacklevel` is the one the calling function would use for its own warnings.
    """
    keys = ('Z', 'Y', 'X')[-ndim:]
    if isinstance(pixel_sizes, dict):
        values = [pixel_sizes.get(k, 1.0) for k in keys]
    else:
        values = list(pixel_sizes)
        if len(values) != ndim:
            raise ValueError(f"pixel_sizes has {len(values)} entries, expected {ndim} ({', '.join(keys)}).")

    missing = [k for k, v in zip(keys, values) if v is None]
    if missing:
        warnings.warn(
            f"pixel_sizes has no value for axis {', '.join(missing)}; using 1.0 for it.",
            UserWarning, stacklevel=stacklevel + 1
        )
    spacing = tuple(1.0 if v is None else float(v) for v in values)
    if not all(np.isfinite(s) and s > 0 for s in spacing):
        raise ValueError(f"pixel_sizes must be finite and positive, got {dict(zip(keys, spacing))}.")
    return spacing


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
    - pixel_sizes: Optional dict mapping dimension to pixel size, e.g., {'Z': 1.0, 'Y': 0.5, 'X': 0.5},
            or a per-axis sequence in (Z)YX order.

    Seeds are local maxima of the spot-blurred image at least 3 pixels of the finest
    axis apart (Chebyshev distance in physical units), so coarser axes such as Z use
    proportionally fewer pixels. Seeds on the image border are kept.

    Returns:
    - labels: Segmented labeled mask.
    """
    if spacing is not None:
        warnings.warn("The 'spacing' parameter is deprecated. Use 'pixel_sizes' instead.", DeprecationWarning, stacklevel=2)
        if pixel_sizes is None:
            pixel_sizes = spacing

    if pixel_sizes is not None:
        sp = _pixel_sizes_to_spacing(pixel_sizes, image.ndim)
    else:
        sp = (1.0,) * image.ndim

    # Convert sigmas into tuples per axis, scaling by pixel size if provided
    def _process_sigma(sigma_val):
        if not isinstance(sigma_val, (list, tuple)):
            sigma_val = [sigma_val] * image.ndim

        if pixel_sizes is not None:
            sigma_val = [s / s_ax for s, s_ax in zip(sigma_val, sp)]

        return tuple(sigma_val)

    spot_sigma_tuple = _process_sigma(spot_sigma)
    outline_sigma_tuple = _process_sigma(outline_sigma)

    # Step 1: Apply Gaussian blur for spot detection directly into a new variable
    blurred_spot = filters.gaussian(image, sigma=spot_sigma_tuple)

    # Step 2: Detect local maxima (spots)
    # min_distance is 3 px of the finest axis, in physical units. With isotropic pixels
    # this is exactly peak_local_max(min_distance=3) (7^n box, Chebyshev spacing >= 3).
    # exclude_border=False: the default (= min_distance) drops every seed within 3 px of
    # a face, so nuclei touching the border, and all nuclei of a thin stack, got no label.
    min_dist_um = 3 * min(sp)
    footprint = np.ones(tuple(2 * max(1, int(round(min_dist_um / s))) + 1 for s in sp), dtype=bool)
    local_maxi = feature.peak_local_max(blurred_spot, footprint=footprint, exclude_border=False)
    if len(local_maxi) > 1:
        # Greedy suppression in descending intensity (peak_local_max returns that order)
        tree = KDTree(local_maxi * np.asarray(sp))
        neighbours = tree.query_ball_point(local_maxi * np.asarray(sp), r=min_dist_um, p=np.inf)
        keep = np.ones(len(local_maxi), dtype=bool)
        for i, nbrs in enumerate(neighbours):
            if not keep[i]:
                continue
            for j in nbrs:
                if j != i and np.max(np.abs((local_maxi[j] - local_maxi[i]) * sp)) < min_dist_um:
                    keep[j] = False
        local_maxi = local_maxi[keep]

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
            'average' averages the slices of each window that lie inside the volume, so
            the first and last slices are not dimmed by the padding.
        pixel_sizes (dict, optional): Dictionary specifying physical pixel dimensions, e.g., {'Z': 0.79, 'Y': 0.468, 'X': 0.468}.

    Returns:
        np.ndarray: An image array with the thick z-slice operation applied.
    """
    if operation not in ('max', 'average'):
        raise ValueError(f"operation must be 'max' or 'average', got {operation!r}.")

    if pixel_sizes is None:
        warnings.warn("pixel_sizes not provided; defaulting to isotropic pixel size of 1.0.", UserWarning, stacklevel=2)

    if pixel_sizes is not None:
        pixel_size_axis = _pixel_sizes_to_spacing(pixel_sizes, nuclei_image.ndim)[axis]
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

    # Pad the image along the specified axis. For 'max', pad with the image minimum so
    # padding never wins over real (possibly negative) values at the edges.
    pad_width = [(0, 0)] * nuclei_image.ndim
    pad_width[axis] = (margin, margin)
    pad_value = np.min(nuclei_image) if operation == 'max' else 0
    padded_image = np.pad(nuclei_image, pad_width=pad_width, mode='constant', constant_values=pad_value)

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

    # Finalize the operation: divide by the number of real slices in each window, so
    # windows that overlap the zero padding at the ends are not dimmed
    if operation == 'average':
        n = nuclei_image.shape[axis]
        k = np.arange(n)
        counts = np.minimum(k + margin, n - 1) - np.maximum(k - margin, 0) + 1
        shape = [1] * nuclei_image.ndim
        shape[axis] = n
        nuclei_image_thick /= counts.reshape(shape)

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
        spacing = _pixel_sizes_to_spacing(pixel_sizes, labels_data.ndim)
        if 'spacing' not in inspect.signature(expand_labels).parameters:
            raise NotImplementedError(
                "Anisotropic expansion requires scikit-image >= 0.21.0 for the `spacing` parameter. "
                "Uniform pixel fallback causes severe structural distortion."
            )
        dilated_labels_data = expand_labels(labels_data, distance=distance, spacing=spacing)
    else:
        dilated_labels_data = expand_labels(labels_data, distance=distance)

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

    The window along each axis is the odd number of pixels nearest to `diameter`
    (diameter 3 px -> 3, 4 px -> 5, 5 px -> 5). Means are computed in float64.

    Returns:
        list: A list of sampled mean intensities for each point. Out-of-bounds points will have NaN.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    if points_3d.ndim == 1 and points_3d.size in (0, image_3d.ndim):
        # [] or a single flat point [z, y, x]
        points_3d = points_3d.reshape(-1, image_3d.ndim)
    if points_3d.ndim != 2 or points_3d.shape[1] != image_3d.ndim:
        raise ValueError(f"points_3d must have shape (N, {image_3d.ndim}), got {points_3d.shape}.")

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
        spacing = _pixel_sizes_to_spacing(pixel_sizes, image_3d.ndim)
    else:
        spacing = [1.0] * image_3d.ndim

    filter_size = []
    for s in spacing:
        # Diameter in pixels, rounded half up (round() is banker's rounding: 2.5 -> 2)
        n = max(1, int(np.floor(diameter / s + 0.5)))
        # Window size must be odd to center perfectly on the coordinate point
        filter_size.append(n if n % 2 == 1 else n + 1)

    filter_size = tuple(filter_size)

    # Use 'reflect' or 'nearest' to ensure boundary points are not diluted by zero-padding.
    # Filter in float64: uniform_filter outputs the input dtype, so integer images
    # had their means truncated (Poisson(5) uint16 -> ~3.9).
    filtered_image = uniform_filter(image_3d, size=filter_size, mode='reflect', output=np.float64)

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


def sample_intensity_along_surface_normals(
    image: np.ndarray,
    surface_points_grid: np.ndarray,
    thickness: float = 3.0,
    num_steps: int = 5,
    interpolation: str = 'nearest',
    pixel_sizes: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None
) -> np.ndarray:
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
    pixel_sizes : dict, list or ndarray, optional
        Dictionary specifying physical pixel dimensions, e.g., {'Z': 0.79, 'Y': 0.468, 'X': 0.468}.
        If provided, normals and offsets are computed in physical space. Assumes Z, Y, X order.

    Returns:
    --------
    sampled_3d : ndarray
        A 3D array of shape (U, V, num_steps) containing the sampled intensities.
    """
    spacing = _ensure_pixel_size_array(pixel_sizes)

    # 1. Convert grid to physical space for accurate normal computation
    physical_grid = surface_points_grid * spacing

    # 2. Estimate Normals numerically using gradients over the physical grid structure
    t_u = np.gradient(physical_grid, axis=0)
    t_v = np.gradient(physical_grid, axis=1)

    physical_normals = np.cross(t_u, t_v)
    norms = np.linalg.norm(physical_normals, axis=2, keepdims=True)
    physical_normals = np.divide(physical_normals, norms, out=np.zeros_like(physical_normals), where=norms != 0)

    # 3. Reshape for sampling
    physical_pts = physical_grid.reshape(-1, 3)
    normals_flat = physical_normals.reshape(-1, 3)

    # 4. Create sampling coordinates along the normal in physical space
    if num_steps == 1:
        offsets = np.zeros(1)  # linspace would put the only sample at -thickness/2
    else:
        offsets = np.linspace(-thickness / 2.0, thickness / 2.0, num_steps)
    physical_sampling_coords = physical_pts[:, np.newaxis, :] + (normals_flat[:, np.newaxis, :] * offsets[np.newaxis, :, np.newaxis])

    # 5. Convert back to pixel space for interpolation
    pixel_sampling_coords = physical_sampling_coords / spacing
    coords_reshaped = pixel_sampling_coords.reshape(-1, 3).T

    # 6. Perform Interpolation
    order = 3 if interpolation == 'bicubic' else 0
    sampled_flat = map_coordinates(image, coords_reshaped, order=order, mode='constant', cval=0)

    # 7. Reshape back to (U, V, steps)
    return sampled_flat.reshape(surface_points_grid.shape[0], surface_points_grid.shape[1], num_steps)


from scipy.interpolate import RegularGridInterpolator
from itertools import product
from scipy.ndimage import map_coordinates

def _resolve_spacing(
    spacing: Optional[Union[Iterable[float], Dict[str, float]]] = None,
    pixel_sizes: Optional[Union[Dict[str, float], Iterable[float]]] = None,
    ndim: int = 3
) -> tuple[float, ...]:
    """
    Resolves physical pixel spacing from either `pixel_sizes` or legacy `spacing` argument.
    Emits UserWarning if neither is provided. Always called from a public function,
    so warnings point at that function's caller.
    """
    if pixel_sizes is not None:
        return _pixel_sizes_to_spacing(pixel_sizes, ndim, stacklevel=3)

    if spacing is not None:
        return _pixel_sizes_to_spacing(spacing, ndim, stacklevel=3)

    warnings.warn("pixel_sizes not provided; defaulting to isotropic pixel size of 1.0.", UserWarning, stacklevel=3)
    return (1.0,) * ndim

def _ensure_pixel_size_array(pixel_sizes: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None) -> np.ndarray:
    """Helper to convert dict or list to [Z, Y, X] numpy array (called from public functions)."""
    if pixel_sizes is None:
        warnings.warn("pixel_sizes not provided; defaulting to isotropic pixel size of 1.0.", UserWarning, stacklevel=3)
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)
    return np.array(_pixel_sizes_to_spacing(pixel_sizes, 3, stacklevel=3), dtype=np.float64)

def create_ellipsoid_struct(
    radius_um: float,
    spacing: Optional[Union[Iterable[float], Dict[str, float]]] = None,
    pixel_sizes: Optional[Union[Dict[str, float], Iterable[float]]] = None
) -> np.ndarray:
    """Creates a boolean 3D ellipsoidal structuring element centered at origin."""
    sp = _resolve_spacing(spacing, pixel_sizes, ndim=3)
    radius_um = max(float(radius_um), 1e-6)

    radii_vox = [radius_um / s for s in sp]
    grids = [np.arange(-np.ceil(r), np.ceil(r) + 1) * s for r, s in zip(radii_vox, sp)]
    mesh = np.meshgrid(*grids, indexing='ij')

    dist_sq = sum(g**2 for g in mesh)
    struct = dist_sq <= (radius_um ** 2)
    # Guarantee at least the single center voxel is active
    if not np.any(struct):
        struct = np.ones((1, 1, 1), dtype=bool)
    return struct


def generate_morphological_surface_mask(
    binary_mask: np.ndarray,
    spacing: Optional[Union[Iterable[float], Dict[str, float]]] = None,
    closing_radius_um: float = 10.0,
    surface_depth_um: float = 2.0,
    downscale_factor: int = 2,
    fill_holes: bool = True,
    pixel_sizes: Optional[Union[Dict[str, float], Iterable[float]]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes a continuous morphological envelope and extracts the physical surface layer.
    """
    sp = _resolve_spacing(spacing, pixel_sizes, ndim=binary_mask.ndim)

    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        return np.zeros_like(binary_mask, dtype=bool), np.zeros_like(binary_mask, dtype=bool)

    # 1. Crop to global foreground bounding box
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    slices = tuple(slice(a, b) for a, b in zip(lo, hi))
    cropped_binary = binary_mask[slices]

    # 2. Downscale by block max-pooling (a block is foreground if any voxel is), so the
    # envelope computed at low resolution still covers every foreground voxel.
    # Nearest-neighbour decimation dropped small objects (17% of voxels at f=2).
    f = int(downscale_factor)
    if f > 1:
        pad = [(0, (-s) % f) for s in cropped_binary.shape]
        padded = np.pad(cropped_binary, pad, mode='constant', constant_values=False)
        blocks_shape = []
        for s in padded.shape:
            blocks_shape += [s // f, f]
        downscaled_binary = padded.reshape(blocks_shape).any(axis=tuple(range(1, 2 * padded.ndim, 2)))
        effective_spacing = tuple(s * downscale_factor for s in sp)
    else:
        downscaled_binary = cropped_binary
        effective_spacing = sp

    # 3. Anisotropic morphological closing with boundary padding to prevent edge erosion
    struct = create_ellipsoid_struct(closing_radius_um, spacing=effective_spacing)
    pad_width = tuple((s, s) for s in struct.shape)
    padded_binary = np.pad(downscaled_binary, pad_width, mode='constant', constant_values=False)

    downscaled_envelope = ndimage.binary_closing(padded_binary, structure=struct)

    if fill_holes:
        downscaled_envelope = ndimage.binary_fill_holes(downscaled_envelope)

    # 4. Extract surface zone using physical distance transform
    dist_from_outside = ndimage.distance_transform_edt(
        downscaled_envelope,
        sampling=effective_spacing
    )
    downscaled_surface = (dist_from_outside > 0) & (dist_from_outside <= surface_depth_um)

    # Safety: if depth is smaller than 1 voxel step, take immediate 1-voxel outer shell
    if not np.any(downscaled_surface) and np.any(downscaled_envelope):
        eroded = ndimage.binary_erosion(downscaled_envelope)
        downscaled_surface = downscaled_envelope & ~eroded

    # Unpad back to downscaled dimensions
    unpad_slices = tuple(
        slice(p[0], p[0] + orig_s)
        for p, orig_s in zip(pad_width, downscaled_binary.shape)
    )
    downscaled_envelope = downscaled_envelope[unpad_slices]
    downscaled_surface = downscaled_surface[unpad_slices]

    # 5. Upscale back to cropped dimensions: block i covers voxels [i*f, (i+1)*f)
    if f > 1:
        crop = tuple(slice(0, s) for s in cropped_binary.shape)
        cropped_envelope = downscaled_envelope
        cropped_surface = downscaled_surface
        for ax in range(cropped_binary.ndim):
            cropped_envelope = np.repeat(cropped_envelope, f, axis=ax)
            cropped_surface = np.repeat(cropped_surface, f, axis=ax)
        # The envelope is a closing, so it must contain the input
        cropped_envelope = cropped_envelope[crop] | cropped_binary
        cropped_surface = cropped_surface[crop]
    else:
        cropped_envelope = downscaled_envelope
        cropped_surface = downscaled_surface

    # 6. Map back to original coordinate volume
    full_envelope = np.zeros_like(binary_mask, dtype=bool)
    surface_mask = np.zeros_like(binary_mask, dtype=bool)

    full_envelope[slices] = cropped_envelope
    surface_mask[slices] = cropped_surface

    return surface_mask, full_envelope


def estimate_inter_label_distance(
    labels: np.ndarray,
    spacing: Optional[Union[Iterable[float], Dict[str, float]]] = None,
    sample_size: int = 500,
    k_nearest: int = 2,
    pixel_sizes: Optional[Union[Dict[str, float], Iterable[float]]] = None,
    seed: Optional[int] = None
) -> dict[str, float]:
    """
    Rapidly estimates characteristic inter-nucleus spacing in physical units (µm).

    Distances are from up to `sample_size` randomly chosen labels (`seed` makes the
    choice reproducible) to their nearest neighbours among *all* labels.
    """
    sp = _resolve_spacing(spacing, pixel_sizes, ndim=labels.ndim)

    unique_ids = np.unique(labels)
    unique_ids = unique_ids[unique_ids != 0]

    min_spacing = min(sp)
    if len(unique_ids) < k_nearest:
        return {
            "median_um": float(min_spacing),
            "p75_um": float(min_spacing),
            "p90_um": float(min_spacing),
            "recommended_closing_um": float(np.round(1.5 * min_spacing, 1))
        }

    # 1. Centroids of all labels: one O(voxels) pass, whatever the number of labels.
    # The neighbours must come from all labels; within a random subsample the nearest
    # neighbour is farther away (2.2x the true spacing for 27,000 labels, 500 sampled).
    centroids_vox = ndimage.center_of_mass(
        labels,
        labels=labels,
        index=unique_ids
    )
    centroids_um = np.asarray(centroids_vox, dtype=np.float64).reshape(len(unique_ids), -1) * np.asarray(sp, dtype=np.float64)

    # 2. Subsample the query points only
    if len(unique_ids) > sample_size:
        rng = np.random.default_rng(seed)
        query_um = centroids_um[rng.choice(len(unique_ids), size=sample_size, replace=False)]
    else:
        query_um = centroids_um

    # 3. KDTree nearest-neighbor query (k=1 is the point itself)
    tree = KDTree(centroids_um)
    distances, _ = tree.query(query_um, k=k_nearest)
    nn_dists_um = distances[:, k_nearest - 1]

    median_d = float(np.percentile(nn_dists_um, 50))
    p75_d = float(np.percentile(nn_dists_um, 75))
    p90_d = float(np.percentile(nn_dists_um, 90))

    # Ensure closing radius is at least 1.0x the smallest voxel spacing
    recommended_closing_um = float(np.round(max(0.6 * p75_d, min_spacing), 1))

    return {
        "median_um": median_d,
        "p75_um": p75_d,
        "p90_um": p90_d,
        "recommended_closing_um": recommended_closing_um
    }

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
    normal = np.asarray(normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("normal must be non-zero.")
    normal = normal / norm
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
    Samples an anisotropic 3D (Z, Y, X) volume on the plane through `p0_phys` with
    normal `normal_phys` (any length; it is normalised).

    Returns the sampled plane(s), shape (u_res, v_res) or (2*num_offsets+1, u_res, v_res),
    and their pixel sizes under the ZYX convention: {'Z': offset step, 'Y': u step,
    'X': v step}. method='cubic' interpolates (spline prefiltered); 'linear' is trilinear.
    """
    if p0_phys is None or normal_phys is None:
        raise ValueError("p0_phys and normal_phys must be provided.")

    normal_phys = np.asarray(normal_phys, dtype=np.float64)
    normal_norm = np.linalg.norm(normal_phys)
    if normal_norm == 0:
        raise ValueError("normal_phys must be non-zero.")
    normal_phys = normal_phys / normal_norm

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

    origin = np.zeros(3)
    if spline_order == 3:
        # Cubic B-spline *interpolation* needs prefiltered coefficients; without them
        # map_coordinates smooths (std of a random volume 0.27 -> 0.10). Compute them
        # once for all offsets, the same way map_coordinates(prefilter=True) would, but
        # only on the sub-volume the planes touch. The prefilter's influence decays as
        # 0.268^k voxels, so a 12-voxel margin makes the crop exact to ~1e-7.
        corners = np.array([
            p0_phys + u * u_dir + v * v_dir + d * normal_phys
            for u in u_range_um for v in v_range_um for d in (offsets[0], offsets[-1])
        ]) / pixel_size
        lo = np.clip(np.floor(corners.min(axis=0)).astype(int) - 12, 0, volume.shape)
        hi = np.clip(np.ceil(corners.max(axis=0)).astype(int) + 13, 0, volume.shape)
        if np.any(hi <= lo):
            # The planes miss the volume entirely: everything samples to cval
            lo, hi = np.zeros(3, dtype=int), np.minimum(np.ones(3, dtype=int), volume.shape)
        origin = lo.astype(np.float64)
        source = ndimage.spline_filter(
            volume[tuple(slice(a, b) for a, b in zip(lo, hi))],
            order=3, output=np.float32, mode='constant'
        )
        is_int = np.issubdtype(volume.dtype, np.integer)
    else:
        source = volume

    # 4. Extract slices directly to the pre-allocated array
    for i, d in enumerate(offsets):
        # Calculate physical coordinates
        phys_z = p0_phys[0] + u_grid * u_dir[0] + v_grid * v_dir[0] + d * normal_phys[0]
        phys_y = p0_phys[1] + u_grid * u_dir[1] + v_grid * v_dir[1] + d * normal_phys[1]
        phys_x = p0_phys[2] + u_grid * u_dir[2] + v_grid * v_dir[2] + d * normal_phys[2]

        # Convert physical metrics to fractional voxel indices
        # map_coordinates operates on dimensions: (3, N)
        coords = np.stack([
            (phys_z / sz).ravel() - origin[0],
            (phys_y / sy).ravel() - origin[1],
            (phys_x / sx).ravel() - origin[2]
        ], axis=0)

        if spline_order == 3:
            plane = map_coordinates(
                source, coords, order=3, mode='constant', cval=0,
                output=np.float32, prefilter=False
            ).reshape(u_res, v_res)
            if is_int:
                info = np.iinfo(volume.dtype)
                plane = np.clip(np.rint(plane), info.min, info.max)
            sampled_volume[i] = plane
        else:
            # Sample directly into the assigned matrix slice
            # The 'output' kwarg forces memory mapping straight to the integer dtype
            sampled_volume[i] = map_coordinates(
                source,
                coords,
                order=1,
                mode='constant',
                cval=0,
                output=volume.dtype,
                prefilter=False
            ).reshape(u_res, v_res)

        # Force clear intermediates for massive planes
        del phys_z, phys_y, phys_x, coords

    if num_offsets == 0:
        sampled_volume = sampled_volume[0]

    # 5. Calculate new pixel spacings
    spacing_u = (u_range_um[1] - u_range_um[0]) / max(1, u_res - 1)
    spacing_v = (v_range_um[1] - v_range_um[0]) / max(1, v_res - 1)
    spacing_n = offset_step_um if num_offsets > 0 else 0.0

    # Axis 0 of each plane is u, axis 1 is v; under the ZYX convention that is Y, X
    output_pixel_sizes = {'Z': spacing_n, 'Y': spacing_u, 'X': spacing_v}
    return sampled_volume, output_pixel_sizes
