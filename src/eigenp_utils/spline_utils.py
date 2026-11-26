# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy",
#     "matplotlib",
# ]
# ///
import warnings

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ._conventions import resolve_pixel_sizes, warn_deprecated

# Function to generate random 3D coordinates
def generate_random_3d_coordinates(num_points=10, seed=None):
    if seed is not None:
        # A private generator: same values as np.random.seed(seed) gave, without
        # reseeding numpy's global random state for the caller
        return np.random.RandomState(seed).rand(num_points, 3) * 100
    return np.random.rand(num_points, 3) * 100  # Scaling to get a wider range


def _mark_points(image, points):
    """Set the voxels containing `points` (one row per point) to 1.

    Voxel centres sit at integer coordinates (as in SciPy, skimage and the
    package's sampling functions): voxel i covers [i - 0.5, i + 0.5). Coordinates
    are rounded half up, ``floor(c + 0.5)``, so a tie x.5 goes to the higher voxel
    (2.5 -> 3, not banker's rounding). Points outside the image, i.e. outside
    [-0.5, n - 0.5) on any axis, are skipped; negative indices never wrap around.

    Changed (D23, R7-02, 2026-11): coordinates are rounded, not floored. Flooring
    put voxel corners at integers, so a spline through integer knots (splev returns
    e.g. 4.999999999999996) was drawn doubled and one voxel off.
    """
    idx = np.floor(np.asarray(points, dtype=np.float64) + 0.5).astype(int)
    shape = np.asarray(image.shape)
    inside = np.all((idx >= 0) & (idx < shape), axis=1)
    image[tuple(idx[inside].T)] = 1


# Function to fit a cubic spline through 3D coordinates
def fit_cubic_spline(coordinates, return_u = False):
    """
    Interpolating B-spline through the last three columns of `coordinates` (N, >= 3).

    The degree is min(3, N - 1): cubic for 4 or more points, quadratic for 3, linear
    for 2. Returns ``tck`` (and the parameter values ``u`` of the points when
    `return_u`).

    Changed (D23, R7-16, 2026-11): 2 and 3 points work (the degree is lowered, as in
    `create_resampled_spline`); splprep raised for them.
    """
    coordinates = np.asarray(coordinates)
    k = min(3, len(coordinates) - 1)
    # Fit a cubic spline (B-spline) to the 3D coordinates
    tck, u = splprep([coordinates[:,-3], coordinates[:,-2], coordinates[:,-1]], s=0, k=k)
    if return_u:
        return tck, u
    else:
        return tck


def create_3d_image_from_spline(spline_tck, shape=None, num_points=2500):
    """
    Create a 3D image from a spline.

    Parameters:
    - spline_tck: The parameters of the fitted spline.
    - shape: The shape of the 3D image (voxel grid). If None, it's set to 1.2 times the max extent of the spline.
    - num_points: The number of points to sample along the spline for high resolution.

    Returns:
    - A 3D numpy array representing the image.

    Changed (D23, R7-02, 2026-11): sample points are rounded to the nearest voxel
    (half up; voxel centres at integer coordinates), not floored, so drawn voxels
    can move by up to one voxel; see `_mark_points` for the tie rule.
    """
    # Sample points along the spline
    u_fine = np.linspace(0, 1, num_points)
    x, y, z = splev(u_fine, spline_tck)

    # Determine the shape of the 3D image if not provided
    if shape is None:
        max_extent = np.max([np.max(coord) for coord in [x, y, z]])
        shape = np.ceil(max_extent * 1.2).astype(int)
        shape = [shape] * 3  # Ensure the shape is 3D

    # Initialize the 3D image
    image = np.zeros(shape, dtype=np.uint8)

    # Mark the path of the spline in the 3D image
    _mark_points(image, np.column_stack([x, y, z]))

    return image


def create_nd_image_from_spline(spline_tck, shape=None, num_points=2500):
    """
    Create an n-dimensional image from a spline.

    Parameters:
    - spline_tck: The parameters of the fitted spline.
    - shape: The shape of the n-dimensional image (hypergrid). If None, it's set to 1.2 times the max extent of the spline.
    - num_points: The number of points to sample along the spline for high resolution.

    Returns:
    - An n-dimensional numpy array representing the image.

    Changed (D23, R7-02, 2026-11): sample points are rounded to the nearest voxel
    (half up; voxel centres at integer coordinates), not floored, so drawn voxels
    can move by up to one voxel; see `_mark_points` for the tie rule.
    """
    # Sample points along the spline
    u_fine = np.linspace(0, 1, num_points)
    points = splev(u_fine, spline_tck)

    # Determine the number of dimensions from the spline output
    num_dims = len(points)

    # Determine the shape of the image if not provided
    if shape is None:
        max_extent = [np.max(coord) for coord in points]
        shape = np.ceil(np.array(max_extent) * 1.2).astype(int)

    # Initialize the n-dimensional image
    image = np.zeros(shape, dtype=np.uint8)

    # Mark the path of the spline in the image
    _mark_points(image, np.column_stack(points))

    return image


# Function to plot the initial points and the spline
def plot_3d_spline(coordinates, spline_tck, num_points=1000, ax=None, title=None):
    """
    Plot the initial points and the cubic spline in 3D.

    Parameters:
    - coordinates: The initial 3D coordinates, columns in (Z, Y, X) order
      (the last three columns are used, as in `fit_cubic_spline`).
    - spline_tck: The parameters of the fitted spline.
    - num_points: Number of points to sample along the spline.
    - ax: Optional existing 3D axes to draw on.
    - title: Optional axes title (default: none).

    Returns:
    - The matplotlib Figure.

    Changed (D23, R7-16, 2026-11): new `title` parameter; there is no default title
    (it was the hard-coded '3D Cubic Spline through Random Points').
    """
    # Sample points along the spline
    u_fine = np.linspace(0, 1, num_points)
    x, y, z = splev(u_fine, spline_tck)

    # Create the 3D plot
    if ax is None:
        fig = plt.figure(figsize=(4, 4), layout = 'constrained')
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # Plot the initial points
    ax.scatter(coordinates[:, -3], coordinates[:, -2], coordinates[:, -1], color='red', label='Initial Points')

    # Plot the spline
    ax.plot(x, y, z, color='blue', label='Cubic Spline')

    # Columns are plotted in (Z, Y, X) order on the plot's x, y, z axes
    ax.set_xlabel('Z')
    ax.set_ylabel('Y')
    ax.set_zlabel('X')
    ax.legend()

    if title is not None:
        ax.set_title(title)
    return fig


# Function to create and resample spline
def create_resampled_spline(points, num_points=1000):
    """
    Resample the interpolating spline through `points` (N, d) to `num_points` points.

    Repeated consecutive points (common in tracked or pixel-snapped polylines) are
    dropped first, because splprep rejects them. The degree is min(3, n - 1) for n
    distinct points; fewer than 2 distinct points raise ValueError.

    Changed (D23, R7-07, 2026-11): repeated consecutive points are dropped instead
    of making splprep fail with 'Error on input data'.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError(f"points must be an (N, d) array with N >= 2, got shape {points.shape}.")
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
    points = points[keep]
    if len(points) < 2:
        raise ValueError(
            f"A spline needs at least 2 distinct points, got {len(points)}."
        )
    # splprep needs more points than the spline degree: lower it for 2-3 points
    tck, _ = splprep(points.T, s=0, k=min(3, len(points) - 1))
    u_fine = np.linspace(0, 1, num_points)
    spline_points = np.array(splev(u_fine, tck)).T
    return spline_points


# Calculate vectors
def _clip_polyline_to_box(points, lo, hi):
    """
    Return the part of a polyline inside the box [lo, hi], with exact entry and exit points.

    Each segment is clipped with the Liang-Barsky parameters; the result runs from the
    first point where the curve enters the box to the last point where it leaves it
    (interior vertices in between are kept). Returns an empty (0, d) array if the
    polyline never touches the box.
    """
    points = np.asarray(points, dtype=float)
    p0, p1 = points[:-1], points[1:]
    d = p1 - p0
    t0 = np.zeros(len(d))
    t1 = np.ones(len(d))
    inside = np.ones(len(d), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(points.shape[1]):
            moving = d[:, k] != 0
            ta = (lo[k] - p0[:, k]) / d[:, k]
            tb = (hi[k] - p0[:, k]) / d[:, k]
            t0 = np.where(moving, np.maximum(t0, np.minimum(ta, tb)), t0)
            t1 = np.where(moving, np.minimum(t1, np.maximum(ta, tb)), t1)
            # a segment parallel to this axis is inside only if it lies within the slab
            inside &= moving | ((p0[:, k] >= lo[k]) & (p0[:, k] <= hi[k]))
    hit = np.flatnonzero(inside & (t0 <= t1))
    if len(hit) == 0:
        return np.empty((0, points.shape[1]))
    first, last = hit[0], hit[-1]
    entry = p0[first] + t0[first] * d[first]
    exit_ = p0[last] + t1[last] * d[last]
    clipped = np.vstack([entry, points[first + 1 : last + 1], exit_])
    # drop repeated points (entry/exit on a vertex), which spline fitting rejects
    keep = np.r_[True, np.any(np.diff(clipped, axis=0) != 0, axis=1)]
    return clipped[keep]


def calculate_vector_difference(spline_origin, spline_endpoint, overlap_only=False, num_points=1000,
                                resample=True):
    '''
    Return the vector difference of equally resampled splines.

    Parameters:
    - spline_origin: The origin spline points (N x d array)
    - spline_endpoint: The endpoint spline points (M x d array)
    - overlap_only: If True, only consider the part of both splines inside the
      intersection of their bounding boxes (each is clipped exactly at the box faces
      and resampled to `num_points`)
    - num_points: Number of points to resample the splines
    - resample: If True (default), both splines are resampled to `num_points`
      points, uniform in the chord-length parameter, so the i-th vectors compare
      corresponding positions along each curve whatever the input sampling. If
      False (and not `overlap_only`), equal-length inputs are paired by index and
      different lengths raise ValueError. Repeated consecutive points are dropped
      before resampling; a spline with fewer than 2 distinct points raises
      ValueError (use ``resample=False`` to pair such inputs by index).

    Returns:
    - vectors: A (num_points, d) array (N x d when paired by index). With
      `overlap_only` and no overlap the vectors are all NaN.

    Changed (D23, R7-07, 2026-11): equal-length inputs are resampled by default (they
    were paired by index and `num_points` was ignored, so two samplings of the same
    curve could differ by the curve's length); new `resample` parameter. No overlap
    returns NaN (it returned zeros, which looks exactly like perfect agreement).
    '''
    spline_origin = np.asarray(spline_origin)
    spline_endpoint = np.asarray(spline_endpoint)

    if overlap_only == False:
        if resample:
            spline_origin = create_resampled_spline(spline_origin, num_points=num_points)
            spline_endpoint = create_resampled_spline(spline_endpoint, num_points=num_points)
        elif spline_origin.shape != spline_endpoint.shape:
            raise ValueError(
                f"With resample=False the splines must have the same shape, got "
                f"{spline_origin.shape} and {spline_endpoint.shape}."
            )
        vectors = spline_endpoint - spline_origin
        return vectors


    # Calculate the overlap region if overlap_only is True
    # Bounding box of the overlapping region
    start = np.maximum(np.min(spline_origin, axis=0), np.min(spline_endpoint, axis=0))
    end = np.minimum(np.max(spline_origin, axis=0), np.max(spline_endpoint, axis=0))

    # Clip each polyline exactly at the box faces; keeping only the sample points that
    # fall inside would start each overlap at its first interior sample, biasing the
    # difference by up to one sample spacing
    overlap_origin = _clip_polyline_to_box(spline_origin, start, end)
    overlap_endpoint = _clip_polyline_to_box(spline_endpoint, start, end)

    # Resample the cropped splines to the common number of points
    # This requires that the overlap region has enough points and spans a valid curve segment
    if len(overlap_origin) > 1 and len(overlap_endpoint) > 1:
        common_origin = create_resampled_spline(overlap_origin, num_points=num_points)
        common_endpoint = create_resampled_spline(overlap_endpoint, num_points=num_points)
    else:
        # No overlap: undefined, not zero
        return np.full((num_points, spline_origin.shape[1]), np.nan)

    vectors = common_endpoint - common_origin
    return vectors


# Calculate tangent vectors along the NT spline
def calculate_tangent_vectors(spline):
    # matth: Use central differences (O(h^2)) instead of forward differences (O(h))
    # to avoid the half-step phase shift and correctly handle boundary conditions.
    tangents = np.gradient(spline, axis=0)
    return tangents


# Function to project vectors onto the plane orthogonal to tangent vectors
def project_onto_plane(vectors, tangent_vectors):
    '''
    v_plane = v - proj_t (v), where
    proj_t (v) = ( <v⋅t> / <t⋅t> ) t
    '''
    # Compute the dot product of each vector with its corresponding tangent vector
    dot_products = np.sum(vectors * tangent_vectors, axis=1)

    # Compute the norm squared of each tangent vector
    tangent_norms_squared = np.sum(tangent_vectors**2, axis=1)

    # Avoid division by zero for zero-length tangents
    # (e.g. if the curve has identical consecutive points)
    valid = tangent_norms_squared > 0

    # Calculate the projection of each vector onto its corresponding tangent vector
    projections = np.zeros(vectors.shape, dtype=np.result_type(vectors, tangent_vectors, float))
    projections[valid] = (
        dot_products[valid, None] / tangent_norms_squared[valid, None]
    ) * tangent_vectors[valid]

    # Subtract the projections from the original vectors to get the projections onto the plane
    projected_vectors = vectors - projections

    return projected_vectors


# Function to normalize vectors to unit length
def normalize_vectors(vectors):
    """Scale each row to unit length; zero vectors stay zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = np.zeros(np.shape(vectors), dtype=float)
    np.divide(vectors, norms, out=unit_vectors, where=norms > 0)
    return unit_vectors

def _length_pixel_sizes(pixel_sizes, ndim, name="pixel_sizes"):
    """Per-axis pixel sizes (µm) for the last `ndim` of Z, Y, X; None or missing -> 1.0."""
    if pixel_sizes is None:
        return (1.0,) * ndim
    if not isinstance(pixel_sizes, dict) and np.ndim(pixel_sizes) > 1:
        pixel_sizes = list(np.ravel(np.asarray(pixel_sizes, dtype=object)))
    values, _, _ = resolve_pixel_sizes(pixel_sizes, ndim, stacklevel=3)
    out = []
    for axis, value in zip(('Z', 'Y', 'X')[-ndim:], values):
        if value is None:
            warnings.warn(
                f"No pixel size for axis {axis} in {name}; using 1.0.", stacklevel=3
            )
            out.append(1.0)
        else:
            out.append(float(value))
    return tuple(out)


def spline_length(points, pixel_sizes=None):
    """
    Arc length of a polyline / sampled spline given as (N, d) points, d = 2 or 3.

    Point-major, like the rest of this module: one row per point, columns in
    (Z, Y, X) or (Y, X) order.

    Parameters:
    - points: (N, 2) or (N, 3) array.
    - pixel_sizes: Pixel sizes for the columns, in the package's usual forms (dict
      {'Z','Y','X'} with an optional length 'unit', a sequence in column order (a
      [Z, Y, X] triple also works for 2D), a scalar) or None (1.0, i.e. pixels).
      Missing entries fall back to 1.0 with a warning (a missing Z with known Y and
      X takes the finest of them).

    Returns:
    - The length in µm (pixels where no size is given), as a float.

    New (D23, R7-06, 2026-11).
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(f"points must have shape (N, 2) or (N, 3), got {points.shape}.")
    sizes = np.asarray(_length_pixel_sizes(pixel_sizes, points.shape[1]))
    if len(points) < 2:
        return 0.0
    steps = np.diff(points, axis=0) * sizes
    return float(np.sum(np.sqrt(np.sum(steps ** 2, axis=1))))


def calculate_spline_length(coords, pixel_sizes=None, resolution=None):
    """
    Calculates the arc length of a 2D or 3D spline given evaluated points.

    Axis-major: `coords` holds one array per axis. For an (N, d) array of points
    use `spline_length`.

    Parameters:
    - coords: List or tuple of arrays in order [Z, Y, X] for 3D or [Y, X] for 2D.
    - pixel_sizes: Pixel sizes as a sequence/array in the same axis order as `coords`
      ([Z, Y, X] or [Y, X]; a [Z, Y, X] triple is also accepted for 2D), a dictionary
      {'Z','Y','X'} (with an optional length 'unit', converted to µm), a scalar, or None.
      A missing Z with known Y and X takes the finest of them; other missing or None
      entries fall back to 1.0. Both warn. Non-finite or non-positive sizes raise.
    - resolution: Deprecated alias of `pixel_sizes`.

    Returns:
    - Real-world length in µm (in pixels for axes without a size).

    Changed (D01, 2026-11): resolution goes through the package's shared
    parser: units are converted to µm, a missing Z takes the finest of Y/X
    (it used to be 1.0), and non-positive sizes raise ValueError.

    Changed (D23, R7-06, R8-18, 2026-11): the parameter is `pixel_sizes`
    (`resolution=` warns, FutureWarning, removed in 0.1.0). An ndarray with 2 or 3
    rows and at most 3 columns, e.g. 3 points as (3, 3), is ambiguous (rows are read
    as axes) and warns; use `spline_length` for (N, d) points.
    """
    if resolution is not None:
        if pixel_sizes is not None:
            raise TypeError("Pass pixel_sizes or the deprecated resolution, not both.")
        warn_deprecated("calculate_spline_length(resolution=...)", "pixel_sizes=")
        pixel_sizes = resolution

    ambiguous = (
        isinstance(coords, np.ndarray) and coords.ndim == 2
        and coords.shape[0] in (2, 3) and coords.shape[1] <= 3
    )
    if ambiguous:
        warnings.warn(
            f"calculate_spline_length reads the rows of a {coords.shape} array as the "
            f"{coords.shape[0]} axes ([Z,] Y, X). If the rows are points, use "
            f"spline_length(points) instead.", UserWarning, stacklevel=2
        )

    ndim = len(coords)
    if ndim not in [2, 3]:
        raise ValueError("coords must be a list of 2 or 3 arrays ([Z], Y, X)")

    sizes = _length_pixel_sizes(pixel_sizes, ndim)
    # Euclidean distance: sqrt(sum of squared, scaled differences)
    scaled_diffs_sq = [np.diff(np.asarray(c) * s) ** 2 for c, s in zip(coords, sizes)]
    segment_lengths = np.sqrt(sum(scaled_diffs_sq))

    return np.sum(segment_lengths)
