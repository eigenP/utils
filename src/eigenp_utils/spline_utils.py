import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to generate random 3D coordinates
def generate_random_3d_coordinates(num_points=10, seed=None):
    if seed is not None:
        np.random.seed(seed)  # For reproducible results
    return np.random.rand(num_points, 3) * 100  # Scaling to get a wider range


# Function to fit a cubic spline through 3D coordinates
def fit_cubic_spline(coordinates, return_u = False):
    # Fit a cubic spline (B-spline) to the 3D coordinates
    tck, u = splprep([coordinates[:,-3], coordinates[:,-2], coordinates[:,-1]], s=0)
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

    # Function to safely mark a point in the image
    def mark_point(image, point):
        try:
            image[point[0], point[1], point[2]] = 1
        except IndexError:  # In case the point is outside the image bounds
            pass

    # Mark the path of the spline in the 3D image
    for i in range(num_points):
        point = (int(x[i]), int(y[i]), int(z[i]))
        mark_point(image, point)

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

    # Function to safely mark a point in the image
    def mark_point(image, point):
        try:
            # Convert point to a tuple of indices and set the value
            image[tuple(point)] = 1
        except IndexError:  # In case the point is outside the image bounds
            pass

    # Convert points to integer indices and mark the path of the spline in the image
    for i in range(num_points):
        point = tuple(int(coord[i]) for coord in points)
        mark_point(image, point)

    return image


# Function to plot the initial points and the spline
def plot_3d_spline(coordinates, spline_tck, num_points=1000):
    """
    Plot the initial points and the cubic spline in 3D.

    Parameters:
    - coordinates: The initial 3D coordinates.
    - spline_tck: The parameters of the fitted spline.
    - num_points: Number of points to sample along the spline.
    """
    # Sample points along the spline
    u_fine = np.linspace(0, 1, num_points)
    x, y, z = splev(u_fine, spline_tck)

    # Create the 3D plot
    fig = plt.figure(figsize=(4, 4), layout = 'constrained')
    ax = fig.add_subplot(111, projection='3d')

    # Plot the initial points
    ax.scatter(coordinates[:, -3], coordinates[:, -2], coordinates[:, -1], color='red', label='Initial Points')

    # Plot the spline
    ax.plot(x, y, z, color='blue', label='Cubic Spline')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    plt.title('3D Cubic Spline through Random Points')
    plt.show()


# Function to create and resample spline
def create_resampled_spline(points, num_points=1000):
    tck, _ = splprep(points.T, s=0)
    u_fine = np.linspace(0, 1, num_points)
    spline_points = np.array(splev(u_fine, tck)).T
    return spline_points


# Calculate vectors
def calculate_vector_difference(spline_origin, spline_endpoint, overlap_only=False, num_points=1000):
    '''
    Return the vector difference of equally resampled splines.

    Parameters:
    - spline_origin: The origin spline points (N x d array)
    - spline_endpoint: The endpoint spline points (N x d array)
    - overlap_only: If True, only consider the overlap region of the splines
    - num_points: Number of points to resample the splines

    Returns:
    - vectors: An N x d array where N is the number of points and d the dimension of the space
    '''

    if overlap_only == False:
        vectors = spline_endpoint - spline_origin
        return vectors

    if spline_origin.shape != spline_endpoint.shape:
        # Resample both splines to ensure they have the same number of points
        spline_origin = create_resampled_spline(spline_origin, num_points=num_points)
        spline_endpoint = create_resampled_spline(spline_endpoint, num_points=num_points)


    # Calculate the overlap region if overlap_only is True
    # Bounding box of the overlapping region
    start = np.maximum(np.min(spline_origin, axis=0), np.min(spline_endpoint, axis=0))
    end = np.minimum(np.max(spline_origin, axis=0), np.max(spline_endpoint, axis=0))

    # Mask to keep only the points that fall inside the geometric bounding box overlap
    mask_origin = np.all((spline_origin >= start) & (spline_origin <= end), axis=1)
    mask_endpoint = np.all((spline_endpoint >= start) & (spline_endpoint <= end), axis=1)

    overlap_origin = spline_origin[mask_origin]
    overlap_endpoint = spline_endpoint[mask_endpoint]

    # Resample the cropped splines to the common number of points
    # This requires that the overlap region has enough points and spans a valid curve segment
    if len(overlap_origin) > 1 and len(overlap_endpoint) > 1:
        common_origin = create_resampled_spline(overlap_origin, num_points=num_points)
        common_endpoint = create_resampled_spline(overlap_endpoint, num_points=num_points)
    else:
        # Fallback if there is no significant overlap
        common_origin = np.zeros((num_points, spline_origin.shape[1]))
        common_endpoint = np.zeros((num_points, spline_endpoint.shape[1]))

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
    projections = np.zeros_like(vectors)
    projections[valid] = (
        dot_products[valid, None] / tangent_norms_squared[valid, None]
    ) * tangent_vectors[valid]

    # Subtract the projections from the original vectors to get the projections onto the plane
    projected_vectors = vectors - projections

    return projected_vectors


# Function to normalize vectors to unit length
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms
    return unit_vectors

def calculate_spline_length(coords, resolution=None):
    """
    Calculates the arc length of a 2D or 3D spline given evaluated points.

    Parameters:
    - coords: List or tuple of arrays in order [Z, Y, X] for 3D or [Y, X] for 2D.
    - resolution: List [Z, Y, X], dictionary {'Z','Y','X'}, or None.

    Returns:
    - Real-world length in units of the pixel size.
    """
    ndim = len(coords)
    if ndim not in [2, 3]:
        raise ValueError("coords must be a list of 2 or 3 arrays ([Z], Y, X)")

    # Default resolutions
    res = {'X': 1.0, 'Y': 1.0, 'Z': 1.0}

    if isinstance(resolution, dict):
        res.update(resolution)
    elif isinstance(resolution, (list, tuple)) and len(resolution) >= 3:
        # Standard convention in this notebook: [Z, Y, X]
        res['Z'], res['Y'], res['X'] = resolution[0], resolution[1], resolution[2]

    # Map coordinates to resolutions:
    # If 3D, coords are [Z, Y, X]
    # If 2D, coords are [Y, X]
    scaled_diffs_sq = []

    if ndim == 3:
        # Scale Z
        scaled_diffs_sq.append(np.diff(coords[0] * res['Z'])**2)
        # Scale Y
        scaled_diffs_sq.append(np.diff(coords[1] * res['Y'])**2)
        # Scale X
        scaled_diffs_sq.append(np.diff(coords[2] * res['X'])**2)
    else:
        # Scale Y
        scaled_diffs_sq.append(np.diff(coords[0] * res['Y'])**2)
        # Scale X
        scaled_diffs_sq.append(np.diff(coords[1] * res['X'])**2)

    # Euclidean distance: sqrt(sum of squared differences)
    segment_lengths = np.sqrt(sum(scaled_diffs_sq))

    return np.sum(segment_lengths)
