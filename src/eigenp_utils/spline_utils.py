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
    start = max(np.min(spline_origin, axis=0), np.min(spline_endpoint, axis=0))
    end = min(np.max(spline_origin, axis=0), np.max(spline_endpoint, axis=0))

    # Resample to the overlap region; you may need a custom function to handle this
    # For simplicity, this example will use linear interpolation to simulate this
    # Usually you would use an interpolation library or extend the splprep/splev usage here
    common_u = np.linspace(start, end, num_points)
    common_origin = np.interp(common_u, np.linspace(0, 1, len(spline_origin)), spline_origin)
    common_endpoint = np.interp(common_u, np.linspace(0, 1, len(spline_endpoint)), spline_endpoint)

    vectors = common_endpoint - common_origin
    return vectors


# Calculate tangent vectors along the NT spline
def calculate_tangent_vectors(spline):
    tangents = np.diff(spline, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])  # To make it the same length as the spline
    return tangents


# Function to project vectors onto the plane orthogonal to tangent vectors
def project_onto_plane(vectors, tangent_vectors):
    '''
    v_plane = v - proj_t (v), where
    proj_t (v) = ( <t⋅t> / <v⋅t> ) t
    '''
    # Compute the dot product of each vector with its corresponding tangent vector
    dot_products = np.sum(vectors * tangent_vectors, axis=1)

    # Compute the norm squared of each tangent vector
    tangent_norms_squared = np.linalg.norm(tangent_vectors, axis=1)**2

    # Calculate the projection of each vector onto its corresponding tangent vector
    projections = (dot_products[:, None] / tangent_norms_squared[:, None]) * tangent_vectors

    # Subtract the projections from the original vectors to get the projections onto the plane
    projected_vectors = vectors - projections

    return projected_vectors


# Function to normalize vectors to unit length
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms
    return unit_vectors
