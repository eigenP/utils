import numpy as np
import pytest
from eigenp_utils.spline_utils import (
    generate_random_3d_coordinates,
    fit_cubic_spline,
    create_3d_image_from_spline,
    create_nd_image_from_spline,
    create_resampled_spline,
    calculate_vector_difference,
    calculate_tangent_vectors,
    project_onto_plane,
    normalize_vectors
)

def test_generate_random_3d_coordinates():
    points = generate_random_3d_coordinates(num_points=5, seed=42)
    assert points.shape == (5, 3)
    assert np.all((points >= 0) & (points <= 100))

def test_fit_cubic_spline():
    points = generate_random_3d_coordinates(num_points=10)
    tck = fit_cubic_spline(points)
    # splprep returns tck, which is actually a tuple of length 3: (t, c, k)
    assert len(tck) == 3

    tck, u = fit_cubic_spline(points, return_u=True)
    assert len(u) == 10

def test_create_3d_image_from_spline():
    points = np.array([
        [10, 10, 10],
        [20, 20, 20],
        [30, 30, 30],
        [40, 40, 40],
        [50, 50, 50]
    ])
    tck = fit_cubic_spline(points)
    img = create_3d_image_from_spline(tck, shape=(60, 60, 60), num_points=100)

    assert img.shape == (60, 60, 60)
    assert np.any(img == 1) # some points must be drawn

def test_create_nd_image_from_spline():
    points = np.array([
        [10, 10],
        [20, 20],
        [30, 30],
        [40, 40]
    ])
    from scipy.interpolate import splprep
    tck, u = splprep(points.T, s=0)
    img = create_nd_image_from_spline(tck, shape=(50, 50), num_points=100)

    assert img.shape == (50, 50)
    assert np.any(img == 1)

def test_create_resampled_spline():
    points = np.array([
        [10, 10, 10],
        [20, 20, 20],
        [30, 30, 30],
        [40, 40, 40]
    ])
    resampled = create_resampled_spline(points, num_points=10)
    assert resampled.shape == (10, 3)

def test_calculate_vector_difference():
    # Straight lines with enough points for splprep (needs m > k, default k=3)
    line1 = np.array([[0,0], [5,5], [10,10], [15,15]])
    line2 = np.array([[0,1], [5,6], [10,11], [15,16]]) # Shifted +1 in Y

    resampled1 = create_resampled_spline(line1, num_points=5)
    resampled2 = create_resampled_spline(line2, num_points=5)

    vectors = calculate_vector_difference(resampled1, resampled2)
    assert vectors.shape == (5, 2)
    assert np.allclose(vectors[:, 1], 1.0) # difference in Y is 1.0
    assert np.allclose(vectors[:, 0], 0.0) # difference in X is 0.0

def test_calculate_tangent_vectors():
    # Line along X axis
    line = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0]
    ])
    tangents = calculate_tangent_vectors(line)

    assert tangents.shape == (3, 3)
    assert np.allclose(tangents[:, 0], 1.0)
    assert np.allclose(tangents[:, 1:], 0.0)

def test_project_onto_plane():
    vectors = np.array([
        [1, 1, 0],
        [-1, 1, 0]
    ])
    tangents = np.array([
        [1, 0, 0], # tangent along X
        [1, 0, 0]
    ])

    # projection should remove the X component
    projected = project_onto_plane(vectors, tangents)
    assert np.allclose(projected[:, 0], 0.0)
    assert np.allclose(projected[:, 1], 1.0)

def test_normalize_vectors():
    vectors = np.array([
        [3, 4], # norm 5
        [1, 0]  # norm 1
    ])
    normalized = normalize_vectors(vectors)
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [1.0, 0.0])
