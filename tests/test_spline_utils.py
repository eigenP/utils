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
    normalize_vectors,
    calculate_spline_length
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

def test_calculate_vector_difference_overlap():
    # line1 from (0,0) to (20,20)
    line1 = np.array([[0,0], [5,5], [10,10], [15,15], [20,20]])
    # line2 from (10,11) to (30,31)
    line2 = np.array([[10,11], [15,16], [20,21], [25,26], [30,31]])

    # Line 1 overlaps the Y range [11, 20] with Line 2.
    # The bounding box of line1 is [[0, 0], [20, 20]]
    # The bounding box of line2 is [[10, 11], [30, 31]]
    # Intersection is start=[10, 11], end=[20, 20]
    # For line1, X in [10, 20], Y in [11, 20] means points between [11, 11] and [20, 20].
    # For line2, X in [10, 20], Y in [11, 20] means points between [10, 11] and [19, 20].
    # The mathematical vector difference between the re-parameterized segments should be [10-11, 11-11] = [-1, 0]
    # Wait, line1 is y=x, line2 is y=x+1.
    # If line1 is parameterized from (11, 11) to (20, 20)
    # and line2 is parameterized from (10, 11) to (19, 20).
    # Then the difference (line2 - line1) at corresponding parameter values is [-1, 0].

    resampled1 = create_resampled_spline(line1, num_points=20)
    resampled2 = create_resampled_spline(line2, num_points=20)

    vectors = calculate_vector_difference(resampled1, resampled2, overlap_only=True, num_points=5)

    # Overlap origin range: from [11.57, 11.57] to [20.0, 20.0]
    # Overlap endpoint range: from [10.0, 11.0] to [18.42, 19.42]
    # Difference = endpoint - origin = [-1.57, -0.57] uniformly over re-parameterization.

    assert vectors.shape == (5, 2)
    diff_y = vectors[0, 1]
    diff_x = vectors[0, 0]
    assert np.allclose(vectors[:, 1], diff_y)
    assert np.allclose(vectors[:, 0], diff_x)

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

def test_calculate_spline_length():
    # 2D case, straight line from (0,0) to (3,4), length should be 5
    coords_2d = [np.array([0, 3]), np.array([0, 4])] # [Y, X]
    length_2d = calculate_spline_length(coords_2d)
    assert np.isclose(length_2d, 5.0)

    # 3D case, straight line from (0,0,0) to (2,3,6), length should be 7
    coords_3d = [np.array([0, 2]), np.array([0, 3]), np.array([0, 6])] # [Z, Y, X]
    length_3d = calculate_spline_length(coords_3d)
    assert np.isclose(length_3d, 7.0)

    # Test custom resolution as list [Z, Y, X] for 3D
    resolution_list = [2.0, 1.0, 0.5] # Z, Y, X
    # After scaling:
    # Z diff: (2-0)*2.0 = 4.0
    # Y diff: (3-0)*1.0 = 3.0
    # X diff: (6-0)*0.5 = 3.0
    # Length = sqrt(4^2 + 3^2 + 3^2) = sqrt(16 + 9 + 9) = sqrt(34)
    length_3d_res_list = calculate_spline_length(coords_3d, resolution=resolution_list)
    assert np.isclose(length_3d_res_list, np.sqrt(34))

    # Test custom resolution as dict for 2D
    resolution_dict = {'Y': 2.0, 'X': 0.5}
    # After scaling:
    # Y diff: (3-0)*2.0 = 6.0
    # X diff: (4-0)*0.5 = 2.0
    # Length = sqrt(6^2 + 2^2) = sqrt(36 + 4) = sqrt(40)
    length_2d_res_dict = calculate_spline_length(coords_2d, resolution=resolution_dict)
    assert np.isclose(length_2d_res_dict, np.sqrt(40))

    # Test exception for 1D coordinates
    with pytest.raises(ValueError, match="coords must be a list of 2 or 3 arrays \\(\\[Z\\], Y, X\\)"):
        calculate_spline_length([np.array([0, 1])])
