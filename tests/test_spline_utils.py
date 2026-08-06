import numpy as np
import pytest

from eigenp_utils.spline_utils import generate_random_3d_coordinates, fit_cubic_spline, create_3d_image_from_spline, create_nd_image_from_spline, create_resampled_spline, calculate_vector_difference, calculate_tangent_vectors, project_onto_plane, normalize_vectors, calculate_spline_length
from eigenp_utils.spline_utils import project_onto_plane, calculate_spline_length



# =========================================
# Source: test_spline_utils.py
# =========================================

def test_generate_random_3d_coordinates():
    """Test that generate random 3d coordinates works as expected."""
    points = generate_random_3d_coordinates(num_points=5, seed=42)
    assert points.shape == (5, 3)
    assert np.all((points >= 0) & (points <= 100))

def test_fit_cubic_spline():
    """Test that fit cubic spline works as expected."""
    points = generate_random_3d_coordinates(num_points=10)
    tck = fit_cubic_spline(points)
    # splprep returns tck, which is actually a tuple of length 3: (t, c, k)
    assert len(tck) == 3

    tck, u = fit_cubic_spline(points, return_u=True)
    assert len(u) == 10

def test_create_3d_image_from_spline():
    """Test that create 3d image from spline works as expected."""
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
    """Test that create nd image from spline works as expected."""
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
    """Test that create resampled spline works as expected."""
    points = np.array([
        [10, 10, 10],
        [20, 20, 20],
        [30, 30, 30],
        [40, 40, 40]
    ])
    resampled = create_resampled_spline(points, num_points=10)
    assert resampled.shape == (10, 3)

def test_calculate_vector_difference():
    """Test that calculate vector difference works as expected."""
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
    """Test that calculate vector difference overlap works as expected."""
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
    """Test that calculate tangent vectors works as expected."""
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
    """Test that project onto plane works as expected."""
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
    """Test that normalize vectors works as expected."""
    vectors = np.array([
        [3, 4], # norm 5
        [1, 0]  # norm 1
    ])
    normalized = normalize_vectors(vectors)
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [1.0, 0.0])

def test_calculate_spline_length():
    """Test that calculate spline length works as expected."""
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

# =========================================
# Source: test_spline_invariants.py
# =========================================

def test_project_onto_plane_orthogonality():
    """
    🔎 Testr: Verify Orthogonality Invariant of project_onto_plane
    💡 What: The projection operator P(v) onto the plane normal to tangent vector t must be orthogonal to t.
    🎯 Why: Orthogonality ensures that all motion along the tangent direction is completely removed.
    🧪 How: Calculate the dot product between the projected vector and the tangent vector. It must be exactly zero.
    📐 Theory: <P(v), t> = 0
    """
    np.random.seed(42)
    vectors = np.random.randn(100, 3) * 10
    tangents = np.random.randn(100, 3) * 5

    # Project
    projected = project_onto_plane(vectors, tangents)

    # Dot product should be strictly zero (within float precision)
    dot_products = np.sum(projected * tangents, axis=1)

    # Using a tight absolute tolerance to verify numerical stability
    assert np.allclose(dot_products, 0.0, atol=1e-12), "Projected vectors are not strictly orthogonal to tangents"


def test_project_onto_plane_idempotence():
    """
    🔎 Testr: Verify Idempotence Invariant of project_onto_plane
    💡 What: P(P(v)) = P(v)
    🎯 Why: Projecting an already-projected vector should have no further effect, verifying P is a true projection operator.
    🧪 How: Project vectors once, then project the result again using the same tangents.
    📐 Theory: P^2 = P
    """
    np.random.seed(43)
    vectors = np.random.randn(100, 3)
    tangents = np.random.randn(100, 3)

    projected_once = project_onto_plane(vectors, tangents)
    projected_twice = project_onto_plane(projected_once, tangents)

    assert np.allclose(projected_once, projected_twice, atol=1e-14), "Projection operator is not idempotent"


def test_project_onto_plane_null_space():
    """
    🔎 Testr: Verify Null Space of project_onto_plane
    💡 What: Projecting a vector that is perfectly aligned with the tangent should yield the zero vector.
    🎯 Why: The tangent direction spans the null space of the projection operator onto the normal plane.
    🧪 How: Pass the tangents themselves as the vectors to be projected.
    📐 Theory: P(a * t) = 0 for any scalar a.
    """
    np.random.seed(44)
    tangents = np.random.randn(100, 3)

    # Vector is a scaled version of tangent
    vectors = tangents * np.random.randn(100, 1)

    projected = project_onto_plane(vectors, tangents)

    assert np.allclose(projected, 0.0, atol=1e-14), "Vectors purely in the tangent space were not fully rejected"


def test_project_onto_plane_pythagorean():
    """
    🔎 Testr: Verify Pythagorean Theorem in Projection
    💡 What: |v|^2 = |P(v)|^2 + |v - P(v)|^2
    🎯 Why: Energy (squared norm) must be conserved between the orthogonal components.
    🧪 How: Check norm squared of original vs sum of norm squared of parallel and orthogonal components.
    📐 Theory: ||v||^2 = ||proj_orthogonal(v)||^2 + ||proj_parallel(v)||^2
    """
    np.random.seed(45)
    vectors = np.random.randn(100, 3)
    tangents = np.random.randn(100, 3)

    projected = project_onto_plane(vectors, tangents)
    # The component parallel to tangent: proj_parallel = v - P(v)
    parallel_component = vectors - projected

    norm_sq_v = np.sum(vectors**2, axis=1)
    norm_sq_proj = np.sum(projected**2, axis=1)
    norm_sq_parallel = np.sum(parallel_component**2, axis=1)

    assert np.allclose(norm_sq_v, norm_sq_proj + norm_sq_parallel, atol=1e-12), "Pythagorean identity violated during projection"


def test_project_onto_plane_degenerate():
    """
    🔎 Testr: Verify Degenerate Behavior of project_onto_plane
    💡 What: Zero-length tangent vectors should safely leave the original vector unchanged (or zero), without NaN generation.
    🎯 Why: Ensures numerical robustness when the curve has identical consecutive points.
    🧪 How: Provide exact zero vectors as tangents.
    """
    vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tangents = np.zeros_like(vectors)

    projected = project_onto_plane(vectors, tangents)

    # Current implementation avoids division by zero and leaves the vector intact
    assert not np.any(np.isnan(projected)), "NaN generated for zero-length tangent"
    assert np.allclose(projected, vectors), "Vectors should pass through unchanged if tangent is zero-length"


def test_calculate_spline_length_analytical_circle():
    """
    🔎 Testr: Verify Analytical Convergence of Arc Length
    💡 What: Discretizing a circle with high resolution must converge to its analytical circumference (2*pi*R).
    🎯 Why: Proves the discrete segment summation is mathematically grounded and free of scaling biases.
    🧪 How: Generate [Y, X] coordinates for a circle and calculate length.
    📐 Theory: lim (n -> inf) sum(||p_i - p_{i-1}||) = 2*pi*R
    """
    # 2D Circle: R = 10
    theta = np.linspace(0, 2 * np.pi, 10000)
    R = 10.0
    y = R * np.sin(theta)
    x = R * np.cos(theta)

    coords = [y, x]
    length = calculate_spline_length(coords)

    expected_length = 2 * np.pi * R
    # Should be extremely close with 10k points
    assert np.isclose(length, expected_length, rtol=1e-5), f"Arc length of circle {length} diverges from 2*pi*R {expected_length}"


def test_calculate_spline_length_isometric_invariance():
    """
    🔎 Testr: Verify Isometric Invariance of Arc Length
    💡 What: Translation and rotation of the point coordinates must strictly NOT change the arc length.
    🎯 Why: Arc length is an intrinsic geometric property independent of the coordinate frame.
    🧪 How: Calculate length of a curve, translate/rotate it, and ensure length is identical.
    """
    # 3D curve
    t = np.linspace(0, 10, 500)
    z = np.sin(t)
    y = np.cos(t)
    x = t * 2.0

    coords_original = [z, y, x]
    len_original = calculate_spline_length(coords_original)

    # 1. Translation
    coords_translated = [z + 100, y - 50, x + 3.14]
    len_translated = calculate_spline_length(coords_translated)
    assert np.isclose(len_original, len_translated, rtol=1e-12), "Translation altered arc length!"

    # 2. Rotation (Rotate around Z axis by pi/4)
    theta = np.pi / 4
    y_rot = y * np.cos(theta) - x * np.sin(theta)
    x_rot = y * np.sin(theta) + x * np.cos(theta)

    coords_rotated = [z, y_rot, x_rot]
    len_rotated = calculate_spline_length(coords_rotated)
    assert np.isclose(len_original, len_rotated, rtol=1e-12), "Rotation altered arc length!"


def test_calculate_spline_length_resolution_scaling():
    """
    🔎 Testr: Verify Resolution Scaling Linearity
    💡 What: Arc length calculated with anisotropic physical resolution must scale segments precisely.
    🎯 Why: Ensures physical coordinates correctly map to voxel coordinates using spacing dictionaries.
    🧪 How: Compare custom resolution dictionary to scaling the original coordinates by the resolution.
    """
    z = np.linspace(0, 5, 100)
    y = np.linspace(0, 10, 100)
    x = np.linspace(0, 15, 100)

    res = {'Z': 2.0, 'Y': 0.5, 'X': 1.0}

    # Method A: Calculate length natively with resolution mapping
    len_mapped = calculate_spline_length([z, y, x], resolution=res)

    # Method B: Pre-scale coordinates and calculate length with unit resolution
    len_prescaled = calculate_spline_length([z * 2.0, y * 0.5, x * 1.0])

    assert np.isclose(len_mapped, len_prescaled, rtol=1e-12), "Resolution mapping is mathematically inconsistent with coordinate scaling"
