import numpy as np
import pytest
from eigenp_utils.spline_utils import project_onto_plane, calculate_spline_length

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
