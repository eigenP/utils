import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import splev, splprep

from eigenp_utils.spline_utils import (
    calculate_spline_length,
    calculate_tangent_vectors,
    calculate_vector_difference,
    create_3d_image_from_spline,
    create_nd_image_from_spline,
    create_resampled_spline,
    fit_cubic_spline,
    generate_random_3d_coordinates,
    normalize_vectors,
    plot_3d_spline,
    project_onto_plane,
)


def test_generate_random_3d_coordinates():
    """Verify shape, bounds, reproducibility, and global RNG isolation.

    Strategy: snapshot and seed NumPy's global random state, call
    `generate_random_3d_coordinates` with an explicit seed, and verify the
    global stream is untouched while output matches `RandomState(seed)`.
    Expected outcome: returns `(5, 3)` values in `[0, 100]` matching the legacy
    `RandomState` sequence without mutating global `np.random` state.
    """
    saved_state = np.random.get_state()
    try:
        np.random.seed(123)
        expected_next = np.random.rand()

        np.random.seed(123)
        pts = generate_random_3d_coordinates(num_points=5, seed=42)
        assert np.random.rand() == expected_next
    finally:
        np.random.set_state(saved_state)

    assert pts.shape == (5, 3)
    assert np.all((pts >= 0.0) & (pts <= 100.0))
    np.testing.assert_array_equal(pts, np.random.RandomState(42).rand(5, 3) * 100)
    np.testing.assert_array_equal(pts, generate_random_3d_coordinates(num_points=5, seed=42))


def test_fit_cubic_spline():
    """Verify cubic B-spline interpolation on 3-column and >3-column coordinates.

    Strategy: fit splines to deterministic 3D points and to a 5-column array
    whose leading two columns hold arbitrary metadata, then evaluate at `u`.
    Expected outcome: `splev(u, tck)` reproduces the input 3D coordinates and
    ignores any columns preceding the last three.
    """
    points = generate_random_3d_coordinates(num_points=10, seed=7)
    tck = fit_cubic_spline(points)
    assert len(tck) == 3

    tck_u, u = fit_cubic_spline(points, return_u=True)
    assert len(u) == 10
    evaluated = np.column_stack(splev(u, tck_u))
    np.testing.assert_allclose(evaluated, points, atol=1e-6)

    # Extra leading columns (e.g. track ID, frame) must be ignored in favour of the last 3
    padded = np.column_stack([np.arange(10) * 999.0, np.full(10, -42.0), points])
    tck_pad, u_pad = fit_cubic_spline(padded, return_u=True)
    np.testing.assert_allclose(u_pad, u, atol=1e-12)
    np.testing.assert_allclose(np.column_stack(splev(u_pad, tck_pad)), points, atol=1e-6)


def test_create_3d_image_from_spline():
    """Verify 3D spline voxelization along a diagonal, auto-sizing, and upper-bound clipping.

    Strategy: fit a straight diagonal 3D spline from 10.5 to 40.5 and voxelize
    with an explicit shape, with `shape=None`, and with a smaller clipping shape.
    Expected outcome: marks the exact diagonal voxels `[10..40]`, auto-sizes to
    `ceil(40.5 * 1.2) = 49` cubed, and clips points >= 30 cleanly without error.
    """
    diag = np.tile(np.array([10.5, 20.5, 30.5, 40.5])[:, None], (1, 3))
    tck = fit_cubic_spline(diag)

    img = create_3d_image_from_spline(tck, shape=(50, 50, 50), num_points=500)
    assert img.shape == (50, 50, 50)
    assert img.dtype == np.uint8
    expected_voxels = np.tile(np.arange(10, 41)[:, None], (1, 3))
    np.testing.assert_array_equal(np.argwhere(img == 1), expected_voxels)

    # shape=None allocates a cube of ceil(max_extent * 1.2) = ceil(48.6) = 49 per axis
    img_auto = create_3d_image_from_spline(tck, shape=None, num_points=500)
    assert img_auto.shape == (49, 49, 49)
    np.testing.assert_array_equal(np.argwhere(img_auto == 1), expected_voxels)

    # Upper-bound clipping skips coordinates >= shape without raising IndexError
    img_clipped = create_3d_image_from_spline(tck, shape=(30, 30, 30), num_points=500)
    expected_clipped = np.tile(np.arange(10, 30)[:, None], (1, 3))
    np.testing.assert_array_equal(np.argwhere(img_clipped == 1), expected_clipped)


def test_create_nd_image_from_spline():
    """Verify nD spline voxelization in 2D and 4D, per-axis auto-sizing, and clipping.

    Strategy: voxelize diagonal 2D and 4D splines with explicit and clipping
    shapes, plus a rectangular 2D spline with `shape=None`.
    Expected outcome: marks exact diagonal hypervoxels, clips out-of-bounds
    upper coordinates, and auto-sizes each axis independently by `ceil(max * 1.2)`.
    """
    pts_2d = np.tile(np.array([5.5, 10.5, 15.5, 20.5])[:, None], (1, 2))
    tck_2d, _ = splprep(pts_2d.T, s=0)

    img_2d = create_nd_image_from_spline(tck_2d, shape=(25, 25), num_points=400)
    assert img_2d.shape == (25, 25) and img_2d.dtype == np.uint8
    expected_2d = np.tile(np.arange(5, 21)[:, None], (1, 2))
    np.testing.assert_array_equal(np.argwhere(img_2d == 1), expected_2d)

    # Upper-bound clipping in 2D
    img_2d_clip = create_nd_image_from_spline(tck_2d, shape=(15, 15), num_points=400)
    expected_2d_clip = np.tile(np.arange(5, 15)[:, None], (1, 2))
    np.testing.assert_array_equal(np.argwhere(img_2d_clip == 1), expected_2d_clip)

    # Non-square 2D auto-sizing: ceil([8.5, 40.5] * 1.2) = [11, 49]
    pts_rect = np.column_stack([np.array([2.5, 4.5, 6.5, 8.5]), np.array([10.5, 20.5, 30.5, 40.5])])
    tck_rect, _ = splprep(pts_rect.T, s=0)
    img_auto = create_nd_image_from_spline(tck_rect, shape=None, num_points=400)
    assert img_auto.shape == (11, 49)
    assert img_auto[2, 10] == 1 and img_auto[8, 40] == 1

    # 4D diagonal hypergrid
    pts_4d = np.tile(np.array([1.5, 2.5, 3.5, 4.5])[:, None], (1, 4))
    tck_4d, _ = splprep(pts_4d.T, s=0)
    img_4d = create_nd_image_from_spline(tck_4d, shape=(6, 6, 6, 6), num_points=200)
    assert img_4d.shape == (6, 6, 6, 6)
    expected_4d = np.tile(np.arange(1, 5)[:, None], (1, 4))
    np.testing.assert_array_equal(np.argwhere(img_4d == 1), expected_4d)


def test_create_resampled_spline():
    """Verify resampled spline endpoints, curve fidelity, and low-degree (2- and 3-point) inputs.

    Strategy: resample a quarter-circle helix (cubic), a 2-point line segment
    (degree 1), and a 3-point symmetric parabola (degree 2).
    Expected outcome: resampled points match exact endpoints, stay on the
    analytical curve, and handle 2-point and 3-point inputs without error.
    """
    theta = np.linspace(0.0, np.pi / 2, 15)
    arc = np.column_stack([np.cos(theta), np.sin(theta), theta])
    resampled = create_resampled_spline(arc, num_points=50)
    assert resampled.shape == (50, 3)
    np.testing.assert_allclose(resampled[0], arc[0], atol=1e-9)
    np.testing.assert_allclose(resampled[-1], arc[-1], atol=1e-9)
    # Every resampled point remains on the unit cylinder x^2 + y^2 = 1
    radii = np.hypot(resampled[:, 0], resampled[:, 1])
    np.testing.assert_allclose(radii, 1.0, atol=1e-3)

    # 2-point input lowers spline degree to k=1 (exact linear interpolation)
    pts2 = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
    res2 = create_resampled_spline(pts2, num_points=11)
    assert res2.shape == (11, 3)
    np.testing.assert_allclose(res2, np.linspace(pts2[0], pts2[1], 11), atol=1e-12)

    # 3-point symmetric input lowers spline degree to k=2 (exact parabola y = x*(2 - x))
    pts3 = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    res3 = create_resampled_spline(pts3, num_points=21)
    assert res3.shape == (21, 2)
    np.testing.assert_allclose(res3[0], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(res3[10], [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(res3[-1], [2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(res3[:, 1], res3[:, 0] * (2.0 - res3[:, 0]), atol=1e-12)


def test_calculate_vector_difference():
    """Verify full-length vector difference between two parallel shifted splines.

    Strategy: construct two parallel 2D lines offset by +1 along Y and compute
    their difference with `overlap_only=False`.
    Expected outcome: returns exact `[0.0, 1.0]` at every sample point.
    """
    line1 = np.array([[0, 0], [5, 5], [10, 10], [15, 15]], dtype=float)
    line2 = np.array([[0, 1], [5, 6], [10, 11], [15, 16]], dtype=float)

    resampled1 = create_resampled_spline(line1, num_points=5)
    resampled2 = create_resampled_spline(line2, num_points=5)

    vectors = calculate_vector_difference(resampled1, resampled2)
    assert vectors.shape == (5, 2)
    np.testing.assert_allclose(vectors, np.tile([0.0, 1.0], (5, 1)), atol=1e-12)


def test_calculate_vector_difference_overlap():
    """Verify exact box-clipped overlap difference and disjoint zero fallback.

    Strategy: compute `overlap_only=True` difference between two partially
    overlapping diagonal lines (`y = x` and `y = x + 1`), then repeat with a
    disjoint shifted copy.
    Expected outcome: overlapping segments clip exactly to box faces yielding
    `[-1.0, 0.0]` everywhere, while disjoint curves return the zero fallback.
    """
    line1 = np.array([[0, 0], [5, 5], [10, 10], [15, 15], [20, 20]], dtype=float)
    line2 = np.array([[10, 11], [15, 16], [20, 21], [25, 26], [30, 31]], dtype=float)

    # Bounding-box intersection is [10, 20] x [11, 20]:
    # line1 clips to (11, 11)->(20, 20), line2 clips to (10, 11)->(19, 20), diff = [-1, 0].
    resampled1 = create_resampled_spline(line1, num_points=20)
    resampled2 = create_resampled_spline(line2, num_points=20)

    vectors = calculate_vector_difference(resampled1, resampled2, overlap_only=True, num_points=5)
    assert vectors.shape == (5, 2)
    np.testing.assert_allclose(vectors, np.tile([-1.0, 0.0], (5, 1)), atol=1e-3)

    # Control: splines without any overlap give the documented zero fallback
    far = create_resampled_spline(line2 + 100, num_points=20)
    np.testing.assert_array_equal(
        calculate_vector_difference(resampled1, far, overlap_only=True, num_points=5),
        np.zeros((5, 2)),
    )


def test_calculate_vector_difference_small_overlap_and_unequal_lengths():
    """Verify tiny identical overlap, unequal-length inputs, and axis-aligned box clipping.

    Strategy: test (1) identical lines overlapping on a tiny corner interval,
    (2) unequal-length splines with `overlap_only=False` and `overlap_only=True`,
    and (3) axis-aligned opposite-direction segments where `d[:, k] == 0`.
    Expected outcome: identical overlap yields exact zeros, unequal-length
    overlap recovers the analytical offset `[-1, 0, -1]`, and axis-aligned
    segments clip cleanly without division-by-zero or falling back to zeros.
    """
    # 1. Tiny corner overlap on the same diagonal: both clipped segments are identical
    a = np.repeat(np.linspace(0, 10, 50)[:, None], 3, axis=1)
    b = np.repeat(np.linspace(9.6, 20, 50)[:, None], 3, axis=1)
    v = calculate_vector_difference(a, b, overlap_only=True, num_points=100)
    assert v.shape == (100, 3)
    np.testing.assert_allclose(v, 0.0, atol=1e-9)

    # 2. Unequal-length splines with overlap_only=False resamples both to num_points
    c = np.c_[np.linspace(0, 10, 50), np.zeros(50), np.zeros(50)]
    d = np.c_[np.linspace(0, 10, 40), np.zeros(40), np.ones(40)]
    v_full = calculate_vector_difference(c, d, overlap_only=False, num_points=100)
    np.testing.assert_allclose(v_full, np.c_[np.zeros(100), np.zeros(100), np.ones(100)], atol=1e-9)

    # 3. Unequal-length splines with overlap_only=True:
    # u1: (t, t, t) for t in [0, 20] (50 pts); u2: (t, t+1, t) for t in [10, 30] (30 pts).
    # Box intersection is [10, 20] x [11, 20] x [10, 20]:
    # u1 clips to (11,11,11)->(20,20,20), u2 clips to (10,11,10)->(19,20,19), diff = [-1, 0, -1].
    t1 = np.linspace(0.0, 20.0, 50)
    t2 = np.linspace(10.0, 30.0, 30)
    u1 = np.column_stack([t1, t1, t1])
    u2 = np.column_stack([t2, t2 + 1.0, t2])
    v_overlap = calculate_vector_difference(u1, u2, overlap_only=True, num_points=40)
    assert v_overlap.shape == (40, 3)
    np.testing.assert_allclose(v_overlap, np.tile([-1.0, 0.0, -1.0], (40, 1)), atol=1e-6)

    # 4. Axis-aligned segments (d[:, 1] == 0) running in opposite directions on y = 5:
    # horiz_a clips to (3, 5)->(10, 5), horiz_b clips to (10, 5)->(3, 5), diff = (7.. -7, 0).
    horiz_a = np.column_stack([np.linspace(0.0, 10.0, 20), np.full(20, 5.0)])
    horiz_b = np.column_stack([np.linspace(13.0, 3.0, 20), np.full(20, 5.0)])
    v_axis = calculate_vector_difference(horiz_a, horiz_b, overlap_only=True, num_points=20)
    expected_axis = np.column_stack([np.linspace(7.0, -7.0, 20), np.zeros(20)])
    np.testing.assert_allclose(v_axis, expected_axis, atol=1e-9)


def test_calculate_tangent_vectors():
    """Verify exact tangents on a line, zero phase shift on a circle, and O(h^2) convergence on a helix.

    Strategy: compute central-difference tangents on (1) a unit-step line,
    (2) a sampled circle checking orthogonality to the radius vector at interior
    points, and (3) a 3D helix at step sizes `h` and `h/2` to measure the error ratio.
    Expected outcome: line tangents equal `[1, 0, 0]`, circle tangents are
    strictly orthogonal to radii (zero half-step phase shift), and halving `h`
    reduces helix derivative error by ~4x (O(h^2)).
    """
    line = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    tangents = calculate_tangent_vectors(line)
    assert tangents.shape == (3, 3)
    np.testing.assert_allclose(tangents[:, 0], 1.0)
    np.testing.assert_allclose(tangents[:, 1:], 0.0)

    # Zero phase shift: central-difference tangents on a circle are strictly perpendicular to radii
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    circle_tangents = calculate_tangent_vectors(circle)
    radial_dots = np.sum(circle_tangents[1:-1] * circle[1:-1], axis=1)
    np.testing.assert_allclose(radial_dots, 0.0, atol=1e-12)

    # O(h^2) convergence on a 3D helix r(t) = (cos t, sin t, 0.5 t)
    def _helix_max_error(n_pts):
        t = np.linspace(0.0, 2.0 * np.pi, n_pts)
        h = t[1] - t[0]
        helix = np.column_stack([np.cos(t), np.sin(t), 0.5 * t])
        approx_deriv = calculate_tangent_vectors(helix) / h
        exact_deriv = np.column_stack([-np.sin(t), np.cos(t), np.full(n_pts, 0.5)])
        return np.max(np.linalg.norm(approx_deriv[1:-1] - exact_deriv[1:-1], axis=1))

    err_coarse = _helix_max_error(51)
    err_fine = _helix_max_error(101)
    assert 3.8 < (err_coarse / err_fine) < 4.2


def test_project_onto_plane_integer_inputs_and_mixed_zero_tangents():
    """Verify integer-input projection does not truncate and mixed zero/non-zero tangents behave cleanly.

    Strategy: pass integer `vectors` and `tangent_vectors` whose projections are
    non-integer rationals, then pass a batch mixing zero and non-zero tangents.
    Expected outcome: results are floating point with exact analytical values
    and zero dot product with non-zero tangents, while rows with zero tangents
    pass through unchanged without NaNs.
    """
    vectors = np.array([[1, 0, 0], [1, 1, 0], [2, 3, 4]], dtype=int)
    tangents = np.array([[1, 1, 0], [1, 2, 0], [1, 1, 1]], dtype=int)

    projected = project_onto_plane(vectors, tangents)
    expected = np.array([[0.5, -0.5, 0.0], [0.4, -0.2, 0.0], [-1.0, 0.0, 1.0]])
    assert np.issubdtype(projected.dtype, np.floating)
    np.testing.assert_allclose(projected, expected, atol=1e-12)
    np.testing.assert_allclose(np.sum(projected * tangents, axis=1), 0.0, atol=1e-12)

    # Mixed batch with both zero and non-zero tangents
    mixed_v = np.array([[1, 2, 3], [1, 0, 0], [4, 5, 6]], dtype=int)
    mixed_t = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=int)
    mixed_out = project_onto_plane(mixed_v, mixed_t)
    assert not np.any(np.isnan(mixed_out))
    np.testing.assert_allclose(mixed_out, [[1.0, 2.0, 3.0], [0.5, -0.5, 0.0], [4.0, 5.0, 6.0]], atol=1e-12)


def test_normalize_vectors():
    """Verify unit-length normalization and warning-free zero-vector handling.

    Strategy: normalize 2D and 3D batches containing both non-zero and zero rows
    with warnings promoted to errors.
    Expected outcome: non-zero rows scale to exact unit length and zero rows
    stay zero without raising `RuntimeWarning`.
    """
    out_2d = normalize_vectors(np.array([[3, 4], [1, 0]]))
    np.testing.assert_allclose(out_2d, [[0.6, 0.8], [1.0, 0.0]], atol=1e-12)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out_3d = normalize_vectors(np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
    np.testing.assert_allclose(out_3d, [[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]], atol=1e-12)


def test_calculate_spline_length_and_resolution_forms():
    """Verify 2D/3D arc length across resolution formats, single-point curves, and validation errors.

    Strategy: compute arc lengths on analytical 2D and 3D lines with `None`,
    dict, list, tuple, and NumPy array resolutions (including `None` entries in
    dicts and sequences), single-point curves, and invalid coordinate dimensions.
    Expected outcome: scaled Euclidean lengths match exact analytical formulas,
    missing/`None` axis resolutions warn and default to 1.0 (a missing Z with
    known Y and X takes the finest of them, D01; units convert to µm), single-point curves
    have length 0.0, and 0D/1D/4D inputs or wrong-length resolutions raise `ValueError`.
    """
    coords_2d = [np.array([0, 3]), np.array([0, 4])]  # [Y, X] -> length 5
    coords_3d = [np.array([0, 2]), np.array([0, 3]), np.array([0, 6])]  # [Z, Y, X] -> length 7
    assert calculate_spline_length(coords_2d) == pytest.approx(5.0)
    assert calculate_spline_length(coords_3d) == pytest.approx(7.0)

    # List, dict, ndarray, and 3-element tuple for 2D
    assert calculate_spline_length(coords_3d, resolution=[2.0, 1.0, 0.5]) == pytest.approx(np.sqrt(34.0))
    assert calculate_spline_length(coords_2d, resolution={"Y": 2.0, "X": 0.5}) == pytest.approx(np.sqrt(40.0))

    line_z = [np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)]
    line_y2d = [np.linspace(0, 10, 11), np.zeros(11)]
    assert calculate_spline_length(line_z, resolution=np.array([2.0, 1.0, 1.0])) == pytest.approx(20.0)
    assert calculate_spline_length(line_y2d, resolution=[0.5, 0.5]) == pytest.approx(5.0)
    assert calculate_spline_length(line_y2d, resolution=(2.0, 0.5, 0.5)) == pytest.approx(5.0)

    # Single-point curves have zero length
    assert calculate_spline_length([np.array([5.0]), np.array([3.0])]) == pytest.approx(0.0)
    assert calculate_spline_length([np.array([1.0]), np.array([2.0]), np.array([3.0])]) == pytest.approx(0.0)

    # None axis in dict or sequence warns and falls back to 1.0
    y_line = [np.zeros(11), np.linspace(0, 10, 11), np.zeros(11)]
    with pytest.warns(UserWarning, match="axis Z"):
        length_dict_none = calculate_spline_length(
            y_line, resolution={"X": 0.5, "Y": 0.5, "Z": None, "unit": "micron"}
        )
    assert length_dict_none == pytest.approx(5.0)

    # D01: a missing Z with known Y/X takes the finest of them (0.5), not 1.0
    with pytest.warns(UserWarning, match="axis Z"):
        length_seq_none = calculate_spline_length(line_z, resolution=[None, 0.5, 0.5])
    assert length_seq_none == pytest.approx(5.0)
    with pytest.warns(UserWarning, match="axis Z"):
        assert calculate_spline_length(line_z, resolution=[None, 0.5, None]) == pytest.approx(10.0)

    # D01: units are converted to µm; non-positive sizes raise
    assert calculate_spline_length(line_z, resolution={"Z": 2000.0, "Y": 1, "X": 1, "unit": "nm"}) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="finite and positive"):
        calculate_spline_length(line_z, resolution=[0.0, 1.0, 1.0])

    # Validation errors: wrong resolution length and unsupported coordinate dimensions (0D, 1D, 4D)
    with pytest.raises(ValueError, match="entries"):
        calculate_spline_length(line_y2d, resolution=[1.0, 2.0, 3.0, 4.0])

    for bad_coords in ([], [np.array([0, 1])], [np.array([0, 1])] * 4):
        with pytest.raises(ValueError, match=r"coords must be a list of 2 or 3 arrays"):
            calculate_spline_length(bad_coords)


def test_project_onto_plane_orthogonality():
    """Verify orthogonality invariant `<P(v), t> = 0` of `project_onto_plane`.

    Strategy: project 100 seeded random 3D vectors onto planes normal to random
    tangent vectors and compute row-wise dot products.
    Expected outcome: all dot products are zero within floating-point precision.
    """
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((100, 3)) * 10
    tangents = rng.standard_normal((100, 3)) * 5

    projected = project_onto_plane(vectors, tangents)
    dot_products = np.sum(projected * tangents, axis=1)
    assert np.allclose(dot_products, 0.0, atol=1e-12), "Projected vectors are not strictly orthogonal to tangents"


def test_project_onto_plane_idempotence():
    """Verify idempotence invariant `P(P(v)) = P(v)` of `project_onto_plane`.

    Strategy: apply `project_onto_plane` twice with the same tangents on seeded
    random 3D vectors.
    Expected outcome: second projection matches the first within `1e-14`.
    """
    rng = np.random.default_rng(43)
    vectors = rng.standard_normal((100, 3))
    tangents = rng.standard_normal((100, 3))

    projected_once = project_onto_plane(vectors, tangents)
    projected_twice = project_onto_plane(projected_once, tangents)
    assert np.allclose(projected_once, projected_twice, atol=1e-14), "Projection operator is not idempotent"


def test_project_onto_plane_null_space():
    """Verify null space invariant `P(a * t) = 0` of `project_onto_plane`.

    Strategy: project scalar multiples of tangent vectors onto their own normal planes.
    Expected outcome: every projected vector vanishes within `1e-14`.
    """
    rng = np.random.default_rng(44)
    tangents = rng.standard_normal((100, 3))
    vectors = tangents * rng.standard_normal((100, 1))

    projected = project_onto_plane(vectors, tangents)
    assert np.allclose(projected, 0.0, atol=1e-14), "Vectors purely in the tangent space were not fully rejected"


def test_project_onto_plane_pythagorean():
    """Verify Pythagorean norm decomposition `||v||^2 = ||P(v)||^2 + ||v - P(v)||^2`.

    Strategy: project seeded random vectors and compare total squared norm
    against the sum of orthogonal and parallel component squared norms.
    Expected outcome: squared norm is conserved across all rows within `1e-12`.
    """
    rng = np.random.default_rng(45)
    vectors = rng.standard_normal((100, 3))
    tangents = rng.standard_normal((100, 3))

    projected = project_onto_plane(vectors, tangents)
    parallel_component = vectors - projected

    norm_sq_v = np.sum(vectors**2, axis=1)
    norm_sq_proj = np.sum(projected**2, axis=1)
    norm_sq_parallel = np.sum(parallel_component**2, axis=1)
    assert np.allclose(norm_sq_v, norm_sq_proj + norm_sq_parallel, atol=1e-12)


def test_project_onto_plane_degenerate():
    """Verify zero-length tangents pass input vectors through unchanged without NaNs.

    Strategy: pass exact zero tangent vectors alongside arbitrary 3D vectors.
    Expected outcome: output contains no NaNs and equals the input vectors.
    """
    vectors = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tangents = np.zeros_like(vectors)

    projected = project_onto_plane(vectors, tangents)
    assert not np.any(np.isnan(projected)), "NaN generated for zero-length tangent"
    assert np.allclose(projected, vectors), "Vectors should pass through unchanged if tangent is zero-length"


def test_calculate_spline_length_analytical_circle():
    """Verify analytical convergence of 2D arc length on a densely sampled circle.

    Strategy: sample a radius-10 circle at 10,000 points and compute arc length.
    Expected outcome: result matches `2 * pi * 10` within `rtol=1e-5`.
    """
    theta = np.linspace(0, 2 * np.pi, 10000)
    radius = 10.0
    coords = [radius * np.sin(theta), radius * np.cos(theta)]
    length = calculate_spline_length(coords)
    assert np.isclose(length, 2 * np.pi * radius, rtol=1e-5)


def test_calculate_spline_length_isometric_invariance():
    """Verify 3D arc length is invariant under rigid translation and rotation.

    Strategy: compute arc length of a 3D helix, then translate and rotate it
    around the Z axis by pi/4 and recompute.
    Expected outcome: translated and rotated arc lengths match within `rtol=1e-12`.
    """
    t = np.linspace(0, 10, 500)
    z = np.sin(t)
    y = np.cos(t)
    x = t * 2.0

    len_original = calculate_spline_length([z, y, x])

    len_translated = calculate_spline_length([z + 100, y - 50, x + 3.14])
    assert np.isclose(len_original, len_translated, rtol=1e-12)

    theta = np.pi / 4
    y_rot = y * np.cos(theta) - x * np.sin(theta)
    x_rot = y * np.sin(theta) + x * np.cos(theta)
    len_rotated = calculate_spline_length([z, y_rot, x_rot])
    assert np.isclose(len_original, len_rotated, rtol=1e-12)


def test_calculate_spline_length_resolution_scaling():
    """Verify anisotropic resolution mapping equals pre-scaling coordinates by pixel sizes.

    Strategy: compare `calculate_spline_length([z, y, x], resolution=res)`
    against `calculate_spline_length([z * sz, y * sy, x * sx])`.
    Expected outcome: both values agree within `rtol=1e-12`.
    """
    z = np.linspace(0, 5, 100)
    y = np.linspace(0, 10, 100)
    x = np.linspace(0, 15, 100)
    res = {"Z": 2.0, "Y": 0.5, "X": 1.0}

    len_mapped = calculate_spline_length([z, y, x], resolution=res)
    len_prescaled = calculate_spline_length([z * 2.0, y * 0.5, x * 1.0])
    assert np.isclose(len_mapped, len_prescaled, rtol=1e-12)


@pytest.mark.parametrize("make_image", [create_3d_image_from_spline, create_nd_image_from_spline])
def test_spline_image_skips_negative_coordinates(make_image):
    """Verify negative spline coordinates are skipped rather than wrapping around.

    Strategy: rasterize a spline descending from `z=2` to `z=-20` into a
    `(40, 40, 40)` grid using both 3D and nD rasterizers.
    Expected outcome: only `z in {0, 1, 2}` contain marked voxels.
    """
    pts = np.c_[np.linspace(2, -20, 6), 5.0 + np.arange(6) * 0.1, np.full(6, 5.0)]
    img = make_image(fit_cubic_spline(pts), (40, 40, 40))
    assert sorted(set(np.nonzero(img)[0].tolist())) == [0, 1, 2]


def test_spline_image_matches_point_loop():
    """Verify vectorized rasterization matches per-point integer flooring on in-bounds splines.

    Strategy: compare `create_3d_image_from_spline` against an explicit loop
    flooring each evaluated point into a `(60, 60, 60)` grid.
    Expected outcome: both grids are bit-for-bit identical.
    """
    pts = generate_random_3d_coordinates(8, seed=3) * 0.5 + 5
    tck = fit_cubic_spline(pts)
    img = create_3d_image_from_spline(tck, (60, 60, 60), num_points=500)

    ref = np.zeros((60, 60, 60), np.uint8)
    for p in np.array(splev(np.linspace(0, 1, 500), tck)).T:
        ref[tuple(int(c) for c in p)] = 1
    np.testing.assert_array_equal(img, ref)


def test_plot_3d_spline_returns_figure():
    """Verify 3D spline plotting creates/reuses axes, labels Z/Y/X, and plots exact spline and scatter data.

    Strategy: call `plot_3d_spline` first with `ax=None` and then with an
    existing 3D `Axes`, inspecting axis labels, plotted line coordinates, and
    scatter point coordinates.
    Expected outcome: returns a `Figure` (reusing `existing_ax.figure` when
    provided), labels axes `("Z", "Y", "X")`, and draws the exact spline curve
    and initial control points.
    """
    pts = generate_random_3d_coordinates(8, seed=1)
    tck = fit_cubic_spline(pts)

    fig = plot_3d_spline(pts, tck, num_points=100)
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("Z", "Y", "X")
    assert len(ax.lines) == 1
    lx, ly, lz = ax.lines[0].get_data_3d()
    expected_curve = np.column_stack(splev(np.linspace(0, 1, 100), tck))
    np.testing.assert_allclose(np.column_stack([lx, ly, lz]), expected_curve)
    assert len(ax.collections) == 1
    np.testing.assert_allclose(np.asarray(ax.collections[0]._offsets3d).T, pts)

    # Existing axes reuse
    existing_fig = plt.figure()
    existing_ax = existing_fig.add_subplot(111, projection="3d")
    returned_fig = plot_3d_spline(pts, tck, num_points=50, ax=existing_ax)
    assert returned_fig is existing_fig
    assert len(existing_ax.lines) == 1
    assert len(existing_ax.lines[0].get_data_3d()[0]) == 50
    assert len(existing_ax.collections) == 1
