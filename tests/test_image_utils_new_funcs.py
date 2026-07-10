import numpy as np
import pytest
from eigenp_utils.image_and_labels_utils import _ensure_pixel_size_array, fit_plane_ransac, generate_plane_basis, sample_volume_plane

def test_ensure_pixel_size_array():
    # Test None
    with pytest.warns(UserWarning):
        res = _ensure_pixel_size_array(None)
    assert np.allclose(res, [1.0, 1.0, 1.0])

    # Test dict
    res = _ensure_pixel_size_array({'Z': 2.0, 'Y': 1.5, 'X': 0.5})
    assert np.allclose(res, [2.0, 1.5, 0.5])

    # Test missing keys in dict fallback to 1.0
    res = _ensure_pixel_size_array({'Z': 2.0, 'X': 0.5})
    assert np.allclose(res, [2.0, 1.0, 0.5])

    # Test list
    res = _ensure_pixel_size_array([2.0, 1.5, 0.5])
    assert np.allclose(res, [2.0, 1.5, 0.5])

def test_generate_plane_basis():
    normal = np.array([1.0, 0.0, 0.0])
    u, v = generate_plane_basis(normal)
    # Check orthogonality
    assert np.isclose(np.dot(u, normal), 0.0)
    assert np.isclose(np.dot(v, normal), 0.0)
    assert np.isclose(np.dot(u, v), 0.0)
    # Check unit length
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.isclose(np.linalg.norm(v), 1.0)

def test_fit_plane_ransac():
    # Points on the plane Z = 2.0
    points_zyx = np.array([
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
        [5.0, 5.0, 5.0] # Outlier
    ])

    p0, normal = fit_plane_ransac(points_zyx, pixel_sizes={'Z': 1.0, 'Y': 1.0, 'X': 1.0}, inlier_threshold_um=0.1)

    assert np.isclose(p0[0], 2.0)
    assert np.isclose(np.abs(normal[0]), 1.0)
    assert np.isclose(normal[1], 0.0)
    assert np.isclose(normal[2], 0.0)

def test_sample_volume_plane():
    volume = np.zeros((10, 10, 10))
    volume[5, :, :] = 1.0

    p0 = np.array([5.0, 5.0, 5.0])
    normal = np.array([1.0, 0.0, 0.0]) # plane Z=5

    sampled, spacing = sample_volume_plane(
        volume, pixel_sizes={'Z': 1.0, 'Y': 1.0, 'X': 1.0},
        p0_phys=p0, normal_phys=normal,
        u_range_um=(-2, 2), v_range_um=(-2, 2),
        u_res=5, v_res=5
    )

    # Should all be 1s since we are sampling the plane at Z=5
    assert np.allclose(sampled, 1.0)
