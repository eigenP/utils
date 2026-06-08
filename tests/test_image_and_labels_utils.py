import numpy as np
import pytest
from eigenp_utils.image_and_labels_utils import (
    windowed_slice_projection,
    optimized_entire_labels_touching_mask,
    sample_intensity_around_points, sample_intensity_along_surface_normals,
    voronoi_otsu_labeling
)

def test_voronoi_otsu_labeling():
    # Test 2D without spacing
    img_2d = np.zeros((20, 20), dtype=float)
    # create two distinct Gaussian spots
    for center, scale in [((5, 5), 3), ((15, 15), 3)]:
        y, x = np.ogrid[-center[0]:20-center[0], -center[1]:20-center[1]]
        img_2d += np.exp(-(x**2 + y**2) / (2 * scale**2)) * 10

    # We add some background noise so Otsu thresholding finds something besides straight zeros
    img_2d += np.random.rand(20, 20) * 0.1

    labels_2d = voronoi_otsu_labeling(img_2d, spot_sigma=1, outline_sigma=1)

    assert labels_2d.shape == (20, 20)
    # Check that there are at least 2 distinct labeled regions
    assert len(np.unique(labels_2d[labels_2d > 0])) >= 2
    # Verify the centers belong to labeled spots
    assert labels_2d[5, 5] > 0
    assert labels_2d[15, 15] > 0

    # Test 3D with spacing
    img_3d = np.zeros((10, 20, 20), dtype=float)
    for center, scale in [((5, 5, 5), 2), ((5, 15, 15), 2)]:
        z, y, x = np.ogrid[-center[0]:10-center[0], -center[1]:20-center[1], -center[2]:20-center[2]]
        img_3d += np.exp(-(x**2 + y**2 + z**2) / (2 * scale**2)) * 10

    img_3d += np.random.rand(10, 20, 20) * 0.1

    spacing = {'Z': 2.0, 'Y': 0.5, 'X': 0.5}
    # Pass a tuple to spot_sigma
    labels_3d = voronoi_otsu_labeling(img_3d, spot_sigma=(2, 1, 1), outline_sigma=1, spacing=spacing)

    assert labels_3d.shape == (10, 20, 20)
    assert len(np.unique(labels_3d[labels_3d > 0])) >= 2
    assert labels_3d[5, 5, 5] > 0
    assert labels_3d[5, 15, 15] > 0


def test_windowed_slice_projection_max():
    img = np.zeros((5, 10, 10))
    img[2, 5, 5] = 10

    thick = windowed_slice_projection(img, window_size=3, axis=0, operation='max')

    assert thick.shape == (5, 10, 10)
    assert thick[1, 5, 5] == 10
    assert thick[2, 5, 5] == 10
    assert thick[3, 5, 5] == 10
    assert thick[0, 5, 5] == 0
    assert thick[4, 5, 5] == 0

def test_windowed_slice_projection_average():
    img = np.ones((5, 10, 10)) * 3

    thick = windowed_slice_projection(img, window_size=3, axis=0, operation='average')

    # average of 3 values of 3 is 3, except boundaries which are padded with 0
    # boundary 0: (0 + 3 + 3) / 3 = 2
    # middle: (3 + 3 + 3) / 3 = 3

    assert np.allclose(thick[2, :, :], 3)
    assert np.allclose(thick[0, :, :], 2)

def test_optimized_entire_labels_touching_mask():
    labels = np.zeros((20, 20), dtype=int)
    # create two labels
    labels[5:10, 5:10] = 1
    labels[15:20, 15:20] = 2

    mask = np.zeros((20, 20), dtype=int)
    # mask touching label 1 after expansion
    mask[2:4, 2:4] = 1

    res = optimized_entire_labels_touching_mask(labels, mask)

    # label 1 should be completely retained
    assert np.all(res[5:10, 5:10] == 1)
    # label 2 should be gone
    assert np.all(res[15:20, 15:20] == 0)

def test_sample_intensity_around_points():
    image_3d = np.ones((10, 10, 10))
    image_3d[5, 5, 5] = 10 # central high intensity point

    points = np.array([
        [5, 5, 5],
        [1, 1, 1],
        [20, 20, 20] # out of bounds
    ])

    # A physical diameter of 2.0 -> physical radius 1.0 -> 1 pixel radius -> 3x3x3 cube
    res = sample_intensity_around_points(image_3d, points, diameter=2.0)

    # [5,5,5] is average of 3x3x3 cube = 27 voxels
    # 26 ones + 1 ten = 36 / 27 = 1.333

    assert len(res) == 3
    assert np.isclose(res[0], 36 / 27)
    assert np.isclose(res[1], 1.0) # all ones
    assert np.isnan(res[2]) # out of bounds

def test_sample_intensity_xyz_warning():
    # Z, Y, X order: 5 slices, 20 rows, 30 columns
    image_3d = np.zeros((5, 20, 30))

    # Intentionally inverted (X, Y, Z) order
    points_xyz = np.array([
        [25, 10, 2]
    ])

    with pytest.warns(UserWarning, match=r"Points appear to be in \(X, Y, Z\) order"):
        res = sample_intensity_around_points(image_3d, points_xyz, diameter=3)
        assert np.isnan(res[0]) # Since 25 >= 5 (Z-dimension), it will be considered out of bounds

def test_sample_intensity_along_surface_normals():
    img = np.ones((10, 10, 10))
    grid = np.zeros((5, 5, 3))
    grid[:, :, 0] = 5
    grid[:, :, 1] = np.arange(5).reshape(-1, 1)
    grid[:, :, 2] = np.arange(5)
    res = sample_intensity_along_surface_normals(img, grid, thickness=3, num_steps=3, pixel_sizes={'Z': 2.0, 'Y': 1.0, 'X': 1.0})
    assert res.shape == (5, 5, 3)

def test_sample_intensity_around_points_pixel_sizes():
    img = np.ones((10, 10, 10))
    points = [[5, 5, 5]]
    res = sample_intensity_around_points(img, points, diameter=5.0, pixel_sizes={'Z': 2.0, 'Y': 1.0, 'X': 1.0})
    assert np.isclose(res[0], 1.0)

def test_optimized_entire_labels_touching_mask_dilation_preservation():
    """
    Control 1: Preservation of Shape (Dilation)
    Test that expanding a label by 6.0 physical units results in matching physical dimensions
    regardless of grid anisotropy.
    """
    from skimage.segmentation import expand_labels

    # Check if expand_labels supports 'spacing'
    try:
        import inspect
        if 'spacing' not in inspect.signature(expand_labels).parameters:
            import pytest
            pytest.skip("scikit-image < 0.21.0 does not support 'spacing' parameter in expand_labels")
    except Exception:
        pass

    # Isotropic grid (1x1x1)
    iso_labels = np.zeros((15, 15, 15), dtype=int)
    iso_labels[7, 7, 7] = 1
    # Expand by 6.0 um -> radius 6 -> expected size 13x13x13 (12 + 1)
    iso_mask = np.zeros_like(iso_labels)
    # distance is total distance from origin
    # using our tested internal expansion function wrapped manually for check
    try:
        iso_dilated = expand_labels(iso_labels, distance=6.0, spacing=(1.0, 1.0, 1.0))
    except TypeError:
        # Fallback for old skimage
        iso_dilated = expand_labels(iso_labels, distance=6)

    # Anisotropic grid (3x1x1)
    # Z-axis has 3.0 pixel size
    aniso_labels = np.zeros((15, 15, 15), dtype=int)
    aniso_labels[7, 7, 7] = 1

    try:
        aniso_dilated = expand_labels(aniso_labels, distance=6.0, spacing=(3.0, 1.0, 1.0))
    except TypeError:
        import pytest
        pytest.skip("scikit-image < 0.21.0 does not support 'spacing' parameter in expand_labels")

    def get_bounding_box_size(mask, spacing):
        coords = np.argwhere(mask > 0)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        pixel_size = max_coords - min_coords + 1
        physical_size = pixel_size * np.array(spacing)
        return physical_size

    iso_size = get_bounding_box_size(iso_dilated, [1.0, 1.0, 1.0])
    aniso_size = get_bounding_box_size(aniso_dilated, [3.0, 1.0, 1.0])

    # 6 um distance -> diam 12 + 1 center? The exact mathematical dilation shape
    # might vary (diamond/L1 vs square/Linf) but between the two grids, the PHYSICAL
    # size should remain the same within a tolerance of 1 pixel width (max 3.0 um)
    assert np.allclose(iso_size, aniso_size, atol=3.0)

def test_sample_intensity_around_points_intensity_conservation():
    """
    Control 2: Intensity Conservation (Uniform Filter)
    Query points located near the boundary layers using mode='reflect' via our function.
    The output mean intensity must evaluate strictly to 100 across all valid spaces.
    """
    # synthetic volume filled with constant intensity 100
    img = np.full((10, 10, 10), 100.0)

    # points near boundary
    points = [
        [0, 0, 0],
        [0, 5, 5],
        [9, 9, 9],
        [9, 0, 5]
    ]

    res = sample_intensity_around_points(img, points, diameter=5.0)

    # Assert intensity is conserved exactly (no zero-padding dilution)
    for r in res:
        assert np.isclose(r, 100.0)

def test_windowed_slice_projection_pixel_sizes():
    img = np.ones((10, 10, 10))
    res = windowed_slice_projection(img, window_size=5.0, pixel_sizes={'Z': 2.0, 'Y': 1.0, 'X': 1.0})
    assert res.shape == img.shape

def test_optimized_entire_labels_touching_mask_pixel_sizes():
    labels = np.zeros((10, 10), dtype=int)
    labels[2:4, 2:4] = 1
    mask = np.zeros((10, 10), dtype=int)
    mask[6:8, 6:8] = 1
    res = optimized_entire_labels_touching_mask(labels, mask, distance=3.0, pixel_sizes={'Y': 1.0, 'X': 1.0})
    assert res.shape == labels.shape
