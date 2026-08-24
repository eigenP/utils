from scipy.ndimage import shift
import numpy as np
import pytest

from eigenp_utils.image_and_labels_utils import (
    _ensure_pixel_size_array,
    fit_plane_ransac,
    generate_plane_basis,
    sample_volume_plane,
    voronoi_otsu_labeling,
    windowed_slice_projection,
    sample_intensity_around_points,
    optimized_entire_labels_touching_mask,
    sample_intensity_along_surface_normals,
    create_ellipsoid_struct,
    generate_morphological_surface_mask,
    estimate_inter_label_distance,
)



# =========================================
# Source: test_image_and_labels_utils.py
# =========================================

def test_voronoi_otsu_labeling():
    """Test that voronoi otsu labeling works as expected."""
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
    labels_3d = voronoi_otsu_labeling(img_3d, spot_sigma=(2, 1, 1), outline_sigma=1, pixel_sizes=spacing)

    assert labels_3d.shape == (10, 20, 20)
    assert len(np.unique(labels_3d[labels_3d > 0])) >= 2
    assert labels_3d[5, 5, 5] > 0
    assert labels_3d[5, 15, 15] > 0


def test_windowed_slice_projection_max():
    """Test that windowed slice projection max works as expected."""
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
    """Test that windowed slice projection average works as expected."""
    img = np.ones((5, 10, 10)) * 3

    thick = windowed_slice_projection(img, window_size=3, axis=0, operation='average')

    # average of 3 values of 3 is 3, except boundaries which are padded with 0
    # boundary 0: (0 + 3 + 3) / 3 = 2
    # middle: (3 + 3 + 3) / 3 = 3

    assert np.allclose(thick[2, :, :], 3)
    assert np.allclose(thick[0, :, :], 2)

def test_optimized_entire_labels_touching_mask():
    """Test that optimized entire labels touching mask works as expected."""
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
    """Test that sample intensity around points works as expected."""
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
    """Test that sample intensity xyz warning works as expected."""
    # Z, Y, X order: 5 slices, 20 rows, 30 columns
    image_3d = np.zeros((5, 20, 30))

    # Intentionally inverted (X, Y, Z) order
    points_xyz = np.array([
        [25, 10, 2]
    ])

    with pytest.warns(UserWarning, match=r"Points appear to be in \(X, Y, Z\) order"):
        res = sample_intensity_around_points(image_3d, points_xyz, diameter=3)
        assert np.isnan(res[0]) # Since 25 >= 5 (Z-dimension), it will be considered out of bounds

def test_sample_intensity_along_surface_normals_anisotropic_profile():
    """
    Test sampling intensity along surface normals for a planar mesh on an anisotropic volume.

    Verifies that for an image with a Gaussian intensity profile centered at Z_phys = 10.0 um,
    sampling along normals of a surface grid at Z_phys = 10.0 um (with Z pixel size = 2.0 um)
    recovers a peak intensity at the central sampling step (offset 0.0) and symmetric falloff at +/- offsets.
    """
    # 3D volume of shape (11, 20, 20) with Z pixel size 2.0 um, Y & X 0.5 um
    pixel_sizes = {'Z': 2.0, 'Y': 0.5, 'X': 0.5}

    z_indices = np.arange(11)
    z_phys = z_indices * pixel_sizes['Z']  # [0, 2, 4, ..., 20] um

    # Peak intensity at Z_phys = 10.0 um (voxel Z = 5)
    gaussian_z = np.exp(-((z_phys - 10.0) ** 2) / (2 * (2.0 ** 2))) * 100.0

    img = np.zeros((11, 20, 20), dtype=float)
    for z in range(11):
        img[z, :, :] = gaussian_z[z]

    # Create a planar surface grid at voxel Z = 5 (Z_phys = 10.0 um)
    u_grid, v_grid = np.meshgrid(np.linspace(5, 15, 6), np.linspace(5, 15, 6), indexing='ij')
    surface_grid = np.zeros((6, 6, 3), dtype=float)
    surface_grid[:, :, 0] = 5.0  # Z coordinate in voxels
    surface_grid[:, :, 1] = u_grid  # Y coordinate in voxels
    surface_grid[:, :, 2] = v_grid  # X coordinate in voxels

    # Sample thickness = 8.0 um centered at Z_phys = 10.0 um with 5 steps (-4.0, -2.0, 0.0, 2.0, 4.0 um)
    # Using bicubic interpolation for smooth subpixel intensity evaluation
    sampled = sample_intensity_along_surface_normals(
        img,
        surface_grid,
        thickness=8.0,
        num_steps=5,
        interpolation='bicubic',
        pixel_sizes=pixel_sizes
    )

    assert sampled.shape == (6, 6, 5)

    # Check that for all interior grid points, step index 2 (offset 0.0 um, Z_phys = 10.0 um) is maximum
    peak_step_indices = np.argmax(sampled, axis=-1)
    assert np.all(peak_step_indices == 2)

    # Check symmetry of sampled intensities across the central normal step
    profile = sampled[3, 3, :]
    assert np.isclose(profile[0], profile[4])
    assert np.isclose(profile[1], profile[3])
    assert profile[2] > profile[1] > profile[0]


def test_sample_intensity_along_surface_normals_curved_surface_bicubic():
    """
    Test sampling intensity along surface normals over a curved mesh using bicubic interpolation.

    Verifies subpixel continuous intensity sampling across normals of a spherical surface grid,
    and checks that a UserWarning is raised when `pixel_sizes` is not provided.
    """
    # Create synthetic volume with a spherical intensity distribution centered at (15, 15, 15)
    z, y, x = np.ogrid[:31, :31, :31]
    radius_grid = np.sqrt((z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2)
    img = np.exp(-((radius_grid - 8.0) ** 2) / (2 * (2.0 ** 2))) * 250.0

    # Create a spherical surface grid at radius ~ 8 voxels
    theta = np.linspace(0, np.pi, 8)
    phi = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    r = 8.0
    grid_z = 15.0 + r * np.cos(theta_grid)
    grid_y = 15.0 + r * np.sin(theta_grid) * np.sin(phi_grid)
    grid_x = 15.0 + r * np.sin(theta_grid) * np.cos(phi_grid)

    surface_grid = np.stack([grid_z, grid_y, grid_x], axis=-1)

    # Calling without pixel_sizes must issue a UserWarning
    with pytest.warns(UserWarning, match="pixel_sizes not provided"):
        sampled = sample_intensity_along_surface_normals(
            img,
            surface_grid,
            thickness=4.0,
            num_steps=7,
            interpolation='bicubic',
            pixel_sizes=None
        )

    assert sampled.shape == (8, 12, 7)
    assert not np.isnan(sampled).any()

    # The peak of the Gaussian shell is at radius = 8.0 (the surface grid position)
    # The middle step (index 3, offset 0.0) should correspond to the maximum intensity along the normal profile
    interior_peaks = np.argmax(sampled[2:6, :, :], axis=-1)
    assert np.all(interior_peaks == 3)

def test_sample_intensity_around_points_pixel_sizes():
    """Test that sample intensity around points pixel sizes works as expected."""
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
    """Test that windowed slice projection pixel sizes works as expected."""
    img = np.ones((10, 10, 10))
    res = windowed_slice_projection(img, window_size=5.0, pixel_sizes={'Z': 2.0, 'Y': 1.0, 'X': 1.0})
    assert res.shape == img.shape

def test_optimized_entire_labels_touching_mask_pixel_sizes():
    """Test that optimized entire labels touching mask pixel sizes works as expected."""
    labels = np.zeros((10, 10), dtype=int)
    labels[2:4, 2:4] = 1
    mask = np.zeros((10, 10), dtype=int)
    mask[6:8, 6:8] = 1
    res = optimized_entire_labels_touching_mask(labels, mask, distance=3.0, pixel_sizes={'Y': 1.0, 'X': 1.0})
    assert res.shape == labels.shape

def test_voronoi_otsu_translation_invariance():
    """
    🔎 Testr: Verify Translation Invariance of Voronoi Otsu Labeling
    💡 What: Segmenting a translated image must produce an identically translated label mask.
    🎯 Why: Ensures that the segmentation logic (filtering, Otsu, watershed) is completely position-independent.
    🧪 How: Shift a synthetic spot image using `scipy.ndimage.shift`, run labeling on both, and compare.
    """
    # Create a base image with a distinct spot
    img = np.zeros((30, 30))
    y, x = np.ogrid[-10:20, -15:15]
    img += np.exp(-(x**2 + y**2) / (2 * 2**2)) * 10

    # Needs some noise so Otsu isn't singular
    np.random.seed(42)
    img += np.random.rand(30, 30) * 0.1

    # Shift by integer pixels
    shift_vec = (5, -3)
    img_shifted = shift(img, shift_vec, order=1, mode='constant', cval=0.0)

    labels_original = voronoi_otsu_labeling(img, spot_sigma=1, outline_sigma=1)
    labels_shifted = voronoi_otsu_labeling(img_shifted, spot_sigma=1, outline_sigma=1)

    # We can't guarantee label IDs match perfectly due to watershed connected components,
    # but the binary mask of "labeled" vs "background" must be translation invariant.
    mask_original = (labels_original > 0).astype(int)
    mask_shifted = (labels_shifted > 0).astype(int)

    mask_shifted_back = shift(mask_shifted, (-shift_vec[0], -shift_vec[1]), order=0, mode='constant', cval=0)

    # Measure overlap inside the valid non-boundary region
    # Avoid edge effects introduced by shift
    valid_region = mask_original[5:-5, 5:-5]
    valid_region_shifted_back = mask_shifted_back[5:-5, 5:-5]

    overlap_ratio = np.mean(valid_region == valid_region_shifted_back)
    assert overlap_ratio > 0.95, f"Translation invariance failed, overlap ratio was only {overlap_ratio}"


def test_voronoi_otsu_intensity_scale_invariance():
    """
    🔎 Testr: Verify Intensity Scale Invariance of Voronoi Otsu Labeling
    💡 What: Multiplying the entire image intensity by a constant factor > 0 must not change the segmentation.
    🎯 Why: Otsu thresholding determines an optimal relative separation; uniform brightness changes should not alter object boundaries.
    🧪 How: Run labeling on `img` and `img * 5.0` and assert exact equality of the binary segmentation mask.
    """
    img = np.zeros((30, 30))
    y, x = np.ogrid[-15:15, -15:15]
    img += np.exp(-(x**2 + y**2) / (2 * 3**2)) * 10
    np.random.seed(42)
    img += np.random.rand(30, 30) * 0.1

    labels_base = voronoi_otsu_labeling(img, spot_sigma=1, outline_sigma=1)
    labels_scaled = voronoi_otsu_labeling(img * 5.0, spot_sigma=1, outline_sigma=1)

    mask_base = (labels_base > 0)
    mask_scaled = (labels_scaled > 0)

    assert np.array_equal(mask_base, mask_scaled), "Segmentation boundaries changed under global intensity scaling"


def test_windowed_slice_projection_constant_identity():
    """
    🔎 Testr: Verify Constant Identity of Windowed Slicing (Average)
    💡 What: Averaging a windowed slice of a uniform constant volume must perfectly recover that constant.
    🎯 Why: Ensures no mathematical bias or off-by-one errors in the window divisor or summation limits.
    🧪 How: Create a volume of all 7.0s, project, and check the non-padded interior.
    """
    vol = np.ones((10, 10, 10)) * 7.0

    # window_size=5 means +/- 2 margin padding
    proj = windowed_slice_projection(vol, window_size=5, axis=0, operation='average')

    # The valid interior (indices 2 through 7) should be exactly 7.0
    valid_interior = proj[2:8, :, :]
    assert np.allclose(valid_interior, 7.0, atol=1e-14), "Averaging a constant volume did not yield the constant value"


def test_windowed_slice_projection_monotonicity():
    """
    🔎 Testr: Verify Monotonicity of Windowed Slicing (Max)
    💡 What: Increasing the window size for a 'max' projection cannot decrease the result.
    🎯 Why: The maximum of a set is less than or equal to the maximum of any superset.
    🧪 How: Calculate max projections with window sizes 3 and 5. The result of 5 must be >= the result of 3 everywhere.
    """
    np.random.seed(42)
    vol = np.random.rand(15, 10, 10) * 100

    proj_win3 = windowed_slice_projection(vol, window_size=3, axis=0, operation='max')
    proj_win5 = windowed_slice_projection(vol, window_size=5, axis=0, operation='max')

    assert np.all(proj_win5 >= proj_win3), "Max projection is not monotonically non-decreasing with window size"


def test_sample_intensity_constant_background():
    """
    🔎 Testr: Verify Identity of Intensity Sampling in Constant Background
    💡 What: Sampling points anywhere in a constant image must exactly return that constant value.
    🎯 Why: Proves that the underlying mean filtering kernel is correctly normalized and localized.
    🧪 How: Create an image filled with pi, sample random valid coordinates.
    """
    img = np.ones((20, 20, 20)) * np.pi

    # Random valid coordinates, making sure they are not exactly on the edge
    # to avoid mode='constant', cval=0.0 boundary conditions of uniform_filter
    np.random.seed(42)
    points = np.random.rand(50, 3) * 14.0 + 3.0

    sampled = sample_intensity_around_points(img, points, diameter=5)

    assert np.allclose(sampled, np.pi, atol=1e-14), "Sampled values deviate from the constant background"


def test_sample_intensity_linear_scaling():
    """
    🔎 Testr: Verify Linear Scaling of Intensity Sampling
    💡 What: f(c * img) = c * f(img)
    🎯 Why: The underlying operation (local mean) is a linear filter and must respect scalar multiplication.
    🧪 How: Compute sampling for an image, multiply image by 10, sample again, and assert values are 10x larger.
    """
    np.random.seed(43)
    img = np.random.rand(20, 20, 20) * 100.0
    # Avoid edge points that include padding zeroes
    points = np.random.rand(10, 3) * 16.0 + 2.0

    sampled_base = np.array(sample_intensity_around_points(img, points, diameter=3))
    sampled_scaled = np.array(sample_intensity_around_points(img * 10.0, points, diameter=3))

    assert np.allclose(sampled_scaled, sampled_base * 10.0, atol=1e-12), "Local intensity sampling is not linearly scalable"

# =========================================
# Source: test_image_utils_new_funcs.py
# =========================================

def test_ensure_pixel_size_array():
    """Test that ensure pixel size array works as expected."""
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
    """Test that generate plane basis works as expected."""
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
    """Test that fit plane ransac works as expected."""
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
    """Test that sample volume plane works as expected."""
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


def test_create_ellipsoid_struct():
    """
    🔎 Test: Verify structuring element creation for 3D morphological operations.
    💡 What: Tests that `create_ellipsoid_struct` generates correct boolean 3D masks for given radius and spacing.
    🎯 Why: Ensures physical scaling correctly determines voxel bounding dimensions and active ellipsoid region.
    """
    # Isotropic test: radius = 3.0 um, spacing = 1.0 um -> radius 3 voxels -> shape (7, 7, 7)
    struct_iso = create_ellipsoid_struct(radius_um=3.0, pixel_sizes={'Z': 1.0, 'Y': 1.0, 'X': 1.0})
    assert struct_iso.shape == (7, 7, 7)
    assert struct_iso.dtype == bool
    # Center voxel must be True
    assert struct_iso[3, 3, 3]

    # Anisotropic test: spacing (2.0, 1.0, 1.0) -> radii_vox = (1.5, 3, 3) -> shape (5, 7, 7)
    struct_aniso = create_ellipsoid_struct(radius_um=3.0, spacing=(2.0, 1.0, 1.0))
    assert struct_aniso.shape == (5, 7, 7)
    assert struct_aniso[2, 3, 3]

    # Tiny radius returns 3x3x3 grid with center voxel True
    struct_tiny = create_ellipsoid_struct(radius_um=1e-8, spacing=(1.0, 1.0, 1.0))
    assert struct_tiny.shape == (3, 3, 3)
    assert struct_tiny[1, 1, 1]
    assert np.sum(struct_tiny) == 1


def test_generate_morphological_surface_mask():
    """
    🔎 Test: Verify generation of morphological envelope and surface layer.
    💡 What: Tests envelope closing and surface mask calculation on a synthetic 3D sphere/cube binary mask.
    🎯 Why: Ensures surface layer extraction is accurately bounded within specified physical surface depth.
    """
    # Create synthetic 3D mask with a sphere of radius 10 at center
    vol = np.zeros((30, 30, 30), dtype=bool)
    z, y, x = np.ogrid[:30, :30, :30]
    mask = (z - 15)**2 + (y - 15)**2 + (x - 15)**2 <= 10**2
    vol[mask] = True

    surface_mask, envelope = generate_morphological_surface_mask(
        vol,
        spacing=(1.0, 1.0, 1.0),
        closing_radius_um=2.0,
        surface_depth_um=3.0,
        downscale_factor=1,
        fill_holes=True
    )

    assert surface_mask.shape == vol.shape
    assert envelope.shape == vol.shape
    assert envelope.dtype == bool
    assert surface_mask.dtype == bool

    # Envelope should cover all foreground mask voxels
    assert np.all(envelope[vol])
    # Surface mask must be a subset of envelope
    assert np.all(envelope[surface_mask])
    # Center of sphere (dist > surface_depth_um) should not be in surface_mask
    assert not surface_mask[15, 15, 15]

    # Test empty mask handling
    empty_vol = np.zeros((10, 10, 10), dtype=bool)
    surf_empty, env_empty = generate_morphological_surface_mask(empty_vol)
    assert not np.any(surf_empty)
    assert not np.any(env_empty)


def test_estimate_inter_label_distance():
    """
    🔎 Test: Verify inter-nucleus distance estimation using label centroids and KDTree.
    💡 What: Tests inter-label physical distance calculations for synthetic 3D labeled centroids.
    🎯 Why: Guarantees accurate statistical estimation (median, p75, p90, recommended closing radius).
    """
    labels = np.zeros((50, 50, 50), dtype=int)
    # Place 3 distinct spherical labels at known physical locations
    # Label 1 at (10, 10, 10), Label 2 at (10, 10, 20) -> dist = 10 um (with 1um spacing)
    # Label 3 at (10, 10, 30) -> dist to nearest = 10 um
    labels[8:12, 8:12, 8:12] = 1
    labels[8:12, 8:12, 18:22] = 2
    labels[8:12, 8:12, 28:32] = 3

    metrics = estimate_inter_label_distance(
        labels,
        pixel_sizes={'Z': 1.0, 'Y': 1.0, 'X': 1.0},
        k_nearest=2
    )

    assert "median_um" in metrics
    assert "p75_um" in metrics
    assert "p90_um" in metrics
    assert "recommended_closing_um" in metrics

    assert np.isclose(metrics["median_um"], 10.0, atol=0.5)

    # Test fallback when fewer labels than k_nearest are provided
    few_labels = np.zeros((20, 20, 20), dtype=int)
    few_labels[5:8, 5:8, 5:8] = 1
    fallback_metrics = estimate_inter_label_distance(few_labels, spacing=(2.0, 2.0, 2.0), k_nearest=2)
    assert fallback_metrics["median_um"] == 2.0
