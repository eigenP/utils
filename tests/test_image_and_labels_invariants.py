import numpy as np
import pytest
from eigenp_utils.image_and_labels_utils import (
    voronoi_otsu_labeling,
    windowed_slice_projection,
    sample_intensity_around_points_optimized
)
from scipy.ndimage import shift

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

    sampled = sample_intensity_around_points_optimized(img, points, diameter=5)

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

    sampled_base = np.array(sample_intensity_around_points_optimized(img, points, diameter=3))
    sampled_scaled = np.array(sample_intensity_around_points_optimized(img * 10.0, points, diameter=3))

    assert np.allclose(sampled_scaled, sampled_base * 10.0, atol=1e-12), "Local intensity sampling is not linearly scalable"
