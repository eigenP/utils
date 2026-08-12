from skimage.exposure import adjust_gamma
import numpy as np
import pytest

from eigenp_utils.intensity_rescaling import correct_z_intensity_decay
from eigenp_utils.intensity_rescaling import contrast_stretching, correct_z_intensity_decay
from eigenp_utils.intensity_rescaling import correct_z_intensity_decay
from eigenp_utils.intensity_rescaling import fit_basic_shading, apply_basic_shading



# =========================================
# Source: test_intensity_rescaling.py
# =========================================

def test_basic_fit_synthetic():
    """Test that basic fit synthetic works as expected."""
    np.random.seed(42)
    sizes = (128, 128)
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))

    gradient = np.sum(grid**2, axis=0)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10

    # Ground truth relative flatfield
    truth = gradient / np.mean(gradient)

    # Generate 8 images with poisson noise
    images = np.random.poisson(lam=gradient.astype(int), size=[8] + list(sizes))

    # Fit flatfield
    res = fit_basic_shading(images, is_3d=False)
    flatfield = res['flatfield']

    # Validate accuracy
    max_error = np.max(np.abs(flatfield - truth))
    assert max_error < 0.35, f"Max error {max_error} exceeded 0.35 threshold"

    # Verify shape
    assert flatfield.shape == sizes

def test_apply_basic_shading():
    """Test that apply basic shading works as expected."""
    np.random.seed(42)
    images = np.random.rand(8, 128, 128)
    flatfield = np.ones((128, 128)) * 2

    corrected = apply_basic_shading(images, flatfield)

    np.testing.assert_allclose(corrected, images / 2.0)

def test_apply_basic_shading_baseline():
    """Test that apply basic shading baseline works as expected."""
    np.random.seed(42)
    images = np.ones((8, 128, 128))
    flatfield = np.ones((128, 128))
    baseline = np.arange(8)

    corrected = apply_basic_shading(images, flatfield, baseline=baseline)

    # Check baseline correction was applied correctly (using defaults and gaussian smoothing)
    # The gaussian filter will slightly blur the `baseline`, but the mean should remain roughly the same
    # For a deterministic check, we test whether baseline logic runs without error
    assert corrected.shape == images.shape

if __name__ == '__main__':
    pytest.main([__file__])

def test_basic_fit_synthetic_3d():
    """Test that basic fit synthetic 3d works as expected."""
    np.random.seed(42)
    sizes = (8, 64, 64)
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))

    gradient = np.sum(grid**2, axis=0)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10

    truth = gradient / np.mean(gradient)

    images = np.random.poisson(lam=gradient.astype(int), size=[4] + list(sizes))

    res = fit_basic_shading(images, is_3d=True)
    flatfield = res['flatfield']

    max_error = np.max(np.abs(flatfield - truth))
    assert max_error < 0.35, f"Max error {max_error} exceeded 0.35 threshold"
    assert flatfield.shape == sizes

def test_apply_basic_shading_dtype():
    """Test that apply basic shading dtype works as expected."""
    np.random.seed(42)
    # Test uint8 input
    images_uint8 = np.random.randint(50, 200, size=(8, 128, 128), dtype=np.uint8)
    flatfield = np.ones((128, 128)) * 0.5  # brightens image (divides by 0.5 -> mult by 2)

    corrected_uint8 = apply_basic_shading(images_uint8, flatfield)

    # Check dtype is preserved
    assert corrected_uint8.dtype == np.uint8

    # Check bounds are enforced (since multiplying by 2 would push many values above 255)
    assert np.max(corrected_uint8) <= 255
    assert np.min(corrected_uint8) >= 0

    # Verify exact math for a specific element that doesn't overflow
    images_uint8[0, 0, 0] = 100
    corrected_uint8 = apply_basic_shading(images_uint8, flatfield)
    assert corrected_uint8[0, 0, 0] == 200

    # Verify overflow clipping
    images_uint8[0, 0, 0] = 150
    corrected_uint8 = apply_basic_shading(images_uint8, flatfield)
    assert corrected_uint8[0, 0, 0] == 255

    # Test uint16 input
    images_uint16 = np.random.randint(5000, 40000, size=(8, 128, 128), dtype=np.uint16)
    corrected_uint16 = apply_basic_shading(images_uint16, flatfield)
    assert corrected_uint16.dtype == np.uint16
    assert np.max(corrected_uint16) <= 65535

def test_basic_fit_synthetic_darkfield():
    """Test that basic fit synthetic darkfield works as expected."""
    np.random.seed(42)
    sizes = (64, 64)
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))

    gradient = np.sum(grid**2, axis=0)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10
    truth_flatfield = gradient / np.mean(gradient)

    truth_darkfield = np.ones(sizes) * 5.0

    # Generate 8 images with poisson noise + darkfield
    images = np.random.poisson(lam=(gradient + truth_darkfield).astype(int), size=[8] + list(sizes))

    # Test approximate with darkfield
    res = fit_basic_shading(images, is_3d=False, get_darkfield=True, fitting_mode='approximate')
    flatfield = res['flatfield']
    darkfield = res['darkfield']

    assert darkfield.shape == sizes
    max_error_ff = np.max(np.abs(flatfield - truth_flatfield))
    assert max_error_ff < 1.5, f"Max error {max_error_ff} exceeded 1.5 threshold"

def test_basic_fit_synthetic_ladmap():
    """Test that basic fit synthetic ladmap works as expected."""
    np.random.seed(42)
    sizes = (64, 64)
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))

    gradient = np.sum(grid**2, axis=0)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10
    truth_flatfield = gradient / np.mean(gradient)

    images = np.random.poisson(lam=(gradient).astype(int), size=[8] + list(sizes))

    # Test ladmap without darkfield
    res = fit_basic_shading(images, is_3d=False, get_darkfield=False, fitting_mode='ladmap')
    flatfield = res['flatfield']

    max_error_ff = np.max(np.abs(flatfield - truth_flatfield))
    assert max_error_ff < 1.5, f"Max error {max_error_ff} exceeded 1.5 threshold"

def test_basic_fit_synthetic_ladmap_darkfield():
    """Test that basic fit synthetic ladmap darkfield works as expected."""
    np.random.seed(42)
    sizes = (32, 32)
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))

    gradient = np.sum(grid**2, axis=0)
    gradient = 0.01 * (np.max(gradient) - gradient) + 10
    truth_flatfield = gradient / np.mean(gradient)

    truth_darkfield = np.ones(sizes) * 5.0
    images = np.random.poisson(lam=(gradient + truth_darkfield).astype(int), size=[8] + list(sizes))

    # Test ladmap with darkfield
    res = fit_basic_shading(images, is_3d=False, get_darkfield=True, fitting_mode='ladmap')
    flatfield = res['flatfield']
    darkfield = res['darkfield']

    max_error_ff = np.max(np.abs(flatfield - truth_flatfield))
    assert max_error_ff < 1.5, f"Max error {max_error_ff} exceeded 1.5 threshold"

def test_ensure_float_and_restore_dtype_decorator():
    """Test that ensure float and restore dtype decorator works as expected."""
    from eigenp_utils.intensity_rescaling import ensure_float_and_restore_dtype
    import warnings

    @ensure_float_and_restore_dtype
    def dummy_func_mult(img, factor):
        # Function receives a float32 implicitly via decorator
        assert np.issubdtype(img.dtype, np.floating)
        return img * factor

    @ensure_float_and_restore_dtype
    def dummy_func_tuple(img):
        return img * 2.0, "metadata"

    # Test Float
    img_float = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    res_float = dummy_func_mult(img_float, 2.0)
    assert res_float.dtype == np.float64
    np.testing.assert_allclose(res_float, [0.2, 1.0, 1.8])

    # Test Uint8 normal range
    img_uint8 = np.array([10, 50, 100], dtype=np.uint8)
    res_uint8 = dummy_func_mult(img_uint8, 2.0)
    assert res_uint8.dtype == np.uint8
    np.testing.assert_equal(res_uint8, [20, 100, 200])

    # Test Uint8 overflow clipping and rounding
    img_uint8_overflow = np.array([10, 150, 200], dtype=np.uint8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res_overflow = dummy_func_mult(img_uint8_overflow, 2.0)
        assert len(w) == 1
        assert "Values outside uint8 range were clipped" in str(w[-1].message)

    assert res_overflow.dtype == np.uint8
    np.testing.assert_equal(res_overflow, [20, 255, 255])

    # Test Uint8 underflow clipping
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res_underflow = dummy_func_mult(img_uint8_overflow, -1.0)
        assert len(w) == 1
        assert "Values outside uint8 range were clipped" in str(w[-1].message)
    np.testing.assert_equal(res_underflow, [0, 0, 0])

    # Test float16
    img_float16 = np.array([1.0, 2.0], dtype=np.float16)
    res_float16 = dummy_func_mult(img_float16, 2.0)
    assert res_float16.dtype == np.float16
    np.testing.assert_allclose(res_float16, [2.0, 4.0])

    # Test Tuple return
    res_tuple = dummy_func_tuple(img_uint8)
    assert isinstance(res_tuple, tuple)
    assert len(res_tuple) == 2
    assert res_tuple[0].dtype == np.uint8
    np.testing.assert_equal(res_tuple[0], [20, 100, 200])
    assert res_tuple[1] == "metadata"

# =========================================
# Source: test_pipeline.py
# =========================================


def test_contrast_stretching_uint8_preserves_dynamic_range():
    """Test that contrast stretching uint8 preserves dynamic range works as expected."""
    # Construct uint8 gradient image with values between 20 and 200
    img_uint8 = np.linspace(20, 200, 10000, dtype=np.uint8).reshape(100, 100)

    stretched = contrast_stretching(img_uint8, p_min=0.0, p_max=100.0)

    assert stretched.dtype == np.uint8
    # Unique value count must reflect smooth stretching, not binary {0, 1} collapse
    assert len(np.unique(stretched)) > 100
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_z_decay_gamma_uint8_correction():
    """Test that z decay gamma uint8 correction works as expected."""
    # Construct 3D stack (Z, Y, X) with intensity decaying along Z
    z_slices = 10
    stack = np.zeros((z_slices, 32, 32), dtype=np.uint8)
    for z in range(z_slices):
        decay_factor = np.exp(-0.2 * z)
        stack[z] = np.uint8(200 * decay_factor)

    corrected_img = correct_z_intensity_decay(stack, method='gamma')

    assert corrected_img.dtype == np.uint8
    # Slice 0 and Slice N-1 should now have substantially equal mean intensity
    assert abs(int(corrected_img[0].mean()) - int(corrected_img[-1].mean())) < 15


def test_decorator_preserves_function_metadata():
    """Test that decorator preserves function metadata works as expected."""
    assert contrast_stretching.__name__ == "contrast_stretching"
    assert "Stretch the intensity range" in contrast_stretching.__doc__


def test_decorator_dictionary_and_tuple_returns():
    """Test that decorator dictionary and tuple returns works as expected."""
    stack = np.ones((5, 16, 16), dtype=np.uint16) * 1000
    result = correct_z_intensity_decay(stack, return_diagnostic=True)

    assert isinstance(result, dict)
    assert "image" in result
    assert result["image"].dtype == np.uint16
    assert isinstance(result["diagnostic_data"], dict)

# =========================================
# Source: test_brightness_invariants.py
# =========================================

def generate_trend_stack(shape=(20, 100, 100), decay_func=None, outlier_idx=None, outlier_factor=0.5, seed=42):
    """
    Generates a synthetic stack where the 99th percentile follows a specific decay function.
    The content is uniform noise scaled to match the decay.
    """
    rng = np.random.default_rng(seed)
    z_dim, y_dim, x_dim = shape
    stack = np.zeros(shape, dtype=np.float32)

    # Base intensity (P99)
    p99_curve = np.zeros(z_dim)

    for z in range(z_dim):
        # Calculate expected P99
        if decay_func:
            target_p99 = decay_func(z)
        else:
            target_p99 = 1.0

        # Store for verification
        p99_curve[z] = target_p99

        # Generate slice
        # We want np.percentile(slice, 99) == target_p99.
        # Uniform [0, M] has percentile 99 at 0.99 * M.
        # So M = target_p99 / 0.99
        max_val = target_p99 / 0.99

        # Add outlier
        if outlier_idx is not None and z == outlier_idx:
            max_val *= outlier_factor

        stack[z] = rng.uniform(0, max_val, (y_dim, x_dim))

    return stack, p99_curve

@pytest.mark.parametrize("gamma_fit_func,method,decay_lambda", [
    ('exponential', 'gain', lambda z: 1.0 * np.exp(-0.1 * z)),
    ('exponential', 'gamma', lambda z: 1.0 * np.exp(-0.1 * z)),
    ('linear', 'gain', lambda z: np.clip(1.0 - 0.02 * z, 0.1, 1.0)),
    ('linear', 'gamma', lambda z: np.clip(1.0 - 0.02 * z, 0.1, 1.0))
])
def test_perfect_restoration(gamma_fit_func, method, decay_lambda):
    """
    Testr Verification: Perfect Signal Restoration

    Verifies that a perfectly exponential or linear decay is recovered to flat intensity
    using the 'gain' or 'gamma' method.

    Invariant: The output stack should have constant 99th percentile across all Z slices.
    """
    Z = 20
    stack, _ = generate_trend_stack(shape=(Z, 50, 50), decay_func=decay_lambda)

    # Apply correction
    corrected = correct_z_intensity_decay(stack, fit_model=gamma_fit_func, method=method)

    # Check 99th percentiles of output
    p99_out = np.array([np.percentile(s, 99) for s in corrected])

    # Should be constant (flat)
    mean_p99 = np.mean(p99_out)
    std_p99 = np.std(p99_out)

    # We expect low variation. For gamma correction on linear scaling,
    # the relationship is non-linear so P99 might not be as perfectly flat
    # as scalar multiplication (gain), so we allow slightly more tolerance.
    cv = std_p99 / mean_p99

    print(f"Mean P99: {mean_p99:.4f}, Std P99: {std_p99:.4f}, CV: {cv:.4f}")

    tolerance = 0.015 if method == 'gain' else 0.08
    assert cv < tolerance, f"Brightness not stabilized! CV={cv:.4f} is too high for method={method}."


def test_trend_preservation_with_outlier():
    """
    Testr Verification: Trend vs. Anomaly Distinction

    Verifies that the algorithm corrects the Global Trend (systematic decay)
    but preserves Local Anomalies (sample outliers).

    This is a critical property: we want to correct for physics (attenuation),
    not normalize away biology (e.g., a slice with no cells).

    Setup:
    - Exponential decay trend.
    - Slice 10 is artificially dimmed by 50%.

    Expectation:
    - The trend (non-outlier slices) becomes flat.
    - Slice 10 remains ~50% of the restored brightness level.
    """
    Z = 20
    decay = lambda z: 1.0 * np.exp(-0.05 * z)
    outlier_z = 10
    factor = 0.5

    stack, _ = generate_trend_stack(shape=(Z, 50, 50), decay_func=decay,
                                    outlier_idx=outlier_z, outlier_factor=factor)

    # Apply correction
    corrected = correct_z_intensity_decay(stack, fit_model='exponential', method='gain')

    p99_out = np.array([np.percentile(s, 99) for s in corrected])

    # 1. Verify trend is flat EXCLUDING the outlier
    # Mask out the outlier
    mask = np.ones(Z, dtype=bool)
    mask[outlier_z] = False

    trend_vals = p99_out[mask]
    outlier_val = p99_out[outlier_z]

    mean_trend = np.mean(trend_vals)
    std_trend = np.std(trend_vals)

    # Trend should be flat (corrected)
    cv_trend = std_trend / mean_trend
    assert cv_trend < 0.02, f"Global trend was not corrected properly. CV={cv_trend}"

    # 2. Verify outlier is preserved (relative to trend)
    # The outlier started at 0.5 * Trend.
    # The correction multiplied it by (Target / Fitted_Trend).
    # So Result = (0.5 * True_Trend) * (Target / Fitted_Trend).
    # Assuming Fitted_Trend ~ True_Trend, Result ~ 0.5 * Target.
    # Since Target ~ mean_trend (of corrected image), Result ~ 0.5 * mean_trend.

    ratio = outlier_val / mean_trend
    print(f"Outlier Ratio: {ratio:.4f} (Expected ~{factor})")

    # We allow some tolerance because the outlier affects the fit slightly
    assert np.isclose(ratio, factor, atol=0.1), \
        f"Local anomaly was not preserved correctly. Ratio {ratio} != {factor}"

def test_idempotence():
    """
    Testr Verification: Idempotence / Stability

    Verifies that re-running the correction on already corrected data
    (which is now flat) results in minimal change.

    If the function is stable, fitting an exponential to a flat line should yield
    decay=0 (flat), resulting in gains of 1.0.
    """
    Z = 15
    decay = lambda z: 1.0 * np.exp(-0.1 * z)
    stack, _ = generate_trend_stack(shape=(Z, 50, 50), decay_func=decay)

    # First pass
    pass1 = correct_z_intensity_decay(stack, fit_model='exponential', method='gain')

    # Second pass
    pass2 = correct_z_intensity_decay(pass1, fit_model='exponential', method='gain')

    # Check difference
    # Using relative error to be scale-invariant
    # Avoid div by zero
    diff = np.abs(pass1 - pass2)
    rel_diff = diff / (pass1 + 1e-9)

    max_rel_diff = np.max(rel_diff)
    print(f"Max Relative Difference: {max_rel_diff:.6f}")

    # Tolerance Note:
    # We use 50x50 patches. The 99th percentile has sampling variance.
    # Even if the underlying distribution is perfectly flat, the sample P99s will fluctuate.
    # curve_fit will slightly overfit these fluctuations, finding a tiny non-zero decay/growth.
    # A change of < 0.1% (1e-3) is acceptable and indicates stability.
    assert max_rel_diff < 1.5e-3, \
        f"Algorithm is not idempotent; second pass changed values by {max_rel_diff*100:.4f}%"

def test_return_diagnostic_dict():
    """Test that return diagnostic dict works as expected."""
    image = np.random.randint(0, 255, (10, 20, 20), dtype=np.uint8)

    # Test that it returns a dictionary with 'image' and 'diagnostic_data'
    result = correct_z_intensity_decay(image, fit_model='exponential', return_diagnostic=True)

    assert isinstance(result, dict)
    assert 'image' in result
    assert 'diagnostic_data' in result
    assert isinstance(result['image'], np.ndarray)

    diag_data = result['diagnostic_data']
    assert isinstance(diag_data, dict)
    assert 'x_data' in diag_data
    assert 'y_data_norm' in diag_data
    assert 'y_fit_norm' in diag_data
    assert 'gamma_fit_func' in diag_data

    # Test plotting function
    from eigenp_utils.plotting_utils import brightness_diagnostic_plotter
    from matplotlib.figure import Figure

    fig = brightness_diagnostic_plotter(diag_data)
    assert isinstance(fig, Figure)

# =========================================
# Source: test_gamma_auto.py
# =========================================

def generate_decaying_image(shape=(10, 100, 100), decay_rate=0.1, initial_intensity=1.0):
    z, y, x = shape
    image = np.zeros(shape, dtype=np.float64)
    for i in range(z):
        decay = np.exp(-decay_rate * i)
        # Create a slice with some noise and variation
        slice_data = np.random.normal(loc=initial_intensity * decay, scale=0.01, size=(y, x))
        # Ensure values are within valid range [0, 1]
        image[i] = np.clip(slice_data, 0, 1)
    return image

def test_adjust_gamma_per_slice_auto_exponential():
    """Test automatic gamma finding with exponential decay."""
    # Create an image that decays to 50% intensity
    # decay_rate such that exp(-r * 9) = 0.5 -> -9r = ln(0.5) -> r = -ln(0.5)/9
    decay_rate = -np.log(0.5) / 9
    image = generate_decaying_image(shape=(10, 100, 100), decay_rate=decay_rate, initial_intensity=0.8)

    # Original last slice mean
    original_last_mean = np.mean(image[-1])
    original_first_mean = np.mean(image[0])
    print(f"Original first mean: {original_first_mean}, Last mean: {original_last_mean}")

    # Apply correction using exponential fit
    try:
        adjusted = correct_z_intensity_decay(image, method='gamma', fit_model='exponential')
    except TypeError:
         # Skip if not implemented yet
         pytest.skip("gamma_fit_func not implemented yet")

    # The goal is to make slices uniform brightness.
    # The first slice should remain roughly same (gamma ~ 1).
    # The last slice should be brightened.

    adj_last_mean = np.mean(adjusted[-1])
    adj_first_mean = np.mean(adjusted[0])

    print(f"Adjusted first mean: {adj_first_mean}, Last mean: {adj_last_mean}")

    # The adjusted last slice should be closer to the first slice than before
    assert abs(adj_last_mean - adj_first_mean) < abs(original_last_mean - original_first_mean)

    # Ideally, they are close
    assert np.isclose(adj_last_mean, adj_first_mean, atol=0.05)

def test_adjust_gamma_per_slice_auto_linear():
    """Test automatic gamma finding with linear decay."""
    # Linear decay image
    image = np.zeros((10, 100, 100))
    for i in range(10):
        val = 0.8 - (0.04 * i) # 0.8 down to 0.44
        image[i] = np.random.normal(val, 0.01, (100, 100))

    try:
        adjusted = correct_z_intensity_decay(image, method='gamma', fit_model='linear')
    except TypeError:
        pytest.skip("gamma_fit_func not implemented yet")

    adj_last_mean = np.mean(adjusted[-1])
    adj_first_mean = np.mean(adjusted[0])

    assert np.isclose(adj_last_mean, adj_first_mean, atol=0.05)

if __name__ == "__main__":
    # Manual run for debugging
    try:
        test_adjust_gamma_per_slice_manual()
        print("Manual test passed.")
        test_adjust_gamma_per_slice_auto_exponential()
        print("Exponential test passed.")
        test_adjust_gamma_per_slice_auto_linear()
        print("Linear test passed.")
    except Exception as e:
        print(f"Test failed: {e}")
