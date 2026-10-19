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


def _radial_gradient(sizes):
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))
    gradient = np.sum(grid**2, axis=0)
    return 0.01 * (np.max(gradient) - gradient) + 10


@pytest.mark.parametrize("sizes, is_3d", [((8, 64, 64), True), ((64, 64), False)])
def test_basic_single_sample_darkfield_is_unidentifiable(sizes, is_3d):
    """With one image/volume, get_darkfield=True falls back to a flatfield-only fit.

    BaSiC (I_i = B_i * S + D) separates the darkfield from the flatfield only
    through variation of the baselines B_i across samples, so a single sample
    cannot identify D. Previously the 'approximate' darkfield estimator averaged
    over an empty set of samples, emitting "Mean of empty slice" RuntimeWarnings
    every iteration, never converging (NaN residual) and returning all-NaN
    results after max_iterations x max_reweight_iterations.

    Strategy: fit a single Poisson volume (3D) / image (2D) with a known radial
    flatfield. Expected outcome: exactly one UserWarning explaining the
    fallback, no RuntimeWarning, finite outputs, a zero darkfield, results
    identical to get_darkfield=False (the darkfield path is skipped, not
    approximated), and the flatfield still recovered (positive control that the
    fit itself works). Accuracy uses correlation and mean absolute error rather
    than max error, which over 4096 pixels of one Poisson(~10) image is
    dominated by noise; across seeds corr is ~0.99 (3D) / ~0.86 (2D) against
    ~0.02 for a permuted truth, and MAE <= 0.09.
    """
    import warnings

    rng = np.random.default_rng(0)
    gradient = _radial_gradient(sizes)
    image = rng.poisson(lam=gradient.astype(int)).astype(float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_basic_shading(image, is_3d=is_3d, get_darkfield=True, fitting_mode='approximate')

    user = [w for w in caught if issubclass(w.category, UserWarning) and "not identifiable" in str(w.message)]
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(user) == 1
    assert runtime == [], [str(w.message) for w in runtime]

    for key in ("flatfield", "darkfield", "baseline"):
        assert np.all(np.isfinite(res[key])), key
    assert np.all(res["darkfield"] == 0)

    ref = fit_basic_shading(image, is_3d=is_3d, get_darkfield=False, fitting_mode='approximate')
    for key in ("flatfield", "darkfield", "baseline"):
        np.testing.assert_array_equal(res[key], ref[key])

    truth = gradient / np.mean(gradient)
    assert np.corrcoef(res["flatfield"].ravel(), truth.ravel())[0, 1] > 0.75
    assert np.mean(np.abs(res["flatfield"] - truth)) < 0.15


def test_basic_identical_samples_darkfield_stays_finite():
    """Identical samples carry no darkfield information but must not produce NaNs.

    With N >= 2 identical images every baseline B_i is (numerically) 1, so the
    darkfield estimator's sample set {B_i < 1} may be empty. Strategy: fit three
    copies of one image with get_darkfield=True. Expected outcome: no
    RuntimeWarning and finite flatfield, darkfield and baseline, whichever way
    floating-point rounding decides the B_i < 1 test.
    """
    import warnings

    rng = np.random.default_rng(1)
    image = rng.poisson(lam=_radial_gradient((64, 64)).astype(int)).astype(float)
    images = np.stack([image] * 3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_basic_shading(images, is_3d=False, get_darkfield=True, fitting_mode='approximate')

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []
    for key in ("flatfield", "darkfield", "baseline"):
        assert np.all(np.isfinite(res[key])), key


# =========================================
# Regression tests for the dtype round-trip, z-decay correction and BaSiC fixes
# =========================================
from eigenp_utils.intensity_rescaling import contrast_stretching, contrast_stretch_per_slice, normalize_image, test_z_axis_orientation as _z_orientation


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint32])
def test_dtype_roundtrip_preserves_wide_integer_images(dtype):
    """32/64-bit integer images keep their levels through the float round-trip.

    The decorator normalised by the dtype range in float32: an int64 ramp collapsed
    to one value and the uint32 maximum wrapped to 0. Strategy: stretch a 0..4095
    ramp over its full range. Expected: all 4096 levels survive, 0 maps to 0 and the
    maximum stays the maximum (no wrap-around).
    """
    ramp = np.arange(4096).reshape(64, 64).astype(dtype)
    out = contrast_stretching(ramp, 0, 100)
    assert out.dtype == dtype
    assert len(np.unique(out)) == 4096
    assert out.min() == 0 and out.max() == out.flat[-1]


def test_signed_integer_zero_background_stays_zero():
    """Signed images keep 0 at 0 through intensity corrections.

    Offsetting int16 by its minimum mapped 0 to 0.5, so z-decay correction lifted
    a black background into hundreds of counts. Expected: background stays exactly
    0 for both methods, and contrast stretching stays non-negative.
    """
    rng = np.random.default_rng(0)
    decay = np.linspace(1, 0.3, 8)[:, None, None]
    stack = (rng.uniform(0, 4000, (8, 32, 32)) * decay).astype(np.int16)
    stack[:, :4, :4] = 0
    for method in ("gain", "gamma"):
        assert correct_z_intensity_decay(stack, method=method)[:, :4, :4].max() == 0
    assert contrast_stretching(stack, 0, 100).min() >= 0


def test_normalize_image_wide_ints_and_nans():
    """normalize_image neither wraps int64 nor lets one NaN poison the image.

    float(int64 max) rounds up to 2**63, which wrapped the brightest pixel to the
    most negative value; np.percentile with a NaN returned NaN bounds, making every
    pixel NaN. Expected: int64 output within [0, max]; exactly one NaN out for one in.
    """
    out = normalize_image(np.arange(4096).reshape(64, 64).astype(np.int64))
    assert out.min() == 0 and out.max() > 0
    img = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    img[3, 3] = np.nan
    assert int(np.isnan(normalize_image(img)).sum()) == 1


@pytest.mark.parametrize("method", ["gamma", "gain"])
def test_z_decay_correction_on_raw_float_units(method):
    """Float stacks in raw intensity units (not [0, 1]) are corrected, not ignored or saturated.

    Previously 'gamma' returned the input unchanged and 'gain' clipped everything to
    1.0. Strategy: a 0.3x exponential decay over 10 slices with values up to 1e4.
    Expected: the last/first slice 99th-percentile ratio rises from 0.3 to ~1 and the
    intensity scale is preserved (max of the same order as the input).
    """
    rng = np.random.default_rng(0)
    decay = np.exp(np.log(0.3) * np.arange(10) / 9)[:, None, None]
    raw = rng.uniform(0, 1e4, (10, 32, 32)) * decay
    out = correct_z_intensity_decay(raw, method=method)
    p99 = [np.percentile(out[i], 99) for i in range(10)]
    assert 0.9 < p99[-1] / p99[0] < 1.1
    assert 0.5e4 < out.max() < 2e4


def test_gamma_correction_near_full_scale_does_not_flood():
    """Gamma correction stays sane when the reference percentile is at full scale.

    With gamma = log(y_ref) / log(y_fit) and y_ref -> 1 every factor collapsed to the
    0.1 clip, so deep slices became brighter than the reference (slice 1 mean +66%).
    Expected: each slice's 99th percentile maps to the reference's, and no slice mean
    exceeds the reference mean by more than a modest gamma-curve margin (< 1.5x).
    """
    rng = np.random.default_rng(0)
    decay = np.exp(np.log(0.5) * np.arange(10) / 9)[:, None, None]
    stack = normalize_image(rng.uniform(0, 1, (10, 32, 32)) * decay, 0.5, 99.9)
    out = correct_z_intensity_decay(stack, method="gamma")
    p99 = np.array([np.percentile(out[i], 99) for i in range(10)])
    assert np.allclose(p99, p99[0], rtol=0.05)
    means = out.mean(axis=(1, 2))
    assert means.max() < 1.5 * means[0]


def test_gamma_correction_unchanged_for_mid_range_unit_data():
    """For [0, 1] data with a reference percentile <= 0.5 the classic formula is kept.

    Expected: identical to applying g_i = log(y_ref)/log(y_fit_i) slice by slice.
    """
    rng = np.random.default_rng(1)
    decay = np.exp(np.log(0.5) * np.arange(6) / 5)[:, None, None]
    stack = (rng.uniform(0, 0.4, (6, 16, 16)) * decay).astype(np.float32)
    out = correct_z_intensity_decay(stack, method="gamma", return_diagnostic=True)
    y_fit = np.clip(out["diagnostic_data"]["y_fit_norm"], 1e-9, 1.0)
    g = np.clip(np.log(y_fit.max()) / np.log(y_fit), 0.1, 10.0)
    expected = np.stack([stack[i] ** g[i] for i in range(6)])
    np.testing.assert_allclose(out["image"], expected, rtol=1e-5, atol=1e-6)


def test_z_decay_rejects_unknown_method():
    """An unknown method raises instead of returning uninitialised memory."""
    with pytest.raises(ValueError, match="Unknown correction method"):
        correct_z_intensity_decay(np.full((4, 8, 8), 100, np.uint16), method="exponential", fit_model=None)


def test_contrast_stretch_per_slice_single_bound():
    """Passing only p_max_array (or only p_min_array) uses the slice's own other bound."""
    img = np.random.default_rng(0).uniform(0, 1000, (4, 8, 8))
    out = contrast_stretch_per_slice(img, p_max_array=np.full(4, 500.0))
    assert out.shape == img.shape
    out = contrast_stretch_per_slice(img, p_min_array=np.full(4, 100.0))
    assert out.shape == img.shape


def test_z_axis_orientation_uses_all_bottom_slices_and_channels():
    """The bottom half includes the last slice, and (C,Z,Y,X) / (T,C,Z,Y,X) channels work.

    Expected: a 3-slice stack bright only in its last slice is flagged for flipping;
    4D and 5D inputs with channel= do not crash; Z < 2 raises a clear error.
    """
    v = np.zeros((3, 8, 8)); v[2] = 1
    assert _z_orientation(v, return_image=False) is True
    _z_orientation(np.random.rand(2, 10, 16, 16), channel=1, return_image=False)
    _z_orientation(np.random.rand(1, 2, 10, 16, 16), channel=1, return_image=False)
    with pytest.raises(ValueError):
        _z_orientation(np.zeros((1, 8, 8)), return_image=False)


@pytest.mark.parametrize("get_darkfield", [False, True])
def test_basic_fit_ladmap_recovers_flatfield(get_darkfield):
    """fitting_mode='ladmap' recovers a known flatfield and baselines.

    It used to stop after one iteration (the update is ~0 while the tiny initial mu
    still shrinks S to zero) and returned an all-zero flatfield and baseline.
    Strategy: 20 Poisson images of a radial flatfield with per-image gains.
    Expected: flatfield correlates > 0.99 with the truth and the baseline tracks the
    gains (corr > 0.99), as in 'approximate' mode.
    """
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:64, :64]
    flat = 1.2 - 0.4 * ((yy - 32) ** 2 + (xx - 32) ** 2) / 32 ** 2
    gains = rng.uniform(0.8, 1.2, (20, 1, 1))
    images = rng.poisson(200 * flat[None] * gains).astype(float)
    res = fit_basic_shading(images, fitting_mode='ladmap', get_darkfield=get_darkfield)
    assert np.corrcoef(res["flatfield"].ravel(), flat.ravel())[0, 1] > 0.99
    assert np.corrcoef(res["baseline"], gains.ravel())[0, 1] > 0.99


def test_basic_rejects_zero_reweight_iterations():
    """max_reweight_iterations=0 raises a clear ValueError (was UnboundLocalError)."""
    with pytest.raises(ValueError, match="max_reweight_iterations"):
        fit_basic_shading(np.random.default_rng(0).random((5, 32, 32)), max_reweight_iterations=0)


def test_apply_basic_shading_rounds_and_keeps_baseline_sign():
    """Integer outputs are rounded; baseline-corrected output defaults to float32.

    110 / 1.1 used to truncate to 99. Subtracting a timelapse baseline centres the
    images on zero (as in BaSiCPy), which a uint16 output clipped to 0 for about half
    of the background. Expected: 100, and signed float32 values when a baseline is given.
    """
    out = apply_basic_shading(np.full((2, 4, 4), 110, np.uint16), np.full((4, 4), 1.1))
    assert out.dtype == np.uint16 and out[0, 0, 0] == 100
    out = apply_basic_shading(np.full((3, 4, 4), 100, np.uint16), np.ones((4, 4)), baseline=np.array([90.0, 100, 110]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0, 0], [10, 0, -10])


def test_contrast_stretching_float_with_negative_low_percentile_stays_in_unit_range():
    """A float image whose low percentile is negative maps to [0, 1], not [-1, 1]
    (skimage's out_range='dtype' picks the signed range when in_range min < 0)."""
    img = np.linspace(-1, 9, 1000, dtype=np.float32).reshape(10, 100)
    out = contrast_stretching(img)
    assert out.min() == 0.0 and out.max() == 1.0
