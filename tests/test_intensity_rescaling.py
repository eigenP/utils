import warnings

import numpy as np
import pytest
import scipy.ndimage as ndimage

from eigenp_utils.intensity_rescaling import (
    adjust_brightness_per_slice,
    adjust_gamma_per_slice,
    apply_basic_shading,
    contrast_stretch_per_slice,
    contrast_stretching,
    correct_z_intensity_decay,
    ensure_float_and_restore_dtype,
    fit_basic_shading,
    normalize_image,
    test_z_axis_orientation as _z_orientation,
)


def _radial_gradient(sizes):
    """Smooth radially symmetric illumination profile (higher at center than corners)."""
    grid = np.array(
        np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing="ij")
    )
    gradient = np.sum(grid**2, axis=0)
    return 0.01 * (np.max(gradient) - gradient) + 10.0


def _generate_trend_stack(
    shape=(20, 100, 100), decay_func=None, outlier_idx=None, outlier_factor=0.5, seed=42
):
    """Generate a synthetic 3D stack whose per-slice 99th percentile follows decay_func."""
    rng = np.random.default_rng(seed)
    z_dim, y_dim, x_dim = shape
    stack = np.zeros(shape, dtype=np.float32)
    p99_curve = np.zeros(z_dim)

    for z in range(z_dim):
        target_p99 = decay_func(z) if decay_func else 1.0
        p99_curve[z] = target_p99
        max_val = target_p99 / 0.99
        if outlier_idx is not None and z == outlier_idx:
            max_val *= outlier_factor
        stack[z] = rng.uniform(0, max_val, (y_dim, x_dim))

    return stack, p99_curve


# --- BaSiC shading estimation & application ---


def test_basic_fit_synthetic():
    """Recover a known 2D radial flatfield from Poisson-noisy images."""
    rng = np.random.default_rng(42)
    sizes = (128, 128)
    gradient = _radial_gradient(sizes)
    truth = gradient / np.mean(gradient)
    images = rng.poisson(lam=gradient.astype(int), size=(8,) + sizes)

    res = fit_basic_shading(images, is_3d=False)
    flatfield = res["flatfield"]

    assert flatfield.shape == sizes
    max_error = float(np.max(np.abs(flatfield - truth)))
    assert max_error < 0.35, f"Max error {max_error} exceeded 0.35 threshold"


def test_basic_fit_synthetic_3d():
    """Recover a known 3D radial flatfield from a 4D stack of Poisson-noisy volumes."""
    rng = np.random.default_rng(42)
    sizes = (8, 64, 64)
    gradient = _radial_gradient(sizes)
    truth = gradient / np.mean(gradient)
    images = rng.poisson(lam=gradient.astype(int), size=(4,) + sizes)

    res = fit_basic_shading(images, is_3d=True)
    flatfield = res["flatfield"]

    assert flatfield.shape == sizes
    max_error = float(np.max(np.abs(flatfield - truth)))
    assert max_error < 0.35, f"Max error {max_error} exceeded 0.35 threshold"


@pytest.mark.parametrize("fitting_mode", ["approximate", "ladmap"])
@pytest.mark.parametrize("get_darkfield", [False, True])
def test_basic_fit_recovers_shading(fitting_mode, get_darkfield):
    """Both fitting modes recover flatfield, varying per-frame gains, and darkfield.

    Strategy: generate 20 Poisson images I_i = 200 * gains_i * flat_true + dark_offset
    with gains spanning [0.5, 1.5] so the darkfield is identifiable from gain variation.
    Expected: flatfield correlates > 0.98 with truth and has MAE < 0.15; baseline
    correlates > 0.98 with gains; darkfield is identically zero when disabled and
    finite positive (matching dark_offset in approximate mode) when enabled.
    """
    rng = np.random.default_rng(0)
    sizes = (64, 64)
    yy, xx = np.mgrid[: sizes[0], : sizes[1]]
    flat_raw = 1.2 - 0.4 * ((yy - 32) ** 2 + (xx - 32) ** 2) / 32**2
    flat_true = flat_raw / np.mean(flat_raw)
    gains = np.linspace(0.5, 1.5, 20)
    dark_offset = 40.0

    expected = 200.0 * flat_true[None] * gains[:, None, None] + dark_offset
    images = rng.poisson(expected).astype(float)

    res = fit_basic_shading(images, fitting_mode=fitting_mode, get_darkfield=get_darkfield)

    assert res["flatfield"].shape == sizes
    assert res["darkfield"].shape == sizes
    assert res["baseline"].shape == (20,)
    for key in ("flatfield", "darkfield", "baseline"):
        assert np.all(np.isfinite(res[key])), key

    assert np.corrcoef(res["flatfield"].ravel(), flat_true.ravel())[0, 1] > 0.98
    assert float(np.mean(np.abs(res["flatfield"] - flat_true))) < 0.15
    assert np.corrcoef(res["baseline"], gains)[0, 1] > 0.98

    if get_darkfield:
        mean_df = float(np.mean(res["darkfield"]))
        if fitting_mode == "approximate":
            assert 0.5 * dark_offset < mean_df < 1.5 * dark_offset, (
                f"Expected darkfield mean near {dark_offset}, got {mean_df:.2f}"
            )
        else:
            assert mean_df > 0.0
    else:
        np.testing.assert_array_equal(res["darkfield"], 0.0)


def test_fit_basic_shading_list_input_and_working_size_roundtrip():
    """fit_basic_shading accepts a list of 2D arrays and downscales/upscales via working_size."""
    rng = np.random.default_rng(5)
    sizes = (64, 64)
    gradient = _radial_gradient(sizes)
    truth = gradient / np.mean(gradient)
    stack = rng.poisson(lam=gradient.astype(int), size=(8,) + sizes)

    res_arr = fit_basic_shading(stack, is_3d=False, working_size=64)
    res_list = fit_basic_shading([stack[i] for i in range(stack.shape[0])], is_3d=False, working_size=64)
    for key in ("flatfield", "darkfield", "baseline"):
        np.testing.assert_array_equal(res_list[key], res_arr[key])

    # Downscale to 32x32 during fitting, then upscale back to (64, 64)
    res_small = fit_basic_shading(stack, is_3d=False, working_size=32)
    assert res_small["flatfield"].shape == sizes
    assert res_small["darkfield"].shape == sizes
    assert res_small["baseline"].shape == (8,)
    assert np.corrcoef(res_small["flatfield"].ravel(), truth.ravel())[0, 1] > 0.98

    with pytest.raises(ValueError, match="fitting_mode"):
        fit_basic_shading(stack, fitting_mode="unknown")


def test_apply_basic_shading():
    """Dividing by a constant flatfield scales float images exactly."""
    rng = np.random.default_rng(42)
    images = rng.random((8, 64, 64))
    flatfield = np.full((64, 64), 2.0)

    corrected = apply_basic_shading(images, flatfield)
    np.testing.assert_allclose(corrected, images / 2.0)


def test_apply_basic_shading_dtype():
    """Input integer dtype is preserved with exact rounding, clamping, and clipping UserWarning."""
    rng = np.random.default_rng(42)
    flatfield = np.full((64, 64), 0.5)  # divides by 0.5 -> multiplies by 2

    images_uint8 = rng.integers(50, 200, size=(4, 64, 64), dtype=np.uint8)
    images_uint8[0, 0, 0] = 100
    images_uint8[0, 0, 1] = 150
    with pytest.warns(UserWarning, match="clips"):
        corrected_uint8 = apply_basic_shading(images_uint8, flatfield)

    assert corrected_uint8.dtype == np.uint8
    assert corrected_uint8[0, 0, 0] == 200
    assert corrected_uint8[0, 0, 1] == 255
    expected_u8 = np.clip(np.round(images_uint8.astype(np.float64) / 0.5), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(corrected_uint8, expected_u8)

    images_uint16 = rng.integers(5000, 45000, size=(4, 64, 64), dtype=np.uint16)
    images_uint16[0, 0, 0] = 20000
    images_uint16[0, 0, 1] = 40000
    with pytest.warns(UserWarning, match="clips"):
        corrected_uint16 = apply_basic_shading(images_uint16, flatfield)

    assert corrected_uint16.dtype == np.uint16
    assert corrected_uint16[0, 0, 0] == 40000
    assert corrected_uint16[0, 0, 1] == 65535
    expected_u16 = np.clip(np.round(images_uint16.astype(np.float64) / 0.5), 0, 65535).astype(np.uint16)
    np.testing.assert_array_equal(corrected_uint16, expected_u16)


def test_apply_basic_shading_gaussian_baseline_zero_flatfield_and_validation():
    """Verify gaussian baseline smoothing, zero-flatfield division guard, and shape validation."""
    images = np.full((6, 8, 8), 100.0, dtype=np.float32)
    flatfield = np.ones((8, 8), dtype=np.float32)
    baseline = np.array([10.0, 20.0, 80.0, 20.0, 10.0, 10.0], dtype=np.float32)

    smoothed = apply_basic_shading(
        images,
        flatfield,
        baseline=baseline,
        baseline_smooth_method="gaussian",
        baseline_smooth_sigma=1.5,
    )
    expected_b = ndimage.gaussian_filter1d(baseline, sigma=1.5)
    np.testing.assert_allclose(smoothed[:, 0, 0], 100.0 - expected_b, rtol=1e-5)

    # Zero-flatfield guard replaces 0 with 1e-9 rather than producing inf/nan
    ff_with_zero = np.ones((8, 8), dtype=np.float64)
    ff_with_zero[0, 0] = 0.0
    imgs_zero = np.ones((2, 8, 8), dtype=np.float64) * 5.0
    imgs_zero[:, 0, 0] = 0.0
    out_guard = apply_basic_shading(imgs_zero, ff_with_zero)
    assert np.all(np.isfinite(out_guard))
    assert out_guard[0, 0, 0] == 0.0
    np.testing.assert_allclose(out_guard[:, 1:, 1:], 5.0)

    # Shape validation errors
    with pytest.raises(ValueError, match="fewer dimensions"):
        apply_basic_shading(np.ones((8, 8)), np.ones((2, 8, 8)))
    with pytest.raises(ValueError, match="Baseline size"):
        apply_basic_shading(np.ones((4, 8, 8)), np.ones((8, 8)), baseline=np.ones(3))


# --- Dtype decorator, contrast stretching & normalization ---


def test_ensure_float_and_restore_dtype_decorator():
    """Decorator converts integer inputs to float, restores dtype on return, and preserves metadata."""

    @ensure_float_and_restore_dtype
    def dummy_func_mult(img, factor):
        """Multiply an image by a scalar factor."""
        assert np.issubdtype(img.dtype, np.floating)
        return img * factor

    @ensure_float_and_restore_dtype
    def dummy_func_tuple(img):
        return img * 2.0, "metadata"

    @ensure_float_and_restore_dtype
    def dummy_func_dict(img):
        return {"image": img * 2.0, "corrected": img * 0.5, "meta": "ok"}

    # Metadata preservation (custom function and public decorated function)
    assert dummy_func_mult.__name__ == "dummy_func_mult"
    assert "Multiply an image" in dummy_func_mult.__doc__
    assert contrast_stretching.__name__ == "contrast_stretching"
    assert "Stretch the intensity range" in contrast_stretching.__doc__

    # Float64 passthrough
    img_float = np.array([0.1, 0.5, 0.9], dtype=np.float64)
    res_float = dummy_func_mult(img_float, 2.0)
    assert res_float.dtype == np.float64
    np.testing.assert_allclose(res_float, [0.2, 1.0, 1.8])

    # Uint8 normal range
    img_uint8 = np.array([10, 50, 100], dtype=np.uint8)
    res_uint8 = dummy_func_mult(img_uint8, 2.0)
    assert res_uint8.dtype == np.uint8
    np.testing.assert_equal(res_uint8, [20, 100, 200])

    # Uint8 overflow and underflow clipping warnings
    img_uint8_overflow = np.array([10, 150, 200], dtype=np.uint8)
    with pytest.warns(RuntimeWarning, match="Values outside uint8 range were clipped"):
        res_overflow = dummy_func_mult(img_uint8_overflow, 2.0)
    assert res_overflow.dtype == np.uint8
    np.testing.assert_equal(res_overflow, [20, 255, 255])

    with pytest.warns(RuntimeWarning, match="Values outside uint8 range were clipped"):
        res_underflow = dummy_func_mult(img_uint8_overflow, -1.0)
    np.testing.assert_equal(res_underflow, [0, 0, 0])

    # Float16 restored to float16
    img_float16 = np.array([1.0, 2.0], dtype=np.float16)
    res_float16 = dummy_func_mult(img_float16, 2.0)
    assert res_float16.dtype == np.float16
    np.testing.assert_allclose(res_float16, [2.0, 4.0])

    # Tuple return converts first element only
    res_tuple = dummy_func_tuple(img_uint8)
    assert isinstance(res_tuple, tuple) and len(res_tuple) == 2
    assert res_tuple[0].dtype == np.uint8
    np.testing.assert_equal(res_tuple[0], [20, 100, 200])
    assert res_tuple[1] == "metadata"

    # Dict return converts 'image' and 'corrected' keys while leaving other keys untouched
    res_dict = dummy_func_dict(img_uint8)
    assert res_dict["image"].dtype == np.uint8
    assert res_dict["corrected"].dtype == np.uint8
    np.testing.assert_equal(res_dict["image"], [20, 100, 200])
    np.testing.assert_equal(res_dict["corrected"], [5, 25, 50])
    assert res_dict["meta"] == "ok"

    # Also verify dict conversion through correct_z_intensity_decay(return_diagnostic=True)
    stack_u16 = np.full((5, 16, 16), 1000, dtype=np.uint16)
    diag_res = correct_z_intensity_decay(stack_u16, return_diagnostic=True)
    assert diag_res["image"].dtype == np.uint16
    assert isinstance(diag_res["diagnostic_data"], dict)


def test_contrast_stretching_uint8_preserves_dynamic_range():
    """Stretching a uint8 gradient preserves smooth dynamic range across [0, 255]."""
    img_uint8 = np.linspace(20, 200, 10000, dtype=np.uint8).reshape(100, 100)
    stretched = contrast_stretching(img_uint8, p_min=0.0, p_max=100.0)

    assert stretched.dtype == np.uint8
    assert len(np.unique(stretched)) > 100
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_contrast_stretching_and_normalize_image_constant_and_dtypes():
    """Constant images produce finite outputs without NaNs, and normalize_image converts dtypes."""
    const_f32 = np.full((16, 16), 0.42, dtype=np.float32)
    out_cs_f32 = contrast_stretching(const_f32)
    assert out_cs_f32.dtype == np.float32
    np.testing.assert_allclose(out_cs_f32, const_f32)

    const_u8 = np.full((16, 16), 77, dtype=np.uint8)
    out_cs_u8 = contrast_stretching(const_u8)
    assert out_cs_u8.dtype == np.uint8
    np.testing.assert_array_equal(out_cs_u8, const_u8)

    out_norm_const = normalize_image(const_f32)
    assert out_norm_const.dtype == np.float32
    assert np.all(np.isfinite(out_norm_const))
    np.testing.assert_array_equal(out_norm_const, np.zeros_like(const_f32))

    # Explicit dtype conversions in normalize_image
    ramp_f32 = np.linspace(10.0, 210.0, 256, dtype=np.float32).reshape(16, 16)
    norm_u8 = normalize_image(ramp_f32, lower_percentile=0.0, upper_percentile=100.0, dtype=np.uint8)
    assert norm_u8.dtype == np.uint8
    assert norm_u8.min() == 0 and norm_u8.max() == 255

    norm_u16 = normalize_image(ramp_f32, lower_percentile=0.0, upper_percentile=100.0, dtype=np.uint16)
    assert norm_u16.dtype == np.uint16
    assert norm_u16.min() == 0 and norm_u16.max() == 65535

    ramp_u16 = np.linspace(100, 50000, 256, dtype=np.uint16).reshape(16, 16)
    norm_f32 = normalize_image(ramp_u16, lower_percentile=0.0, upper_percentile=100.0, dtype=np.float32)
    assert norm_f32.dtype == np.float32
    np.testing.assert_allclose([norm_f32.min(), norm_f32.max()], [0.0, 1.0], atol=1e-6)


# --- Z-axis intensity decay correction & deprecated wrappers ---


def test_z_decay_gamma_uint8_correction():
    """Gamma decay correction restores a decaying uint8 Z-stack to uniform brightness."""
    z_slices = 10
    stack = np.zeros((z_slices, 32, 32), dtype=np.uint8)
    for z in range(z_slices):
        stack[z] = np.uint8(200 * np.exp(-0.2 * z))

    corrected_img = correct_z_intensity_decay(stack, method="gamma")

    assert corrected_img.dtype == np.uint8
    assert abs(int(corrected_img[0].mean()) - int(corrected_img[-1].mean())) < 15


@pytest.mark.parametrize(
    "gamma_fit_func,method,decay_lambda",
    [
        ("exponential", "gain", lambda z: 1.0 * np.exp(-0.1 * z)),
        ("exponential", "gamma", lambda z: 1.0 * np.exp(-0.1 * z)),
        ("linear", "gain", lambda z: np.clip(1.0 - 0.02 * z, 0.1, 1.0)),
        ("linear", "gamma", lambda z: np.clip(1.0 - 0.02 * z, 0.1, 1.0)),
    ],
)
def test_perfect_restoration(gamma_fit_func, method, decay_lambda):
    """Exponential and linear Z-decays are stabilized to flat per-slice 99th percentiles."""
    stack, _ = _generate_trend_stack(shape=(20, 50, 50), decay_func=decay_lambda)
    corrected = correct_z_intensity_decay(stack, fit_model=gamma_fit_func, method=method)

    p99_out = np.array([np.percentile(s, 99) for s in corrected])
    cv = float(np.std(p99_out) / np.mean(p99_out))

    tolerance = 0.015 if method == "gain" else 0.08
    assert cv < tolerance, f"Brightness not stabilized! CV={cv:.4f} is too high for method={method}."


def test_trend_preservation_with_outlier():
    """Corrects systematic global Z-decay while preserving a localized 50%-dimmed outlier slice."""
    Z = 20
    outlier_z = 10
    factor = 0.5
    stack, _ = _generate_trend_stack(
        shape=(Z, 50, 50),
        decay_func=lambda z: 1.0 * np.exp(-0.05 * z),
        outlier_idx=outlier_z,
        outlier_factor=factor,
    )

    corrected = correct_z_intensity_decay(stack, fit_model="exponential", method="gain")
    p99_out = np.array([np.percentile(s, 99) for s in corrected])

    mask = np.ones(Z, dtype=bool)
    mask[outlier_z] = False
    mean_trend = float(np.mean(p99_out[mask]))
    cv_trend = float(np.std(p99_out[mask]) / mean_trend)
    assert cv_trend < 0.02, f"Global trend was not corrected properly. CV={cv_trend}"

    ratio = float(p99_out[outlier_z] / mean_trend)
    assert np.isclose(ratio, factor, atol=0.1), f"Outlier ratio {ratio} != {factor}"


def test_idempotence():
    """Re-running Z-decay correction on an already-corrected stack leaves it virtually unchanged."""
    stack, _ = _generate_trend_stack(shape=(15, 50, 50), decay_func=lambda z: 1.0 * np.exp(-0.1 * z))

    pass1 = correct_z_intensity_decay(stack, fit_model="exponential", method="gain")
    pass2 = correct_z_intensity_decay(pass1, fit_model="exponential", method="gain")

    max_rel_diff = float(np.max(np.abs(pass1 - pass2) / (pass1 + 1e-9)))
    assert max_rel_diff < 1.5e-3, (
        f"Algorithm is not idempotent; second pass changed values by {max_rel_diff * 100:.4f}%"
    )


def test_return_diagnostic_dict():
    """return_diagnostic=True returns expected keys, shapes, fitted curve, and model tag."""
    Z = 10
    stack, _ = _generate_trend_stack(shape=(Z, 32, 32), decay_func=lambda z: 0.9 * np.exp(-0.1 * z), seed=7)
    image_u8 = np.clip(np.round(stack * 255), 0, 255).astype(np.uint8)

    result = correct_z_intensity_decay(image_u8, fit_model="exponential", return_diagnostic=True)

    assert set(result.keys()) == {"image", "diagnostic_data"}
    assert result["image"].shape == image_u8.shape
    assert result["image"].dtype == np.uint8

    diag = result["diagnostic_data"]
    assert set(diag.keys()) == {"x_data", "y_data_norm", "y_fit_norm", "gamma_fit_func"}
    np.testing.assert_array_equal(diag["x_data"], np.arange(Z))
    assert diag["gamma_fit_func"] == "exponential"
    assert diag["y_data_norm"].shape == (Z,)
    assert diag["y_fit_norm"].shape == (Z,)

    expected_p99 = np.array([np.nanpercentile(image_u8[i] / 255.0, 99) for i in range(Z)])
    np.testing.assert_allclose(diag["y_data_norm"], expected_p99, rtol=1e-5)
    assert np.all(np.diff(diag["y_fit_norm"]) <= 0.0)
    np.testing.assert_allclose(diag["y_fit_norm"], diag["y_data_norm"], atol=0.03)


def test_deprecated_z_decay_wrappers():
    """Deprecated adjust_gamma_per_slice and adjust_brightness_per_slice warn and remap kwargs."""
    stack, _ = _generate_trend_stack(shape=(8, 32, 32), decay_func=lambda z: 0.8 * np.exp(-0.1 * z), seed=9)

    ref_gamma = correct_z_intensity_decay(stack, method="gamma", fit_model="linear")
    with pytest.warns(DeprecationWarning, match="adjust_gamma_per_slice is deprecated"):
        out_gamma = adjust_gamma_per_slice(stack, gamma_fit_func="linear", FLIP_Z_AXIS=True)
    np.testing.assert_array_equal(out_gamma, ref_gamma)

    ref_gain = correct_z_intensity_decay(stack, method="gain", fit_model="exponential")
    with pytest.warns(DeprecationWarning, match="adjust_brightness_per_slice is deprecated"):
        out_gain = adjust_brightness_per_slice(
            stack, method="gain", gamma_fit_func="exponential", FLIP_Z_AXIS=False
        )
    np.testing.assert_array_equal(out_gain, ref_gain)


def test_correct_z_intensity_decay_fit_model_none_custom_callable_and_edge_cases():
    """Verify fit_model=None ramp, custom callable model, all-zero stack, and error paths."""
    rng = np.random.default_rng(13)
    stack = rng.uniform(0.1, 0.8, size=(5, 16, 16)).astype(np.float32)

    # fit_model=None uses a linear ramp of factors from 1.0 to final_gamma
    out_none = correct_z_intensity_decay(stack, method="gamma", fit_model=None, final_gamma=0.6)
    factors = np.linspace(1.0, 0.6, 5)
    expected_none = np.stack([stack[i] ** factors[i] for i in range(5)])
    np.testing.assert_allclose(out_none, expected_none, rtol=1e-5)

    # Custom callable fit_model
    def constant_decay(x_data, y_data):
        return np.linspace(1.0, 0.5, len(x_data))

    res_custom = correct_z_intensity_decay(
        stack, method="gain", fit_model=constant_decay, return_diagnostic=True
    )
    assert res_custom["diagnostic_data"]["gamma_fit_func"] == "custom_callable"
    gains = 1.0 / np.linspace(1.0, 0.5, 5)
    expected_custom = np.stack([np.clip(stack[i] * gains[i], 0.0, 1.0) for i in range(5)])
    np.testing.assert_allclose(res_custom["image"], expected_custom, rtol=1e-5)

    # All-zero stack stays all zeros cleanly without NaNs or division by zero
    zeros = np.zeros((4, 8, 8), dtype=np.float32)
    for method in ("gamma", "gain"):
        for model in ("linear", "exponential"):
            out_zero = correct_z_intensity_decay(zeros, method=method, fit_model=model)
            assert np.all(np.isfinite(out_zero))
            np.testing.assert_array_equal(out_zero, zeros)

    # Validation errors
    with pytest.raises(ValueError, match="3D numpy array"):
        correct_z_intensity_decay(np.ones((16, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        correct_z_intensity_decay(np.full((4, 8, 8), -0.5, dtype=np.float32), method="gamma")
    with pytest.raises(RuntimeError, match="Z-decay fitting failed"):
        correct_z_intensity_decay(stack, fit_model="quadratic")
    with pytest.raises(RuntimeError, match="Z-decay fitting failed"):
        correct_z_intensity_decay(stack, fit_model=123)


# --- Regressions & remaining API coverage ---


@pytest.mark.parametrize("sizes, is_3d", [((8, 64, 64), True), ((64, 64), False)])
def test_basic_single_sample_darkfield_is_unidentifiable(sizes, is_3d):
    """With one image/volume, get_darkfield=True falls back to a flatfield-only fit.

    BaSiC (I_i = B_i * S + D) separates the darkfield from the flatfield only
    through variation of the baselines B_i across samples, so a single sample
    cannot identify D. Strategy: fit a single Poisson volume (3D) / image (2D)
    with a known radial flatfield. Expected outcome: exactly one UserWarning
    explaining the fallback, no RuntimeWarning, finite outputs, a zero
    darkfield, results identical to get_darkfield=False, and the flatfield
    still recovered (corr > 0.75, MAE < 0.15).
    """
    rng = np.random.default_rng(0)
    gradient = _radial_gradient(sizes)
    image = rng.poisson(lam=gradient.astype(int)).astype(float)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_basic_shading(image, is_3d=is_3d, get_darkfield=True, fitting_mode="approximate")

    user = [w for w in caught if issubclass(w.category, UserWarning) and "not identifiable" in str(w.message)]
    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(user) == 1
    assert runtime == [], [str(w.message) for w in runtime]

    for key in ("flatfield", "darkfield", "baseline"):
        assert np.all(np.isfinite(res[key])), key
    assert np.all(res["darkfield"] == 0)

    ref = fit_basic_shading(image, is_3d=is_3d, get_darkfield=False, fitting_mode="approximate")
    for key in ("flatfield", "darkfield", "baseline"):
        np.testing.assert_array_equal(res[key], ref[key])

    truth = gradient / np.mean(gradient)
    assert np.corrcoef(res["flatfield"].ravel(), truth.ravel())[0, 1] > 0.75
    assert np.mean(np.abs(res["flatfield"] - truth)) < 0.15


def test_basic_identical_samples_darkfield_stays_finite():
    """Identical samples carry no darkfield information but must stay finite with no RuntimeWarning."""
    rng = np.random.default_rng(1)
    image = rng.poisson(lam=_radial_gradient((64, 64)).astype(int)).astype(float)
    images = np.stack([image] * 3)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fit_basic_shading(images, is_3d=False, get_darkfield=True, fitting_mode="approximate")

    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []
    for key in ("flatfield", "darkfield", "baseline"):
        assert np.all(np.isfinite(res[key])), key


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint32])
def test_dtype_roundtrip_preserves_wide_integer_images(dtype):
    """32/64-bit integer images keep all 4096 ramp levels without collapsing or wrapping."""
    ramp = np.arange(4096).reshape(64, 64).astype(dtype)
    out = contrast_stretching(ramp, 0, 100)
    assert out.dtype == dtype
    assert len(np.unique(out)) == 4096
    assert out.min() == 0 and out.max() == out.flat[-1]


def test_signed_integer_zero_background_stays_zero():
    """Signed int16 images keep a zero background at 0 through z-decay and contrast stretching."""
    rng = np.random.default_rng(0)
    decay = np.linspace(1, 0.3, 8)[:, None, None]
    stack = (rng.uniform(0, 4000, (8, 32, 32)) * decay).astype(np.int16)
    stack[:, :4, :4] = 0
    for method in ("gain", "gamma"):
        assert correct_z_intensity_decay(stack, method=method)[:, :4, :4].max() == 0
    assert contrast_stretching(stack, 0, 100).min() >= 0


def test_normalize_image_wide_ints_and_nans():
    """normalize_image neither wraps int64 nor lets a single NaN poison the whole image."""
    out = normalize_image(np.arange(4096).reshape(64, 64).astype(np.int64))
    assert out.min() == 0 and out.max() > 0
    img = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    img[3, 3] = np.nan
    assert int(np.isnan(normalize_image(img)).sum()) == 1


@pytest.mark.parametrize("method", ["gamma", "gain"])
def test_z_decay_correction_on_raw_float_units(method):
    """Float stacks in raw intensity units (not [0, 1]) are corrected without saturation."""
    rng = np.random.default_rng(0)
    decay = np.exp(np.log(0.3) * np.arange(10) / 9)[:, None, None]
    raw = rng.uniform(0, 1e4, (10, 32, 32)) * decay
    out = correct_z_intensity_decay(raw, method=method)
    p99 = [np.percentile(out[i], 99) for i in range(10)]
    assert 0.9 < p99[-1] / p99[0] < 1.1
    assert 0.5e4 < out.max() < 2e4


def test_gamma_correction_near_full_scale_does_not_flood():
    """Gamma correction stays well-behaved when the reference percentile is near 1.0."""
    rng = np.random.default_rng(0)
    decay = np.exp(np.log(0.5) * np.arange(10) / 9)[:, None, None]
    stack = normalize_image(rng.uniform(0, 1, (10, 32, 32)) * decay, 0.5, 99.9)
    out = correct_z_intensity_decay(stack, method="gamma")
    p99 = np.array([np.percentile(out[i], 99) for i in range(10)])
    assert np.allclose(p99, p99[0], rtol=0.05)
    means = out.mean(axis=(1, 2))
    assert means.max() < 1.5 * means[0]


def test_gamma_correction_unchanged_for_mid_range_unit_data():
    """For [0, 1] data with reference percentile <= 0.5, the classic power formula is preserved."""
    rng = np.random.default_rng(1)
    decay = np.exp(np.log(0.5) * np.arange(6) / 5)[:, None, None]
    stack = (rng.uniform(0, 0.4, (6, 16, 16)) * decay).astype(np.float32)
    out = correct_z_intensity_decay(stack, method="gamma", return_diagnostic=True)
    y_fit = np.clip(out["diagnostic_data"]["y_fit_norm"], 1e-9, 1.0)
    g = np.clip(np.log(y_fit.max()) / np.log(y_fit), 0.1, 10.0)
    expected = np.stack([stack[i] ** g[i] for i in range(6)])
    np.testing.assert_allclose(out["image"], expected, rtol=1e-5, atol=1e-6)


def test_z_decay_rejects_unknown_method():
    """An unknown correction method raises ValueError instead of returning uninitialised memory."""
    with pytest.raises(ValueError, match="Unknown correction method"):
        correct_z_intensity_decay(np.full((4, 8, 8), 100, np.uint16), method="exponential", fit_model=None)


def test_contrast_stretch_per_slice_single_bound():
    """Passing only p_max_array or only p_min_array stretches against each slice's own other bound."""
    base = np.array([[0.0, 250.0], [500.0, 1000.0]], dtype=np.float64)
    img = np.stack([base, base, base], axis=0)

    p_max = np.array([500.0, 1000.0, 250.0])
    out_max = contrast_stretch_per_slice(img, p_max_array=p_max)
    assert out_max.shape == img.shape
    np.testing.assert_allclose(out_max[0], [[0.0, 0.5], [1.0, 1.0]])
    np.testing.assert_allclose(out_max[1], [[0.0, 0.25], [0.5, 1.0]])
    np.testing.assert_allclose(out_max[2], [[0.0, 1.0], [1.0, 1.0]])

    p_min = np.array([0.0, 200.0, 500.0])
    out_min = contrast_stretch_per_slice(img, p_min_array=p_min)
    assert out_min.shape == img.shape
    np.testing.assert_allclose(out_min[0], [[0.0, 0.25], [0.5, 1.0]])
    np.testing.assert_allclose(out_min[1], [[0.0, 50.0 / 800.0], [300.0 / 800.0, 1.0]])
    np.testing.assert_allclose(out_min[2], [[0.0, 0.0], [0.0, 1.0]])


def test_contrast_stretch_per_slice_flip_z_axis_and_errors():
    """FLIP_Z_AXIS=True reverses p_min_array/p_max_array along Z; missing bounds or non-3D raise."""
    base = np.array([[0.0, 100.0], [200.0, 400.0]], dtype=np.float64)
    img = np.stack([base, base, base], axis=0)
    p_min = np.array([0.0, 50.0, 100.0])
    p_max = np.array([100.0, 200.0, 400.0])

    flipped = contrast_stretch_per_slice(img, p_min_array=p_min, p_max_array=p_max, FLIP_Z_AXIS=True)
    direct = contrast_stretch_per_slice(img, p_min_array=p_min[::-1], p_max_array=p_max[::-1], FLIP_Z_AXIS=False)
    np.testing.assert_allclose(flipped, direct)

    with pytest.raises(ValueError, match="3D numpy array"):
        contrast_stretch_per_slice(base, p_max_array=p_max)
    with pytest.raises(ValueError, match="no difference"):
        contrast_stretch_per_slice(img)


def test_contrast_stretch_per_slice_float_with_negative_lower_bound_maps_to_unit_range():
    """Float slices whose lower bound is negative map to [0, 1], not [-1, 1].

    Strategy: stretch a float32 ramp from -1 to 9 between explicit bounds
    (-1, 9), and stack it with the same ramp shifted to 0..10; with only
    p_max_array given, each slice's own minimum (-1 or 0) is the lower bound.
    The sibling ``contrast_stretching`` already maps such data to [0, 1].
    Expected outcome: every slice spans exactly [0, 1] and equals
    (v - lower) / (upper - lower), so the shifted slices are identical whether
    or not their lower bound is negative. Negative control: a uint16 stack
    keeps its dtype and the full [0, 65535] output range.
    """
    ramp = np.linspace(-1.0, 9.0, 100, dtype=np.float32).reshape(10, 10)
    expected = (ramp.astype(np.float64) + 1.0) / 10.0
    img = np.stack([ramp, ramp + 1.0])

    both = contrast_stretch_per_slice(img, p_min_array=np.array([-1.0, 0.0]), p_max_array=np.array([9.0, 10.0]))
    only_max = contrast_stretch_per_slice(img, p_max_array=np.array([9.0, 10.0]))
    for out in (both, only_max):
        assert out.dtype == np.float32
        for z in range(2):
            assert out[z].min() == 0.0 and out[z].max() == 1.0
            np.testing.assert_allclose(out[z], expected, atol=1e-6)

    u16 = np.stack([np.linspace(100, 1000, 64).reshape(8, 8)] * 2).astype(np.uint16)
    out_u16 = contrast_stretch_per_slice(u16, p_min_array=np.array([100, 100]), p_max_array=np.array([1000, 1000]))
    assert out_u16.dtype == np.uint16
    assert out_u16.min() == 0 and out_u16.max() == 65535


def test_z_axis_orientation_uses_all_bottom_slices_and_channels():
    """Bottom half includes the last slice, channel selection discriminates per-channel brightness,
    and Z < 2 raises ValueError."""
    v_bot = np.zeros((3, 8, 8))
    v_bot[2] = 1.0
    assert _z_orientation(v_bot, return_image=False) is True

    v_top = np.zeros((3, 8, 8))
    v_top[0] = 1.0
    assert _z_orientation(v_top, return_image=False) is False

    # 4D (C, Z, Y, X): channel 0 bright at top (False), channel 1 bright at bottom (True)
    vol_4d = np.zeros((2, 6, 8, 8), dtype=np.float32)
    vol_4d[0, 0] = 10.0
    vol_4d[1, -1] = 10.0
    assert _z_orientation(vol_4d, channel=0, return_image=False) is False
    assert _z_orientation(vol_4d, channel=1, return_image=False) is True

    # 5D (T, C, Z, Y, X): channel 0 bright at top (False), channel 1 bright at bottom (True)
    vol_5d = vol_4d[np.newaxis, ...]
    assert _z_orientation(vol_5d, channel=0, return_image=False) is False
    assert _z_orientation(vol_5d, channel=1, return_image=False) is True

    with pytest.raises(ValueError):
        _z_orientation(np.zeros((1, 8, 8)), return_image=False)


def test_z_axis_orientation_return_image_and_invalid_channel_dims():
    """return_image=True returns (flipped_or_original, flag), and channel on <4D (2D) raises."""
    vol_needs_flip = np.arange(4 * 4 * 4, dtype=np.float32).reshape(4, 4, 4)
    flipped, flag = _z_orientation(vol_needs_flip, return_image=True)
    assert flag is True
    np.testing.assert_array_equal(flipped, vol_needs_flip[::-1, :, :])

    vol_ok = vol_needs_flip[::-1, :, :].copy()
    unchanged, flag_ok = _z_orientation(vol_ok, return_image=True)
    assert flag_ok is False
    np.testing.assert_array_equal(unchanged, vol_ok)

    # 4D (C, Z, Y, X) flips along Z (axis -3), keeping channel order intact
    vol_4d = np.zeros((2, 4, 4, 4), dtype=np.float32)
    vol_4d[0, 0] = 1.0
    vol_4d[1, 3] = 5.0
    flipped_4d, flag_4d = _z_orientation(vol_4d, channel=1, return_image=True)
    assert flag_4d is True
    np.testing.assert_array_equal(flipped_4d, vol_4d[..., ::-1, :, :])

    with pytest.raises(ValueError, match="channel given"):
        _z_orientation(np.zeros((8, 8)), channel=0, return_image=False)


def test_basic_rejects_zero_reweight_iterations():
    """max_reweight_iterations=0 raises a clear ValueError instead of UnboundLocalError."""
    with pytest.raises(ValueError, match="max_reweight_iterations"):
        fit_basic_shading(np.random.default_rng(0).random((5, 32, 32)), max_reweight_iterations=0)


def test_apply_basic_shading_rounds_and_keeps_baseline_sign():
    """Integer outputs are rounded; baseline-corrected output defaults to signed float32."""
    out = apply_basic_shading(np.full((2, 4, 4), 110, np.uint16), np.full((4, 4), 1.1))
    assert out.dtype == np.uint16 and out[0, 0, 0] == 100
    out = apply_basic_shading(
        np.full((3, 4, 4), 100, np.uint16), np.ones((4, 4)), baseline=np.array([90.0, 100.0, 110.0])
    )
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0, 0], [10.0, 0.0, -10.0])


def test_contrast_stretching_float_with_negative_low_percentile_stays_in_unit_range():
    """A float image whose low percentile is negative still maps to [0, 1], not [-1, 1]."""
    img = np.linspace(-1, 9, 1000, dtype=np.float32).reshape(10, 100)
    out = contrast_stretching(img)
    assert out.min() == 0.0 and out.max() == 1.0


def test_apply_basic_shading_offset_casts_to_input_dtype_and_round_trips():
    """Zero-centred uint16 result shifted by 'baseline_mean' keeps input dtype and round-trips."""
    rng = np.random.default_rng(0)
    flatfield = np.linspace(0.6, 1.2, 64).reshape(8, 8)
    baseline = np.array([900.0, 1000.0, 1100.0])
    images = np.round(baseline[:, None, None] * flatfield + rng.normal(0, 30, (3, 8, 8)) + 100)
    images = images.astype(np.uint16)
    darkfield = np.full((8, 8), 100.0)

    centred = apply_basic_shading(images, flatfield, darkfield, baseline=baseline)
    assert centred.dtype == np.float32 and (centred < 0).mean() > 0.3

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        shifted, off = apply_basic_shading(
            images,
            flatfield,
            darkfield,
            baseline=baseline,
            offset="baseline_mean",
            return_offset=True,
        )
    assert shifted.dtype == np.uint16
    assert off == pytest.approx(1000.0)
    np.testing.assert_allclose(shifted.astype(np.float64) - off, centred, atol=0.5 + 1e-3)

    _, off_num = apply_basic_shading(
        images, flatfield, darkfield, baseline=baseline, offset=250, return_offset=True
    )
    assert off_num == 250.0
    _, off_none = apply_basic_shading(images, flatfield, darkfield, return_offset=True)
    assert off_none == 0.0


def test_apply_basic_shading_warns_when_integer_cast_clips():
    """Casting a zero-centred result to uint16 without an offset emits a clipping UserWarning."""
    rng = np.random.default_rng(1)
    flatfield = np.ones((16, 16))
    baseline = np.array([500.0, 500.0])
    images = (500 + rng.normal(0, 20, (2, 16, 16))).round().astype(np.uint16)
    with pytest.warns(UserWarning, match=r"clips .*% of pixels at 0"):
        clipped = apply_basic_shading(images, flatfield, baseline=baseline, output_dtype=np.uint16)
    assert 0.3 * 20 < clipped.mean() < 0.5 * 20


def test_apply_basic_shading_offset_errors():
    """offset='baseline_mean' requires a baseline, and unknown offset strings raise ValueError."""
    images = np.ones((2, 4, 4), np.uint16)
    with pytest.raises(ValueError, match="requires a baseline"):
        apply_basic_shading(images, np.ones((4, 4)), offset="baseline_mean")
    with pytest.raises(ValueError, match="offset must be"):
        apply_basic_shading(images, np.ones((4, 4)), baseline=np.zeros(2), offset="median")
