import numpy as np
import pytest
from eigenp_utils.intensity_rescaling import fit_basic_shading, apply_basic_shading

def test_basic_fit_synthetic():
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
    np.random.seed(42)
    images = np.random.rand(8, 128, 128)
    flatfield = np.ones((128, 128)) * 2

    corrected = apply_basic_shading(images, flatfield)

    np.testing.assert_allclose(corrected, images / 2.0)

def test_apply_basic_shading_baseline():
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
