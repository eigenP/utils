import numpy as np
import pytest

from eigenp_utils.clahe_equalize_adapthist import (
    _my_clahe_,
    clip_histogram,
    map_histogram,
)


def _low_contrast_sine_image(shape=(128, 128), seed=42):
    """Low-contrast 2D sine pattern in [0, 1] with small Gaussian noise."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 4 * np.pi, shape[0])
    x = np.linspace(0, 4 * np.pi, shape[1])
    yv, xv = np.meshgrid(y, x, indexing="ij")
    signal = 0.5 + 0.1 * np.sin(xv) * np.cos(yv)
    noise = rng.normal(0.0, 0.01, shape)
    return np.clip(signal + noise, 0.0, 1.0)


def test_clahe_matches_skimage_reference():
    """_my_clahe_ agrees with skimage.exposure.equalize_adapthist on natural images.

    A port error set bin_size = 1 + nbins // nbins == 2, so only half the
    histogram bins were used: on skimage's camera image the output mean was
    0.376 against skimage's 0.515, with compressed highlights. Strategy: run both
    implementations with the same clip limit on two reference images. Expected:
    near-identical outputs (correlation > 0.999, means within 0.01); the small
    residual comes from skimage quantising to 16384 gray levels internally.
    """
    skd = pytest.importorskip("skimage.data")
    from skimage.exposure import equalize_adapthist

    for name in ("camera", "coins"):
        image = getattr(skd, name)()
        ours = _my_clahe_(image, clip_limit=0.01)
        ref = equalize_adapthist(image, clip_limit=0.01)
        assert np.corrcoef(ours.ravel(), ref.ravel())[0, 1] > 0.999, name
        assert abs(ours.mean() - ref.mean()) < 0.01, name


def test_contrast_monotonicity():
    """Increasing clip_limit monotonically increases output contrast (StdDev) up to AHE saturation."""
    img = _low_contrast_sine_image()
    limits = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    stds = [
        float(np.std(_my_clahe_(img, clip_limit=lim, kernel_size=(32, 32)).astype(np.float64)))
        for lim in limits
    ]

    for i in range(len(stds) - 1):
        assert stds[i + 1] >= stds[i] - 1e-6, (
            f"Contrast dropped between limit {limits[i]} and {limits[i + 1]}: "
            f"{stds[i]:.6f} -> {stds[i + 1]:.6f}"
        )

    assert stds[-1] > stds[0] * 1.2, (
        "Max limit (AHE) should produce significantly higher contrast than min limit."
    )


def test_clip_limit_boundary_invariants():
    """Verify exact boundary behavior at clip_limit=0, near-zero, 1.0, and None.

    Invariants:
    - clip_limit=None is bit-for-bit identical to clip_limit=1.0 (unclipped AHE).
    - clip_limit=0 is bit-for-bit identical to clip_limit=1e-5 for a 32x32 kernel
      (both clamp clim to 1) and preserves strong linear correlation (> 0.99) with input.
    - clip_limit=0 produces strictly lower contrast than clip_limit=1.0.
    """
    img = _low_contrast_sine_image()
    out_zero = _my_clahe_(img, clip_limit=0, kernel_size=(32, 32))
    out_tiny = _my_clahe_(img, clip_limit=1e-5, kernel_size=(32, 32))
    out_max = _my_clahe_(img, clip_limit=1.0, kernel_size=(32, 32))
    out_none = _my_clahe_(img, clip_limit=None, kernel_size=(32, 32))

    np.testing.assert_array_equal(out_none, out_max)
    np.testing.assert_array_equal(out_zero, out_tiny)

    corr = np.corrcoef(img.ravel(), out_zero.astype(np.float64).ravel())[0, 1]
    assert corr > 0.99, f"Expected near-linear correlation for clip_limit=0, got {corr:.6f}"
    assert np.std(out_zero.astype(np.float64)) < np.std(out_max.astype(np.float64))


def test_clahe_rejects_kernel_size_of_wrong_length():
    """A kernel_size whose length differs from image.ndim raises a clear ValueError.

    The check built the exception without raising it, so the call failed later
    with an unrelated broadcasting error. Expected: ValueError naming kernel_size.
    """
    image = np.random.default_rng(0).random((32, 32))
    with pytest.raises(ValueError, match="kernel_size"):
        _my_clahe_(image, kernel_size=(8, 8, 8))


@pytest.mark.parametrize("kernel_size", [None, 8, (4, 8, 8)])
def test_clahe_3d_volume_matches_skimage(kernel_size):
    """3D (Z, Y, X) volume CLAHE agrees with skimage for None, scalar, and 3-tuple kernel_size."""
    from skimage.exposure import equalize_adapthist

    rng = np.random.default_rng(7)
    z = np.linspace(0, 2 * np.pi, 24)
    y = np.linspace(0, 3 * np.pi, 32)
    x = np.linspace(0, 4 * np.pi, 32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    volume = np.clip(
        0.5 + 0.12 * np.sin(zz) + 0.1 * np.cos(yy) * np.sin(xx) + rng.normal(0, 0.02, zz.shape),
        0.0,
        1.0,
    )

    ours = _my_clahe_(volume, kernel_size=kernel_size, clip_limit=0.05)
    ref = equalize_adapthist(volume, kernel_size=kernel_size, clip_limit=0.05)

    assert ours.shape == volume.shape
    assert np.corrcoef(ours.astype(np.float64).ravel(), ref.ravel())[0, 1] > 0.99
    assert abs(float(ours.astype(np.float64).mean()) - float(ref.mean())) < 0.02


def test_output_float_dtype_options():
    """output_float_dtype defaults to float16 and honors float32 and float64 consistently."""
    img = _low_contrast_sine_image(shape=(64, 64), seed=1)

    out_default = _my_clahe_(img, clip_limit=0.02, kernel_size=(16, 16))
    out_f32 = _my_clahe_(img, clip_limit=0.02, kernel_size=(16, 16), output_float_dtype=np.float32)
    out_f64 = _my_clahe_(img, clip_limit=0.02, kernel_size=(16, 16), output_float_dtype=np.float64)

    assert out_default.dtype == np.float16
    assert out_f32.dtype == np.float32
    assert out_f64.dtype == np.float64

    np.testing.assert_allclose(out_f32, out_f64, atol=1e-5)
    np.testing.assert_allclose(out_default.astype(np.float64), out_f64, atol=5e-3)


def test_uint16_input_and_unit_range_invariant():
    """uint16 inputs are equalized into finite [0, 1] output matching skimage."""
    from skimage.exposure import equalize_adapthist

    img_u16 = np.round(_low_contrast_sine_image(shape=(64, 64), seed=3) * 65535).astype(np.uint16)
    out = _my_clahe_(img_u16, clip_limit=0.02, kernel_size=(16, 16), output_float_dtype=np.float32)
    ref = equalize_adapthist(img_u16, clip_limit=0.02, kernel_size=(16, 16))

    assert out.shape == img_u16.shape
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isclose(out.min(), 0.0, atol=1e-5)
    assert np.isclose(out.max(), 1.0, atol=1e-5)
    assert np.corrcoef(out.ravel(), ref.ravel())[0, 1] > 0.99


@pytest.mark.parametrize("shape,kernel_size", [((5, 7), None), ((6, 6), (4, 4))])
def test_small_image_handling(shape, kernel_size):
    """Images smaller than 8 px along an axis run cleanly and return finite values in [0, 1]."""
    img = np.random.default_rng(11).random(shape)
    out = _my_clahe_(img, kernel_size=kernel_size, clip_limit=0.01)

    assert out.shape == shape
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("fill_value", [0.0, 0.5, 1.0])
def test_constant_image_produces_no_nan(fill_value):
    """Constant images produce finite, constant output in [0, 1] with no NaNs."""
    img = np.full((32, 32), fill_value, dtype=np.float64)
    out = _my_clahe_(img, clip_limit=0.01, kernel_size=(8, 8))

    assert out.shape == img.shape
    assert np.all(np.isfinite(out))
    assert np.all(out == out.flat[0])
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_clip_histogram_bounds_and_count_conservation():
    """clip_histogram caps every bin at <= clip_limit and conserves total pixel count."""
    # Case 1: no bin exceeds clip_limit -> histogram is unchanged
    already_valid = np.array([1, 4, 7, 3, 2, 0, 5, 6], dtype=int)
    res_valid = clip_histogram(already_valid.copy(), clip_limit=10)
    np.testing.assert_array_equal(res_valid, already_valid)

    # Case 2: excess redistributed evenly across empty bins
    spiky = np.array([24, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    res_spiky = clip_histogram(spiky.copy(), clip_limit=10)
    assert np.all(res_spiky <= 10)
    assert res_spiky.sum() == spiky.sum()

    # Case 3: multi-bin clipping exercising low_mask, mid_mask, and remainder loop
    skewed = np.array([50, 20, 15, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    res_skewed = clip_histogram(skewed.copy(), clip_limit=10)
    assert np.all(res_skewed <= 10)
    assert res_skewed.sum() == skewed.sum()


def test_map_histogram_monotonicity_and_bounds():
    """map_histogram produces a non-decreasing LUT bounded within [min_val, max_val]."""
    hists = np.array([
        [4] * 16,
        [32, 16, 8, 4, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 4, 8, 16, 32],
    ], dtype=int)

    lut = map_histogram(hists, min_val=10, max_val=240, n_pixels=64)

    assert lut.shape == hists.shape
    assert np.issubdtype(lut.dtype, np.integer)
    assert np.all(np.diff(lut, axis=-1) >= 0)
    assert lut.min() >= 10
    assert lut.max() <= 240
    np.testing.assert_array_equal(lut[:, -1], 240)
