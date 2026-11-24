"""Tests for ``eigenp_utils.clahe_equalize_adapthist`` (``clahe_u8`` and deprecated ``_my_clahe_``)."""
import numpy as np
import pytest

from eigenp_utils.clahe_equalize_adapthist import (
    _my_clahe_,
    clahe_u8,
    clip_histogram,
    map_histogram,
)

# _my_clahe_ is deprecated (use clahe_u8); its tests stay until removal, and
# only this expected warning is ignored for them
_IGNORE_MY_CLAHE_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:_my_clahe_ is deprecated:DeprecationWarning"
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


def _camera():
    """Return scikit-image's 512x512 uint8 camera reference image."""
    skd = pytest.importorskip("skimage.data")
    return skd.camera()


# ---------------------------------------------------------------------------
# Deprecated _my_clahe_ & histogram primitives
# ---------------------------------------------------------------------------


def test_my_clahe_warns_deprecated():
    """Verify ``_my_clahe_`` emits a ``DeprecationWarning`` pointing to ``clahe_u8`` at the caller line."""
    image = np.random.default_rng(0).random((32, 32))
    with pytest.warns(DeprecationWarning, match="clahe_u8") as record:
        _my_clahe_(image, kernel_size=8)
    assert record[0].filename == __file__


@_IGNORE_MY_CLAHE_DEPRECATION
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


@_IGNORE_MY_CLAHE_DEPRECATION
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


@_IGNORE_MY_CLAHE_DEPRECATION
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


@_IGNORE_MY_CLAHE_DEPRECATION
def test_clahe_rejects_kernel_size_of_wrong_length():
    """A kernel_size whose length differs from image.ndim raises a clear ValueError.

    The check built the exception without raising it, so the call failed later
    with an unrelated broadcasting error. Expected: ValueError naming kernel_size.
    """
    image = np.random.default_rng(0).random((32, 32))
    with pytest.raises(ValueError, match="kernel_size"):
        _my_clahe_(image, kernel_size=(8, 8, 8))


@_IGNORE_MY_CLAHE_DEPRECATION
@pytest.mark.parametrize("kernel_size", [None, 8, (4, 8, 8)])
def test_clahe_3d_volume_matches_skimage(kernel_size):
    """3D (Z, Y, X) volume CLAHE agrees with skimage for None, scalar, and 3-tuple kernel_size."""
    pytest.importorskip("skimage.exposure")
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


@_IGNORE_MY_CLAHE_DEPRECATION
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


@_IGNORE_MY_CLAHE_DEPRECATION
def test_uint16_input_and_unit_range_invariant():
    """uint16 inputs are equalized into finite [0, 1] output matching skimage."""
    pytest.importorskip("skimage.exposure")
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


@_IGNORE_MY_CLAHE_DEPRECATION
@pytest.mark.parametrize("shape,kernel_size", [((5, 7), None), ((6, 6), (4, 4))])
def test_small_image_handling(shape, kernel_size):
    """Images smaller than 8 px along an axis run cleanly and return finite values in [0, 1]."""
    img = np.random.default_rng(11).random(shape)
    out = _my_clahe_(img, kernel_size=kernel_size, clip_limit=0.01)

    assert out.shape == shape
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0


@_IGNORE_MY_CLAHE_DEPRECATION
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
    already_valid = np.array([1, 4, 7, 3, 2, 0, 5, 6], dtype=int)
    res_valid = clip_histogram(already_valid.copy(), clip_limit=10)
    np.testing.assert_array_equal(res_valid, already_valid)

    spiky = np.array([24, 0, 0, 0, 0, 0, 0, 0], dtype=int)
    res_spiky = clip_histogram(spiky.copy(), clip_limit=10)
    assert np.all(res_spiky <= 10)
    assert res_spiky.sum() == spiky.sum()

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


# ---------------------------------------------------------------------------
# clahe_u8: fast 8-bit N-D CLAHE
# ---------------------------------------------------------------------------


def test_clahe_u8_numpy_matches_opencv():
    """The NumPy backend reproduces OpenCV's CLAHE to within rounding.

    OpenCV is only a test oracle here (not a dependency). Strategy: kernel sizes
    dividing the image, three clip regimes (identity-like, default, AHE).
    Expected: max absolute difference <= 2 gray levels, mean < 0.5.
    """
    cv2 = pytest.importorskip("cv2")
    image = _camera()
    for clip in (0.0, 0.01, None):
        for ks in ((64, 64), (128, 32)):
            a = clahe_u8(image, kernel_size=ks, clip_limit=clip, backend="numpy")
            cv_clip = 0.0 if clip is None else max(256 * clip, 1e-6)
            tiles = (image.shape[1] // ks[1], image.shape[0] // ks[0])
            b = cv2.createCLAHE(clipLimit=cv_clip, tileGridSize=tiles).apply(image)
            d = np.abs(a.astype(int) - b.astype(int))
            assert d.max() <= 2 and d.mean() < 0.5, (clip, ks, d.max(), d.mean())


def test_clahe_u8_numba_bit_identical_to_numpy():
    """The numba backend reproduces the NumPy backend bit-for-bit across 1D/2D/3D and clip regimes."""
    pytest.importorskip("numba")
    from scipy import ndimage

    rng = np.random.default_rng(4)
    vol = ndimage.gaussian_filter(rng.random((21, 37, 45)), 2)
    vol = (255 * (vol - vol.min()) / np.ptp(vol)).astype(np.uint8)
    image = _camera()
    cases = [
        (vol, (5, 9, 12)),        # 3D, every axis non-divisible
        (vol[:2], (2, 16, 16)),   # thin stack: row-parallel kernel
        (image, (64, 64)),        # 2D
        (image[:301, :257], 40),  # 2D, non-divisible
        (image[100], 16),         # 1D
    ]
    for arr, ks in cases:
        for clip in (0.0, 0.01, None):
            a = clahe_u8(arr, kernel_size=ks, clip_limit=clip, backend="numba")
            b = clahe_u8(arr, kernel_size=ks, clip_limit=clip, backend="numpy")
            np.testing.assert_array_equal(a, b, err_msg=f"{arr.shape} {ks} {clip}")


def test_clahe_u8_falls_back_to_numpy_without_numba(monkeypatch):
    """backend='auto' falls back cleanly to NumPy when numba is unavailable; backend='numba' raises."""
    import eigenp_utils.clahe_equalize_adapthist as mod

    monkeypatch.setattr(mod, "_numba_kernels", lambda: None)
    image = _camera()
    np.testing.assert_array_equal(
        clahe_u8(image, kernel_size=64), clahe_u8(image, kernel_size=64, backend="numpy")
    )
    with pytest.raises(ImportError, match="numba"):
        clahe_u8(image, backend="numba")


def test_clahe_u8_matches_skimage_reference():
    """clahe_u8 agrees with skimage.exposure.equalize_adapthist on a natural image (r > 0.99, mean diff < 3)."""
    pytest.importorskip("skimage.exposure")
    from skimage.exposure import equalize_adapthist

    image = _camera()
    ours = clahe_u8(image, clip_limit=0.01, backend="numpy").astype(float)
    ref = equalize_adapthist(image, clip_limit=0.01) * 255
    assert np.corrcoef(ours.ravel(), ref.ravel())[0, 1] > 0.99
    assert abs(ours.mean() - ref.mean()) < 3


def test_clahe_u8_axis_permutation_equivariance():
    """The N-D path treats every axis alike: permuting axes commutes with CLAHE to within 2 gray levels."""
    from scipy import ndimage

    rng = np.random.default_rng(1)
    vol = ndimage.gaussian_filter(rng.random((21, 37, 45)), 2)
    vol = (255 * (vol - vol.min()) / np.ptp(vol)).astype(np.uint8)
    ks = (5, 9, 12)
    out = clahe_u8(vol, kernel_size=ks)
    for perm in ((2, 0, 1), (1, 2, 0)):
        out_p = clahe_u8(vol.transpose(perm), kernel_size=tuple(ks[p] for p in perm))
        d = np.abs(out_p.astype(int) - out.transpose(perm).astype(int))
        assert d.max() <= 2, (perm, d.max())


def test_clahe_u8_z_constant_volume_equals_2d():
    """A volume constant along Z with one Z tile matches 2D AHE of a slice bit-for-bit."""
    image = _camera()[::2, ::2]
    vol = np.repeat(image[None], 6, axis=0)
    out3 = clahe_u8(vol, kernel_size=(6, 32, 32), clip_limit=None, backend="numpy")
    out2 = clahe_u8(image, kernel_size=(32, 32), clip_limit=None, backend="numpy")
    for z in range(6):
        np.testing.assert_array_equal(out3[z], out2)


def test_clahe_u8_percentile_scaling_ignores_hot_pixel():
    """Non-uint8 input is scaled between percentiles so a single 1e6 hot pixel is harmless."""
    from scipy import ndimage
    from eigenp_utils.clahe_equalize_adapthist import _to_uint8

    rng = np.random.default_rng(2)
    image = ndimage.gaussian_filter(rng.random((256, 256)), 3).astype(np.float32)
    hot = image.copy()
    hot[250, 250] = 1e6
    pct = (0.1, 99.9)
    d8 = np.abs(_to_uint8(image, pct).astype(int) - _to_uint8(hot, pct).astype(int))
    d8[250, 250] = 0
    assert d8.max() <= 1

    def corner_diff(percentiles):
        a = clahe_u8(image, kernel_size=32, percentiles=percentiles, backend="numpy")
        b = clahe_u8(hot, kernel_size=32, percentiles=percentiles, backend="numpy")
        assert int(a.max()) - int(a.min()) > 200
        return np.abs(a[:128, :128].astype(int) - b[:128, :128].astype(int)).mean()

    assert corner_diff(pct) < 1.5
    assert corner_diff((0, 100)) > 20


def test_clahe_u8_small_int_conversion_matches_float_reference():
    """16/8-bit integer histogram+LUT conversion matches float64 percentiles within 1 gray level."""
    from eigenp_utils.clahe_equalize_adapthist import _to_uint8

    rng = np.random.default_rng(3)
    for dt in (np.uint16, np.int16, np.int8):
        info = np.iinfo(dt)
        a = rng.integers(info.min, info.max, (300, 301)).astype(dt)
        lo, hi = np.percentile(a.astype(float), (0.1, 99.9), method="lower")
        ref = np.rint(np.clip((a.astype(float) - lo) * 255 / (hi - lo), 0, 255))
        out = _to_uint8(a, (0.1, 99.9))
        assert out.dtype == np.uint8
        assert np.abs(out.astype(int) - ref.astype(int)).max() <= 1, dt


def test_clahe_u8_identity_and_edge_cases():
    """Low clip ~ identity; constant images, 1D, oversized kernels, and size-1 axes do not fail."""
    image = _camera()
    out0 = clahe_u8(image, clip_limit=0.0, backend="numpy")
    assert np.corrcoef(out0.ravel().astype(float), image.ravel())[0, 1] > 0.99

    const = np.full((40, 30), 77, np.uint8)
    out = clahe_u8(const, kernel_size=8)
    assert out.shape == const.shape and np.ptp(out) == 0

    for arr, ks in ((image[100], 16), (image[:7, :5], 64), (image[None, :1, :50], None)):
        res = clahe_u8(arr, kernel_size=ks)
        assert res.shape == arr.shape and res.dtype == np.uint8


def test_clahe_u8_rejects_bad_arguments():
    """Wrong kernel_size length, unknown backend and >3D numba raise ValueError; 4D 'auto' succeeds."""
    image = np.zeros((16, 16), np.uint8)
    with pytest.raises(ValueError, match="kernel_size"):
        clahe_u8(image, kernel_size=(4, 4, 4))
    with pytest.raises(ValueError, match="backend"):
        clahe_u8(image, backend="opencv")
    vol4 = np.random.default_rng(5).integers(0, 255, (3, 4, 16, 16), dtype=np.uint8)
    assert clahe_u8(vol4, kernel_size=(2, 2, 8, 8)).shape == vol4.shape
    pytest.importorskip("numba")
    with pytest.raises(ValueError, match="3 dimensions"):
        clahe_u8(vol4, backend="numba")
