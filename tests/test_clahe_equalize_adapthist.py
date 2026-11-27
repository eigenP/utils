"""Tests for ``eigenp_utils.clahe_equalize_adapthist`` (``clahe_u8`` and deprecated ``_my_clahe_``)."""
import sys

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
    "ignore:_my_clahe_ is deprecated:FutureWarning"
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
    """Verify ``_my_clahe_`` emits a ``FutureWarning`` naming the removal version and ``clahe_u8``, at the caller line.

    Changed (D08): it was a DeprecationWarning, which Python hides when the
    warning is attributed to library code.
    """
    image = np.random.default_rng(0).random((32, 32))
    with pytest.warns(FutureWarning, match=r"removed in eigenp_utils 0\.1\.0.*clahe_u8") as record:
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
    """With ``tile_centres='opencv'`` the NumPy backend reproduces OpenCV's CLAHE to within rounding.

    OpenCV is only a test oracle here (not a dependency). Strategy: kernel sizes
    dividing the image, three clip regimes (identity-like, default, AHE).
    Expected: max absolute difference <= 2 gray levels, mean < 0.5.
    """
    cv2 = pytest.importorskip("cv2")
    image = _camera()
    for clip in (0.0, 0.01, None):
        for ks in ((64, 64), (128, 32)):
            a = clahe_u8(image, kernel_size=ks, clip_limit=clip, backend="numpy", tile_centres="opencv")
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
            for tc in ("pixel", "opencv"):
                a = clahe_u8(arr, kernel_size=ks, clip_limit=clip, backend="numba", tile_centres=tc)
                b = clahe_u8(arr, kernel_size=ks, clip_limit=clip, backend="numpy", tile_centres=tc)
                np.testing.assert_array_equal(a, b, err_msg=f"{arr.shape} {ks} {clip} {tc}")


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


@pytest.mark.parametrize("tile_centres", ["pixel", "opencv"])
def test_clahe_u8_axis_permutation_equivariance(tile_centres):
    """Permuting axes commutes with CLAHE to within 2 grey levels (fixed-point rounding).

    Every axis uses the same tile-centre convention, so no axis is special.
    (An interim D20 version centred only the leading axes; permutations that
    moved an axis into the leading position then differed by up to 12 levels.)
    Expected outcome: max |diff| <= 2 for every permutation, both conventions.
    """
    from scipy import ndimage

    rng = np.random.default_rng(1)
    vol = ndimage.gaussian_filter(rng.random((21, 37, 45)), 2)
    vol = (255 * (vol - vol.min()) / np.ptp(vol)).astype(np.uint8)
    ks = (5, 9, 12)
    out = clahe_u8(vol, kernel_size=ks, tile_centres=tile_centres)
    for perm in ((0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)):
        out_p = clahe_u8(vol.transpose(perm), kernel_size=tuple(ks[p] for p in perm),
                         tile_centres=tile_centres)
        d = np.abs(out_p.astype(int) - out.transpose(perm).astype(int))
        assert d.max() <= 2, (perm, d.max(), d.mean())


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


def test_clahe_u8_sparse_signal_and_column_patterns():
    """Verify sparse signal is not blanked (R6-02) and the percentile sample does not alias with columns (R6-13).

    Strategy: (+) 15 bright spots on zeros in float32 and uint16 (degenerate
    (0.1, 99.9) range); (+) a 64x512x512 float volume whose every 4th column is
    offset by -50 (a strided sample with step 4 sees one column phase only);
    (-) dense data keep the previous mapping (no warning).
    Expected outcome: spots map to 255 with a UserWarning instead of 0; the
    sampled percentiles match the exact ones within 0.5% of the range.
    """
    from eigenp_utils.clahe_equalize_adapthist import _to_uint8

    img = np.zeros((200, 200), np.float32)
    img[::50, ::15][:3, :5] = 1000.0
    for arr in (img, img.astype(np.uint16)):
        with pytest.warns(UserWarning, match="percentiles"):
            out = _to_uint8(arr, (0.1, 99.9))
        assert out.max() == 255 and out[1, 1] == 0

    rng = np.random.default_rng(0)
    vol = rng.normal(300, 20, (64, 512, 512)).astype(np.float32)
    vol[..., ::4] -= 50
    exact = np.percentile(vol, (0.1, 99.9))
    out = _to_uint8(vol, (0.1, 99.9))
    # the value that maps to grey level 0 / 255 is within 0.5% of the exact percentile range
    span = exact[1] - exact[0]
    assert abs(vol[out == 0].max() - exact[0]) < 0.005 * span
    assert abs(vol[out == 255].min() - exact[1]) < 0.005 * span


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


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_clahe_u8_empty_input_returns_empty_uint8(backend):
    """Empty arrays (e.g. empty tiles of a tiled or dask pipeline) pass through instead of crashing.

    Strategy: run ``clahe_u8`` on 1D/2D/3D arrays with a size-0 axis, in uint8,
    uint16 and float32, on both backends. A size-0 axis made the tile size 0,
    which raised ZeroDivisionError in the tile grid; ``median_filter_fast``
    already returns empty input unchanged.
    Expected outcome: an empty uint8 array of the input shape; a kernel_size of
    the wrong length is still rejected. Negative control: a non-empty image is
    still equalised (non-constant output, identical on both backends).
    """
    if backend == "numba":
        pytest.importorskip("numba")
    for shape in ((0,), (0, 16), (16, 0), (0, 16, 16), (4, 0, 16)):
        for dtype in (np.uint8, np.uint16, np.float32):
            out = clahe_u8(np.zeros(shape, dtype), backend=backend)
            assert out.shape == shape and out.dtype == np.uint8, (shape, dtype)
    with pytest.raises(ValueError, match="kernel_size"):
        clahe_u8(np.zeros((0, 16), np.uint8), kernel_size=(4, 4, 4), backend=backend)

    image = np.random.default_rng(3).integers(0, 255, (24, 32), dtype=np.uint8)
    out = clahe_u8(image, kernel_size=8, backend=backend)
    assert out.shape == image.shape and np.ptp(out) > 0
    np.testing.assert_array_equal(out, clahe_u8(image, kernel_size=8, backend="numpy"))


@pytest.mark.skipif(sys.platform == "emscripten", reason="no subprocesses in Pyodide")
def test_clahe_module_runs_standalone_with_sibling_conventions(tmp_path):
    """The module keeps its PEP 723 standalone use (D09-A): copied next to
    ``_conventions.py`` and loaded by file path with ``eigenp_utils`` not
    importable, ``clahe_u8`` still works."""
    import shutil
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    import eigenp_utils.clahe_equalize_adapthist as mod

    src = Path(mod.__file__).parent
    for name in ("clahe_equalize_adapthist.py", "_conventions.py"):
        shutil.copy(src / name, tmp_path / name)
    code = textwrap.dedent(f"""
        import sys, runpy, importlib.abc
        import numpy as np
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "eigenp_utils" or name.startswith("eigenp_utils."):
                    raise ModuleNotFoundError(name)
        sys.meta_path.insert(0, Block())
        ns = runpy.run_path({str(tmp_path / 'clahe_equalize_adapthist.py')!r})
        out = ns["clahe_u8"](np.random.default_rng(0).random((64, 64)).astype("float32"))
        assert out.dtype == np.uint8
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"MPLBACKEND": "Agg", "PATH": ""})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


# ---------------------------------------------------------------------------
# D20: pixel-centred tile coordinates on the folded (leading) axes
# ---------------------------------------------------------------------------


def _d20_backends():
    """Backends to check: NumPy always, numba when importable."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return ["numpy"]
    return ["numpy", "numba"]


def _d20_image2d():
    """Deterministic 97x131 uint8 ramp with pseudo-random texture (no RNG, so the hash is portable)."""
    yy, xx = np.mgrid[:97, :131].astype(np.int64)
    noise = (xx * 7919 + yy * 104729 + xx * yy * 31) % 41 - 20
    return np.clip(40 + xx + yy // 2 + noise, 0, 255).astype(np.uint8)


# sha256 of the base-commit (33bb6ff) 2D output; identical on both backends
_D20_2D_HASHES = {
    (None, 0.01): "b0aa4a2ac807e92864677290102b18528d80187da55f6a89743d1efae507ff97",
    ((16, 20), 0.03): "d8ad57b1f04a3a68092314bdc11334eed94941973718b1bd36983c5f418071bc",
    (7, None): "a19f7868a670bf2bd0e1eaa283835032bacba4539aa754b2db32d122ab4c7386",
}


@pytest.mark.parametrize("backend", _d20_backends())
def test_clahe_u8_opencv_tile_centres_reproduce_base_2d(backend):
    """(-) control for D20: ``tile_centres='opencv'`` reproduces the previous 2D output, bit for bit.

    Strategy: equalise a deterministic 2D image with three kernel/clip settings
    and hash the output. Expected outcome: the sha256 equals the output of the
    base commit (before D20), on both backends.
    """
    import hashlib

    img = _d20_image2d()
    for (ks, clip), digest in _D20_2D_HASHES.items():
        out = clahe_u8(img, kernel_size=ks, clip_limit=clip, backend=backend, tile_centres="opencv")
        assert hashlib.sha256(out.tobytes()).hexdigest() == digest, (ks, clip)


# sha256 of clahe_u8 on _d20_volume() at the pre-D20 base commit (33bb6ff),
# where every axis used OpenCV's convention (numpy and numba agree)
_D20_3D_BASE_HASHES = {
    (None, 0.01): "7d902f7304f7fbb1186516b6f8a84a422930cea2fec247d7db482d4287571428",
    ((2, 12, 10), 0.01): "11bba981b5f0bca8bd9f40293fa6b0fee4fa0c18276b4bfaabf60da90a3a5b31",
    ((5, 16, 8), None): "001ee89d4c05a9a77ea960251e09ac71b6c328ea970a8becdabc7a59dc7a0f4b",
    ((1, 16, 16), 0.0): "cc0d809f00199522343370f087cb55b189a0c303e5b427eb70e8201df9133b2a",
}


def _d20_volume():
    """The smooth random 10x48x40 uint8 volume of the Z-flip test."""
    from scipy import ndimage

    rng = np.random.default_rng(7)
    vol = ndimage.gaussian_filter(rng.random((10, 48, 40)), (0.5, 3, 3))
    return (255 * (vol - vol.min()) / np.ptp(vol)).astype(np.uint8)


@pytest.mark.parametrize("backend", _d20_backends())
def test_clahe_u8_opencv_tile_centres_reproduce_base_3d(backend):
    """(-) control for D20: ``tile_centres='opencv'`` is the previous N-D behaviour, bit for bit.

    Expected outcome: the sha256 of four kernel/clip settings equals the output
    of the pre-D20 base commit, on both backends.
    """
    import hashlib

    vol = _d20_volume()
    for (ks, clip), digest in _D20_3D_BASE_HASHES.items():
        out = clahe_u8(vol, kernel_size=ks, clip_limit=clip, backend=backend, tile_centres="opencv")
        assert hashlib.sha256(out.tobytes()).hexdigest() == digest, (ks, clip)


@pytest.mark.parametrize("backend", _d20_backends())
def test_clahe_u8_pixel_tile_centres_commute_with_flips_on_every_axis(backend):
    """(+) control for D20: with pixel-centred tiles, flipping Y or X (not only Z) commutes with CLAHE.

    OpenCV's half-pixel shift makes even its own 2D CLAHE direction-dependent;
    the pixel convention does not. Expected outcome: ||C(flip x) - flip C(x)||_inf
    == 0 on every axis whose length is a multiple of the tile, and > 0 on Y or X
    with ``tile_centres='opencv'`` (negative control).
    """
    vol = _d20_volume()
    ks = (5, 16, 8)
    for axis in range(3):
        flip = [slice(None)] * 3
        flip[axis] = slice(None, None, -1)
        flip = tuple(flip)
        a = clahe_u8(vol[flip], kernel_size=ks, backend=backend)[flip]
        b = clahe_u8(vol, kernel_size=ks, backend=backend)
        assert np.abs(a.astype(int) - b.astype(int)).max() == 0, axis
    a = clahe_u8(vol[:, :, ::-1], kernel_size=ks, backend=backend, tile_centres="opencv")[:, :, ::-1]
    b = clahe_u8(vol, kernel_size=ks, backend=backend, tile_centres="opencv")
    assert np.abs(a.astype(int) - b.astype(int)).max() > 0


def test_clahe_u8_pixel_tile_centres_stay_close_to_opencv_in_2d():
    """D20 bound: the default 2D output moves only slightly from the OpenCV convention.

    Strategy: camera-like texture at default and explicit tile sizes. Expected
    outcome: at the default tile (1/8 of each dim) max |diff| <= 4 and mean < 0.6
    grey levels versus ``tile_centres='opencv'``; the change grows for small tiles.
    """
    img = _camera()
    for clip in (0.01, None):
        a = clahe_u8(img, clip_limit=clip).astype(int)
        b = clahe_u8(img, clip_limit=clip, tile_centres="opencv").astype(int)
        d = np.abs(a - b)
        assert d.max() <= 4 and d.mean() < 0.6, (clip, d.max(), d.mean())
        assert d.max() > 0  # the conventions do differ


def test_clahe_u8_rejects_unknown_tile_centres():
    """``tile_centres`` accepts only 'pixel' and 'opencv'."""
    with pytest.raises(ValueError, match="tile_centres"):
        clahe_u8(_camera(), tile_centres="center")


@pytest.mark.parametrize("backend", _d20_backends())
def test_clahe_u8_z_flip_equivariance(backend):
    """(+) control for D20: flipping Z commutes with CLAHE when Z is a multiple of the Z tile.

    Strategy: a smooth random 10x48x40 volume (Z < 16, so the default Z tile is
    1 voxel), and explicit Z tiles of 2 and 5. Before D20 the half-pixel OpenCV
    shift on Z made slice i a 50/50 blend of the LUTs of slices i-1 and i, and
    flipping Z changed the output by up to 64 grey levels.
    Expected outcome: ||C(flip_z x) - flip_z C(x)||_inf == 0.
    """
    from scipy import ndimage

    rng = np.random.default_rng(7)
    vol = ndimage.gaussian_filter(rng.random((10, 48, 40)), (0.5, 3, 3))
    vol = (255 * (vol - vol.min()) / np.ptp(vol)).astype(np.uint8)
    for ks in (None, (2, 12, 10), (5, 16, 8)):
        a = clahe_u8(vol[::-1], kernel_size=ks, backend=backend)
        b = clahe_u8(vol, kernel_size=ks, backend=backend)[::-1]
        assert np.abs(a.astype(int) - b.astype(int)).max() == 0, ks


@pytest.mark.parametrize("backend", _d20_backends())
def test_clahe_u8_unit_z_tiles_keep_slices_independent(backend):
    """(+) control for D20: with 1-voxel Z tiles, each slice is equalised on its own.

    Strategy: a (C, Y, X) stack with kernel_size (1, 16, 16); perturb slice j
    only. Expected outcome: every other slice i is unchanged, and each slice
    equals the 2D CLAHE of that slice alone (no LUT leaks between channels).
    """
    rng = np.random.default_rng(8)
    stack = rng.integers(0, 256, (4, 64, 48), dtype=np.uint8)
    ks = (1, 16, 16)
    base = clahe_u8(stack, kernel_size=ks, backend=backend)
    pert = stack.copy()
    pert[2] = 255 - pert[2]
    out = clahe_u8(pert, kernel_size=ks, backend=backend)
    for i in (0, 1, 3):
        np.testing.assert_array_equal(out[i], base[i], err_msg=f"slice {i}")
    assert np.any(out[2] != base[2])
    for i in range(4):
        np.testing.assert_array_equal(
            base[i], clahe_u8(stack[i], kernel_size=ks[1:], backend=backend), err_msg=f"slice {i}"
        )


def test_clahe_u8_warns_on_trailing_colour_axis():
    """C-07: a trailing axis of length 3 or 4 is tiled as if it were spatial, so clahe_u8 warns.

    Strategy: a flat pure-red (32, 32, 3) uint8 image (channel-last RGB, as
    ``color_coded_projection`` returns). Expected outcome: a UserWarning that
    names the channel axis and points to channel-first processing. Negative
    control: a (3, 32, 32) channel-first stack and a (32, 32) image do not warn.
    """
    import warnings

    rgb = np.zeros((32, 32, 3), np.uint8)
    rgb[..., 0] = 255
    with pytest.warns(UserWarning, match="trailing axis of length 3"):
        clahe_u8(rgb, backend="numpy")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clahe_u8(np.moveaxis(rgb, -1, 0), kernel_size=(1, 8, 8), backend="numpy")
        clahe_u8(rgb[..., 0], backend="numpy")


def test_clahe_u8_nan_policy():
    """R6-15: NaN input maps to 0 with a UserWarning instead of silently becoming black.

    Strategy: a float ramp with a NaN block. Expected outcome: a UserWarning
    that counts the NaN pixels, and those pixels are equalised from grey level
    0 (identical to the same image with the block set to the data minimum).
    Negative control: finite float input does not warn.
    """
    import warnings

    img = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
    nan_img = img.copy()
    nan_img[:8, :8] = np.nan
    with pytest.warns(UserWarning, match="64 NaN"):
        out = clahe_u8(nan_img, kernel_size=16, backend="numpy")
    ref_img = img.copy()
    ref_img[:8, :8] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = clahe_u8(ref_img, kernel_size=16, backend="numpy")
    np.testing.assert_array_equal(out, ref)
