import unittest
import numpy as np
import pytest
import warnings
from eigenp_utils.clahe_equalize_adapthist import _my_clahe_

# _my_clahe_ is deprecated (use clahe_u8); its tests stay until removal, and
# only this expected warning is ignored for them
_IGNORE_MY_CLAHE_DEPRECATION = pytest.mark.filterwarnings(
    "ignore:_my_clahe_ is deprecated:DeprecationWarning"
)


def test_my_clahe_warns_deprecated():
    """_my_clahe_ emits a DeprecationWarning pointing to clahe_u8.

    Expected: one DeprecationWarning whose message names clahe_u8, attributed
    to the caller (stacklevel=2), so users see their own line in the warning.
    """
    image = np.random.default_rng(0).random((32, 32))
    with pytest.warns(DeprecationWarning, match="clahe_u8") as record:
        _my_clahe_(image, kernel_size=8)
    assert record[0].filename == __file__


@_IGNORE_MY_CLAHE_DEPRECATION
class TestCLAHECorrectness(unittest.TestCase):
    """
    Testr 🔎 Verification: CLAHE Correctness

    This test verifies the algorithmic intent and mathematical properties of
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Verified Properties:
    1. Monotonicity of Contrast: Increasing `clip_limit` (the contrast limit)
       must monotonically increase (or saturate) the entropy/contrast of the output.
       (Low limit -> Identity-like; High limit -> AHE-like).

    2. Identity Convergence: As `clip_limit` -> 0 (but > 0), the output should
       converge towards a linear rescaling of the input (preserving relative contrast),
       rather than the aggressive reshaping of Histogram Equalization.

    3. Monotone Zero Behavior: A `clip_limit` of 0 should correspond to
       Identity (Minimum Contrast), preserving continuity with limit->0.
    """

    def setUp(self):
        # Create a synthetic image: A low-contrast sine wave + noise
        # This gives CLAHE something to work with (expand contrast).
        # We use a float image [0, 1] range.
        np.random.seed(42)
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        xv, yv = np.meshgrid(x, y)

        # Base signal: Sine wave (approx range [0.4, 0.6])
        signal = 0.5 + 0.1 * np.sin(xv) * np.cos(yv)

        # Add noise to ensure histograms are not degenerate (avoiding artifacts)
        noise = np.random.normal(0, 0.01, signal.shape)

        self.img = np.clip(signal + noise, 0, 1)
        self.initial_std = np.std(self.img)

    def test_contrast_monotonicity(self):
        """
        Verify that increasing clip_limit increases output contrast (StdDev).
        """
        # We verify monotonicity across a range of limits.
        # Note: saturation usually occurs around 0.05-0.10 for smooth images,
        # as the 'clip count' becomes larger than any bin count.
        limits = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        stds = []

        # Run CLAHE with increasing limits
        for lim in limits:
            # We use a fixed kernel size to isolate clip_limit effect
            # Kernel 32x32 -> 1024 pixels.
            out = _my_clahe_(self.img, clip_limit=lim, kernel_size=(32, 32))
            stds.append(np.std(out))

        print("\nCLAHE Contrast Monotonicity Check:")
        for l, s in zip(limits, stds):
            print(f"  Limit={l:.4f}, StdDev={s:.4f}")

        # 1. Check Non-Decreasing Monotonicity
        for i in range(len(stds) - 1):
            self.assertGreaterEqual(stds[i+1], stds[i] - 1e-6,
                f"Contrast dropped between limit {limits[i]} and {limits[i+1]}")

        # 2. Check Significant Increase
        # The highest limit (AHE) should produce significantly more contrast than the lowest (Identity-like)
        # We use a relaxed 1.2x factor (20% increase) to be robust to image content.
        self.assertGreater(stds[-1], stds[0] * 1.2,
            "Max limit should produce significantly higher contrast than min limit")

    def test_identity_convergence(self):
        """
        Verify that very low clip_limit preserves the original signal shape (linear scaling).
        """
        # Very low limit -> Uniform Histogram -> Identity Mapping (Linear Rescaling)
        limit = 1e-5
        out = _my_clahe_(self.img, clip_limit=limit, kernel_size=(32, 32))

        # Compute correlation between input and output
        # If it's a linear transform, correlation should be 1.0
        flat_in = self.img.ravel()
        flat_out = out.ravel()

        corr = np.corrcoef(flat_in, flat_out)[0, 1]

        print(f"\nIdentity Convergence (Limit={limit}): Correlation={corr:.6f}")

        self.assertGreater(corr, 0.99,
            "Output with near-zero clip_limit should be highly correlated with input (Linear Transform).")

    def test_zero_behavior(self):
        """
        Verify that clip_limit=0 produces Minimum Contrast (Identity),
        and clip_limit=1.0/None produces Maximum Contrast (AHE).
        """
        # Run with limit=0 (Identity)
        out_zero = _my_clahe_(self.img, clip_limit=0, kernel_size=(32, 32))
        std_zero = np.std(out_zero)

        # Run with limit=1.0 (Max Contrast explicitly)
        out_max = _my_clahe_(self.img, clip_limit=1.0, kernel_size=(32, 32))
        std_max = np.std(out_max)

        # Run with limit=None (Default/Max)
        out_none = _my_clahe_(self.img, clip_limit=None, kernel_size=(32, 32))
        std_none = np.std(out_none)

        print(f"\nZero Check: Std(0)={std_zero:.4f}, Std(1.0)={std_max:.4f}, Std(None)={std_none:.4f}")

        # Std(0) should be strictly less than Std(1.0)
        self.assertLess(std_zero, std_max,
            msg="clip_limit=0 (Identity) should have lower contrast than AHE.")

        # Std(1.0) should equal Std(None)
        self.assertAlmostEqual(std_max, std_none, delta=1e-5,
            msg="clip_limit=None should be equivalent to clip_limit=1.0 (AHE)")

        # Zero should be close to very small limit
        out_small = _my_clahe_(self.img, clip_limit=1e-5, kernel_size=(32, 32))
        std_small = np.std(out_small)
        self.assertAlmostEqual(std_zero, std_small, delta=1e-2,
            msg="clip_limit=0 should be continuous with small limits.")

if __name__ == '__main__':
    unittest.main()


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
    import pytest
    skd = pytest.importorskip("skimage.data")
    from skimage.exposure import equalize_adapthist

    for name in ("camera", "coins"):
        image = getattr(skd, name)()
        ours = _my_clahe_(image, clip_limit=0.01)
        ref = equalize_adapthist(image, clip_limit=0.01)
        assert np.corrcoef(ours.ravel(), ref.ravel())[0, 1] > 0.999, name
        assert abs(ours.mean() - ref.mean()) < 0.01, name


@_IGNORE_MY_CLAHE_DEPRECATION
def test_clahe_rejects_kernel_size_of_wrong_length():
    """A kernel_size whose length differs from image.ndim raises a clear ValueError.

    The check built the exception without raising it, so the call failed later
    with an unrelated broadcasting error. Expected: ValueError naming kernel_size.
    """
    import pytest
    image = np.random.default_rng(0).random((32, 32))
    with pytest.raises(ValueError, match="kernel_size"):
        _my_clahe_(image, kernel_size=(8, 8, 8))


# ---------------------------------------------------------------------------
# clahe_u8: fast 8-bit CLAHE
# ---------------------------------------------------------------------------

from eigenp_utils.clahe_equalize_adapthist import clahe_u8


def _camera():
    import pytest
    skd = pytest.importorskip("skimage.data")
    return skd.camera()  # 512 x 512 uint8


def test_clahe_u8_numpy_matches_opencv():
    """The NumPy backend reproduces OpenCV's CLAHE to within rounding.

    OpenCV is only a test oracle here (not a dependency). Both use
    origin-anchored tiles, a single clip/redistribute pass and bilinear
    interpolation of per-tile LUTs; clahe_u8 works in int16 fixed point and
    its clip_limit c equals OpenCV's clipLimit = 256 * c. Strategy: kernel
    sizes dividing the image (so OpenCV's tile size equals ours), three clip
    regimes (identity-like, default, AHE). Expected: max absolute difference
    <= 2 gray levels, mean < 0.5.
    """
    import pytest
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
    """The numba backend reproduces the NumPy backend exactly.

    The kernels mirror the NumPy arithmetic (float32 leading-axis LUT blend,
    half-even rounding to int16, 7-bit fixed-point lerps; histograms with
    reflect-completed trailing tiles done by index reflection instead of
    padding). Strategy: 1D, 2D, 3D, non-divisible shapes, anisotropic tiles,
    three clip regimes; a thin stack also exercises the row-parallel kernel.
    Expected: identical arrays.
    """
    import pytest
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
    """backend='auto' works when numba is unavailable (e.g. Pyodide).

    Strategy: make the numba loader report "not importable". Expected: 'auto'
    returns the NumPy result; an explicit backend='numba' raises ImportError.
    """
    import pytest
    import eigenp_utils.clahe_equalize_adapthist as mod
    monkeypatch.setattr(mod, "_numba_kernels", lambda: None)
    image = _camera()
    np.testing.assert_array_equal(
        clahe_u8(image, kernel_size=64), clahe_u8(image, kernel_size=64, backend="numpy")
    )
    with pytest.raises(ImportError, match="numba"):
        clahe_u8(image, backend="numba")


def test_clahe_u8_matches_skimage_reference():
    """clahe_u8 agrees with skimage.exposure.equalize_adapthist on a natural image.

    The differences (tile anchoring, single-pass redistribution, 8-bit LUTs)
    are approximation choices, not a different algorithm. Expected: output
    correlation > 0.99 and means within 3 gray levels.
    """
    from skimage.exposure import equalize_adapthist
    image = _camera()
    ours = clahe_u8(image, clip_limit=0.01, backend="numpy").astype(float)
    ref = equalize_adapthist(image, clip_limit=0.01) * 255
    assert np.corrcoef(ours.ravel(), ref.ravel())[0, 1] > 0.99
    assert abs(ours.mean() - ref.mean()) < 3


def test_clahe_u8_axis_permutation_equivariance():
    """The N-D path treats every axis alike: permuting axes commutes with CLAHE.

    The implementation folds leading axes into a per-slice LUT and interpolates
    the last two axes per pixel, so a bug in either half breaks equivariance.
    Strategy: a random smooth 3D volume with non-divisible shape and anisotropic
    kernel; compare clahe(transpose(v)) with transpose(clahe(v)). Expected: equal
    to within 2 gray levels (int16 rounding happens in a different order).
    """
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
    """A volume constant along Z with one Z tile equals 2D CLAHE of a slice.

    With kernel_size[0] == Z there is a single contextual region along Z, whose
    histogram is the slice histogram times Z, so the unclipped (AHE) LUT is the
    2D one. This isolates the leading-axis LUT folding. (With clipping, the
    integer clip count and redistribution remainder do not scale exactly with
    Z, so only AHE is compared.) Expected: exact equality.
    """
    image = _camera()[::2, ::2]
    vol = np.repeat(image[None], 6, axis=0)
    out3 = clahe_u8(vol, kernel_size=(6, 32, 32), clip_limit=None, backend="numpy")
    out2 = clahe_u8(image, kernel_size=(32, 32), clip_limit=None, backend="numpy")
    for z in range(6):
        np.testing.assert_array_equal(out3[z], out2)


def test_clahe_u8_percentile_scaling_ignores_hot_pixel():
    """Non-uint8 input is scaled between percentiles, so one hot pixel is harmless.

    The extra sample only moves the 99.9th percentile by one order statistic.
    Negative control: min/max scaling (percentiles=(0, 100)) lets the 1e6
    pixel map everything else to gray level 0. Expected: 8-bit conversions
    differ by <= 1 level away from the hot pixel; CLAHE (which amplifies
    local contrast) changes the far corner by < 1.5 levels on average, versus
    > 20 levels under min/max scaling.
    """
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
    """16/8-bit integer input goes through a histogram + LUT path; check it.

    Signed types are histogrammed through a two's-complement view, which is
    easy to get wrong by the offset. Reference: percentiles (method='lower')
    and linear scaling computed in float64. Expected: <= 1 gray level
    difference (round-half-even ties) for uint16, int16 and int8.
    """
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
    """Low clip ~ identity; degenerate shapes and constant images do not fail.

    Expected: clip_limit=0 output is highly correlated with the input
    (r > 0.99); a constant image gives a constant image; 1D input, a kernel
    larger than the image and single-pixel axes return the input shape.
    """
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
    """Wrong kernel_size length, unknown backend and >3D numba raise ValueError.

    Also checks that 'auto' handles 4D input (NumPy path) instead of failing.
    """
    import pytest
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
