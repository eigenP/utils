"""Tests for eigenp_utils.median_filter.median_filter_fast.

Oracle: scipy.ndimage.median_filter(size=(1, ..., k, k), mode='reflect'), an
exact median, for finite data; for NaN, np.median over the same reflected
windows (np.median propagates NaN). The comparator networks are checked
exhaustively with the 0-1 principle.
"""
import itertools
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy import ndimage

from eigenp_utils import median_filter as mf
from eigenp_utils.median_filter import median_filter_fast

BACKENDS = [
    "numpy",
    pytest.param("numba", marks=pytest.mark.skipif(
        mf._numba_kernels() is None, reason="numba not installed")),
]
DTYPES = [np.uint8, np.uint16, np.int16, np.float32, np.float64]


def _random(shape, dtype, rng, levels=None):
    """Random image; `levels` small -> many ties (the hard case for selection)."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        a = rng.standard_normal(shape) * 1e3
        if levels is not None:
            a = np.round(a / 1e3 * levels / 4)
        return a.astype(dtype)
    info = np.iinfo(dtype)
    hi = info.max if levels is None else min(info.max, info.min + levels)
    return rng.integers(info.min, hi, shape, endpoint=True).astype(dtype)


def _scipy(a, k, axes=(-2, -1)):
    size = [1] * a.ndim
    for ax in axes:
        size[ax] = k
    return ndimage.median_filter(a, size=size, mode="reflect")


def _bits(a):
    """Bit pattern view, so equality also checks signed zeros and NaN payloads."""
    return a.view(np.dtype(f"u{a.dtype.itemsize}"))


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [3, 5])
def test_networks_sort_columns_and_select_median(k):
    """The column and window comparator networks are correct (0-1 principle).

    A comparator network commutes with monotone maps, so it sorts (selects)
    correctly for all inputs iff it does for all 0-1 inputs; thresholding
    keeps columns sorted, so for the window network it suffices to test the
    0-1 inputs whose k columns are each sorted ((k + 1) ** k cases).
    Expected: the column network sorts all 2 ** k binary vectors; the window
    network leaves the median (k * k // 2-th smallest) on its output wire.
    """
    colsort, window, out_wire = mf._NETWORKS[k]

    vecs = np.array(list(itertools.product([False, True], repeat=k)))
    wires = mf._apply_network([vecs[:, i] for i in range(k)], colsort)
    np.testing.assert_array_equal(np.stack(wires, 1), np.sort(vecs, axis=1))

    # column c with n ones (at the top ranks): rank r is 1 iff r >= k - n
    ones = np.array(list(itertools.product(range(k + 1), repeat=k)))
    wires = [ones[:, c] >= k - r for c in range(k) for r in range(k)]
    expected = ones.sum(1) > k * k // 2   # median is 1 iff ones outnumber zeros
    got = mf._apply_network(wires, window)[out_wire]
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# Exactness against scipy
# ---------------------------------------------------------------------------

SHAPES = [(1, 1), (1, 9), (9, 1), (2, 2), (2, 3), (4, 4), (31, 17), (3, 13, 19),
          (2, 3, 8, 7)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_scipy_reflect(backend, dtype):
    """Bit-for-bit equal to scipy.ndimage.median_filter(mode='reflect').

    Strategy: odd shapes, 1-pixel-wide rows/columns, images smaller than the
    window (reflection past the far edge), 3D and 4D stacks (scipy size 1 on
    leading axes), each with full-range values and with 4 levels (ties), for
    k = 3, 5 (networks) and 7 (selection fallback), plus k = 1 (identity).
    Expected: identical arrays, input dtype kept.
    """
    rng = np.random.default_rng(0)
    for shape in SHAPES:
        for levels in (None, 4):
            a = _random(shape, dtype, rng, levels)
            for k in (1, 3, 5, 7):
                out = median_filter_fast(a, k, backend=backend)
                assert out.dtype == a.dtype and out.shape == a.shape
                np.testing.assert_array_equal(
                    out, _scipy(a, k), err_msg=f"{shape} k={k} levels={levels}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_constant_and_extreme_values(backend):
    """Constant images stay constant; dtype extremes and +-inf are ordered values.

    Expected: a constant image (any dtype, including iinfo min/max) is
    returned unchanged; float images with +-inf and finfo max/min match scipy.
    """
    for dtype in DTYPES:
        info = np.iinfo(dtype) if np.dtype(dtype).kind in "ui" else np.finfo(dtype)
        for v in (info.min, info.max, 0):
            a = np.full((5, 6), v, dtype)
            for k in (3, 5):
                np.testing.assert_array_equal(median_filter_fast(a, k, backend=backend), a)
    rng = np.random.default_rng(1)
    for dtype in (np.float32, np.float64):
        fi = np.finfo(dtype)
        a = rng.choice(np.array([-np.inf, np.inf, fi.min, fi.max, 0, 1, -1], dtype),
                       (23, 19))
        for k in (3, 5, 7):
            np.testing.assert_array_equal(median_filter_fast(a, k, backend=backend),
                                          _scipy(a, k))


@pytest.mark.parametrize("backend", BACKENDS)
def test_slices_are_independent_and_axes_select_the_plane(backend):
    """Leading axes are independent slices; `axes` picks the filtered plane.

    Strategy: a CZYX stack where one slice is replaced by a spike image; the
    result must equal per-slice 2D calls (so nothing leaks across Z or C).
    Filtering axes (0, 2) of a ZYX volume must equal moving those axes last,
    filtering and moving back, and equal scipy with size (k, 1, k).
    Non-contiguous and big-endian inputs give the same result.
    """
    rng = np.random.default_rng(2)
    stack = _random((2, 3, 21, 18), np.float32, rng)
    stack[1, 2] = 0
    stack[1, 2, 10, 9] = 1e6
    out = median_filter_fast(stack, 3, backend=backend)
    for c, z in np.ndindex(stack.shape[:2]):
        np.testing.assert_array_equal(out[c, z], median_filter_fast(stack[c, z], 3, backend=backend))
    assert out[1, 2].max() == 0  # an isolated spike is removed

    vol = _random((9, 6, 11), np.uint16, rng)
    a = median_filter_fast(vol, 5, axes=(0, 2), backend=backend)
    np.testing.assert_array_equal(a, _scipy(vol, 5, axes=(0, 2)))
    np.testing.assert_array_equal(a, median_filter_fast(vol, 5, axes=(2, 0), backend=backend))

    strided = vol[:, ::2, ::-1]
    np.testing.assert_array_equal(median_filter_fast(strided, 3, backend=backend),
                                  _scipy(strided, 3))
    big = vol.astype(">u2")
    out_big = median_filter_fast(big, 3, backend=backend)
    assert out_big.dtype == np.uint16
    np.testing.assert_array_equal(out_big, _scipy(vol, 3))


# ---------------------------------------------------------------------------
# NaN and signed zeros
# ---------------------------------------------------------------------------

def _nan_oracle(a, k):
    """np.median over the reflected k x k windows (propagates NaN)."""
    p = k // 2
    pad = [(0, 0)] * (a.ndim - 2) + [(p, p), (p, p)]
    win = np.lib.stride_tricks.sliding_window_view(np.pad(a, pad, mode="symmetric"),
                                                   (k, k), axis=(-2, -1))
    return np.median(win, axis=(-2, -1)).astype(a.dtype)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nan_propagates_to_every_window_containing_it(backend, dtype):
    """NaN policy: any NaN in the (reflected) k x k window gives NaN.

    Strategy: (1) an interior NaN in a finite image must turn exactly the
    k x k block around it into NaN and leave every other pixel equal to the
    NaN-free filter; (2) random NaNs (incl. on borders) in a stack match
    np.median over the same reflected windows (np.median propagates NaN).
    """
    rng = np.random.default_rng(3)
    a = _random((15, 16), dtype, rng)
    for k in (3, 5, 7):
        p = k // 2
        b = a.copy()
        b[7, 6] = np.nan
        out = median_filter_fast(b, k, backend=backend)
        expected_nan = np.zeros(a.shape, bool)
        expected_nan[7 - p:7 + p + 1, 6 - p:6 + p + 1] = True
        np.testing.assert_array_equal(np.isnan(out), expected_nan)
        clean = median_filter_fast(a, k, backend=backend)
        np.testing.assert_array_equal(out[~expected_nan], clean[~expected_nan])

    stack = _random((2, 12, 13), dtype, rng)
    stack[rng.random(stack.shape) < 0.03] = np.nan
    stack[0, 0, 0] = stack[1, -1, 5] = np.nan  # corner / border
    for k in (3, 5, 7):
        np.testing.assert_array_equal(median_filter_fast(stack, k, backend=backend),
                                      _nan_oracle(stack, k))


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_median_is_positive_zero(backend):
    """-0.0 and +0.0 tie; a zero median is returned as +0.0 on both backends.

    This makes the output independent of which tied zero a network keeps.
    Expected: an all -0.0 image and a mixed +-0.0 image give +0.0 everywhere.
    """
    for dtype in (np.float32, np.float64):
        neg = np.full((6, 7), -0.0, dtype)
        mixed = neg.copy()
        mixed[::2] = 0.0
        for a in (neg, mixed):
            for k in (3, 5, 7):
                out = median_filter_fast(a, k, backend=backend)
                assert not np.signbit(out).any()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def test_numba_bit_identical_to_numpy():
    """The numba and NumPy backends return bit-identical arrays.

    Both compute exact order statistics with the same NaN and signed-zero
    rules. Strategy: all supported dtypes, k = 3, 5, 7, data with ties, and
    for floats NaN, +-inf and +-0.0; compare raw bits. Expected: identical.
    """
    pytest.importorskip("numba")
    rng = np.random.default_rng(4)
    for dtype in DTYPES:
        a = _random((3, 37, 41), dtype, rng, levels=8)
        if np.dtype(dtype).kind == "f":
            special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype)
            mask = rng.random(a.shape) < 0.1
            a[mask] = rng.choice(special, mask.sum())
        for k in (3, 5, 7):
            x = median_filter_fast(a, k, backend="numba")
            y = median_filter_fast(a, k, backend="numpy")
            np.testing.assert_array_equal(_bits(x), _bits(y), err_msg=f"{dtype} k={k}")


def test_auto_falls_back_to_numpy_without_numba(monkeypatch):
    """backend='auto' works when numba is unavailable (e.g. Pyodide).

    Strategy: make the numba loader report "not importable". Expected: 'auto'
    returns the NumPy result; an explicit backend='numba' raises ImportError.
    """
    monkeypatch.setattr(mf, "_numba_kernels", lambda: None)
    a = _random((20, 30), np.float32, np.random.default_rng(5))
    np.testing.assert_array_equal(median_filter_fast(a), median_filter_fast(a, backend="numpy"))
    with pytest.raises(ImportError, match="numba"):
        median_filter_fast(a, backend="numba")


def test_imports_and_runs_without_skimage_or_numba():
    """The module needs only numpy (scipy not needed at run time either).

    Strategy: in a subprocess, block scikit-image and numba (any import of
    them raises ImportError), import the module, filter with 'auto'.
    Expected: runs on the NumPy backend, gives the expected 3 x 3 median of
    a hand-checkable image, and never imports skimage.
    """
    code = textwrap.dedent(
        """
        import sys
        sys.modules["skimage"] = None
        sys.modules["numba"] = None
        import numpy as np
        from eigenp_utils.median_filter import median_filter_fast

        img = np.zeros((6, 6), np.uint16)
        img[4, 4] = 1000           # isolated spike -> removed
        img[0:3, 0:3] = 7          # 3 x 3 corner block: 9/9 sevens at (0, 0) and
        out = median_filter_fast(img)  # (1, 1), only 4/9 at its corner (2, 2)
        assert out.dtype == np.uint16 and out[4, 4] == 0
        assert out[0, 0] == 7 and out[1, 1] == 7 and out[2, 2] == 0
        try:
            median_filter_fast(img, backend="numba")
            raise SystemExit("numba backend should be unavailable")
        except ImportError:
            pass
        assert "skimage" not in [m.split(".")[0] for m in sys.modules if sys.modules[m]]
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "ok"


# ---------------------------------------------------------------------------
# OpenCV border convention (documentation check; OpenCV is not a dependency)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_opencv_agrees_for_3x3_and_in_the_interior_for_5x5(dtype):
    """cv2.medianBlur uses BORDER_REPLICATE; scipy 'reflect' is half-sample symmetric.

    For k = 3 the one-pixel extension is the same (a | a b c), so the outputs
    are identical. For k = 5 they agree away from the borders; only the
    outer k // 2 rows/columns may differ. Expected exactly that.
    """
    cv2 = pytest.importorskip("cv2")
    a = _random((40, 37), dtype, np.random.default_rng(6))
    np.testing.assert_array_equal(median_filter_fast(a, 3), cv2.medianBlur(a, 3))
    ours, theirs = median_filter_fast(a, 5), cv2.medianBlur(a, 5)
    np.testing.assert_array_equal(ours[2:-2, 2:-2], theirs[2:-2, 2:-2])


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_rejects_bad_arguments():
    """Invalid size, axes, backend, dimensionality or dtype raise clearly.

    Expected: even / non-positive / non-int sizes, repeated or out-of-range
    axes, unknown backend and 1D input raise ValueError; bool, float16 and
    complex raise TypeError (no silent conversion). Empty slices pass through.
    """
    img = np.zeros((8, 8), np.float32)
    for size in (0, 2, -3, 3.0, True, (3, 3)):
        with pytest.raises(ValueError, match="size"):
            median_filter_fast(img, size)
    for axes in ((0, 0), (0,), (0, 1, 2), (0, 5), 3):
        with pytest.raises(ValueError):
            median_filter_fast(img, axes=axes)
    with pytest.raises(ValueError, match="backend"):
        median_filter_fast(img, backend="opencv")
    with pytest.raises(ValueError, match="2 dimensions"):
        median_filter_fast(np.zeros(8, np.float32))
    for dtype in (bool, np.float16, np.complex64):
        with pytest.raises(TypeError):
            median_filter_fast(img.astype(dtype))
    empty = np.zeros((0, 5, 4), np.uint8)
    assert median_filter_fast(empty).shape == empty.shape
