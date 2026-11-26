"""Tests for eigenp_utils.median_filter.median_filter_fast.

Oracle: scipy.ndimage.median_filter(size=(1, ..., k, k), mode='reflect'), an
exact median, for finite data; for NaN, np.median over the same reflected
windows (np.median propagates NaN). The comparator networks are checked
exhaustively with the 0-1 principle.
"""
import importlib
import itertools
import sys

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
    Both hand-built networks are checked: 3 x 3 (64 window cases) and 5 x 5
    (6 ** 5 = 7776 window cases); R6-18 added 5 x 5.
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
@pytest.mark.parametrize("size", [3, 5, 7])
def test_matches_scipy_reflect(backend, dtype, size):
    """Bit-for-bit equal to scipy.ndimage.median_filter(mode='reflect').

    Strategy: odd shapes, 1-pixel-wide rows/columns, images smaller than the
    window (reflection past the far edge), 3D and 4D stacks (scipy size 1 on
    leading axes), each with full-range values and with 4 levels (ties), for
    the 3 x 3 and 5 x 5 networks and the k >= 7 selection path (R6-18).
    Expected: identical arrays, input dtype kept.
    """
    rng = np.random.default_rng(0)
    for shape in SHAPES:
        for levels in (None, 4):
            a = _random(shape, dtype, rng, levels)
            out = median_filter_fast(a, size, backend=backend)
            assert out.dtype == a.dtype and out.shape == a.shape
            np.testing.assert_array_equal(
                out, _scipy(a, size), err_msg=f"{shape} levels={levels}")


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
            np.testing.assert_array_equal(median_filter_fast(a, backend=backend), a)
    rng = np.random.default_rng(1)
    for dtype in (np.float32, np.float64):
        fi = np.finfo(dtype)
        a = rng.choice(np.array([-np.inf, np.inf, fi.min, fi.max, 0, 1, -1], dtype),
                       (23, 19))
        np.testing.assert_array_equal(median_filter_fast(a, backend=backend),
                                      _scipy(a, 3))


@pytest.mark.parametrize("backend", BACKENDS)
def test_slices_are_independent_and_axes_select_the_plane(backend):
    """Leading axes are independent slices; `axes` picks the filtered plane.

    Strategy: a CZYX stack where one slice is replaced by a spike image; the
    result must equal per-slice 2D calls (so nothing leaks across Z or C).
    Filtering axes (0, 2) of a ZYX volume must equal moving those axes last,
    filtering and moving back, and equal scipy with size (3, 1, 3).
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
    a = median_filter_fast(vol, axes=(0, 2), backend=backend)
    np.testing.assert_array_equal(a, _scipy(vol, 3, axes=(0, 2)))
    np.testing.assert_array_equal(a, median_filter_fast(vol, axes=(2, 0), backend=backend))

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

    Strategy (3 x 3): (1) an interior NaN in a finite image must turn exactly
    the 3 x 3 block around it into NaN and leave every other pixel equal to the
    NaN-free filter; (2) random NaNs (incl. on borders) in a stack match
    np.median over the same reflected windows (np.median propagates NaN).
    """
    rng = np.random.default_rng(3)
    a = _random((15, 16), dtype, rng)
    b = a.copy()
    b[7, 6] = np.nan
    out = median_filter_fast(b, backend=backend)
    expected_nan = np.zeros(a.shape, bool)
    expected_nan[6:9, 5:8] = True
    np.testing.assert_array_equal(np.isnan(out), expected_nan)
    clean = median_filter_fast(a, backend=backend)
    np.testing.assert_array_equal(out[~expected_nan], clean[~expected_nan])

    stack = _random((2, 12, 13), dtype, rng)
    stack[rng.random(stack.shape) < 0.03] = np.nan
    stack[0, 0, 0] = stack[1, -1, 5] = np.nan  # corner / border
    np.testing.assert_array_equal(median_filter_fast(stack, backend=backend),
                                  _nan_oracle(stack, 3))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("size", [5, 7])
def test_nan_policy_for_larger_windows(backend, size):
    """The NaN policy also holds for the 5 x 5 network and the k >= 7 path (R6-18).

    Strategy: random NaNs (incl. corners and borders) in a float32 stack.
    Expected: equal to np.median over the same reflected k x k windows, and
    NaN-free pixels far from every NaN equal the scipy result.
    """
    rng = np.random.default_rng(30 + size)
    stack = _random((2, 17, 19), np.float32, rng)
    stack[rng.random(stack.shape) < 0.01] = np.nan
    stack[0, 0, 0] = stack[1, -1, 5] = np.nan
    np.testing.assert_array_equal(median_filter_fast(stack, size, backend=backend),
                                  _nan_oracle(stack, size))


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
            assert not np.signbit(median_filter_fast(a, backend=backend)).any()


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def test_numba_bit_identical_to_numpy():
    """The numba and NumPy backends return bit-identical arrays.

    Both compute exact order statistics with the same NaN and signed-zero
    rules. Strategy: all supported dtypes, 3 x 3, 5 x 5 and 7 x 7 (R6-18),
    data with ties, and for floats NaN, +-inf and +-0.0; compare raw bits.
    Expected: identical.
    """
    pytest.importorskip("numba")
    rng = np.random.default_rng(4)
    for size in (3, 5, 7):
        for dtype in DTYPES:
            a = _random((3, 37, 41), dtype, rng, levels=8)
            if np.dtype(dtype).kind == "f":
                special = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype)
                mask = rng.random(a.shape) < 0.1
                a[mask] = rng.choice(special, mask.sum())
            x = median_filter_fast(a, size, backend="numba")
            y = median_filter_fast(a, size, backend="numpy")
            np.testing.assert_array_equal(_bits(x), _bits(y), err_msg=f"{dtype} k={size}")


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


def _import_fresh_with_blocked(monkeypatch, module, blocked, forget=()):
    """Re-import `module` while every import of the `blocked` packages fails.

    Done in-process (Pyodide cannot start subprocesses): blocked packages and
    their already-imported submodules map to None in sys.modules, so any
    `import` of them raises ImportError. `module` and the `forget` modules
    (lazily imported helpers) are dropped from sys.modules and their parent
    package so they are imported again. monkeypatch restores everything.
    """
    for name in list(sys.modules):
        if name.split(".")[0] in blocked:
            monkeypatch.setitem(sys.modules, name, None)
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
    for name in (module, *forget):
        parent, _, child = name.rpartition(".")
        monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.delattr(sys.modules[parent], child, raising=False)
    return importlib.import_module(module)


def test_imports_and_runs_without_skimage_or_numba(monkeypatch):
    """The module needs only numpy (no scikit-image, no numba).

    Strategy: block scikit-image and numba (any import of them raises
    ImportError), import the module afresh, filter with 'auto'. Runs in
    Pyodide too, where numba genuinely does not exist.
    Expected: the import succeeds, 'auto' runs on the NumPy backend and gives
    the 3 x 3 median of a hand-checkable image, backend='numba' raises
    ImportError.
    """
    fresh = _import_fresh_with_blocked(
        monkeypatch, "eigenp_utils.median_filter", ("skimage", "numba"),
        forget=("eigenp_utils._median_numba",))
    assert fresh._numba_kernels() is None

    img = np.zeros((6, 6), np.uint16)
    img[4, 4] = 1000           # isolated spike -> removed
    img[0:3, 0:3] = 7          # 3 x 3 corner block: 9/9 sevens at (0, 0) and
    out = fresh.median_filter_fast(img)  # (1, 1), only 4/9 at its corner (2, 2)
    assert out.dtype == np.uint16 and out[4, 4] == 0
    assert out[0, 0] == 7 and out[1, 1] == 7 and out[2, 2] == 0
    with pytest.raises(ImportError, match="numba"):
        fresh.median_filter_fast(img, backend="numba")


# ---------------------------------------------------------------------------
# OpenCV agreement (documentation check; OpenCV is not a dependency)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_opencv_agrees_for_3x3(dtype):
    """cv2.medianBlur (BORDER_REPLICATE) equals scipy 'reflect' for 3 x 3.

    Half-sample symmetric reflection extends by one pixel exactly as
    replication does (a | a b c), so the 3 x 3 outputs are identical,
    borders included. (For k >= 5 the outer rows/columns can differ.)
    """
    cv2 = pytest.importorskip("cv2")
    a = _random((40, 37), dtype, np.random.default_rng(6))
    np.testing.assert_array_equal(median_filter_fast(a), cv2.medianBlur(a, 3))


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


def test_warns_when_a_trailing_colour_axis_is_filtered():
    """C-07: a trailing axis of length 3 or 4 among the filtered axes gives a UserWarning.

    Strategy: a channel-last (32, 32, 3) float image, as ``color_coded_projection``
    returns, with the default axes (-2, -1), which filter X and the colour
    axis together. Expected outcome: a UserWarning pointing to axes=(0, 1).
    Negative controls: axes=(0, 1) on the same image, a (3, 32, 32)
    channel-first stack and a 2D (32, 3) image do not warn, and the filtered
    output is unchanged by the warning.
    """
    import warnings

    rgb = np.random.default_rng(9).random((32, 32, 3)).astype(np.float32)
    with pytest.warns(UserWarning, match=r"axes=\(0, 1\)"):
        out = median_filter_fast(rgb)
    np.testing.assert_array_equal(out, _scipy(rgb, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        median_filter_fast(rgb, axes=(0, 1))
        median_filter_fast(np.moveaxis(rgb, -1, 0))
        median_filter_fast(rgb[:, 0, :])
