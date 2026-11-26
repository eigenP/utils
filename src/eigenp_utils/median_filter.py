# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
# ]
# ///
"""Fast k x k median filter over two axes of an N-D stack (numba or NumPy).

`median_filter_fast` replaces ``cv2.medianBlur`` (OpenCV) and
``scipy.ndimage.median_filter`` for per-slice 2D medians of image stacks. It
returns exactly ``scipy.ndimage.median_filter(image, size=(1, ..., k, k),
mode='reflect')`` for finite data.

Algorithm (k = 3 and 5): each output row sorts the k values of every input
column once with a sorting network (shared by the k windows that contain the
column), then a pruned selection network picks the median of the k x k window
from its k sorted columns: 12 min/max operations per pixel for 3 x 3 (the
classic max-of-lows / median-of-mids / min-of-highs), 108 for 5 x 5 (Batcher
merges of the sorted columns, pruned to the median and greedily minimised). A
comparator network computes the exact order statistic, so the result is exact
for every dtype. Other odd k use a per-pixel selection (numba) or
``np.partition`` (NumPy).

The networks are data (`_COLSORT3`, `_NET3`, `_COLSORT5`, `_NET5`) checked by
the 0-1 principle in ``tests/test_median_filter.py``; the NumPy backend applies
them to whole row blocks, and ``_median_numba.py`` has them unrolled.
"""
import warnings

import numpy as np

__all__ = ["median_filter_fast"]


# ---------------------------------------------------------------------------
# Comparator networks. Ops are (i, j, kind) on wires w: 'ce' sets
# (w[i], w[j]) = (min, max); 'min' sets only w[i] = min(w[i], w[j]); 'max'
# sets only w[j] = max(w[i], w[j]) (the other output is never used).
# Column networks sort k values ascending (wire = row offset). Window networks
# take k sorted columns, wire = k * column + rank, and leave the median on
# wire _OUT{k}.
# ---------------------------------------------------------------------------

_COLSORT3 = ((0, 1, 'ce'), (1, 2, 'ce'), (0, 1, 'ce'))
# med3(max of lows, med3 of mids, min of highs)  (Paeth / Smith)
_NET3 = (
    (0, 3, 'max'), (3, 6, 'max'),                   # w6 = max(lows)
    (2, 5, 'min'), (2, 8, 'min'),                   # w2 = min(highs)
    (1, 4, 'ce'), (4, 7, 'min'), (1, 4, 'max'),     # w4 = med(mids)
    (6, 4, 'ce'), (4, 2, 'min'), (6, 4, 'max'),     # w4 = med(w6, w4, w2)
)
_OUT3 = 4

# Optimal 9-comparator sorting network for 5 values
_COLSORT5 = (
    (0, 1, 'ce'), (3, 4, 'ce'), (2, 4, 'ce'), (2, 3, 'ce'), (0, 3, 'ce'),
    (0, 2, 'ce'), (1, 4, 'ce'), (1, 3, 'ce'), (1, 2, 'ce'),
)
# Odd-even merges of the sorted columns ((0, 1), (2, 3)), 4, pruned to the
# median and greedily minimised against all column-sorted 0-1 inputs: 108 ops
_NET5 = (
    (0, 5, 'ce'), (4, 9, 'ce'), (4, 5, 'ce'), (2, 7, 'ce'), (2, 4, 'ce'),
    (7, 5, 'ce'), (1, 6, 'ce'), (3, 8, 'ce'), (3, 6, 'ce'), (1, 2, 'ce'),
    (3, 4, 'ce'), (6, 7, 'ce'), (8, 5, 'ce'), (10, 15, 'ce'), (14, 19, 'ce'),
    (14, 15, 'ce'), (12, 17, 'ce'), (12, 14, 'ce'), (17, 15, 'ce'), (11, 16, 'ce'),
    (13, 18, 'ce'), (13, 16, 'ce'), (11, 12, 'ce'), (13, 14, 'ce'), (16, 17, 'ce'),
    (18, 15, 'ce'), (0, 10, 'max'), (5, 15, 'ce'), (5, 10, 'ce'), (4, 14, 'ce'),
    (4, 5, 'max'), (14, 10, 'ce'), (2, 12, 'max'), (7, 17, 'min'), (7, 12, 'ce'),
    (7, 5, 'max'), (12, 14, 'ce'), (1, 11, 'max'), (9, 19, 'ce'), (9, 11, 'ce'),
    (6, 16, 'ce'), (6, 9, 'max'), (16, 11, 'ce'), (3, 13, 'max'), (8, 18, 'min'),
    (8, 13, 'ce'), (8, 9, 'ce'), (13, 16, 'ce'), (8, 5, 'ce'), (9, 12, 'ce'),
    (13, 14, 'ce'), (10, 20, 'min'), (5, 10, 'max'), (14, 24, 'min'),
    (14, 10, 'min'), (15, 22, 'min'), (12, 15, 'min'), (12, 14, 'max'),
    (11, 21, 'min'), (9, 11, 'max'), (16, 11, 'min'), (19, 23, 'min'),
    (13, 19, 'min'), (8, 13, 'max'), (13, 16, 'min'), (13, 14, 'max'),
)
_OUT5 = 14

_NETWORKS = {3: (_COLSORT3, _NET3, _OUT3), 5: (_COLSORT5, _NET5, _OUT5)}


def _apply_network(wires, ops):
    """Apply comparator ops to a list of arrays (or scalars), in place."""
    for i, j, kind in ops:
        a, b = wires[i], wires[j]
        if kind == 'ce':
            wires[i], wires[j] = np.minimum(a, b), np.maximum(a, b)
        elif kind == 'min':
            wires[i] = np.minimum(a, b)
        else:
            wires[j] = np.maximum(a, b)
    return wires


def _reflect_index(idx, n):
    """scipy 'reflect' (d c b a | a b c d | d c b a) for any distance past the edge."""
    m = np.mod(idx, 2 * n)
    return np.where(m >= n, 2 * n - 1 - m, m)


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------

def _window_any(mask, k, n, W):
    """(n, W) OR of a padded (n + k - 1, W + k - 1) boolean mask over k x k windows."""
    col = mask[0:n].copy()
    for d in range(1, k):
        col |= mask[d:d + n]
    out = col[:, 0:W].copy()
    for c in range(1, k):
        out |= col[:, c:c + W]
    return out


def _median_numpy(img, k, chunk_px=1 << 16):
    """(S, H, W) -> (S, H, W) k x k median per slice, row blocks of ~chunk_px pixels."""
    S, H, W = img.shape
    p = k // 2
    out = np.empty_like(img)
    yi = _reflect_index(np.arange(-p, H + p), H)
    xi = _reflect_index(np.arange(-p, W + p), W)
    is_float = img.dtype.kind == 'f'
    net = _NETWORKS.get(k)
    # np.partition holds k * k copies of the block: keep it ~chunk_px * 9
    budget = chunk_px if net is not None else max(1, (9 * chunk_px) // (k * k))
    rows = max(1, budget // (W + 2 * p))
    for s in range(S):
        src = img[s]
        for r0 in range(0, H, rows):
            r1 = min(H, r0 + rows)
            n = r1 - r0
            block = np.take(np.take(src, yi[r0:r1 + 2 * p], axis=0), xi, axis=1)
            if net is not None:
                colsort, window, out_wire = net
                cols = _apply_network([block[d:d + n] for d in range(k)], colsort)
                wires = [cols[r][:, c:c + W] for c in range(k) for r in range(k)]
                res = _apply_network(wires, window)[out_wire]
            else:
                stack = np.stack([block[dy:dy + n, dx:dx + W]
                                  for dy in range(k) for dx in range(k)])
                res = np.partition(stack, k * k // 2, axis=0)[k * k // 2]
            dst = out[s, r0:r1]
            dst[...] = res
            if is_float:
                nan = np.isnan(block)
                if nan.any():
                    dst[_window_any(nan, k, n, W)] = np.nan
    if is_float:
        # -0.0 and +0.0 tie: return +0.0 whichever one the selection kept
        out += out.dtype.type(0)
    return out


# ---------------------------------------------------------------------------
# numba backend
# ---------------------------------------------------------------------------

def _numba_kernels():
    """The compiled numba backend, or None when numba is not importable."""
    try:
        from . import _median_numba
    except ImportError:
        return None
    return _median_numba


def _median_numba(img, k, kernels):
    S, H, W = img.shape
    out = np.empty_like(img)
    zero = np.zeros(1, img.dtype)
    n_rows = S * H
    # ~8 row blocks per thread: balances threads without per-row overhead
    n_blocks = 8 * kernels.numba.get_num_threads()
    rows_per_block = max(1, -(-n_rows // n_blocks))
    p = k // 2
    # source column of each padded column
    xi = _reflect_index(np.arange(-p, W + p), W).astype(np.intp)
    if k == 3:
        kernels.median3(img, out, zero, xi, rows_per_block)
    elif k == 5:
        kernels.median5(img, out, zero, xi, rows_per_block)
    else:
        kernels.median_k(img, out, zero, k, xi, rows_per_block)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _check_dtype(dtype):
    if dtype.kind in 'ui' or (dtype.kind == 'f' and dtype.itemsize in (4, 8)):
        return
    raise TypeError(
        f"median_filter_fast supports integer, float32 and float64 images, not "
        f"{dtype}; convert explicitly (e.g. float16 -> float32, bool -> uint8)."
    )


def median_filter_fast(image, size=3, axes=(-2, -1), backend='auto'):
    """k x k median filter of every 2D slice of an N-D stack.

    Equals ``scipy.ndimage.median_filter(image, size=s, mode='reflect')`` with
    ``s = k`` on ``axes`` and 1 elsewhere (bit for bit, finite data), at
    OpenCV-like speed: each 2D slice (e.g. every (Y, X) plane of a ZYX or CZYX
    stack) is filtered independently.

    Parameters
    ----------
    image : ndarray, at least 2D
        Axis order is free: ``axes`` names the two filtered (spatial) axes,
        by default the last two, i.e. (..., Y, X). All other axes (Z, C, T)
        are independent slices, so the package's channel-first convention
        (C, ..., Y, X) needs no option. A channel-last (Y, X, 3/4) image
        needs ``axes=(0, 1)``: with the default axes its colour axis is
        filtered as if it were spatial, and a UserWarning says so (C-07). Supported dtypes: uint8, uint16, float32
        and float64, plus the other integer types. bool, float16 and complex
        raise TypeError (convert explicitly).
    size : int
        Odd window side k (default 3). 3 and 5 use sorting networks; other
        odd k fall back to a per-pixel selection, O(k^2) per pixel. 1 returns
        a copy.
    axes : (int, int)
        The two axes the k x k window spans. Default (-2, -1).
    backend : {'auto', 'numba', 'numpy'}
        ``'numba'`` (used by ``'auto'`` when numba is importable) runs fused
        loops parallel over rows; threads follow numba's settings
        (``NUMBA_NUM_THREADS`` / ``numba.set_num_threads``). The first call per
        dtype in a fresh environment compiles for a few seconds; the compiled
        code is cached on disk. ``'numpy'`` is a vectorised fallback (Pyodide,
        no numba). Both return bit-identical arrays.

    Returns
    -------
    out : ndarray
        Same shape and dtype as ``image`` (never converted).

    Notes
    -----
    Borders use scipy's ``mode='reflect'``, i.e. half-sample symmetric
    extension ``d c b a | a b c d | d c b a`` (NumPy's ``'symmetric'``), also
    for windows larger than the image. ``cv2.medianBlur`` uses
    ``BORDER_REPLICATE`` (``a a a | a b c d``): identical for k = 3, but for
    k >= 5 the outer ``k // 2 - 1`` rows and columns can differ.

    NaN policy (floats): NaN propagates. Any NaN inside the k x k window,
    counting reflected border pixels, gives NaN; scipy's result is
    unspecified for NaN. +-inf are ordinary values. Signed zeros: -0.0 and
    +0.0 compare equal, and a zero median is returned as +0.0.

    Examples
    --------
    >>> stack = np.random.default_rng(0).random((4, 64, 64), dtype=np.float32)
    >>> out = median_filter_fast(stack, size=3)          # per Z slice
    >>> from scipy import ndimage
    >>> np.array_equal(out, ndimage.median_filter(stack, size=(1, 3, 3), mode='reflect'))
    True
    """
    if backend not in ('auto', 'numba', 'numpy'):
        raise ValueError(f"Unknown backend: {backend!r}")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)):
        raise ValueError(f"`size` must be an odd positive int, got {size!r}")
    k = int(size)
    if k < 1 or k % 2 == 0:
        raise ValueError(f"`size` must be an odd positive int, got {size!r}")

    image = np.asarray(image)
    _check_dtype(image.dtype)
    if image.ndim < 2:
        raise ValueError("`image` must have at least 2 dimensions")
    try:
        ax = tuple(int(a) for a in axes)
    except TypeError:
        raise ValueError(f"`axes` must be two distinct axes, got {axes!r}") from None
    if len(ax) != 2:
        raise ValueError(f"`axes` must be two distinct axes, got {axes!r}")
    ax = tuple(_normalize_axis(a, image.ndim) for a in ax)
    if ax[0] == ax[1]:
        raise ValueError(f"`axes` must be two distinct axes, got {axes!r}")
    last = image.ndim - 1
    if (image.ndim >= 3 and image.shape[-1] in (3, 4) and last in ax
            and k > 1 and image.size):
        warnings.warn(
            f"median_filter_fast filters axes {tuple(ax)} of this {image.shape} "
            f"image, including its trailing axis of length {image.shape[-1]}: if "
            "it holds colour channels (channel-last RGB/RGBA), they are mixed. "
            "Use axes=(0, 1) for a (Y, X, C) image, or put channels first.",
            UserWarning, stacklevel=2)

    kernels = None
    if backend != 'numpy':
        kernels = _numba_kernels()
        if kernels is None and backend == 'numba':
            raise ImportError("backend='numba' requires numba")

    if not image.dtype.isnative:
        image = image.astype(image.dtype.newbyteorder('='))
    moved = np.moveaxis(image, ax, (-2, -1))
    if k == 1 or moved.size == 0:
        return np.moveaxis(moved.copy(), (-2, -1), ax)
    H, W = moved.shape[-2:]
    img = np.ascontiguousarray(moved).reshape(-1, H, W)

    if kernels is not None:
        res = _median_numba(img, k, kernels)
    else:
        res = _median_numpy(img, k)
    return np.moveaxis(res.reshape(moved.shape), (-2, -1), ax)


def _normalize_axis(a, ndim):
    if not -ndim <= a < ndim:
        raise ValueError(f"axis {a} is out of bounds for an array of dimension {ndim}")
    return a % ndim
