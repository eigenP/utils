# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "numba",
# ]
# ///
"""numba kernels for `median_filter.median_filter_fast` (imported lazily).

Importing this module raises ImportError when numba is missing, which makes
`median_filter_fast` fall back to its NumPy backend. `median3` / `median5` are
the comparator networks of `median_filter` (`_COLSORT3`, `_NET3`, ...)
unrolled (generated code: edit the networks there, then regenerate), so they
compute the same exact order statistic as the NumPy backend; with the shared
NaN and signed-zero rules both backends are bit-identical.
"""
import numba
import numpy as np


@numba.njit(cache=True)
def _lo(a, b):
    return b if b < a else a


@numba.njit(cache=True)
def _hi(a, b):
    return a if b < a else b


@numba.njit(cache=True)
def _reflect(i, n):
    """scipy 'reflect' index (half-sample symmetric), any distance past the edge."""
    m = i % (2 * n)  # Python semantics: non-negative
    return 2 * n - 1 - m if m >= n else m


@numba.njit(cache=True)
def _nan_fix(img, s, y, k, xi, o, flag):
    """o[x] = NaN where the k x k window centred on (y, x) holds a NaN.

    Called only for rows whose window rows contain a NaN. `xi` maps padded
    columns to source columns; `flag` is scratch (one bool per padded column).
    """
    H = img.shape[1]
    W = img.shape[2]
    p = k // 2
    for j in range(W + 2 * p):
        f = False
        for d in range(k):
            v = img[s, _reflect(y - p + d, H), xi[j]]
            f |= v != v
        flag[j] = f
    for x in range(W):
        f = False
        for c in range(k):
            f |= flag[x + c]
        if f:
            o[x] = np.nan


@numba.njit(parallel=True, cache=True)
def median3(img, out, zero, xi, rows_per_block):
    """3 x 3 median of each (H, W) slice of a C-contiguous (S, H, W) array.

    Generated from median_filter._COLSORT3 / _NET3: per row, sort every
    padded column once (c0 <= ... <= c2), then run the window network.
    """
    S, H, W = img.shape
    n_rows = S * H
    n_blocks = (n_rows + rows_per_block - 1) // rows_per_block
    z = zero[0]
    for b in numba.prange(n_blocks):
        c0 = np.empty(W + 2, img.dtype)
        c1 = np.empty(W + 2, img.dtype)
        c2 = np.empty(W + 2, img.dtype)
        flag = np.empty(W + 2, np.bool_)
        for row in range(b * rows_per_block, min(n_rows, (b + 1) * rows_per_block)):
            s = row // H
            y = row - s * H
            r0 = img[s, _reflect(y - 1, H)]
            r1 = img[s, y]
            r2 = img[s, _reflect(y + 1, H)]
            has_nan = False
            for x in range(W):
                v0 = r0[x]
                v1 = r1[x]
                v2 = r2[x]
                has_nan |= (v0 != v0) | (v1 != v1) | (v2 != v2)
                v0, v1 = _lo(v0, v1), _hi(v0, v1)
                v1, v2 = _lo(v1, v2), _hi(v1, v2)
                v0, v1 = _lo(v0, v1), _hi(v0, v1)
                c0[x + 1] = v0
                c1[x + 1] = v1
                c2[x + 1] = v2
            # reflected border columns: sorted copies of their source columns
            for j in range(1):
                c0[j] = c0[xi[j] + 1]
                c1[j] = c1[xi[j] + 1]
                c2[j] = c2[xi[j] + 1]
            for j in range(W + 1, W + 2):
                c0[j] = c0[xi[j] + 1]
                c1[j] = c1[xi[j] + 1]
                c2[j] = c2[xi[j] + 1]
            o = out[s, y]
            for x in range(W):
                w0 = c0[x]
                w1 = c1[x]
                w2 = c2[x]
                w3 = c0[x + 1]
                w4 = c1[x + 1]
                w5 = c2[x + 1]
                w6 = c0[x + 2]
                w7 = c1[x + 2]
                w8 = c2[x + 2]
                w3 = _hi(w0, w3)
                w6 = _hi(w3, w6)
                w2 = _lo(w2, w5)
                w2 = _lo(w2, w8)
                w1, w4 = _lo(w1, w4), _hi(w1, w4)
                w4 = _lo(w4, w7)
                w4 = _hi(w1, w4)
                w6, w4 = _lo(w6, w4), _hi(w6, w4)
                w4 = _lo(w4, w2)
                w4 = _hi(w6, w4)
                o[x] = w4 + z
            if has_nan:
                _nan_fix(img, s, y, 3, xi, o, flag)


@numba.njit(parallel=True, cache=True)
def median5(img, out, zero, xi, rows_per_block):
    """5 x 5 median of each (H, W) slice of a C-contiguous (S, H, W) array.

    Generated from median_filter._COLSORT5 / _NET5: per row, sort every
    padded column once (c0 <= ... <= c4), then run the window network.
    """
    S, H, W = img.shape
    n_rows = S * H
    n_blocks = (n_rows + rows_per_block - 1) // rows_per_block
    z = zero[0]
    for b in numba.prange(n_blocks):
        c0 = np.empty(W + 4, img.dtype)
        c1 = np.empty(W + 4, img.dtype)
        c2 = np.empty(W + 4, img.dtype)
        c3 = np.empty(W + 4, img.dtype)
        c4 = np.empty(W + 4, img.dtype)
        flag = np.empty(W + 4, np.bool_)
        for row in range(b * rows_per_block, min(n_rows, (b + 1) * rows_per_block)):
            s = row // H
            y = row - s * H
            r0 = img[s, _reflect(y - 2, H)]
            r1 = img[s, _reflect(y - 1, H)]
            r2 = img[s, y]
            r3 = img[s, _reflect(y + 1, H)]
            r4 = img[s, _reflect(y + 2, H)]
            has_nan = False
            for x in range(W):
                v0 = r0[x]
                v1 = r1[x]
                v2 = r2[x]
                v3 = r3[x]
                v4 = r4[x]
                has_nan |= (v0 != v0) | (v1 != v1) | (v2 != v2) | (v3 != v3) | (v4 != v4)
                v0, v1 = _lo(v0, v1), _hi(v0, v1)
                v3, v4 = _lo(v3, v4), _hi(v3, v4)
                v2, v4 = _lo(v2, v4), _hi(v2, v4)
                v2, v3 = _lo(v2, v3), _hi(v2, v3)
                v0, v3 = _lo(v0, v3), _hi(v0, v3)
                v0, v2 = _lo(v0, v2), _hi(v0, v2)
                v1, v4 = _lo(v1, v4), _hi(v1, v4)
                v1, v3 = _lo(v1, v3), _hi(v1, v3)
                v1, v2 = _lo(v1, v2), _hi(v1, v2)
                c0[x + 2] = v0
                c1[x + 2] = v1
                c2[x + 2] = v2
                c3[x + 2] = v3
                c4[x + 2] = v4
            # reflected border columns: sorted copies of their source columns
            for j in range(2):
                c0[j] = c0[xi[j] + 2]
                c1[j] = c1[xi[j] + 2]
                c2[j] = c2[xi[j] + 2]
                c3[j] = c3[xi[j] + 2]
                c4[j] = c4[xi[j] + 2]
            for j in range(W + 2, W + 4):
                c0[j] = c0[xi[j] + 2]
                c1[j] = c1[xi[j] + 2]
                c2[j] = c2[xi[j] + 2]
                c3[j] = c3[xi[j] + 2]
                c4[j] = c4[xi[j] + 2]
            o = out[s, y]
            for x in range(W):
                w0 = c0[x]
                w1 = c1[x]
                w2 = c2[x]
                w3 = c3[x]
                w4 = c4[x]
                w5 = c0[x + 1]
                w6 = c1[x + 1]
                w7 = c2[x + 1]
                w8 = c3[x + 1]
                w9 = c4[x + 1]
                w10 = c0[x + 2]
                w11 = c1[x + 2]
                w12 = c2[x + 2]
                w13 = c3[x + 2]
                w14 = c4[x + 2]
                w15 = c0[x + 3]
                w16 = c1[x + 3]
                w17 = c2[x + 3]
                w18 = c3[x + 3]
                w19 = c4[x + 3]
                w20 = c0[x + 4]
                w21 = c1[x + 4]
                w22 = c2[x + 4]
                w23 = c3[x + 4]
                w24 = c4[x + 4]
                w0, w5 = _lo(w0, w5), _hi(w0, w5)
                w4, w9 = _lo(w4, w9), _hi(w4, w9)
                w4, w5 = _lo(w4, w5), _hi(w4, w5)
                w2, w7 = _lo(w2, w7), _hi(w2, w7)
                w2, w4 = _lo(w2, w4), _hi(w2, w4)
                w7, w5 = _lo(w7, w5), _hi(w7, w5)
                w1, w6 = _lo(w1, w6), _hi(w1, w6)
                w3, w8 = _lo(w3, w8), _hi(w3, w8)
                w3, w6 = _lo(w3, w6), _hi(w3, w6)
                w1, w2 = _lo(w1, w2), _hi(w1, w2)
                w3, w4 = _lo(w3, w4), _hi(w3, w4)
                w6, w7 = _lo(w6, w7), _hi(w6, w7)
                w8, w5 = _lo(w8, w5), _hi(w8, w5)
                w10, w15 = _lo(w10, w15), _hi(w10, w15)
                w14, w19 = _lo(w14, w19), _hi(w14, w19)
                w14, w15 = _lo(w14, w15), _hi(w14, w15)
                w12, w17 = _lo(w12, w17), _hi(w12, w17)
                w12, w14 = _lo(w12, w14), _hi(w12, w14)
                w17, w15 = _lo(w17, w15), _hi(w17, w15)
                w11, w16 = _lo(w11, w16), _hi(w11, w16)
                w13, w18 = _lo(w13, w18), _hi(w13, w18)
                w13, w16 = _lo(w13, w16), _hi(w13, w16)
                w11, w12 = _lo(w11, w12), _hi(w11, w12)
                w13, w14 = _lo(w13, w14), _hi(w13, w14)
                w16, w17 = _lo(w16, w17), _hi(w16, w17)
                w18, w15 = _lo(w18, w15), _hi(w18, w15)
                w10 = _hi(w0, w10)
                w5, w15 = _lo(w5, w15), _hi(w5, w15)
                w5, w10 = _lo(w5, w10), _hi(w5, w10)
                w4, w14 = _lo(w4, w14), _hi(w4, w14)
                w5 = _hi(w4, w5)
                w14, w10 = _lo(w14, w10), _hi(w14, w10)
                w12 = _hi(w2, w12)
                w7 = _lo(w7, w17)
                w7, w12 = _lo(w7, w12), _hi(w7, w12)
                w5 = _hi(w7, w5)
                w12, w14 = _lo(w12, w14), _hi(w12, w14)
                w11 = _hi(w1, w11)
                w9, w19 = _lo(w9, w19), _hi(w9, w19)
                w9, w11 = _lo(w9, w11), _hi(w9, w11)
                w6, w16 = _lo(w6, w16), _hi(w6, w16)
                w9 = _hi(w6, w9)
                w16, w11 = _lo(w16, w11), _hi(w16, w11)
                w13 = _hi(w3, w13)
                w8 = _lo(w8, w18)
                w8, w13 = _lo(w8, w13), _hi(w8, w13)
                w8, w9 = _lo(w8, w9), _hi(w8, w9)
                w13, w16 = _lo(w13, w16), _hi(w13, w16)
                w8, w5 = _lo(w8, w5), _hi(w8, w5)
                w9, w12 = _lo(w9, w12), _hi(w9, w12)
                w13, w14 = _lo(w13, w14), _hi(w13, w14)
                w10 = _lo(w10, w20)
                w10 = _hi(w5, w10)
                w14 = _lo(w14, w24)
                w14 = _lo(w14, w10)
                w15 = _lo(w15, w22)
                w12 = _lo(w12, w15)
                w14 = _hi(w12, w14)
                w11 = _lo(w11, w21)
                w11 = _hi(w9, w11)
                w16 = _lo(w16, w11)
                w19 = _lo(w19, w23)
                w13 = _lo(w13, w19)
                w13 = _hi(w8, w13)
                w13 = _lo(w13, w16)
                w14 = _hi(w13, w14)
                o[x] = w14 + z
            if has_nan:
                _nan_fix(img, s, y, 5, xi, o, flag)


@numba.njit(cache=True)
def _select(a, n, kth):
    """kth smallest of a[:n] (Wirth's selection, in place). No NaN allowed."""
    lo = 0
    hi = n - 1
    while lo < hi:
        pivot = a[kth]
        i = lo
        j = hi
        while True:
            while a[i] < pivot:
                i += 1
            while pivot < a[j]:
                j -= 1
            if i <= j:
                t = a[i]
                a[i] = a[j]
                a[j] = t
                i += 1
                j -= 1
            if i > j:
                break
        if j < kth:
            lo = i
        if kth < i:
            hi = j
    return a[kth]


@numba.njit(parallel=True, cache=True)
def median_k(img, out, zero, k, xi, rows_per_block):
    """k x k median (any odd k) by per-pixel selection, O(k^2) per pixel."""
    S, H, W = img.shape
    p = k // 2
    n_rows = S * H
    n_blocks = (n_rows + rows_per_block - 1) // rows_per_block
    z = zero[0]
    for b in numba.prange(n_blocks):
        win = np.empty(k * k, img.dtype)
        yi = np.empty(k, np.intp)
        for row in range(b * rows_per_block, min(n_rows, (b + 1) * rows_per_block)):
            s = row // H
            y = row - s * H
            for d in range(k):
                yi[d] = _reflect(y - p + d, H)
            o = out[s, y]
            for x in range(W):
                has_nan = False
                n = 0
                for d in range(k):
                    r = img[s, yi[d]]
                    for c in range(k):
                        v = r[xi[x + c]]
                        has_nan |= v != v
                        win[n] = v
                        n += 1
                if has_nan:
                    o[x] = np.nan
                else:
                    o[x] = _select(win, n, n // 2) + z
