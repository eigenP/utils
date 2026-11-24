"""numba kernels for `clahe_equalize_adapthist.clahe_u8` (imported lazily).

Importing this module raises ImportError when numba is missing, which makes
`clahe_u8` fall back to its NumPy backend. The arithmetic mirrors
`_clahe_u8_numpy` operation by operation (float32 leading-axis LUT blend,
round-half-even to int16, 7-bit fixed-point lerps), so both backends are
bit-identical.
"""
import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def tile_hist(img, kz, ky, kx, nz, ny, nx):
    """(nz*ny*nx, 256) histograms of origin-anchored tiles of a 3D uint8 volume.

    Indices past the end reflect (numpy 'reflect' / OpenCV BORDER_REFLECT_101),
    which completes trailing partial tiles without padding a copy.
    """
    Z, H, W = img.shape
    n_tiles = nz * ny * nx
    hist = np.zeros((n_tiles, 256), np.int64)
    for t in numba.prange(n_tiles):
        iz = t // (ny * nx)
        iy = (t // nx) % ny
        ix = t % nx
        h = hist[t]
        for z in range(iz * kz, iz * kz + kz):
            zz = z if z < Z else 2 * (Z - 1) - z
            for y in range(iy * ky, iy * ky + ky):
                yy = y if y < H else 2 * (H - 1) - y
                for x in range(ix * kx, ix * kx + kx):
                    xx = x if x < W else 2 * (W - 1) - x
                    h[img[zz, yy, xx]] += 1
    return hist


@numba.njit(cache=True)
def _slice_lut(lut, t, w):
    """float32 blend of LUT planes t and t + 1, rounded half-even to int16."""
    lo = lut[t]
    hi = lut[t + 1]
    gy, gx, nb = lo.shape
    sl = np.empty((gy, gx, nb), np.int16)
    for i in range(gy):
        for j in range(gx):
            for v in range(nb):
                a = lo[i, j, v]
                if w != 0:
                    a = a + w * (hi[i, j, v] - a)
                sl[i, j, v] = np.int16(np.rint(a))
    return sl


@numba.njit(cache=True)
def _map_row(src, dst, sl, t0, wyy, tx, wx, w_bits):
    """Fixed-point bilinear mapping of one image row (4 LUT lookups / pixel)."""
    half = 1 << (w_bits - 1)
    for x in range(src.shape[0]):
        t1 = tx[x]
        wxx = wx[x]
        v = src[x]
        a = np.int32(sl[t0, t1, v])
        b = np.int32(sl[t0, t1 + 1, v])
        top = a + (((b - a) * wxx + half) >> w_bits)
        c = np.int32(sl[t0 + 1, t1, v])
        d = np.int32(sl[t0 + 1, t1 + 1, v])
        bot = c + (((d - c) * wxx + half) >> w_bits)
        dst[x] = np.uint8(top + (((bot - top) * wyy + half) >> w_bits))


@numba.njit(parallel=True, cache=True)
def _interp_by_slice(img, lut, tz, wz, ty, wy, tx, wx, w_bits, out):
    for z in numba.prange(img.shape[0]):
        sl = _slice_lut(lut, tz[z], wz[z])
        for y in range(img.shape[1]):
            _map_row(img[z, y], out[z, y], sl, ty[y], wy[y], tx, wx, w_bits)


@numba.njit(parallel=True, cache=True)
def _interp_by_row(img, lut, tz, wz, ty, wy, tx, wx, w_bits, out):
    for z in range(img.shape[0]):
        sl = _slice_lut(lut, tz[z], wz[z])
        for y in numba.prange(img.shape[1]):
            _map_row(img[z, y], out[z, y], sl, ty[y], wy[y], tx, wx, w_bits)


def interp(img, lut, tz, wz, ty, wy, tx, wx, w_bits):
    """Map a 3D uint8 volume through edge-padded tile LUTs (gz, gy, gx, 256).

    Per Z slice: blend the two neighbouring LUT planes in float32 and round to
    int16 (as the NumPy backend), then bilinear interpolation in fixed point.
    Threads split slices, or rows when there are fewer slices than threads
    (2D images, thin stacks).
    """
    out = np.empty(img.shape, np.uint8)
    if img.shape[0] >= 2 * numba.get_num_threads():
        _interp_by_slice(img, lut, tz, wz, ty, wy, tx, wx, w_bits, out)
    else:
        _interp_by_row(img, lut, tz, wz, ty, wy, tx, wx, w_bits, out)
    return out
