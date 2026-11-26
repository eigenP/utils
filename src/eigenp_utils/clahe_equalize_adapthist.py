# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scikit-image",
# ]
# ///

"""
Adapted code from "Contrast Limited Adaptive Histogram Equalization" by Karel
Zuiderveld <karel@cv.ruu.nl>, Graphics Gems IV, Academic Press, 1994.

http://tog.acm.org/resources/GraphicsGems/

The Graphics Gems code is copyright-protected.  In other words, you cannot
claim the text of the code as your own and resell it. Using the code is
permitted in any program, product, or library, non-commercial or commercial.
Giving credit is not required, though is a nice gesture.  The code comes as-is,
and if there are any flaws or problems with any Gems code, nobody involved with
Gems - authors, editors, publishers, or webmasters - are to be held
responsible.  Basically, don't be a jerk, and remember that anything free
comes with no guarantee.
"""
import numbers
import warnings

import numpy as np

try:
    from ._conventions import (
        _degenerate_fallback, check_zyx_input, robust_range, sample_values, warn_deprecated)
except ImportError:  # run as a standalone script: load _conventions.py from next to this file
    import importlib.util as _ilu
    import pathlib as _pathlib

    _spec = _ilu.spec_from_file_location(
        "_eigenp_conventions", _pathlib.Path(__file__).resolve().with_name("_conventions.py"))
    _conv = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_conv)
    _degenerate_fallback, check_zyx_input, robust_range, sample_values, warn_deprecated = (
        _conv._degenerate_fallback, _conv.check_zyx_input, _conv.robust_range,
        _conv.sample_values, _conv.warn_deprecated)

# scikit-image is imported inside the legacy `_my_clahe_` only, so `clahe_u8`
# (numba / NumPy) works in minimal environments without scikit-image.





def _my_clahe_(image, kernel_size=None,
                       clip_limit=0.01, nbins=256, output_float_dtype = np.float16):
    """Contrast Limited Adaptive Histogram Equalization (CLAHE).

    .. deprecated::
        Emits a FutureWarning; removed in eigenp_utils 0.1.0 (D08). Use
        :func:`clahe_u8`: same algorithm and ``clip_limit`` scale, N-D,
        5-150x faster, robust percentile input scaling. It returns uint8 in
        [0, 255] rather than float in [0, 1] (divide by 255 if needed). For a
        float result faithful to skimage, use
        ``skimage.exposure.equalize_adapthist`` (but note that there
        ``clip_limit=0`` means unclipped AHE, while here it means identity).

    An algorithm for local contrast enhancement, that uses histograms computed
    over different tile regions of the image. Local details can therefore be
    enhanced even in regions that are darker or lighter than most of the image.

    Parameters
    ----------
    image : (N1, ...,NN) ndarray
        Input image. Every axis is treated as spatial: there is no color-channel
        handling, so an (M, N, 3) RGB image is equalised as a 3-slice volume,
        which mixes the channels and shifts hues. To equalise a color image,
        run this on its HSV V channel (e.g. ``skimage.color.rgb2hsv``) and convert
        back.
    kernel_size : int or array_like, optional
        Defines the shape of contextual regions used in the algorithm. If
        iterable is passed, it must have the same number of elements as
        ``image.ndim``. If integer, it is broadcasted to each `image`
        dimension. By default, ``kernel_size`` is 1/8 of ``image`` height by
        1/8 of its width.
    clip_limit : float or None, optional
        Clipping limit, normalized between 0 and 1 (higher values give more
        contrast). 0.0 gives Identity (min contrast). 1.0 (or None) gives
        standard AHE (max contrast).
    nbins : int, optional
        Number of gray bins for histogram ("data range").
    nbins = 2 ** 8 : number of grayscale levels to use in CLAHE algorithm


    Returns
    -------
    out : (N1, ...,NN) ndarray
        Equalized image with float16 dtype.

    See Also
    --------
    equalize_hist, rescale_intensity

    Notes
    -----
    Adapted from ``skimage.exposure.equalize_adapthist``, without its
    color-image (HSV) handling.

    .. versionchanged:: 0.17
        The values returned by this function are slightly shifted upwards
        because of an internal change in rounding behavior.

    References
    ----------
    .. [1] http://tog.acm.org/resources/GraphicsGems/
    .. [2] https://en.wikipedia.org/wiki/CLAHE#CLAHE
    """

    warn_deprecated(
        "_my_clahe_",
        "clahe_u8 (same clip_limit scale, much faster, returns uint8 in "
        "[0, 255] instead of float in [0, 1])",
    )
    from skimage.exposure import rescale_intensity
    from skimage.util import img_as_uint

    float_dtype = output_float_dtype
    image = img_as_uint(image)
    image = np.round(
        rescale_intensity(image, out_range=(0, nbins - 1))
    ).astype(np.min_scalar_type(nbins))

    if kernel_size is None:
        kernel_size = tuple([max(s // 8, 1) for s in image.shape])
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * image.ndim
    elif len(kernel_size) != image.ndim:
        raise ValueError(f'Incorrect value of `kernel_size`: {kernel_size}')

    kernel_size = [int(k) for k in kernel_size]

    image = _clahe(image, kernel_size, clip_limit, nbins)
    image = image.astype(float_dtype, copy=False)
    return rescale_intensity(image)


def _clahe(image, kernel_size, clip_limit, nbins):
    """Contrast Limited Adaptive Histogram Equalization.

    Parameters
    ----------
    image : (N1,...,NN) ndarray
        Input image.
    kernel_size : int or N-tuple of int
        Defines the shape of contextual regions used in the algorithm.
    clip_limit : float
        Normalized clipping limit between 0 and 1 (higher values give more
        contrast).
    nbins : int
        Number of gray bins for histogram ("data range").

    Returns
    -------
    out : (N1,...,NN) ndarray
        Equalized image.

    The number of "effective" graylevels in the output image is set by `nbins`;
    selecting a small value (e.g. 128) speeds up processing and still produces
    an output image of good quality. A clip limit of 1.0 (or None) results
    in standard (non-contrast limited) AHE. A clip limit of 0.0 results in
    Identity (minimum contrast).
    """
    ndim = image.ndim
    dtype = image.dtype

    # pad the image such that the shape in each dimension
    # - is a multiple of the kernel_size and
    # - is preceded by half a kernel size
    pad_start_per_dim = [k // 2 for k in kernel_size]

    pad_end_per_dim = [(k - s % k) % k + int(np.ceil(k / 2.))
                       for k, s in zip(kernel_size, image.shape)]

    image = np.pad(image, [[p_i, p_f] for p_i, p_f in
                           zip(pad_start_per_dim, pad_end_per_dim)],
                   mode='reflect')

    # The image was already quantised to [0, nbins - 1] by the caller, so each
    # gray level is its own histogram bin. (skimage quantises to NR_OF_GRAY
    # levels and bins with 1 + NR_OF_GRAY // nbins; substituting nbins for
    # NR_OF_GRAY there gave bin_size == 2 and used only half the bins.)

    # calculate graylevel mappings for each contextual region
    # rearrange image into flattened contextual regions
    ns_hist = [int(s / k) - 1 for s, k in zip(image.shape, kernel_size)]
    hist_blocks_shape = np.array([ns_hist, kernel_size]).T.flatten()
    hist_blocks_axis_order = np.array([np.arange(0, ndim * 2, 2),
                                       np.arange(1, ndim * 2, 2)]).flatten()
    hist_slices = [slice(k // 2, k // 2 + n * k)
                   for k, n in zip(kernel_size, ns_hist)]
    hist_blocks = image[tuple(hist_slices)].reshape(hist_blocks_shape)
    hist_blocks = np.transpose(hist_blocks, axes=hist_blocks_axis_order)
    hist_block_assembled_shape = hist_blocks.shape
    hist_blocks = hist_blocks.reshape((np.prod(ns_hist), -1))

    # Calculate actual clip limit
    # matth: Ensure monotonicity.
    # clip_limit=0.0 -> clim=1 (Identity, min contrast)
    # clip_limit=1.0 -> clim=N (AHE, max contrast)
    # clip_limit=None -> clim=N (AHE, explicit)

    n_pixels = np.prod(kernel_size)

    if clip_limit is None:
        clim = n_pixels
    else:
        clim = int(np.clip(clip_limit * n_pixels, 1, n_pixels))

    hist = np.apply_along_axis(np.bincount, -1, hist_blocks, minlength=nbins)
    hist = np.apply_along_axis(clip_histogram, -1, hist, clip_limit=clim)
    hist = map_histogram(hist, 0, nbins - 1, np.prod(kernel_size))
    hist = hist.reshape(hist_block_assembled_shape[:ndim] + (-1,))

    # duplicate leading mappings in each dim
    map_array = np.pad(hist,
                       [[1, 1] for _ in range(ndim)] + [[0, 0]],
                       mode='edge')

    # Perform multilinear interpolation of graylevel mappings
    # using the convention described here:
    # https://en.wikipedia.org/w/index.php?title=Adaptive_histogram_
    # equalization&oldid=936814673#Efficient_computation_by_interpolation

    # rearrange image into blocks for vectorized processing
    ns_proc = [int(s / k) for s, k in zip(image.shape, kernel_size)]
    blocks_shape = np.array([ns_proc, kernel_size]).T.flatten()
    blocks_axis_order = np.array([np.arange(0, ndim * 2, 2),
                                  np.arange(1, ndim * 2, 2)]).flatten()
    blocks = image.reshape(blocks_shape)
    blocks = np.transpose(blocks, axes=blocks_axis_order)
    blocks_flattened_shape = blocks.shape
    blocks = np.reshape(blocks, (np.prod(ns_proc),
                                 np.prod(blocks.shape[ndim:])))

    # calculate interpolation coefficients
    coeffs = np.meshgrid(*tuple([np.arange(k) / k
                                 for k in kernel_size[::-1]]), indexing='ij')
    coeffs = [np.transpose(c).flatten() for c in coeffs]
    inv_coeffs = [1 - c for dim, c in enumerate(coeffs)]

    # sum over contributions of neighboring contextual
    # regions in each direction
    result = np.zeros(blocks.shape, dtype=np.float32)
    for iedge, edge in enumerate(np.ndindex(*([2] * ndim))):

        edge_maps = map_array[tuple([slice(e, e + n)
                                     for e, n in zip(edge, ns_proc)])]
        edge_maps = edge_maps.reshape((np.prod(ns_proc), -1))

        # apply map
        edge_mapped = np.take_along_axis(edge_maps, blocks, axis=-1)

        # interpolate
        edge_coeffs = np.prod([[inv_coeffs, coeffs][e][d]
                                  for d, e in enumerate(edge[::-1])], 0)

        result += (edge_mapped * edge_coeffs).astype(result.dtype)

    result = result.astype(dtype)

    # rebuild result image from blocks
    result = result.reshape(blocks_flattened_shape)
    blocks_axis_rebuild_order =\
        np.array([np.arange(0, ndim),
                  np.arange(ndim, ndim * 2)]).T.flatten()
    result = np.transpose(result, axes=blocks_axis_rebuild_order)
    result = result.reshape(image.shape)

    # undo padding
    unpad_slices = tuple([slice(p_i, s - p_f) for p_i, p_f, s in
                          zip(pad_start_per_dim, pad_end_per_dim,
                              image.shape)])
    result = result[unpad_slices]

    return result


def clip_histogram(hist, clip_limit):
    """Perform clipping of the histogram and redistribution of bins.

    The histogram is clipped and the number of excess pixels is counted.
    Afterwards the excess pixels are equally redistributed across the
    whole histogram (providing the bin count is smaller than the cliplimit).

    Parameters
    ----------
    hist : ndarray
        Histogram array.
    clip_limit : int
        Maximum allowed bin count.

    Returns
    -------
    hist : ndarray
        Clipped histogram.
    """
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit
    hist[excess_mask] = clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = n_excess // hist.size  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = np.logical_and(hist >= upper, hist < clip_limit)
    mid = hist[mid_mask]
    n_excess += mid.sum() - mid.size * clip_limit
    hist[mid_mask] = clip_limit

    while n_excess > 0:  # Redistribute remaining excess
        prev_n_excess = n_excess
        for index in range(hist.size):
            under_mask = hist < clip_limit
            step_size = max(1, np.count_nonzero(under_mask) // n_excess)
            under_mask = under_mask[index::step_size]
            hist[index::step_size][under_mask] += 1
            n_excess -= np.count_nonzero(under_mask)
            if n_excess <= 0:
                break
        if prev_n_excess == n_excess:
            break

    return hist


def map_histogram(hist, min_val, max_val, n_pixels):
    """Calculate the equalized lookup table (mapping).

    It does so by cumulating the input histogram.
    Histogram bins are assumed to be represented by the last array dimension.

    Parameters
    ----------
    hist : ndarray
        Clipped histogram.
    min_val : int
        Minimum value for mapping.
    max_val : int
        Maximum value for mapping.
    n_pixels : int
        Number of pixels in the region.

    Returns
    -------
    out : ndarray
       Mapped intensity LUT.
    """
    out = np.cumsum(hist, axis=-1).astype(float)
    out *= (max_val - min_val) / n_pixels
    out += min_val
    np.clip(out, a_min=None, a_max=max_val, out=out)

    return out.astype(int)


# ---------------------------------------------------------------------------
# Fast 8-bit CLAHE (segmentation preprocessing)
# ---------------------------------------------------------------------------

_NBINS_U8 = 256
# Values sampled (fixed seed) to estimate the percentiles of non-uint8 input
_SAMPLE_U8 = 1 << 22


def _to_uint8(image, percentiles):
    """uint8 passes through; anything else is rescaled between two percentiles."""
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.dtype == bool:
        return image.astype(np.uint8) * 255
    if image.dtype in (np.uint16, np.int16, np.int8):
        return _small_int_to_uint8(image, percentiles)
    # Percentiles on a fixed-seed random sample of <= 4M values (a strided
    # sample aliases with image columns, R6-13): exact enough for
    # segmentation bounds, and O(1) in the volume size beyond that. A
    # degenerate range (sparse signal) falls back to the data maximum, with a
    # warning, instead of a black image (R6-02).
    if sample_values(image, n=_SAMPLE_U8).size == 0:
        return np.zeros(image.shape, np.uint8)
    lo, hi = robust_range(image, *percentiles, n=_SAMPLE_U8, stacklevel=3)
    out = np.empty(image.shape, np.float32)
    if hi > lo:
        np.subtract(image, lo, out=out, casting="unsafe")
        out *= np.float32(255.0 / (hi - lo))
    else:
        out.fill(0.0)
    np.nan_to_num(out, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
    np.clip(out, 0.0, 255.0, out=out)
    out += np.float32(0.5)
    return out.astype(np.uint8)


def _small_int_to_uint8(image, percentiles):
    """<= 16-bit integers: percentiles from a histogram, then one LUT gather."""
    info = np.iinfo(image.dtype)
    offset = -int(info.min)
    codes = image.reshape(-1)
    if offset:
        codes = codes.view(np.dtype(image.dtype.str.replace('i', 'u')))
        # two's complement view: v + offset == (code ^ sign bit)
        sign = np.array(1 << (8 * image.dtype.itemsize - 1), dtype=codes.dtype)
        codes = codes ^ sign
    # histogram of a fixed-seed random sample (<= 4M values), as for floats
    if codes.size > _SAMPLE_U8:
        sample = codes[np.random.default_rng(0).integers(0, codes.size, _SAMPLE_U8)]
    else:
        sample = codes
    counts = np.bincount(sample, minlength=info.max + offset + 1)
    cdf = np.cumsum(counts)
    # 'lower'-style percentiles on the exact histogram
    ranks = np.asarray(percentiles, float) / 100 * (cdf[-1] - 1)
    lo, hi = (int(v) for v in np.searchsorted(cdf, ranks, side='right'))
    if hi <= lo and codes.size:
        lo, hi = _degenerate_fallback(lo, hi, int(codes.min()), int(codes.max()),
                                      *percentiles, stacklevel=4)
    levels = np.arange(counts.size, dtype=np.float32)
    if hi > lo:
        lut = (levels - lo) * np.float32(255.0 / (hi - lo))
        np.clip(lut, 0, 255, out=lut)
        lut = np.rint(lut).astype(np.uint8)
    else:
        lut = np.zeros(counts.size, np.uint8)
    return np.take(lut, codes, mode='clip').reshape(image.shape)


def _parse_kernel_size(kernel_size, shape):
    ndim = len(shape)
    if kernel_size is None:
        kernel_size = tuple(max(s // 8, 1) for s in shape)
    elif isinstance(kernel_size, numbers.Number):
        kernel_size = (kernel_size,) * ndim
    elif len(kernel_size) != ndim:
        raise ValueError(f'Incorrect value of `kernel_size`: {kernel_size}')
    return tuple(int(min(max(int(k), 1), s)) for k, s in zip(kernel_size, shape))


def _tile_grid(shape, ks):
    """Number of origin-anchored tiles per axis (trailing partial tiles count)."""
    return [-(-s // k) for s, k in zip(shape, ks)]


def _tile_hist_u8_numpy(img, ks):
    """(n_tiles, 256) int64 histograms of origin-anchored tiles, C order.

    Trailing partial tiles are completed by reflection (numpy 'reflect', i.e.
    OpenCV's BORDER_REFLECT_101).
    """
    ndim = img.ndim
    nt = _tile_grid(img.shape, ks)
    pad = [(0, n * k - s) for n, k, s in zip(nt, ks, img.shape)]
    if any(p for _, p in pad):
        img = np.pad(img, pad, mode='reflect')

    # (n0, k0, n1, k1, ...) -> (n0, n1, ..., k0, k1, ...) -> (T, K); uint8 copy
    split = [d for n, k in zip(nt, ks) for d in (n, k)]
    order = list(range(0, 2 * ndim, 2)) + list(range(1, 2 * ndim, 2))
    blocks = img.reshape(split).transpose(order).reshape(int(np.prod(nt)), -1)
    n_tiles, n_px = blocks.shape

    hist = np.empty((n_tiles, _NBINS_U8), np.int64)
    if n_px >= 1024:
        # bincount on uint8 is ~3x faster than on a widened tile*256 + v index
        for t in range(n_tiles):
            hist[t] = np.bincount(blocks[t], minlength=_NBINS_U8)
    else:
        # Many small tiles: batch them (offset by tile * 256) to amortise calls
        per_call = max(1, (1 << 20) // n_px)
        for t0 in range(0, n_tiles, per_call):
            b = blocks[t0:t0 + per_call]
            idx = b + (np.arange(b.shape[0], dtype=np.int32) * _NBINS_U8)[:, None]
            hist[t0:t0 + b.shape[0]] = np.bincount(
                idx.ravel(), minlength=b.shape[0] * _NBINS_U8
            ).reshape(-1, _NBINS_U8)
    return hist


def _hist_to_lut(hist, nt, n_px, clip_limit):
    """Clip, redistribute and cumulate tile histograms into float32 LUTs.

    Returns shape (n_0 + 2, ..., n_d + 2, 256): the tile grid edge-padded by
    one tile per axis, so interpolation never needs bounds checks.
    """
    ndim = len(nt)
    if clip_limit is not None:
        # Normalised limit as in _my_clahe_: 0 -> identity, 1 -> AHE. Unlike
        # skimage and OpenCV, where 0 disables clipping (AHE, R6-08); the
        # count is an integer, as in both of them.
        clim = int(np.clip(int(clip_limit * n_px), 1, n_px))
        # OpenCV's single clip + redistribution pass, vectorised over tiles:
        # every bin gets excess // 256, then the remainder r adds 1 to bins
        # 0, s, 2s, ... (r of them) with s = max(256 // r, 1). Bins may end
        # slightly above the limit (skimage iterates to remove that).
        excess = np.maximum(hist - clim, 0).sum(axis=1, keepdims=True)
        hist = np.minimum(hist, clim)  # do not modify the caller's array
        batch, residual = np.divmod(excess, _NBINS_U8)
        hist += batch
        step = np.maximum(_NBINS_U8 // np.maximum(residual, 1), 1)
        bins = np.arange(_NBINS_U8)[None, :]
        hist = hist + ((bins % step == 0) & (bins // step < residual))

    lut = np.cumsum(hist, axis=1).astype(np.float32)
    lut *= np.float32((_NBINS_U8 - 1) / n_px)
    np.clip(lut, 0, _NBINS_U8 - 1, out=lut)
    lut = lut.reshape(tuple(nt) + (_NBINS_U8,))
    return np.pad(lut, [(1, 1)] * ndim + [(0, 0)], mode='edge')


def _tile_luts_u8(img, ks, clip_limit):
    """Edge-padded float32 tile LUTs (NumPy histograms)."""
    hist = _tile_hist_u8_numpy(img, ks)
    return _hist_to_lut(hist, _tile_grid(img.shape, ks), int(np.prod(ks)), clip_limit)


def _interp_coords(size, k, centred=False):
    """Padded-grid lower tile index and weight for each coordinate.

    ``centred=False`` is OpenCV's convention, f = i / k - 0.5 (used on Y/X so
    2D output matches OpenCV). ``centred=True`` measures from the true tile
    centre, f = (i + 0.5) / k - 0.5 (used on the folded leading axes, D20): a
    1-voxel tile then gives weight 0, so thin Z (or C, T) tiles do not blend
    50% of the neighbouring slice's LUT, and flipping the axis commutes with
    CLAHE when the axis is a multiple of the tile.
    """
    i = np.arange(size, dtype=np.float64)
    f = ((i + 0.5) / k - 0.5) if centred else (i / k - 0.5)
    t = np.floor(f)
    w = (f - t).astype(np.float32)
    return (t.astype(np.int64) + 1), w


# Fixed-point interpolation weights: (b - a) * w must fit int16 for
# |b - a| <= 255, so w has 7 fractional bits (255 * 128 = 32640).
_W_BITS = 7
_W_ONE = 1 << _W_BITS
_W_HALF = 1 << (_W_BITS - 1)


def _clahe_u8_numpy(img, ks, clip_limit, chunk_px=1 << 16):
    """CLAHE mapping + interpolation in int16 fixed point.

    The leading axes are folded into one (gy, gx, 256) LUT per 2D slice, then
    each pixel needs 4 gathers and 3 lerps. int16 (vs float32) halves the
    memory traffic of this bandwidth-bound loop (~2.5x faster); the cost is
    uint8-rounded LUTs (as in OpenCV) and 7-bit weights, i.e. <= ~1 gray level.
    """
    lut = _tile_luts_u8(img, ks, clip_limit)
    *lead_shape, H, W = img.shape
    gx = lut.shape[-2]

    ty, wy = _interp_coords(H, ks[-2])
    tx, wx = _interp_coords(W, ks[-1])
    # Flat index of the (ty, tx) corner in a (gy, gx, 256) slice LUT
    base = ((ty * gx).astype(np.int32)[:, None] + tx.astype(np.int32)[None, :])
    base *= _NBINS_U8
    o_x, o_y = _NBINS_U8, gx * _NBINS_U8
    wx = np.round(wx * _W_ONE).astype(np.int16)[None, :]
    wy = np.round(wy * _W_ONE).astype(np.int16)[:, None]

    lead_coords = [_interp_coords(s, k, centred=True) for s, k in zip(lead_shape, ks[:-2])]
    rows = max(1, chunk_px // W)  # keep temporaries cache-resident
    out = np.empty(img.shape, np.uint8)

    def lerp_(a, b, w):
        # a += round((b - a) * w / 2**_W_BITS), in place; b is scratch
        b -= a
        b *= w
        b += _W_HALF
        b >>= _W_BITS
        a += b

    for lead in np.ndindex(*lead_shape):
        # Multilinear interpolation of the LUT grid along the leading axes
        # (tiny: gy * gx * 256 values) leaves a 2D bilinear problem per slice.
        slab = lut
        for d, i in enumerate(lead):
            t, w = lead_coords[d][0][i], lead_coords[d][1][i]
            lo, hi = slab[t], slab[t + 1]
            slab = lo + w * (hi - lo) if w else lo
        flat = np.rint(slab).astype(np.int16).reshape(-1)
        f_00, f_01, f_10, f_11 = flat, flat[o_x:], flat[o_y:], flat[o_y + o_x:]
        src, dst = img[lead], out[lead]

        for r0 in range(0, H, rows):
            r1 = min(r0 + rows, H)
            idx = base[r0:r1] + src[r0:r1]
            # indices are in range by construction; mode='clip' skips the
            # bounds check (~25% faster than the default 'raise')
            top = np.take(f_00, idx, mode='clip')
            lerp_(top, np.take(f_01, idx, mode='clip'), wx)
            bot = np.take(f_10, idx, mode='clip')
            lerp_(bot, np.take(f_11, idx, mode='clip'), wx)
            lerp_(top, bot, wy[r0:r1])
            dst[r0:r1] = top
    return out


def _numba_kernels():
    """The compiled numba backend, or None when numba is not importable."""
    try:
        from . import _clahe_numba
    except ImportError:
        return None
    return _clahe_numba


def _clahe_u8_numba(img, ks, clip_limit, kernels):
    """Fused, multithreaded equivalent of `_clahe_u8_numpy` for ndim <= 3."""
    # Treat 1D/2D as (1, ..., N) volumes with unit kernels on the added axes
    lead = 3 - img.ndim
    vol = img.reshape((1,) * lead + img.shape)
    ks3 = (1,) * lead + tuple(ks)
    nt = _tile_grid(vol.shape, ks3)

    hist = kernels.tile_hist(np.ascontiguousarray(vol), *ks3, *nt)
    lut = _hist_to_lut(hist, nt, int(np.prod(ks3)), clip_limit)

    # Z (the folded leading axis) is pixel-centred, Y/X follow OpenCV (D20)
    (tz, wz), (ty, wy), (tx, wx) = (_interp_coords(s, k, centred=(d == 0))
                                    for d, (s, k) in enumerate(zip(vol.shape, ks3)))
    wy = np.round(wy * _W_ONE).astype(np.int32)
    wx = np.round(wx * _W_ONE).astype(np.int32)
    out = kernels.interp(vol, lut, tz, wz, ty, wy, tx, wx, _W_BITS)
    return out.reshape(img.shape)


def clahe_u8(image, kernel_size=None, clip_limit=0.01,
             percentiles=(0.1, 99.9), backend='auto'):
    """Fast N-D CLAHE on 8 bits, for segmentation preprocessing (uint8 in and out).

    Same contextual-region algorithm as ``_my_clahe_`` / skimage, but it trades
    exactness for speed: 256 bins, OpenCV's tiling (anchored at the origin,
    trailing partial tiles completed by reflection) and its single-pass clip
    redistribution, int16 fixed-point interpolation, uint8 output. The leading
    axes are folded into one LUT per 2D slice, so every pixel needs 4 lookups
    whatever the dimensionality.

    **Every axis is spatial.** There is no channel axis: contextual regions
    span all axes, so a (C, Y, X) or (T, Y, X) stack shares histograms across
    C or T unless its kernel_size is 1 on that axis, and a channel-last RGB
    (Y, X, 3) image mixes its colours (a UserWarning flags a trailing axis of
    length 3 or 4). Put channels first and use ``kernel_size=(1, ky, kx)``,
    or call this per channel.

    Tile coordinates: Y and X (the last two axes) use OpenCV's convention, so
    2D output matches OpenCV. The leading axes are pixel-centred, so with a
    1-voxel tile along Z (the default when Z < 16) each slice uses its own
    LUT, and flipping Z commutes with CLAHE when Z is a multiple of the tile.
    Changed (D20, 2026-11): the leading axes used OpenCV's half-pixel shift,
    which blended 50% of the previous slice's (or channel's) LUT into each
    slice when the tile was 1 voxel thick, and made the result depend on the
    Z direction (up to 64 grey levels). 2D output is unchanged, bit for bit.
    Changed (C-07 / R6-15, 2026-11): a UserWarning is now emitted for a
    trailing axis of length 3 or 4 (a likely channel-last colour axis) and
    for NaN input (mapped to 0, as before); the output is unchanged.

    Two backends give bit-identical results:

    * ``'numba'`` (used by ``'auto'`` when numba is importable and
      ``image.ndim <= 3``): fused, multithreaded loops, ~1 ns/voxel on 4 cores.
      The first call in a fresh environment compiles for a few seconds; the
      compiled code is cached on disk. Threads follow numba's settings
      (``NUMBA_NUM_THREADS`` / ``numba.set_num_threads``).
    * ``'numpy'``: vectorised fallback (Pyodide, no numba, ndim > 3), ~5-10x
      slower.

    Parameters
    ----------
    image : (N1, ..., NN) ndarray
        uint8 is used as is (no stretching, so dim uint8 data stay dim while
        the same counts in uint16 are stretched). bool maps to {0, 255}. Any
        other dtype is mapped to [0, 255] between the ``percentiles`` (so
        isolated hot/dead pixels do not set the range). NaN pixels become 0
        before equalisation, with a UserWarning: they then take part in the
        tile histograms and come out as equalised grey levels, so mask them
        afterwards if needed (uint8 cannot hold NaN).
    kernel_size : int or sequence of int, optional
        Tile shape. Default 1/8 of each dimension (at least 1).
    clip_limit : float or None
        Normalised clip limit as in ``_my_clahe_``: 1 or None = AHE (maximum
        contrast), equivalent to OpenCV's ``clipLimit = 256 * clip_limit``.
        **0 means identity (minimum contrast) here, unlike skimage and
        OpenCV, where 0 disables clipping and gives AHE.** Use None for AHE;
        ``surface_extraction`` relies on 0 meaning "off".
    percentiles : (float, float)
        Input range for non-uint8 images, default (0.1, 99.9): the processing
        default shared with ``intensity_rescaling`` (D02/D27; the viewers
        display up to p99.99 instead). Estimated on a fixed-seed random
        sample of at most 4M values. When the two percentiles are equal
        (signal on fewer than (100 - upper)% of the voxels) the range falls
        back to the data maximum (and minimum), with a UserWarning.
        Changed (D02, 2026-11): that case used to give an all-black input
        (all-zero for integer input); the sample used to be strided, which
        aliases with image columns.
    backend : {'auto', 'numba', 'numpy'}

    Returns
    -------
    out : ndarray of uint8, same shape as ``image``.

    Changed (D31, 2026-11): an xarray.DataArray with more than 3 dims (e.g. the
    STCZYX output of load_image) raises TypeError; select one image with
    ``eigenp_utils.image_io.to_zyx`` first. 2-D / 3-D DataArrays work as before.
    """
    check_zyx_input(image, "clahe_u8")
    if backend not in ('auto', 'numba', 'numpy'):
        raise ValueError(f'Unknown backend: {backend!r}')
    image = np.asarray(image)
    if image.ndim >= 3 and image.shape[-1] in (3, 4):
        warnings.warn(
            f"clahe_u8 tiles every axis, including the trailing axis of length "
            f"{image.shape[-1]} of this {image.shape} image: if it holds colour "
            "channels (RGB/RGBA), they are mixed. Put channels first and use "
            "kernel_size=(1, ky, kx), or equalise each channel separately.",
            UserWarning, stacklevel=2)
    if image.dtype.kind == 'f':
        n_nan = int(np.count_nonzero(np.isnan(image)))
        if n_nan:
            warnings.warn(
                f"clahe_u8: {n_nan} NaN pixels are mapped to 0 and equalised with "
                "the rest (uint8 output cannot hold NaN); mask them afterwards "
                "if needed.", UserWarning, stacklevel=2)
    img = _to_uint8(image, percentiles)
    if img.ndim == 0:
        raise ValueError('`image` must have at least one dimension')
    ks = _parse_kernel_size(kernel_size, img.shape)
    if img.size == 0:
        # A size-0 axis gives a 0-wide tile (no tile grid); pass empty input
        # through, as median_filter_fast does
        return np.zeros(img.shape, np.uint8)

    if backend != 'numpy':
        kernels = _numba_kernels()
        if backend == 'numba':
            if kernels is None:
                raise ImportError("backend='numba' requires numba")
            if img.ndim > 3:
                raise ValueError("backend='numba' supports up to 3 dimensions")
        if kernels is not None and img.ndim <= 3:
            return _clahe_u8_numba(img, ks, clip_limit, kernels)

    if img.ndim == 1:
        return _clahe_u8_numpy(img[None], (1,) + ks, clip_limit)[0]
    return _clahe_u8_numpy(img, ks, clip_limit)
