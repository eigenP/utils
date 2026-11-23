"""Minimal 2D tile stitching for STCZYX xarray images (see :func:`stitch_tiles`)."""

import warnings

import numpy as np
import xarray as xr

from .image_io import _STCZYX, array_to_stczyx_xarray, get_stczyx_pixel_sizes


def _overlap(a0, a1, b0, b1):
    """Intersection [max(a0, b0), min(a1, b1)) of two 1D intervals (may be empty)."""
    return max(a0, b0), min(a1, b1)


def _register_pair(img_i, img_j, offset, min_overlap, max_shift):
    """Refine the (dy, dx) offset of tile j relative to tile i by NCC template matching.

    ``offset`` is the predicted top-left of j minus that of i, in pixels. The
    predicted overlap of j is matched inside tile i's overlap grown by the
    search radius ``m = min(max_shift, narrower overlap // 2)`` (shrinking the
    template where tile i ends), so every shift up to ``m`` is searched
    exhaustively with no FFT wrap-around. Only side neighbours are registered:
    a pair must overlap by at least half the tile along one axis (corner pairs
    share little texture and give biased shifts).
    Returns (refined offset, NCC at the best shift) or None.
    """
    from skimage.feature import match_template

    h, w = img_i.shape
    (y0, y1), (x0, x1) = (_overlap(0, h, offset[0], offset[0] + h),
                          _overlap(0, w, offset[1], offset[1] + w))
    oy, ox = y1 - y0, x1 - x0
    if oy < min_overlap or ox < min_overlap or (oy < h / 2 and ox < w / 2):
        return None
    # One search radius for both axes (stage errors are isotropic), capped by
    # half the narrow overlap so the template keeps at least half of it
    my = mx = min(max_shift, min(oy, ox) // 2)

    # Search region in tile i, and the template (j's predicted overlap) shrunk
    # where the search region is clipped by the edge of tile i
    sy0, sy1, sx0, sx1 = max(y0 - my, 0), min(y1 + my, h), max(x0 - mx, 0), min(x1 + mx, w)
    ty0, ty1 = y0 + (my - (y0 - sy0)), y1 - (my - (sy1 - y1))
    tx0, tx1 = x0 + (mx - (x0 - sx0)), x1 - (mx - (sx1 - x1))
    if ty1 - ty0 < min_overlap // 2 or tx1 - tx0 < min_overlap // 2:
        return None
    search = img_i[sy0:sy1, sx0:sx1]
    template = img_j[ty0 - offset[0]:ty1 - offset[0], tx0 - offset[1]:tx1 - offset[1]]
    if np.ptp(template) == 0 or np.ptp(search) == 0:
        return None  # featureless: nothing to register

    # img_j(v) = img_i(v + offset + e): the template sits at (ty0, tx0) + e in tile i
    score = match_template(search, template)
    peak = np.unravel_index(np.argmax(score), score.shape)
    e = np.array([sy0 + peak[0] - ty0, sx0 + peak[1] - tx0])
    if np.any(np.abs(e) >= my):
        return None  # best match on the search border: the true shift is likely outside it
    return np.asarray(offset) + e, float(score[peak])


def _solve_positions(prior, edges, prior_weight, max_residual):
    """Weighted least squares for tile positions, dropping the worst edge until all fit.

    Minimises sum_ij w_ij |t_j - t_i - d_ij|^2 + prior_weight * sum_i |t_i - prior_i|^2,
    independently for y and x. Returns (positions, kept edges).
    """
    n = len(prior)
    edges = list(edges)
    while True:
        rows, rhs = [], []
        for i, j, d, w in edges:
            row = np.zeros(n)
            row[j], row[i] = 1.0, -1.0
            rows.append(np.sqrt(w) * row)
            rhs.append(np.sqrt(w) * np.asarray(d, float))
        sw = np.sqrt(prior_weight)
        rows.extend(sw * np.eye(n))
        rhs.extend(sw * prior)
        positions = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
        if not edges:
            return positions, edges
        residuals = [np.linalg.norm(positions[j] - positions[i] - d) for i, j, d, _ in edges]
        worst = int(np.argmax(residuals))
        if residuals[worst] <= max_residual:
            return positions, edges
        edges.pop(worst)


def stitch_tiles(
    tiles: xr.DataArray,
    registration_channel=0,
    stage_to_image=None,
    min_overlap: int = 16,
    max_shift=None,
    min_ncc: float = 0.6,
    max_residual: float = 2.0,
    prior_weight: float = 0.01,
    blending: str = 'linear',
) -> xr.DataArray:
    """
    Stitch the S tiles of an STCZYX image into one 2D mosaic (S = 1).

    A minimal translation-only stitcher: stage positions give the initial
    placement, side-by-side neighbours are registered by normalised
    cross-correlation (template matching) on the max projection over Z and T
    of one channel, positions are solved
    globally by weighted least squares (with outlier rejection), and tiles are
    fused with linear (feathered) blending. Shifts are whole pixels; there is
    no rotation, Z alignment or per-timepoint registration. For more control
    (affine or 3D registration, lazy fusion of large datasets, napari) use
    `multiview-stitcher` (Marvin Albert, https://github.com/multiview-stitcher/multiview-stitcher).

    Apply shading (flatfield) correction first, e.g. with ``fit_basic_shading``:
    vignetting causes seams and biases the registration.

    Parameters:
    -----------
    tiles : xr.DataArray
        STCZYX image (as from :func:`array_to_stczyx_xarray`) with stage
        positions in the coordinates ``stage_y`` / ``stage_x`` along S, in the
        unit of the pixel sizes (calibrated Y/X coordinates or
        ``attrs['pixel_sizes']``). All tiles share one Y/X shape.
    registration_channel : int or str
        Channel (index, or name in the C coordinate) used for registration.
        The resulting positions are applied to all channels.
    stage_to_image : (2, 2) array, optional
        Matrix mapping stage (y, x) to image (row, col) direction, for cameras
        that are flipped or rotated relative to the stage, e.g.
        ``[[-1, 0], [0, 1]]`` when the stage Y axis points up in the image.
        Default: identity. If the stage-only placement is scrambled, try flips.
    min_overlap : int
        Minimum predicted overlap in pixels (along both Y and X) for a pair of
        tiles to be registered. Corner (diagonal) pairs, overlapping by less
        than half the tile along both axes, are never registered.
    max_shift : int, optional
        Largest accepted correction of a stage position, in pixels (the stage
        error, relative between neighbours). The search radius is also capped
        at half the overlap width, so the relative stage error must stay below
        half the overlap. Default: no cap beyond that.
    min_ncc : float (>= 0)
        Pairs whose aligned overlap has a normalised cross-correlation below
        this are discarded (e.g. empty background); such tiles keep the stage
        placement relative to their neighbours.
    max_residual : float
        Pairwise shifts disagreeing with the global solution by more than this
        many pixels are dropped one at a time (worst first) and the solve repeated.
    prior_weight : float
        Weight (> 0) of the stage positions in the global solve, relative to a
        pair with NCC = 1. Small values anchor the mosaic without overriding
        registrations.
    blending : {'linear', 'overwrite'}
        'linear' weights each tile by its distance to the tile edge;
        'overwrite' pastes tiles in S order (later tiles on top).

    Returns:
    --------
    xr.DataArray
        STCZYX mosaic with S = 1, the input dtype, physical Y/X coordinates
        starting at 0 and ``attrs['pixel_sizes']``. ``attrs['stitching']``
        records the tile positions (pixels, top-left, ``(y, x)``), the stage-only
        positions and the number of registered pairs used.
    """
    warnings.warn(
        "stitch_tiles is a minimal translation-only 2D stitcher. For more control "
        "(3D/affine registration, large datasets) consider multiview-stitcher "
        "(Marvin Albert, https://github.com/multiview-stitcher/multiview-stitcher).",
        UserWarning,
        stacklevel=2,
    )
    if tuple(tiles.dims) != tuple(_STCZYX):
        raise ValueError(
            f"tiles must have dims {tuple(_STCZYX)}, got {tuple(tiles.dims)}; "
            "convert with array_to_stczyx_xarray first."
        )
    missing = [c for c in ('stage_y', 'stage_x') if c not in tiles.coords]
    if missing:
        raise ValueError(
            f"tiles lack the coordinates {missing}; pass stage_positions to array_to_stczyx_xarray."
        )
    if blending not in ('linear', 'overwrite'):
        raise ValueError(f"blending must be 'linear' or 'overwrite', got {blending!r}.")
    sizes = get_stczyx_pixel_sizes(tiles)
    if sizes['Y'] is None or sizes['X'] is None:
        raise ValueError("tiles have no Y/X pixel sizes; pass pixel_sizes to array_to_stczyx_xarray.")
    pixel_yx = np.array([sizes['Y'], sizes['X']])

    n_tiles = tiles.sizes['S']
    h, w = tiles.sizes['Y'], tiles.sizes['X']
    if max_shift is None:
        max_shift = max(h, w)
    if not prior_weight > 0:
        raise ValueError(f"prior_weight must be > 0 (it anchors unregistered tiles), got {prior_weight}.")
    if not min_ncc >= 0:
        raise ValueError(f"min_ncc must be >= 0, got {min_ncc}.")

    # 1. Stage positions -> prior top-left pixel positions
    stage = np.column_stack([tiles.coords['stage_y'].values, tiles.coords['stage_x'].values]).astype(float)
    transform = np.eye(2) if stage_to_image is None else np.asarray(stage_to_image, float)
    if transform.shape != (2, 2):
        raise ValueError(f"stage_to_image must be a 2x2 matrix, got shape {transform.shape}.")
    prior = (stage @ transform.T) / pixel_yx
    prior -= prior.min(axis=0)

    # 2. Pairwise registration on the max projection of one channel
    if isinstance(registration_channel, str):
        reg = tiles.sel(C=registration_channel)
    else:
        reg = tiles.isel(C=registration_channel)
    projections = np.asarray(reg.max(dim=('T', 'Z')).values, dtype=np.float32)

    prior_int = np.round(prior).astype(int)
    edges = []
    for i in range(n_tiles):
        for j in range(i + 1, n_tiles):
            offset = prior_int[j] - prior_int[i]
            result = _register_pair(projections[i], projections[j], offset, min_overlap, max_shift)
            if result is not None and result[1] >= min_ncc:
                edges.append((i, j, result[0], result[1]))

    # 3. Global positions
    positions, used = _solve_positions(prior, edges, prior_weight, max_residual)
    positions = np.round(positions - positions.min(axis=0)).astype(int)

    # 4. Fusion
    out_h, out_w = positions[:, 0].max() + h, positions[:, 1].max() + w
    lead = (tiles.sizes['T'], tiles.sizes['C'], tiles.sizes['Z'])
    canvas = np.zeros(lead + (out_h, out_w), dtype=np.float64 if blending == 'linear' else tiles.dtype)
    if blending == 'linear':
        ramp_y = np.minimum(np.arange(1, h + 1), np.arange(h, 0, -1)).astype(float)
        ramp_x = np.minimum(np.arange(1, w + 1), np.arange(w, 0, -1)).astype(float)
        feather = np.outer(ramp_y, ramp_x)
        weight = np.zeros((out_h, out_w))
    for s in range(n_tiles):
        y, x = positions[s]
        tile = np.asarray(tiles.isel(S=s).values)
        if blending == 'linear':
            canvas[..., y:y + h, x:x + w] += tile * feather
            weight[y:y + h, x:x + w] += feather
        else:
            canvas[..., y:y + h, x:x + w] = tile
    if blending == 'linear':
        canvas = np.divide(canvas, weight, out=np.zeros_like(canvas), where=weight > 0)
        if np.issubdtype(tiles.dtype, np.integer):
            info = np.iinfo(tiles.dtype)
            canvas = np.clip(np.round(canvas), info.min, info.max)
        canvas = canvas.astype(tiles.dtype)

    mosaic = array_to_stczyx_xarray(
        canvas, 'TCZYX',
        channel_names=list(tiles.coords['C'].values),
        attrs={k: v for k, v in tiles.attrs.items() if k != 'pixel_sizes'},
        pixel_sizes=sizes,
    )
    mosaic = mosaic.assign_coords(T=tiles.coords['T'], Z=tiles.coords['Z'])
    mosaic.name = tiles.name
    mosaic.attrs['stitching'] = {
        'positions_px': positions.tolist(),
        'stage_positions_px': np.round(prior).astype(int).tolist(),
        'n_pairs': len(used),
        'n_pairs_registered': len(edges),
    }
    return mosaic
