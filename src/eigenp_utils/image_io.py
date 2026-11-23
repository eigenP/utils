import warnings
from pathlib import Path

import numpy as np
import xarray as xr
import tifffile
import xml.etree.ElementTree as ET


_STCZYX = 'STCZYX'


def _relabel_dims(da: xr.DataArray, axes: str) -> xr.DataArray:
    """Rename the dims of ``da`` positionally to the letters of ``axes``.

    Goes through temporary names so that permutations of existing names
    (e.g. nd2's ``('P', 'S')`` -> ``('S', 'C')``) do not collide.
    Coordinates of a dim relabelled to a different spatial letter (e.g. a
    time axis in ms relabelled as Z) are dropped: they are not positions along
    the new axis and would otherwise be read as a pixel size.
    """
    if tuple(da.dims) == tuple(axes):
        return da
    stale = [d for d, a in zip(da.dims, axes)
             if a in _SPATIAL and str(d).upper() != a and d in da.coords]
    da = da.drop_vars(stale)
    tmp = {d: f'__stczyx_tmp_{i}' for i, d in enumerate(da.dims)}
    da = da.rename(tmp)
    return da.rename({tmp[d]: a for d, a in zip(tmp, axes)})


_SPATIAL = ('Z', 'Y', 'X')


def _normalize_pixel_sizes(pixel_sizes):
    """{'Z', 'Y', 'X', 'unit'} dict (None where unknown) from a dict or a (Z, Y, X) sequence."""
    if pixel_sizes is None:
        return {'Z': None, 'Y': None, 'X': None, 'unit': None}
    if isinstance(pixel_sizes, dict):
        keyed = {str(k).upper(): v for k, v in pixel_sizes.items()}
        out = {ax: keyed.get(ax) for ax in _SPATIAL}
        out['unit'] = pixel_sizes.get('unit')
    else:
        values = list(pixel_sizes)
        if len(values) != 3:
            raise ValueError(f"pixel_sizes must be a dict or a (Z, Y, X) sequence, got {pixel_sizes!r}.")
        out = dict(zip(_SPATIAL, values))
        out['unit'] = None
    for ax in _SPATIAL:
        if out[ax] is not None:
            v = float(out[ax])
            if not (np.isfinite(v) and v > 0):
                raise ValueError(f"pixel_sizes['{ax}'] must be finite and positive, got {out[ax]!r}.")
            out[ax] = v
    return out


def _coord_spacing(da: xr.DataArray, dim: str):
    """Pixel size implied by a dim's coordinate: the step of uniformly spaced float coords, else None.

    Integer coordinates are read as indices (uncalibrated), not as physical positions.
    """
    if dim not in da.coords or da.sizes[dim] < 2:
        return None
    values = np.asarray(da.coords[dim].values)
    if not np.issubdtype(values.dtype, np.floating):
        return None
    steps = np.diff(values)
    step = float(np.median(steps))
    if not np.isfinite(step) or step == 0 or not np.allclose(steps, step, rtol=1e-3, atol=0):
        return None
    return abs(step)


def _coord_unit(da: xr.DataArray):
    for dim in _SPATIAL:
        if dim in da.coords:
            unit = da.coords[dim].attrs.get('units', da.coords[dim].attrs.get('unit'))
            if unit is not None:
                return unit
    return None


def get_stczyx_pixel_sizes(da: xr.DataArray) -> dict:
    """
    Pixel sizes ``{'Z', 'Y', 'X', 'unit'}`` of an xarray image (None where unknown).

    The coordinates are the source of truth: they follow slicing and
    downsampling (``da[..., ::2, ::2]``, ``coarsen``), while ``attrs`` are a
    snapshot taken at conversion and are dropped by many xarray operations.
    Per axis, the step of uniformly spaced float coordinates is used; axes
    without such coordinates fall back to ``da.attrs['pixel_sizes']``.
    Any uniform float coordinate counts as calibrated, including ones that are
    not physical (e.g. ``coarsen(...).mean()`` of an uncalibrated array turns
    the integer index into 0.5, 2.5, ... and reads as a pixel size of 2).
    The result can be passed directly as ``pixel_sizes=`` to the plotting and
    image-processing functions of this package.
    """
    stored = _normalize_pixel_sizes(da.attrs.get('pixel_sizes'))
    out = {}
    for ax in _SPATIAL:
        step = _coord_spacing(da, ax) if ax in da.dims else None
        out[ax] = step if step is not None else stored[ax]
    out['unit'] = _coord_unit(da) or stored['unit']
    return out


def set_channel_info(da: xr.DataArray, info: dict, strict: bool = True) -> xr.DataArray:
    """Attach per-channel metadata (colour, role, ...) as coordinates along ``C``.

    Channel names stay the ``C`` coordinate (e.g. read from the file by
    the source reader); everything else about a channel becomes an *auxiliary
    coordinate* on the ``C`` dimension, so it follows selection, reordering and
    slicing automatically (``da.sel(C=["DAPI", "SOX2"]).color``) instead of
    living in parallel lists or dicts that can fall out of sync.

    Parameters
    ----------
    da : xr.DataArray
        Image with a ``C`` dimension whose coordinate holds channel names.
    info : dict
        ``{channel_name: {field: value, ...}}``, e.g.
        ``{"DAPI": {"color": "gray", "role": "dna"}}``. Fields missing for a
        channel are ``""`` when the field's given values are all strings,
        else ``NaN``.
    strict : bool
        Raise if ``info`` names a channel that is not in ``da.C`` (catches typos
        and channel-order assumptions); channels without an entry only warn.

    Returns
    -------
    xr.DataArray
        A new DataArray with one coordinate per field along ``C``.

    Examples
    --------
    >>> da = set_channel_info(da, {"DAPI": {"color": "gray", "role": "dna"},
    ...                            "TBXT": {"color": "yellow", "role": "marker"}})
    >>> channels_where(da, role="dna")
    ['DAPI']
    """
    if "C" not in da.dims:
        raise ValueError("DataArray has no 'C' dimension")
    if "C" not in da.coords:
        da = da.assign_coords(C=np.arange(da.sizes["C"]))
    labels = list(da.coords["C"].values)
    names = [str(c) for c in labels]
    # match keys by value first (int channel labels), then by their string form
    key_of = {}
    for k in info:
        if k in labels:
            key_of[names[labels.index(k)]] = k
        elif str(k) in names:
            key_of[str(k)] = k
    unknown = [k for k in info if k not in key_of.values()]
    if unknown and strict:
        raise KeyError(f"channels {unknown} are not in the image; available: {names}")
    missing = [n for n in names if n not in key_of]
    if missing:
        warnings.warn(f"no channel info for {missing}", stacklevel=2)
    fields = []
    for v in info.values():
        for f in v:
            if f not in fields:
                fields.append(f)
    for f in fields:
        if f in da.dims or f in ("C", "S", "T", "Z", "Y", "X"):
            raise ValueError(f"field name {f!r} clashes with a dimension name")
    by_name = {n: info[k] for n, k in key_of.items()}
    coords = {}
    for f in fields:
        given = [v[f] for v in by_name.values() if f in v]
        # xarray turns None into NaN in object coordinates; use "" for string fields
        fill = "" if given and all(isinstance(g, str) for g in given) else np.nan
        arr = np.empty(len(names), dtype=object)  # 1-D even for tuple values (RGB colours)
        for i, n in enumerate(names):
            arr[i] = by_name.get(n, {}).get(f, fill)
        coords[f] = ("C", arr)
    return da.assign_coords(coords)


def channels_where(da: xr.DataArray, **criteria) -> list:
    """Channel labels (in ``C`` order, as stored in ``C``) whose auxiliary coordinates match ``criteria``.

    Each criterion is ``field=value`` (equality), ``field=[v1, v2]`` (any of)
    or ``field=callable`` (predicate on the value). Use the result with
    ``da.sel(C=...)``.

    >>> channels_where(da, role=["mito", "perox"])
    ['TOMM20', 'PMP70']
    """
    if "C" not in da.dims:
        raise ValueError("DataArray has no 'C' dimension")
    labels = list(da.coords["C"].values) if "C" in da.coords else list(range(da.sizes["C"]))
    keep = [True] * len(labels)
    for field, want in criteria.items():
        if field not in da.coords or da.coords[field].dims != ("C",):
            raise KeyError(f"no channel coordinate {field!r}; set it with set_channel_info")
        vals = list(da.coords[field].values)
        for i, v in enumerate(vals):
            if callable(want):
                ok = bool(want(v))
            elif not isinstance(want, (str, bytes)) and hasattr(want, "__iter__"):
                ok = v in list(want)  # any of (list, tuple, set, ndarray)
            else:
                ok = v == want
            keep[i] = keep[i] and ok
    return [lab for lab, k in zip(labels, keep) if k]  # native labels: usable in da.sel(C=...)


def channels_by(da: xr.DataArray, field: str, values, allow_missing: bool = False, unique: bool = True) -> list:
    """Channel names whose ``field`` coordinate equals each of ``values``, **in the order requested**.

    Use it when channel order matters (e.g. a model expecting
    ``[membrane, nucleus]``): ``da.sel(C=channels_by(da, "role", ["membrane", "dna"]))``.

    Parameters
    ----------
    field : str
        Coordinate along ``C`` (see :func:`set_channel_info`), e.g. ``"role"``.
    values : value or list of values
        Requested values, in output order.
    allow_missing : bool
        Skip values no channel has instead of raising ``KeyError``.
    unique : bool
        Raise ``ValueError`` if a value matches more than one channel; with
        ``unique=False`` all matches are returned (in ``C`` order per value).
    """
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        values = [values]
    out = []
    for v in values:
        names = channels_where(da, **{field: v})
        if len(names) > 1 and unique:
            raise ValueError(f"{field}={v!r} matches several channels {names}; expected one")
        if not names:
            if allow_missing:
                continue
            present = sorted({str(x) for x in da.coords[field].values if isinstance(x, str) and x})
            raise KeyError(f"no channel with {field}={v!r}; present: {present}")
        out.extend(names)
    return out


def _read_tiff(path, input_axes, series=None):
    with tifffile.TiffFile(path) as tif:
        all_series = list(tif.series)
        if series is not None:
            chosen = [all_series[series]]
        elif len(all_series) > 1 and len({(s.shape, s.axes, str(s.dtype)) for s in all_series}) == 1:
            chosen = all_series  # e.g. multi-position OME-TIFF: stack along S
        else:
            chosen = all_series[:1]
            if len(all_series) > 1:
                warnings.warn(
                    f"{Path(path).name}: {len(all_series)} series of different shapes; read series 0 "
                    "(pass series= to choose)",
                    stacklevel=3,
                )
        file_axes = chosen[0].axes.upper()
        data = chosen[0].asarray() if len(chosen) == 1 else np.stack([s.asarray() for s in chosen])
    if input_axes is not None:
        axes = input_axes.upper()
    else:
        # tifffile's 'S' is samples per pixel (RGB / planar components), not scenes
        if "S" in file_axes:
            if "C" in file_axes:
                raise ValueError(
                    f"{Path(path).name}: TIFF axes {file_axes!r} have both samples ('S') and channels ('C'); "
                    "pass input_axes="
                )
            file_axes = file_axes.replace("S", "C")
        axes = ("S" + file_axes) if len(chosen) > 1 else file_axes
    if set(axes) - set(_STCZYX):
        raise ValueError(
            f"{Path(path).name}: TIFF axes {axes!r} contain letters outside STCZYX (tifffile uses e.g. 'Q', 'I' "
            "for unknown axes); pass input_axes="
        )
    return data, get_tiff_voxel_size(path), axes


def load_image(
    source,
    *,
    input_axes: str = None,
    channel_names=None,
    channel_info: dict = None,
    pixel_sizes=None,
    strict: bool = True,
    series: int = None,
) -> xr.DataArray:
    """Convert an image (TIFF file, array, or DataArray from any reader) to a strict STCZYX ``xarray.DataArray``.

    Format-agnostic: files other than TIFF are read with their own reader
    (anything returning an array or an ``xarray.DataArray`` with single-letter
    dims, e.g. a vendor reader's ``to_xarray()``), and the result is passed
    here. Channel names stay as provided by the source (the ``C`` coordinate
    of a DataArray); ``channel_info`` attaches per-channel metadata as
    coordinates along ``C`` (:func:`set_channel_info`).

    Parameters
    ----------
    source : str, Path, array or xr.DataArray
        ``.tif`` / ``.tiff`` path (axes from tifffile unless ``input_axes``;
        voxel size via :func:`get_tiff_voxel_size`), or an in-memory array /
        DataArray (as :func:`array_to_stczyx_xarray`; calibrated coordinates
        give the pixel sizes). A reader's dim named ``P`` (position) maps to
        ``S``.
    input_axes : str, optional
        Axis letters for arrays, or TIFFs whose axes tifffile cannot name.
    channel_names : list of str, optional
        Define / override channel names (needed for plain arrays and most TIFFs).
    channel_info : dict, optional
        ``{channel_name: {field: value}}`` keyed by the **final** channel names
        (the file's names, or ``channel_names`` if given), e.g.
        ``{"DAPI": {"role": "dna", "color": "gray"}}``; unknown names raise when
        ``strict``.
    pixel_sizes : dict or (Z, Y, X) sequence, optional
        Overrides the source's values **per axis** (``{"Z": 0.5}`` keeps the
        file's X/Y).
    series : int, optional
        TIFF only: read this series. By default series of identical shape
        (multi-position OME-TIFF) are stacked along ``S``; otherwise series 0
        is read with a warning. tifffile's samples axis (RGB / planar 'S') is
        read as ``C``.

    Returns
    -------
    xr.DataArray
        Dims ``S, T, C, Z, Y, X``; raw file values (no dtype change); physical
        coordinates where pixel sizes are known; ``attrs["source"]`` for files.
    """
    attrs = {}
    file_ps = None
    data = source
    if isinstance(source, (str, Path)):
        path = Path(source)
        attrs["source"] = str(path)
        name = path.name.lower()
        if name.endswith((".tif", ".tiff")):
            data, file_ps, input_axes = _read_tiff(path, input_axes, series)
            if file_ps and file_ps.get("unit") is None and file_ps.get("X") == 1.0 and file_ps.get("Y") == 1.0:
                file_ps = {**file_ps, "X": None, "Y": None}  # resolution tags absent: 1.0 means "unknown"
        else:
            raise ValueError(
                f"unsupported file type: {path.name}; load_image reads .tif/.tiff. For other formats read "
                "the file with its own reader (to an array or xarray.DataArray with named dims) and pass that"
            )
    given = _normalize_pixel_sizes(pixel_sizes)
    base = _normalize_pixel_sizes(file_ps)
    ps = {a: given[a] if given[a] is not None else base[a] for a in ("Z", "Y", "X")}
    ps["unit"] = given["unit"] or base["unit"]
    if all(ps[a] is None for a in ("Z", "Y", "X")):
        ps = None
    original = None
    if channel_names is not None and hasattr(data, "coords") and "C" in getattr(data, "coords", {}):
        original = [str(c) for c in data.coords["C"].values]
    da = array_to_stczyx_xarray(data, input_axes=input_axes, channel_names=channel_names, attrs=attrs or None, pixel_sizes=ps)
    if channel_info:
        if original and strict and any(str(k) in original and str(k) not in map(str, da.coords["C"].values) for k in channel_info):
            raise KeyError(
                f"channel_info uses the original channel names {original}, but channel_names renamed them to "
                f"{[str(c) for c in da.coords['C'].values]}; key channel_info by the new names"
            )
        da = set_channel_info(da, channel_info, strict=strict)
    return da


def _normalize_stage_positions(stage_positions, n_series: int) -> dict:
    """{'stage_z', 'stage_y', 'stage_x'} -> float arrays of length n_series from a dict or an (S, 2|3) array."""
    if isinstance(stage_positions, dict):
        keyed = {str(k).upper().removeprefix('STAGE_'): v for k, v in stage_positions.items()}
        unknown = sorted(set(keyed) - set(_SPATIAL))
        if unknown:
            raise ValueError(f"stage_positions keys must be Z, Y, X, got {unknown}.")
        columns = {ax: keyed[ax] for ax in _SPATIAL if ax in keyed}
    else:
        table = np.asarray(stage_positions, dtype=float)
        if table.ndim == 1 and n_series == 1:
            table = table[None, :]
        if table.ndim != 2 or table.shape[1] not in (2, 3):
            raise ValueError(
                "stage_positions must be a dict {'Y': [...], 'X': [...]} (optionally 'Z') "
                f"or an (S, 2) (Y, X) / (S, 3) (Z, Y, X) array, got shape {table.shape}."
            )
        columns = dict(zip(_SPATIAL[-table.shape[1]:], table.T))

    out = {}
    for ax, values in columns.items():
        values = np.atleast_1d(np.asarray(values, dtype=float))
        if values.shape != (n_series,):
            raise ValueError(
                f"stage_positions['{ax}'] has {values.size} entries, but S has size {n_series}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"stage_positions['{ax}'] contains non-finite values.")
        out[f'stage_{ax.lower()}'] = values
    return out


def array_to_stczyx_xarray(
    image,
    input_axes: str = None,
    channel_names=None,
    attrs: dict = None,
    pixel_sizes=None,
    stage_positions=None,
) -> xr.DataArray:
    """
    Converts an array into an xarray.DataArray with strict 'STCZYX' dimension order.
    Missing dimensions are expanded with a size of 1.

    Accepts NumPy arrays, dask arrays and existing ``xr.DataArray`` objects
    (e.g. from ``nd2.ND2File.to_xarray()`` or ``aicsimageio``). Dask data
    stays lazy: nothing is loaded into memory. Other array-likes (e.g. zarr
    arrays) are loaded by xarray; wrap them with ``dask.array.from_zarr`` to
    keep them lazy.

    Dimension letters: S = scene / series / stage position, T = time,
    C = channel, Z, Y, X.

    Parameters:
    -----------
    image : array-like or xr.DataArray
        The input image (e.g. from ``tifffile.imread`` or ``nd2``).
    input_axes : str, optional
        The dimension order of the input (e.g. 'ZCYX', 'TCYX'). Required for
        plain arrays. For a DataArray it defaults to its dims, and if given it
        relabels them positionally (e.g. nd2's 'TPZCYX' -> 'TSZCYX'). A
        DataArray dim named 'P' (nd2's stage position) is read as 'S' when no
        'S' dim exists. Note that nd2 names the RGB-component axis 'S': pass
        ``input_axes`` explicitly for RGB nd2 files.
    channel_names : str or list of str, optional
        Names for the channel dimension. Must match the size of 'C'. A single
        string is accepted for a single channel. Overrides existing 'C' coords.
    attrs : dict, optional
        Metadata attributes to add. For a DataArray they are merged into (and
        take precedence over) its existing attrs.
    pixel_sizes : dict or (Z, Y, X) sequence, optional
        Physical pixel sizes, e.g. ``{'Z': 2.0, 'Y': 0.329, 'X': 0.329,
        'unit': 'µm'}`` as returned by ``get_tiff_voxel_size``. None entries
        mean unknown. Given sizes become physical coordinates
        (``origin + i * size``; a DataArray's existing origin is kept). If a
        DataArray already has calibrated coordinates with a different step, a
        warning is issued and the given size wins. Axes not given are read
        from calibrated (uniform float) coordinates, e.g. nd2's Y/X.
    stage_positions : dict or array, optional
        Stage position of each S entry (tile / position), as reported by the
        microscope, in the physical unit of ``pixel_sizes`` (usually µm):
        ``{'Y': [...], 'X': [...]}`` (optionally ``'Z'``), or an ``(S, 2)``
        (Y, X) / ``(S, 3)`` (Z, Y, X) array. Stored as non-dimension coordinates
        ``stage_y``, ``stage_x`` (and ``stage_z``) along S, so they follow
        selection along S (``da.isel(S=[0, 2])``). No orientation transform is
        applied: the stage axes may be flipped or rotated relative to the
        image axes, which a stitcher must calibrate. Existing ``stage_*``
        coordinates of a DataArray are kept unless overridden here.

    Returns:
    --------
    xr.DataArray
        6D DataArray strictly ordered as STCZYX. Existing coordinates (e.g.
        physical Y/X positions) and the name of a DataArray input are kept;
        dims without coordinates get integer coordinates (0, 1, 2, ...).
        When any pixel size is known, ``attrs['pixel_sizes']`` holds
        ``{'Z', 'Y', 'X', 'unit'}`` (None where unknown) and calibrated
        coordinates carry ``attrs['units']``. The coordinates stay the source
        of truth after slicing or downsampling: read sizes back with
        :func:`get_stczyx_pixel_sizes`, not from ``attrs``.
    """
    # 1. Wrap into an initial DataArray with dims from input_axes
    if isinstance(image, xr.DataArray):
        da = image
        if input_axes is None:
            dims = [str(d).upper() for d in da.dims]
            if 'P' in dims and 'S' not in dims:
                dims[dims.index('P')] = 'S'
            input_axes = ''.join(dims) if all(len(d) == 1 for d in dims) else None
            if input_axes is None:
                raise ValueError(
                    f"Cannot map DataArray dims {tuple(da.dims)} to STCZYX; "
                    "pass input_axes (e.g. 'TCYX') to relabel them."
                )
    else:
        if input_axes is None:
            raise ValueError("input_axes is required when the input is not an xarray.DataArray.")
        if not hasattr(image, 'shape') or not hasattr(image, 'ndim'):
            image = np.asarray(image)
        da = None

    input_axes = input_axes.upper()
    ndim = image.ndim

    # 2. Validate input dimensions
    if ndim != len(input_axes):
        raise ValueError(
            f"Shape mismatch: Array has {ndim} dims {tuple(image.shape)}, "
            f"but input_axes string '{input_axes}' defines {len(input_axes)} dims."
        )
    unknown = sorted(set(input_axes) - set(_STCZYX))
    if unknown:
        raise ValueError(
            f"input_axes '{input_axes}' contains {unknown}; allowed letters are {_STCZYX}."
        )
    if len(set(input_axes)) != len(input_axes):
        raise ValueError(f"input_axes '{input_axes}' repeats a dimension.")

    if da is None:
        da = xr.DataArray(image, dims=list(input_axes))
    else:
        da = _relabel_dims(da, input_axes)

    # 3. Expand missing dimensions (size 1) and enforce STCZYX order
    missing_dims = [dim for dim in _STCZYX if dim not in input_axes]
    if missing_dims:
        da = da.expand_dims({dim: 1 for dim in missing_dims})
    da = da.transpose(*_STCZYX)

    # 4. Populate coordinates (keep existing ones)
    coords = {}
    if channel_names is not None:
        if isinstance(channel_names, str):
            channel_names = [channel_names]
        channel_names = list(channel_names)
        if len(channel_names) != da.sizes['C']:
            raise ValueError(
                f"Provided {len(channel_names)} channel names, "
                f"but C dimension has size {da.sizes['C']}."
            )
        coords['C'] = channel_names

    # Physical pixel sizes: explicit argument > calibrated coords > stored attrs
    given = _normalize_pixel_sizes(
        pixel_sizes if pixel_sizes is not None else (attrs or {}).get('pixel_sizes')
    )
    stored = _normalize_pixel_sizes(da.attrs.get('pixel_sizes'))
    sizes = {}
    for dim in _SPATIAL:
        step = _coord_spacing(da, dim)
        if given[dim] is None and step is None and stored[dim] is not None:
            given[dim] = stored[dim]  # build physical coords from the stored sizes
        if given[dim] is not None:
            sizes[dim] = given[dim]
            if step is not None and not np.isclose(step, given[dim], rtol=1e-3):
                warnings.warn(
                    f"pixel_sizes['{dim}'] = {given[dim]} differs from the {dim} coordinate "
                    f"step {step}; rebuilding the {dim} coordinates from pixel_sizes.",
                    UserWarning, stacklevel=2,
                )
            if step is None or not np.isclose(step, given[dim], rtol=1e-3):
                # Keep the origin and direction of float (physical) coords;
                # integer coords are indices and start the physical axis at 0
                origin, sign = 0.0, 1.0
                if dim in da.coords and np.issubdtype(da.coords[dim].dtype, np.floating):
                    values = da.coords[dim].values
                    origin = float(values[0])
                    if values.size > 1 and values[-1] < values[0]:
                        sign = -1.0
                coords[dim] = origin + sign * np.arange(da.sizes[dim]) * given[dim]
        else:
            sizes[dim] = step if step is not None else stored[dim]
    unit = given['unit'] or _coord_unit(da) or stored['unit']

    for dim in _STCZYX:
        if dim not in coords and dim not in da.coords:
            # Default to standard integer coordinates (0, 1, 2...)
            coords[dim] = np.arange(da.sizes[dim])
    da = da.assign_coords(coords)

    # 5. Append metadata attributes (copy: never modify the caller's dicts)
    new_attrs = dict(da.attrs)
    new_attrs.update(attrs or {})
    new_attrs['axes'] = _STCZYX
    new_attrs.pop('pixel_sizes', None)
    if any(sizes[dim] is not None for dim in _SPATIAL):
        new_attrs['pixel_sizes'] = {**sizes, 'unit': unit}
        if unit is not None:
            # New coordinate variables: never touch the attrs of the input's coords
            da = da.assign_coords({
                dim: (dim, da.coords[dim].values, {**da.coords[dim].attrs, 'units': unit})
                for dim in _SPATIAL if _coord_spacing(da, dim) is not None
            })
    da.attrs = new_attrs

    # 6. Stage positions: non-dimension coordinates along S
    if stage_positions is not None:
        stage = _normalize_stage_positions(stage_positions, da.sizes['S'])
        coord_attrs = {'units': unit} if unit is not None else {}
        da = da.assign_coords({name: ('S', values, coord_attrs) for name, values in stage.items()})

    return da


def numpy_to_stczyx_xarray(
    image_array: np.ndarray,
    input_axes: str,
    channel_names: list = None,
    attrs: dict = None
) -> xr.DataArray:
    """Deprecated alias of :func:`array_to_stczyx_xarray`."""
    warnings.warn(
        "numpy_to_stczyx_xarray is deprecated. Use array_to_stczyx_xarray instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return array_to_stczyx_xarray(
        image_array, input_axes, channel_names=channel_names, attrs=attrs
    )


# Length units in metres, keyed by the spellings ImageJ, OME and TIFF use
_UNIT_IN_METRES = {
    'm': 1.0, 'meter': 1.0, 'metre': 1.0,
    'cm': 1e-2, 'centimeter': 1e-2,
    'mm': 1e-3, 'millimeter': 1e-3,
    'µm': 1e-6, 'μm': 1e-6, 'um': 1e-6, 'micron': 1e-6, 'microns': 1e-6, 'micrometer': 1e-6,
    'nm': 1e-9, 'nanometer': 1e-9,
    'å': 1e-10, 'angstrom': 1e-10,
    'inch': 0.0254, 'in': 0.0254,
}


def _unit_in_metres(unit):
    if unit is None:
        return None
    # ImageJ writes µ as the escape sequence 'µ'
    key = str(unit).strip().replace('\\u00B5', 'µ').replace('\\u00b5', 'µ').lower()
    return _UNIT_IN_METRES.get(key)


def get_tiff_voxel_size(filepath):
    """
    Attempts to read X, Y, and Z physical pixel sizes from a TIFF file.
    Returns a dictionary with the sizes and one common unit:
    ``{'X': ..., 'Y': ..., 'Z': ..., 'unit': ...}`` (None where unknown).

    - ImageJ stacks follow ImageJ's own decoding: Z is ``abs(spacing)``, and a
      multi-slice stack without a ``spacing`` entry has the default step 1.0
      (ImageJ omits the entry when the step is 1.0). A ``zunit`` that differs
      from ``unit`` is converted.
    - OME-TIFF sizes are read with their per-axis units (default µm).
    - X/Y fall back to the TIFF resolution tags (inch or cm).

    When the axes have different units (e.g. X/Y from resolution tags in cm and
    Z from OME in µm), all sizes are converted to µm.
    """
    voxel_data = {'X': None, 'Y': None, 'Z': None, 'unit': None}
    units = {'X': None, 'Y': None, 'Z': None}

    with tifffile.TiffFile(filepath) as tif:

        # ---------------------------------------------------------
        # 1. Try ImageJ Metadata (Very common for 3D TIFFs)
        # ---------------------------------------------------------
        if tif.is_imagej:
            metadata = tif.imagej_metadata
            if metadata:
                unit = metadata.get('unit')
                units['X'] = units['Y'] = unit
                spacing = metadata.get('spacing')
                if spacing is not None:
                    voxel_data['Z'] = abs(float(spacing))
                elif int(metadata.get('slices', 1) or 1) > 1:
                    voxel_data['Z'] = 1.0
                units['Z'] = metadata.get('zunit', unit)

        # ---------------------------------------------------------
        # 2. Try OME-TIFF Metadata (Common in microscopy)
        # ---------------------------------------------------------
        elif tif.is_ome:
            ome_xml = tif.ome_metadata
            if ome_xml:
                # Parse the XML to find PhysicalSize attributes
                root = ET.fromstring(ome_xml)
                # Namespace handling for OME-XML
                ns = {'ome': root.tag.split('}')[0].strip('{')}
                pixels = root.find('.//ome:Pixels', ns)

                if pixels is not None:
                    for ax in ('X', 'Y', 'Z'):
                        value = pixels.get(f'PhysicalSize{ax}')
                        if value:
                            voxel_data[ax] = float(value)
                            units[ax] = pixels.get(f'PhysicalSize{ax}Unit', 'µm')

        # ---------------------------------------------------------
        # 3. Read Standard TIFF Tags (Fallback for X and Y)
        # ---------------------------------------------------------
        # If X and Y weren't found in OME, check standard tags on the first page
        if voxel_data['X'] is None or voxel_data['Y'] is None:
            tags = tif.pages[0].tags

            # Resolution is typically stored as (numerator, denominator) representing pixels per unit
            x_res_tag = tags.get('XResolution')
            y_res_tag = tags.get('YResolution')
            res_unit_tag = tags.get('ResolutionUnit')

            if x_res_tag and y_res_tag:
                x_num, x_den = x_res_tag.value
                y_num, y_den = y_res_tag.value

                # Convert "pixels per unit" to "units per pixel"
                if x_num > 0 and y_num > 0:
                    voxel_data['X'] = x_den / x_num
                    voxel_data['Y'] = y_den / y_num

                # Standard TIFF ResolutionUnit: 1=None, 2=Inch, 3=Centimeter.
                # ImageJ's own 'unit' defines its resolution tags, so it takes precedence.
                if res_unit_tag and units['X'] is None:
                    unit_val = res_unit_tag.value
                    if unit_val == 2:
                        units['X'] = units['Y'] = 'inch'
                    elif unit_val == 3:
                        units['X'] = units['Y'] = 'cm'

    # ---------------------------------------------------------
    # 4. Report all sizes in one unit
    # ---------------------------------------------------------
    present = {ax: units[ax] for ax in ('X', 'Y', 'Z') if voxel_data[ax] is not None}
    first_unit = next((u for u in present.values() if u is not None), units['X'])
    # Spellings of the same unit ('um', 'µm', 'micron') count as one unit
    distinct = {_unit_in_metres(u) or u for u in present.values() if u is not None}
    if len(distinct) <= 1:
        voxel_data['unit'] = first_unit
    elif all(_unit_in_metres(u) is not None for u in present.values()):
        for ax, u in present.items():
            voxel_data[ax] = voxel_data[ax] * _unit_in_metres(u) / 1e-6
        voxel_data['unit'] = 'µm'
    else:
        warnings.warn(
            f"Pixel sizes of {filepath} use units {present} that could not be converted "
            "to a common unit; 'unit' refers to X.",
            UserWarning,
            stacklevel=2,
        )
        voxel_data['unit'] = units['X']

    return voxel_data
