import warnings

import numpy as np
import xarray as xr
import tifffile
import xml.etree.ElementTree as ET


_STCZYX = 'STCZYX'


def _relabel_dims(da: xr.DataArray, axes: str) -> xr.DataArray:
    """Rename the dims of ``da`` positionally to the letters of ``axes``.

    Goes through temporary names so that permutations of existing names
    (e.g. nd2's ``('P', 'S')`` -> ``('S', 'C')``) do not collide.
    """
    if tuple(da.dims) == tuple(axes):
        return da
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


def array_to_stczyx_xarray(
    image,
    input_axes: str = None,
    channel_names=None,
    attrs: dict = None,
    pixel_sizes=None,
) -> xr.DataArray:
    """
    Converts an array into an xarray.DataArray with strict 'STCZYX' dimension order.
    Missing dimensions are expanded with a size of 1.

    Accepts NumPy arrays, lazy arrays (dask, zarr, ...) and existing
    ``xr.DataArray`` objects (e.g. from ``nd2.ND2File.to_xarray()`` or
    ``aicsimageio``). Lazy data stays lazy: nothing is loaded into memory.

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
        if given[dim] is not None:
            sizes[dim] = given[dim]
            if step is not None and not np.isclose(step, given[dim], rtol=1e-3):
                warnings.warn(
                    f"pixel_sizes['{dim}'] = {given[dim]} differs from the {dim} coordinate "
                    f"step {step}; rebuilding the {dim} coordinates from pixel_sizes.",
                    UserWarning, stacklevel=2,
                )
            if step is None or not np.isclose(step, given[dim], rtol=1e-3):
                origin = float(da.coords[dim].values[0]) if step is not None else 0.0
                coords[dim] = origin + np.arange(da.sizes[dim]) * given[dim]
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
