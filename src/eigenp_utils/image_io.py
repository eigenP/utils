import warnings

import numpy as np
import xarray as xr
import tifffile
import xml.etree.ElementTree as ET


def numpy_to_stczyx_xarray(
    image_array: np.ndarray,
    input_axes: str,
    channel_names: list = None,
    attrs: dict = None
) -> xr.DataArray:
    """
    Converts a NumPy array into an xarray.DataArray with strict 'STCZYX' dimension order.
    Missing dimensions in `input_axes` are expanded with a size of 1.

    Parameters:
    -----------
    image_array : np.ndarray
        The input image array (e.g., from tifffile.imread).
    input_axes : str
        The dimension order of the input array (e.g., 'ZCYX', 'TCYX').
    channel_names : list of str, optional
        Names for the channel dimension. Must match the size of 'C'.
    attrs : dict, optional
        Dictionary of metadata attributes to add to the xarray.

    Returns:
    --------
    xr.DataArray
        6D DataArray strictly ordered as STCZYX.
    """
    input_axes = input_axes.upper()
    target_axes = list('STCZYX')

    # 1. Validate input dimensions
    if len(image_array.shape) != len(input_axes):
        raise ValueError(
            f"Shape mismatch: Array has {len(image_array.shape)} dims {image_array.shape}, "
            f"but input_axes string '{input_axes}' defines {len(input_axes)} dims."
        )

    # 2. Wrap into an initial DataArray
    da = xr.DataArray(image_array, dims=list(input_axes))

    # 3. Expand missing dimensions (set size to 1)
    missing_dims = [dim for dim in target_axes if dim not in input_axes]
    for dim in missing_dims:
        da = da.expand_dims({dim: 1})

    # 4. Transpose to strictly enforce STCZYX order
    da = da.transpose(*target_axes)

    # 5. Populate coordinates
    coords = {}
    for dim in target_axes:
        if dim == 'C' and channel_names is not None:
            if len(channel_names) != da.sizes['C']:
                raise ValueError(
                    f"Provided {len(channel_names)} channel names, "
                    f"but C dimension has size {da.sizes['C']}."
                )
            coords[dim] = channel_names
        else:
            # Default to standard integer coordinates (0, 1, 2...)
            coords[dim] = np.arange(da.sizes[dim])

    da = da.assign_coords(coords)

    # 6. Append metadata attributes (copy: never modify the caller's dict)
    attrs = dict(attrs or {})
    attrs['axes'] = 'STCZYX'
    da.attrs.update(attrs)

    return da


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
