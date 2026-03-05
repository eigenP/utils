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

    # 6. Append metadata attributes
    if attrs is None:
        attrs = {}
    attrs['axes'] = 'STCZYX'
    da.attrs.update(attrs)

    return da


def get_tiff_voxel_size(filepath):
    """
    Attempts to read X, Y, and Z physical pixel sizes from a TIFF file.
    Returns a dictionary with the sizes and units.
    """
    voxel_data = {'X': None, 'Y': None, 'Z': None, 'unit': None}

    with tifffile.TiffFile(filepath) as tif:

        # ---------------------------------------------------------
        # 1. Try ImageJ Metadata (Very common for 3D TIFFs)
        # ---------------------------------------------------------
        if tif.is_imagej:
            metadata = tif.imagej_metadata
            if metadata:
                # ImageJ stores Z spacing explicitly
                voxel_data['Z'] = metadata.get('spacing')
                voxel_data['unit'] = metadata.get('unit')

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
                    voxel_data['X'] = float(pixels.get('PhysicalSizeX')) if pixels.get('PhysicalSizeX') else None
                    voxel_data['Y'] = float(pixels.get('PhysicalSizeY')) if pixels.get('PhysicalSizeY') else None
                    voxel_data['Z'] = float(pixels.get('PhysicalSizeZ')) if pixels.get('PhysicalSizeZ') else None
                    voxel_data['unit'] = pixels.get('PhysicalSizeXUnit', 'µm') # Usually microns

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

                # Standard TIFF ResolutionUnit: 1=None, 2=Inch, 3=Centimeter
                if res_unit_tag:
                    unit_val = res_unit_tag.value
                    if unit_val == 2:
                        voxel_data['unit'] = 'inch'
                    elif unit_val == 3:
                        voxel_data['unit'] = 'cm'

    return voxel_data
