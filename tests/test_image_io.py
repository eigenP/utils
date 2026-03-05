import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock
from eigenp_utils.image_io import numpy_to_stczyx_xarray, get_tiff_voxel_size

def test_numpy_to_stczyx_xarray_basic():
    """Test basic conversion to STCZYX."""
    arr = np.ones((10, 20))
    da = numpy_to_stczyx_xarray(arr, 'YX')

    assert da.shape == (1, 1, 1, 1, 10, 20)
    assert list(da.dims) == ['S', 'T', 'C', 'Z', 'Y', 'X']
    assert da.attrs['axes'] == 'STCZYX'

def test_numpy_to_stczyx_xarray_complex():
    """Test complex conversion with channel names and attrs."""
    arr = np.ones((5, 3, 10, 20))
    channels = ['DAPI', 'GFP', 'RFP']
    attrs = {'experiment': 'test1'}

    da = numpy_to_stczyx_xarray(arr, 'ZCYX', channel_names=channels, attrs=attrs)

    assert da.shape == (1, 1, 3, 5, 10, 20)
    assert list(da.dims) == ['S', 'T', 'C', 'Z', 'Y', 'X']
    assert da.coords['C'].values.tolist() == channels
    assert da.attrs['axes'] == 'STCZYX'
    assert da.attrs['experiment'] == 'test1'

def test_numpy_to_stczyx_xarray_errors():
    """Test error handling in numpy_to_stczyx_xarray."""
    arr = np.ones((10, 20))

    # Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        numpy_to_stczyx_xarray(arr, 'ZYX')

    # Channel names mismatch
    with pytest.raises(ValueError, match="Provided 2 channel names"):
        numpy_to_stczyx_xarray(np.ones((3, 10, 20)), 'CYX', channel_names=['C1', 'C2'])

@patch('eigenp_utils.image_io.tifffile.TiffFile')
def test_get_tiff_voxel_size_imagej(mock_tifffile):
    """Test reading ImageJ metadata."""
    mock_tif = MagicMock()
    mock_tif.is_imagej = True
    mock_tif.imagej_metadata = {'spacing': 0.5, 'unit': 'micron'}
    mock_x_res = MagicMock()
    mock_x_res.value = (10000, 10000) # num, den
    mock_y_res = MagicMock()
    mock_y_res.value = (10000, 10000)
    mock_unit = MagicMock()
    mock_unit.value = 3 # cm

    mock_tags = MagicMock()
    mock_tags.get.side_effect = lambda key: {
        'XResolution': mock_x_res,
        'YResolution': mock_y_res,
        'ResolutionUnit': mock_unit
    }.get(key)

    mock_page = MagicMock()
    mock_page.tags = mock_tags
    mock_tif.pages = [mock_page]

    # Mock context manager
    mock_tifffile.return_value.__enter__.return_value = mock_tif

    res = get_tiff_voxel_size("dummy.tif")

    assert res['Z'] == 0.5
    # Since ImageJ usually sets Z from its own metadata but standard tags sets unit:
    # Actually get_tiff_voxel_size logic currently overrides unit with cm if standard tags fall through.
    # In the code, unit is 'micron' from imagej, but standard tags rewrite 'unit' to 'cm'.
    # Let's verify X/Y calculation (10000/10000 = 1.0)
    assert res['X'] == 1.0
    assert res['Y'] == 1.0

@patch('eigenp_utils.image_io.tifffile.TiffFile')
def test_get_tiff_voxel_size_ome(mock_tifffile):
    """Test reading OME-TIFF metadata."""
    mock_tif = MagicMock()
    mock_tif.is_imagej = False
    mock_tif.is_ome = True

    ome_xml = '''
    <OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
        <Image>
            <Pixels PhysicalSizeX="0.2" PhysicalSizeY="0.2" PhysicalSizeZ="0.5" PhysicalSizeXUnit="um" />
        </Image>
    </OME>
    '''
    mock_tif.ome_metadata = ome_xml
    mock_tifffile.return_value.__enter__.return_value = mock_tif

    res = get_tiff_voxel_size("dummy.tif")

    assert res['X'] == 0.2
    assert res['Y'] == 0.2
    assert res['Z'] == 0.5
    assert res['unit'] == 'um'
