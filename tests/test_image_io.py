import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock
from eigenp_utils.image_io import (
    array_to_stczyx_xarray,
    numpy_to_stczyx_xarray,
    get_tiff_voxel_size,
)

def test_array_to_stczyx_xarray_basic():
    """A plain YX array becomes 6D STCZYX with singleton S, T, C, Z and integer coords."""
    arr = np.ones((10, 20))
    da = array_to_stczyx_xarray(arr, 'YX')

    assert da.shape == (1, 1, 1, 1, 10, 20)
    assert list(da.dims) == ['S', 'T', 'C', 'Z', 'Y', 'X']
    assert da.attrs['axes'] == 'STCZYX'

def test_array_to_stczyx_xarray_complex():
    """ZCYX input is reordered to STCZYX; channel names and attrs are attached."""
    arr = np.ones((5, 3, 10, 20))
    channels = ['DAPI', 'GFP', 'RFP']
    attrs = {'experiment': 'test1'}

    da = array_to_stczyx_xarray(arr, 'ZCYX', channel_names=channels, attrs=attrs)

    assert da.shape == (1, 1, 3, 5, 10, 20)
    assert list(da.dims) == ['S', 'T', 'C', 'Z', 'Y', 'X']
    assert da.coords['C'].values.tolist() == channels
    assert da.attrs['axes'] == 'STCZYX'
    assert da.attrs['experiment'] == 'test1'

def test_array_to_stczyx_xarray_errors():
    """Mismatched ndim and wrong-length channel_names raise ValueError."""
    arr = np.ones((10, 20))

    # Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch"):
        array_to_stczyx_xarray(arr, 'ZYX')

    # Channel names mismatch
    with pytest.raises(ValueError, match="Provided 2 channel names"):
        array_to_stczyx_xarray(np.ones((3, 10, 20)), 'CYX', channel_names=['C1', 'C2'])

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


# =========================================
# Source: bug-scan regressions (low-severity sweep)
# =========================================


def test_array_to_stczyx_xarray_does_not_modify_attrs():
    """The caller's attrs dict is copied, not mutated, and 'axes' is overwritten to STCZYX."""
    meta = {'source': 'x.tif', 'axes': 'ZCYX'}
    da = array_to_stczyx_xarray(np.zeros((2, 1, 3, 3)), 'ZCYX', attrs=meta)
    assert meta == {'source': 'x.tif', 'axes': 'ZCYX'}
    assert da.attrs['axes'] == 'STCZYX' and da.attrs['source'] == 'x.tif'


def _imagej_stack(tmp_path, metadata):
    import tifffile

    path = tmp_path / "ij.tif"
    tifffile.imwrite(
        path, np.zeros((4, 8, 8), np.uint8), imagej=True, resolution=(2.0, 2.0),
        metadata={'axes': 'ZYX', **metadata},
    )
    return get_tiff_voxel_size(path)


def test_get_tiff_voxel_size_imagej_default_and_negative_spacing(tmp_path):
    # ImageJ omits 'spacing' when the z-step is 1.0
    res = _imagej_stack(tmp_path, {'unit': 'micron'})
    assert res == {'X': 0.5, 'Y': 0.5, 'Z': 1.0, 'unit': 'micron'}
    # ImageJ reads a negative spacing as its absolute value
    res = _imagej_stack(tmp_path, {'unit': 'micron', 'spacing': -2.0})
    assert res['Z'] == 2.0


def test_get_tiff_voxel_size_imagej_zunit_is_converted(tmp_path):
    res = _imagej_stack(tmp_path, {'unit': 'micron', 'spacing': 500.0, 'zunit': 'nm'})
    assert res['X'] == pytest.approx(0.5)
    assert res['Z'] == pytest.approx(0.5)
    assert res['unit'] == 'µm'


@patch('eigenp_utils.image_io.tifffile.TiffFile')
def test_get_tiff_voxel_size_ome_z_with_xy_from_resolution_tags(mock_tifffile):
    """Z from OME in µm, X/Y from resolution tags in cm: one common unit."""
    mock_tif = MagicMock()
    mock_tif.is_imagej = False
    mock_tif.is_ome = True
    mock_tif.ome_metadata = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image>'
        '<Pixels PhysicalSizeZ="2.0" PhysicalSizeZUnit="µm" /></Image></OME>'
    )
    res_tag = MagicMock(value=(10000, 1))  # 10000 px per cm -> 1e-4 cm = 1 µm
    unit_tag = MagicMock(value=3)
    mock_tif.pages = [MagicMock()]
    mock_tif.pages[0].tags.get.side_effect = lambda k: {
        'XResolution': res_tag, 'YResolution': res_tag, 'ResolutionUnit': unit_tag
    }.get(k)
    mock_tifffile.return_value.__enter__.return_value = mock_tif

    res = get_tiff_voxel_size("dummy.tif")
    assert res['unit'] == 'µm'
    assert res['X'] == pytest.approx(1.0) and res['Y'] == pytest.approx(1.0)
    assert res['Z'] == pytest.approx(2.0)


def test_array_to_stczyx_xarray_from_dataarray_keeps_coords_and_attrs():
    """A YX DataArray with physical coords (as from nd2.to_xarray) is expanded to STCZYX.

    Y/X float coords, the name and existing attrs must survive; the new
    singleton dims get integer coords; channel_names may be a single string.
    """
    y = np.arange(4) * 0.329
    x = np.arange(5) * 0.329
    src = xr.DataArray(
        np.arange(20, dtype=np.uint16).reshape(4, 5),
        dims=('Y', 'X'), coords={'Y': y, 'X': x},
        name='img', attrs={'metadata': 'nd2-meta'},
    )
    da = array_to_stczyx_xarray(src, channel_names='BF', attrs={'experiment': 'e1'})

    assert da.dims == tuple('STCZYX')
    assert da.shape == (1, 1, 1, 1, 4, 5)
    np.testing.assert_array_equal(da.coords['Y'].values, y)
    np.testing.assert_array_equal(da.coords['X'].values, x)
    assert da.coords['C'].values.tolist() == ['BF']
    assert da.coords['S'].values.tolist() == [0]
    assert da.name == 'img'
    assert da.attrs == {'metadata': 'nd2-meta', 'experiment': 'e1', 'axes': 'STCZYX'}
    np.testing.assert_array_equal(da.values[0, 0, 0, 0], src.values)
    # The input DataArray is untouched
    assert src.dims == ('Y', 'X') and src.attrs == {'metadata': 'nd2-meta'}


def test_array_to_stczyx_xarray_dataarray_relabel_and_position_alias():
    """nd2's 'P' (stage position) maps to 'S'; input_axes relabels DataArray dims positionally.

    The data must follow its axis: element [p, c, y, x] of the input ends up at
    [p, 0, c, 0, y, x]. Relabelling with a permutation of existing names
    ('S' -> 'C', 'C' -> 'S') must not collide.
    """
    arr = np.random.default_rng(0).random((3, 2, 4, 5))
    src = xr.DataArray(arr, dims=('P', 'C', 'Y', 'X'),
                       coords={'C': ['DAPI', 'GFP']})
    da = array_to_stczyx_xarray(src)
    assert da.sizes['S'] == 3 and da.sizes['C'] == 2
    assert da.coords['C'].values.tolist() == ['DAPI', 'GFP']
    np.testing.assert_array_equal(da.values[:, 0, :, 0], arr)

    swapped = xr.DataArray(arr, dims=('S', 'C', 'Y', 'X'))
    da2 = array_to_stczyx_xarray(swapped, input_axes='CSYX')
    assert da2.sizes['C'] == 3 and da2.sizes['S'] == 2
    np.testing.assert_array_equal(da2.values[:, 0, :, 0], arr.transpose(1, 0, 2, 3))


def test_array_to_stczyx_xarray_stays_lazy_with_dask():
    """Dask input (nd2 / zarr readers) stays a dask array; values match the eager result."""
    dask_array = pytest.importorskip('dask.array')
    arr = np.arange(2 * 6 * 7).reshape(2, 6, 7)
    lazy = dask_array.from_array(arr, chunks=(1, 6, 7))

    da = array_to_stczyx_xarray(lazy, 'ZYX')
    assert isinstance(da.data, dask_array.Array)
    np.testing.assert_array_equal(da.values, array_to_stczyx_xarray(arr, 'ZYX').values)

    da_x = array_to_stczyx_xarray(xr.DataArray(lazy, dims=('Z', 'Y', 'X')))
    assert isinstance(da_x.data, dask_array.Array)


def test_array_to_stczyx_xarray_rejects_bad_axes():
    """Unknown letters, repeated letters, unmappable DataArray dims and missing axes raise."""
    arr = np.ones((3, 4, 5))
    with pytest.raises(ValueError, match="allowed letters"):
        array_to_stczyx_xarray(arr, 'QYX')
    with pytest.raises(ValueError, match="repeats"):
        array_to_stczyx_xarray(arr, 'YYX')
    with pytest.raises(ValueError, match="input_axes is required"):
        array_to_stczyx_xarray(arr)
    with pytest.raises(ValueError, match="Cannot map"):
        array_to_stczyx_xarray(xr.DataArray(arr, dims=('time', 'Y', 'X')))
    # ... but relabelling such a DataArray works
    da = array_to_stczyx_xarray(xr.DataArray(arr, dims=('time', 'Y', 'X')), input_axes='TYX')
    assert da.sizes['T'] == 3


def test_numpy_to_stczyx_xarray_is_deprecated_alias():
    """The old name warns with DeprecationWarning and returns the same result as the new one."""
    arr = np.ones((3, 10, 20))
    with pytest.warns(DeprecationWarning, match="array_to_stczyx_xarray"):
        old = numpy_to_stczyx_xarray(arr, 'CYX', channel_names=['a', 'b', 'c'])
    new = array_to_stczyx_xarray(arr, 'CYX', channel_names=['a', 'b', 'c'])
    xr.testing.assert_identical(old, new)
