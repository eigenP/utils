import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from eigenp_utils.image_io import array_to_stczyx_xarray
from eigenp_utils.stitching import stitch_tiles

pytestmark = pytest.mark.filterwarnings("ignore:stitch_tiles is a minimal:UserWarning")

TILE = 120
NOMINAL = np.array([0, 90, 180])  # 30 px nominal overlap


def _scene(shape=(320, 320), seed=0):
    """Textured ground-truth scene: smoothed noise plus blobs, in uint16."""
    rng = np.random.default_rng(seed)
    img = ndimage.gaussian_filter(rng.random(shape), 3)
    img = (img - img.min()) / np.ptp(img)
    return np.round(img * 60000).astype(np.uint16)


def _tiles(scene, jitter_seed=1, jitter=5, extra_leading=()):
    """Cut a 3x3 grid of tiles at nominal + random jitter positions (the true positions).

    Returns (tiles array (S, *extra_leading, Y, X), true top-left positions, nominal positions).
    """
    rng = np.random.default_rng(jitter_seed)
    nominal = np.array([(y, x) for y in NOMINAL for x in NOMINAL]) + jitter
    true = nominal + rng.integers(-jitter, jitter + 1, size=nominal.shape)
    cuts = np.stack([scene[..., y:y + TILE, x:x + TILE] for y, x in true])
    return cuts, true, nominal


def _expected(scene, true):
    """Ground truth over the mosaic's bounding box and the mask of pixels some tile covers."""
    lo = true.min(axis=0)
    hi = true.max(axis=0) + TILE
    mask = np.zeros(hi - lo, bool)
    for y, x in true - lo:
        mask[y:y + TILE, x:x + TILE] = True
    return scene[..., lo[0]:hi[0], lo[1]:hi[1]], mask


def test_stitch_tiles_recovers_true_positions_and_scene():
    """Positive control: stage reports nominal grid positions, tiles were taken at jittered ones.

    The stitcher must recover the true relative positions exactly (integer shifts,
    noise-free data) and reproduce the ground-truth scene over the covered area,
    for every channel, although only channel 0 is registered.
    """
    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    second = scene.max() - cuts  # a different channel with the same geometry
    data = np.stack([cuts, second], axis=1)  # S, C, Y, X
    px = 0.5
    da = array_to_stczyx_xarray(
        data, 'SCYX', channel_names=['A', 'B'],
        pixel_sizes={'Y': px, 'X': px, 'unit': 'µm'},
        stage_positions={'Y': nominal[:, 0] * px, 'X': nominal[:, 1] * px},
    )
    mosaic = stitch_tiles(da)

    positions = np.array(mosaic.attrs['stitching']['positions_px'])
    np.testing.assert_array_equal(positions, true - true.min(axis=0))
    assert mosaic.attrs['stitching']['n_pairs'] >= 12  # the 12 edge-sharing neighbours

    expected, mask = _expected(scene, true)
    assert mosaic.dims == tuple('STCZYX') and mosaic.sizes['S'] == 1
    assert mosaic.dtype == np.uint16
    a = mosaic.sel(C='A').values[0, 0, 0]
    b = mosaic.sel(C='B').values[0, 0, 0]
    np.testing.assert_array_equal(a[mask], expected[mask])
    np.testing.assert_array_equal(b[mask], (scene.max() - expected)[mask])
    assert np.all(a[~mask] == 0)  # gaps between jittered tiles stay empty
    assert mosaic.attrs['pixel_sizes'] == {'Z': None, 'Y': px, 'X': px, 'unit': 'µm'}
    np.testing.assert_allclose(np.diff(mosaic.coords['X'].values), px)


def test_stitch_tiles_stage_only_placement_is_off_without_registration():
    """With registration disabled (min_ncc > 1), the mosaic is the stage-only placement,
    which differs from the truth by the jitter; this shows the registration does the work."""
    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    da = array_to_stczyx_xarray(cuts, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                stage_positions={'Y': nominal[:, 0], 'X': nominal[:, 1]})
    mosaic = stitch_tiles(da, min_ncc=1.1)
    assert mosaic.attrs['stitching']['n_pairs'] == 0
    np.testing.assert_array_equal(mosaic.attrs['stitching']['positions_px'],
                                  nominal - nominal.min(axis=0))


def test_stitch_tiles_featureless_tiles_fall_back_to_stage():
    """Negative control: constant tiles have no features; every pair is rejected and the
    positions equal the stage positions (no drift driven by noise or edge effects)."""
    cuts = np.full((9, TILE, TILE), 100, np.uint16)
    nominal = np.array([(y, x) for y in NOMINAL for x in NOMINAL])
    da = array_to_stczyx_xarray(cuts, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                stage_positions={'Y': nominal[:, 0], 'X': nominal[:, 1]})
    mosaic = stitch_tiles(da)
    assert mosaic.attrs['stitching']['n_pairs'] == 0
    np.testing.assert_array_equal(mosaic.attrs['stitching']['positions_px'], nominal)
    assert np.all(mosaic.values == 100)


def test_stitch_tiles_flipped_stage_axis_and_z_t_projection():
    """A stage whose Y axis points up in the image is fixed with stage_to_image.

    Tiles have T and Z planes; registration uses the max over T and Z, and every
    plane is placed with the same positions.
    """
    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    planes = np.stack([cuts, cuts // 2], axis=1)            # S, Z, Y, X
    data = np.stack([planes, planes // 3], axis=1)         # S, T, Z, Y, X
    da = array_to_stczyx_xarray(data, 'STZYX', pixel_sizes=(2.0, 1.0, 1.0),
                                stage_positions={'Y': -nominal[:, 0], 'X': nominal[:, 1]})
    mosaic = stitch_tiles(da, stage_to_image=[[-1, 0], [0, 1]], blending='overwrite')

    np.testing.assert_array_equal(mosaic.attrs['stitching']['positions_px'], true - true.min(axis=0))
    expected, mask = _expected(scene, true)
    np.testing.assert_array_equal(mosaic.values[0, 0, 0, 0][mask], expected[mask])
    np.testing.assert_array_equal(mosaic.values[0, 1, 0, 1][mask], (expected // 2 // 3)[mask])
    np.testing.assert_array_equal(mosaic.coords['Z'].values, [0.0, 2.0])


def test_stitch_tiles_rejects_bad_input_and_warns():
    """Missing stage coords, missing pixel sizes, wrong dims and bad options raise;
    every call points users to multiview-stitcher."""
    tiles = np.zeros((2, 8, 8))
    no_stage = array_to_stczyx_xarray(tiles, 'SYX', pixel_sizes=(1, 1, 1))
    with pytest.raises(ValueError, match="stage_y"):
        stitch_tiles(no_stage)
    no_px = array_to_stczyx_xarray(tiles, 'SYX', stage_positions={'Y': [0, 0], 'X': [0, 4]})
    with pytest.raises(ValueError, match="pixel sizes"):
        stitch_tiles(no_px)
    with pytest.raises(ValueError, match="dims"):
        stitch_tiles(xr.DataArray(tiles, dims=('S', 'Y', 'X')))
    ok = array_to_stczyx_xarray(tiles, 'SYX', pixel_sizes=(1, 1, 1),
                                stage_positions={'Y': [0, 0], 'X': [0, 4]})
    with pytest.raises(ValueError, match="blending"):
        stitch_tiles(ok, blending='max')
    with pytest.warns(UserWarning, match="multiview-stitcher"):
        stitch_tiles(ok, min_overlap=4)
