import numpy as np
import pytest
import xarray as xr
from scipy import ndimage

from eigenp_utils.image_io import array_to_stczyx_xarray, get_stczyx_pixel_sizes
from eigenp_utils.stitching import _solve_positions, stitch_tiles

pytestmark = pytest.mark.filterwarnings("ignore:stitch_tiles is a minimal:UserWarning")

TILE = 120
NOMINAL = np.array([0, 90, 180])  # 30 px nominal overlap


def _scene(shape=(320, 320), seed=0):
    """Textured ground-truth scene: smoothed noise plus blobs, in uint16."""
    rng = np.random.default_rng(seed)
    img = ndimage.gaussian_filter(rng.random(shape), 3)
    img = (img - img.min()) / np.ptp(img)
    return np.round(img * 60000).astype(np.uint16)


def _tiles(scene, jitter_seed=1, jitter=5):
    """Cut a 3x3 grid of tiles at nominal + random jitter positions (the true positions).

    Returns (tiles array (S, Y, X), true top-left positions, nominal positions).
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

    Strategy: build a 2-channel 3x3 grid with random stage jitter up to +/-5 px
    and stitch using channel 0 only.
    Expected outcome: recovers exact true integer positions using strictly the
    12 side-neighbour pairs (excluding the 4 diagonal corner pairs), reproduces
    both channels over the covered area, leaves gaps at 0, and preserves dtype,
    pixel sizes, DataArray name, and custom attrs.
    """
    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    second = scene.max() - cuts  # a different channel with the same geometry
    data = np.stack([cuts, second], axis=1)  # S, C, Y, X
    px = 0.5
    da = array_to_stczyx_xarray(
        data, 'SCYX', channel_names=['A', 'B'],
        attrs={'experiment': 'mosaic_1'},
        pixel_sizes={'Y': px, 'X': px, 'unit': 'µm'},
        stage_positions={'Y': nominal[:, 0] * px, 'X': nominal[:, 1] * px},
    )
    da.name = 'tile_grid'
    mosaic = stitch_tiles(da)

    positions = np.array(mosaic.attrs['stitching']['positions_px'])
    np.testing.assert_array_equal(positions, true - true.min(axis=0))
    assert mosaic.attrs['stitching']['n_pairs'] == 12  # strictly the 12 side neighbours, no diagonals

    expected, mask = _expected(scene, true)
    assert mosaic.dims == tuple('STCZYX') and mosaic.sizes['S'] == 1
    assert mosaic.dtype == np.uint16
    assert mosaic.name == 'tile_grid'
    assert mosaic.attrs['experiment'] == 'mosaic_1'
    a = mosaic.sel(C='A').values[0, 0, 0]
    b = mosaic.sel(C='B').values[0, 0, 0]
    np.testing.assert_array_equal(a[mask], expected[mask])
    np.testing.assert_array_equal(b[mask], (scene.max() - expected)[mask])
    assert np.all(a[~mask] == 0)  # gaps between jittered tiles stay empty
    assert mosaic.attrs['pixel_sizes'] == {'Z': None, 'Y': px, 'X': px, 'unit': 'µm'}
    np.testing.assert_allclose(np.diff(mosaic.coords['X'].values), px)


def test_stitch_tiles_stage_only_placement_is_off_without_registration():
    """With registration disabled (min_ncc > 1), the mosaic falls back to the jittered stage grid.

    Strategy: verify nominal stage positions genuinely differ from true tile
    positions, then call ``stitch_tiles(..., min_ncc=1.1)``.
    Expected outcome: zero pairs are registered and output positions equal the
    normalized stage positions rather than ground truth.
    """
    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    assert not np.array_equal(nominal - nominal.min(axis=0), true - true.min(axis=0))
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


def test_stitch_tiles_anisotropic_pixels_and_stage_origin_invariance():
    """Verify anisotropic Y/X pixel scaling and translation invariance of the stage origin.

    Strategy: construct tiles with distinct ``Y=0.5`` and ``X=1.25`` pixel sizes
    and stitch once at the original stage coordinates and once after adding an
    arbitrary constant offset ``(432.5, -187.25)`` to every tile's stage position.
    Expected outcome: anisotropic pixel sizes recover exact true pixel positions
    (no Y/X axis swap) and shifting the stage origin yields an identical mosaic.
    """
    scene = _scene(seed=7)
    cuts, true, nominal = _tiles(scene, jitter_seed=7)
    py, px = 0.5, 1.25
    da = array_to_stczyx_xarray(
        cuts, 'SYX',
        pixel_sizes={'Y': py, 'X': px, 'unit': 'µm'},
        stage_positions={'Y': nominal[:, 0] * py, 'X': nominal[:, 1] * px},
    )
    mosaic = stitch_tiles(da)
    np.testing.assert_array_equal(mosaic.attrs['stitching']['positions_px'], true - true.min(axis=0))
    assert mosaic.attrs['pixel_sizes'] == {'Z': None, 'Y': py, 'X': px, 'unit': 'µm'}

    da_shifted = array_to_stczyx_xarray(
        cuts, 'SYX',
        pixel_sizes={'Y': py, 'X': px, 'unit': 'µm'},
        stage_positions={'Y': nominal[:, 0] * py + 432.5, 'X': nominal[:, 1] * px - 187.25},
    )
    mosaic_shifted = stitch_tiles(da_shifted)
    xr.testing.assert_identical(mosaic, mosaic_shifted)


def test_stitch_tiles_max_shift_disjoint_and_noise_robustness():
    """Verify ``max_shift`` bounds, disjoint-tile stage fallback, and modest-noise robustness.

    Strategy: (1) construct two overlapping tiles with a 6 px relative stage
    error and test ``max_shift=4`` (must reject border peak) vs ``max_shift=10``
    (must recover exact shift); (2) place two tiles with no overlap (disjoint);
    (3) add Gaussian noise (sigma=1500 on 0..60000 uint16) to a 3x3 jittered grid.
    Expected outcome: (1) too-small ``max_shift`` registers 0 pairs while
    adequate ``max_shift`` recovers exact positions; (2) disjoint tiles stay at
    stage positions with zero-filled gap; (3) noisy 3x3 grid still recovers
    exact integer positions.
    """
    scene = _scene(seed=9)
    # 1. Two tiles with 30 px nominal overlap and 6 px true relative shift
    t0 = scene[20:20 + TILE, 20:20 + TILE]
    t1 = scene[20:20 + TILE, 20 + 96:20 + 96 + TILE]  # true dx = 96, nominal dx = 90 (error = +6)
    pair = array_to_stczyx_xarray(
        np.stack([t0, t1]), 'SYX',
        pixel_sizes=(1.0, 1.0, 1.0),
        stage_positions={'Y': [0.0, 0.0], 'X': [0.0, 90.0]},
    )
    too_tight = stitch_tiles(pair, max_shift=4)
    assert too_tight.attrs['stitching']['n_pairs'] == 0
    enough = stitch_tiles(pair, max_shift=10)
    assert enough.attrs['stitching']['n_pairs'] == 1
    np.testing.assert_array_equal(enough.attrs['stitching']['positions_px'], [[0, 0], [0, 96]])

    # 2. Disjoint tiles: gap of 10 px between tiles stays zero-filled
    disjoint = array_to_stczyx_xarray(
        np.stack([np.full((40, 40), 50, np.uint16), np.full((40, 40), 80, np.uint16)]),
        'SYX',
        pixel_sizes=(1.0, 1.0, 1.0),
        stage_positions={'Y': [0.0, 0.0], 'X': [0.0, 50.0]},
    )
    m_dis = stitch_tiles(disjoint)
    assert m_dis.attrs['stitching']['n_pairs'] == 0
    assert m_dis.sizes['X'] == 90
    assert np.all(m_dis.values[0, 0, 0, 0, :, 40:50] == 0)

    # 3. Modest additive Gaussian noise still recovers exact 3x3 positions
    cuts, true, nominal = _tiles(scene, jitter_seed=9)
    rng = np.random.default_rng(42)
    noisy = np.clip(cuts.astype(np.float64) + rng.normal(0.0, 1500.0, cuts.shape), 0, 65535).astype(np.uint16)
    da_noisy = array_to_stczyx_xarray(
        noisy, 'SYX',
        pixel_sizes=(1.0, 1.0, 1.0),
        stage_positions={'Y': nominal[:, 0], 'X': nominal[:, 1]},
    )
    m_noisy = stitch_tiles(da_noisy)
    np.testing.assert_array_equal(m_noisy.attrs['stitching']['positions_px'], true - true.min(axis=0))


def test_stitch_tiles_rejects_bad_input_and_warns(monkeypatch):
    """Missing stage coords, missing pixel sizes, wrong dims, bad options, and bad transforms raise;
    the pointer to multiview-stitcher is shown once per process.

    Changed deliberately (D32): the warning used to fire on every call; it now fires on the
    first call of the process only (the module flag is reset here to observe it).
    """
    import warnings

    import eigenp_utils.stitching as st

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
    with pytest.raises(ValueError, match="prior_weight"):
        stitch_tiles(ok, prior_weight=0)  # unregistered tiles would collapse onto each other
    with pytest.raises(ValueError, match="prior_weight"):
        stitch_tiles(ok, prior_weight=np.nan)
    with pytest.raises(ValueError, match="min_ncc"):
        stitch_tiles(ok, min_ncc=-1)
    with pytest.raises(ValueError, match="min_ncc"):
        stitch_tiles(ok, min_ncc=np.nan)
    with pytest.raises(ValueError, match="2x2"):
        stitch_tiles(ok, stage_to_image=np.eye(3))
    with pytest.raises(KeyError):
        stitch_tiles(ok, registration_channel='missing_channel')

    monkeypatch.setattr(st, "_MINIMAL_WARNING_SHOWN", False)
    with pytest.warns(UserWarning, match="multiview-stitcher"):
        stitch_tiles(ok, min_overlap=4)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        stitch_tiles(ok, min_overlap=4)  # second call: silent


def test_stitch_tiles_register_false_places_tiles_at_stage_positions():
    """D32: register=False is the stage-only mode (no pair is registered), replacing min_ncc=1.1.

    (+) jittered tiles: register=False gives exactly the stage grid and n_pairs_registered == 0.
    (-) it equals the old min_ncc=1.1 idiom, and register=True (default) still recovers the truth.
    """
    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    da = array_to_stczyx_xarray(cuts, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                stage_positions={'Y': nominal[:, 0], 'X': nominal[:, 1]})
    stage_only = stitch_tiles(da, register=False)
    st = stage_only.attrs['stitching']
    assert st['n_pairs_registered'] == 0 and st['n_pairs'] == 0
    np.testing.assert_array_equal(st['positions_px'], nominal - nominal.min(axis=0))
    xr.testing.assert_identical(stage_only, stitch_tiles(da, min_ncc=1.1))
    registered = stitch_tiles(da)
    np.testing.assert_array_equal(registered.attrs['stitching']['positions_px'], true - true.min(axis=0))


def test_stitch_tiles_converts_stage_units_to_the_pixel_unit():
    """R8-11: stage positions in nm with pixel sizes in µm are converted, not read as µm.

    (+) stage coords in nm (units='nm'): same positions as the same stage given in µm.
    (-) an unconvertible stage unit raises a clear ValueError naming both units.
    """
    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    px = 0.5  # µm
    um = array_to_stczyx_xarray(cuts, 'SYX', pixel_sizes={'Z': 1, 'Y': px, 'X': px, 'unit': 'µm'},
                                stage_positions={'Y': nominal[:, 0] * px, 'X': nominal[:, 1] * px})
    nm = um.assign_coords(
        stage_y=('S', um.stage_y.values * 1000, {'units': 'nm'}),
        stage_x=('S', um.stage_x.values * 1000, {'units': 'nm'}),
    )
    a = stitch_tiles(um, register=False).attrs['stitching']['positions_px']
    b = stitch_tiles(nm, register=False).attrs['stitching']['positions_px']
    assert a == b
    bad = um.assign_coords(stage_y=('S', um.stage_y.values, {'units': 'furlong'}),
                           stage_x=('S', um.stage_x.values, {'units': 'furlong'}))
    with pytest.raises(ValueError, match="furlong"):
        stitch_tiles(bad, register=False)


def test_stitch_tiles_refuses_an_implausibly_large_canvas():
    """R8-11: stage positions far apart relative to the tiles (e.g. nm read as µm) raise before allocating.

    (+) stage values 1000x too large: ValueError mentioning the canvas size, instead of a
        34 GiB allocation; max_area_ratio=None lifts the check.
    (-) a normal mosaic (area ~ n_tiles * tile area) passes.
    """
    tiles = np.zeros((2, 64, 64), np.uint16)
    big = array_to_stczyx_xarray(tiles, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                 stage_positions={'Y': [0, 48000], 'X': [0, 48000]})
    with pytest.raises(ValueError, match="canvas"):
        stitch_tiles(big, register=False)
    near = array_to_stczyx_xarray(tiles, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                  stage_positions={'Y': [0, 0], 'X': [0, 50]})
    assert stitch_tiles(near, register=False).sizes['X'] == 114
    # two tiles 1000 px apart: 64 x 1064 canvas > 4 x the tile area, refused unless asked for
    far = array_to_stczyx_xarray(tiles, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                 stage_positions={'Y': [0, 0], 'X': [0, 1000]})
    with pytest.raises(ValueError, match="max_area_ratio"):
        stitch_tiles(far, register=False)
    assert stitch_tiles(far, register=False, max_area_ratio=None).sizes['X'] == 1064


@pytest.mark.parametrize("seed", range(2, 12))
def test_stitch_tiles_exact_over_many_jitters(seed):
    """Seed sweep of the positive control (stage error up to +/-5 px, 30 px overlap):
    every jitter pattern must give the exact true positions, not just the default seed."""
    scene = _scene(seed=seed)
    cuts, true, nominal = _tiles(scene, jitter_seed=seed)
    da = array_to_stczyx_xarray(cuts, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                stage_positions={'Y': nominal[:, 0], 'X': nominal[:, 1]})
    positions = np.array(stitch_tiles(da).attrs['stitching']['positions_px'])
    np.testing.assert_array_equal(positions, true - true.min(axis=0))


@pytest.mark.parametrize("shape", [(80, 150), (150, 80)])
def test_stitch_tiles_non_square_tiles_and_transposed_stage(shape):
    """Non-square tiles on a grid whose stage reports (x, y) swapped are placed exactly
    when stage_to_image is the transpose matrix."""
    th, tw = shape
    step_y, step_x = th - 25, tw - 25
    rng = np.random.default_rng(3)
    scene = _scene((3 * step_y + th + 20, 3 * step_x + tw + 20), seed=3)
    nominal = np.array([(y, x) for y in np.arange(3) * step_y for x in np.arange(3) * step_x]) + 5
    true = nominal + rng.integers(-5, 6, size=nominal.shape)
    cuts = np.stack([scene[y:y + th, x:x + tw] for y, x in true])
    # The stage reports image rows on its X axis and columns on its Y axis
    da = array_to_stczyx_xarray(cuts, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                stage_positions={'Y': nominal[:, 1], 'X': nominal[:, 0]})
    mosaic = stitch_tiles(da, stage_to_image=[[0, 1], [1, 0]])
    np.testing.assert_array_equal(mosaic.attrs['stitching']['positions_px'], true - true.min(axis=0))
    assert mosaic.sizes['Y'] == np.ptp(true[:, 0]) + th and mosaic.sizes['X'] == np.ptp(true[:, 1]) + tw


def test_stitch_tiles_linear_blending_and_overwrite_order():
    """Verify exact feather weights for float64 and uint8 linear blending, plus overwrite order.

    Strategy: place two featureless tiles overlapping by 20 columns in both
    ``float64`` (100 vs 200) and ``uint8`` (100 vs 201), and compare against the
    closed-form distance-to-edge feather formula and ``blending='overwrite'``.
    Expected outcome: float64 matches the exact weighted blend; uint8 rounds
    properly without truncation or dtype change; overwrite lets S=1 win.
    """
    h, w = 10, 50
    tiles = np.stack([np.full((h, w), 100.0), np.full((h, w), 200.0)])
    da = array_to_stczyx_xarray(tiles, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                stage_positions={'Y': [0, 0], 'X': [0, 30]})
    row = stitch_tiles(da).values[0, 0, 0, 0, h // 2]
    assert row.shape == (80,)
    np.testing.assert_array_equal(row[:30], 100.0)
    np.testing.assert_array_equal(row[50:], 200.0)
    overlap = row[30:50]
    assert np.all(np.diff(overlap) > 0)
    # Column 30 + k: tile 0 weight min(31 + k, 20 - k), tile 1 weight k + 1
    k = np.arange(20)
    w0, w1 = np.minimum(31 + k, 20 - k), k + 1
    np.testing.assert_allclose(overlap, (100.0 * w0 + 200.0 * w1) / (w0 + w1))

    # Integer rounding and dtype preservation
    tiles_u8 = np.stack([np.full((h, w), 100, np.uint8), np.full((h, w), 201, np.uint8)])
    da_u8 = array_to_stczyx_xarray(tiles_u8, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                   stage_positions={'Y': [0, 0], 'X': [0, 30]})
    mosaic_u8 = stitch_tiles(da_u8)
    assert mosaic_u8.dtype == np.uint8
    expected_u8 = np.round((100.0 * w0 + 201.0 * w1) / (w0 + w1)).astype(np.uint8)
    np.testing.assert_array_equal(mosaic_u8.values[0, 0, 0, 0, h // 2, 30:50], expected_u8)

    over = stitch_tiles(da, blending='overwrite').values[0, 0, 0, 0, h // 2]
    np.testing.assert_array_equal(over[30:50], 200.0)
    np.testing.assert_array_equal(over[:30], 100.0)


def test_stitch_tiles_registration_channel_by_name_and_single_tile():
    """Registering on a named channel uses that channel only: channel 'blank' is featureless,
    so registering on it leaves the stage placement, while 'tex' recovers the truth.
    A single tile is returned unchanged."""
    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    data = np.stack([np.zeros_like(cuts), cuts], axis=1)
    da = array_to_stczyx_xarray(data, 'SCYX', channel_names=['blank', 'tex'], pixel_sizes=(1, 1, 1),
                                stage_positions={'Y': nominal[:, 0], 'X': nominal[:, 1]})
    np.testing.assert_array_equal(stitch_tiles(da, registration_channel='tex').attrs['stitching']['positions_px'],
                                  true - true.min(axis=0))
    assert stitch_tiles(da, registration_channel='blank').attrs['stitching']['n_pairs'] == 0

    one = array_to_stczyx_xarray(cuts[:1], 'SYX', pixel_sizes=(1, 1, 1),
                                 stage_positions={'Y': [7.0], 'X': [3.0]})
    np.testing.assert_array_equal(stitch_tiles(one).values[0], one.values[0])


def test_solve_positions_drops_inconsistent_edge():
    """Three tiles in a row with consistent edges 0->1 and 1->2 (+10 px each) and a wrong,
    weaker edge 0->2 (+30 px instead of +20): outlier rejection drops the wrong edge and
    the solution is exact; with a huge max_residual the wrong edge biases the result."""
    prior = np.array([[0.0, 0.0], [0.0, 12.0], [0.0, 24.0]])
    edges = [(0, 1, np.array([0, 10]), 1.0), (1, 2, np.array([0, 10]), 1.0), (0, 2, np.array([0, 30]), 0.7)]
    positions, kept = _solve_positions(prior, edges, prior_weight=1e-3, max_residual=1.0)
    assert len(kept) == 2 and all(e[:2] != (0, 2) for e in kept)
    np.testing.assert_allclose(positions[:, 1] - positions[0, 1], [0, 10, 20], atol=0.05)
    biased, kept_all = _solve_positions(prior, edges, prior_weight=1e-3, max_residual=100.0)
    assert len(kept_all) == 3 and abs(biased[2, 1] - biased[0, 1] - 20) > 1


def test_stitch_tiles_keeps_channel_and_plane_aux_coords():
    """Per-channel metadata (colour, role) and per-plane coordinates survive stitching.

    Strategy: attach ``color`` / ``role`` along C with ``set_channel_info`` (the
    per-channel metadata convention the viewers read colours from) plus aux
    coordinates along T and Z, stitch, and stitch the same tiles without them.
    The mosaic used to be rebuilt with only the C names, so colours were lost.
    Expected outcome: the mosaic carries the same C/T/Z aux coordinates with the
    same values and they follow ``sel`` along C. Negative control: pixel values
    and positions equal those of the plain tiles, and the per-tile S coordinates
    (``stage_y`` / ``stage_x``) are not copied onto the single mosaic S entry.
    """
    from eigenp_utils.image_io import set_channel_info

    scene = _scene()
    cuts, true, nominal = _tiles(scene)
    data = np.stack([np.stack([cuts, cuts // 2], axis=1), np.stack([cuts // 3, cuts // 4], axis=1)], axis=1)
    plain = array_to_stczyx_xarray(data, 'SCZYX', channel_names=['nuc', 'mem'], pixel_sizes=(2.0, 1.0, 1.0),
                                   stage_positions={'Y': nominal[:, 0], 'X': nominal[:, 1]})
    da = set_channel_info(plain, {'nuc': {'color': 'cyan', 'role': 'dna'},
                                  'mem': {'color': 'magenta', 'role': 'membrane'}})
    da = da.assign_coords(z_um=('Z', [10.0, 12.0]), t_s=('T', [3.5]))

    mosaic = stitch_tiles(da)
    for name, dim in (('color', 'C'), ('role', 'C'), ('z_um', 'Z'), ('t_s', 'T')):
        assert mosaic.coords[name].dims == (dim,), name
        np.testing.assert_array_equal(mosaic.coords[name].values, da.coords[name].values)
    assert mosaic.sel(C='mem').coords['color'].item() == 'magenta'

    reference = stitch_tiles(plain)
    np.testing.assert_array_equal(mosaic.values, reference.values)
    assert mosaic.attrs['stitching'] == reference.attrs['stitching']
    np.testing.assert_array_equal(mosaic.attrs['stitching']['positions_px'], true - true.min(axis=0))
    assert 'stage_y' not in mosaic.coords and 'stage_x' not in mosaic.coords


@pytest.mark.parametrize("unit, scale", [("mm", 1e-3), ("nm", 1e3)])
def test_stitch_tiles_converts_stage_units_when_pixel_sizes_have_no_unit(unit, scale):
    """R8-11/D01: unit-less pixel sizes are µm, so stage coords in mm or nm are still converted.

    (+) stage in mm or nm with pixel_sizes given as a plain tuple (unit None): same positions
        as the stage given in µm (before the fix, mm stacked both tiles at 0 and nm raised a
        misleading canvas-size error).
    (-) the same stage given without a unit keeps the pixel unit (reference positions).
    """
    tiles = np.zeros((2, 32, 32), np.uint8)
    um = array_to_stczyx_xarray(tiles, 'SYX', pixel_sizes=(1.0, 0.5, 0.5),
                                stage_positions={'Y': [0, 0], 'X': [0, 14]})
    assert get_stczyx_pixel_sizes(um)['unit'] is None
    ref = stitch_tiles(um, register=False).attrs['stitching']['positions_px']
    assert np.array_equal(ref, [[0, 0], [0, 28]])
    other = um.assign_coords(
        stage_y=('S', um.stage_y.values * scale, {'units': unit}),
        stage_x=('S', um.stage_x.values * scale, {'units': unit}),
    )
    np.testing.assert_array_equal(stitch_tiles(other, register=False).attrs['stitching']['positions_px'], ref)


def test_stitch_tiles_canvas_check_accepts_a_diagonal_strip():
    """R8-11: a diagonal strip of touching tiles is a legitimate layout and is not refused.

    (+) five 100 x 100 tiles stepping (90, 90) px: a 460 x 460 canvas is stitched (it was
        refused at 4.23x the summed tile area).
    (-) the 1000x-too-large stage in test_stitch_tiles_refuses_an_implausibly_large_canvas
        is still refused.
    """
    tiles = np.ones((5, 100, 100), np.uint8)
    steps = np.arange(5) * 90.0
    da = array_to_stczyx_xarray(tiles, 'SYX', pixel_sizes=(1.0, 1.0, 1.0),
                                stage_positions={'Y': steps, 'X': steps})
    out = stitch_tiles(da, register=False)
    assert out.sizes['Y'] == 460 and out.sizes['X'] == 460
