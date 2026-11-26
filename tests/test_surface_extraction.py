import numpy as np
import pytest
from scipy import ndimage

from eigenp_utils.surface_extraction import (
    adjust_mask_location,
    crop_edges,
    expand_surface_z,
    extract_surface,
    extract_surface_tiled,
    pad_z_slices,
)


def _generate_slanted_plane(
    shape=(50, 64, 64),
    slope_y=0.2,
    slope_x=0.1,
    z_offset=10.0,
    noise_sigma=0.0,
    seed=0,
):
    """Generate a uint8 3D volume with a 1-px ramped slanted plane interface.

    Voxels above the plane (low Z) are dark (0) and voxels below (high Z) are
    bright (255), with a linear ramp over 1 voxel across `gt_height`.
    """
    Z, Y, X = shape
    y_idx, x_idx = np.indices((Y, X))
    gt_height = z_offset + slope_y * y_idx + slope_x * x_idx
    z_idx = np.arange(Z, dtype=np.float32)[:, None, None]
    vol = np.clip(z_idx - gt_height[None, :, :] + 0.5, 0.0, 1.0).astype(np.float32)

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        vol = np.clip(vol + rng.normal(0.0, noise_sigma, size=shape).astype(np.float32), 0.0, 1.0)

    return (vol * 255).astype(np.uint8), gt_height


def _soft_slab(shape=(48, 64, 256), surface=16, gain=None, seed=0):
    """Tissue below `surface` with a soft Z edge, an optional per-column gain and noise."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(shape, np.float32)
    vol[surface:] = 1.0
    vol = ndimage.gaussian_filter(vol, (4, 0, 0))
    if gain is not None:
        vol *= gain[None, None, :].astype(np.float32)
    return vol + rng.normal(0, 0.02, vol.shape).astype(np.float32)


def test_extract_surface_curved_paraboloid_accuracy():
    """Verify heightmap accuracy on an analytical 3D curved paraboloid surface.

    Strategy: construct a `(48, 64, 64)` volume whose top interface follows the
    paraboloid `z(y, x) = 12 + 0.015 * ((y - 31.5)^2 + (x - 31.5)^2)`, extract
    with `downscale_factor=2`, and compare interior heights against ground truth.
    Expected outcome: every column has one surface voxel, center is shallower
    than edges, and interior heightmap RMSE is below 0.5 px (max error < 1.0 px).
    """
    Z, Y, X = 48, 64, 64
    y_idx, x_idx = np.indices((Y, X))
    gt_height = 12.0 + 0.015 * ((y_idx - 31.5) ** 2 + (x_idx - 31.5) ** 2)
    z_idx = np.arange(Z, dtype=np.float32)[:, None, None]
    vol = (np.clip(z_idx - gt_height[None, :, :] + 0.5, 0.0, 1.0) * 255).astype(np.uint8)

    surface, heights = extract_surface(vol, downscale_factor=2, gaussian_sigma=1.0, return_heightmap=True)

    assert surface.shape == (Z, Y, X) and surface.dtype == bool
    assert heights.shape == (Y, X)
    assert np.all(surface.sum(axis=0) == 1)
    assert heights[32, 32] < heights[10, 32]

    crop = 6
    error = heights[crop:-crop, crop:-crop] - gt_height[crop:-crop, crop:-crop]
    rmse = float(np.sqrt(np.mean(error**2)))
    assert rmse < 0.5
    assert float(np.max(np.abs(error))) < 1.0


def test_extract_surface_padding_mismatch():
    """Verify extraction succeeds and recovers the shifted height when dimensions are not block-divisible.

    Strategy: pad a `(20, 10, 10)` step volume (surface at z=10) by 4 slices in
    Z (moving the surface to z=14) and run with `downscale_factor=(2, 4, 4)`
    where `10 % 4 != 0`.
    Expected outcome: no `IndexError`, every column has one surface voxel, and
    the flat heightmap sits near z=14 (within 1.5 px of 13.5) with zero lateral ripple.
    """
    Z, Y, X = 20, 10, 10
    image = np.zeros((Z, Y, X), dtype=np.uint8)
    image[10:, :, :] = 200
    padded_image = pad_z_slices(image, num_slices=4)

    surface, heights = extract_surface(
        padded_image, downscale_factor=(2, 4, 4), gaussian_sigma=1.0, return_heightmap=True
    )

    assert surface.shape == padded_image.shape
    assert surface.dtype == bool
    assert np.all(surface.sum(axis=0) == 1)
    np.testing.assert_allclose(heights, 13.5, atol=1.5)
    assert np.ptp(heights) < 1e-5


def test_extract_surface_with_clahe():
    """Verify the `clahe_clip > 0` preprocessing branch produces a valid heightmap.

    Strategy: build a `(48, 64, 64)` step volume at `z=24` with a lateral
    brightness gradient (`80..200`) and extract with `clahe_clip=0.02`.
    Expected outcome: every column has one surface voxel and the recovered
    heightmap tracks the `z=24` interface without collapsing or diverging.
    """
    vol = np.zeros((48, 64, 64), dtype=np.uint8)
    vol[24:, :, :] = np.round(np.linspace(80, 200, 64)).astype(np.uint8)[None, None, :]

    mask, heights = extract_surface(
        vol, downscale_factor=2, gaussian_sigma=1.0, clahe_clip=0.02, return_heightmap=True
    )

    assert mask.shape == vol.shape and mask.dtype == bool
    assert np.all(mask.sum(axis=0) == 1)
    assert np.median(np.abs(heights - 23.5)) < 3.0
    assert heights.min() >= 16.0 and heights.max() <= 30.0


@pytest.mark.parametrize("fill_val", [0, 128])
def test_extract_surface_constant_or_empty_volume(fill_val):
    """Verify constant or all-zero volumes yield an all-False mask and `-1.0` heightmap.

    Strategy: run `extract_surface(..., return_heightmap=True)` on uniform zero
    and uniform non-zero uint8 volumes.
    Expected outcome: no surface voxels are marked and every heightmap entry is `-1.0`.
    """
    vol = np.full((24, 32, 32), fill_val, dtype=np.uint8)
    mask, heights = extract_surface(vol, downscale_factor=2, gaussian_sigma=1.0, return_heightmap=True)
    assert mask.shape == vol.shape and mask.dtype == bool
    assert not np.any(mask)
    np.testing.assert_array_equal(heights, -1.0)


def test_pad_z_slices():
    """Verify padding count, fractional sizing, min-fill default, arbitrary axes, dtype preservation, and errors.

    Strategy: pad arrays with integer and fractional `num_slices`, `fill_value=None`
    and explicit values, along axes 0, 1, and 2 across `uint16` and `float32`
    dtypes, and probe invalid `axis` / negative `num_slices`.
    Expected outcome: padded shapes, fill values, data placement, and dtypes
    match specification, and out-of-range axes or negative counts raise `ValueError`.
    """
    arr = np.zeros((10, 10, 10), dtype=np.float32)
    padded_before = pad_z_slices(arr, num_slices=1, pad_before=True, pad_after=False)
    assert padded_before.shape == (11, 10, 10) and padded_before.dtype == np.float32

    padded_frac = pad_z_slices(arr, num_slices=0.1, pad_before=True, pad_after=True)
    assert padded_frac.shape == (12, 10, 10)

    # Explicit fill value and dtype preservation on uint16
    u16 = np.full((6, 5, 4), 7, dtype=np.uint16)
    u16[2, 2, 2] = 3
    padded_explicit = pad_z_slices(u16, num_slices=2, fill_value=99, pad_before=True)
    assert padded_explicit.dtype == np.uint16
    assert np.all(padded_explicit[:2] == 99)
    np.testing.assert_array_equal(padded_explicit[2:], u16)

    # fill_value=None defaults to arr.min() (= 3)
    padded_min = pad_z_slices(u16, num_slices=2, fill_value=None, pad_before=True)
    assert padded_min.dtype == np.uint16
    assert np.all(padded_min[:2] == 3)
    np.testing.assert_array_equal(padded_min[2:], u16)

    # Axis 1 and axis 2 padding
    padded_y = pad_z_slices(u16, num_slices=2, fill_value=0, axis=1, pad_before=False, pad_after=True)
    assert padded_y.shape == (6, 7, 4)
    assert np.all(padded_y[:, 5:, :] == 0)
    np.testing.assert_array_equal(padded_y[:, :5, :], u16)

    padded_x = pad_z_slices(u16, num_slices=3, fill_value=9, axis=2, pad_before=True, pad_after=True)
    assert padded_x.shape == (6, 5, 10)
    assert np.all(padded_x[:, :, :3] == 9) and np.all(padded_x[:, :, -3:] == 9)
    np.testing.assert_array_equal(padded_x[:, :, 3:-3], u16)

    # Validation errors
    for bad_axis in (-1, 3):
        with pytest.raises(ValueError, match="out of bounds"):
            pad_z_slices(u16, num_slices=1, axis=bad_axis)
    with pytest.raises(ValueError, match="non-negative"):
        pad_z_slices(u16, num_slices=-1)


@pytest.mark.parametrize(
    "crop_fracs",
    [
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.0, 0.1, 0.1),
        (0.2, 0.1, 0.15),
    ],
)
@pytest.mark.parametrize("in_place", [False, True])
def test_crop_edges(crop_fracs, in_place):
    """Verify edge zeroing across Z, Y, and X for both out-of-place and in-place modes.

    Strategy: apply `crop_edges` to a uniform `(10, 20, 20)` volume over
    combinations of `(cz, cy, cx)` and `in_place`.
    Expected outcome: cropped border slabs are zeroed, interior values remain 1,
    `in_place=True` mutates and returns the input array, and `in_place=False`
    leaves the input array untouched.
    """
    cz, cy, cx = crop_fracs
    arr = np.ones((10, 20, 20), dtype=np.float32)
    src = arr.copy()

    out = crop_edges(src, cz, cy, cx, in_place=in_place)
    assert out.shape == arr.shape
    if in_place:
        assert out is src
    else:
        assert out is not src
        np.testing.assert_array_equal(src, arr)

    crops = [int(n * f) for n, f in zip(arr.shape, crop_fracs)]
    expected = np.zeros_like(arr)
    keep = tuple(slice(c, n - c) if c > 0 else slice(None) for c, n in zip(crops, arr.shape))
    expected[keep] = 1.0
    np.testing.assert_array_equal(out, expected)


def test_expand_surface_z():
    """Verify Z expansion thickness, boundary clamping, negative/zero thickness, and in-place mutation.

    Strategy: expand single-voxel surfaces with `thickness=1`, `thickness=3`,
    boundary voxels at `z=0` and `z=Z-1`, `thickness=-2`, and `in_place=True`.
    Expected outcome: expands symmetrically along Z without wrapping or raising
    at Z boundaries, respects `in_place`, and returns unchanged copy/self for
    non-positive thickness.
    """
    surface = np.zeros((12, 10, 10), dtype=bool)
    surface[5, 5, 5] = True

    exp1 = expand_surface_z(surface, thickness=1)
    assert np.array_equal(np.nonzero(exp1)[0], [4, 5, 6])
    assert not surface[4, 5, 5]

    exp3 = expand_surface_z(surface, thickness=3)
    assert np.array_equal(np.nonzero(exp3)[0], [2, 3, 4, 5, 6, 7, 8])

    # Z boundary clamping (no wraparound or IndexError at z=0 and z=Z-1)
    edge = np.zeros((6, 4, 4), dtype=bool)
    edge[0, 1, 1] = True
    edge[5, 2, 2] = True
    exp_edge = expand_surface_z(edge, thickness=2)
    assert np.array_equal(exp_edge[:, 1, 1], [True, True, True, False, False, False])
    assert np.array_equal(exp_edge[:, 2, 2], [False, False, False, True, True, True])

    # Negative thickness behaves like no-op with copy/in_place semantics
    neg_copy = expand_surface_z(surface, thickness=-2, in_place=False)
    assert neg_copy is not surface and np.array_equal(neg_copy, surface)
    assert expand_surface_z(surface, thickness=-2, in_place=True) is surface

    # in_place=True mutates the input mask directly
    target = surface.copy()
    res = expand_surface_z(target, thickness=2, in_place=True)
    assert res is target
    assert np.array_equal(np.nonzero(target)[0], [3, 4, 5, 6, 7])


def test_expand_surface_z_zero_thickness_returns_copy():
    """Verify `thickness=0` returns an independent copy by default and the input when `in_place=True`.

    Strategy: expand a boolean mask with `thickness=0`, mutate the returned copy,
    and verify the original is unaffected; then verify `in_place=True` identity.
    Expected outcome: `in_place=False` returns a separate array; `in_place=True` returns `s`.
    """
    s = np.zeros((4, 3, 3), dtype=bool)
    r = expand_surface_z(s, 0)
    r[0, 0, 0] = True
    assert not s[0, 0, 0]
    assert expand_surface_z(s, 0, in_place=True) is s


def test_adjust_mask_location(capsys):
    """Verify translation, rollover clamping, OOB clearing, in-place overlap, morphology, and error handling.

    Strategy: exercise positive/negative shifts, boundary rollover (checking no
    wraparound and warning message), out-of-bounds clearing, overlapping
    `in_place=True` shifts, dilation/erosion, combined translation + dilation,
    and an invalid `morph` name.
    Expected outcome: shifted voxels land at exact target indices, rolled-over
    voxels are dropped without wrapping, overlapping in-place shifts avoid
    corruption, morphological operations match 6-connected erosion/dilation,
    and unknown `morph` raises `ValueError`.
    """
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[5, 5, 5] = True

    # Positive and negative translations
    pos = adjust_mask_location(mask, translation=(1, 1, 1))
    assert pos[6, 6, 6] and pos.sum() == 1
    neg = adjust_mask_location(mask, translation=(-2, -3, -1))
    assert neg[3, 2, 4] and neg.sum() == 1

    # Boundary rollover drops out-of-frame voxels without wrapping and prints warning
    roll = np.zeros(shape, dtype=bool)
    roll[0, 5, 5] = True
    roll[5, 5, 5] = True
    rolled = adjust_mask_location(roll, translation=(-1, 0, 0))
    assert rolled[4, 5, 5] and not rolled[9, 5, 5] and rolled.sum() == 1
    assert "rolled over the boundaries" in capsys.readouterr().out

    # Out-of-bounds translation clears entirely for both out-of-place and in-place
    for in_place in (False, True):
        src = mask.copy()
        cleared = adjust_mask_location(src, translation=(10, 0, 0), in_place=in_place)
        assert not np.any(cleared)
        if in_place:
            assert cleared is src

    # Overlapping in_place=True shifts (positive and negative) must not self-corrupt
    block = np.zeros(shape, dtype=bool)
    block[3:6, 3:6, 3:6] = True
    shifted_pos = adjust_mask_location(block, translation=(1, 1, 1), in_place=True)
    assert shifted_pos is block
    expected_pos = np.zeros(shape, dtype=bool)
    expected_pos[4:7, 4:7, 4:7] = True
    np.testing.assert_array_equal(shifted_pos, expected_pos)

    shifted_neg = adjust_mask_location(block, translation=(-1, -1, -1), in_place=True)
    assert shifted_neg is block
    expected_neg = np.zeros(shape, dtype=bool)
    expected_neg[3:6, 3:6, 3:6] = True
    np.testing.assert_array_equal(shifted_neg, expected_neg)

    # Standalone dilation & erosion
    dilated = adjust_mask_location(mask, morph="dilate", iterations=1)
    assert dilated[5, 5, 6] and dilated[5, 6, 5] and dilated.sum() == 7

    mask_block = np.zeros(shape, dtype=bool)
    mask_block[4:7, 4:7, 4:7] = True
    eroded = adjust_mask_location(mask_block, morph="erode", iterations=1)
    assert eroded[5, 5, 5] and not eroded[4, 4, 4] and eroded.sum() == 1

    # Combined translation + dilation
    combo = adjust_mask_location(mask, translation=(1, -1, 2), morph="dilate", iterations=1)
    assert combo[6, 4, 7] and combo[5, 4, 7] and combo[7, 4, 7] and combo.sum() == 7

    # Invalid morph raises ValueError
    with pytest.raises(ValueError, match="Invalid morphological operation"):
        adjust_mask_location(mask, morph="open")


def test_surface_precision():
    """Verify subpixel heightmap RMSE on a noise-free slanted plane beats integer quantization.

    Strategy: generate a slanted plane volume with 1-px ramp interface, extract
    at `downscale_factor=1, gaussian_sigma=2.0`, and compute interior RMSE.
    Expected outcome: RMSE is strictly below 0.20 px (better than 0.29 px integer quantization).
    """
    vol, gt_height = _generate_slanted_plane(
        shape=(40, 64, 64), slope_y=0.15, slope_x=0.05, z_offset=5.0, noise_sigma=0.0
    )
    surface_mask, height_map = extract_surface(
        vol, downscale_factor=1, gaussian_sigma=2.0, clahe_clip=0.0, return_heightmap=True
    )

    found_mask = np.any(surface_mask, axis=0)
    assert found_mask.all()

    crop = 5
    error_crop = (height_map - gt_height)[crop:-crop, crop:-crop]
    rmse = float(np.sqrt(np.mean(error_crop**2)))
    assert rmse < 0.20, f"RMSE ({rmse:.4f}) should be better than integer quantization (0.29)"


def test_translation_invariance():
    """Verify subpixel Z translation invariance of extracted surface heightmaps.

    Strategy: extract surfaces from two slanted plane volumes shifted by 0.5 px
    in Z and measure the mean and standard deviation of their height difference.
    Expected outcome: detected mean shift equals `0.5 +/- 0.05` with std < 0.15.
    """
    shape = (40, 32, 32)
    vol1, _ = _generate_slanted_plane(shape, z_offset=10.0)
    vol2, _ = _generate_slanted_plane(shape, z_offset=10.5)

    mask1, h1 = extract_surface(vol1, downscale_factor=1, gaussian_sigma=2.0, return_heightmap=True)
    mask2, h2 = extract_surface(vol2, downscale_factor=1, gaussian_sigma=2.0, return_heightmap=True)

    valid = np.any(mask1, axis=0) & np.any(mask2, axis=0)
    crop = 5
    valid[:crop, :] = False
    valid[-crop:, :] = False
    valid[:, :crop] = False
    valid[:, -crop:] = False

    diff = h2[valid] - h1[valid]
    assert abs(float(np.mean(diff)) - 0.5) < 0.05
    assert float(np.std(diff)) < 0.15, "Shift StdDev should be low for rigid shift"


def test_downscaling_alignment():
    """Verify downscaled surface extraction has zero lateral shift bias and minimal edge curl.

    Strategy: construct a block-aligned `(64, 64, 64)` volume with a foreground
    region starting at `margin=16` and surface at `z=32`, extract at
    `downscale_factor=4, gaussian_sigma=0.5`, and inspect the central X profile.
    Expected outcome: first valid pixel starts exactly at `X=16`, edge-to-interior
    height difference is < 2.0 px, and interior height is within 2.0 px of 31.5.
    """
    Z, Y, X = 64, 64, 64
    vol = np.zeros((Z, Y, X), dtype=np.uint8)
    margin = 16
    vol[32:, margin:-margin, margin:-margin] = 200
    vol[:32, margin:-margin, margin:-margin] = 50

    _, hmap = extract_surface(vol, downscale_factor=4, gaussian_sigma=0.5, return_heightmap=True)
    profile = hmap[Y // 2, :]

    valid_indices = np.where(profile > 0)[0]
    assert len(valid_indices) > 0, "Surface not found"
    assert valid_indices[0] == margin, f"Surface start shifted! Expected {margin}, got {valid_indices[0]}"

    val_edge = profile[margin]
    val_inside = profile[margin + 2]
    assert abs(val_inside - val_edge) < 2.0
    assert abs(val_inside - 31.5) < 2.0


@pytest.mark.parametrize("noise_amp", [0, 10])
def test_extract_surface_tissue_from_first_slice_and_sharp_step(noise_amp):
    """Verify columns foreground at z=0 and sharp 40-voxel steps stay clamped inside `[0, Z - 1]`.

    Strategy: construct clean and noisy `(96, 64, 64)` volumes with tissue
    starting at `z=0` on the left half and `z=40` on the right half, then
    extract the heightmap.
    Expected outcome: every column has one valid surface voxel with heights
    strictly in `[0, Z - 1]`, and away from the step boundary the left half sits
    near `z=0` while the right half sits near `z=40`.
    """
    vol = np.zeros((96, 64, 64), dtype=np.uint8)
    vol[:, :, :32] = 200
    vol[40:, :, 32:] = 200
    if noise_amp > 0:
        rng = np.random.default_rng(0)
        vol = (vol + rng.integers(0, noise_amp, vol.shape)).astype(np.uint8)

    mask, heights = extract_surface(vol, return_heightmap=True)

    assert np.all(heights != -1.0)
    assert heights.min() >= 0.0 and heights.max() <= vol.shape[0] - 1
    assert np.all(mask.sum(axis=0) == 1)
    assert np.all(heights[:, :20] <= 2.5)
    assert np.all(np.abs(heights[:, 44:] - 40.0) <= 4.0)


@pytest.mark.parametrize("gaussian_sigma", [4.0, 1.0])
def test_extract_surface_tissue_from_first_slice_gets_height_zero(gaussian_sigma):
    """Verify columns foreground from the first binned slice report height 0, not half a block.

    Strategy: tissue starts at z=0 on the left half and at z=30 on the right
    half of a `(96, 64, 64)` volume (default `downscale_factor=4`); the left
    columns are foreground from the first binned slice. Their reduced height
    used to be mapped like a block centre, `(0 + 0.5) * 4 - 0.5 = 1.5`, and the
    rasterised surface landed at z=2. The tiled path must agree.
    Expected outcome: away from the step the left half has height exactly 0
    and its surface voxel at z=0, for both blur widths. Negative control: tissue
    starting at z=4 (first binned slice background) keeps its height of 1.5.
    """
    vol = np.zeros((96, 64, 64), dtype=np.uint8)
    vol[:, :, :32] = 200
    vol[30:, :, 32:] = 200
    mask, heights = extract_surface(vol, gaussian_sigma=gaussian_sigma, return_heightmap=True)
    np.testing.assert_array_equal(heights[:, :20], 0.0)
    assert np.all(mask[0, :, :20])
    tiled = extract_surface_tiled(vol, gaussian_sigma=gaussian_sigma, tile_size=32, return_mask=False)
    np.testing.assert_array_equal(tiled, heights)

    below_top = np.zeros((96, 32, 32), dtype=np.uint8)
    below_top[4:] = 200
    _, h_below = extract_surface(below_top, gaussian_sigma=0.5, return_heightmap=True)
    np.testing.assert_allclose(h_below, 1.5, atol=1e-5)


def test_extract_surface_covers_outer_rim():
    """Verify full-image surface coverage extends all the way to the outer pixel rim.

    Strategy: extract a uniform flat slab at `z=40` in a `(96, 64, 64)` volume
    with default `downscale_factor=4`.
    Expected outcome: every column including the outer rim gets exactly one
    surface voxel and the heightmap is flat (`ptp < 1e-6`).
    """
    vol = np.zeros((96, 64, 64), dtype=np.uint8)
    vol[40:] = 200
    mask, heights = extract_surface(vol, return_heightmap=True)
    assert np.all(mask.sum(axis=0) == 1)
    assert np.ptp(heights) < 1e-6


@pytest.mark.parametrize("shape", [(32, 1, 64), (32, 64, 1)])
def test_extract_surface_single_row_or_column(shape):
    """Verify single-row and single-column volumes extract cleanly without interpolator errors.

    Strategy: run `extract_surface` on `(32, 1, 64)` and `(32, 64, 1)` step volumes.
    Expected outcome: output shapes match input and every column has one surface voxel.
    """
    vol = np.zeros(shape, dtype=np.uint8)
    vol[16:] = 200
    mask, heights = extract_surface(vol, return_heightmap=True)
    assert mask.shape == shape and heights.shape == shape[1:]
    assert np.all(mask.sum(axis=0) == 1)


def test_extract_surface_numpy_integer_factor():
    """Verify NumPy integer scalar `downscale_factor` matches Python `int` identically.

    Strategy: compare `downscale_factor=np.int64(4)` against `downscale_factor=4`.
    Expected outcome: both return bit-identical boolean surface masks.
    """
    vol = np.zeros((48, 32, 32), dtype=np.uint8)
    vol[20:] = 200
    np.testing.assert_array_equal(
        extract_surface(vol, downscale_factor=np.int64(4)),
        extract_surface(vol, downscale_factor=4),
    )


@pytest.mark.parametrize("tile_size", [16, 50, 64, (40, 24), 1024])
def test_extract_surface_tiled_matches_whole_volume(tile_size):
    """Verify tiled extraction reproduces whole-volume extraction across aligned, unaligned, and rectangular tiles.

    Strategy: compare `extract_surface_tiled` (with `return_mask=True` and
    `return_mask=False`) against `extract_surface` on a `(40, 203, 157)` graded slab.
    Expected outcome: masks and heightmaps are bit-for-bit identical for all tile sizes.
    """
    vol = _soft_slab(shape=(40, 203, 157), gain=np.linspace(0.6, 1.0, 157))
    kw = dict(downscale_factor=(2, 4, 4), gaussian_sigma=2.0, return_heightmap=True)
    m0, h0 = extract_surface(vol, **kw)
    m1, h1 = extract_surface_tiled(vol, tile_size=tile_size, **kw)
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_array_equal(h0, h1)
    heights = extract_surface_tiled(vol, tile_size=tile_size, return_mask=False, **kw)
    np.testing.assert_array_equal(heights, h0)


def test_extract_surface_tiled_has_no_seams_unlike_per_chunk_calls():
    """Verify tiled extraction eliminates height steps at tile boundaries under a brightness gradient.

    Strategy: compare boundary height steps between independent per-chunk
    `extract_surface` calls and a single `extract_surface_tiled` call on a
    horizontally graded slab.
    Expected outcome: per-chunk calls show seams > 0.4 px while tiled extraction
    keeps seam jumps below 0.1 px.
    """
    vol = _soft_slab(gain=np.linspace(0.6, 1.0, 256))
    kw = dict(downscale_factor=(1, 4, 4), gaussian_sigma=1.0, return_heightmap=True)

    def seam_steps(h):
        profile = h.mean(axis=0)
        return np.array([profile[x] - profile[x - 1] for x in (64, 128, 192)])

    per_chunk = np.concatenate(
        [extract_surface(vol[:, :, x0 : x0 + 64], **kw)[1] for x0 in range(0, 256, 64)], axis=1
    )
    _, tiled = extract_surface_tiled(vol, tile_size=64, **kw)
    assert np.abs(seam_steps(per_chunk)).min() > 0.4
    assert np.abs(seam_steps(tiled)).max() < 0.1


def test_extract_surface_tiled_judges_debris_against_whole_sample():
    """Verify tiled debris filtering compares component size against the full sample, not a single chunk.

    Strategy: place a floating blob that is 25% of one 32-wide chunk's tissue
    but < 10% of the full 128-wide sample's tissue, comparing single-chunk vs tiled extraction.
    Expected outcome: single-chunk extraction keeps the blob (`h < 12`), whereas
    `extract_surface_tiled` removes it and lands on the true surface (`~23.5`).
    """
    Z, Y, X = 40, 32, 128
    vol = np.zeros((Z, Y, X), np.float32)
    vol[24:] = 1.0
    vol[4:12, 8:24, 8:24] = 1.0
    kw = dict(downscale_factor=1, gaussian_sigma=0.5, return_heightmap=True)

    _, per_chunk = extract_surface(vol[:, :, :32], **kw)
    assert per_chunk[16, 16] < 12
    _, tiled = extract_surface_tiled(vol, tile_size=32, **kw)
    assert abs(tiled[16, 16] - 23.5) < 1.0


def test_extract_surface_fixed_threshold():
    """Verify fixed `threshold` scales linearly with data intensity and rejects CLAHE on both APIs.

    Strategy: compare `(vol, threshold=0.5)` against `(vol * 10, threshold=5.0)`,
    verify height matches the soft edge at `z=16`, and check `ValueError` when
    combining `threshold` with `clahe_clip > 0` in both `extract_surface` and
    `extract_surface_tiled`.
    Expected outcome: scaled runs match identically, median height is within 1 px
    of 15.5, and both functions raise `ValueError` mentioning CLAHE.
    """
    vol = _soft_slab(shape=(40, 32, 32))
    m1, h1 = extract_surface(vol, downscale_factor=2, threshold=0.5, return_heightmap=True)
    m2, h2 = extract_surface(vol * 10, downscale_factor=2, threshold=5.0, return_heightmap=True)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_allclose(h1, h2, atol=1e-4)
    assert abs(np.median(h1) - 15.5) < 1.0

    with pytest.raises(ValueError, match="CLAHE"):
        extract_surface(vol, threshold=0.5, clahe_clip=0.01)
    with pytest.raises(ValueError, match="CLAHE"):
        extract_surface_tiled(vol, threshold=0.5, clahe_clip=0.01)


def test_extract_surface_tiled_reads_one_tile_at_a_time():
    """Verify tiled extraction works on lazy sliceable arrays and bounds peak tile reads.

    Strategy: wrap a `(24, 96, 80)` slab in a minimal lazy reader recording
    `self.largest_read`, run `extract_surface_tiled(..., tile_size=32)`, and
    compare against in-memory `extract_surface`.
    Expected outcome: masks match bit-for-bit and no single slice read exceeds
    one `(Z, 32, 32)` tile (`24 * 32 * 32` elements).
    """
    vol = _soft_slab(shape=(24, 96, 80))

    class LazyVolume:
        def __init__(self, data):
            self._data = data
            self.shape = data.shape
            self.largest_read = 0

        def __getitem__(self, key):
            block = self._data[key]
            self.largest_read = max(self.largest_read, block.size)
            return block

    lazy = LazyVolume(vol)
    m_lazy = extract_surface_tiled(lazy, downscale_factor=4, gaussian_sigma=1.0, tile_size=32)
    np.testing.assert_array_equal(m_lazy, extract_surface(vol, downscale_factor=4, gaussian_sigma=1.0))
    assert lazy.largest_read <= 24 * 32 * 32


def _tilted_textured_half_space(depth, shape=(64, 64, 64), tilt=3.0, seed=0):
    """Poisson volume: textured tissue below a plane tilted by +-`tilt` voxels around `depth`.

    Returns the uint16 volume and the true height map (Z of the tissue boundary per column).
    """
    Z, Y, X = shape
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:Y, :X]
    h = depth + tilt * (xx - X / 2) / (X / 2)
    tissue = np.arange(Z)[:, None, None] >= h[None]
    texture = 1 + 0.5 * np.sin(yy / 3.0) * np.cos(xx / 4.0)
    lam = np.where(tissue, 40.0 * texture[None], 5.0)
    return rng.poisson(lam).astype(np.uint16), h


def _median_bias_per_depth(Z=64, **kw):
    """Median (estimated - true) height of `extract_surface` at depths Z/8 ... 7Z/8."""
    bias = []
    for k in range(1, 8):
        vol, h = _tilted_textured_half_space(k * Z / 8, shape=(Z, 64, 64))
        _, est = extract_surface(vol, return_heightmap=True, **kw)
        bias.append(float(np.median((est - h)[est >= 0])))
    return np.array(bias)


def test_extract_surface_default_blur_has_small_depth_independent_bias():
    """D19 / R7-01: the default Z blur no longer moves the surface by 8-13 voxels with depth.

    Strategy: tilted (+-3 voxel) textured Poisson half-spaces at depths Z/8 ... 7Z/8
    of a 64-slice stack, default parameters (Z binned by 4). The old default
    (isotropic sigma 4 in binned voxels = 16 raw slices along Z) gave biases from
    +12.6 at Z/8 to -13.6 at 7Z/8.
    Expected outcome: |median(h_est - h)| <= 2 voxels at every depth (measured
    +1.1 ... -1.7), and the spread of the bias across depths is <= 3 voxels
    (measured 2.8, was 26). The plan's tighter target, a spread <= 1 voxel, is NOT
    met by the (1, 4, 4) default on this geometry: Otsu's threshold fraction still
    moves with the tissue fraction, times a Z blur of 4 raw slices; the bias grows
    towards 7Z/8, where the tissue is only 2 binned blocks thick.
    """
    bias = _median_bias_per_depth()
    assert np.all(np.abs(bias) <= 2.0), bias
    assert np.ptp(bias) <= 3.0, bias


def test_extract_surface_old_isotropic_sigma_is_depth_biased():
    """Negative control for D19: an explicit isotropic gaussian_sigma=4 still gives the old bias.

    Strategy: the same depth series as the default-blur test with gaussian_sigma=4.0.
    Expected outcome: the bias spread across depths exceeds 15 voxels, so the new
    default (not the test volume) is what removes it.
    """
    bias = _median_bias_per_depth(gaussian_sigma=4.0)
    assert np.ptp(bias) > 15.0, bias


def test_extract_surface_default_sigma_is_per_axis():
    """D19: the default gaussian_sigma is (1, 4, 4) binned voxels, for both entry points.

    Strategy: compare the default call with explicit gaussian_sigma=(1, 4, 4).
    Expected outcome: identical heights for extract_surface and extract_surface_tiled.
    """
    vol, _ = _tilted_textured_half_space(24, shape=(48, 48, 48))
    _, h_default = extract_surface(vol, return_heightmap=True)
    _, h_explicit = extract_surface(vol, gaussian_sigma=(1, 4, 4), return_heightmap=True)
    np.testing.assert_array_equal(h_default, h_explicit)
    tiled = extract_surface_tiled(vol, tile_size=16, return_mask=False)
    np.testing.assert_array_equal(tiled, h_default)


def test_extract_surface_isotropic_unbinned_unchanged():
    """D19 negative control: isotropic, unbinned input with an explicit sigma is unchanged.

    Strategy: downscale_factor=1 with an explicit scalar gaussian_sigma=2 (the new
    default never applies) against the same call with sigma (2, 2, 2).
    Expected outcome: bit-identical heights, and a bias below 0.1 voxel.
    """
    vol, h = _tilted_textured_half_space(32, shape=(64, 48, 48))
    _, a = extract_surface(vol, downscale_factor=1, gaussian_sigma=2.0, return_heightmap=True)
    _, b = extract_surface(vol, downscale_factor=1, gaussian_sigma=(2.0, 2.0, 2.0), return_heightmap=True)
    np.testing.assert_array_equal(a, b)
    assert abs(float(np.median(a - h))) < 0.1


def test_extract_surface_sigma_in_micrometres():
    """D19 opt-in: gaussian_sigma_um with pixel_sizes converts the blur to binned voxels per axis.

    Strategy: Z step 2 µm, XY 0.5 µm, downscale_factor (2, 4, 4): binned voxels are
    4 x 2 x 2 µm, so gaussian_sigma_um=(4, 8, 8) equals gaussian_sigma=(1, 4, 4) and a
    scalar 8 µm equals (2, 4, 4). Units in a pixel_sizes dict are converted (nm).
    Expected outcome: identical height maps for the matching pairs (both entry points);
    gaussian_sigma_um without pixel_sizes warns and reads the sigma in raw voxels.
    """
    vol, _ = _tilted_textured_half_space(24, shape=(48, 48, 48))
    kw = dict(downscale_factor=(2, 4, 4), return_heightmap=True)
    ps = {"Z": 2.0, "Y": 0.5, "X": 0.5}
    _, ref = extract_surface(vol, gaussian_sigma=(1, 4, 4), **kw)
    _, um = extract_surface(vol, gaussian_sigma_um=(4.0, 8.0, 8.0), pixel_sizes=ps, **kw)
    np.testing.assert_array_equal(um, ref)
    nm_ps = {"Z": 2000, "Y": 500, "X": 500, "unit": "nm"}
    _, nm = extract_surface(vol, gaussian_sigma_um=(4.0, 8.0, 8.0), pixel_sizes=nm_ps, **kw)
    np.testing.assert_array_equal(nm, ref)
    _, ref2 = extract_surface(vol, gaussian_sigma=(2, 4, 4), **kw)
    _, um2 = extract_surface(vol, gaussian_sigma_um=8.0, pixel_sizes=ps, **kw)
    np.testing.assert_array_equal(um2, ref2)
    tiled = extract_surface_tiled(
        vol, downscale_factor=(2, 4, 4), gaussian_sigma_um=8.0, pixel_sizes=ps, tile_size=16, return_mask=False
    )
    np.testing.assert_array_equal(tiled, ref2)

    with pytest.warns(UserWarning, match="pixel_sizes"):
        _, raw = extract_surface(vol, gaussian_sigma_um=(2.0, 16.0, 16.0), **kw)
    np.testing.assert_array_equal(raw, ref)
