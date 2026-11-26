import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from eigenp_utils.extended_depth_of_focus import apply_median_filter, best_focus_image


def _generate_focus_stack(shape, z_map, sigma=1.0, seed=42):
    """Generate a synthetic focus stack whose local high-frequency amplitude peaks at z_map."""
    Z, H, W = shape
    rng = np.random.default_rng(seed)
    texture = rng.uniform(0, 1, (H, W))
    z_indices = np.arange(Z).reshape(-1, 1, 1)
    weights = np.exp(-((z_indices - z_map) ** 2) / (2 * sigma**2))
    stack = texture * weights
    return (stack - stack.min()) / (stack.max() - stack.min() + 1e-8)


def _generate_feature_stack(shape=(10, 200, 200), feature_size=60, seed=42):
    """Generate a 3D stack with a central square feature focused at Z=7 on a Z=2 background."""
    Z, H, W = shape
    rng = np.random.default_rng(seed)
    texture = rng.random((H, W))

    depth_map = np.full((H, W), 2, dtype=int)
    mid_h, mid_w = H // 2, W // 2
    half_size = feature_size // 2
    depth_map[
        mid_h - half_size : mid_h + half_size,
        mid_w - half_size : mid_w + half_size,
    ] = 7

    stack = np.zeros(shape, dtype=np.float32)
    mask2 = depth_map == 7
    mask1 = ~mask2
    for z in range(Z):
        sigma1 = abs(2 - z) * 0.5
        sigma2 = abs(7 - z) * 0.5
        layer1 = gaussian_filter(texture, sigma=sigma1) if sigma1 > 0 else texture
        layer2 = gaussian_filter(texture, sigma=sigma2) if sigma2 > 0 else texture
        stack[z] = layer1 * mask1 + layer2 * mask2

    return stack, depth_map


def _generate_synthetic_focal_stack(sharp_image, num_slices=25, k_defocus=0.8, seed=42):
    """Generate a 3D focal stack from a 2D sharp texture using a smooth depth surface z_gt(y, x)."""
    rng = np.random.default_rng(seed)
    H, W = sharp_image.shape
    y, x = np.mgrid[0:H, 0:W]

    z_mid = num_slices / 2.0
    z_gt = (
        z_mid
        + 0.35 * z_mid * np.sin(2.0 * np.pi * y / H)
        + 0.25 * z_mid * np.cos(2.0 * np.pi * x / W)
    ).astype(np.float32)

    max_sigma = k_defocus * num_slices
    sigma_steps = np.linspace(0, max_sigma, 35)
    blurred_levels = [gaussian_filter(sharp_image, sigma=s) for s in sigma_steps]

    focal_stack = np.zeros((num_slices, H, W), dtype=np.float32)
    for z in range(num_slices):
        dist = np.abs(z - z_gt)
        level_idx = np.clip(
            (dist / max_sigma * (len(sigma_steps) - 1)).astype(int),
            0,
            len(sigma_steps) - 1,
        )

        slice_z = np.zeros((H, W), dtype=np.float32)
        for idx in np.unique(level_idx):
            mask = level_idx == idx
            slice_z[mask] = blurred_levels[idx][mask]

        noise = rng.normal(0.0, 0.005, (H, W)).astype(np.float32)
        focal_stack[z] = np.clip(slice_z + noise, 0.0, 1.0)

    return focal_stack, z_gt


def _textured_stack(shape, seed=0):
    """Random uniform float32 stack for edge-case and regression tests."""
    return np.random.default_rng(seed).random(shape).astype(np.float32)


def test_subpixel_focus_recovery():
    """Recover a constant subpixel depth plane with high precision via log-parabolic interpolation."""
    Z, H, W = 10, 64, 64
    z_true = 4.3
    stack = _generate_focus_stack((Z, H, W), z_true, sigma=1.5)

    _, height_map = best_focus_image(stack, patch_size=11, return_heightmap=True)
    valid_region = height_map[10:-10, 10:-10]

    mean_error = float(np.mean(valid_region) - z_true)
    rmse = float(np.sqrt(np.mean((valid_region - z_true) ** 2)))

    assert abs(mean_error) < 0.02, f"Mean error {mean_error} is too high (expected < 0.02)"
    assert rmse < 0.05, f"RMSE {rmse} is too high (expected < 0.05)"


def test_slanted_plane_reconstruction():
    """Reconstruct a linearly slanted focus plane z(x, y) = 2.5 + 0.05 * x."""
    Z, H, W = 10, 64, 64
    _, x = np.mgrid[:H, :W]
    z_map = 2.5 + 0.05 * x

    stack = _generate_focus_stack((Z, H, W), z_map, sigma=1.5)
    _, height_map = best_focus_image(stack, patch_size=11, return_heightmap=True)

    margin = 10
    error = height_map[margin:-margin, margin:-margin] - z_map[margin:-margin, margin:-margin]
    rmse = float(np.sqrt(np.mean(error**2)))

    assert rmse < 0.20, f"Slanted plane RMSE {rmse} is too high (expected < 0.20)"


def test_identity_reproduction_at_boundaries():
    """A uniform constant stack reproduces identically everywhere with no boundary signal loss."""
    val = 100.0
    stack = np.full((3, 64, 64), val, dtype=np.float32)
    result = best_focus_image(stack)

    assert np.isclose(float(np.mean(result)), val, atol=0.5)
    assert np.allclose(result, val, atol=1e-3), (
        f"Identity reproduction failed! Min value is {result.min():.4f} (expected {val})."
    )


def test_partition_of_unity_via_linear_ramp():
    """Blending weights sum to 1 everywhere so a stack of identical linear ramps is preserved."""
    H, W = 64, 64
    x = np.linspace(0, 100, W)
    y = np.linspace(0, 100, H)
    xv, yv = np.meshgrid(x, y)
    ramp = (xv + yv).astype(np.float32)

    result = best_focus_image(np.array([ramp, ramp, ramp]))
    max_diff = float(np.max(np.abs(result - ramp)))

    assert max_diff < 0.1, f"Ramp reconstruction error {max_diff:.4f} violates partition of unity."


def test_interpolation_quality():
    """Cubic Z-interpolation recovers subpixel peak intensity above linear-node attenuation."""
    nz, ny, nx = 6, 64, 64
    stack = np.zeros((nz, ny, nx), dtype=np.float32)

    true_peak_z = 2.5
    sigma_z = 1.0
    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny // 2, nx // 2
    xy_profile = np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * 5.0**2))

    for z in range(nz):
        stack[z] = xy_profile * np.exp(-((z - true_peak_z) ** 2) / (2 * sigma_z**2))

    result = best_focus_image(stack, patch_size=32)
    reconstructed_peak = float(result[cy, cx])

    # Value at adjacent integer nodes (z=2, 3) is exp(-0.5^2 / 2) ~= 0.882; cubic should reach > 0.94
    assert reconstructed_peak > 0.94, (
        f"Reconstructed peak {reconstructed_peak:.4f} is too low (expected > 0.94)."
    )


def test_best_focus_checkerboard_reconstruction():
    """Identify and reconstruct alternating sharp/blurred quadrants across a 2-slice checkerboard."""
    H, W = 512, 512
    patch_size = 64
    rng = np.random.default_rng(42)

    sharp_texture = rng.uniform(0, 100, (H, W)).astype(np.float32)
    blurred_texture = gaussian_filter(sharp_texture, sigma=5.0)

    slice0 = np.zeros((H, W), dtype=np.float32)
    slice1 = np.zeros((H, W), dtype=np.float32)
    mid_y, mid_x = H // 2, W // 2

    slice0[:mid_y, :mid_x] = sharp_texture[:mid_y, :mid_x]
    slice0[mid_y:, mid_x:] = sharp_texture[mid_y:, mid_x:]
    slice0[:mid_y, mid_x:] = blurred_texture[:mid_y, mid_x:]
    slice0[mid_y:, :mid_x] = blurred_texture[mid_y:, :mid_x]

    slice1[:mid_y, :mid_x] = blurred_texture[:mid_y, :mid_x]
    slice1[mid_y:, mid_x:] = blurred_texture[mid_y:, mid_x:]
    slice1[:mid_y, mid_x:] = sharp_texture[:mid_y, mid_x:]
    slice1[mid_y:, :mid_x] = sharp_texture[mid_y:, :mid_x]

    result_img, height_map = best_focus_image(
        np.array([slice0, slice1]), patch_size=patch_size, return_heightmap=True
    )

    margin = patch_size
    tl_region = height_map[margin : mid_y - margin, margin : mid_x - margin]
    tr_region = height_map[margin : mid_y - margin, mid_x + margin : W - margin]

    assert tl_region.mean() < 0.1, "Top-Left quadrant should select slice 0."
    assert tr_region.mean() > 0.9, "Top-Right quadrant should select slice 1."

    mask = np.ones((H, W), dtype=bool)
    mask[mid_y - patch_size : mid_y + patch_size, :] = False
    mask[:, mid_x - patch_size : mid_x + patch_size] = False

    mse_perfect = float(np.mean((result_img[mask] - sharp_texture[mask]) ** 2))
    mse_blurred = float(np.mean((result_img[mask] - blurred_texture[mask]) ** 2))
    assert mse_perfect < 0.05 * mse_blurred

    std_input = float(np.std(sharp_texture[mask]))
    std_output = float(np.std(result_img[mask]))
    assert std_output > 0.95 * std_input


def test_feature_preservation():
    """Preserve a moderate-sized elevated square feature (Z=7 on Z=2 background) through median smoothing."""
    stack, truth_map = _generate_feature_stack(shape=(10, 200, 200), feature_size=60)
    _, height_map = best_focus_image(stack, return_heightmap=True)

    center_depth = float(np.mean(height_map[90:110, 90:110]))
    assert center_depth > 5.0, f"Feature erased: center depth {center_depth:.2f} too close to background (2)."

    mse = float(np.mean((height_map - truth_map) ** 2))
    assert mse < 4.0


def test_synthetic_focal_stack_reconstruction():
    """Recover depth map (MAE < 1.5 slices) and outperform MIP/Mean in PSNR and SSIM on a smooth 3D surface."""
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    rng = np.random.default_rng(42)
    H, W = 256, 256
    y, x = np.mgrid[0:H, 0:W]
    base_texture = np.zeros((H, W), dtype=np.float32)
    for _ in range(60):
        cy, cx = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
        rad = rng.uniform(5, 15)
        base_texture += np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * rad**2))

    speckle = rng.uniform(0.2, 1.0, (H, W)).astype(np.float32)
    base_texture = base_texture * speckle
    base_texture = (base_texture - base_texture.min()) / (base_texture.max() - base_texture.min() + 1e-8)

    focal_stack, z_gt = _generate_synthetic_focal_stack(base_texture, num_slices=25, seed=43)

    patch_size = 32
    reconstructed_img, estimated_heightmap = best_focus_image(
        focal_stack, patch_size=patch_size, return_heightmap=True
    )

    mip_baseline = np.max(focal_stack, axis=0)
    mean_baseline = np.mean(focal_stack, axis=0)
    valid_crop = np.s_[patch_size:-patch_size, patch_size:-patch_size]

    mae_z = float(np.mean(np.abs(estimated_heightmap[valid_crop] - z_gt[valid_crop])))
    psnr_edf = psnr(base_texture[valid_crop], reconstructed_img[valid_crop], data_range=1.0)
    ssim_edf = ssim(base_texture[valid_crop], reconstructed_img[valid_crop], data_range=1.0)
    psnr_mip = psnr(base_texture[valid_crop], mip_baseline[valid_crop], data_range=1.0)
    ssim_mip = ssim(base_texture[valid_crop], mip_baseline[valid_crop], data_range=1.0)
    psnr_mean = psnr(base_texture[valid_crop], mean_baseline[valid_crop], data_range=1.0)
    ssim_mean = ssim(base_texture[valid_crop], mean_baseline[valid_crop], data_range=1.0)

    assert mae_z < 1.5, f"MAE Z error {mae_z:.4f} exceeds 1.5 slices"
    assert psnr_edf > psnr_mip
    assert psnr_edf > psnr_mean
    assert ssim_edf > ssim_mip
    assert ssim_edf > ssim_mean


def test_best_focus_image_accepts_pathlike(tmp_path):
    """Accept an os.PathLike TIFF path and match array input."""
    import tifffile

    stack = (_textured_stack((5, 40, 40)) * 1000).astype(np.uint16)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, stack)
    np.testing.assert_allclose(best_focus_image(path), best_focus_image(stack))


def test_best_focus_image_accepts_str_path(tmp_path):
    """Accept a string file path and match array input."""
    import tifffile

    stack = (_textured_stack((5, 40, 40), seed=3) * 1000).astype(np.uint16)
    path = tmp_path / "stack_str.tif"
    tifffile.imwrite(path, stack)
    np.testing.assert_allclose(best_focus_image(str(path)), best_focus_image(stack))


@pytest.mark.parametrize("shape", [(5, 8, 8), (5, 25, 25), (5, 40, 40)])
def test_best_focus_image_small_images_have_no_seam_holes(shape):
    """Default patch sizes of 1-4 px must not leave zero-weight seams or crash."""
    stack = _textured_stack(shape) + 1.0  # every input value is >= 1
    out = best_focus_image(stack)
    assert out.shape == shape[1:]
    assert out.min() >= 1.0 - 1e-5  # a zero-weight seam gives 0


def test_best_focus_image_keeps_negative_values():
    """Blending must commute with an intensity offset (no clamp at 0)."""
    rng = np.random.default_rng(1)
    stack = gaussian_filter(rng.normal(size=(7, 64, 64)), (0, 1, 1)).astype(np.float32)
    direct = best_focus_image(stack, patch_size=32)
    shifted = best_focus_image(stack + 10.0, patch_size=32) - 10.0
    assert (direct < 0).mean() > 0.3
    np.testing.assert_allclose(direct, shifted, atol=1e-3)


def test_best_focus_heightmap_stays_in_stack():
    """Linear extrapolation beyond outermost patch centres must stay clipped within [0, Z-1]."""
    Z, H, W = 7, 96, 96
    rng = np.random.default_rng(2)
    texture = rng.random((H, W))
    x = np.arange(W)
    z_map = np.clip(0.3 + (x - 12) / 16, 0, Z - 1)[None, :].repeat(H, 0)
    zz = np.arange(Z)[:, None, None]
    stack = (texture[None] * np.exp(-((zz - z_map[None]) ** 2) / 0.5)).astype(np.float32)
    _, height = best_focus_image(stack, patch_size=24, return_heightmap=True)
    assert height.min() >= 0.0 and height.max() <= Z - 1


def test_apply_median_filter_spike_constant_and_reflect_borders():
    """apply_median_filter removes interior/edge spikes, preserves constants, and uses reflect borders."""
    # Constant preservation (no border drop)
    const = np.full((7, 7), 3.5, dtype=np.float32)
    np.testing.assert_array_equal(apply_median_filter(const), const)

    # Interior and non-corner edge isolated spikes are removed by the disk(1) cross footprint
    with_spikes = np.full((9, 9), 5.0, dtype=np.float32)
    with_spikes[4, 4] = 50.0
    with_spikes[0, 4] = 50.0
    filtered = apply_median_filter(with_spikes)
    np.testing.assert_array_equal(filtered, np.full((9, 9), 5.0, dtype=np.float32))

    # Reflect border behavior: at corner (0, 0), both reflected out-of-bounds cardinal neighbors
    # mirror (0, 0) itself, so 3 of the 5 disk(1) samples equal the corner value
    corner_spike = np.full((7, 7), 2.0, dtype=np.float32)
    corner_spike[0, 0] = 50.0
    filtered_corner = apply_median_filter(corner_spike)
    assert filtered_corner[0, 0] == 50.0
    assert filtered_corner[0, 1] == 2.0
    assert filtered_corner[1, 0] == 2.0


def test_best_focus_image_single_slice_identity():
    """A Z=1 stack returns the single slice unchanged and an all-zero height map."""
    rng = np.random.default_rng(10)
    single = rng.uniform(5.0, 20.0, size=(1, 32, 32)).astype(np.float32)

    fused, height_map = best_focus_image(single, return_heightmap=True)

    assert fused.shape == (32, 32)
    assert height_map.shape == (32, 32)
    np.testing.assert_allclose(fused, single[0], atol=1e-5)
    np.testing.assert_array_equal(height_map, np.zeros((32, 32), dtype=np.float32))


@pytest.mark.parametrize("bad_patch_size", [0, -1, -10])
def test_best_focus_image_rejects_invalid_patch_size(bad_patch_size):
    """patch_size < 1 raises a ValueError."""
    stack = _textured_stack((3, 16, 16))
    with pytest.raises(ValueError, match="patch_size must be >= 1"):
        best_focus_image(stack, patch_size=bad_patch_size)


@pytest.mark.parametrize("patch_size", [1, 2, 32])
def test_best_focus_image_patch_size_edge_cases(patch_size):
    """patch_size=1, patch_size=2 (zero overlap), and single-patch (>= max(H, W)) preserve identity."""
    rng = np.random.default_rng(12)
    plane = rng.uniform(1.0, 10.0, size=(20, 20)).astype(np.float32)
    stack = np.stack([plane, plane, plane], axis=0)

    out = best_focus_image(stack, patch_size=patch_size)

    assert out.shape == (20, 20)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, plane, atol=1e-4)


@pytest.mark.parametrize("shape", [(16, 16), (2, 3, 16, 16)])
def test_best_focus_image_rejects_non_3d_input(shape):
    """Non-3D inputs raise a ValueError."""
    arr = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match="Image not 3D"):
        best_focus_image(arr)


@pytest.mark.parametrize("patch_size", [None, 11])
def test_best_focus_image_non_square_prime_shape(patch_size):
    """Non-square prime lateral dimensions (5, 37, 83) reconstruct cleanly without seam artifacts."""
    Z, H, W = 5, 37, 83
    stack = _generate_focus_stack((Z, H, W), z_map=2.0, sigma=1.2, seed=99)

    fused, height_map = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

    assert fused.shape == (H, W)
    assert height_map.shape == (H, W)
    assert np.all(np.isfinite(fused))
    assert height_map.min() >= 0.0 and height_map.max() <= Z - 1

    # Also verify exact identity reproduction on the same non-square prime shape
    plane = np.random.default_rng(100).uniform(2.0, 12.0, size=(H, W)).astype(np.float32)
    ident = best_focus_image(np.repeat(plane[None], Z, axis=0), patch_size=patch_size)
    np.testing.assert_allclose(ident, plane, atol=1e-4)


def test_best_focus_default_patch_is_32_px_capped_at_image_size():
    """D24 / R7-08: the default patch size is a fixed 32 px, capped at min(H, W).

    Strategy: compare the default call with explicit patch sizes on a 300x260 image
    (the old default was min(H, W) // 10 = 26) and on a 20x24 image (cap 20).
    Expected outcome: identical outputs to patch_size=32 and patch_size=20.
    """
    big = _textured_stack((6, 300, 260), seed=5)
    np.testing.assert_array_equal(best_focus_image(big), best_focus_image(big, patch_size=32))
    small = _textured_stack((6, 20, 24), seed=6)
    np.testing.assert_array_equal(best_focus_image(small), best_focus_image(small, patch_size=20))


def test_best_focus_crop_and_full_image_agree():
    """D24 / R7-08: the default focus map does not depend on the field of view.

    Strategy: a 12 x 512 x 512 stack whose focus height varies sinusoidally
    (period 200 px); compare the height map of a 160 x 160 top-left crop with the
    same region of the full image (the patch grids coincide at the origin). The old
    default used 51 px patches on the full image and 16 px patches on the crop.
    Expected outcome: mean |h_full - h_crop| over the crop's interior < 0.1 slice,
    and both recover the true heights within 0.35 slice on average.
    """
    Z, H, W = 12, 512, 512
    yy, xx = np.mgrid[:H, :W]
    z_map = 5.5 + 3.0 * np.sin(2 * np.pi * xx / 200.0) * np.cos(2 * np.pi * yy / 260.0)
    stack = _generate_focus_stack((Z, H, W), z_map, sigma=1.5, seed=1).astype(np.float32)
    _, h_full = best_focus_image(stack, return_heightmap=True)
    _, h_crop = best_focus_image(stack[:, :160, :160], return_heightmap=True)
    inner = (slice(16, 144), slice(16, 144))
    assert np.mean(np.abs(h_full[:160, :160][inner] - h_crop[inner])) < 0.1
    assert np.mean(np.abs(h_crop[inner] - z_map[:160, :160][inner])) < 0.35
    assert np.mean(np.abs(h_full[:160, :160][inner] - z_map[:160, :160][inner])) < 0.35


def test_best_focus_test_parameter_is_deprecated():
    """R7-15 / D08: the unused `test=` parameter warns (FutureWarning, removed in 0.1.0)."""
    stack = _textured_stack((4, 32, 32), seed=7)
    with pytest.warns(FutureWarning, match=r"test.*0\.1\.0"):
        out = best_focus_image(stack, test=True)
    np.testing.assert_array_equal(out, best_focus_image(stack))


def test_best_focus_output_is_float32():
    """R7-14: the fused image (and height map) are float32 whatever the input dtype (documented)."""
    stack = (_textured_stack((4, 32, 32), seed=8) * 1000).astype(np.uint16)
    fused, height = best_focus_image(stack, return_heightmap=True)
    assert fused.dtype == np.float32 and height.dtype == np.float32
