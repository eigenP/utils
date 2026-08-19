from scipy.ndimage import gaussian_filter
import numpy as np
import pytest
import scipy.ndimage as ndi
import unittest

from eigenp_utils.extended_depth_of_focus import best_focus_image


# =========================================
# Source: test_depth_of_focus_subpixel.py
# =========================================


class TestDepthOfFocusSubpixel(unittest.TestCase):
    """
    Testr 🔎 Verification: Subpixel Depth of Focus

    Validates that the extended depth of focus algorithm correctly identifies
    the subpixel peak of a focus stack using log-parabolic interpolation.
    """

    def generate_focus_stack(self, shape, z_map, sigma=1.0, seed=42):
        """
        Generates a synthetic focus stack where the "focus score" (high frequency content)
        follows a Gaussian profile centered at z_map.

        The image intensity is modeled as:
        I(x,y,z) = Noise(x,y) * exp( - (z - z_map(x,y))^2 / (2 * sigma^2) )

        This ensures that the Laplacian energy (and thus the focus metric) also follows
        a Gaussian profile, allowing us to test the peak finding logic.
        """
        Z, H, W = shape
        rng = np.random.default_rng(seed)

        # Generate high-frequency texture (white noise)
        # Using uniform noise in [0, 1]
        texture = rng.uniform(0, 1, (H, W))

        # Precompute grid for Z
        z_indices = np.arange(Z).reshape(-1, 1, 1)

        # Compute Gaussian weights
        # z_map can be (H, W) or scalar
        # (Z, 1, 1) - (H, W) broadcasts to (Z, H, W)
        delta_z = z_indices - z_map
        weights = np.exp(- (delta_z**2) / (2 * sigma**2))

        # Modulate texture
        # (Z, H, W) = (H, W) * (Z, H, W)
        stack = texture * weights

        # Normalize to 0-1 (optional but good practice)
        stack = (stack - stack.min()) / (stack.max() - stack.min() + 1e-8)

        return stack

    def test_subpixel_focus_recovery(self):
        """
        Verifies that best_focus_image recovers a constant subpixel depth plane
        with high precision using log-parabolic interpolation.
        """
        Z, H, W = 10, 64, 64
        z_true = 4.3  # Subpixel peak
        sigma = 1.5

        stack = self.generate_focus_stack((Z, H, W), z_true, sigma=sigma)

        # Run best_focus_image
        # return_heightmap=True returns (focused_image, height_map)
        _, height_map = best_focus_image(stack, patch_size=11, return_heightmap=True)

        # Calculate statistics
        # Ignore boundaries where padding/validity might be an issue (10 pixels)
        # The algorithm uses padding, but let's be safe.
        valid_region = height_map[10:-10, 10:-10]

        mean_error = np.mean(valid_region) - z_true
        rmse = np.sqrt(np.mean((valid_region - z_true)**2))

        print(f"True Z: {z_true}")
        print(f"Recovered Mean Z: {np.mean(valid_region):.4f}")
        print(f"Mean Error: {mean_error:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # The log-parabolic interpolation should be extremely accurate for Gaussian inputs.
        # We expect error < 0.05 pixels (conservative).
        # Ideally it should be < 0.01.
        self.assertLess(abs(mean_error), 0.02, f"Mean error {mean_error} is too high (expected < 0.02)")
        self.assertLess(rmse, 0.05, f"RMSE {rmse} is too high (expected < 0.05)")

    def test_slanted_plane_reconstruction(self):
        """
        Verifies that best_focus_image correctly reconstructs a slanted plane.
        z(x, y) = 2.0 + slope * x
        """
        Z, H, W = 10, 64, 64
        slope = 0.05
        z_start = 2.5

        # Create z_map (H, W)
        y, x = np.mgrid[:H, :W]
        z_map = z_start + slope * x

        stack = self.generate_focus_stack((Z, H, W), z_map, sigma=1.5)

        _, height_map = best_focus_image(stack, patch_size=11, return_heightmap=True)

        # Check valid region
        margin = 10
        valid_z_map = z_map[margin:-margin, margin:-margin]
        valid_height_map = height_map[margin:-margin, margin:-margin]

        error = valid_height_map - valid_z_map
        rmse = np.sqrt(np.mean(error**2))

        print(f"Slanted Plane RMSE: {rmse:.4f}")

        # Allow slightly higher error for spatially varying map due to patch averaging effects
        # but regular grid interpolation should handle it well.
        self.assertLess(rmse, 0.20, f"Slanted plane RMSE {rmse} is too high (expected < 0.20)")


# =========================================
# Source: test_focus_properties.py
# =========================================

class TestFocusProperties(unittest.TestCase):
    """
    Testr 🔎 Verification: Focus Stacking Invariants

    This test verifies fundamental mathematical properties of the Extended Depth of Focus (EDoF)
    algorithm, specifically 'Identity Reproduction' and 'Partition of Unity'.

    Property 1: Identity Reproduction
    If the input stack consists of identical slices (S(z) = Image I), the output
    must be exactly I, regardless of the focus metric or patch blending logic.

    Property 2: Boundary Integrity (Partition of Unity)
    The blending weights used to fuse patches must sum to 1.0 everywhere.
    A common failure mode is signal loss at image boundaries due to
    unconditional tapering (windowing) of edge patches.
    """

    def test_identity_reproduction_at_boundaries(self):
        """
        Verifies that a uniform white stack results in a uniform white image,
        ensuring no signal is lost at the boundaries due to weighting artifacts.
        """
        # 1. Setup
        # Create a stack of 3 slices, 64x64, all 1.0 (float)
        # We use a value of 100.0 to make relative errors obvious/easy to reason about.
        val = 100.0
        shape = (3, 64, 64)
        stack = np.full(shape, val, dtype=np.float32)

        # 2. Run Algorithm
        # We use default patch size (which would be 64//10 = 6).
        # Overlap will be 6//3 = 2.
        # This ensures we have multiple patches and boundaries are relevant.
        result = best_focus_image(stack)

        # 3. Verification

        # A) Global Conservation
        # Mean should be 100.0
        mean_val = np.mean(result)
        print(f"Mean Value: {mean_val:.4f} (Expected {val})")
        self.assertAlmostEqual(mean_val, val, delta=0.5,
            msg=f"Global signal loss detected! Mean is {mean_val} instead of {val}")

        # B) Boundary Integrity (The Critical Check)
        # We check the minimum value in the image.
        # If boundary tapering is buggy, pixels at the edge will be 0 or < 100.
        min_val = np.min(result)
        print(f"Min Value: {min_val:.4f} (Expected {val})")

        # Visualizing the artifact if present
        if min_val < val * 0.9:
            print("Low values detected! Top-Left corner slice:")
            print(result[:5, :5])

        # We allow tiny floating point error (e.g. 1e-5), but not massive signal loss.
        self.assertTrue(np.allclose(result, val, atol=1e-3),
            f"Identity reproduction failed! Min value is {min_val} (should be {val}). "
            "This likely indicates incorrect weight tapering at image boundaries.")

    def test_partition_of_unity_via_linear_ramp(self):
        """
        Verifies that a spatially varying signal (linear ramp) is perfectly reconstructed.
        If weights don't sum to 1, the ramp will be distorted.
        """
        # Stack where every slice is the same Linear Ramp
        H, W = 64, 64
        x = np.linspace(0, 100, W)
        y = np.linspace(0, 100, H)
        xv, yv = np.meshgrid(x, y)
        ramp = (xv + yv).astype(np.float32) # Range 0 to 200

        stack = np.array([ramp, ramp, ramp])

        result = best_focus_image(stack)

        # Compare
        diff = np.abs(result - ramp)
        max_diff = np.max(diff)

        print(f"Max Ramp Reconstruction Error: {max_diff:.4f}")

        self.assertLess(max_diff, 0.1,
            "Ramp reconstruction failed. Partition of Unity violated?")


# =========================================
# Source: test_edof_interpolation.py
# =========================================


def test_interpolation_quality():
    """
    Verifies that the reconstruction preserves peak intensity for sub-pixel focus.
    We create a stack where the focus is exactly at z=2.5.
    The intensity profile in Z is Gaussian.
    We check the max intensity of the reconstructed image.
    Linear interpolation should underestimate the peak.
    Cubic interpolation should be closer to the true peak (1.0).
    """

    # 1. Create synthetic stack
    nz, ny, nx = 6, 64, 64
    stack = np.zeros((nz, ny, nx), dtype=np.float32)

    # Focus peak at z = 2.5
    true_peak_z = 2.5
    sigma_z = 1.0

    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny//2, nx//2
    sigma_xy = 5.0
    xy_profile = np.exp(-((y-cy)**2 + (x-cx)**2) / (2 * sigma_xy**2))

    for z in range(nz):
        intensity_factor = np.exp(-(z - true_peak_z)**2 / (2 * sigma_z**2))
        stack[z] = xy_profile * intensity_factor

    result = best_focus_image(stack, patch_size=32)

    reconstructed_peak = result[cy, cx]

    print(f"Reconstructed Peak: {reconstructed_peak:.4f}")

    val_at_node = np.exp(-(0.5)**2 / 2) # 0.882
    print(f"Value at nodes (z=2,3): {val_at_node:.4f}")
    print(f"True Peak: 1.0000")

    threshold = 0.94

    assert reconstructed_peak > threshold, f"Reconstruction {reconstructed_peak:.4f} is too low. Expected > {threshold:.4f}"


# =========================================
# Source: test_depth_of_focus.py
# =========================================


def test_best_focus_checkerboard_reconstruction():
    """
    Testr Verification: Checkerboard Depth Field Reconstruction

    This test verifies that the `best_focus_image` algorithm correctly identifies and reconstructs
    regions of focus from a 3D stack.
    """

    # 1. Setup Parameters
    H, W = 512, 512
    patch_size = 64
    np.random.seed(42)

    # 2. Generate Textures
    sharp_texture = np.random.uniform(0, 100, (H, W)).astype(np.float32)
    blurred_texture = ndi.gaussian_filter(sharp_texture, sigma=5.0)

    # 3. Construct Checkerboard Stack
    slice0 = np.zeros((H, W), dtype=np.float32)
    slice1 = np.zeros((H, W), dtype=np.float32)

    mid_y, mid_x = H // 2, W // 2

    slice0[:mid_y, :mid_x] = sharp_texture[:mid_y, :mid_x] # TL
    slice0[mid_y:, mid_x:] = sharp_texture[mid_y:, mid_x:] # BR
    slice0[:mid_y, mid_x:] = blurred_texture[:mid_y, mid_x:] # TR
    slice0[mid_y:, :mid_x] = blurred_texture[mid_y:, :mid_x] # BL

    slice1[:mid_y, :mid_x] = blurred_texture[:mid_y, :mid_x] # TL
    slice1[mid_y:, mid_x:] = blurred_texture[mid_y:, mid_x:] # BR
    slice1[:mid_y, mid_x:] = sharp_texture[:mid_y, mid_x:] # TR
    slice1[mid_y:, :mid_x] = sharp_texture[mid_y:, :mid_x] # BL

    stack = np.array([slice0, slice1])

    result_img, height_map = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

    margin = patch_size

    tl_region = height_map[margin : mid_y-margin, margin : mid_x-margin]
    tr_region = height_map[margin : mid_y-margin, mid_x+margin : W-margin]

    print(f"TL Mean Index: {tl_region.mean():.4f} (Expected 0)")
    print(f"TR Mean Index: {tr_region.mean():.4f} (Expected 1)")

    assert tl_region.mean() < 0.1, "Top-Left quadrant should be mostly index 0 (Sharp in Slice 0)"
    assert tr_region.mean() > 0.9, "Top-Right quadrant should be mostly index 1 (Sharp in Slice 1)"

    mask = np.ones((H, W), dtype=bool)
    mask[mid_y-patch_size:mid_y+patch_size, :] = False
    mask[:, mid_x-patch_size:mid_x+patch_size] = False

    mse_perfect = np.mean((result_img[mask] - sharp_texture[mask])**2)
    mse_blurred = np.mean((result_img[mask] - blurred_texture[mask])**2)

    print(f"MSE vs Perfect: {mse_perfect:.4f}")
    print(f"MSE vs Blurred: {mse_blurred:.4f}")

    assert mse_perfect < 0.05 * mse_blurred, \
        f"Reconstruction failed to recover sharp texture. MSE_perfect={mse_perfect}, MSE_blurred={mse_blurred}"

    std_input = np.std(sharp_texture[mask])
    std_output = np.std(result_img[mask])

    print(f"Input Std: {std_input:.4f}, Output Std: {std_output:.4f}")

    assert std_output > 0.95 * std_input, \
        f"Output lost significant contrast. InStd={std_input}, OutStd={std_output}"


def generate_feature_stack(shape=(10, 200, 200), feature_size=60):
    """
    Generates a 3D stack with a central square feature at Z=7, background Z=2.
    """
    Z, H, W = shape
    rng = np.random.default_rng(42)
    texture = rng.random((H, W))

    depth_map = np.zeros((H, W), dtype=int) + 2

    mid_h, mid_w = H // 2, W // 2
    half_size = feature_size // 2

    depth_map[mid_h - half_size : mid_h + half_size,
              mid_w - half_size : mid_w + half_size] = 7

    stack = np.zeros(shape, dtype=np.float32)

    for z in range(Z):
        dist_1 = abs(2 - z)
        dist_2 = abs(7 - z)

        sigma1 = dist_1 * 0.5
        sigma2 = dist_2 * 0.5

        layer1 = gaussian_filter(texture, sigma=sigma1) if sigma1 > 0 else texture
        layer2 = gaussian_filter(texture, sigma=sigma2) if sigma2 > 0 else texture

        mask2 = (depth_map == 7)
        mask1 = ~mask2

        stack[z] = layer1 * mask1 + layer2 * mask2

    return stack, depth_map


def test_feature_preservation():
    """
    Verifies that best_focus_image preserves features of moderate size.
    A feature of size 60x60 in a 200x200 image corresponds to approx 4x4 patches (grid 14x14).
    A 3x3 median filter (disk(1)) should preserve it.
    """
    stack, truth_map = generate_feature_stack(shape=(10, 200, 200), feature_size=60)

    fused, height_map = best_focus_image(stack, return_heightmap=True)

    mid_h, mid_w = 100, 100
    center_depth = np.mean(height_map[mid_h-10:mid_h+10, mid_w-10:mid_w+10])

    print(f"Center Depth: {center_depth:.2f} (Expected ~7)")

    assert center_depth > 5.0, \
        f"Feature erased! Center depth {center_depth:.2f} is too close to background (2). Over-smoothing detected."

    mse = np.mean((height_map - truth_map)**2)
    print(f"MSE: {mse:.4f}")
    assert mse < 4.0


def test_focal_plane_indices():
    """
    Verifies that the returned height map contains valid indices within [0, Z-1].
    """
    stack, _ = generate_feature_stack(shape=(5, 100, 100))
    fused, height_map = best_focus_image(stack, return_heightmap=True)

    assert height_map.min() >= 0
    assert height_map.max() < 5


def generate_synthetic_focal_stack(sharp_image, num_slices=25, k_defocus=0.8):
    """
    Generates a synthetic 3D focal stack from a 2D sharp texture using a smooth ground-truth depth surface z_GT(y, x).
    """
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
        level_idx = np.clip((dist / max_sigma * (len(sigma_steps) - 1)).astype(int), 0, len(sigma_steps) - 1)

        slice_z = np.zeros((H, W), dtype=np.float32)
        for idx in np.unique(level_idx):
            mask = level_idx == idx
            slice_z[mask] = blurred_levels[idx][mask]

        noise = np.random.normal(0, 0.005, (H, W)).astype(np.float32)
        focal_stack[z] = np.clip(slice_z + noise, 0.0, 1.0)

    return focal_stack, z_gt


def test_synthetic_focal_stack_reconstruction():
    """
    Verifies that best_focus_image accurately recovers depth map and focused texture
    on a synthetic focal stack with a spatially varying ground-truth depth field.

    Invariants and Metrics asserted:
    1. Height map MAE relative to ground truth depth map z_gt is < 1.5 slices.
    2. Reconstructed image PSNR and SSIM exceed both Maximum Intensity Projection (MIP)
       and Mean baselines.
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim

    np.random.seed(42)
    H, W = 256, 256
    y, x = np.mgrid[0:H, 0:W]
    base_texture = np.zeros((H, W), dtype=np.float32)
    for _ in range(60):
        cy, cx = np.random.uniform(20, H - 20), np.random.uniform(20, W - 20)
        rad = np.random.uniform(5, 15)
        base_texture += np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * rad**2))

    rng = np.random.default_rng(42)
    speckle = rng.uniform(0.2, 1.0, (H, W)).astype(np.float32)
    base_texture = base_texture * speckle
    base_texture = (base_texture - base_texture.min()) / (base_texture.max() - base_texture.min() + 1e-8)

    num_slices = 25
    focal_stack, z_gt = generate_synthetic_focal_stack(base_texture, num_slices=num_slices)

    patch_size = 32
    reconstructed_img, estimated_heightmap = best_focus_image(
        focal_stack, patch_size=patch_size, return_heightmap=True
    )

    mip_baseline = np.max(focal_stack, axis=0)
    mean_baseline = np.mean(focal_stack, axis=0)

    valid_crop = np.s_[patch_size:-patch_size, patch_size:-patch_size]

    z_error = np.abs(estimated_heightmap[valid_crop] - z_gt[valid_crop])
    mae_z = np.mean(z_error)

    psnr_edf = psnr(base_texture[valid_crop], reconstructed_img[valid_crop], data_range=1.0)
    ssim_edf = ssim(base_texture[valid_crop], reconstructed_img[valid_crop], data_range=1.0)

    psnr_mip = psnr(base_texture[valid_crop], mip_baseline[valid_crop], data_range=1.0)
    ssim_mip = ssim(base_texture[valid_crop], mip_baseline[valid_crop], data_range=1.0)

    psnr_mean = psnr(base_texture[valid_crop], mean_baseline[valid_crop], data_range=1.0)
    ssim_mean = ssim(base_texture[valid_crop], mean_baseline[valid_crop], data_range=1.0)

    print(f"Height map MAE: {mae_z:.4f}")
    print(f"EDF PSNR: {psnr_edf:.2f} dB, SSIM: {ssim_edf:.4f}")
    print(f"MIP PSNR: {psnr_mip:.2f} dB, SSIM: {ssim_mip:.4f}")
    print(f"Mean PSNR: {psnr_mean:.2f} dB, SSIM: {ssim_mean:.4f}")

    assert mae_z < 1.5, f"MAE Z error {mae_z:.4f} exceeds 1.5 slices"
    assert psnr_edf > psnr_mip, f"EDF PSNR ({psnr_edf:.2f}) failed to beat MIP ({psnr_mip:.2f})"
    assert psnr_edf > psnr_mean, f"EDF PSNR ({psnr_edf:.2f}) failed to beat Mean ({psnr_mean:.2f})"
    assert ssim_edf > ssim_mip, f"EDF SSIM ({ssim_edf:.4f}) failed to beat MIP ({ssim_mip:.4f})"
    assert ssim_edf > ssim_mean, f"EDF SSIM ({ssim_edf:.4f}) failed to beat Mean ({ssim_mean:.4f})"


if __name__ == "__main__":
    unittest.main()
