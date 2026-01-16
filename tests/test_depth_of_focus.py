
import numpy as np
import scipy.ndimage as ndi
import pytest
from eigenp_utils.extended_depth_of_focus import best_focus_image

def test_best_focus_checkerboard_reconstruction():
    """
    Testr Verification: Checkerboard Depth Field Reconstruction

    This test verifies that the `best_focus_image` algorithm correctly identifies and reconstructs
    regions of focus from a 3D stack.

    The Setup:
    - We create a "perfect" sharp texture (white noise).
    - We create a "blurred" version (Gaussian blur).
    - We construct a 2-slice stack where the focus plane follows a 2x2 checkerboard pattern.
      - Quadrant 1 (Top-Left):  Slice 0 is Sharp
      - Quadrant 2 (Top-Right): Slice 1 is Sharp
      - Quadrant 3 (Bot-Left):  Slice 1 is Sharp
      - Quadrant 4 (Bot-Right): Slice 0 is Sharp

    The Invariants:
    1. Focus Map Accuracy: The algorithm must recover a height map (index map) that matches
       the checkerboard pattern (0s and 1s in correct quadrants), ignoring boundary effects.
    2. Signal Preservation: The output image must closely resemble the sharp texture (low MSE)
       and be significantly different from the blurred texture.
    3. Contrast Conservation: The standard deviation of the output should match the input signal,
       proving that the blending didn't wash out features.
    """

    # 1. Setup Parameters
    H, W = 512, 512
    patch_size = 64
    np.random.seed(42)

    # 2. Generate Textures
    # Signal scale 100.0 to make errors obvious
    # Using random noise ensures high Laplacian energy
    sharp_texture = np.random.uniform(0, 100, (H, W)).astype(np.float32)
    blurred_texture = ndi.gaussian_filter(sharp_texture, sigma=5.0)

    # 3. Construct Checkerboard Stack
    # Slices
    slice0 = np.zeros((H, W), dtype=np.float32)
    slice1 = np.zeros((H, W), dtype=np.float32)

    mid_y, mid_x = H // 2, W // 2

    # Slice 0: Sharp in TL, BR. Blurred in TR, BL.
    slice0[:mid_y, :mid_x] = sharp_texture[:mid_y, :mid_x] # TL
    slice0[mid_y:, mid_x:] = sharp_texture[mid_y:, mid_x:] # BR
    slice0[:mid_y, mid_x:] = blurred_texture[:mid_y, mid_x:] # TR
    slice0[mid_y:, :mid_x] = blurred_texture[mid_y:, :mid_x] # BL

    # Slice 1: Blurred in TL, BR. Sharp in TR, BL.
    slice1[:mid_y, :mid_x] = blurred_texture[:mid_y, :mid_x] # TL
    slice1[mid_y:, mid_x:] = blurred_texture[mid_y:, mid_x:] # BR
    slice1[:mid_y, mid_x:] = sharp_texture[:mid_y, mid_x:] # TR
    slice1[mid_y:, :mid_x] = sharp_texture[mid_y:, :mid_x] # BL

    stack = np.array([slice0, slice1])

    # 4. Run Algorithm
    # Note: We expect the algorithm to pick the sharpest texture in each patch.
    result_img, height_map = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

    # 5. Verify Focus Map (Indices)
    # We sample the center of each quadrant to avoid patch boundary artifacts
    margin = patch_size  # Stay away from edges and center cross

    # Quadrant Centers
    tl_idx = height_map[mid_y//2, mid_x//2]
    tr_idx = height_map[mid_y//2, mid_x + mid_x//2]
    bl_idx = height_map[mid_y + mid_y//2, mid_x//2]
    br_idx = height_map[mid_y + mid_y//2, mid_x + mid_x//2]

    # Also check average over a region in the quadrant to be robust
    tl_region = height_map[margin : mid_y-margin, margin : mid_x-margin]
    tr_region = height_map[margin : mid_y-margin, mid_x+margin : W-margin]

    print(f"TL Mean Index: {tl_region.mean():.4f} (Expected 0)")
    print(f"TR Mean Index: {tr_region.mean():.4f} (Expected 1)")

    # Assertions
    assert tl_region.mean() < 0.1, "Top-Left quadrant should be mostly index 0 (Sharp in Slice 0)"
    assert tr_region.mean() > 0.9, "Top-Right quadrant should be mostly index 1 (Sharp in Slice 1)"

    # 6. Verify Reconstruction Quality (MSE)
    # Compare output to the "Perfect Composite" (which is just sharp_texture everywhere)
    # We exclude the boundaries (cross in the middle) from the stats calculation
    mask = np.ones((H, W), dtype=bool)
    # Mask out the center cross seam (width approx 2 patches)
    mask[mid_y-patch_size:mid_y+patch_size, :] = False
    mask[:, mid_x-patch_size:mid_x+patch_size] = False

    mse_perfect = np.mean((result_img[mask] - sharp_texture[mask])**2)
    mse_blurred = np.mean((result_img[mask] - blurred_texture[mask])**2)

    print(f"MSE vs Perfect: {mse_perfect:.4f}")
    print(f"MSE vs Blurred: {mse_blurred:.4f}")

    # The reconstruction should be MUCH closer to perfect than to blurred
    # Typically < 1% of the blurred error
    assert mse_perfect < 0.05 * mse_blurred, \
        f"Reconstruction failed to recover sharp texture. MSE_perfect={mse_perfect}, MSE_blurred={mse_blurred}"

    # 7. Verify Contrast Preservation (Standard Deviation)
    # Blending can sometimes lower contrast. We want to ensure we kept the signal.
    std_input = np.std(sharp_texture[mask])
    std_output = np.std(result_img[mask])

    print(f"Input Std: {std_input:.4f}, Output Std: {std_output:.4f}")

    # Allow small drop due to potential blending/interpolation, but should be close
    assert std_output > 0.95 * std_input, \
        f"Output lost significant contrast. InStd={std_input}, OutStd={std_output}"
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from eigenp_utils.extended_depth_of_focus import best_focus_image

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
    A 7x7 median filter (disk(3)) would erase this feature (16 pixels < 25 threshold).
    A 3x3 median filter (disk(1)) should preserve it.
    """
    stack, truth_map = generate_feature_stack(shape=(10, 200, 200), feature_size=60)

    # Run reconstruction
    fused, height_map = best_focus_image(stack, return_heightmap=True)

    # Check if the feature is preserved.
    # We check the max depth value in the center region.
    mid_h, mid_w = 100, 100
    # Average depth in the feature center
    center_depth = np.mean(height_map[mid_h-10:mid_h+10, mid_w-10:mid_w+10])

    print(f"Center Depth: {center_depth:.2f} (Expected ~7)")

    # If erased, center_depth would be close to 2.
    # If preserved, center_depth should be close to 7.

    assert center_depth > 5.0, \
        f"Feature erased! Center depth {center_depth:.2f} is too close to background (2). Over-smoothing detected."

    # Also check MSE globally
    mse = np.mean((height_map - truth_map)**2)
    print(f"MSE: {mse:.4f}")
    # MSE should be reasonable
    assert mse < 4.0

def test_focal_plane_indices():
    """
    Verifies that the returned height map contains valid indices within [0, Z-1].
    """
    stack, _ = generate_feature_stack(shape=(5, 100, 100))
    fused, height_map = best_focus_image(stack, return_heightmap=True)

    assert height_map.min() >= 0
    assert height_map.max() < 5
