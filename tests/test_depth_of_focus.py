
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
