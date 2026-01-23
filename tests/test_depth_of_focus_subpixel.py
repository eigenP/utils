
import pytest
import numpy as np
from eigenp_utils.extended_depth_of_focus import best_focus_image

def generate_slanted_plane_stack(shape=(20, 100, 100), slope=0.1, texture_freq=0.5):
    """
    Generates a Z-stack where the optimal focus is a slanted plane.
    z_opt(y, x) = slope * x + Z/2
    Texture is high frequency to allow focus detection.
    """
    Z, H, W = shape
    stack = np.zeros(shape, dtype=np.float32)

    # Grid
    y = np.arange(H)
    x = np.arange(W)
    xx, yy = np.meshgrid(x, y)

    # Optimal Z plane
    z_opt = (slope * (xx - W/2)) + Z/2

    # Generate stack
    for z in range(Z):
        # Distance from optimal focus
        dist = np.abs(z - z_opt)

        # Blur kernel size increases with distance
        # Focus profile (Gaussian in Z)
        sigma_z = 2.0
        focus_weight = np.exp(- (dist**2) / (2 * sigma_z**2))

        # Texture: checkerboard/sinusoidal
        texture = np.sin(texture_freq * xx) * np.sin(texture_freq * yy)

        stack[z] = texture * focus_weight + 0.05 * np.random.normal(size=(H, W))

    return stack, z_opt

def test_depth_of_focus_subpixel_accuracy():
    """
    Verifies that best_focus_image returns a sub-pixel precise height map
    and reduces quantization error on a slanted plane.
    """
    Z, H, W = 30, 64, 64
    slope = 0.2

    # 1. Generate Synthetic Data
    stack, z_opt = generate_slanted_plane_stack(shape=(Z, H, W), slope=slope, texture_freq=2.0)

    # Normalize
    stack -= stack.min()
    stack /= stack.max()

    # 2. Run EDoF
    # Small patch size to evaluate height map resolution
    patch_size = 16
    img_recon, height_map = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

    # 3. Analyze Height Map
    # Crop borders to avoid boundary artifacts
    border = patch_size // 2
    hm_crop = height_map[border:-border, border:-border]
    gt_crop = z_opt[border:-border, border:-border]

    # RMSE Calculation
    rmse = np.sqrt(np.mean((hm_crop - gt_crop)**2))
    print(f"RMSE: {rmse:.4f}")

    # Check for quantization
    unique_vals = np.unique(hm_crop)
    # Check if values are floats with fractional parts
    # We check if any value has a significant fractional part
    has_fractional = np.any(np.abs(unique_vals - np.round(unique_vals)) > 1e-3)

    # 4. Assertions

    # Ideally, we want RMSE < 0.5 (integer quantization error ~0.5/sqrt(12)?? No, step is 1, error is uniform [-0.5, 0.5], std is 1/sqrt(12) = 0.29. But here we have patches and noise)
    # The integer baseline was RMSE ~ 2.15 (likely due to patch size averaging or noise).
    # We assert improvement over integer baseline.

    # Note: If the code is not yet updated, this test might fail on 'has_fractional' or RMSE.
    # For now, I will assert that it IS better than integer baseline if it passes,
    # but I expect this test to FAIL currently.

    if not has_fractional:
        pytest.fail(f"Height map appears quantized (Integer values only). RMSE: {rmse:.4f}")

    assert rmse < 0.5, f"RMSE {rmse:.4f} is too high (Expected < 0.5 for subpixel)"

if __name__ == "__main__":
    test_depth_of_focus_subpixel_accuracy()
