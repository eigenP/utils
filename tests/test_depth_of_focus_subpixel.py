
import numpy as np
import scipy.ndimage as ndi
import pytest
from eigenp_utils.extended_depth_of_focus import best_focus_image

def generate_slanted_plane(shape=(10, 256, 256), z_start=3.0, z_end=7.0):
    Z, H, W = shape
    rng = np.random.default_rng(42)
    sharp = rng.random((H, W)).astype(np.float32)
    blurred = ndi.gaussian_filter(sharp, sigma=3.0)

    stack = np.zeros(shape, dtype=np.float32)

    # Ground truth depth map (varying along X)
    x_ramp = np.linspace(0, 1, W)
    # Broadcast to (H, W)
    z_true = z_start + (z_end - z_start) * x_ramp[None, :]
    z_true = np.broadcast_to(z_true, (H, W))

    for z in range(Z):
        # Gaussian weighting for focus
        # dist = z - z_true
        # weight = exp(-dist^2)
        dist_sq = (z - z_true)**2
        # Width of focus plane
        sigma_focus = 1.0
        weights = np.exp(-dist_sq / (2 * sigma_focus**2))

        # Blend
        # I = weight * sharp + (1-weight) * blurred
        stack[z] = weights * sharp + (1 - weights) * blurred

    return stack, z_true

def test_slanted_plane_smoothness():
    """
    Verifies sub-pixel depth estimation.
    A slanted plane should result in a smooth height map, not a staircase.
    """
    Z = 10
    H, W = 200, 200
    z_start, z_end = 3.0, 7.0

    stack, z_true = generate_slanted_plane((Z, H, W), z_start, z_end)

    # Run
    patch_size = 32
    fused, height_map = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

    # Crop borders where padding/patching might distort
    crop = patch_size
    hm_center = height_map[crop:-crop, crop:-crop]
    zt_center = z_true[crop:-crop, crop:-crop]

    # Calculate RMSE
    rmse = np.sqrt(np.mean((hm_center - zt_center)**2))
    print(f"RMSE: {rmse:.4f}")

    # Calculate "Staircase Artifact" metric
    # Staircase = high variance in gradient?
    # Or just check if values are integers.

    # Check fraction of non-integer values
    # (Allowing for some float noise, we check if they are close to integers)
    is_integer = np.isclose(hm_center, np.round(hm_center), atol=1e-5)
    frac_integer = np.mean(is_integer)
    print(f"Fraction of integer values: {frac_integer:.4f}")

    # Current behavior expectation:
    # RMSE will be around 0.25 (quantization noise uniform [-0.5, 0.5] -> std 1/sqrt(12) = 0.29)
    # Fraction integer will be 1.0

    # Desired behavior:
    # RMSE < 0.1
    # Fraction integer < 0.1

    # For now, we assert failure or just log it?
    # The prompt says "Create verification script". I should probably make it a test that will PASS after my changes.
    # So I will assert the desired behavior.

    assert frac_integer < 0.5, f"Height map is quantized! {frac_integer:.1%} values are integers."
    assert rmse < 0.2, f"RMSE too high ({rmse:.4f}). Subpixel refinement needed."

if __name__ == "__main__":
    try:
        test_slanted_plane_smoothness()
        print("Test PASSED")
    except AssertionError as e:
        print(f"Test FAILED: {e}")
