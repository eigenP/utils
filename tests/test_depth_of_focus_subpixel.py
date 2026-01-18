
import numpy as np
import scipy.ndimage as ndi
import pytest
from eigenp_utils.extended_depth_of_focus import best_focus_image

def test_subpixel_focus_accuracy():
    """
    Matth Journal Regression Test: Sub-pixel Focus Estimation

    Verifies that the depth map estimation is not quantized to integer Z-steps
    but uses interpolation to recover the continuous focal plane.

    Setup:
    - 64x64 image, 10 Z-slices.
    - True focal plane is a slant from Z=3.0 to Z=6.0 across the X axis.
    - We generate the stack by blurring a texture based on distance to this plane.
    """
    H, W = 128, 128
    Z = 10
    patch_size = 32

    # 1. Generate Slanted Plane Truth
    # Varies from 3.0 to 6.0 across width
    x = np.linspace(0, 1, W)
    y = np.linspace(0, 1, H)
    xv, yv = np.meshgrid(x, y)

    true_height_map = 3.0 + 3.0 * xv # Linear ramp from 3 to 6

    # 2. Generate Texture Stack
    rng = np.random.default_rng(42)
    texture = rng.uniform(0, 255, (H, W)).astype(np.float32)

    stack = np.zeros((Z, H, W), dtype=np.float32)

    for z in range(Z):
        # Distance from this slice z to the true focal plane
        # dist_map = abs(z - true_height_map)

        # We need to blur spatially.
        # This is tricky to simulate perfectly fast, but we can approximate:
        # For each pixel, the blur radius is proportional to distance.
        # But varying sigma per pixel is slow.
        # Instead, we can just generate layers.
        # Wait, simple way:
        # For each Z, we have a fixed Z. The true focus is at Z_true(x,y).
        # Distance D(x,y) = |z - Z_true(x,y)|.
        # We want to apply blur sigma(x,y) = coeff * D(x,y).
        # Spatially varying blur is hard.

        # Alternative: Just make the signal vary in amplitude?
        # No, extended depth of focus works on sharpness (Laplacian).
        # Laplacian response L ~ 1 / sigma^2 roughly (for Gaussian).
        # If we simulate the *score* directly?
        # No, `best_focus_image` computes the score from the image.

        # Let's do a block-wise approximation or just generating the stack properly.
        # Actually, for a small image (128x128), we can just loop or use a few discrete sigmas.
        # Or, just use the fact that if we have a texture, the sharpest version is at Z_true.
        # If we just blend a sharp texture with a blurred one based on distance?

        # Let's try to simulate the "Gaussian Spot" logic from drift correction but for focus.
        # Just create a stack where for each pixel (x,y), the intensity is maximized at true_height_map(x,y)
        # and falls off like a Gaussian in Z.
        # Pixel(x,y,z) = Texture(x,y) * exp( - (z - true_height_map(x,y))^2 / (2 * width^2) )
        # This modulates *intensity*.
        # `best_focus_image` uses Laplacian Energy (contrast).
        # If we modulate contrast:
        # Stack(x,y,z) = Texture(x,y) * exp(...) + 128
        # Then the Laplacian will be Texture_Laplacian * exp(...).
        # Its square will be max at the same Z.
        # This works perfectly for testing peak finding.

        # Width of the focus curve (depth of field).
        # Let's say sigma_z = 1.5 slices.
        sigma_z = 1.5

        weight = np.exp( - (z - true_height_map)**2 / (2 * sigma_z**2) )

        # We add some background so it's not black
        stack[z] = texture * weight + 50.0

    # 3. Run Algorithm
    # We rely on the returned height map
    fused, height_map = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

    # 4. Analysis

    # Check for quantization
    # If all values are integers (or very close), it's failing the sub-pixel requirement.
    non_integer_fraction = np.mean(np.abs(height_map - np.round(height_map)) > 0.001)

    print(f"Fraction of non-integer heights: {non_integer_fraction:.2f}")

    # Current behavior (integer argmax) should give 0.0
    # Desired behavior (sub-pixel) should give > 0.0 (likely ~1.0 since it's a ramp)

    # Also check Error
    # Ignore boundary patches where padding/artifacts might occur
    margin = 10
    mask = np.ones_like(height_map, dtype=bool)
    mask[:margin, :] = False
    mask[-margin:, :] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False

    rmse = np.sqrt(np.mean((height_map[mask] - true_height_map[mask])**2))
    print(f"RMSE vs Truth: {rmse:.4f}")

    # For integer quantization of a ramp, error is uniform [-0.5, 0.5], variance 1/12 approx 0.083, RMSE ~0.29.
    # With subpixel, it should be much better.

    # Assertion 1: Must be sub-pixel (fails if implementation is just argmax)
    assert non_integer_fraction > 0.5, "Height map contains mostly integers! Sub-pixel refinement missing."

    # Assertion 2: Precision
    # With sigma=1.5 and correct parabolic fit, error should be very low.
    # We'll set a loose bound for now that is tighter than quantization noise.
    # RMSE < 0.1 (Quantization is ~0.29)
    assert rmse < 0.15, f"RMSE {rmse:.4f} is too high (Quantization limit ~0.29)."

if __name__ == "__main__":
    # Manually run if executed as script
    test_subpixel_focus_accuracy()
