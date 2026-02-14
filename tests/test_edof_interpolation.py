
import numpy as np
import pytest
from eigenp_utils.extended_depth_of_focus import best_focus_image

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

    # Create a texture (e.g. a bright spot)
    # For simplicity, just a uniform plane that varies in Z
    # Or a single pixel to check impulse response?
    # Let's use a small Gaussian spot in XY as well to ensure it's "focused"
    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny//2, nx//2
    sigma_xy = 5.0
    xy_profile = np.exp(-((y-cy)**2 + (x-cx)**2) / (2 * sigma_xy**2))

    for z in range(nz):
        # Gaussian intensity profile in Z
        # Note: In real EDOF, "focus" means sharpness (gradient), not just intensity.
        # But best_focus_image reconstructs the PIXEL VALUE.
        # If the pixel value varies with Z (as it does in focus), we want to recover the value at best_z.
        # For a fluorescent bead, intensity is max at focus.
        intensity_factor = np.exp(-(z - true_peak_z)**2 / (2 * sigma_z**2))
        stack[z] = xy_profile * intensity_factor

    # 2. Run best_focus_image
    # We force the patch size to be large so we essentially treat it as one block
    # to avoid patch boundary artifacts complicating the measurement.
    # However, best_focus_image calculates focus metric.
    # For a pure intensity gaussian, the "sharpness" (Laplacian) is also max at the peak intensity
    # because Laplacian of Gaussian ~ Gaussian (roughly, second derivative).
    # d2/dx2 (exp(-x^2)) = (4x^2 - 2) exp(-x^2). The envelope is still Gaussian-ish.

    result = best_focus_image(stack, patch_size=32)

    # 3. Measure Peak Intensity
    # The true peak intensity at (cy, cx) should be 1.0 * 1.0 = 1.0 (at z=2.5)
    # At z=2 and z=3, intensity is exp(-0.5^2/2) = exp(-0.125) = 0.882
    # Linear interp: (0.882 + 0.882) / 2 = 0.882.
    # We expect the result to be significantly higher than 0.882 if cubic is used.

    reconstructed_peak = result[cy, cx]

    print(f"Reconstructed Peak: {reconstructed_peak:.4f}")

    # Theoretical limits
    val_at_node = np.exp(-(0.5)**2 / 2) # 0.882
    print(f"Value at nodes (z=2,3): {val_at_node:.4f}")
    print(f"True Peak: 1.0000")

    # For linear, it should be close to 0.882
    # For cubic, it should be > 0.95 maybe?

    # Check if we are doing better than linear
    # We set a threshold halfway between linear and perfect
    # Linear is approx 0.8825. 0.94 is a safe threshold for cubic (0.9522).
    threshold = 0.94

    assert reconstructed_peak > threshold, f"Reconstruction {reconstructed_peak:.4f} is too low (Linear behavior?). Expected > {threshold:.4f}"


if __name__ == "__main__":
    test_interpolation_quality()
