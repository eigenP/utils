
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from eigenp_utils.extended_depth_of_focus import best_focus_image

def generate_synthetic_stack(shape=(10, 64, 64), focus_z=4.5):
    """
    Generates a stack where the focus varies with Z.
    The "image" is random noise blurred by a gaussian kernel whose sigma depends on |z - focus_z|.
    Sigma is minimal (but non-zero) at focus_z.
    """
    rng = np.random.default_rng(42)
    # Fixed texture at the "object plane"
    texture = rng.random(shape[1:])

    stack = np.zeros(shape, dtype=np.float32)

    for z in range(shape[0]):
        # Distance from focal plane
        dist = abs(z - focus_z)
        # Sigma grows with distance (blur)
        sigma = 0.5 + 2.0 * dist
        # Blur the texture
        stack[z] = gaussian_filter(texture, sigma=sigma)

    return stack

def test_subpixel_accuracy():
    # Focus is exactly between slice 4 and 5
    true_z = 4.5
    stack = generate_synthetic_stack(shape=(10, 64, 64), focus_z=true_z)

    # Run EDOF with sub-pixel refinement
    final_img, height_map = best_focus_image(stack, patch_size=32, return_heightmap=True)

    print(f"Height map stats: Mean={height_map.mean():.4f}, Min={height_map.min()}, Max={height_map.max()}")

    # The height map should NOT be integers anymore
    is_integer = np.all(np.mod(height_map, 1) == 0)

    # Error from true_z
    rmse = np.sqrt(np.mean((height_map - true_z)**2))
    print(f"RMSE from true z={true_z}: {rmse:.4f}")

    # Assertions
    assert not is_integer, "Height map should contain sub-pixel values"
    assert rmse < 0.1, f"RMSE {rmse:.4f} is too high (expected < 0.1 for sub-pixel accuracy)"

def test_reconstruction_subpixel():
    # Test if reconstruction actually interpolates between slices
    # Create a stack where slice 4 is 0 and slice 5 is 100.
    # If best_z is 4.5, result should be 50.

    stack = np.zeros((10, 64, 64), dtype=np.float32)

    # We need to trick the focus scoring to pick 4.5

    # Stack with texture AND intensity gradient
    rng = np.random.default_rng(42)
    texture = rng.random((64, 64))

    stack_grad = np.zeros((10, 64, 64), dtype=np.float32)
    for z in range(10):
        # Texture sharpness peaks at 4.5
        dist = abs(z - 4.5)
        sigma = 0.5 + 2.0 * dist
        # Normalize texture to mean 0, std 1 before adding to mean intensity
        t = gaussian_filter(texture, sigma=sigma)
        t = (t - t.mean()) / t.std()

        # Mean intensity increases with Z: z * 10
        # Slice 4: 40. Slice 5: 50. Expected at 4.5: 45.
        stack_grad[z] = t + z * 10.0

    final_img, height_map = best_focus_image(stack_grad, patch_size=32, return_heightmap=True)

    # Check estimated Z
    est_z = height_map.mean()
    print(f"Estimated Z mean: {est_z:.4f}")
    print(f"Estimated Z min: {height_map.min():.4f}")
    print(f"Estimated Z max: {height_map.max():.4f}")

    assert abs(est_z - 4.5) < 0.1

    # Check output intensity
    mean_val = final_img.mean()
    min_val = final_img.min()
    max_val = final_img.max()
    print(f"Output Intensity: Mean={mean_val:.4f}, Min={min_val:.4f}, Max={max_val:.4f}")

    # Debug: Check if any pixel is exactly 0 (indicating no coverage)
    zeros = np.sum(final_img == 0)
    print(f"Number of zero pixels: {zeros}")

    assert abs(mean_val - 45.0) < 1.0, f"Reconstruction did not interpolate intensity. Got {mean_val}, expected 45."

if __name__ == "__main__":
    test_subpixel_accuracy()
    test_reconstruction_subpixel()
