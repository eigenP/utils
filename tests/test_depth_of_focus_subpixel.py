import unittest
import numpy as np
import scipy.ndimage as ndi
from eigenp_utils.extended_depth_of_focus import best_focus_image

class TestDepthOfFocusSubpixel(unittest.TestCase):
    """
    Testr Verification: Sub-pixel Depth Estimation

    This test verifies that the `best_focus_image` algorithm provides sub-pixel accuracy
    when estimating the depth map of a continuous surface.

    The Setup:
    - A synthetic "Slanted Plane" where the true depth z depends linearly on x and y.
    - We simulate the focus stack by modulating a high-frequency texture with a Gaussian profile in Z.
      I(z, y, x) = Texture(y, x) * exp( - (z - z_true(y, x))^2 / (2 * width^2) )
    - This creates a signal where the Laplacian (focus measure) peaks continuously at z_true.

    The Invariants:
    1. RMSE < 0.20: The Root Mean Square Error of the recovered height map vs ground truth
       must be significantly lower than the theoretical limit of integer quantization.
       (Uniform quantization noise for interval 1 is 1/sqrt(12) approx 0.29).
       We expect sub-pixel logic to achieve ~0.1 or better.
    """

    def test_slanted_plane_accuracy(self):
        H, W = 256, 256
        Z = 20
        np.random.seed(42)

        # 1. Generate Ground Truth Depth Map
        # Slanted plane from z=5 to z=15
        y, x = np.mgrid[0:H, 0:W]

        # Slopes
        # z = 5 + (10 * y / H)
        # varying only in Y for simplicity of visualization/debugging,
        # but coupled with patches it tests the blending.
        z_true = 5.0 + 10.0 * (y / float(H)) + 2.0 * (x / float(W))
        # Range approx 5 to 17. Fits in Z=20.

        # 2. Generate Texture
        # White noise
        texture = np.random.uniform(0, 100, (H, W)).astype(np.float32)

        # 3. Generate Stack
        stack = np.zeros((Z, H, W), dtype=np.float32)
        width = 1.5 # Sigma of the focus peak in Z

        # Vectorized generation
        # (Z, H, W) grid
        z_coords = np.arange(Z).reshape(Z, 1, 1)

        # Gaussian envelope
        envelope = np.exp( - (z_coords - z_true)**2 / (2 * width**2) )

        # Apply to texture
        # Broadcasting: (Z, H, W) = (H, W) * (Z, H, W)
        stack = texture * envelope

        # 4. Run Algorithm
        # Use a reasonably small patch size to resolve the slope
        # Slope is 10/256 = 0.04 slices per pixel.
        # Patch size 32 => Change of 1.2 slices per patch.
        # This tests the algorithm's ability to interpolate.
        patch_size = 32

        # We assume the algorithm returns a zoomed height map.
        # Since we want to check raw accuracy of the estimation, we care about 'height_map_full'.
        _, height_map_full = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

        # 5. Evaluate RMSE
        # We crop the boundaries where padding/patch artifacts might occur.
        margin = patch_size
        valid_mask = np.s_[margin:-margin, margin:-margin]

        diff = height_map_full[valid_mask] - z_true[valid_mask]
        rmse = np.sqrt(np.mean(diff**2))

        print(f"\nSub-pixel Depth RMSE: {rmse:.4f} pixels")
        print(f"GT Range: {z_true[valid_mask].min():.2f} - {z_true[valid_mask].max():.2f}")
        print(f"Est Range: {height_map_full[valid_mask].min():.2f} - {height_map_full[valid_mask].max():.2f}")
        print(f"Mean Diff: {np.mean(diff):.4f}")

        # Theoretical limit for integer quantization (Uniform[-0.5, 0.5]) is 0.288
        # We expect significant improvement.
        self.assertLess(rmse, 0.20,
            f"RMSE ({rmse:.4f}) should be better than integer quantization limit (~0.29).")

        # If it was integer only, RMSE would be ~0.30.
        # With parabolic fit, we usually get < 0.10 for this setup.

if __name__ == '__main__':
    unittest.main()
