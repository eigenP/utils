
import unittest
import numpy as np
from eigenp_utils.extended_depth_of_focus import best_focus_image

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

if __name__ == "__main__":
    unittest.main()
