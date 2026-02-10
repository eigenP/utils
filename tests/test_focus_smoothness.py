
import unittest
import numpy as np
from eigenp_utils.extended_depth_of_focus import best_focus_image

class TestFocusSmoothness(unittest.TestCase):
    def test_smoothness_slanted_plane(self):
        """
        Generates a synthetic stack with a slanted plane of focus and dense noise texture.
        Verifies that the reconstructed image amplitude is uniform (no block artifacts).
        """
        # 1. Synthesis Parameters
        W, H = 100, 500  # Increase H for better averaging
        n_slices = 20
        sigma_z = 2.0  # Depth of Field

        # Ground Truth Depth Map: Plane tilting from z=5 to z=15
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xv, yv = np.meshgrid(x, y)
        gt_depth = 5.0 + 10.0 * xv  # Gradient along X axis

        # Dense noise texture (Uniform 0-1)
        np.random.seed(42)
        texture = np.random.uniform(0, 1, (H, W)).astype(np.float32)

        # Generate Stack
        stack = np.zeros((n_slices, H, W), dtype=np.float32)
        for z in range(n_slices):
            # Distance from focus plane
            dist = z - gt_depth
            # Focus decay (Gaussian profile)
            focus_weight = np.exp(-0.5 * (dist / sigma_z)**2)
            stack[z] = texture * focus_weight

        # 2. Run EDoF
        # Using patch_size=20, so boundaries at 20, 40, 60, 80
        img, height_map = best_focus_image(stack, patch_size=20, return_heightmap=True)

        # 3. Analyze Smoothness

        # We calculate the mean profile along columns.
        # Ideally, it should be Mean(Texture) = 0.5.
        # If there are block artifacts (wrong Z), intensity drops, so mean drops.

        profile = np.mean(img, axis=0)
        expected_mean = 0.5

        # Check standard deviation of the profile (fluctuation across X)
        # Expected std err of mean = sigma_noise / sqrt(H) = 0.29 / sqrt(500) = 0.013
        profile_std = np.std(profile)
        print(f"Profile Std Dev: {profile_std:.4f}")
        print(f"Profile Mean: {np.mean(profile):.4f}")

        # If artifacts are present, they cause systematic dips at regular intervals.
        # This increases profile_std.

        # Also check Height Map smoothness
        laplacian_h = np.abs(np.gradient(np.gradient(height_map, axis=0), axis=0)) + \
                      np.abs(np.gradient(np.gradient(height_map, axis=1), axis=1))
        mean_laplacian = np.mean(laplacian_h)
        print(f"Mean Laplacian of Height Map: {mean_laplacian:.4f}")

        # Check RMSE of height map
        err = height_map - gt_depth
        rmse = np.sqrt(np.mean(err**2))
        print(f"Height Map RMSE: {rmse:.4f}")

        self.assertLess(mean_laplacian, 0.1, "Height map is not smooth.")
        self.assertLess(profile_std, 0.025, "Reconstructed image profile has high variance (artifacts).")
        self.assertLess(rmse, 0.2, "Height map RMSE is too high.")

if __name__ == '__main__':
    unittest.main()
