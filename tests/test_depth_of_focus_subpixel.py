import unittest
import numpy as np
from eigenp_utils.extended_depth_of_focus import best_focus_image

class TestDepthOfFocusSubpixel(unittest.TestCase):
    def test_subpixel_accuracy_slanted_plane(self):
        """
        Generates a synthetic stack with a slanted plane of focus.
        Verifies that subpixel estimation reduces quantization error.

        Prior to improvement (integer argmax + naive zoom), RMSE was ~0.44 px.
        With parabolic interpolation + RegularGridInterpolator, RMSE is ~0.05 px.
        With Log-Parabolic interpolation, RMSE is ~0.018 px.
        """
        # 1. Synthesis Parameters
        W, H = 128, 128
        n_slices = 10
        sigma_z = 1.0  # Depth of Field (thickness of focus)

        # Ground Truth Depth Map: Plane tilting from z=3 to z=7
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        xv, yv = np.meshgrid(x, y)
        gt_depth = 3.0 + 4.0 * xv  # Gradient along X axis

        # Texture: Random noise
        np.random.seed(42)
        texture = np.random.rand(H, W)

        # Generate Stack
        stack = np.zeros((n_slices, H, W), dtype=np.float32)
        for z in range(n_slices):
            # Distance from focus plane
            dist = z - gt_depth
            # Focus decay (Gaussian profile)
            focus_weight = np.exp(-0.5 * (dist / sigma_z)**2)
            # Image at slice z is texture weighted by focus
            stack[z] = texture * focus_weight

        # 2. Run EDoF (requesting heightmap)
        img, height_map = best_focus_image(stack, patch_size=16, return_heightmap=True)

        # 3. Analyze Error
        # Crop borders to avoid boundary effects of the patch system
        margin = 16
        valid_mask = np.s_[margin:-margin, margin:-margin]

        error = height_map[valid_mask] - gt_depth[valid_mask]
        rmse = np.sqrt(np.mean(error**2))

        print(f"RMSE (subpixel): {rmse:.4f} pixels")

        # 4. Assertions
        # Quantization noise limit for integer steps is ~0.29 px.
        # We demand significantly better than that.
        self.assertLess(rmse, 0.05, "Subpixel accuracy failed! RMSE is too high.")

        return rmse

if __name__ == '__main__':
    unittest.main()
