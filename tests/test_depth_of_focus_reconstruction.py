import unittest
import numpy as np
from eigenp_utils.extended_depth_of_focus import best_focus_image

class TestDepthOfFocusReconstruction(unittest.TestCase):
    def test_peak_intensity_recovery(self):
        """
        Verifies that Cubic Hermite Spline interpolation recovers the peak intensity
        of a Gaussian focus profile better than Linear Interpolation.
        """
        W, H = 128, 128
        n_slices = 5
        sigma_z = 1.0
        true_peak_z = 1.5

        # Texture: Deterministic high-frequency pattern to minimize variance noise
        # A checkerboard pattern
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        texture = ((x + y) % 2).astype(np.float32)

        # Simulate stack
        stack = np.zeros((n_slices, H, W), dtype=np.float32)
        for z in range(n_slices):
            val = np.exp(-0.5 * ((z - true_peak_z) / sigma_z)**2)
            stack[z] = texture * val

        gt_mean_intensity = np.mean(texture) * 1.0

        # Run with larger patch to ensure stable focus metric
        recon_img, height_map = best_focus_image(stack, patch_size=32, return_heightmap=True)

        # Check detected height map
        # We expect ~1.5 everywhere
        center_height = np.mean(height_map[32:-32, 32:-32])
        print(f"Mean Detected Height: {center_height:.4f}")

        # Measure center intensity
        center_val = np.mean(recon_img[32:-32, 32:-32])

        print(f"Reconstructed Mean Intensity: {center_val:.4f}")
        print(f"Ground Truth Mean Intensity:  {gt_mean_intensity:.4f}")

        error = abs(gt_mean_intensity - center_val)
        print(f"Error: {error:.4f}")

        expected_linear_error = (1.0 - np.exp(-0.125)) * np.mean(texture)
        print(f"Expected Linear Error: {expected_linear_error:.4f}")

        # With deterministic texture, the peak detection should be very accurate.
        # We expect the error to be close to the theoretical limit of cubic interpolation.

        self.assertLess(error, expected_linear_error * 0.5,
                       f"Reconstruction error {error:.4f} is not significantly better than linear {expected_linear_error:.4f}")

if __name__ == '__main__':
    unittest.main()
