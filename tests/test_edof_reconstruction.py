
import unittest
import numpy as np
from eigenp_utils.extended_depth_of_focus import best_focus_image

class TestEDOFIntensityRecovery(unittest.TestCase):
    def test_intensity_recovery_peak_between_slices(self):
        """
        Verifies that the EDoF reconstruction preserves intensity when the focus peak
        falls exactly between two slices.
        """
        W, H = 64, 64
        n_slices = 10
        sigma_z = 1.0

        # Place peak exactly at 4.5 (midway between slice 4 and 5)
        gt_depth = 4.5

        # Texture: High frequency noise to ensure robust focus scoring
        np.random.seed(42)
        base_texture = np.random.rand(H, W) + 1.0 # Mean ~1.5, range [1, 2]

        stack = np.zeros((n_slices, H, W), dtype=np.float32)

        # Generate stack
        for z in range(n_slices):
            dist = z - gt_depth
            focus_weight = np.exp(-0.5 * (dist / sigma_z)**2)
            stack[z] = base_texture * focus_weight

        # Run EDoF
        # Using a patch size that covers the image roughly to minimize boundary issues
        # effectively testing the core reconstruction logic
        img = best_focus_image(stack, patch_size=32)

        # Analyze the central region
        margin = 16
        center_region = img[margin:-margin, margin:-margin]
        gt_region = base_texture[margin:-margin, margin:-margin]

        # We compute the ratio of reconstructed intensity to the base texture intensity
        # Since stack[z] = base * weight, reconstructed = base * interpolated_weight
        # ratio = reconstructed / base = interpolated_weight

        # We average the ratio over pixels to smooth out numerical noise
        ratio_map = center_region / gt_region
        mean_recovery = np.mean(ratio_map)

        print(f"Mean Recovery Ratio: {mean_recovery:.4f}")

        # Theoretical Max (Focus Peak): 1.0
        # Linear Interpolation at midpoint (z=4.5):
        # w(4) = exp(-0.5 * 0.5^2) = 0.8825
        # w(5) = 0.8825
        # Linear(4.5) = 0.8825

        # Cubic Interpolation at midpoint:
        # Should be significantly higher, close to 0.95 or better
        # (calculated ~0.952 in thought process)

        self.assertGreater(mean_recovery, 0.90,
                           f"Intensity recovery {mean_recovery:.4f} too low! Likely linear interpolation.")

if __name__ == '__main__':
    unittest.main()
