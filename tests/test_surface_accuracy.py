
import unittest
import numpy as np
from eigenp_utils.surface_extraction import extract_surface
from scipy.ndimage import gaussian_filter

class TestSurfaceAccuracy(unittest.TestCase):
    """
    Matth 🧠 Verification: Surface Extraction Accuracy

    Validates that the surface extraction algorithm correctly identifies
    the surface of a 3D object and measures the precision of the height map.
    """

    def generate_slanted_plane(self, shape=(50, 64, 64), slope_y=0.2, slope_x=0.1, z_offset=10, noise_sigma=0.0):
        """
        Generates a 3D volume with a slanted plane interface.
        Voxels ABOVE the plane (low Z) are dark (background).
        Voxels BELOW the plane (high Z) are bright (foreground).
        """
        Z, Y, X = shape
        vol = np.zeros(shape, dtype=np.float32)

        y_idx, x_idx = np.indices((Y, X))

        # True surface height at each (y, x)
        gt_height = z_offset + slope_y * y_idx + slope_x * x_idx

        # Fill volume
        # I(z) = 0 if z < h, 1 if z > h
        # Ramp: I(z) = clip(z - h + 0.5, 0, 1)

        for z in range(Z):
            # Signed distance from surface (positive = inside object, below surface)
            dist = z - gt_height

            # Clamp to 0-1 (linear ramp over 1 pixel thickness)
            val = np.clip(dist + 0.5, 0.0, 1.0)

            vol[z] = val

        # Add noise
        if noise_sigma > 0:
            vol += np.random.normal(0, noise_sigma, size=shape)
            vol = np.clip(vol, 0.0, 1.0)

        # Scale to uint8 for the algorithm
        vol_u8 = (vol * 255).astype(np.uint8)

        return vol_u8, gt_height

    def test_surface_precision(self):
        """
        Measure the RMSE of the extracted surface against the ground truth.
        """
        # Parameters
        shape = (40, 64, 64)
        slope_y = 0.15
        slope_x = 0.05
        z_offset = 20.0

        # No noise to test pure quantization error
        vol, gt_height = self.generate_slanted_plane(shape, slope_y, slope_x, z_offset, noise_sigma=0.0)

        # Run extraction
        # We increase gaussian_sigma to 2.0 to ensure the sharp step function of the synthetic data
        # is blurred enough to create a gradient for subpixel interpolation.
        # This simulates real microscopy data which is always band-limited (PSF).
        surface_mask, height_map = extract_surface(vol, downscale_factor=1, gaussian_sigma=2.0, clahe_clip=0.0, return_heightmap=True)

        found_mask = np.any(surface_mask, axis=0)

        # Compare ground truth to the floating point height map
        error = height_map[found_mask] - gt_height[found_mask]

        H, W = height_map.shape
        crop = 5
        valid_crop = found_mask[crop:-crop, crop:-crop]
        error_crop = error.reshape(H, W)[crop:-crop, crop:-crop][valid_crop]

        rmse = np.sqrt(np.mean(error_crop**2))

        print(f"Surface RMSE (Float): {rmse:.4f} pixels")

        # With sufficient smoothing, subpixel interpolation works effectively.
        self.assertLess(rmse, 0.20, "RMSE should be better than integer quantization (0.29)")

    def test_surface_precision_downscaled(self):
        """
        Measure the RMSE of the extracted surface against the ground truth with downscaling.
        """
        # Parameters
        shape = (40, 64, 64)
        slope_y = 0.15
        slope_x = 0.05
        z_offset = 20.0

        # No noise to test pure quantization error
        vol, gt_height = self.generate_slanted_plane(shape, slope_y, slope_x, z_offset, noise_sigma=0.0)

        # Run extraction with downscale_factor=4
        surface_mask, height_map = extract_surface(vol, downscale_factor=4, gaussian_sigma=2.0, clahe_clip=0.0, return_heightmap=True)

        found_mask = np.any(surface_mask, axis=0)

        # Compare ground truth to the floating point height map
        error = height_map[found_mask] - gt_height[found_mask]

        H, W = height_map.shape
        crop = 5
        valid_crop = found_mask[crop:-crop, crop:-crop]
        error_crop = error.reshape(H, W)[crop:-crop, crop:-crop][valid_crop]

        rmse = np.sqrt(np.mean(error_crop**2))

        # Mean Error (Bias)
        bias = np.mean(error_crop)

        print(f"Surface RMSE (Downscaled): {rmse:.4f} pixels")
        print(f"Surface Bias (Downscaled): {bias:.4f} pixels")

        # Tolerance relaxed due to downscaling (sz=4 implies +/- 2 pixel uncertainty)
        # Observed RMSE ~ 2.0 pixels
        self.assertLess(rmse, 2.5, "RMSE should be within half block size")

        # Check bias
        self.assertLess(abs(bias), 2.5, "Systematic shift should be within half block size")

    def test_translation_invariance(self):
        """
        The surface should not change shape when the object is shifted in Z.
        Subpixel extraction should shift smoothly.
        """
        shape = (40, 32, 32)
        # Shift by 0.5 pixels
        vol1, gt1 = self.generate_slanted_plane(shape, z_offset=10.0)
        vol2, gt2 = self.generate_slanted_plane(shape, z_offset=10.5)

        # Use sigma=2.0 for smoothing
        mask1, h1 = extract_surface(vol1, downscale_factor=1, gaussian_sigma=2.0, return_heightmap=True)
        mask2, h2 = extract_surface(vol2, downscale_factor=1, gaussian_sigma=2.0, return_heightmap=True)

        valid = (np.any(mask1, axis=0)) & (np.any(mask2, axis=0))

        crop = 5
        valid[:crop, :] = False
        valid[-crop:, :] = False
        valid[:, :crop] = False
        valid[:, -crop:] = False

        z1 = h1[valid]
        z2 = h2[valid]

        mean_shift = np.mean(z2 - z1)
        print(f"Detected Shift (True=0.5): {mean_shift:.4f}")

        diff_std = np.std(z2 - z1)
        print(f"Shift StdDev: {diff_std:.4f}")

        self.assertAlmostEqual(mean_shift, 0.5, delta=0.05)

        # With sigma=2.0, the "staircase" effect is smoothed out, so the shift should be consistent.
        self.assertLess(diff_std, 0.15, "Shift StdDev should be low for rigid shift")

if __name__ == '__main__':
    unittest.main()
