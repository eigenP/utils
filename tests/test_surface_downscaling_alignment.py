
import unittest
import numpy as np
from eigenp_utils.surface_extraction import extract_surface
from scipy.ndimage import gaussian_filter

class TestSurfaceDownscalingAlignment(unittest.TestCase):
    """
    Matth 🧠 Verification: Surface Extraction Downscaling Alignment

    Validates that the surface extraction algorithm correctly handles coordinate
    transforms when downscaling is active, ensuring zero systematic bias (shift)
    for slanted planes.
    """

    def generate_slanted_plane(self, shape=(50, 100, 100), slope_x=0.1, slope_y=0.0, z_offset=20.0):
        """
        Generates a volume with a sharp sigmoidal transition at z = slope_x * x + slope_y * y + z_offset.
        """
        Z, Y, X = shape
        grid_y, grid_x = np.mgrid[0:Y, 0:X]
        true_surface = slope_x * grid_x + slope_y * grid_y + z_offset

        grid_z = np.arange(Z)[:, None, None]

        # Sigmoid transition centered at true_surface
        # Steep transition to mimic binary object but with subpixel information available via smoothing
        vol = 1.0 / (1.0 + np.exp(-2.0 * (grid_z - true_surface)))
        vol = (vol * 255).astype(np.uint8)

        return vol, true_surface

    def test_downscaling_bias_x(self):
        """
        Test alignment for X-slope with downscale_factor=4.
        If X-alignment is wrong (shift), a slope in X will cause a height bias.
        """
        vol, true_surface = self.generate_slanted_plane(slope_x=0.1, slope_y=0.0)

        # Use return_heightmap=True to get float precision
        # Use small sigma to minimize smoothing bias, but enough to allow subpixel interp
        mask, height_map = extract_surface(
            vol,
            downscale_factor=4,
            gaussian_sigma=1.0,
            return_heightmap=True
        )

        # Crop edges to avoid boundary artifacts
        crop = 10
        diff = height_map[crop:-crop, crop:-crop] - true_surface[crop:-crop, crop:-crop]

        rmse = np.sqrt(np.mean(diff**2))
        bias = np.mean(diff)

        print(f"X-Slope Bias: {bias:.4f}, RMSE: {rmse:.4f}")

        # With correct alignment, bias should be negligible (< 0.2 pixel)
        # Without fix, bias was ~ -1.5 pixels
        self.assertLess(abs(bias), 0.2, "Systematic bias should be negligible (< 0.2 px) after coordinate correction")
        self.assertLess(rmse, 0.5, "RMSE should be low (< 0.5 px)")

    def test_downscaling_bias_y(self):
        """
        Test alignment for Y-slope with downscale_factor=4.
        """
        vol, true_surface = self.generate_slanted_plane(slope_x=0.0, slope_y=0.1)

        mask, height_map = extract_surface(
            vol,
            downscale_factor=4,
            gaussian_sigma=1.0,
            return_heightmap=True
        )

        crop = 10
        diff = height_map[crop:-crop, crop:-crop] - true_surface[crop:-crop, crop:-crop]

        rmse = np.sqrt(np.mean(diff**2))
        bias = np.mean(diff)

        print(f"Y-Slope Bias: {bias:.4f}, RMSE: {rmse:.4f}")

        self.assertLess(abs(bias), 0.2, "Systematic bias should be negligible (< 0.2 px)")
        self.assertLess(rmse, 0.5, "RMSE should be low (< 0.5 px)")

if __name__ == '__main__':
    unittest.main()
