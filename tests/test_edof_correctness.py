
import unittest
import numpy as np
import time
from eigenp_utils.extended_depth_of_focus import best_focus_image
from skimage.data import binary_blobs
from scipy.ndimage import gaussian_filter

class TestEDOFCorrectness(unittest.TestCase):

    def generate_slanted_plane(self, shape=(30, 256, 256), slope=0.05, texture_sigma=1.0):
        Z, H, W = shape

        # Ground truth texture (in focus everywhere)
        # Binary blobs smoothed to create continuous texture
        np.random.seed(42)
        texture = binary_blobs(length=max(H, W), blob_size_fraction=0.05).astype(float)
        texture = texture[:H, :W]
        texture = gaussian_filter(texture, sigma=texture_sigma)
        texture = (texture - texture.min()) / (texture.max() - texture.min())

        # Ground truth depth map
        y = np.arange(H)
        x = np.arange(W)
        X, Y = np.meshgrid(x, y)

        # Plane: z = slope*x + slope*y + Z/3
        depth_map = slope * X + slope * Y + Z / 3.0

        # Generate stack
        stack = np.zeros(shape, dtype=np.float32)

        # Gaussian PSF width (sigma_z)
        sigma_z = 1.5

        # Vectorized stack generation is memory hungry, let's loop
        # I(z, y, x) = Texture(y, x) * exp( - (z - depth_map(y,x))^2 / 2*sigma^2 )

        for z in range(Z):
            dist = z - depth_map
            weight = np.exp( - (dist**2) / (2 * sigma_z**2) )
            stack[z] = texture * weight

        return stack, texture, depth_map

    def test_reconstruction_accuracy(self):
        print("Generating data...")
        stack, gt_texture, gt_depth = self.generate_slanted_plane()

        print("Running best_focus_image...")
        start_time = time.time()

        # Run with default patch size
        result, height_map = best_focus_image(stack, return_heightmap=True)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time:.4f}s")

        # Calculate RMSE
        result_norm = (result - result.mean()) / result.std()
        gt_norm = (gt_texture - gt_texture.mean()) / gt_texture.std()

        rmse = np.sqrt(np.mean((result_norm - gt_norm)**2))
        print(f"Image RMSE (Normalized): {rmse:.4f}")

        # Check depth map accuracy
        rmse_depth = np.sqrt(np.mean((height_map - gt_depth)**2))
        print(f"Depth RMSE: {rmse_depth:.4f}")

        # Assertions based on observed improvement
        # Previous Image RMSE was ~0.35
        # Previous Depth RMSE was ~0.98
        self.assertLess(rmse, 0.32, "Image RMSE should be improved (< 0.32)")
        self.assertLess(rmse_depth, 1.0, "Depth RMSE should be reasonable (< 1.0)")

        # Ensure it is fast (benchmark was 0.07s vs 0.27s)
        # 1.0s is a safe upper bound for CI
        self.assertLess(execution_time, 1.0, "Execution time should be fast")

if __name__ == "__main__":
    unittest.main()
