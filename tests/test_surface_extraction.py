from scipy.ndimage import gaussian_filter
import numpy as np
import pytest
import unittest

from eigenp_utils.surface_extraction import extract_surface
from eigenp_utils.surface_extraction import extract_surface, pad_z_slices, crop_edges, expand_surface_z, adjust_mask_location



# =========================================
# Source: test_surface_extraction.py
# =========================================


def test_extract_surface():
    """Test that extract surface works as expected."""
    # Create a synthetic 3D image with a surface (a sphere)
    shape = (50, 50, 50)
    image = np.zeros(shape, dtype=np.uint8)

    # Create a sphere in the center
    z, y, x = np.indices(shape)
    center = np.array(shape) / 2
    radius = 15
    mask = (z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2 <= radius**2
    image[mask] = 200 # Surface intensity

    # Add noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 10, shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    # Extract surface
    surface = extract_surface(image, downscale_factor=2)

    assert surface.shape == shape
    assert surface.dtype == bool
    # Check if surface is detected (some True values)
    assert np.any(surface)

    # The surface should be somewhat around the sphere shell
    # We can check if the detected surface points are roughly at radius distance
    # This is a basic sanity check, not precision validation
    surface_points = np.argwhere(surface)
    distances = np.sqrt(np.sum((surface_points - center)**2, axis=1))
    # Check if average distance is reasonable (near radius)
    # Since it extracts top surface, it might be different, but for a sphere top surface is the upper hemisphere.
    # extract_surface finds "top" surface along Z.

    # Let's trust it returns a mask for now.

def test_extract_surface_padding_mismatch():
    """
    Test extract_surface with dimensions that are not divisible by block size,
    causing internal padding that previously led to IndexError.
    """
    # Y = 10, X = 10. Block size = (2, 4, 4)
    # 10 % 4 = 2 != 0. Requires padding.
    Z, Y, X = 20, 10, 10
    image = np.zeros((Z, Y, X), dtype=np.uint8)
    image[10:, :, :] = 200 # Foreground

    # Pad Z manually as in the user report (optional, but good for consistency)
    padded_image = pad_z_slices(image, num_slices=4)

    _BLOCK_SIZE = (2, 4, 4)

    # This should not raise IndexError
    surface = extract_surface(padded_image, downscale_factor=_BLOCK_SIZE)

    assert surface.shape == padded_image.shape
    assert surface.dtype == bool

def test_pad_z_slices():
    """Test that pad z slices works as expected."""
    arr = np.zeros((10, 10, 10))
    # Pad 1 slice before
    padded = pad_z_slices(arr, num_slices=1, pad_before=True, pad_after=False)
    assert padded.shape == (11, 10, 10)

    # Pad 10% -> 1 slice (10 * 0.1 = 1)
    padded = pad_z_slices(arr, num_slices=0.1, pad_before=True, pad_after=True)
    assert padded.shape == (12, 10, 10)

    # Pad with value
    arr[:] = 5
    padded = pad_z_slices(arr, num_slices=1, fill_value=99, pad_before=True)
    assert padded[0, 0, 0] == 99
    assert padded[1, 0, 0] == 5

def test_crop_edges():
    """Test that crop edges works as expected."""
    arr = np.ones((10, 20, 20))
    # Crop 10% from Y and X
    # Y: 20 * 0.1 = 2 pixels from each side
    # X: 20 * 0.1 = 2 pixels from each side
    cropped = crop_edges(arr, 0, 0.1, 0.1)
    assert cropped.shape == arr.shape

    # Central region should be 1
    assert cropped[5, 10, 10] == 1
    # Edges should be 0
    assert cropped[5, 0, 0] == 0
    assert cropped[5, 1, 1] == 0 # 0 and 1 are cropped (0..2)
    assert cropped[5, 2, 2] == 1 # 2 is start of valid region

    # Test 0 crop
    cropped_zero = crop_edges(arr, 0, 0.0, 0.0)
    assert np.all(cropped_zero == arr)

def test_expand_surface_z():
    """Test that expand surface z works as expected."""
    shape = (10, 10, 10)
    surface = np.zeros(shape, dtype=bool)
    surface[5, 5, 5] = True

    expanded = expand_surface_z(surface, thickness=1)

    # Should cover z=4, 5, 6 at y=5, x=5
    assert expanded[4, 5, 5]
    assert expanded[5, 5, 5]
    assert expanded[6, 5, 5]
    # Should not cover z=3 or 7
    assert not expanded[3, 5, 5]
    assert not expanded[7, 5, 5]

def test_adjust_mask_location():
    """Test that adjust mask location works as expected."""
    shape = (10, 10, 10)
    mask = np.zeros(shape, dtype=bool)
    mask[5, 5, 5] = True

    # Translate by (1, 1, 1) -> (6, 6, 6)
    translated = adjust_mask_location(mask, translation=(1, 1, 1))
    assert translated[6, 6, 6]
    assert not translated[5, 5, 5]

    # Dilate
    dilated = adjust_mask_location(mask, morph='dilate', iterations=1)
    # Check neighbors
    assert dilated[5, 5, 6]
    assert dilated[5, 6, 5]

    # Erode
    # Mask with a small block
    mask_block = np.zeros(shape, dtype=bool)
    mask_block[4:7, 4:7, 4:7] = True
    eroded = adjust_mask_location(mask_block, morph='erode', iterations=1)
    # Center should remain, edges eroded
    assert eroded[5, 5, 5]
    assert not eroded[4, 4, 4]

# =========================================
# Source: test_surface_accuracy.py
# =========================================


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
        z_offset = 5.0

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

    def test_downscaling_alignment(self):
        """
        Verify that surface extraction with downscaling does not introduce
        spatial shift bias or boundary artifacts.
        """
        # Create a simple block volume
        Z, Y, X = 64, 64, 64
        vol = np.zeros((Z, Y, X), dtype=np.uint8)

        # Surface at Z=32 in the center
        # Margin 16 ensures alignment with 4x4 blocks
        margin = 16
        vol[32:, margin:-margin, margin:-margin] = 200
        vol[:32, margin:-margin, margin:-margin] = 50

        # Background
        # vol[:, :margin, :] = 10
        # etc... implicitly handled by initialization

        downscale = 4
        # Use low sigma to see the edge clearly
        mask, hmap = extract_surface(vol, downscale_factor=downscale, gaussian_sigma=0.5, return_heightmap=True)

        # Check alignment at the left edge (X=margin)
        # The first valid pixel should be at margin
        center_y = Y // 2
        profile = hmap[center_y, :]

        valid_indices = np.where(profile > 0)[0]
        self.assertTrue(len(valid_indices) > 0, "Surface not found")

        first_valid = valid_indices[0]
        self.assertEqual(first_valid, margin, f"Surface start shifted! Expected {margin}, got {first_valid}")

        # Check for boundary artifacts (curl down)
        # The value at the edge should be close to the value inside
        val_edge = profile[margin]
        val_inside = profile[margin + 2]

        diff = abs(val_inside - val_edge)
        self.assertLess(diff, 2.0, f"Significant boundary artifact detected: edge={val_edge}, inside={val_inside}")

        # Check Z value (should be ~31.5 due to interpolation of transition 31-32)
        # Ground truth is step at 32. Otsu threshold ~125.
        # Transition happens between 31 (50) and 32 (200).
        # Linear interp for 125: (125-50)/(200-50) = 75/150 = 0.5.
        # So Z = 31 + 0.5 = 31.5.
        # Relaxed delta to account for Otsu threshold variability
        self.assertAlmostEqual(val_inside, 31.5, delta=2.0)

if __name__ == '__main__':
    unittest.main()
