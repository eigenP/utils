
import numpy as np
from src.eigenp_utils.surface_extraction import extract_surface
import matplotlib.pyplot as plt

def test_outlier_failure():
    # 1. Create a synthetic volume (100x100x100)
    # Surface at z=50 (flat)
    shape = (100, 100, 100)
    vol = np.zeros(shape, dtype=np.uint8)

    # Fill surface
    vol[50:, :, :] = 100

    # 2. Add "floating debris" (outliers) above the surface
    # Make it a larger blob to survive binning (downscale=4 -> bin 4x4x4)
    # A 5x5x5 blob at z=20. Brighter than surface.
    vol[15:25, 45:55, 45:55] = 200

    # 3. Run extract_surface
    # downscale=4
    surf_mask_ds, depth_map_ds = extract_surface(
        vol,
        downscale_factor=4,
        gaussian_sigma=1.0,
        return_heightmap=True
    )

    # Check region around the debris
    # Map back to original coords
    # Center of debris is roughly 50, 50
    z_center = depth_map_ds[50, 50]
    print(f"Depth at debris center (50,50) [Downscale=4]: {z_center:.2f}")

    # The surface is at 50. Debris is at ~15-25.
    # If z_center is < 30, it picked the debris.

    # Check neighbor far from debris
    z_neighbor = depth_map_ds[10, 10]
    print(f"Depth at background (10,10): {z_neighbor:.2f}")

    if z_center < 30:
        print("FAILURE: Picked debris instead of surface.")
    else:
        print("SUCCESS: Ignored debris.")

if __name__ == "__main__":
    test_outlier_failure()
