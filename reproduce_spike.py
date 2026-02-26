
import numpy as np
from src.eigenp_utils.surface_extraction import extract_surface
import matplotlib.pyplot as plt

def test_outlier_rejection():
    # 1. Create a synthetic volume (100x100x100)
    # Surface at z=50 (flat)
    shape = (100, 100, 100)
    vol = np.zeros(shape, dtype=np.uint8)

    # Fill surface
    vol[50:, :, :] = 100

    # 2. Add "floating debris" (outliers) above the surface
    # Single pixel spike at z=20
    # At downscale=4, this pixel might be averaged, but if bright enough...
    # Let's make a 2x2 block to survive binning (downscale=4 -> bin size 4x4x4)
    # Actually, binning is mean.
    # If we have a bright spot 255 at z=20.
    vol[20, 50, 50] = 255

    # 3. Run extract_surface
    # We expect the surface at (50, 50) to be near z=20 instead of z=50
    # downscale=1 to verify basic behavior first
    surf_mask_naive, depth_map_naive = extract_surface(
        vol,
        downscale_factor=1,
        gaussian_sigma=1.0,
        return_heightmap=True
    )

    z_at_spike = depth_map_naive[50, 50]
    print(f"Depth at spike (50,50) [No Downscale]: {z_at_spike:.2f} (Expected ~20 or ~50)")

    # 4. Run with downscale=4 (standard usage)
    # The spike is single pixel. Binning (4x4x4) will average it.
    # 255 / 64 = 4. Might be below threshold.
    # Let's make the spike bigger or brighter.
    # Block of 4x4 at z=20
    vol[20:24, 48:52, 48:52] = 255

    surf_mask_ds, depth_map_ds = extract_surface(
        vol,
        downscale_factor=4,
        gaussian_sigma=1.0,
        return_heightmap=True
    )

    # Check region around the spike
    # Map back to original coords
    z_center = depth_map_ds[50, 50]
    print(f"Depth at spike (50,50) [Downscale=4]: {z_center:.2f}")

    # Check neighbors (should be ~50)
    z_neighbor = depth_map_ds[50, 60]
    print(f"Depth at neighbor (50,60): {z_neighbor:.2f}")

    return depth_map_ds

if __name__ == "__main__":
    test_outlier_rejection()
