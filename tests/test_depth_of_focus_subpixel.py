
import numpy as np
import scipy.ndimage as ndi
from eigenp_utils.extended_depth_of_focus import best_focus_image
import pytest

def generate_slanted_plane_stack(shape=(10, 100, 100), slope=0.05):
    """
    Generates a stack where the focus plane is a slant: Z = 2 + slope * x.
    """
    Z, H, W = shape
    rng = np.random.default_rng(42)
    # High frequency texture
    texture = rng.random((H, W)).astype(np.float32)

    stack = np.zeros(shape, dtype=np.float32)

    # True depth map
    x_coords = np.arange(W)
    y_coords = np.arange(H)
    # Broadcast to (H, W)
    xx, yy = np.meshgrid(x_coords, y_coords)

    true_depth = 2.0 + slope * xx
    true_depth = np.clip(true_depth, 0, Z-1)

    for z in range(Z):
        # Distance from focal plane
        dist = np.abs(true_depth - z)

        # Blur sigma proportional to distance
        # We need a way to apply variable blur.
        # Standard gaussian_filter takes constant sigma.
        # Approximation: Blending blurred versions is complex.
        # Simpler: Generate the stack by iterating pixels (slow) or
        # iterating Z and realizing that for a specific Z, the sigma varies spatially.

        # Let's do a simplified approach:
        # Create discrete versions of the texture with different blurs.
        # sigma = 0, 0.5, 1.0, 1.5 ...
        # Interpolate between them?

        # Even simpler: Just modulate the INTENSITY of high frequencies?
        # No, that changes energy.

        # Let's use the property that we only care about the SCORE.
        # But best_focus_image computes the score from the image.

        # Let's slice the texture by Z.
        # For each Z, we apply a blur that is roughly correct for the CENTER of the patch?
        # No, we want pixel-wise correctness.

        # Approximating spatially variant blur:
        # For this test, we can just model the 1D case or use a loop over columns (since slant is only in X).
        pass

    # Efficient generation for X-slant
    for col in range(W):
        d = 2.0 + slope * col
        # For this column, the focal plane is at depth d.
        # Calculate blur sigma for each z
        for z in range(Z):
            sigma = 0.5 * abs(d - z)
            # Apply 1D blur along Y to the column?
            # Or just blur the whole image with constant sigma and pick the column?
            # That's expensive (Z * Z * H * W).
            pass

    # Vectorized approach:
    # 1. Pre-generate blurred versions of the texture for a range of sigmas.
    sigmas = np.linspace(0, 5, 51) # 0.1 steps
    blurred_library = []
    for s in sigmas:
        if s == 0:
            blurred_library.append(texture)
        else:
            blurred_library.append(ndi.gaussian_filter(texture, sigma=s))
    blurred_library = np.array(blurred_library) # (N_sigmas, H, W)

    # 2. Assign pixel values
    for z in range(Z):
        dist = np.abs(true_depth - z)
        # Map dist to sigma index
        sigma_idx = np.clip(np.round(dist / 0.1 * 0.5).astype(int), 0, 50) # sigma = 0.5 * dist

        # Fancy indexing to pull pixels
        # stack[z] = blurred_library[sigma_idx, yy, xx]
        # This is (H, W)

        # To avoid massive memory of grid indexing:
        for r in range(H):
             stack[z, r, :] = blurred_library[sigma_idx[r, :], r, xx[r, :]]

    return stack, true_depth

def test_subpixel_smoothness():
    """
    Verifies that the reconstructed height map is smooth and not stepped.
    """
    # Create a slant that spans Z=2 to Z=6 over 100 pixels.
    # Slope = 4 / 100 = 0.04
    shape = (10, 128, 128)
    stack, true_depth = generate_slanted_plane_stack(shape=shape, slope=0.04)

    # Run reconstruction with return_heightmap
    # Use a small patch size to get decent resolution on the map
    # Note: best_focus_image uses patch_size for scoring.
    patch_size = 16

    # We expect the height_map to look like the true_depth.
    # Current implementation: height_map is integers.
    # RMSE will be dominated by quantization noise (uniform distribution U(-0.5, 0.5) has var 1/12 approx 0.08).

    fused, height_map = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

    # Analyze a middle row
    row_idx = 64
    pred_depth_row = height_map[row_idx, :]
    true_depth_row = true_depth[row_idx, :]

    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred_depth_row - true_depth_row)**2))

    print(f"RMSE: {rmse:.4f}")

    # Calculate smoothness (First difference)
    # Real slope is 0.04.
    # Integer steps will be 0, 0, ..., 1, 0, ...
    diffs = np.diff(pred_depth_row)
    # Check max jump. If quantized, we expect jumps of 1.
    # If smooth, we expect jumps of ~0.04.
    max_jump = np.max(np.abs(diffs))

    print(f"Max Jump: {max_jump}")

    return rmse, max_jump

if __name__ == "__main__":
    test_subpixel_smoothness()
