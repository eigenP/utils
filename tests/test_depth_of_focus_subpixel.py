
import numpy as np
import scipy.ndimage as ndi
from eigenp_utils.extended_depth_of_focus import best_focus_image

def generate_slanted_plane(shape=(10, 100, 100), z_start=3.0, z_end=7.0):
    """
    Generates a stack where the focus plane varies linearly from z_start to z_end along the X axis.
    This simulates a slanted surface.
    """
    Z, H, W = shape
    rng = np.random.default_rng(42)
    # High frequency texture
    texture = rng.random((H, W)).astype(np.float32) * 100

    # Ground truth depth map (varies along X)
    x_coords = np.linspace(0, 1, W)
    depth_profile = z_start + (z_end - z_start) * x_coords
    depth_map = np.tile(depth_profile, (H, 1))

    stack = np.zeros(shape, dtype=np.float32)

    # Pre-compute blurs for efficiency
    max_sigma = max(abs(z_start - Z), abs(z_end - 0)) * 0.5
    sigmas = np.linspace(0, max_sigma, 20)
    blurred_stack = []
    for s in sigmas:
        if s == 0:
            blurred_stack.append(texture)
        else:
            blurred_stack.append(ndi.gaussian_filter(texture, sigma=s))
    blurred_stack = np.array(blurred_stack)

    # Fill stack
    for x in range(W):
         # Depth at this column
         d_val = depth_map[0, x]
         for z in range(Z):
             # Distance from focus
             dist = np.abs(d_val - z)
             # Sigma for blur
             s_val = dist * 0.5

             # Find closest pre-blurred
             best_k = np.argmin(np.abs(sigmas - s_val))
             stack[z, :, x] = blurred_stack[best_k, :, x]

    return stack, depth_map

def test_subpixel_refinement():
    """
    Verifies that the height map returned by best_focus_image contains sub-pixel information
    (smooth transitions) rather than discrete integer steps, when provided with a continuous surface.
    """
    H, W = 100, 100
    Z_dim = 10
    z_start, z_end = 3.0, 7.0

    stack, truth_map = generate_slanted_plane((Z_dim, H, W), z_start, z_end)

    # Run best_focus_image
    # Patch size 20 means 5 patches along X.
    # The returned height map is zoomed to full size.
    _, height_map = best_focus_image(stack, patch_size=20, return_heightmap=True)

    # Analyze the height map along the middle row
    mid_y = H // 2
    profile = height_map[mid_y, :]

    # Check smoothness (first difference)
    # A staircase (integer steps) has 0 diffs mostly, and large jumps (1.0).
    # A smooth ramp has small constant diffs (~(7-3)/100 = 0.04).
    diffs = np.diff(profile)
    diff_std = np.std(diffs)

    print(f"Diff Std: {diff_std:.4f}")

    # Integer steps would have high variance in diffs (zeros and ones).
    # Smooth ramp has low variance.
    # Baseline (Integer) was ~0.22. Improved (Subpixel) is ~0.03.
    assert diff_std < 0.1, f"Height map is not smooth (Diff Std={diff_std:.4f}). Expected < 0.1 for sub-pixel accuracy."

    # Check uniqueness
    unique_vals = np.unique(profile)
    n_unique = len(unique_vals)
    print(f"Number of unique depth values: {n_unique}")

    # Integer map would have ~5 values. Float map should have many.
    assert n_unique > 20, f"Height map seems discrete ({n_unique} unique values). Expected > 20 for sub-pixel accuracy."

if __name__ == "__main__":
    test_subpixel_refinement()
