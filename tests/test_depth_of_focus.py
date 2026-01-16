import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from eigenp_utils.extended_depth_of_focus import best_focus_image

def generate_feature_stack(shape=(10, 200, 200), feature_size=60):
    """
    Generates a 3D stack with a central square feature at Z=7, background Z=2.
    """
    Z, H, W = shape
    rng = np.random.default_rng(42)
    texture = rng.random((H, W))

    depth_map = np.zeros((H, W), dtype=int) + 2

    mid_h, mid_w = H // 2, W // 2
    half_size = feature_size // 2

    depth_map[mid_h - half_size : mid_h + half_size,
              mid_w - half_size : mid_w + half_size] = 7

    stack = np.zeros(shape, dtype=np.float32)

    for z in range(Z):
        dist_1 = abs(2 - z)
        dist_2 = abs(7 - z)

        sigma1 = dist_1 * 0.5
        sigma2 = dist_2 * 0.5

        layer1 = gaussian_filter(texture, sigma=sigma1) if sigma1 > 0 else texture
        layer2 = gaussian_filter(texture, sigma=sigma2) if sigma2 > 0 else texture

        mask2 = (depth_map == 7)
        mask1 = ~mask2

        stack[z] = layer1 * mask1 + layer2 * mask2

    return stack, depth_map

def test_feature_preservation():
    """
    Verifies that best_focus_image preserves features of moderate size.
    A feature of size 60x60 in a 200x200 image corresponds to approx 4x4 patches (grid 14x14).
    A 7x7 median filter (disk(3)) would erase this feature (16 pixels < 25 threshold).
    A 3x3 median filter (disk(1)) should preserve it.
    """
    stack, truth_map = generate_feature_stack(shape=(10, 200, 200), feature_size=60)

    # Run reconstruction
    fused, height_map = best_focus_image(stack, return_heightmap=True)

    # Check if the feature is preserved.
    # We check the max depth value in the center region.
    mid_h, mid_w = 100, 100
    # Average depth in the feature center
    center_depth = np.mean(height_map[mid_h-10:mid_h+10, mid_w-10:mid_w+10])

    print(f"Center Depth: {center_depth:.2f} (Expected ~7)")

    # If erased, center_depth would be close to 2.
    # If preserved, center_depth should be close to 7.

    assert center_depth > 5.0, \
        f"Feature erased! Center depth {center_depth:.2f} is too close to background (2). Over-smoothing detected."

    # Also check MSE globally
    mse = np.mean((height_map - truth_map)**2)
    print(f"MSE: {mse:.4f}")
    # MSE should be reasonable
    assert mse < 4.0

def test_focal_plane_indices():
    """
    Verifies that the returned height map contains valid indices within [0, Z-1].
    """
    stack, _ = generate_feature_stack(shape=(5, 100, 100))
    fused, height_map = best_focus_image(stack, return_heightmap=True)

    assert height_map.min() >= 0
    assert height_map.max() < 5
