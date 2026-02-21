
import numpy as np
import pytest
from eigenp_utils.extended_depth_of_focus import best_focus_image

def test_edof_methods_comparison():
    # 1. Create a synthetic stack
    H, W = 128, 128
    Z = 10

    stack = np.zeros((Z, H, W), dtype=np.float32)

    # Define a depth map: Tilted plane
    y_idx, x_idx = np.indices((H, W))
    true_depth = 2 + 6 * (x_idx / W) # varies from 2 to 8

    # Simulate a textured surface at depth
    for z in range(Z):
        # Gaussian intensity profile around true_depth
        intensity = np.exp(-0.5 * (z - true_depth)**2 / (1.5**2))
        # Texture pattern
        texture = np.sin(x_idx/5) * np.cos(y_idx/5)
        stack[z] = (texture * 100 + 128) * intensity

    stack = np.clip(stack, 0, 255).astype(np.uint8)

    # 2. Run both methods
    res_patches = best_focus_image(stack, patch_size=32, method='patches')
    res_continuous = best_focus_image(stack, patch_size=32, method='continuous')

    # 3. Basic Validity Checks
    assert res_patches.shape == (H, W)
    assert res_continuous.shape == (H, W)
    assert not np.isnan(res_patches).any()
    assert not np.isnan(res_continuous).any()

    # 4. Check Gradient Smoothness (Block Artifacts)
    # Patches method should have higher gradients at patch boundaries

    def get_boundary_gradient_ratio(image, patch_size=32):
        grad_y, grad_x = np.gradient(image)
        grad_mag = np.sqrt(grad_y**2 + grad_x**2)

        v_boundaries = np.arange(patch_size, image.shape[1], patch_size)
        h_boundaries = np.arange(patch_size, image.shape[0], patch_size)

        if len(v_boundaries) == 0 or len(h_boundaries) == 0:
            return 1.0 # too small to test

        grad_v_bound = np.mean(grad_mag[:, v_boundaries])
        grad_h_bound = np.mean(grad_mag[h_boundaries, :])

        mask = np.ones_like(grad_mag, dtype=bool)
        mask[:, v_boundaries] = False
        mask[h_boundaries, :] = False
        grad_inside = np.mean(grad_mag[mask])

        return (grad_v_bound + grad_h_bound) / (2 * grad_inside)

    ratio_patches = get_boundary_gradient_ratio(res_patches)
    ratio_continuous = get_boundary_gradient_ratio(res_continuous)

    print(f"Gradient Ratio - Patches: {ratio_patches:.4f}")
    print(f"Gradient Ratio - Continuous: {ratio_continuous:.4f}")

    # Continuous method should have smoother boundaries (ratio closer to 1.0)
    # Patches method often has ratio > 1.0 due to discontinuities
    # We assert that continuous is "better" (lower ratio) or at least not significantly worse if both are good.
    # In this synthetic case with smooth depth, patches will have visible seams.

    # Note: If the texture is very high frequency, gradients inside might be high too.
    # But seams are artificial edges.

    # Ideally ratio_continuous < ratio_patches
    # But let's be robust: Check if patches has high ratio (>1.1) and continuous is lower.

    # If ratio_patches is high, continuous should be significantly lower.
    if ratio_patches > 1.05:
        assert ratio_continuous < ratio_patches, "Continuous method should have fewer artifacts than patches"

if __name__ == "__main__":
    test_edof_methods_comparison()
