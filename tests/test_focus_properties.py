import numpy as np
import pytest
import scipy.ndimage as ndi
from eigenp_utils.extended_depth_of_focus import best_focus_image

def test_partition_of_unity():
    """
    Testr Verification: Partition of Unity / Constant Normalization

    If the input stack is constant (1.0) everywhere, the output must be exactly 1.0.
    This verifies that the weighted blending and subsequent normalization by 'counts'
    correctly reconstructs the signal without gain or loss.
    """
    H, W = 100, 100
    stack = np.ones((3, H, W), dtype=np.float32)

    # We use a patch size that isn't a perfect divisor to trigger boundary logic
    patch_size = 30

    result = best_focus_image(stack, patch_size=patch_size)

    # Check bounds
    min_val = result.min()
    max_val = result.max()

    print(f"Partition of Unity: Range [{min_val}, {max_val}]")

    # We expect floating point precision errors, but very small
    assert np.allclose(result, 1.0, atol=1e-5), \
        f"Partition of Unity failed. Range: [{min_val}, {max_val}]"

def test_identity_reproduction():
    """
    Testr Verification: Identity Reproduction (Lossless Blending)

    If every slice in the stack is identical, the focus metric (Laplacian)
    is identical for all slices (0 difference). The selection might be arbitrary (usually index 0),
    but the reconstruction should result in exactly the input image.

    This verifies that the patch extraction, weighting, accumulation, and normalization
    pipeline is mathematically lossless for a flat field (single image).
    """
    H, W = 128, 128
    np.random.seed(42)
    # Random noise texture
    img = np.random.uniform(0, 100, (H, W)).astype(np.float32)

    # Stack of identical images
    stack = np.stack([img, img, img])

    result = best_focus_image(stack, patch_size=32)

    diff = np.abs(result - img)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"Identity Reproduction: Max Diff {max_diff}, Mean Diff {mean_diff}")

    # Should be effectively zero
    assert np.allclose(result, img, atol=1e-4), \
        f"Identity Reproduction failed. Max diff: {max_diff}"


def test_translation_equivariance():
    """
    Testr Verification: Translation Equivariance (Grid Stability)

    A patch-based focus algorithm imposes a fixed grid on the image.
    If we shift the image, the grid relative to the features changes.
    Ideally, the result should simply be the shifted version of the original result.

    Large deviations indicate that the reconstruction is highly sensitive to
    where features fall relative to patch boundaries (Grid Artifacts).
    """
    H, W = 200, 200
    np.random.seed(1337)

    # Create a background with noise
    background = np.random.uniform(0, 10, (H, W)).astype(np.float32)

    # Create a sharp feature (high frequency) at a specific location
    feature = np.zeros((H, W), dtype=np.float32)
    cy, cx = 100, 100
    # A sharp box
    feature[cy-20:cy+20, cx-20:cx+20] = 100.0
    # Add noise to feature to ensure it has texture for Laplacian
    feature[cy-20:cy+20, cx-20:cx+20] += np.random.uniform(0, 50, (40, 40))

    # Stack: Slice 0 is background, Slice 1 has the feature
    # To make it challenging, we blur Slice 0's feature area so Slice 1 is definitely sharper there
    slice0 = background.copy()
    slice0[cy-20:cy+20, cx-20:cx+20] = ndi.gaussian_filter(feature[cy-20:cy+20, cx-20:cx+20], sigma=5)

    slice1 = background.copy()
    # Add feature to slice 1
    slice1 = np.maximum(slice1, feature)

    stack = np.stack([slice0, slice1])

    # 1. Compute Focus on Original
    res1 = best_focus_image(stack, patch_size=32)

    # 2. Shift the Input Stack
    dy, dx = 5, 5 # Shift by partial patch amount (32//6 approx)
    stack_shifted = np.zeros_like(stack)
    for z in range(stack.shape[0]):
        stack_shifted[z] = ndi.shift(stack[z], (dy, dx), mode='reflect')

    # 3. Compute Focus on Shifted
    res2 = best_focus_image(stack_shifted, patch_size=32)

    # 4. Shift the Result 1 to match
    res1_shifted = ndi.shift(res1, (dy, dx), mode='reflect')

    # Compare in the central region to avoid boundary effects of the shift
    margin = 20
    diff = np.abs(res2 - res1_shifted)
    central_diff = diff[margin:-margin, margin:-margin]

    mae = np.mean(central_diff)
    rmse = np.sqrt(np.mean(central_diff**2))

    print(f"Translation Equivariance: MAE={mae:.4f}, RMSE={rmse:.4f}")

    # The error won't be zero because the grid alignment changed decisions slightly,
    # but it should be small compared to the feature contrast (~100).
    # We expect RMSE < 5.0 (5% of signal range)
    assert rmse < 5.0, f"Grid artifacts are too high. RMSE={rmse:.4f}"


def test_permutation_invariance():
    """
    Testr Verification: Permutation Invariance (Order Independence)

    The order of slices in the stack (Z-order) should not affect the final pixel values,
    only the indices in the height map.
    If 'best_focus_image' is biased (e.g. prefers lower Z when scores are tied,
    or processes Z sequentially with leakage), the output intensities might change.
    """
    H, W = 64, 64
    np.random.seed(99)

    # Two textures with different intensities
    t1 = np.random.uniform(0, 50, (H, W)).astype(np.float32)
    t2 = np.random.uniform(50, 100, (H, W)).astype(np.float32)

    stack1 = np.stack([t1, t2])
    stack2 = np.stack([t2, t1])

    res1, hmap1 = best_focus_image(stack1, patch_size=16, return_heightmap=True)
    res2, hmap2 = best_focus_image(stack2, patch_size=16, return_heightmap=True)

    # Intensities should be identical
    diff = np.abs(res1 - res2)
    max_diff = diff.max()

    print(f"Permutation Invariance: Max Diff {max_diff}")
    assert max_diff < 1e-4, "Output intensity depends on Z-order!"

    sum_hmap = hmap1 + hmap2

    # With random float noise, ties are extremely unlikely.
    assert np.allclose(sum_hmap, 1), "Height maps are not complementary!"
