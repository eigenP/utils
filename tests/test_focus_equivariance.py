
import pytest
import numpy as np
from eigenp_utils.extended_depth_of_focus import best_focus_image

def test_z_flip_invariance():
    """
    Testr 🔎: Verify Z-Flip Symmetry (Time Reversal Invariance).

    If we reverse the order of slices in the Z-stack, the resulting focused image
    should be identical (pixel-perfect), and the height map should be exactly inverted.

    This guarantees that the algorithm treats the Z-dimension purely as a search space
    and has no directional bias (e.g. favoring earlier slices in ties).
    """
    # Setup
    Z, H, W = 10, 64, 64
    rng = np.random.default_rng(42)
    # Use float32 to match internal precision
    stack = rng.random((Z, H, W)).astype(np.float32) * 100.0

    # Run Original
    img_orig, hmap_orig = best_focus_image(stack, patch_size=16, return_heightmap=True)

    # Run Flipped
    stack_flipped = stack[::-1].copy()
    img_flip, hmap_flip = best_focus_image(stack_flipped, patch_size=16, return_heightmap=True)

    # Verify Image Identity
    # Tolerance due to potential accumulation order differences in summation?
    # float32 accumulation order is not associative.
    # However, the patching order in XY is the same. The Z-search finds an index.
    # If the index is correct, the patch extraction is identical.
    # The blending is XY-based.
    # So it should be exact or very close.
    np.testing.assert_allclose(img_flip, img_orig, rtol=1e-5, atol=1e-5,
        err_msg="Flipping Z-stack changed the output image.")

    # Verify Height Map Inversion
    # hmap_flip should be (Z - 1) - hmap_orig
    expected_hmap = (Z - 1) - hmap_orig

    # Check for mismatches
    mismatches = np.sum(hmap_flip != expected_hmap)
    total_pixels = H * W
    mismatch_rate = mismatches / total_pixels

    # We allow a tiny fraction of mismatches due to ties in Laplacian energy
    # where index selection might be stable (first max) and thus differ on reversal.
    # But with continuous random noise, ties are virtually impossible.
    assert mismatch_rate == 0.0, f"Height map did not invert correctly. Mismatch rate: {mismatch_rate:.2%}"

def test_affine_intensity_equivariance():
    """
    Testr 🔎: Verify Affine Intensity Equivariance.

    Transform: I' = a * I + b
    Expected:  Out' = a * Out + b

    This verifies that the focus metric (Laplacian) and blending (linear)
    preserve the linear photometric relationships.
    """
    Z, H, W = 5, 50, 50
    rng = np.random.default_rng(123)
    stack = rng.random((Z, H, W)).astype(np.float32)

    a, b = 5.5, 10.0
    stack_prime = a * stack + b

    # Run
    img, _ = best_focus_image(stack, patch_size=16, return_heightmap=True)
    img_prime, _ = best_focus_image(stack_prime, patch_size=16, return_heightmap=True)

    # Predict
    img_predicted = a * img + b

    np.testing.assert_allclose(img_prime, img_predicted, rtol=1e-4, atol=1e-4,
        err_msg="Affine transformation of input did not result in affine transformation of output.")

def test_transpose_invariance():
    """
    Testr 🔎: Verify XY Transpose Equivariance (Geometric Isometry).

    If we swap X and Y axes of the input stack, the output image
    should be the transpose of the original output.

    This is a critical regression test for:
    1. Hardcoded dimension indexing (using shape[0] for width, etc).
    2. Asymmetric padding logic.
    3. Asymmetric kernel application.
    """
    # Use rectangular dimensions to catch index swaps
    Z, H, W = 5, 64, 96
    patch_size = 24 # Overlap ~8.
    # 64: ~3 patches. 96: ~4-5 patches.

    rng = np.random.default_rng(999)
    stack = rng.random((Z, H, W)).astype(np.float32)

    # Original
    img, hmap = best_focus_image(stack, patch_size=patch_size, return_heightmap=True)

    # Transposed Input: (Z, W, H)
    stack_T = stack.transpose(0, 2, 1).copy()

    # Run on Transposed
    img_T, hmap_T = best_focus_image(stack_T, patch_size=patch_size, return_heightmap=True)

    # Verify Structure
    assert img.shape == (H, W)
    assert img_T.shape == (W, H)

    # Compare
    # img_T should be equal to img.T
    np.testing.assert_allclose(img_T, img.T, rtol=1e-5, atol=1e-5,
        err_msg="Output of transposed stack is not the transpose of the output.")

    # Verify Height Map Transpose
    np.testing.assert_array_equal(hmap_T, hmap.T,
        err_msg="Height map of transposed stack is not the transpose of the height map.")
