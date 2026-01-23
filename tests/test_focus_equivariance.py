import numpy as np
import pytest
from eigenp_utils.extended_depth_of_focus import best_focus_image

def generate_unique_focus_stack(shape=(10, 128, 128), seed=42):
    """
    Generates a stack where the focus metric (local variance) has a unique peak along Z.

    We create a "Depth Map" D(y, x) varying smoothly.
    The stack S(z, y, x) consists of noise modulated by a Gaussian centered at D(y, x).

    S(z, y, x) = Noise(y, x) * exp( - (z - D(y, x))^2 / sigma )

    This ensures that the "sharpness" (amplitude of noise) is maximized exactly at z = D(y, x).
    The 'Noise' is high-frequency enough to drive the Laplacian metric.
    """
    Z, H, W = shape
    rng = np.random.default_rng(seed)

    # Base high-frequency texture (white noise)
    texture = rng.uniform(0, 1, (H, W)).astype(np.float32)

    # Depth map: A tilted plane or sine wave to cover all Zs
    y_coords, x_coords = np.mgrid[0:H, 0:W]

    # Oscillating depth map from 1 to Z-2 (to avoid edge boundary effects being dominant)
    depth_map = (Z/2) + (Z/3) * np.sin(4 * np.pi * x_coords / W) * np.cos(4 * np.pi * y_coords / H)
    depth_map = np.clip(depth_map, 0, Z-1)

    stack = np.zeros(shape, dtype=np.float32)
    sigma = 1.5 # Width of the focus peak

    for z in range(Z):
        # Gaussian weight for this Z
        weight = np.exp( - (z - depth_map)**2 / (2 * sigma**2) )
        # Add some base intensity to verify affine properties later
        stack[z] = 10.0 + 50.0 * texture * weight

    return stack

def test_z_flip_invariance():
    """
    Testr Verification: Z-Flip Invariance

    If we reverse the stack order (Z-flip), the physical selection of "best pixels"
    should remain identical (because the sharpest plane is the same physical plane).

    Mathematically:
    Let S be the stack. S_rev(z) = S(N-1-z).
    If BestZ(S) at (y,x) is k, then BestZ(S_rev) at (y,x) should be N-1-k.
    The resulting pixel value S[k] == S_rev[N-1-k].

    This verifies:
    1. No "first vs last" bias in argmax (due to unique peak).
    2. No off-by-one errors in loop indexing.
    3. Correct handling of padding/reflection in Z-dependent logic (if any).
    """
    stack = generate_unique_focus_stack(shape=(12, 64, 64))

    # Forward run
    img_fwd = best_focus_image(stack)

    # Reverse run
    stack_rev = stack[::-1].copy() # Ensure contiguous
    img_rev = best_focus_image(stack_rev)

    # Verify
    # We use a slightly generous tolerance because float32 accumulations in different orders
    # might differ slightly (though here it's selection + local blend, order should imply same ops).
    # But wait, median filter on index map:
    # Median([k, ...]) vs Median([N-1-k, ...])
    # Median(N-1 - X) = N-1 - Median(X). This holds for odd windows.
    # So the selected indices should map perfectly.

    np.testing.assert_allclose(img_fwd, img_rev, rtol=1e-5, atol=1e-5,
                               err_msg="Focus fusion is not invariant to Z-stack reversal.")

def test_affine_intensity_equivariance():
    """
    Testr Verification: Affine Intensity Equivariance

    The selection of the focal plane should depend on *contrast* (relative changes),
    not absolute intensity.

    f(alpha * S + beta) approx alpha * f(S) + beta

    Strictly:
    Laplacian is linear: L(aS+b) = a L(S).
    Energy: (a L)^2 = a^2 L^2.
    Argmax(a^2 E) == Argmax(E) (for a != 0).
    So the *index map* should be identical.

    The reconstruction is a weighted average:
    Sum(w * (a I + b)) / Sum(w) = a * (Sum(w I)/Sum(w)) + b * (Sum(w)/Sum(w))
    = a * Result + b * 1.

    So it should be *exactly* equivariant (within float precision).
    """
    stack = generate_unique_focus_stack(shape=(8, 64, 64))

    alpha = 2.5
    beta = 100.0

    stack_trans = stack * alpha + beta

    img_orig = best_focus_image(stack)
    img_trans = best_focus_image(stack_trans)

    img_expected = img_orig * alpha + beta

    np.testing.assert_allclose(img_trans, img_expected, rtol=1e-4, atol=1e-4,
                               err_msg="Focus fusion is not equivariant to affine intensity changes.")

def test_transpose_invariance():
    """
    Testr Verification: XY Transpose Invariance

    Verifies that X and Y dimensions are treated symmetrically.

    f(S.T) == f(S).T

    This checks for bugs where 'pad_x' is used for Y, or patch grids are calculated asymmetrically.
    """
    # Use non-square shape to catch dimension swapping bugs
    stack = generate_unique_focus_stack(shape=(8, 96, 64))

    img_orig = best_focus_image(stack)

    # Transpose stack: (Z, H, W) -> (Z, W, H)
    stack_T = stack.transpose(0, 2, 1).copy()

    img_T = best_focus_image(stack_T)

    np.testing.assert_allclose(img_T, img_orig.T, rtol=1e-5, atol=1e-5,
                               err_msg="Focus fusion is not invariant to XY transposition.")
