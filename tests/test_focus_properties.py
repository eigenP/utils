import unittest
import numpy as np
from eigenp_utils.extended_depth_of_focus import best_focus_image

class TestFocusProperties(unittest.TestCase):
    """
    Testr 🔎 Verification: Focus Stacking Invariants

    This test verifies fundamental mathematical properties of the Extended Depth of Focus (EDoF)
    algorithm, specifically 'Identity Reproduction' and 'Partition of Unity'.

    Property 1: Identity Reproduction
    If the input stack consists of identical slices (S(z) = Image I), the output
    must be exactly I, regardless of the focus metric or patch blending logic.

    Property 2: Boundary Integrity (Partition of Unity)
    The blending weights used to fuse patches must sum to 1.0 everywhere.
    A common failure mode is signal loss at image boundaries due to
    unconditional tapering (windowing) of edge patches.
    """

    def test_identity_reproduction_at_boundaries(self):
        """
        Verifies that a uniform white stack results in a uniform white image,
        ensuring no signal is lost at the boundaries due to weighting artifacts.
        """
        # 1. Setup
        # Create a stack of 3 slices, 64x64, all 1.0 (float)
        # We use a value of 100.0 to make relative errors obvious/easy to reason about.
        val = 100.0
        shape = (3, 64, 64)
        stack = np.full(shape, val, dtype=np.float32)

        # 2. Run Algorithm
        # We use default patch size (which would be 64//10 = 6).
        # Overlap will be 6//3 = 2.
        # This ensures we have multiple patches and boundaries are relevant.
        result = best_focus_image(stack)

        # 3. Verification

        # A) Global Conservation
        # Mean should be 100.0
        mean_val = np.mean(result)
        print(f"Mean Value: {mean_val:.4f} (Expected {val})")
        self.assertAlmostEqual(mean_val, val, delta=0.5,
            msg=f"Global signal loss detected! Mean is {mean_val} instead of {val}")

        # B) Boundary Integrity (The Critical Check)
        # We check the minimum value in the image.
        # If boundary tapering is buggy, pixels at the edge will be 0 or < 100.
        min_val = np.min(result)
        print(f"Min Value: {min_val:.4f} (Expected {val})")

        # Visualizing the artifact if present
        if min_val < val * 0.9:
            print("Low values detected! Top-Left corner slice:")
            print(result[:5, :5])

        # We allow tiny floating point error (e.g. 1e-5), but not massive signal loss.
        self.assertTrue(np.allclose(result, val, atol=1e-3),
            f"Identity reproduction failed! Min value is {min_val} (should be {val}). "
            "This likely indicates incorrect weight tapering at image boundaries.")

    def test_partition_of_unity_via_linear_ramp(self):
        """
        Verifies that a spatially varying signal (linear ramp) is perfectly reconstructed.
        If weights don't sum to 1, the ramp will be distorted.
        """
        # Stack where every slice is the same Linear Ramp
        H, W = 64, 64
        x = np.linspace(0, 100, W)
        y = np.linspace(0, 100, H)
        xv, yv = np.meshgrid(x, y)
        ramp = (xv + yv).astype(np.float32) # Range 0 to 200

        stack = np.array([ramp, ramp, ramp])

        result = best_focus_image(stack)

        # Compare
        diff = np.abs(result - ramp)
        max_diff = np.max(diff)

        print(f"Max Ramp Reconstruction Error: {max_diff:.4f}")

        self.assertLess(max_diff, 0.1,
            "Ramp reconstruction failed. Partition of Unity violated?")

if __name__ == '__main__':
    unittest.main()
