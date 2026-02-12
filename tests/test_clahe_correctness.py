import unittest
import numpy as np
import warnings
from eigenp_utils.clahe_equalize_adapthist import _my_clahe_

class TestCLAHECorrectness(unittest.TestCase):
    """
    Testr 🔎 Verification: CLAHE Correctness

    This test verifies the algorithmic intent and mathematical properties of
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Verified Properties:
    1. Monotonicity of Contrast: Increasing `clip_limit` (the contrast limit)
       must monotonically increase (or saturate) the entropy/contrast of the output.
       (Low limit -> Identity-like; High limit -> AHE-like).

    2. Identity Convergence: As `clip_limit` -> 0 (but > 0), the output should
       converge towards a linear rescaling of the input (preserving relative contrast),
       rather than the aggressive reshaping of Histogram Equalization.

    3. Monotone Zero Behavior: A `clip_limit` of 0 should correspond to
       Identity (Minimum Contrast), preserving continuity with limit->0.
    """

    def setUp(self):
        # Create a synthetic image: A low-contrast sine wave + noise
        # This gives CLAHE something to work with (expand contrast).
        # We use a float image [0, 1] range.
        np.random.seed(42)
        x = np.linspace(0, 4*np.pi, 128)
        y = np.linspace(0, 4*np.pi, 128)
        xv, yv = np.meshgrid(x, y)

        # Base signal: Sine wave (approx range [0.4, 0.6])
        signal = 0.5 + 0.1 * np.sin(xv) * np.cos(yv)

        # Add noise to ensure histograms are not degenerate (avoiding artifacts)
        noise = np.random.normal(0, 0.01, signal.shape)

        self.img = np.clip(signal + noise, 0, 1)
        self.initial_std = np.std(self.img)

    def test_contrast_monotonicity(self):
        """
        Verify that increasing clip_limit increases output contrast (StdDev).
        """
        # We verify monotonicity across a range of limits.
        # Note: saturation usually occurs around 0.05-0.10 for smooth images,
        # as the 'clip count' becomes larger than any bin count.
        limits = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        stds = []

        # Run CLAHE with increasing limits
        for lim in limits:
            # We use a fixed kernel size to isolate clip_limit effect
            # Kernel 32x32 -> 1024 pixels.
            out = _my_clahe_(self.img, clip_limit=lim, kernel_size=(32, 32))
            stds.append(np.std(out))

        print("\nCLAHE Contrast Monotonicity Check:")
        for l, s in zip(limits, stds):
            print(f"  Limit={l:.4f}, StdDev={s:.4f}")

        # 1. Check Non-Decreasing Monotonicity
        for i in range(len(stds) - 1):
            self.assertGreaterEqual(stds[i+1], stds[i] - 1e-6,
                f"Contrast dropped between limit {limits[i]} and {limits[i+1]}")

        # 2. Check Significant Increase
        # The highest limit (AHE) should produce significantly more contrast than the lowest (Identity-like)
        # We use a relaxed 1.2x factor (20% increase) to be robust to image content.
        self.assertGreater(stds[-1], stds[0] * 1.2,
            "Max limit should produce significantly higher contrast than min limit")

    def test_identity_convergence(self):
        """
        Verify that very low clip_limit preserves the original signal shape (linear scaling).
        """
        # Very low limit -> Uniform Histogram -> Identity Mapping (Linear Rescaling)
        limit = 1e-5
        out = _my_clahe_(self.img, clip_limit=limit, kernel_size=(32, 32))

        # Compute correlation between input and output
        # If it's a linear transform, correlation should be 1.0
        flat_in = self.img.ravel()
        flat_out = out.ravel()

        corr = np.corrcoef(flat_in, flat_out)[0, 1]

        print(f"\nIdentity Convergence (Limit={limit}): Correlation={corr:.6f}")

        self.assertGreater(corr, 0.99,
            "Output with near-zero clip_limit should be highly correlated with input (Linear Transform).")

    def test_zero_behavior(self):
        """
        Verify that clip_limit=0 produces Minimum Contrast (Identity),
        and clip_limit=1.0/None produces Maximum Contrast (AHE).
        """
        # Run with limit=0 (Identity)
        out_zero = _my_clahe_(self.img, clip_limit=0, kernel_size=(32, 32))
        std_zero = np.std(out_zero)

        # Run with limit=1.0 (Max Contrast explicitly)
        out_max = _my_clahe_(self.img, clip_limit=1.0, kernel_size=(32, 32))
        std_max = np.std(out_max)

        # Run with limit=None (Default/Max)
        out_none = _my_clahe_(self.img, clip_limit=None, kernel_size=(32, 32))
        std_none = np.std(out_none)

        print(f"\nZero Check: Std(0)={std_zero:.4f}, Std(1.0)={std_max:.4f}, Std(None)={std_none:.4f}")

        # Std(0) should be strictly less than Std(1.0)
        self.assertLess(std_zero, std_max,
            msg="clip_limit=0 (Identity) should have lower contrast than AHE.")

        # Std(1.0) should equal Std(None)
        self.assertAlmostEqual(std_max, std_none, delta=1e-5,
            msg="clip_limit=None should be equivalent to clip_limit=1.0 (AHE)")

        # Zero should be close to very small limit
        out_small = _my_clahe_(self.img, clip_limit=1e-5, kernel_size=(32, 32))
        std_small = np.std(out_small)
        self.assertAlmostEqual(std_zero, std_small, delta=1e-2,
            msg="clip_limit=0 should be continuous with small limits.")

if __name__ == '__main__':
    unittest.main()
