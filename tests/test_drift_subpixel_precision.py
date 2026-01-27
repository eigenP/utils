# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "pandas",
#     "scikit-image",
#     "tqdm",
# ]
# ///

import unittest
import numpy as np
from scipy.ndimage import center_of_mass, shift
from eigenp_utils.maxproj_registration import apply_drift_correction_2D

class TestDriftSubpixelPrecision(unittest.TestCase):
    """
    Testr 🔎 Verification: Subpixel Drift Correction Precision

    This test verifies that the 'subpixel' mode of drift correction provides
    mathematically superior stability compared to 'integer' mode.

    It validates:
    1. Precision: Subpixel correction reduces centroid jitter to < 0.1 pixels
       (approaching interpolation limits), whereas integer correction is bounded by 0.5 pixels.
    2. Accumulation: Drift accumulation retains floating point precision.
    3. Inverse Consistency: Correcting a known shift restores the image to its original state (minimized residuals).
    """

    def generate_moving_gaussian(self, shape=(20, 64, 64), drift_per_frame=(0.33, 0.66), sigma=3.0):
        """
        Generates a synthetic video of a drifting 2D Gaussian.
        drift_per_frame: (dy, dx)
        """
        T, H, W = shape
        video = np.zeros(shape, dtype=np.float32)

        # Base grid
        y = np.arange(H)
        x = np.arange(W)
        yy, xx = np.meshgrid(y, x, indexing='ij')

        # Center
        cy, cx = H // 2, W // 2
        dy, dx = drift_per_frame

        gt_positions = []

        for t in range(T):
            curr_cy = cy + dy * t
            curr_cx = cx + dx * t
            gt_positions.append((curr_cy, curr_cx))

            # G = exp( -((y-cy)^2 + (x-cx)^2) / (2*sigma^2) )
            gauss = np.exp(-((yy - curr_cy)**2 + (xx - curr_cx)**2) / (2 * sigma**2))
            video[t] = gauss

        return video, np.array(gt_positions)

    def test_subpixel_accuracy_improvement(self):
        """
        Verifies that 'subpixel' method yields significantly lower centroid error than 'integer'.
        """
        # Setup: Drift that is explicitly non-integer (e.g. 0.4 px/frame)
        # Over 10 frames, this accumulates to 4.0 px, but intermediate frames are at .4, .8, .2, .6...
        # Integer correction will snap these, causing "staircase" jitter.
        # Subpixel correction should smooth it out.

        drift_rate = (0.4, 0.4)
        n_frames = 15
        video, _ = self.generate_moving_gaussian(shape=(n_frames, 64, 64), drift_per_frame=drift_rate)

        # 1. Run Integer Correction
        corrected_int, _ = apply_drift_correction_2D(video, method='integer', save_drift_table=False)

        # 2. Run Subpixel Correction
        corrected_sub, _ = apply_drift_correction_2D(video, method='subpixel', save_drift_table=False)

        # 3. Calculate Centroid Stability (Standard Deviation of positions)
        def get_centroid_std(stack):
            centroids = np.array([center_of_mass(frame) for frame in stack])
            # We expect the centroid to be constant (stationary object)
            # So standard deviation represents the "jitter" or error.
            return np.std(centroids, axis=0) # (std_y, std_x)

        std_int = get_centroid_std(corrected_int)
        std_sub = get_centroid_std(corrected_sub)

        print(f"\nCentroid Jitter (Std Dev):")
        print(f"Integer:  Y={std_int[0]:.4f}, X={std_int[1]:.4f}")
        print(f"Subpixel: Y={std_sub[0]:.4f}, X={std_sub[1]:.4f}")

        # Assertions

        # Integer jitter is dominated by quantization noise (uniform [-0.5, 0.5] variance is 1/12 ~= 0.08,
        # but can be higher depending on interference patterns).
        # We expect Subpixel jitter to be much lower.

        # Metric: Improvement Factor
        improvement_y = std_int[0] / (std_sub[0] + 1e-9)
        improvement_x = std_int[1] / (std_sub[1] + 1e-9)

        print(f"Improvement Factor: Y={improvement_y:.1f}x, X={improvement_x:.1f}x")

        self.assertLess(std_sub[0], 0.05, "Subpixel Y-stability is poor (> 0.05 px)")
        self.assertLess(std_sub[1], 0.05, "Subpixel X-stability is poor (> 0.05 px)")

        # Ensure subpixel is actually better than integer (discriminative test)
        self.assertGreater(improvement_y, 2.0, "Subpixel method did not significantly outperform integer method")

    def test_cycle_consistency_exact(self):
        """
        Verifies that correcting a known shift restores the image with minimal residual.
        """
        # 1. Create a frame
        H, W = 64, 64
        y, x = np.mgrid[:H, :W]
        cy, cx = H//2, W//2
        frame0 = np.exp(-((y-cy)**2 + (x-cx)**2) / (2 * 4.0**2)).astype(np.float32)

        # 2. Create a drifted frame (shift by 0.5, 0.5)
        # Using scipy.ndimage.shift directly to create "Ground Truth" drifted image
        # Note: apply_drift_correction also uses scipy.ndimage.shift with order=3 (bicubic).
        # So we expect near-perfect inversion if the estimator finds the right shift.

        drift = (0.5, 0.5)
        frame1 = shift(frame0, drift, order=3, mode='constant', cval=0.0)

        # Construct video: Frame 0 (Ref), Frame 1 (Drifted)
        video = np.array([frame0, frame1])

        # 3. Apply Correction
        # This will estimate drift frame1->frame0 (should be ~0.5, 0.5)
        # And shift frame1 back by (-0.5, -0.5).
        corrected, table = apply_drift_correction_2D(video, method='subpixel', save_drift_table=False)

        # 4. Check Residual
        # Corrected Frame 1 should match Frame 0
        residual = np.abs(corrected[1] - frame0)
        max_residual = np.max(residual)
        mse = np.mean(residual**2)

        print(f"\nCycle Consistency MSE: {mse:.2e}, Max Diff: {max_residual:.4f}")

        # Verify Estimation Accuracy from table
        # Table accumulates drift. Frame 1 cum drift should be +drift.
        # Wait, implementation:
        # dx, dy is drift between current and previous.
        # cum is sum.
        # We drifted by +0.5. Estimator should see shift of +0.5.
        # Then correction applies -0.5.

        est_dy = table['cum_dy'].iloc[-1]
        est_dx = table['cum_dx'].iloc[-1]

        print(f"Estimated Drift: dy={est_dy:.4f}, dx={est_dx:.4f}")

        # The drift table records the CORRECTION shift needed (inverse of object motion).
        # We moved object by +0.5, so we need -0.5 to correct it.
        self.assertAlmostEqual(est_dy, -drift[0], delta=0.05, msg="Drift estimation inaccurate")
        self.assertAlmostEqual(est_dx, -drift[1], delta=0.05, msg="Drift estimation inaccurate")

        # MSE should be very low (interpolation error only)
        self.assertLess(mse, 1e-5, "Subpixel correction failed to restore original image")

if __name__ == '__main__':
    unittest.main()
