# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-image",
#     "scipy",
# ]
# ///

import unittest
import numpy as np
from scipy.ndimage import center_of_mass
from eigenp_utils.maxproj_registration import apply_drift_correction_2D

class TestDriftCorrectionAccuracy(unittest.TestCase):
    """
    Testr 🔎 Verification: Drift Correction Accuracy

    This test verifies the end-to-end correctness of the drift correction pipeline.
    It checks two key properties:
    1. Estimation Accuracy: The algorithm detects sub-pixel drift with high precision.
    2. Correction Stability: The integer-based correction stabilizes the image
       within the theoretical quantization bound (+/- 0.5 pixels).

    Unlike unit tests that mock the estimator, this tests the interaction between
    signal processing (phase correlation) and discrete actuation (integer shifting).
    """

    def generate_moving_gaussian(self, shape=(20, 64, 64), drift_per_frame=(0.4, 0.7), sigma=3.0):
        """
        Generates a synthetic video of a drifting 2D Gaussian.

        Args:
            shape: (T, Y, X)
            drift_per_frame: (dy, dx) shift per frame
            sigma: standard deviation of the Gaussian
        """
        T, H, W = shape
        video = np.zeros(shape, dtype=np.float32)

        # Grid coordinates
        y = np.arange(H)
        x = np.arange(W)
        yy, xx = np.meshgrid(y, x, indexing='ij')

        # Start at center
        cy, cx = H // 2, W // 2
        dy, dx = drift_per_frame

        ground_truth_positions = []

        for t in range(T):
            # Calculate current center
            # Note: The object moves by +drift, so we shift center by +drift
            curr_cy = cy + dy * t
            curr_cx = cx + dx * t

            ground_truth_positions.append((curr_cy, curr_cx))

            # Generate Gaussian
            # G = exp( -((y-cy)^2 + (x-cx)^2) / (2*sigma^2) )
            gauss = np.exp(-((yy - curr_cy)**2 + (xx - curr_cx)**2) / (2 * sigma**2))
            video[t] = gauss

        # Normalize to reasonable range for uint8 if needed, but float is fine for processing
        # The drift correction code handles float inputs (via _2D_weighted_image and fft)

        return video, np.array(ground_truth_positions)

    def test_subpixel_drift_recovery(self):
        """
        Verifies that sub-pixel drift is accurately estimated and accumulated.
        """
        # 1. Setup
        n_frames = 20
        # Drift: 0.3 px Y, 0.4 px X per frame
        # Total drift ~ 6px Y, 8px X
        drift_rate = (0.3, 0.4)

        video, gt_pos = self.generate_moving_gaussian(
            shape=(n_frames, 64, 64),
            drift_per_frame=drift_rate,
            sigma=2.5
        )

        # 2. Run Drift Correction
        # Note: apply_drift_correction_2D prints to stdout (tqdm), which we might ignore
        corrected_video, drift_table = apply_drift_correction_2D(video, save_drift_table=False)

        # 3. Verify Estimation (Drift Table)
        # drift_table contains 'dx', 'dy', 'cum_dx', 'cum_dy'
        # 'dx' in table is the correction shift.
        # If object moves +0.4, correction dx should be -0.4 to bring it back.
        # Check cumulative drift

        final_cum_dy = drift_table['cum_dy'].iloc[-1]
        final_cum_dx = drift_table['cum_dx'].iloc[-1]

        # Expected cumulative correction: -(N-1) * drift_rate
        # Frame 0 is Ref. Frame 1 needs -drift. Frame 19 needs -19*drift.
        expected_dy = -(n_frames - 1) * drift_rate[0]
        expected_dx = -(n_frames - 1) * drift_rate[1]

        print(f"\nEstimated Cumulative Drift: dy={final_cum_dy:.4f}, dx={final_cum_dx:.4f}")
        print(f"Expected Cumulative Drift:  dy={expected_dy:.4f}, dx={expected_dx:.4f}")

        # Tolerance: Sub-pixel estimation is usually accurate to ~0.05 px
        # Accumulated error over 20 frames might be slightly higher.
        self.assertAlmostEqual(final_cum_dy, expected_dy, delta=0.5,
                               msg="Cumulative Y drift estimation failed")
        self.assertAlmostEqual(final_cum_dx, expected_dx, delta=0.5,
                               msg="Cumulative X drift estimation failed")

        # 4. Verify Correction (Image Stability)
        # Calculate centroid of corrected video frames
        # They should all be close to the centroid of Frame 0

        centroids = []
        for t in range(n_frames):
            cy, cx = center_of_mass(corrected_video[t])
            centroids.append((cy, cx))
        centroids = np.array(centroids)

        # Target: Frame 0 position
        target_y, target_x = centroids[0]

        # Error metrics
        # Since correction is integer-only, the corrected position will snap to grid.
        # The centroid of a snapped Gaussian will oscillate around the true center.
        # Max error should be roughly 0.5 pixels.

        y_errors = np.abs(centroids[:, 0] - target_y)
        x_errors = np.abs(centroids[:, 1] - target_x)

        max_err_y = np.max(y_errors)
        max_err_x = np.max(x_errors)

        print(f"Max Centroid Jitter: Y={max_err_y:.4f} px, X={max_err_x:.4f} px")

        # We allow slightly more than 0.5 because center_of_mass on a discrete grid
        # can shift slightly if the Gaussian tails are clipped or integer shifting
        # alters the sampling symmetry. 0.7 is a safe bound for "Integer Corrected".
        self.assertLess(max_err_y, 0.7, "Corrected Y position drifts/jitters too much")
        self.assertLess(max_err_x, 0.7, "Corrected X position drifts/jitters too much")

    def test_drift_direction_sign(self):
        """
        Verifies that positive object motion results in negative correction shifts.
        This catches sign inversion bugs.
        """
        # Object moves RIGHT (+X)
        drift_rate = (0.0, 1.0)
        # Using 64x64 to avoid windowing artifacts on 32x32 images
        video, _ = self.generate_moving_gaussian(shape=(5, 64, 64), drift_per_frame=drift_rate)

        _, drift_table = apply_drift_correction_2D(video, save_drift_table=False)

        # Correction 'dx' for Frame 1 (relative to Frame 0) should be -1.0
        # cum_dx at end (Frame 4) should be approx -4.0 (4 steps)

        final_dx = drift_table['cum_dx'].iloc[-1]

        # Print for debugging
        print(f"\nDrift Table:\n{drift_table}")
        print(f"Final Cum DX: {final_dx}")

        # Should be close to -4.0
        self.assertLess(final_dx, -3.5, "Positive object motion should yield negative correction")

if __name__ == '__main__':
    unittest.main()
