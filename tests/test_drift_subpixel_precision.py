import unittest
import numpy as np
from scipy.ndimage import center_of_mass
from eigenp_utils.maxproj_registration import apply_drift_correction_2D

class TestDriftSubpixelPrecision(unittest.TestCase):
    """
    Testr 🔎 Verification: Subpixel Drift Correction Stability

    This test verifies that the 'subpixel' mode of the drift correction algorithm
    provides significantly higher alignment precision than the default 'integer' mode.

    The Invariants:
    1. Subpixel Stability: For an object drifting at non-integer rates (e.g., 0.33 px/frame),
       subpixel correction should stabilize the centroid to within 0.1 pixels.
       (Integer correction is expected to have ~0.3 px RMSE due to quantization).
    2. Cycle Consistency (Identity): If an object is stationary (drift=0), the
       correction should not alter the image (identity transform).
    """

    def generate_drifting_stack(self, n_frames=30, drift_rate=(0.33, 0.33), size=128):
        """
        Generates a stack of frames with a drifting 2D Gaussian.
        Using size=128 to ensure the object stays within the central flat region of the windowing function.
        (Window overlap is size//3. For 64, flat region is small. For 128, it's ~42px wide).
        """
        frames = np.zeros((n_frames, size, size), dtype=np.float32)

        # Initial Center
        cy, cx = size / 2.0, size / 2.0

        y, x = np.mgrid[0:size, 0:size]

        ground_truth_positions = []

        for t in range(n_frames):
            # Calculate current position
            # Object moves by +drift
            curr_cy = cy + drift_rate[0] * t
            curr_cx = cx + drift_rate[1] * t

            ground_truth_positions.append((curr_cy, curr_cx))

            # Generate Gaussian
            sigma = 3.0
            gaussian = np.exp(-((y - curr_cy)**2 + (x - curr_cx)**2) / (2 * sigma**2))
            frames[t] = gaussian

        return frames, np.array(ground_truth_positions)

    def measure_stability_rmse(self, video_data):
        """
        Calculates the RMSE of the centroid position relative to the mean centroid.
        Lower RMSE = Better Stability.
        """
        centroids = []
        for t in range(video_data.shape[0]):
            c = center_of_mass(video_data[t])
            centroids.append(c)
        centroids = np.array(centroids)

        # Calculate deviations from the mean position (target is stability)
        mean_pos = np.mean(centroids, axis=0)
        deviations = centroids - mean_pos

        # RMSE distance
        rmse = np.sqrt(np.mean(np.sum(deviations**2, axis=1)))
        return rmse

    def test_subpixel_vs_integer_precision(self):
        """
        Verifies that subpixel correction reduces jitter by >5x compared to integer correction
        for non-integer drifts.
        """
        # 1. Setup
        # Drift rate 0.33 is worst-case for integer snapping (accumulates error)
        drift_rate = (0.33, 0.33)
        video, _ = self.generate_drifting_stack(n_frames=30, drift_rate=drift_rate, size=128)

        # 2. Run Integer Correction (Baseline)
        corrected_int, _ = apply_drift_correction_2D(
            video.copy(),
            method='integer',
            save_drift_table=False
        )
        rmse_int = self.measure_stability_rmse(corrected_int)

        # 3. Run Subpixel Correction (Test Subject)
        corrected_sub, _ = apply_drift_correction_2D(
            video.copy(),
            method='subpixel',
            save_drift_table=False
        )
        rmse_sub = self.measure_stability_rmse(corrected_sub)

        print(f"\nStability Comparison (RMSE):")
        print(f"Integer Correction: {rmse_int:.4f} pixels")
        print(f"Subpixel Correction: {rmse_sub:.4f} pixels")
        print(f"Improvement Factor: {rmse_int / rmse_sub:.2f}x")

        # 4. Assertions
        # Integer RMSE should be roughly 0.3 (std dev of uniform[-0.5, 0.5] is ~0.29)
        # Subpixel RMSE should be very low (< 0.1)

        self.assertLess(rmse_sub, 0.1, "Subpixel correction failed to stabilize image (< 0.1 px)")
        self.assertLess(rmse_sub, 0.2 * rmse_int, "Subpixel correction not significantly better than integer")

    def test_cycle_consistency_stationary(self):
        """
        Verifies that running correction on a stationary object results in Identity
        (no shifts, no blur).
        """
        drift_rate = (0.0, 0.0)
        video, _ = self.generate_drifting_stack(n_frames=10, drift_rate=drift_rate, size=64)

        # Subpixel method uses interpolation, so we check for degradation
        corrected, table = apply_drift_correction_2D(video.copy(), method='subpixel')

        # Check detected drift
        max_cum_drift = np.max(np.abs(table[['cum_dx', 'cum_dy']].values))
        print(f"\nStationary Drift Detected: {max_cum_drift:.4f}")
        self.assertLess(max_cum_drift, 0.05, "False drift detected on stationary object")

        # Check image fidelity (MSE)
        mse = np.mean((corrected - video)**2)
        print(f"Stationary Reconstruction MSE: {mse:.6f}")
        self.assertLess(mse, 1e-6, "Subpixel correction degraded stationary image")

if __name__ == '__main__':
    unittest.main()
