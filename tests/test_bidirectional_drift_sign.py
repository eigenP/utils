
import unittest
import numpy as np
from scipy.ndimage import center_of_mass
from eigenp_utils.maxproj_registration import apply_drift_correction_2D

class TestBidirectionalDriftSign(unittest.TestCase):
    """
    Testr 🔎 Verification: Bidirectional Drift Correction Sign

    This test verifies that the bidirectional drift estimation (`reverse_time='both'`)
    correctly identifies the direction of motion and applies a counter-shift.

    A regression was identified where `reverse_time='both'` averaged forward and backward
    drifts with the wrong sign, leading to positive feedback (doubling the drift)
    instead of negative feedback (correcting it).

    Guarantee:
    If object moves +D, correction must be -D.
    """

    def generate_drifting_blob(self, n_frames=10, size=64, drift=(0.0, 1.0)):
        """
        Generates a video with a Gaussian object drifting in +X.
        """
        frames = np.zeros((n_frames, size, size), dtype=np.float32)
        cy, cx = size / 2.0, size / 2.0

        y, x = np.mgrid[0:size, 0:size]

        for t in range(n_frames):
            curr_cy = cy + drift[0] * t
            curr_cx = cx + drift[1] * t

            sigma = 3.0
            gauss = np.exp(-((y - curr_cy)**2 + (x - curr_cx)**2) / (2 * sigma**2))
            frames[t] = gauss

        return frames

    def test_correction_opposes_drift(self):
        # 1. Setup: Drift +1.0 px/frame in X
        # Total drift over 9 steps: +9.0 px.
        drift_rate = (0.0, 1.0)
        n_frames = 10
        video = self.generate_drifting_blob(n_frames=n_frames, drift=drift_rate)

        # 2. Run Bidirectional Correction
        corrected, table = apply_drift_correction_2D(video, reverse_time='both', save_drift_table=False)

        # 3. Analyze Results
        # Check drift table accumulation
        final_cum_dx = table['cum_dx'].iloc[-1]

        print(f"\nDrift Analysis (Bidirectional):")
        print(f"Motion: +X direction")
        print(f"Final Estimated Cumulative Drift (Correction): {final_cum_dx:.4f}")

        # If object moves +X, correction (cum_dx) should be Negative (-X)
        # Expected approx -9.0.
        # Allow some underestimation (e.g. -7.0) due to windowing bias,
        # but ensure it is clearly negative and substantial.
        self.assertLess(final_cum_dx, -5.0,
            f"Correction should be negative (opposing drift). Got {final_cum_dx}.")

        # 4. Check Image Stability
        # Calculate centroid of last corrected frame.
        # Should be near center (32, 32).

        c_last = center_of_mass(corrected[-1])
        c_first = center_of_mass(corrected[0])

        displacement = c_last[1] - c_first[1] # X displacement

        print(f"Residual Displacement in Corrected Video: {displacement:.4f} px")

        # With -7.6 correction vs +9.0 motion, residual is +1.4.
        # This confirms we didn't DOUBLE the drift (residual would be +18).
        self.assertLess(np.abs(displacement), 2.5,
            "Corrected video drift is too high. Correction direction or magnitude failed.")

if __name__ == '__main__':
    unittest.main()
