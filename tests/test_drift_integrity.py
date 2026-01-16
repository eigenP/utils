# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-image",
#     "tqdm",
# ]
# ///

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import unittest
from eigenp_utils.maxproj_registration import apply_drift_correction_2D

class TestDriftIntegrity(unittest.TestCase):
    """
    Testr 🔎 Verification: Drift Integrity

    This test verifies that the drift correction algorithm correctly accumulates
    fractional drift over time without suffering from integrator windup (precision loss)
    due to premature integer casting.

    Mathematical Guarantee:
    If drift d_t is estimated at each step t, the total correction at step T
    must be approx sum(d_t from 0 to T).

    If the implementation casts d_t + cum_d_{t-1} to integer at each step,
    fractional parts are discarded, leading to O(T) error accumulation.

    Correct implementation should accumulate in float and only cast for the final shift application.
    """

    def test_fractional_drift_accumulation(self):
        # Create dummy video (Time=20, X=10, Y=10)
        video = np.zeros((20, 10, 10), dtype=np.uint8)

        # We simulate a constant drift of 0.5 pixels per frame.
        # Over 19 intervals (frames 1 to 19), total drift should be 9.5 pixels.

        # When reverse_time=False (default), the code calls phase_cross_correlation
        # exactly TWICE per time point (once for X, once for Y).
        # Shift is calculated as (shift_x[0], shift_y[0]).

        # If the object moves +0.5 pixels/frame, the shift required to register
        # frame t to t-1 is -0.5.

        ret_x = (np.array([0.5]), 0, 0) # Use 0.5 to signify positive accumulation if we want positive cum_dx
        ret_y = (np.array([0.0]), 0, 0)

        # Note: The sign of accumulation depends on the code.
        # dx, dy = shift_x[0], shift_y[0]
        # dx, dy = dx * DRIFT_SIGN (1)
        # cum_dx += dx.
        # So if we return 0.5, cum_dx increases by 0.5.

        side_effect = []
        # Loop runs for range(1, 20) -> 19 iterations
        for _ in range(19):
            side_effect.append(ret_x)
            side_effect.append(ret_y)

        with patch('eigenp_utils.maxproj_registration.phase_cross_correlation', side_effect=side_effect):
            corrected, drift_table = apply_drift_correction_2D(video, save_drift_table=False)

        last_row = drift_table.iloc[-1]
        final_cum_dx = last_row['cum_dx']
        final_cum_dy = last_row['cum_dy']

        print(f"\nDrift Table Tail:\n{drift_table.tail(3)}")
        print(f"Final Accumulation: dx={final_cum_dx}, dy={final_cum_dy}")

        # Expected: 9.5 (19 * 0.5)

        # If the bug was present (int casting), 0.5 would be truncated to 0 at each step (if int(0.5)=0)
        # or 1 (if int(0.5+prev) jumps).
        # Actually int(0.5) = 0. So it would stay 0.

        self.assertGreater(final_cum_dx, 9.0, "Drift accumulation was lost! Likely due to integer casting.")
        self.assertAlmostEqual(final_cum_dx, 9.5, delta=1e-5, msg="Drift should accumulate to exactly 9.5 with float precision")

if __name__ == '__main__':
    unittest.main()
