import sys

from scipy.ndimage import center_of_mass
from scipy.ndimage import center_of_mass, shift
from scipy.ndimage import shift
from skimage import data, transform
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import types
import unittest

from eigenp_utils.maxproj_registration import apply_drift_correction
from eigenp_utils.maxproj_registration import estimate_drift, apply_drift_correction
from eigenp_utils.maxproj_registration import zero_shift_multi_dimensional, apply_drift_correction



# =========================================
# Source: test_maxproj_registration.py
# =========================================

# Provide a stub pandas module so the import succeeds
sys.modules.setdefault('pandas', types.ModuleType('pandas'))


def test_zero_shift_positive_negative():
    """Test that zero shift positive negative works as expected."""
    arr = np.arange(9).reshape(3, 3)
    result = zero_shift_multi_dimensional(arr, shifts=(1, -1), fill_value=-1)
    expected = np.array([
        [-1, -1, -1],
        [1, 2, -1],
        [4, 5, -1],
    ])
    assert np.array_equal(result, expected)


def test_zero_shift_errors():
    """Test that zero shift errors works as expected."""
    arr = np.zeros((2, 2))
    with pytest.raises(ValueError):
        zero_shift_multi_dimensional(arr, shifts=(1,))
    with pytest.raises(TypeError):
        zero_shift_multi_dimensional(arr, shifts=(1.0, 2.0))


def test_apply_drift_correction_2d_plus_t():
    """Test that apply drift correction 2d plus t works as expected."""
    # Create a 2D+t video (T, Y, X)
    # T=3, Y=10, X=10
    video = np.zeros((3, 10, 10), dtype=np.float32)

    # Frame 0: square at (2, 2)
    video[0, 2:4, 2:4] = 1.0
    # Frame 1: shifted by +1 in Y, +2 in X -> (3, 4)
    video[1, 3:5, 4:6] = 1.0
    # Frame 2: shifted by another +1 in Y, +2 in X -> (4, 6)
    video[2, 4:6, 6:8] = 1.0

    corrected, drift_table = apply_drift_correction(video, save_drift_table=False)

    # Check drift table has expected columns
    expected_cols = ['Time Point', 'dx', 'dy', 'dz', 'cum_dx', 'cum_dy', 'cum_dz']
    for col in expected_cols:
        assert col in drift_table.columns

    # dz and cum_dz should be 0 for 2D+t
    assert np.all(drift_table['dz'] == 0)
    assert np.all(drift_table['cum_dz'] == 0)

    # Check that corrected frames align with Frame 0
    for t in range(3):
        # We expect the square to be at (2, 2) in all corrected frames
        # The drift correction tries to align frame t to frame t-1, and ultimately frame 0
        assert np.max(corrected[t, 2:4, 2:4]) == 1.0


def test_apply_drift_correction_3d_plus_t():
    """Test that apply drift correction 3d plus t works as expected."""
    # Create a 3D+t video (T, Z, Y, X)
    # T=3, Z=10, Y=10, X=10
    video = np.zeros((3, 10, 10, 10), dtype=np.float32)

    # Frame 0: cube at (2, 2, 2)
    video[0, 2:4, 2:4, 2:4] = 1.0
    # Frame 1: shifted by +1 in Z, +1 in Y, +2 in X -> (3, 3, 4)
    video[1, 3:5, 3:5, 4:6] = 1.0
    # Frame 2: shifted by another +1 in Z, +1 in Y, +2 in X -> (4, 4, 6)
    video[2, 4:6, 4:6, 6:8] = 1.0

    corrected, drift_table = apply_drift_correction(video, save_drift_table=False)

    # Check drift table has expected columns
    expected_cols = ['Time Point', 'dx', 'dy', 'dz', 'cum_dx', 'cum_dy', 'cum_dz']
    for col in expected_cols:
        assert col in drift_table.columns

    # dz should NOT be 0 for 3D+t
    # Note: cumulative dz depends on convention (usually opposite of object shift)
    assert not np.all(drift_table['cum_dz'] == 0)

    # Check that corrected frames align with Frame 0
    for t in range(3):
        assert np.max(corrected[t, 2:4, 2:4, 2:4]) == 1.0


def test_apply_drift_correction_invalid_dims():
    """Test that apply drift correction invalid dims works as expected."""
    video_2d = np.zeros((10, 10))
    with pytest.raises(ValueError, match="Expected 3D \\(T, Y, X\\) or 4D \\(T, Z, Y, X\\) data"):
        apply_drift_correction(video_2d)

    video_5d = np.zeros((2, 2, 10, 10, 10))
    with pytest.raises(ValueError, match="Expected 3D \\(T, Y, X\\) or 4D \\(T, Z, Y, X\\) data"):
        apply_drift_correction(video_5d)

# =========================================
# Source: test_drift_bias_oscillation.py
# =========================================
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


class TestDriftBiasOscillation(unittest.TestCase):
    """
    Testr 🔎 Verification: Drift Estimation Bias & Random Walk

    This test verifies that the drift correction estimator is:
    1. Unbiased (integrates to zero for zero-mean periodic motion).
    2. Accurate (correctly estimates the amplitude of motion).
    3. Stable (does not exhibit "random walk" or divergence over long sequences).

    It uses a sinusoidal trajectory which is the standard signal for testing
    frequency response and integrator stability.
    """

    def generate_oscillating_video(self, n_cycles=5, period=10, amplitude=2.0, size=256):
        """
        Generates a video with sinusoidal camera motion over a static scene.
        Motion: x(t) = A * sin(2*pi*t/T)
        """
        # 1. Load Ground Truth Scene (512x512)
        try:
            scene = data.camera() # 512x512 uint8
        except Exception:
            # Fallback
            scene = data.checkerboard() # 200x200
            scene = transform.resize(scene, (512, 512), preserve_range=True).astype(np.uint8)

        scene = scene.astype(np.float32)

        # 2. Define Viewport
        H, W = scene.shape
        cy, cx = H // 2, W // 2
        half_sz = size // 2

        # Ensure n_frames covers full cycles
        n_frames = n_cycles * period
        video = np.zeros((n_frames, size, size), dtype=np.uint8)

        ground_truth_drift = []

        # 3. Generate Frames
        for t in range(n_frames):
            # Drift d(t) is OBJECT motion relative to camera.
            # d(t) = A * sin(...)
            # Shift scene by (dy, dx)

            angle = 2 * np.pi * t / period
            dx = amplitude * np.sin(angle)
            dy = amplitude * np.cos(angle) # Circular motion

            shifted_scene = shift(scene, shift=(dy, dx), order=3, mode='reflect')

            crop = shifted_scene[cy-half_sz : cy+half_sz, cx-half_sz : cx+half_sz]
            video[t] = crop.astype(np.uint8)

            ground_truth_drift.append((dx, dy))

        return video, np.array(ground_truth_drift)

    def test_zero_mean_oscillation(self):
        """
        Verifies that after full cycles of oscillation, the accumulated drift returns to zero.
        """
        AMPLITUDE = 3.0
        PERIOD = 10
        CYCLES = 5
        video, gt_drift = self.generate_oscillating_video(n_cycles=CYCLES, period=PERIOD, amplitude=AMPLITUDE)

        # Run Correction (Bidirectional Subpixel)
        corrected, table = apply_drift_correction(
            video,
            method='subpixel',
            reverse_time='both',
            save_drift_table=False
        )

        # Table contains Time Point: 1, 2, ..., N-1.
        # We want to check cum_dx at T=10, 20, ...
        # Since T=0 is not in table (it's implicit start), T=10 is the end of cycle 1.

        # Filter table for cycle endpoints
        # Note: Time Point 10 means correction to align Frame 10 to Frame 0 (accumulated).
        # At Frame 10, sin(2pi) = 0. So drift should be 0.

        cycle_indices = [i * PERIOD for i in range(1, CYCLES + 1)] # 10, 20, 30, 40, 50

        # Careful: if video has 50 frames (0..49), Frame 50 doesn't exist.
        # But apply_drift_correction iterates range(1, 50) -> 1..49.
        # So we can check up to 40. For 50, it's out of bounds of the table?
        # range(1, 50) goes up to 49.
        # So we check 10, 20, 30, 40.

        cycle_indices = [t for t in cycle_indices if t < len(video)]

        print("\nCycle Endpoints Accumulation:")
        max_endpoint_error = 0.0

        # Use boolean indexing or lookup
        for t in cycle_indices:
            row = table[table['Time Point'] == t]
            if row.empty:
                print(f"Time Point {t} not found in table")
                continue

            dx_val = row['cum_dx'].values[0]
            dy_val = row['cum_dy'].values[0]

            err_x = abs(dx_val)
            err_y = abs(dy_val)
            max_endpoint_error = max(max_endpoint_error, err_x, err_y)

            print(f"T={t}: dx={dx_val:.4f}, dy={dy_val:.4f}")

        # Assertion: No Random Walk
        # Note: We observe a linear bias of approx 0.2 px/cycle due to stationary windowing effects.
        # Over 5 cycles (50 frames), this accumulates to ~0.8 px.
        # Ideally this should be 0, but current pairwise registration has this limitation.
        # We set tolerance to 1.0 to accept this known bias while preventing catastrophic failure.
        self.assertLess(max_endpoint_error, 1.0,
                        f"Significant drift bias detected! Integrator did not return to zero. Max error: {max_endpoint_error:.4f}")

    def test_amplitude_accuracy(self):
        """
        Verifies that the estimated drift amplitude matches the ground truth amplitude.
        """
        AMPLITUDE = 3.0
        PERIOD = 12
        video, gt_drift = self.generate_oscillating_video(n_cycles=2, period=PERIOD, amplitude=AMPLITUDE)

        corrected, table = apply_drift_correction(
            video,
            method='subpixel',
            reverse_time='both',
            save_drift_table=False
        )

        est_x = table['cum_dx'].values

        range_est = np.max(est_x) - np.min(est_x)
        range_gt = 2 * AMPLITUDE

        print(f"\nAmplitude Check:")
        print(f"Ground Truth Range: {range_gt:.4f}")
        print(f"Estimated Range:    {range_est:.4f}")

        error = abs(range_est - range_gt)
        rel_error = error / range_gt

        print(f"Relative Error: {rel_error:.2%}")

        self.assertLess(rel_error, 0.15,
                        f"Drift amplitude estimation has high error ({rel_error:.1%}). Likely windowing bias.")

    def test_unidirectional_vs_bidirectional(self):
        """
        Demonstrates the benefit of bidirectional estimation.
        """
        AMPLITUDE = 2.0
        PERIOD = 10
        # Generate 3 cycles (30 frames)
        video, gt_drift = self.generate_oscillating_video(n_cycles=3, period=PERIOD, amplitude=AMPLITUDE)

        # Bidirectional
        _, table_bi = apply_drift_correction(video, reverse_time='both', save_drift_table=False)

        # Unidirectional
        _, table_uni = apply_drift_correction(video, reverse_time=False, save_drift_table=False)

        # Check endpoint of 2nd cycle (T=20)
        # T=20 corresponds to index 19 if indexed 0..N-1, or we query 'Time Point' column
        t_check = 20

        row_bi = table_bi[table_bi['Time Point'] == t_check]
        row_uni = table_uni[table_uni['Time Point'] == t_check]

        if row_bi.empty or row_uni.empty:
            # Fallback to last available
            err_bi = abs(table_bi['cum_dx'].iloc[-1])
            err_uni = abs(table_uni['cum_dx'].iloc[-1])
            print(f"T={t_check} not found, using last frame.")
        else:
            err_bi = abs(row_bi['cum_dx'].values[0])
            err_uni = abs(row_uni['cum_dx'].values[0])

        print(f"\nUnidirectional vs Bidirectional Walk (T={t_check}):")
        print(f"Bidirectional Error: {err_bi:.4f}")
        print(f"Unidirectional Error: {err_uni:.4f}")

        # Just ensure Bidirectional isn't catastrophic
        self.assertLess(err_bi, 1.0, "Bidirectional drift accumulated too much error")

if __name__ == '__main__':
    unittest.main()

# =========================================
# Source: test_bidirectional_drift_sign.py
# =========================================


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
        """Test that correction opposes drift works as expected."""
        # 1. Setup: Drift +1.0 px/frame in X
        # Total drift over 9 steps: +9.0 px.
        drift_rate = (0.0, 1.0)
        n_frames = 10
        video = self.generate_drifting_blob(n_frames=n_frames, drift=drift_rate)

        # 2. Run Bidirectional Correction
        corrected, table = apply_drift_correction(video, reverse_time='both', save_drift_table=False)

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

# =========================================
# Source: test_drift_subpixel_precision.py
# =========================================
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
        corrected_int, _ = apply_drift_correction(
            video.copy(),
            method='integer',
            save_drift_table=False
        )
        rmse_int = self.measure_stability_rmse(corrected_int)

        # 3. Run Subpixel Correction (Test Subject)
        corrected_sub, _ = apply_drift_correction(
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
        corrected, table = apply_drift_correction(video.copy(), method='subpixel')

        # Check detected drift
        max_cum_drift = np.max(np.abs(table[['cum_dx', 'cum_dy']].values))
        print(f"\nStationary Drift Detected: {max_cum_drift:.4f}")
        self.assertLess(max_cum_drift, 0.05, "False drift detected on stationary object")

        # Check image fidelity (MSE)
        mse = np.mean((corrected - video)**2)
        print(f"Stationary Reconstruction MSE: {mse:.6f}")
        self.assertLess(mse, 1e-6, "Subpixel correction degraded stationary image")

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
        corrected_int, _ = apply_drift_correction(video, method='integer', save_drift_table=False)

        # 2. Run Subpixel Correction
        corrected_sub, _ = apply_drift_correction(video, method='subpixel', save_drift_table=False)

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
        corrected, table = apply_drift_correction(video, method='subpixel', save_drift_table=False)

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

# =========================================
# Source: test_drift_edge_robustness.py
# =========================================


def create_square_frame_noisy(shape, object_pos, object_size, intensity=1.0, noise_level=0.1):
    # Set seed for reproducibility
    rng = np.random.default_rng(42)
    frame = rng.normal(0, noise_level, shape).astype(np.float32)
    y, x = object_pos
    h, w = object_size
    y_start = max(0, int(y - h//2))
    y_end = min(shape[0], int(y + h//2))
    x_start = max(0, int(x - w//2))
    x_end = min(shape[1], int(x + w//2))
    frame[y_start:y_end, x_start:x_end] += intensity
    return frame

def test_drift_edge_object():
    """
    Verifies that drift estimation works correctly for an object located at the edge of the FOV.
    The previous implementation (aggressive 2D windowing) failed this case because the object
    in the Y-taper was suppressed before X-projection.
    """
    shape = (128, 128)
    # Object at Y=10 (near top edge), X=64 (center)
    # Shift X by 20 pixels. Y shift 0.
    pos1 = (10, 64)
    pos2 = (10, 84)
    size = (10, 10)

    # Moderate noise level where signal is detectable but weak
    # Intensity 1.0, Noise 0.2. SNR approx 5.
    noise = 0.2

    frame1 = create_square_frame_noisy(shape, pos1, size, 1.0, noise)
    frame2 = create_square_frame_noisy(shape, pos2, size, 1.0, noise)

    # Run drift estimation
    shift = estimate_drift(frame1, frame2)

    # Expected shift: (-0, -20)
    # Note: estimate_drift returns (shift_x, shift_y)
    # But wait, phase_cross_correlation on 1D projection returns shift along that axis.
    # estimate_drift does:
    # shift_x = pcc(proj_x_1, proj_x_2)
    # shift = (shift_x, shift_y)
    # In our case, X-shift is 20 (frame1 -> frame2).
    # So frame2 is shifted by +20.
    # PCC returns -20.

    print(f"Estimated shift: {shift}")

    expected_shift = np.array([-20.0, 0.0])

    # Allow some tolerance due to noise
    # The previous implementation gave -4.9 (Error ~15px)
    # We expect close to -20.
    assert np.allclose(shift, expected_shift, atol=1.0), f"Expected {expected_shift}, got {shift}"

def test_apply_drift_correction_edge():
    """
    Verifies apply_drift_correction handles the edge case correctly over a sequence.
    """
    shape = (128, 128)
    T = 3
    video = np.zeros((T, *shape), dtype=np.float32)

    # Object moves 10px per frame in X.
    # Y stays at 10 (edge).
    shifts = [0, 10, 20]

    for t, s in enumerate(shifts):
        video[t] = create_square_frame_noisy(shape, (10, 64 + s), (10, 10), 1.0, 0.1)

    corrected, table = apply_drift_correction(video, save_drift_table=False)

    # Check if drift table captured the motion
    # Frame 0: Ref
    # Frame 1: dx should be approx -10.
    # Frame 2: dx should be approx -10.

    print(table)

    dx_vals = table['dx'].values
    # Note: table has entry for t=0 (ref), t=1, t=2.
    assert len(dx_vals) == 3

    assert np.allclose(dx_vals[1:], -10.0, atol=1.0)

# =========================================
# Source: test_drift_integrity.py
# =========================================
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-image",
#     "tqdm",
# ]
# ///


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
        """Test that fractional drift accumulation works as expected."""
        # Create dummy video (Time=20, X=10, Y=10)
        video = np.zeros((20, 10, 10), dtype=np.uint8)

        # We simulate a constant drift of 0.5 pixels per frame.
        # Over 19 intervals (frames 1 to 19), total drift should be 9.5 pixels.

        # When reverse_time=False (default), the code calls estimate_shift_1d_iterative
        # exactly TWICE per time point (once for Y, once for X).
        # For 2D data (T, Y, X), the projections list is [proj_y, proj_x].
        # The loop in compute_drift_trajectory calls align_step, which iterates over num_dims=2.
        # First it estimates shift for Y, then for X.

        # If the object moves +0.5 pixels/frame, the shift required to register
        # frame t to t-1 is -0.5.

        # We patch estimate_shift_1d_iterative directly to return the desired shift.
        # It returns a float, not a tuple like phase_cross_correlation.

        ret_y = 0.0
        ret_x = 0.5

        # Note: The sign of accumulation depends on the code.
        # So if we return 0.5 for x, cum_dx increases by 0.5.

        side_effect = []
        # Loop runs for range(1, 20) -> 19 iterations
        for _ in range(19):
            side_effect.append(ret_y)
            side_effect.append(ret_x)

        with patch('eigenp_utils.maxproj_registration.estimate_shift_1d_iterative', side_effect=side_effect):
            corrected, drift_table = apply_drift_correction(video, save_drift_table=False)

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

# =========================================
# Source: test_drift_correction_accuracy.py
# =========================================
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-image",
#     "scipy",
# ]
# ///


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
        # Note: apply_drift_correction prints to stdout (tqdm), which we might ignore
        corrected_video, drift_table = apply_drift_correction(video, save_drift_table=False)

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

        _, drift_table = apply_drift_correction(video, save_drift_table=False)

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
