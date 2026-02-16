
import numpy as np
import pytest
from src.eigenp_utils.maxproj_registration import compute_drift_trajectory, apply_drift_correction_2D

def test_compute_drift_trajectory_forward():
    """
    Test that compute_drift_trajectory correctly computes positions for a known shift.
    Synthetic data: Square moving by (1, 1) per frame.
    Pos[0] = (0,0)
    Pos[1] = (1,1)
    ...
    Pos[4] = (4,4)
    Trajectory should reflect this relative motion.
    """
    T, H, W = 5, 20, 20
    projections_x = np.zeros((T, W), dtype=np.float32)
    projections_y = np.zeros((T, H), dtype=np.float32)

    # Create moving peaks
    for t in range(T):
        # Peak at t+5 to avoid boundary issues with windowing
        # We use a small gaussian or block to avoid subpixel ambiguity if needed,
        # but single pixel peak is fine for integer shift test.
        projections_x[t, t+5] = 1.0
        projections_y[t, t+5] = 1.0

    positions, steps = compute_drift_trajectory(projections_x, projections_y, mode='forward')

    # Expected:
    # t=0: Pos=(0,0)
    # t=1: Shift t->t-1 is (-1, -1). Pos[1] = Pos[0] - (-1, -1) = (1, 1).
    # ...
    # Pos[t] = (t, t)

    expected_positions = np.array([[t, t] for t in range(T)], dtype=np.float32)

    np.testing.assert_allclose(positions, expected_positions, atol=0.1)

def test_apply_drift_correction_reverse_indexing_bug():
    """
    Regression test for the off-by-one indexing bug in reverse mode.
    Ensures that frame 0 is corrected properly and aligns with the reference frame (last frame).
    """
    T, H, W = 5, 20, 20 # Increased size to avoid edge effects
    video = np.zeros((T, H, W), dtype=np.uint8)
    for t in range(T):
        # Center the motion somewhat
        video[t, t+5, t+5] = 100 # Peak moves (5,5) -> (9,9)

    # We expect all frames to be aligned to frame 4 (9,9)
    corrected, table = apply_drift_correction_2D(video, reverse_time=True)

    # Check peaks
    for t in range(T):
        if corrected[t].max() == 0:
            pytest.fail(f"Frame {t} is empty after correction")

        peak_idx = np.unravel_index(np.argmax(corrected[t]), corrected[t].shape)
        # Expect (9, 9)
        # Using slice to handle potential off-by-one if interpolation blurs
        # But max pixel should be at 9,9
        np.testing.assert_array_equal(peak_idx, (9, 9), err_msg=f"Frame {t} failed alignment")

    # Check drift table consistency
    # cum_dx for frame 0 should be shift applied.
    # Shift = Ref - Pos[0] = (9,9) - (5,5) = (4,4).
    # table entry for t=0
    row0 = table[table['Time Point'] == 0].iloc[0]
    # cum_dx matches x-shift (second component of shift tuple)
    assert row0['cum_dx'] == 4.0
    assert row0['cum_dy'] == 4.0

def test_apply_drift_correction_bidirectional():
    """
    Test bidirectional mode.
    Synthetic data with noise: moving by (1, 1).
    Bidirectional averaging should be robust.
    """
    T, H, W = 5, 20, 20
    video = np.zeros((T, H, W), dtype=np.uint8)
    for t in range(T):
        # Peak at t+5
        video[t, t+5, t+5] = 100

    # Reference frame 0.
    # Pos[t] = (t, t). Ref = (0, 0).
    # Shift[t] = (0,0) - (t,t) = (-t, -t).
    # Frame t at (t+5, t+5) shifted by (-t, -t) -> (5, 5).
    # So all frames should align to (5, 5).

    corrected, table = apply_drift_correction_2D(video, reverse_time='both')

    for t in range(T):
        peak_idx = np.unravel_index(np.argmax(corrected[t]), corrected[t].shape)
        np.testing.assert_array_equal(peak_idx, (5, 5), err_msg=f"Frame {t} failed alignment in both mode")

def test_apply_drift_correction_subpixel():
    """
    Test subpixel mode runs without error and produces reasonable output.
    """
    T, H, W = 3, 20, 20
    video = np.zeros((T, H, W), dtype=np.float32)
    # create gaussian blob moving by 0.5 px
    y, x = np.mgrid[:H, :W]

    for t in range(T):
        # center = 10 + t*0.5
        c = 10.0 + t * 0.5
        video[t] = np.exp(-((x-c)**2 + (y-c)**2)/2)

    # Align to frame 0
    # Pos[t] = (0.5t, 0.5t)
    # Shift[t] = (-0.5t, -0.5t)
    # Corrected should be centered at 10.0

    corrected, table = apply_drift_correction_2D(video, reverse_time=False, method='subpixel')

    # Check center of mass of corrected[2] (t=2)
    # Should be close to 10.0
    import scipy.ndimage
    cm = scipy.ndimage.center_of_mass(corrected[2])
    np.testing.assert_allclose(cm, (10.0, 10.0), atol=0.2)
