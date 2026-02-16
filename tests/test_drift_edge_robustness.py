
import numpy as np
import pytest
from eigenp_utils.maxproj_registration import estimate_drift_2D, apply_drift_correction_2D

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
    shift = estimate_drift_2D(frame1, frame2)

    # Expected shift: (-0, -20)
    # Note: estimate_drift_2D returns (shift_x, shift_y)
    # But wait, phase_cross_correlation on 1D projection returns shift along that axis.
    # estimate_drift_2D does:
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
    Verifies apply_drift_correction_2D handles the edge case correctly over a sequence.
    """
    shape = (128, 128)
    T = 3
    video = np.zeros((T, *shape), dtype=np.float32)

    # Object moves 10px per frame in X.
    # Y stays at 10 (edge).
    shifts = [0, 10, 20]

    for t, s in enumerate(shifts):
        video[t] = create_square_frame_noisy(shape, (10, 64 + s), (10, 10), 1.0, 0.1)

    corrected, table = apply_drift_correction_2D(video, save_drift_table=False)

    # Check if drift table captured the motion
    # Frame 0: Ref
    # Frame 1: dx should be approx -10.
    # Frame 2: dx should be approx -10.

    print(table)

    dx_vals = table['dx'].values
    # Note: table has entry for t=0, t=1, t=2 (with t=0 having dx=0)
    assert len(dx_vals) == 3

    # Frame 0 is 0. Frame 1, 2 are -10.
    assert dx_vals[0] == 0.0
    assert np.allclose(dx_vals[1:], -10.0, atol=1.0)
