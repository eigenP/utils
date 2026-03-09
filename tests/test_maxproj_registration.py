import sys
import types
import numpy as np
import pytest

# Provide a stub pandas module so the import succeeds
sys.modules.setdefault('pandas', types.ModuleType('pandas'))

from eigenp_utils.maxproj_registration import zero_shift_multi_dimensional, apply_drift_correction
import numpy as np

def test_zero_shift_positive_negative():
    arr = np.arange(9).reshape(3, 3)
    result = zero_shift_multi_dimensional(arr, shifts=(1, -1), fill_value=-1)
    expected = np.array([
        [-1, -1, -1],
        [1, 2, -1],
        [4, 5, -1],
    ])
    assert np.array_equal(result, expected)


def test_zero_shift_errors():
    arr = np.zeros((2, 2))
    with pytest.raises(ValueError):
        zero_shift_multi_dimensional(arr, shifts=(1,))
    with pytest.raises(TypeError):
        zero_shift_multi_dimensional(arr, shifts=(1.0, 2.0))


def test_apply_drift_correction_2d_plus_t():
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
    video_2d = np.zeros((10, 10))
    with pytest.raises(ValueError, match="Expected 3D \\(T, Y, X\\) or 4D \\(T, Z, Y, X\\) data"):
        apply_drift_correction(video_2d)

    video_5d = np.zeros((2, 2, 10, 10, 10))
    with pytest.raises(ValueError, match="Expected 3D \\(T, Y, X\\) or 4D \\(T, Z, Y, X\\) data"):
        apply_drift_correction(video_5d)
