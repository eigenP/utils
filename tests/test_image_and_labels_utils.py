import numpy as np
import pytest
from eigenp_utils.image_and_labels_utils import (
    windowed_slice_projection,
    optimized_entire_labels_touching_mask,
    sample_intensity_around_points_optimized
)

def test_windowed_slice_projection_max():
    img = np.zeros((5, 10, 10))
    img[2, 5, 5] = 10

    thick = windowed_slice_projection(img, window_size=3, axis=0, operation='max')

    assert thick.shape == (5, 10, 10)
    assert thick[1, 5, 5] == 10
    assert thick[2, 5, 5] == 10
    assert thick[3, 5, 5] == 10
    assert thick[0, 5, 5] == 0
    assert thick[4, 5, 5] == 0

def test_windowed_slice_projection_average():
    img = np.ones((5, 10, 10)) * 3

    thick = windowed_slice_projection(img, window_size=3, axis=0, operation='average')

    # average of 3 values of 3 is 3, except boundaries which are padded with 0
    # boundary 0: (0 + 3 + 3) / 3 = 2
    # middle: (3 + 3 + 3) / 3 = 3

    assert np.allclose(thick[2, :, :], 3)
    assert np.allclose(thick[0, :, :], 2)

def test_optimized_entire_labels_touching_mask():
    labels = np.zeros((20, 20), dtype=int)
    # create two labels
    labels[5:10, 5:10] = 1
    labels[15:20, 15:20] = 2

    mask = np.zeros((20, 20), dtype=int)
    # mask touching label 1 after expansion
    mask[2:4, 2:4] = 1

    res = optimized_entire_labels_touching_mask(labels, mask)

    # label 1 should be completely retained
    assert np.all(res[5:10, 5:10] == 1)
    # label 2 should be gone
    assert np.all(res[15:20, 15:20] == 0)

def test_sample_intensity_around_points_optimized():
    image_3d = np.ones((10, 10, 10))
    image_3d[5, 5, 5] = 10 # central high intensity point

    points = np.array([
        [5, 5, 5],
        [1, 1, 1],
        [20, 20, 20] # out of bounds
    ])

    res = sample_intensity_around_points_optimized(image_3d, points, diameter=3)

    # [5,5,5] is average of 3x3x3 cube = 27 voxels
    # 26 ones + 1 ten = 36 / 27 = 1.333

    assert len(res) == 3
    assert np.isclose(res[0], 36 / 27)
    assert np.isclose(res[1], 1.0) # all ones
    assert np.isnan(res[2]) # out of bounds
