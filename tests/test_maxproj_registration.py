import sys
import types
import numpy as np
import pytest

# Provide a stub pandas module so the import succeeds
sys.modules.setdefault('pandas', types.ModuleType('pandas'))

from eigenp_utils.maxproj_registration import zero_shift_multi_dimensional, _2D_weighted_image


def test_2D_weighted_image():
    # Test with a small image
    image = np.ones((10, 10), dtype=np.float32)
    overlap = 3
    weighted = _2D_weighted_image(image, overlap)

    # Check shape
    assert weighted.shape == image.shape

    # Center should be 1
    assert weighted[5, 5] == 1.0

    # Edges should be tapered (less than 1)
    # The weight function is 3x^2 - 2x^3. At x=0, w=0. At x=1, w=1.
    # The implementation generates weights for i in range(overlap).
    # i=0 -> x=0 -> w=0.
    # i=1 -> x=0.5 -> w=0.5.
    # i=2 -> x=1 -> w=1.

    # Top left corner (0, 0)
    # y-weight is w[0]=0. x-weight is w[0]=0.
    # value = 1 * 0 * 0 = 0.
    assert weighted[0, 0] == 0.0

    # Top edge center (0, 5)
    # y-weight is 0. x-weight is 1 (center).
    # value = 1 * 0 * 1 = 0.
    assert weighted[0, 5] == 0.0

    # (1, 5) -> y-weight is w[1] (at x=0.5) = 0.5. x-weight is 1.
    # value = 0.5.
    assert np.isclose(weighted[1, 5], 0.5, atol=1e-5)

    # Test with overlap = 0 (should return original)
    weighted_no_overlap = _2D_weighted_image(image, 0)
    assert np.array_equal(weighted_no_overlap, image)


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
