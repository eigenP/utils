import pytest
import numpy as np
from eigenp_utils.intensity_rescaling import contrast_stretching, correct_z_intensity_decay


def test_contrast_stretching_uint8_preserves_dynamic_range():
    # Construct uint8 gradient image with values between 20 and 200
    img_uint8 = np.linspace(20, 200, 10000, dtype=np.uint8).reshape(100, 100)

    stretched = contrast_stretching(img_uint8, p_min=0.0, p_max=100.0)

    assert stretched.dtype == np.uint8
    # Unique value count must reflect smooth stretching, not binary {0, 1} collapse
    assert len(np.unique(stretched)) > 100
    assert stretched.min() == 0
    assert stretched.max() == 255


def test_z_decay_gamma_uint8_correction():
    # Construct 3D stack (Z, Y, X) with intensity decaying along Z
    z_slices = 10
    stack = np.zeros((z_slices, 32, 32), dtype=np.uint8)
    for z in range(z_slices):
        decay_factor = np.exp(-0.2 * z)
        stack[z] = np.uint8(200 * decay_factor)

    corrected_img = correct_z_intensity_decay(stack, method='gamma')

    assert corrected_img.dtype == np.uint8
    # Slice 0 and Slice N-1 should now have substantially equal mean intensity
    assert abs(int(corrected_img[0].mean()) - int(corrected_img[-1].mean())) < 15


def test_decorator_preserves_function_metadata():
    assert contrast_stretching.__name__ == "contrast_stretching"
    assert "Stretch the intensity range" in contrast_stretching.__doc__


def test_decorator_dictionary_and_tuple_returns():
    stack = np.ones((5, 16, 16), dtype=np.uint16) * 1000
    result = correct_z_intensity_decay(stack, return_diagnostic=True)

    assert isinstance(result, dict)
    assert "image" in result
    assert result["image"].dtype == np.uint16
    assert isinstance(result["diagnostic_data"], dict)
