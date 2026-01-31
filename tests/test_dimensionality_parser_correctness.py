
import pytest
import numpy as np
from eigenp_utils.dimensionality_parser import dimensionality_parser

# --- Helper Functions ---

@dimensionality_parser(target_dims='YX')
def op_identity(image):
    """Returns the image as-is. Preserves Y and X."""
    return image

@dimensionality_parser(target_dims='ZYX')
def op_project(image):
    """Max projection along Z. Reduces Z, preserves Y and X."""
    return np.max(image, axis=0)

@dimensionality_parser(target_dims='YX')
def op_resize(image):
    """Subsamples Y and X by 2. Resizes Y and X."""
    return image[::2, ::2]

# --- Tests ---

def test_broadcasting_identity():
    """
    Testr 🔎: Verify that identity operation broadcasts correctly over extra dimensions.
    Input: (C, Y, X) -> Output: (C, Y, X)
    Invariant: Shape preservation for non-target dims.
    """
    # Shape: (2, 10, 10) -> C=2, Y=10, X=10
    inp = np.random.rand(2, 10, 10)
    out = op_identity(inp)

    assert out.shape == (2, 10, 10)
    np.testing.assert_array_equal(out, inp)

def test_broadcasting_projection():
    """
    Testr 🔎: Verify that reduction operation broadcasts correctly.
    Input: (C, Z, Y, X) -> Output: (C, Y, X)
    Invariant: Reduced dimension (Z) is removed, extra dim (C) is preserved.
    """
    # Shape: (2, 5, 10, 10) -> C=2, Z=5, Y=10, X=10
    inp = np.random.rand(2, 5, 10, 10)
    out = op_project(inp)

    assert out.shape == (2, 10, 10)

    # Verify values for first channel
    expected_c0 = np.max(inp[0], axis=0)
    np.testing.assert_array_equal(out[0], expected_c0)

def test_broadcasting_resize():
    """
    Testr 🔎: Verify that resizing operation broadcasts correctly.
    Input: (C, Y, X) -> Output: (C, Y/2, X/2)
    Invariant: Resized dimensions are handled correctly, not treated as 'reduced' (removed).

    Failure Prediction: The parser might incorrectly infer Y and X as 'reduced'
    because output size != input size, leading to iteration errors or incorrect shape logic.
    """
    # Shape: (2, 10, 10) -> C=2, Y=10, X=10
    inp = np.random.rand(2, 10, 10)
    out = op_resize(inp)

    # Expected output: (2, 5, 5)
    assert out.shape == (2, 5, 5)

    # Verify values
    expected_c0 = inp[0, ::2, ::2]
    np.testing.assert_array_equal(out[0], expected_c0)
