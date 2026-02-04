# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pytest",
# ]
# ///

import unittest
import numpy as np
import pytest
from eigenp_utils.dimensionality_parser import dimensionality_parser

class TestDimensionalityParserResizing(unittest.TestCase):
    """
    Testr 🔎 Verification: Dimensionality Parser Resizing & Reduction

    The `dimensionality_parser` decorator allows functions written for lower-dimensional
    inputs (e.g., 2D images) to be applied automatically to higher-dimensional stacks
    (e.g., 3D videos, 4D hyperstacks).

    This test verifies two critical behaviors:
    1.  **Rank-Preserving Resizing**: If a function changes the spatial dimensions
        (e.g., downscaling) but preserves the rank (2D -> 2D), the parser must
        correctly reconstruct the output stack with the new dimensions, rather than
        assuming dimensions were reduced.
    2.  **Dimension Reduction**: If a function reduces the rank (e.g., Max Projection
        2D -> 1D), the parser must correctly identify the missing dimension and
        exclude it from the output shape.

    The bug identified was that `dimensionality_parser` used strict size equality
    to determine if a dimension was preserved. Resizing (size mismatch) was
    incorrectly interpreted as reduction, causing output shape mismatch errors.
    """

    def test_rank_preserving_resizing(self):
        """
        Scenario: Downscaling a 2D image by factor of 2.
        Input: (T, 10, 10)
        Function: (10, 10) -> (5, 5)
        Expected Output: (T, 5, 5)
        """

        @dimensionality_parser(target_dims='YX')
        def resize_by_half(img):
            # Simple slicing to simulate resizing
            return img[::2, ::2]

        # Input: 2 frames of 10x10
        stack = np.zeros((2, 10, 10), dtype=np.float32)
        # Add some data to verify mapping
        stack[0, 0, 0] = 1.0
        stack[1, 0, 0] = 2.0

        # Execute
        try:
            result = resize_by_half(stack)
        except ValueError as e:
            self.fail(f"Resizing failed with ValueError: {e}")
        except Exception as e:
            self.fail(f"Resizing failed with unexpected exception: {e}")

        # Verify Shape
        self.assertEqual(result.shape, (2, 5, 5),
            f"Expected shape (2, 5, 5), got {result.shape}")

        # Verify Content
        self.assertEqual(result[0, 0, 0], 1.0)
        self.assertEqual(result[1, 0, 0], 2.0)

    def test_dimension_reduction(self):
        """
        Scenario: Max Projection along X.
        Input: (T, 10, 10)
        Function: (10, 10) -> (10,)
        Expected Output: (T, 10)
        """

        @dimensionality_parser(target_dims='YX')
        def max_project_x(img):
            # Project along X (axis 1 of input 2D array)
            return np.max(img, axis=1)

        stack = np.random.rand(2, 10, 10)

        result = max_project_x(stack)

        self.assertEqual(result.shape, (2, 10),
            f"Expected shape (2, 10), got {result.shape}")

    def test_identity_preservation(self):
        """
        Scenario: Identity (No change).
        Input: (T, 10, 10)
        Function: (10, 10) -> (10, 10)
        Expected Output: (T, 10, 10)
        """

        @dimensionality_parser(target_dims='YX')
        def identity(img):
            return img

        stack = np.zeros((2, 10, 10))
        result = identity(stack)

        self.assertEqual(result.shape, (2, 10, 10))

if __name__ == '__main__':
    unittest.main()
