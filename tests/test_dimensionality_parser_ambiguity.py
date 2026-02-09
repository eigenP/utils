
import unittest
import numpy as np
from eigenp_utils.dimensionality_parser import dimensionality_parser

class TestDimensionalityParserAmbiguity(unittest.TestCase):
    """
    Testr 🔎 Verification: Dimensionality Parser Ambiguity (Square Inputs)

    This test verifies that the parser correctly handles square inputs where
    dimensions are ambiguous. It primarily ensures that the new "Unique Probe"
    logic is executed without error and produces consistent results.
    """

    def test_square_reduction_probe_path(self):
        """
        Verify that square reduction works correctly without error.
        This exercises the unique probe logic.
        """
        @dimensionality_parser(target_dims='YX')
        def reduce_axis_1(img):
            return np.max(img, axis=1)

        # Input: (T, Y, X) = (2, 10, 10).
        # This triggers the ambiguity (10, 10) for target_dims='YX'.
        # Probe runs with (2, 3) (or similar unique).
        # Probe result determines reduced_dims.

        stack = np.zeros((2, 10, 10))

        res = reduce_axis_1(stack)

        # Check shape correctness (T, Y) -> (2, 10)
        self.assertEqual(res.shape, (2, 10))

if __name__ == '__main__':
    unittest.main()
