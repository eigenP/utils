import unittest
import numpy as np
from eigenp_utils.dimensionality_parser import dimensionality_parser

class TestDimensionalityParserAmbiguity(unittest.TestCase):
    """
    Testr 🔎 Verification: Dimensionality Parser Ambiguity

    This test targets a specific flaw in the heuristic used to identify reduced dimensions.
    When input dimensions have identical sizes (e.g. a square image), and a reduction operation
    (like max projection) removes one of them, the parser can incorrectly identify *which*
    dimension was removed if it relies solely on matching output sizes to input sizes.

    This ambiguity leads to crashes when combined with `iterate_dims`, as the parser may
    fail to allocate the correct output slicer for the dimension it falsely believes was reduced.
    """

    def test_square_array_reduction_iteration(self):
        """
        Verify that iterating over a preserved dimension works correctly even if
        the input array is square and the other dimension is reduced.
        """
        # 1. Setup
        # Input: (20, 20) Square Array.
        # Dimensions: Y, X
        N = 20
        image = np.random.random((N, N)).astype(np.float32)

        # 2. Define Function
        # We define a function that reduces X (axis 1) but preserves Y (axis 0).
        # We make it robust to 1D input because iterate_dims slices the array, passing 1D rows.

        @dimensionality_parser(target_dims='YX')
        def reduce_x(img):
            # If we get a 2D array (e.g. dummy input), max over axis 1.
            # If we get a 1D array (sliced row), max over axis 0 (global max).
            if img.ndim == 2:
                return np.max(img, axis=1)
            else:
                return np.max(img)

        # 3. Execution
        # We iterate over Y.
        # The parser logic runs BEFORE the iteration loop to determine output shapes.
        # If the parser incorrectly thinks Y is REDUCED (because it matched X's size to Output of dummy),
        # it will fail to find Y in main_dims when setting up output_shape, raising ValueError.

        try:
            # Iterate Y indices 0, 10.
            # If Y is preserved, main_dims=['Y']. Output shape becomes [2].
            # If Y is reduced (bug), main_dims=['X']. Output shape update fails (ValueError).
            result = reduce_x(image, iterate_dims={'Y': f'0:{N}:10'})
        except ValueError as e:
            self.fail(f"Dimensionality Parser crashed on square array iteration logic: {e}")

        # 4. Verification
        # Check shape: Should be (2,) because we iterated 2 steps on Y, and X was reduced.
        self.assertEqual(result.shape, (2,), "Result shape should match iteration count")

        # Check values
        # Row 0 max, Row 10 max
        expected = np.array([np.max(image[0]), np.max(image[10])], dtype=np.float32)
        np.testing.assert_allclose(result, expected, err_msg="Computed result does not match expected max projection")

if __name__ == '__main__':
    unittest.main()
