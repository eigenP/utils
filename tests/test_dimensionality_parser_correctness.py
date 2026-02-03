
import pytest
import numpy as np
from eigenp_utils.dimensionality_parser import dimensionality_parser

def test_dimensionality_parser_resizing_correctness():
    """
    Testr 🔎: Verify that dimensionality_parser handles resizing operations correctly.

    Intent: The parser claims to support operations that change the size of dimensions
    without reducing the rank (e.g., downscaling). It should not treat resized dimensions
    as 'reduced' and should correctly map the output shape.

    Failure Mode: If the parser relies solely on size matching, it will misinterpret
    a size change (10 -> 5) as a reduction, leading to an incorrect output shape calculation
    and a crash during assignment.
    """

    @dimensionality_parser(target_dims='YX')
    def downscale_2x(img):
        # Input (H, W). Output (H//2, W//2)
        h, w = img.shape
        return np.zeros((h // 2, w // 2), dtype=img.dtype)

    # 1. 2D Input (Direct pass-through of target dims)
    img_2d = np.zeros((10, 10))
    res_2d = downscale_2x(img_2d)
    assert res_2d.shape == (5, 5), f"2D Resizing failed. Expected (5, 5), got {res_2d.shape}"

    # 2. 3D Input (Iteration over S)
    # Input (S=2, Y=10, X=10) -> Output (S=2, Y=5, X=5)
    img_3d = np.zeros((2, 10, 10))
    res_3d = downscale_2x(img_3d)

    assert res_3d.shape == (2, 5, 5), f"3D Resizing iteration failed. Expected (2, 5, 5), got {res_3d.shape}"


def test_dimensionality_parser_reduction_logic():
    """
    Testr 🔎: Verify dimensionality_parser correctly identifies reduced dimensions.

    Intent: When a function reduces specific axes (e.g., max projection), the parser
    must remove those axes from the output shape construction.
    """

    @dimensionality_parser(target_dims='YX')
    def max_proj_x(img):
        # Input (Y, X). Output (Y,)
        return np.max(img, axis=1)

    # Non-square input to avoid ambiguity heuristic issues for this basic test
    # Y=10, X=20
    img = np.random.rand(2, 10, 20)

    # Expected: Iterate S(2). For each, reduce X. Output (S, Y) -> (2, 10)
    res = max_proj_x(img)

    assert res.shape == (2, 10), f"Reduction failed. Expected (2, 10), got {res.shape}"


def test_dimensionality_parser_square_ambiguity():
    """
    Testr 🔎: Probe behavior on ambiguous square inputs.

    Intent: Verify behavior when input is square (10, 10) and one axis is reduced.
    Current heuristic might be fragile. This test documents the behavior.
    """
    @dimensionality_parser(target_dims='YX')
    def mean_last_axis(img):
        # Reduces X (axis 1). Keeps Y (axis 0).
        return np.mean(img, axis=1)

    img = np.zeros((2, 10, 10))
    # Mark the Y axis
    # Y=0 -> all 1s (mean 1)
    # Y=1 -> all 0s (mean 0)
    img[0, 0, :] = 1.0

    res = mean_last_axis(img)

    # If parser correctly kept Y, res[0, 0] should be 1.0
    # If parser incorrectly kept X (thinking Y was reduced),
    # and if it assigns based on kept dim...
    # Actually, if result is (2, 10), it's structurally identical.
    # The question is semantic mapping.

    assert res.shape == (2, 10)
    assert np.isclose(res[0, 0], 1.0), "Data mapping incorrect in square ambiguity case."
