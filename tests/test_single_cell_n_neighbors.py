import pytest
import numpy as np

# This will fail initially because the function is not yet implemented
def test_pacmap_heuristic_n_neighbors():
    try:
        from eigenp_utils.single_cell import pacmap_heuristic_n_neighbors
    except ImportError:
        pytest.fail("pacmap_heuristic_n_neighbors is not implemented yet")

    # n <= 10000 should return 10
    assert pacmap_heuristic_n_neighbors(100) == 10
    assert pacmap_heuristic_n_neighbors(10000) == 10

    # n > 10000 should return int(round(10 + 15 * (np.log10(n) - 4)))
    assert pacmap_heuristic_n_neighbors(50000) == 20
    assert pacmap_heuristic_n_neighbors(100000) == 25
    assert pacmap_heuristic_n_neighbors(1000000) == 40
