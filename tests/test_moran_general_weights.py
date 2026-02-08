
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pytest
from eigenp_utils.single_cell import morans_i_all_fast

class MockAdata:
    def __init__(self, X, var_names):
        self.X = X
        self.var_names = var_names
        self.shape = X.shape
        self.layers = {}
        self.obsp = {}

def test_moran_bias_isolated_nodes():
    """
    Test that morans_i_all_fast produces correct results even with isolated nodes
    or non-row-standardized weights, compared to a naive exact implementation.
    """
    # Create a small dataset: 4 nodes
    # Nodes 0, 1 connected (clique)
    # Nodes 2, 3 isolated

    # Adjacency:
    # 0: [0, 1, 0, 0]
    # 1: [1, 0, 0, 0]
    # 2: [0, 0, 0, 0]
    # 3: [0, 0, 0, 0]

    W = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float32)

    W_csr = sp.csr_matrix(W)

    # Feature x:
    # 0, 1 have value 10
    # 2, 3 have value -10
    # Global mean = 0
    # Connected nodes are positively correlated (10 next to 10).
    # Expected I should be positive.

    x = np.array([10, 10, -10, -10], dtype=np.float32).reshape(4, 1)

    adata = MockAdata(x, np.array(["gene1"]))

    # Manual Calculation:
    # Mean = 0
    # Denominator = sum(x^2) = 100+100+100+100 = 400
    # Numerator term 1: sum_ij w_ij (xi-u)(xj-u)
    # i=0, j=1: 1 * 10 * 10 = 100
    # i=1, j=0: 1 * 10 * 10 = 100
    # Others 0.
    # Num = 200.
    # S0 = 2.
    # I = (N/S0) * (Num/Den) = (4/2) * (200/400) = 2 * 0.5 = 1.0.

    # Run morans_i_all_fast with explicit W (not row standardized!)
    # Note: morans_i_all_fast assumes W_rowstd is row-standardized by default if passed?
    # No, it just uses it as W. The argument name implies intent, but math should hold if we fix it.
    # However, if we pass W_rowstd, the function uses it.

    res = morans_i_all_fast(adata, W_rowstd=W_csr, block_genes=100)

    print(f"\nExact I: 1.0")
    print(f"Computed I (Not Row Std): {res['I'][0]}")

    # Case 2: Row Standardized manually (handling islands)
    # Row sums: [1, 1, 0, 0]
    # W_rs = W.
    # So same result expected.

    # Case 3: Shift Mean
    # x = [20, 20, 0, 0] -> Mean = 10.
    # x - u = [10, 10, -10, -10]. Same deviations.
    # Should get I = 1.0.

    x_shifted = np.array([20, 20, 0, 0], dtype=np.float32).reshape(4, 1)
    adata_shifted = MockAdata(x_shifted, np.array(["gene1"]))

    res_shifted = morans_i_all_fast(adata_shifted, W_rowstd=W_csr, block_genes=100)

    print(f"Exact I (Shifted): 1.0")
    print(f"Computed I (Shifted): {res_shifted['I'][0]}")

    # Check if they match
    assert np.isclose(res_shifted['I'][0], 1.0, atol=1e-5), f"Shifted mean failed: {res_shifted['I'][0]}"

    # Case 4: General Weights (Not Row Std)
    # W = 2 * Identity (Self loops? No, usually 0 diag).
    # W = 2 * adjacency
    W2 = 2 * W_csr
    # S0 = 4.
    # Num = 1 * 2 * 10 * 10 + 1 * 2 * 10 * 10 = 400.
    # Den = 400.
    # I = (4/4) * (400/400) = 1.0.

    res_scaled = morans_i_all_fast(adata, W_rowstd=W2, block_genes=100)
    print(f"Exact I (Scaled W): 1.0")
    print(f"Computed I (Scaled W): {res_scaled['I'][0]}")

    assert np.isclose(res_scaled['I'][0], 1.0, atol=1e-5), f"Scaled W failed: {res_scaled['I'][0]}"

if __name__ == "__main__":
    test_moran_bias_isolated_nodes()
