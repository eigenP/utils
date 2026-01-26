
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pytest
from eigenp_utils.single_cell import morans_i_all_fast

class MockAdata:
    def __init__(self, X):
        self.X = X
        self.layers = {}
        self.var_names = np.arange(X.shape[1]).astype(str)
        self.obsp = {} # Needed if internal calls check obsp

    def copy(self):
        return self

def test_moran_general_weights():
    """
    Verifies that morans_i_all_fast computes the correct Moran's I
    for general (non-row-standardized) weights.
    The previous implementation was biased for such weights.
    """
    # 1. Create a small synthetic dataset
    np.random.seed(42)
    n_cells = 20
    n_genes = 5
    X = np.random.rand(n_cells, n_genes).astype(np.float32)
    mock_adata = MockAdata(X)

    # 2. Create an irregular weight matrix
    # Ring graph + random edges
    rows = []
    cols = []
    for i in range(n_cells):
        # Ring connections
        rows.append(i); cols.append((i + 1) % n_cells)
        rows.append((i + 1) % n_cells); cols.append(i)

    # Add random extra edges to make it irregular
    # Ensure symmetry just to be nice (Moran's I doesn't strictly require it but common)
    for _ in range(10):
        u = np.random.randint(0, n_cells)
        v = np.random.randint(0, n_cells)
        if u != v:
            rows.append(u); cols.append(v)
            rows.append(v); cols.append(u)

    data = np.ones(len(rows))
    W = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    # Ensure duplicates are summed if any
    W.sum_duplicates()

    # Verify W is not row-standardized
    row_sums = W.sum(axis=1).A1
    if np.allclose(row_sums, row_sums[0]):
        # If randomly regular, add one more edge
        W[0, 2] = 1
        W[2, 0] = 1

    # 3. Compute Moran's I using morans_i_all_fast
    # We pass W directly as W_rowstd (even though it's not rowstd)
    df = morans_i_all_fast(mock_adata, W_rowstd=W, source="X", deduplicate="none")

    # 4. Compute Moran's I manually for verification
    # I = (N / S0) * (num / den)
    # num = sum_ij w_ij (xi - xbar)(xj - xbar)
    # den = sum_i (xi - xbar)^2

    S0 = W.sum()
    results_manual = []

    # Get indices from sparse matrix for manual loop (or use matrix math)
    # Matrix math: z.T @ W @ z

    for g in range(n_genes):
        x = X[:, g]
        xbar = x.mean()
        z = x - xbar

        # num = z.T @ W @ z
        num = z.T @ (W @ z)
        den = np.sum(z**2)

        I_manual = (n_cells / S0) * (num / den)
        results_manual.append(I_manual)

    # Sort df by gene to match
    df['gene_idx'] = df['gene'].astype(int)
    df = df.sort_values('gene_idx')

    results_code = df['I'].values

    # We expect this to FAIL with the old implementation
    # But for the test suite, we assert comparison.
    # The test is expected to fail initially.

    print(f"\nMax Diff: {np.max(np.abs(results_code - results_manual))}")

    np.testing.assert_allclose(results_code, results_manual, atol=1e-5, err_msg="Moran's I calculation mismatch for general weights")
