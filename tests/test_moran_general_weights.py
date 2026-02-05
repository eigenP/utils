
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from eigenp_utils.single_cell import morans_i_all_fast

def build_grid_graph(n_rows, n_cols):
    """Builds a 4-connected grid graph adjacency matrix."""
    n = n_rows * n_cols
    rows = []
    cols = []

    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            # Neighbors: up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    j = nr * n_cols + nc
                    rows.append(i)
                    cols.append(j)

    data = np.ones(len(rows))
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return W

def naive_morans_i(x, W, center=True):
    """Naive, readable implementation of Moran's I."""
    n = len(x)
    S0 = W.sum()

    if center:
        x_bar = x.mean()
        z = x - x_bar
    else:
        z = x

    # General Formula: (N / S0) * (Z^T W Z) / (Z^T Z)
    # Note: Z is column vector (n, 1) or array (n,)
    if z.ndim == 1:
        z = z[:, None] # (n, 1)

    num = (n / S0) * (z.T @ W @ z)[0, 0]
    den = (z.T @ z)[0, 0]

    if den == 0:
        return np.nan

    return num / den

def test_morans_i_general_weights_star():
    """
    Test case from reproduction: Star Graph with Hub signal.
    This fails if the algorithm assumes row-standardization.
    """
    # 1. Setup Data: Star Graph
    n_nodes = 5
    # Hub has high expression
    X = np.array([10.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Adjacency (Symmetric Binary)
    # 0 connected to 1, 2, 3, 4
    rows = [0, 0, 0, 0, 1, 2, 3, 4]
    cols = [1, 2, 3, 4, 0, 0, 0, 0]
    data = np.ones(len(rows), dtype=np.float32)
    W_general = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))

    adata = AnnData(X=X[:, None])
    adata.var_names = ["GeneHub"]
    adata.obs_names = [str(i) for i in range(n_nodes)]

    # 2. Ground Truth
    i_true = naive_morans_i(X, W_general, center=True)

    # 3. Run Fast Implementation
    res = morans_i_all_fast(adata, W_rowstd=W_general)
    i_fast = res.loc[0, "I"]

    print(f"True: {i_true}, Fast: {i_fast}")

    assert np.isclose(i_fast, i_true, atol=1e-5), \
        f"Star Graph: Fast I ({i_fast}) != True I ({i_true})"

def test_morans_i_general_weights_random():
    """
    Test on a random matrix with general (non-row-standardized) weights.
    """
    n = 50
    # Symmetric, but arbitrary weights
    # W = A + A.T
    A = sp.random(n, n, density=0.1, random_state=42)
    W = A + A.T
    # W is not row-standardized. Row sums vary.

    rng = np.random.default_rng(123)
    x = rng.uniform(0, 10, n)

    adata = AnnData(X=x[:, None])
    adata.var_names = ["gene1"]

    i_true = naive_morans_i(x, W, center=True)
    res = morans_i_all_fast(adata, W_rowstd=W)
    i_fast = res.loc[0, "I"]

    assert np.isclose(i_fast, i_true, atol=1e-5), \
        f"Random General W: Fast I ({i_fast}) != True I ({i_true})"

def test_morans_i_disconnected_components():
    """
    Test graph with disconnected components.
    Row sums are zero for isolated nodes.
    """
    n = 10
    # Nodes 0-4 connected (clique), 5-9 connected (clique). Disconnected.
    # Actually, simpler: just use random block diagonal.

    # Create block diag
    B1 = np.ones((5,5)) - np.eye(5)
    B2 = np.ones((5,5)) - np.eye(5)
    W_dense = np.zeros((10,10))
    W_dense[:5, :5] = B1
    W_dense[5:, 5:] = B2
    W = sp.csr_matrix(W_dense)

    # Signal: 0-4 high, 5-9 low.
    x = np.concatenate([np.ones(5)*10, np.zeros(5)])

    adata = AnnData(X=x[:, None])
    adata.var_names = ["block_sig"]

    i_true = naive_morans_i(x, W, center=True)
    res = morans_i_all_fast(adata, W_rowstd=W)
    i_fast = res.loc[0, "I"]

    assert np.isclose(i_fast, i_true, atol=1e-5), \
        f"Disconnected: Fast I ({i_fast}) != True I ({i_true})"

def test_morans_i_row_standardized_regression():
    """
    Test that it still works for row-standardized weights (Regression).
    """
    n = 20
    W_raw = build_grid_graph(4, 5)

    # Row standardize
    row_sums = np.array(W_raw.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    W_std = sp.diags(1.0/row_sums) @ W_raw

    rng = np.random.default_rng(999)
    x = rng.standard_normal(n)

    adata = AnnData(X=x[:, None])
    adata.var_names = ["regress"]

    i_true = naive_morans_i(x, W_std, center=True)
    res = morans_i_all_fast(adata, W_rowstd=W_std)
    i_fast = res.loc[0, "I"]

    assert np.isclose(i_fast, i_true, atol=1e-5), \
        f"RowStd Regression: Fast I ({i_fast}) != True I ({i_true})"
