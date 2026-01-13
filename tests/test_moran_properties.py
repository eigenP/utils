
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

def row_standardize(W):
    """Row-standardizes a sparse matrix."""
    row_sums = np.array(W.sum(axis=1)).flatten()
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    inv_sums = 1.0 / row_sums
    D = sp.diags(inv_sums)
    return D @ W

def naive_morans_i(x, W, center=True):
    """Naive, readable implementation of Moran's I."""
    n = len(x)
    S0 = W.sum()

    if center:
        x_bar = x.mean()
        z = x - x_bar
    else:
        z = x

    num = (n / S0) * (z.T @ W @ z)
    den = z.T @ z

    return num / den

def test_morans_i_eigenvectors():
    """
    Verifies Moran's I on eigenvectors of a grid graph.
    1. Constant vector -> I = 1 (if not centered).
    2. Checkerboard vector -> I = -1.
    """
    n_rows, n_cols = 10, 10
    n = n_rows * n_cols
    W = build_grid_graph(n_rows, n_cols)
    W_std = row_standardize(W)

    # 1. Constant Vector
    # Note: morans_i_all_fast centers by default. We must disable centering to test the constant vector
    # (otherwise it becomes the zero vector and I is undefined/NaN).
    x_const = np.ones(n)

    adata_const = AnnData(X=x_const[:, None])
    adata_const.var_names = ["const"]

    # Test uncentered constant vector
    res_const = morans_i_all_fast(adata_const, W_rowstd=W_std, center=False)
    # The function upper-cases gene names by default due to deduplication
    i_const = res_const.set_index("gene").loc["CONST", "I"]

    # For a row-standardized matrix, the constant vector is an eigenvector with eval 1.
    # I = (n/S0) * (x'Wx)/(x'x). S0 = n for row-standardized.
    # I = (n/n) * (x'x)/(x'x) = 1.
    assert np.isclose(i_const, 1.0, atol=1e-5), f"Constant vector I should be 1.0, got {i_const}"

    # 2. Checkerboard Vector (Bipartite)
    # +1, -1 pattern
    x_check = np.zeros(n)
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            x_check[i] = 1 if (r + c) % 2 == 0 else -1

    adata_check = AnnData(X=x_check[:, None])
    adata_check.var_names = ["checkerboard"]

    # Checkerboard should be mean 0 (balanced grid), so center=True/False shouldn't matter much,
    # but let's stick to center=True (default) to test that path too.
    res_check = morans_i_all_fast(adata_check, W_rowstd=W_std, center=True)
    i_check = res_check.set_index("gene").loc["CHECKERBOARD", "I"]

    # For bipartite graph, x_check is eigenvector with eval -1.
    # I should be -1.
    assert np.isclose(i_check, -1.0, atol=1e-5), f"Checkerboard vector I should be -1.0, got {i_check}"


def test_morans_i_null_hypothesis():
    """
    Verifies that for random noise, E[I] = -1/(n-1).
    """
    n_rows, n_cols = 20, 20
    n = n_rows * n_cols
    W = build_grid_graph(n_rows, n_cols)
    W_std = row_standardize(W)

    # Generate many random genes
    n_genes = 2000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_genes))

    adata = AnnData(X=X)
    adata.var_names = [f"g_{i}" for i in range(n_genes)]

    res = morans_i_all_fast(adata, W_rowstd=W_std)

    mean_I = res["I"].mean()
    expected_I = -1.0 / (n - 1)

    # Tolerance check:
    # Var(I) approx 1/n = 1/400. Std(I) = 0.05.
    # SEM = 0.05 / sqrt(2000) ~ 0.001.
    # 3 sigma ~ 0.003.

    assert np.isclose(mean_I, expected_I, atol=0.005), \
        f"Mean I ({mean_I:.5f}) should be close to expected ({expected_I:.5f}) for random noise."


def test_morans_i_algebraic_correctness():
    """
    Verifies that the optimized expansion in `morans_i_all_fast` matches
    the naive calculation exactly (within float precision).
    """
    n = 50
    W = sp.random(n, n, density=0.1, random_state=42)
    W_std = row_standardize(W)

    rng = np.random.default_rng(123)
    x = rng.uniform(0, 10, n) # Non-centered data

    adata = AnnData(X=x[:, None])
    adata.var_names = ["gene1"]

    # Run optimized
    res_opt = morans_i_all_fast(adata, W_rowstd=W_std, center=True)
    i_opt = res_opt.iloc[0]["I"]

    # Run naive
    i_naive = naive_morans_i(x, W_std, center=True)

    assert np.isclose(i_opt, i_naive, atol=1e-5), \
        f"Optimized implementation ({i_opt}) does not match naive ({i_naive})"
