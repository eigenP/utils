
import pytest
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
from eigenp_utils.single_cell import morans_i_all_fast

def test_morans_i_correctness():
    """
    Verify that the optimized morans_i_all_fast produces the same results
    as a reference implementation (or just self-consistency checks).
    Since we don't have the 'old' function readily available as a separate import,
    we check against known properties and simple manual calculation for small data.
    """
    # 1. Create small synthetic data
    n_cells = 50
    n_genes = 10

    # Gene expression: random
    rng = np.random.default_rng(42)
    X = rng.random((n_cells, n_genes)).astype(np.float32)

    # Neighbors: random graph
    # Create a symmetric adjacency matrix
    A = rng.random((n_cells, n_cells)) < 0.2
    np.fill_diagonal(A, 0)
    A = A.astype(np.float32)
    # Symmetrize
    A = (A + A.T) > 0
    A = A.astype(np.float32)

    adata = sc.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Store connectivities
    adata.obsp["connectivities"] = sp.csr_matrix(A)

    # 2. Run Moran's I
    # We run with block_genes=2 to force looping and block handling
    df = morans_i_all_fast(adata, block_genes=2, center=True)

    assert "gene" in df.columns
    assert "I" in df.columns
    assert "z_score" in df.columns
    assert "pval_z" in df.columns

    assert len(df) == n_genes

    # Check that I values are within [-1, 1] (mostly)
    # Moran's I can exceed bounds slightly but usually fits.
    # For random data, I should be close to 0.

    print("\nMoran's I results (head):")
    print(df.head())

    # Verify no NaNs in I (unless variance is 0)
    assert not df["I"].isna().any()

    # 3. Check consistency with manual calculation for first gene
    g0 = df.iloc[0]["gene"] # gene with highest I
    idx = int(g0.split("_")[1])
    x = X[:, idx]

    # Row normalize W
    rs = np.array(A.sum(axis=1)).flatten()
    W_norm = A / rs[:, None]

    # Moran's I formula: (N/S0) * (z' W z) / (z' z)
    # Here S0 = N because of row normalization? No, S0 = sum(W_norm).
    S0 = W_norm.sum()
    x_mean = x.mean()
    z = x - x_mean

    num = (z @ W_norm @ z)
    den = (z @ z)

    I_manual = (n_cells / S0) * (num / den)

    I_calc = df.iloc[0]["I"]

    print(f"Manual I for {g0}: {I_manual}")
    print(f"Calculated I: {I_calc}")

    assert np.isclose(I_manual, I_calc, atol=1e-5)

def test_morans_i_sparse_input():
    """Check handling of sparse input matrices."""
    n_cells = 100
    n_genes = 20
    rng = np.random.default_rng(123)
    X = sp.random(n_cells, n_genes, density=0.1, format="csr", dtype=np.float32)

    # Create connectivity
    A = sp.random(n_cells, n_cells, density=0.1, format="csr", dtype=np.float32)

    adata = sc.AnnData(X=X)
    adata.obsp["connectivities"] = A
    adata.var_names = [f"G{i}" for i in range(n_genes)]

    # Run
    df = morans_i_all_fast(adata, block_genes=5)
    assert len(df) == n_genes
    assert not df["I"].isna().all()

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_morans_i_correctness()
        print("test_morans_i_correctness PASSED")
        test_morans_i_sparse_input()
        print("test_morans_i_sparse_input PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        raise
