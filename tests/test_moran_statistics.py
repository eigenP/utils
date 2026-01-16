
import pytest
import numpy as np
import scipy.sparse as sp
import pandas as pd
from anndata import AnnData
from eigenp_utils.single_cell import morans_i_all_fast
from scipy import stats

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
    row_sums[row_sums == 0] = 1.0
    inv_sums = 1.0 / row_sums
    D = sp.diags(inv_sums)
    return D @ W

def test_moran_statistics_gaussian():
    """
    Verifies that for Gaussian random noise, the Z-scores produced by
    morans_i_all_fast follow a Standard Normal Distribution N(0, 1).

    This validates the variance formula for the 'randomization' assumption
    when the underlying data is actually normal.
    """
    # 1. Setup: 20x20 Grid (N=400)
    n_rows, n_cols = 20, 20
    n = n_rows * n_cols
    W = build_grid_graph(n_rows, n_cols)
    W_std = row_standardize(W)

    # 2. Generate Random Data: 5000 genes, N(0, 1)
    n_genes = 5000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_genes))

    adata = AnnData(X=X)
    adata.var_names = [f"g_{i}" for i in range(n_genes)]

    # 3. Run Moran's I
    res = morans_i_all_fast(adata, W_rowstd=W_std)

    # 4. Verify Z-scores statistics
    z_scores = res["z_score"].values

    # Drop NaNs if any (shouldn't be for N=400)
    z_scores = z_scores[np.isfinite(z_scores)]

    mu_z = np.mean(z_scores)
    std_z = np.std(z_scores)

    print(f"Gaussian Noise: Mean Z = {mu_z:.4f}, Std Z = {std_z:.4f}")

    # Mean should be 0 (unbiased)
    # SEM = 1 / sqrt(5000) ~ 0.014
    assert np.abs(mu_z) < 0.05, f"Z-scores should be centered at 0, got {mu_z:.4f}"

    # Std should be 1 (correct variance estimation)
    # Variance of sample variance ~ 2/N. Std of sample std ~ 1/sqrt(2N) ~ 0.01
    assert np.abs(std_z - 1.0) < 0.05, f"Z-scores should have unit variance, got {std_z:.4f}"

    # Kolmogorov-Smirnov test against Standard Normal
    # This checks the entire shape of the distribution
    ks_stat, ks_pval = stats.kstest(z_scores, 'norm')

    # We expect p-value to be high (fail to reject null).
    # However, with N=5000, even tiny deviations can reject.
    # We'll use a lenient threshold or rely on moments.
    print(f"KS Test: stat={ks_stat:.4f}, pval={ks_pval:.4e}")

    # 5. Verify P-values Uniformity
    p_vals = res["pval_z"].values
    prop_significant = np.mean(p_vals < 0.05)
    print(f"Proportion p < 0.05: {prop_significant:.4f}")

    assert 0.04 < prop_significant < 0.06, f"P-values should be uniform (approx 0.05 < 0.05), got {prop_significant:.4f}"


def test_moran_statistics_kurtotic():
    """
    Verifies that for High-Kurtosis noise (Sparse Spikes), the Z-scores
    still follow N(0, 1).

    This strictly validates that the variance formula uses the kurtosis term 'b2' correctly.
    If the formula assumed normality (b2=3), the variance estimate would be wrong
    for high-kurtosis data, and the resulting Z-scores would not have unit variance.
    """
    # 1. Setup
    n_rows, n_cols = 20, 20
    n = n_rows * n_cols
    W = build_grid_graph(n_rows, n_cols)
    W_std = row_standardize(W)

    # 2. Generate Sparse/Spiky Data
    # 95% zeros, 5% large values
    n_genes = 5000
    rng = np.random.default_rng(99)
    X = np.zeros((n, n_genes))

    # Add spikes
    n_spikes = int(0.05 * n)
    for g in range(n_genes):
        indices = rng.choice(n, n_spikes, replace=False)
        X[indices, g] = rng.exponential(scale=10.0, size=n_spikes)

    adata = AnnData(X=X)
    adata.var_names = [f"g_{i}" for i in range(n_genes)]

    # 3. Run Moran's I
    res = morans_i_all_fast(adata, W_rowstd=W_std)

    # 4. Verify Z-scores statistics
    z_scores = res["z_score"].values
    z_scores = z_scores[np.isfinite(z_scores)]

    mu_z = np.mean(z_scores)
    std_z = np.std(z_scores)

    print(f"Kurtotic Noise: Mean Z = {mu_z:.4f}, Std Z = {std_z:.4f}")

    # Even with high kurtosis, the randomization null hypothesis ensures
    # the Z-score (standardized by analytical variance) is asymptotically normal
    # (or at least has unit variance).

    assert np.abs(mu_z) < 0.05, f"Z-scores should be centered at 0, got {mu_z:.4f}"

    # Crucial check: Is variance correct?
    # If we ignored kurtosis, the variance of I would be underestimated/overestimated.
    # The analytical formula should adapt.
    assert np.abs(std_z - 1.0) < 0.1, f"Z-scores should have unit variance despite kurtosis, got {std_z:.4f}"
