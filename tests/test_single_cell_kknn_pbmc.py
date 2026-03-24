import numpy as np
import pandas as pd
import scanpy as sc
import pytest
from eigenp_utils.single_cell import compute_kknn_neighbors

def old_compute_kknn_neighbors(
    adata_query: sc.AnnData,
    adata_ref: sc.AnnData,
    use_rep: str = "X_pca",
    query_use_rep: str = "X_pca",
    n_neighbors: int = 10,
    min_neighbors: int = 3,
    max_neighbors: int = 20,
    quantile_bins: int = 10
):
    """The original unoptimized, heuristic method for comparison"""
    from sklearn.neighbors import NearestNeighbors

    X_query = adata_query.obsm[query_use_rep]
    X_ref = adata_ref.obsm[use_rep]

    N_ref = X_ref.shape[0]
    N_query = X_query.shape[0]

    nn = NearestNeighbors(n_neighbors=max_neighbors, algorithm='auto', n_jobs=-1)
    nn.fit(X_ref)
    distances, indices = nn.kneighbors(X_query)

    curvatures = np.zeros(N_query)
    m = X_ref.shape[1]

    for i in range(N_query):
        neighs_idx = indices[i]
        amostras = X_ref[neighs_idx]

        ni = len(neighs_idx)
        if ni > 1:
            I = np.cov(amostras, rowvar=False)
            if I.ndim == 0:
                I = np.array([[I]])
        else:
            I = np.eye(m)

        try:
            eigvals = np.linalg.eigvalsh(I)
        except np.linalg.LinAlgError:
            eigvals = np.ones(m)

        eigvals = np.maximum(eigvals, 0)
        total_var = np.sum(eigvals)

        if total_var > 0:
            num_small = max(1, m // 2)
            curvatures[i] = np.sum(eigvals[:num_small]) / total_var
        else:
            curvatures[i] = 0.0

    ptp = curvatures.max() - curvatures.min()
    if ptp == 0:
        K = np.zeros_like(curvatures)
    else:
        K = (curvatures - curvatures.min()) / (ptp + 1e-9)

    intervalos = np.linspace(0.0, 1.0, quantile_bins + 1)[1:-1]
    quantis = np.quantile(K, intervalos)
    bins = np.array(quantis)
    disc_curv = np.digitize(K, bins)

    pruned_distances = []
    pruned_indices = []

    for i in range(N_query):
        bin_idx = disc_curv[i]
        fraction = bin_idx / max(1, (quantile_bins - 1))
        keep = int(round(max_neighbors - fraction * (max_neighbors - min_neighbors)))
        keep = max(min_neighbors, min(max_neighbors, keep))

        pruned_distances.append(distances[i, :keep])
        pruned_indices.append(indices[i, :keep])

    return pruned_distances, pruned_indices


def test_compute_kknn_neighbors_pbmc_concordance():
    # Use the pbmc3k dataset
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')

    # Split into ref and query (70/30)
    rng = np.random.default_rng(42)
    n_cells = adata.n_obs
    idx = rng.permutation(n_cells)
    split = int(0.7 * n_cells)

    adata_ref = adata[idx[:split]].copy()
    adata_query = adata[idx[split:]].copy()

    # Run the NEW method
    new_dists, new_idxs = compute_kknn_neighbors(
        adata_query,
        adata_ref,
        n_neighbors=15,
        min_neighbors=5,
        max_neighbors=30
    )

    # Run the OLD method
    old_dists, old_idxs = old_compute_kknn_neighbors(
        adata_query,
        adata_ref,
        n_neighbors=15,
        min_neighbors=5,
        max_neighbors=30
    )

    # Compare
    new_lengths = np.array([len(idx) for idx in new_idxs])
    old_lengths = np.array([len(idx) for idx in old_idxs])

    # They shouldn't be identical because the math is different, but they should be
    # highly correlated (i.e. areas of high curvature still prune more neighbors)
    from scipy.stats import spearmanr
    corr, _ = spearmanr(new_lengths, old_lengths)

    print(f"Spearman correlation between old and new neighborhood sizes: {corr:.3f}")
    assert corr > 0.3, "The new dimensionality metric should broadly correlate with the old curvature metric."

    # The new method should still obey the bounds
    assert new_lengths.min() >= 5
    assert new_lengths.max() <= 30
