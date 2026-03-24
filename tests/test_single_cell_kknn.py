import numpy as np
import pandas as pd
import scanpy as sc
import pytest
from eigenp_utils.single_cell import compute_kknn_neighbors, kknn_ingest

def test_compute_kknn_neighbors():
    rng = np.random.default_rng(42)
    n_ref = 100
    n_query = 10
    n_pcs = 5

    X_ref = rng.normal(size=(n_ref, n_pcs))
    X_query = rng.normal(size=(n_query, n_pcs))

    adata_ref = sc.AnnData(np.zeros((n_ref, 10)))
    adata_ref.obsm["X_pca"] = X_ref

    adata_query = sc.AnnData(np.zeros((n_query, 10)))
    adata_query.obsm["X_pca"] = X_query

    dists, idxs = compute_kknn_neighbors(
        adata_query,
        adata_ref,
        n_neighbors=10,
        min_neighbors=3,
        max_neighbors=20
    )

    assert len(dists) == n_query
    assert len(idxs) == n_query

    # Check that lengths are between min and max
    lengths = [len(d) for d in dists]
    assert min(lengths) >= 3
    assert max(lengths) <= 20

def test_kknn_ingest():
    rng = np.random.default_rng(42)
    n_ref = 100
    n_query = 10
    n_pcs = 5

    X_ref = rng.normal(size=(n_ref, n_pcs))
    X_query = rng.normal(size=(n_query, n_pcs))

    adata_ref = sc.AnnData(np.zeros((n_ref, 10)))
    adata_ref.obsm["X_pca"] = X_ref
    adata_ref.obsm["X_umap"] = rng.normal(size=(n_ref, 2))
    adata_ref.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], size=n_ref))

    adata_query = sc.AnnData(np.zeros((n_query, 10)))
    adata_query.obsm["X_pca"] = X_query

    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["cell_type"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        n_neighbors=10
    )

    assert "X_umap" in adata_query.obsm
    assert adata_query.obsm["X_umap"].shape == (n_query, 2)

    assert "cell_type" in adata_query.obs
    assert len(adata_query.obs["cell_type"]) == n_query
    assert isinstance(adata_query.obs["cell_type"].dtype, pd.CategoricalDtype)

    assert "mapping_confidence_cell_type" in adata_query.obs
    conf = adata_query.obs["mapping_confidence_cell_type"].values
    assert np.all((conf >= 0) & (conf <= 1.0))
