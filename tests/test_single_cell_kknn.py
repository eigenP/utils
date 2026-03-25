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
    n_genes = 20

    # Need real variance for sc.pp.pca to work
    adata_ref = sc.AnnData(rng.normal(size=(n_ref, n_genes)))
    adata_ref.obsm["X_umap"] = rng.normal(size=(n_ref, 2))
    adata_ref.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], size=n_ref))

    adata_query = sc.AnnData(rng.normal(size=(n_query, n_genes)))

    # We don't precompute PCA on query. kknn_ingest should do it.

    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["cell_type"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        n_neighbors=10,
        recompute_ref_PCA=True,
        save_ref_PCA_key="X_pca_projected"
    )

    # Check if projection was saved
    assert "X_pca_projected" in adata_query.obsm
    assert adata_query.obsm["X_pca_projected"].shape[0] == n_query

    # Check if other mappings worked
    assert "X_umap_kknn" in adata_query.obsm
    assert adata_query.obsm["X_umap_kknn"].shape == (n_query, 2)

    assert "cell_type_kknn" in adata_query.obs
    assert len(adata_query.obs["cell_type_kknn"]) == n_query
    assert isinstance(adata_query.obs["cell_type_kknn"].dtype, pd.CategoricalDtype)

    # Check if the k count was saved
    assert "kknn_k" in adata_query.obs

    assert "mapping_confidence_cell_type_kknn" in adata_query.obs
    conf = adata_query.obs["mapping_confidence_cell_type_kknn"].values
    assert np.all((conf >= 0) & (conf <= 1.0))

def test_kknn_ingest_no_recompute_no_save():
    rng = np.random.default_rng(42)
    n_ref = 100
    n_query = 10
    n_genes = 20

    adata_ref = sc.AnnData(rng.normal(size=(n_ref, n_genes)))
    sc.tl.pca(adata_ref)
    adata_ref.obsm["X_umap"] = rng.normal(size=(n_ref, 2))
    adata_ref.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], size=n_ref))

    adata_query = sc.AnnData(rng.normal(size=(n_query, n_genes)))

    # Check what happens without recomputing PCA and without saving the key (temporary key)
    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["cell_type"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        n_neighbors=10,
        recompute_ref_PCA=False,
        save_ref_PCA_key=None
    )

    # Temporary PCA projection should be deleted
    assert not any(k.startswith("__temp_ingest_") for k in adata_query.obsm.keys())

    # Still maps labels
    assert "cell_type_kknn" in adata_query.obs
