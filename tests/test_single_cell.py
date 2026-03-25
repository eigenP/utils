import pytest
import scanpy as sc
import numpy as np
from eigenp_utils.single_cell import compute_kknn_neighbors

def test_kknn_caching():
    # Setup dummy data
    np.random.seed(42)
    adata_ref = sc.AnnData(np.random.rand(100, 10))
    adata_ref.obsm['X_pca'] = np.random.rand(100, 5)

    adata_query = sc.AnnData(np.random.rand(50, 10))
    adata_query.obsm['X_pca'] = np.random.rand(50, 5)

    # Run first time, should cache
    assert 'kknn_curvature_bounds' not in adata_ref.uns
    distances1, indices1 = compute_kknn_neighbors(adata_query, adata_ref, n_neighbors=10)
    assert 'kknn_curvature_bounds' in adata_ref.uns

    # Store bounds to compare
    bounds1 = adata_ref.uns['kknn_curvature_bounds']

    # Run second time, should use cache
    distances2, indices2 = compute_kknn_neighbors(adata_query, adata_ref, n_neighbors=10)
    bounds2 = adata_ref.uns['kknn_curvature_bounds']

    # Compare
    assert bounds1 == bounds2
    np.testing.assert_array_equal(indices1[0], indices2[0])
    np.testing.assert_array_almost_equal(distances1[0], distances2[0])


def test_kknn_same_query_ref():
    import pandas as pd
    from eigenp_utils.single_cell import kknn_classifier
    np.random.seed(42)
    adata = sc.AnnData(np.random.rand(100, 10))
    adata.obsm['X_pca'] = np.random.rand(100, 5)
    adata.obs['labels'] = np.random.choice(['A', 'B'], 100)

    # Run the classifier logic, which sets query=ref
    kknn_classifier(adata, 'labels', use_rep='X_pca', inplace=True)

    assert 'labels_kknn' in adata.obs

    # Run it again, checking cache is hit
    bounds1 = adata.uns['kknn_curvature_bounds']
    kknn_classifier(adata, 'labels', use_rep='X_pca', inplace=True)
    bounds2 = adata.uns['kknn_curvature_bounds']
    assert bounds1 == bounds2


def test_kknn_ingest_caching():
    import scanpy as sc
    import numpy as np
    from eigenp_utils.single_cell import kknn_ingest

    np.random.seed(42)
    adata_ref = sc.AnnData(np.random.rand(100, 10))
    adata_ref.var_names = [f"gene_{i}" for i in range(10)]
    adata_ref.obsm['X_pca'] = np.random.rand(100, 5)
    adata_ref.obs['cell_type'] = np.random.choice(['T', 'B', 'M'], 100)

    # Needs PCs for ingest
    adata_ref.uns['pca'] = {'params': {'zero_center': True, 'use_highly_variable': False}}
    adata_ref.varm['PCs'] = np.random.rand(10, 5)

    adata_query1 = sc.AnnData(np.random.rand(50, 10))
    adata_query1.var_names = [f"gene_{i}" for i in range(10)]

    adata_query2 = sc.AnnData(np.random.rand(30, 10))
    adata_query2.var_names = [f"gene_{i}" for i in range(10)]

    # Ingest first batch
    assert 'kknn_curvature_bounds' not in adata_ref.uns
    kknn_ingest(adata_query1, adata_ref, obs_keys=['cell_type'], recompute_ref_PCA=False)
    assert 'kknn_curvature_bounds' in adata_ref.uns

    bounds1 = adata_ref.uns['kknn_curvature_bounds']

    # Ingest second batch
    kknn_ingest(adata_query2, adata_ref, obs_keys=['cell_type'], recompute_ref_PCA=False)
    bounds2 = adata_ref.uns['kknn_curvature_bounds']

    assert bounds1 == bounds2
