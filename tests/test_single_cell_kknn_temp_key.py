import anndata as ad
import numpy as np
import pandas as pd
from eigenp_utils.single_cell import kknn_ingest

def test_kknn_ingest_temp_key_cleanup():
    # Create dummy reference
    X_ref = np.random.randn(100, 20)
    obs_ref = pd.DataFrame({"label": ["A", "B"] * 50})
    obsm_ref = {"X_pca": X_ref[:, :5], "X_umap": X_ref[:, :2]}
    varm_ref = {"PCs": np.random.randn(20, 5)}
    uns_ref = {"pca": {"params": {"zero_center": True, "use_highly_variable": False}}}

    adata_ref = ad.AnnData(X=X_ref, obs=obs_ref, obsm=obsm_ref, varm=varm_ref, uns=uns_ref)

    # Create dummy query
    X_query = np.random.randn(50, 20)
    adata_query = ad.AnnData(X=X_query)

    # We want to use use_rep="X_pca" to trigger the temp key path,
    # and use barycenter="lle" which accesses adata_query.obsm[query_use_rep]
    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["label"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        barycenter="lle",
        recompute_ref_PCA=False # avoid sc.tl.pca
    )

    # Verify temp key has been cleaned up
    assert not any(k.startswith("__temp_ingest_") for k in adata_query.obsm.keys())
