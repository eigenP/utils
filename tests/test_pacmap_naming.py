
import scanpy as sc
import numpy as np
import pytest
from eigenp_utils.single_cell import tl_pacmap

def test_pacmap_key_naming():
    try:
        import pacmap
    except ImportError:
        pytest.skip("PaCMAP not installed")

    n_obs = 50
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    # Test default (n_components=2) -> X_pacmap
    # n_neighbors=5 to avoid error on small dataset
    tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X")
    assert "X_pacmap" in adata.obsm
    assert "X_pacmap_2" not in adata.obsm

    # Test n_components=3 -> X_pacmap_3
    tl_pacmap(adata, n_neighbors=5, n_components=3, use_rep="X")
    assert "X_pacmap_3" in adata.obsm
    # Ensure dimensions are correct
    assert adata.obsm["X_pacmap_3"].shape == (n_obs, 3)

    # Test n_components=4 -> X_pacmap_4
    tl_pacmap(adata, n_neighbors=5, n_components=4, use_rep="X")
    assert "X_pacmap_4" in adata.obsm
    assert adata.obsm["X_pacmap_4"].shape == (n_obs, 4)

    # Verify X_pacmap is still the 2D one (it wasn't overwritten by 3 or 4)
    assert adata.obsm["X_pacmap"].shape == (n_obs, 2)
