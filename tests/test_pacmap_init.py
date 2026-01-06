
import pytest
import scanpy as sc
import numpy as np
import warnings
from eigenp_utils.single_cell import tl_pacmap

try:
    import pacmap
    PACMAP_INSTALLED = True
except ImportError:
    PACMAP_INSTALLED = False

@pytest.mark.skipif(not PACMAP_INSTALLED, reason="PaCMAP not installed")
def test_tl_pacmap_init_large_features():
    # Case 1: > 100 features (should default to PCA)
    n_obs = 50
    n_vars = 101
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    # Run pacmap
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always") # Cause all warnings to always be triggered.
        tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X")

        # Check that NO "Switching initialization" warning was issued
        for warning in w:
            assert "Switching initialization" not in str(warning.message)

    assert "X_pacmap" in adata.obsm

@pytest.mark.skipif(not PACMAP_INSTALLED, reason="PaCMAP not installed")
def test_tl_pacmap_init_override():
    # Case 3: <= 100 features but user specifies init='pca'
    # Should use 'pca' and NOT warn about switching
    n_obs = 50
    n_vars = 50 # <= 100
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Pass init explicitly via kwargs
        tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X", init="pca")

        # Check that NO "Switching initialization" warning was issued
        for warning in w:
            assert "Switching initialization" not in str(warning.message)

    assert "X_pacmap" in adata.obsm

@pytest.mark.skipif(not PACMAP_INSTALLED, reason="PaCMAP not installed")
def test_tl_pacmap_init_small_features():
    # Case 2: <= 100 features (should switch to random and warn)
    n_obs = 50
    n_vars = 50 # <= 100
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    # Run pacmap
    # Expect a warning about switching initialization
    with pytest.warns(UserWarning, match="Switching initialization"):
        tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X")

    assert "X_pacmap" in adata.obsm
