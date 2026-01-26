import numpy as np
import scanpy as sc
import pytest
from eigenp_utils.single_cell import run_triku

@pytest.fixture
def adata_triku_ready():
    """Returns an AnnData object with counts and log1p layer, ready for neighbors."""
    n_cells = 50
    n_genes = 100
    X = np.random.randint(0, 10, size=(n_cells, n_genes)).astype(float)
    obs = dict(batch=np.random.choice(["a", "b"], size=n_cells))
    var = dict(gene_name=[f"gene_{i}" for i in range(n_genes)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.layers["log1p"] = adata.X.copy()
    return adata

def test_triku_standard_neighbors(adata_triku_ready):
    """Test run_triku with standard neighbors (uns['neighbors'])."""
    try:
        import triku
    except ImportError:
        pytest.skip("triku not installed")

    # Compute standard neighbors
    sc.pp.pca(adata_triku_ready)
    sc.pp.neighbors(adata_triku_ready)

    # Run Triku
    res_df = run_triku(adata_triku_ready, layer="log1p", n_features=20)

    # Check outputs
    assert "triku_distance" in adata_triku_ready.var
    assert "triku_highly_variable" in adata_triku_ready.var
    assert not res_df.empty

def test_triku_custom_knn_key(adata_triku_ready):
    """Test run_triku with custom knn_key (e.g. 'scvi')."""
    try:
        import triku
    except ImportError:
        pytest.skip("triku not installed")

    # Compute custom neighbors
    sc.pp.pca(adata_triku_ready)
    sc.pp.neighbors(adata_triku_ready, key_added="scvi")

    # Verify we have the custom key but NOT standard neighbors
    assert "scvi" in adata_triku_ready.uns
    assert "neighbors" not in adata_triku_ready.uns

    # Run Triku with knn_key
    res_df = run_triku(adata_triku_ready, layer="log1p", knn_key="scvi", n_features=20)

    # Check outputs
    assert "triku_distance" in adata_triku_ready.var
    assert "triku_highly_variable" in adata_triku_ready.var
    assert not res_df.empty

    # Verify that run_triku didn't pollute the original object with 'neighbors'
    # (It modifies a copy internally)
    assert "neighbors" not in adata_triku_ready.uns

def test_triku_missing_neighbors_error(adata_triku_ready):
    """Test that run_triku raises ValueError if neighbors are missing."""
    try:
        import triku
    except ImportError:
        pytest.skip("triku not installed")

    with pytest.raises(ValueError, match="Neighbors not found in adata.uns"):
        run_triku(adata_triku_ready, layer="log1p")

def test_triku_missing_knn_key_error(adata_triku_ready):
    """Test that run_triku raises ValueError if provided knn_key is missing."""
    try:
        import triku
    except ImportError:
        pytest.skip("triku not installed")

    with pytest.raises(ValueError, match="Key 'non_existent' not found in adata.uns"):
        run_triku(adata_triku_ready, layer="log1p", knn_key="non_existent")
