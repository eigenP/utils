import numpy as np
import scanpy as sc
import pytest
from eigenp_utils.single_cell import preprocess_subset

@pytest.fixture
def adata_integer_counts():
    """Adata with integer counts in .X and .layers['counts']"""
    X = np.random.randint(0, 10, size=(100, 50)).astype(float)
    obs = dict(batch=np.random.choice(["a", "b"], size=100))
    var = dict(gene_name=[f"gene_{i}" for i in range(50)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata

@pytest.fixture
def adata_float_layer():
    """Adata with float layer (e.g. scvi_normalized) and NO counts layer."""
    X = np.random.exponential(size=(100, 50)) + 0.1 # Ensure positive
    obs = dict(batch=np.random.choice(["a", "b"], size=100))
    var = dict(gene_name=[f"gene_{i}" for i in range(50)])
    # Populate .X with X so filtering doesn't drop everything
    adata = sc.AnnData(X=X.copy(), obs=obs, var=var)
    adata.layers["scvi_normalized"] = X.copy()
    return adata

def test_standard_workflow(adata_integer_counts):
    """Test standard workflow with integer counts: log1p, scaling, etc."""
    adata = preprocess_subset(
        adata_integer_counts,
        counts_layer="counts",
        X_layer_for_pca="log1p",
        hvg_flavor="seurat",
        scale_max_value=10.0,
        copy=True
    )

    assert "log1p" in adata.layers
    # Check if data was scaled (mean approx 0)
    assert np.allclose(adata.X.mean(axis=0), 0, atol=1.0) # lenient check
    assert adata.shape[1] == 50 # No subsetting unless n_top_genes < 50
    assert "X_pca" in adata.obsm

def test_custom_layer_workflow(adata_float_layer):
    """Test using a custom float layer (e.g. scvi) without counts."""
    # Should warn about missing counts but proceed
    with pytest.warns(UserWarning, match="Counts layer .* not found"):
        adata = preprocess_subset(
            adata_float_layer,
            counts_layer="counts", # Does not exist
            X_layer_for_pca="scvi_normalized",
            hvg_flavor="seurat", # standard seurat (not v3) works on data
            scale_data=False, # New parameter to skip scaling
            copy=True
        )

    # log1p should NOT be created if not requested and counts missing
    assert "log1p" not in adata.layers
    # X should be the scvi layer (unscaled)
    assert np.allclose(adata.X, adata.layers["scvi_normalized"])
    assert "X_pca" in adata.obsm

def test_custom_layer_scaling(adata_float_layer):
    """Test using a custom float layer WITH scaling."""
    with pytest.warns(UserWarning, match="Counts layer .* not found"):
        adata = preprocess_subset(
            adata_float_layer,
            counts_layer="counts",
            hvg_flavor="seurat", # seurat_v3 would fail without counts
            X_layer_for_pca="scvi_normalized",
            scale_data=True,
            copy=True
        )

    # X should be scaled
    assert not np.allclose(adata.X, adata.layers["scvi_normalized"])
    assert np.allclose(adata.X.mean(axis=0), 0, atol=1.0)

def test_seurat_v3_missing_counts_error(adata_float_layer):
    """Test that seurat_v3 flavor raises error if counts are missing."""
    with pytest.raises(ValueError, match="flavor='seurat_v3' requires raw counts"):
        preprocess_subset(
            adata_float_layer,
            hvg_flavor="seurat_v3",
            counts_layer="counts",
            copy=True
        )

