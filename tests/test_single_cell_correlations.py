import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from eigenp_utils.single_cell import find_correlated_features

@pytest.fixture
def pbmc_adata():
    # Create a dummy PBMC-like dataset
    np.random.seed(42)
    n_cells = 100
    n_genes = 20

    # Base expression
    X = np.random.poisson(lam=1.0, size=(n_cells, n_genes)).astype(float)

    # Introduce correlation
    # Gene 0 and Gene 1 are highly correlated
    X[:, 1] = X[:, 0] * 2 + np.random.normal(scale=0.5, size=n_cells)

    # Gene 0 and Gene 2 are anti-correlated
    X[:, 2] = -X[:, 0] + np.random.normal(scale=0.5, size=n_cells) + 10

    # Ensure non-negative
    X[X < 0] = 0

    adata = sc.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]

    # Add an obs column that correlates with Gene 0
    adata.obs["score_A"] = adata.X[:, 0].toarray().ravel() + np.random.normal(scale=0.1, size=n_cells)

    # Add a dense layer
    adata.layers["counts"] = adata.X.toarray()

    return adata


def test_find_correlated_features_pearson_sparse(pbmc_adata):
    """Test Pearson correlation calculation on sparse matrices."""
    res = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"]
    )

    assert "pearson" in res.columns
    assert res.index[0] == "Gene0"
    assert res.loc["Gene0", "pearson"] == pytest.approx(1.0)
    assert res.loc["Gene1", "pearson"] > 0.8  # Highly correlated
    assert res.loc["Gene2", "pearson"] < -0.8 # Anti-correlated


def test_find_correlated_features_dense_layer(pbmc_adata):
    """Test correlation on dense layer vs sparse X."""
    res_sparse = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"]
    )

    res_dense = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        layer="counts",
        metrics=["pearson"]
    )

    np.testing.assert_allclose(
        res_sparse["pearson"].values,
        res_dense["pearson"].values,
        atol=1e-6
    )


def test_find_correlated_features_target_obs(pbmc_adata):
    """Test using an .obs column as the target."""
    res = find_correlated_features(
        pbmc_adata,
        target="score_A",
        metrics=["pearson"]
    )

    assert res.loc["Gene0", "pearson"] > 0.9
    assert res.loc["Gene1", "pearson"] > 0.8
    assert res.loc["Gene2", "pearson"] < -0.8


def test_find_correlated_features_multiple_metrics(pbmc_adata):
    """Test requesting multiple metrics at once."""
    res = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson", "spearman", "wasserstein"]
    )

    assert all(col in res.columns for col in ["pearson", "spearman", "wasserstein"])

    # Wasserstein distance of a variable to itself (Z-scored) should be 0
    assert res.loc["Gene0", "wasserstein"] == pytest.approx(0.0, abs=1e-7)

    # Spearman should also show strong correlation
    assert res.loc["Gene1", "spearman"] > 0.8


def test_find_correlated_features_exclude(pbmc_adata):
    """Test the exclude_features parameter."""
    res = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        exclude_features=["Gene0", "Gene1", "NonExistentGene"]
    )

    assert "Gene0" not in res.index
    assert "Gene1" not in res.index
    assert "Gene2" in res.index


def test_find_correlated_features_graph_smoothing(pbmc_adata):
    """Test graph-smoothed feature correlation calculation."""
    # Compute neighbors so `adata.obsp['connectivities']` exists
    sc.pp.neighbors(pbmc_adata, n_neighbors=5, use_rep="X")

    # Store a copy of original data to ensure it is not mutated
    X_orig = pbmc_adata.X.copy()
    connectivities_orig = pbmc_adata.obsp["connectivities"].copy()

    res_no_graph = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"]
    )

    res_with_graph = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"],
        use_graph=True,
        weights_key="connectivities"
    )

    # Both should have computed pearson correlation
    assert "pearson" in res_with_graph.columns

    # Gene0 vs Gene0 should still be 1.0
    assert res_with_graph.loc["Gene0", "pearson"] == pytest.approx(1.0)

    # The diffused distances will be numerically different from the raw ones
    assert res_with_graph.loc["Gene1", "pearson"] != res_no_graph.loc["Gene1", "pearson"]

    # Ensure no mutation of original data
    assert (pbmc_adata.X != X_orig).nnz == 0
    assert (pbmc_adata.obsp["connectivities"] != connectivities_orig).nnz == 0


def test_find_correlated_features_sorting(pbmc_adata):
    """Test sorting behavior (wasserstein ascending, others descending)."""
    # Pearson: descending
    res_pearson = find_correlated_features(pbmc_adata, target="Gene0", metrics=["pearson"])
    assert res_pearson.index[0] == "Gene0"
    assert res_pearson.index[-1] == "Gene2" # Most anti-correlated

    # Wasserstein: ascending (0 is best)
    res_wass = find_correlated_features(pbmc_adata, target="Gene0", metrics=["wasserstein"])
    assert res_wass.index[0] == "Gene0"

    # Multiple metrics: sorts by first
    res_multi = find_correlated_features(
        pbmc_adata, target="Gene0", metrics=["wasserstein", "pearson"]
    )
    assert res_multi.index[0] == "Gene0"
    assert list(res_multi.index) == list(res_wass.index)
