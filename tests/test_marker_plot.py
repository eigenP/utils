
import pytest
import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
from unittest.mock import patch
from eigenp_utils.single_cell import plot_marker_genes_dict_on_embedding

def test_plot_marker_genes_dict_on_embedding():
    # Setup dummy AnnData
    n_obs = 100
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Add dummy UMAP
    adata.obsm["X_umap"] = np.random.rand(n_obs, 2)

    # Define marker genes (some existing, some missing to test check_gene_adata implicitly)
    marker_genes = {
        "TissueA": ["gene_0", "gene_1"],
        "TissueB": ["gene_2", "gene_missing"]
    }

    # Mock score_genes to avoid data distribution requirements and ensure it "succeeds"
    # We need side_effect to actually add the score to adata.obs so sc.pl.embedding can plot it
    def score_genes_side_effect(adata, gene_list, score_name, **kwargs):
        adata.obs[score_name] = np.random.rand(adata.n_obs)

    with patch("scanpy.tl.score_genes", side_effect=score_genes_side_effect):
        # Run function
        axes_list = plot_marker_genes_dict_on_embedding(
            adata,
            marker_genes,
            basis="X_umap",
            show=False # Ensure it doesn't block
        )

    # Assertions
    assert isinstance(axes_list, list)
    # TissueA has 2 valid genes + 1 score -> 3 plots
    # TissueB has 1 valid gene + 1 score -> 2 plots
    # Total = 5
    assert len(axes_list) == 5, f"Expected 5 axes, got {len(axes_list)}"

    for ax in axes_list:
        assert isinstance(ax, plt.Axes)
        # Check if title or label logic worked (optional, but good)
        # We can check if ylabel is set as expected (Tissue Name + \n)

    print("Test passed successfully!")

def test_missing_basis():
    n_obs = 10
    n_vars = 10
    adata = AnnData(X=np.random.rand(n_obs, n_vars))
    marker_genes = {"A": ["gene_0"]}

    try:
        plot_marker_genes_dict_on_embedding(adata, marker_genes, basis="X_pca")
    except ValueError as e:
        assert "compute it and add in obsm, or choose from available keys" in str(e)
        print("Missing basis test passed!")
        return

    raise AssertionError("Did not raise ValueError for missing basis")

if __name__ == "__main__":
    # Manually running checks if not using pytest directly,
    # but we will likely run with pytest or python
    try:
        test_plot_marker_genes_dict_on_embedding()
        test_missing_basis()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed: {repr(e)}")
        exit(1)
