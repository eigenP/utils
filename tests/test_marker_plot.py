
import pytest
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eigenp_utils.single_cell import plot_marker_genes_dict_on_embedding

def test_plot_marker_genes_dict_on_embedding():
    # 1. Create a dummy AnnData object
    n_obs = 100
    n_vars = 50
    adata = sc.AnnData(np.random.rand(n_obs, n_vars))

    # Add fake gene names
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]

    # Add fake UMAP embedding
    adata.obsm['X_umap'] = np.random.rand(n_obs, 2)

    # Add fake clusters
    adata.obs['leiden'] = np.random.choice(['0', '1', '2'], size=n_obs)

    # 2. Define a marker dictionary
    marker_dict = {
        "Group A": ["gene_0", "gene_1"],
        "Group B": ["gene_2"]
    }

    # 3. Call the function
    axes = plot_marker_genes_dict_on_embedding(
        adata,
        marker_dict,
        use_rep="X_umap",
        show=False
    )

    # 4. Verify output
    assert isinstance(axes, list)
    assert len(axes) > 0
    # Expected number of plots: 3 genes = 3 plots
    # But wait, sc.pl.embedding with color=list plots multiple panels.
    # The function iterates over groups.
    # Group A: 2 genes -> 2 panels? Or 1? scanpy plots each gene in a panel.
    # Group B: 1 gene -> 1 panel.
    # Total panels: 3.
    assert len(axes) == 3
    for ax in axes:
        assert isinstance(ax, plt.Axes)
