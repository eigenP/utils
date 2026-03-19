import numpy as np
import anndata
import matplotlib
matplotlib.use('Agg') # prevent plotting windows
import matplotlib.pyplot as plt
from eigenp_utils.single_cell import plot_marker_genes_dict_on_embedding

def test_plot_marker_genes_dict_on_embedding_methods():
    np.random.seed(42)
    X = np.random.uniform(0, 10, (20, 3))
    adata = anndata.AnnData(X=X)
    adata.var_names = ["G0", "G1", "G2"]
    adata.obsm["X_umap"] = np.random.uniform(0, 1, (20, 2))

    markers = {
        "Type1": ["G0", "G1"],
    }
    neg_markers = {
        "Type1": ["G2"]
    }

    # Test default
    axes = plot_marker_genes_dict_on_embedding(adata, markers)
    assert len(axes) > 0
    assert "Type1_score" not in adata.obs # ensure cleaned up

    # Test binned
    axes_binned = plot_marker_genes_dict_on_embedding(adata, markers, score_method="binned", use_raw=False)
    assert len(axes_binned) > 0

    # Test multiple methods with negative markers
    axes_multi = plot_marker_genes_dict_on_embedding(
        adata,
        markers,
        negative_marker_genes=neg_markers,
        score_method=["scanpy", "binned", "binned_weighted"],
        use_raw=False
    )
    assert len(axes_multi) > 0

    # Assert temporary columns are cleaned up
    assert "Type1_score_scanpy" not in adata.obs
    assert "Type1_score_binned" not in adata.obs
    assert "Type1_score_binned_weighted" not in adata.obs
