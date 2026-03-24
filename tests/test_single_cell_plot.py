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
        score_method=["scanpy", "binned", "binned_weighted", "net_scanpy", "net_binned", "net_binned_weighted"],
        use_raw=False
    )
    assert len(axes_multi) > 0

    # Assert temporary columns are cleaned up
    assert "Type1_score_scanpy" not in adata.obs
    assert "Type1_score_binned" not in adata.obs
    assert "Type1_score_binned_weighted" not in adata.obs
    assert "Type1_score_net_scanpy" not in adata.obs
    assert "Type1_score_net_binned" not in adata.obs
    assert "Type1_score_net_binned_weighted" not in adata.obs


def test_binned_vs_net_binned():
    # specifically test that binned and net_binned produce different scores when negative markers are present
    from eigenp_utils.single_cell import score_celltypes
    np.random.seed(42)
    # 10 cells, 2 positive genes, 1 negative gene
    X = np.random.uniform(0, 10, (10, 3))
    adata = anndata.AnnData(X=X)
    adata.var_names = ["P1", "P2", "N1"]

    markers = {"T1": ["P1", "P2"]}
    neg_markers = {"T1": ["N1"]}

    df_binned = score_celltypes(adata, markers, cell_type_negative_markers_dict=neg_markers, score_method="binned", use_raw=False)
    df_net_binned = score_celltypes(adata, markers, cell_type_negative_markers_dict=neg_markers, score_method="net_binned", use_raw=False)

    # "binned" should ignore neg_markers and just be positive
    # "net_binned" should be positive - negative
    # They should not be equal.
    assert not np.allclose(df_binned["T1"], df_net_binned["T1"]), "binned and net_binned should not be identical when negative markers exist"

    # Validate that net_binned scores are clipped to [0, 1]
    assert np.all((df_net_binned["T1"] >= 0.0) & (df_net_binned["T1"] <= 1.0)), "net_binned scores should be clipped between 0 and 1"

    # Also check if no negative markers passed, they are equal
    df_binned_noneg = score_celltypes(adata, markers, cell_type_negative_markers_dict=None, score_method="binned", use_raw=False)
    df_net_binned_noneg = score_celltypes(adata, markers, cell_type_negative_markers_dict=None, score_method="net_binned", use_raw=False)

    assert np.allclose(df_binned_noneg["T1"], df_net_binned_noneg["T1"]), "binned and net_binned should be identical when no negative markers exist"
