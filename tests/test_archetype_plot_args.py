
import scanpy as sc
import numpy as np
import pandas as pd
import anndata
from unittest.mock import patch, MagicMock
from eigenp_utils.single_cell import plot_archetype_summary

def create_dummy_adata():
    X = np.random.rand(10, 5)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(5)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = np.random.rand(10, 2)
    return adata

def create_dummy_archetype_results():
    return {
        "archetypes": np.random.rand(3, 10), # 3 archetypes, 10 cells
        "clusters": np.array([1, 1, 2, 3, 3]), # 5 genes
        "gene_corrs": np.random.rand(5),
        "gene_list": [f"gene_{i}" for i in range(5)]
    }

@patch("scanpy.pl.embedding")
def test_plot_archetype_summary_defaults(mock_embedding):
    """Test default colormaps."""
    adata = create_dummy_adata()
    results = create_dummy_archetype_results()

    plot_archetype_summary(adata, results, archetype_id=1, k=2)

    assert mock_embedding.call_count == 2

    # First call: Archetype score. Should use 'PiYG' by default.
    call1_args, call1_kwargs = mock_embedding.call_args_list[0]
    assert call1_kwargs.get('cmap') == 'PiYG'
    assert 'archetype_1_score' in call1_kwargs.get('color') or call1_kwargs.get('color') == 'archetype_1_score'

    # Second call: Top genes. Should use 'Purples' by default.
    call2_args, call2_kwargs = mock_embedding.call_args_list[1]
    assert call2_kwargs.get('cmap') == 'Purples'

@patch("scanpy.pl.embedding")
def test_plot_archetype_summary_custom(mock_embedding):
    """Test custom colormaps."""
    adata = create_dummy_adata()
    results = create_dummy_archetype_results()

    plot_archetype_summary(adata, results, archetype_id=1, k=2, cmap="Reds", archetype_cmap="Blues")

    assert mock_embedding.call_count == 2

    # First call: Archetype score. Should use 'Blues'.
    _, call1_kwargs = mock_embedding.call_args_list[0]
    assert call1_kwargs.get('cmap') == 'Blues'

    # Second call: Top genes. Should use 'Reds'.
    _, call2_kwargs = mock_embedding.call_args_list[1]
    assert call2_kwargs.get('cmap') == 'Reds'
