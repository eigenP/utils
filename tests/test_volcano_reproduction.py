
import pytest
import numpy as np
import scanpy as sc
import pandas as pd
from eigenp_utils.single_cell import plot_volcano_adata
import matplotlib.pyplot as plt

def test_volcano_reproduction():
    # Create a dummy AnnData
    adata = sc.AnnData(np.random.rand(100, 50))
    adata.obs['leiden'] = np.random.choice(['0', '1'], 100)
    adata.var_names = [f'Gene{i}' for i in range(50)]

    # Mock rank_genes_groups output
    # Scanpy stores it as structured arrays usually
    # We define dtypes to mimic structured arrays
    dtype = [('0', 'f4'), ('1', 'f4')]

    # Create structured arrays
    pvals_adj_data = np.zeros((50,), dtype=dtype)
    pvals_adj_data['0'] = np.random.rand(50)
    pvals_adj_data['1'] = np.random.rand(50)

    logfoldchanges_data = np.zeros((50,), dtype=dtype)
    logfoldchanges_data['0'] = np.random.randn(50)
    logfoldchanges_data['1'] = np.random.randn(50)

    names_dtype = [('0', 'U10'), ('1', 'U10')]
    names_data = np.zeros((50,), dtype=names_dtype)
    names_data['0'] = [f'Gene{i}' for i in range(50)]
    names_data['1'] = [f'Gene{i}' for i in range(50)]

    adata.uns['rank_genes_groups'] = {
        'pvals_adj': pvals_adj_data,
        'logfoldchanges': logfoldchanges_data,
        'names': names_data
    }

    # Test 1: group as string (should pass)
    try:
        plot_volcano_adata(adata, 'rank_genes_groups', group='0', show=False)
    except Exception as e:
        pytest.fail(f"Failed with group='0' (str): {e}")

    # Test 2: group as list of one element (should pass now)
    try:
        plot_volcano_adata(adata, 'rank_genes_groups', group=['0'], show=False)
    except Exception as e:
        pytest.fail(f"Failed with group=['0'] (list): {e}")

    # Test 3: group as list of multiple elements (should raise ValueError)
    with pytest.raises(ValueError, match="only supports plotting a single group"):
        plot_volcano_adata(adata, 'rank_genes_groups', group=['0', '1'], show=False)

if __name__ == "__main__":
    test_volcano_reproduction()
