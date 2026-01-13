
import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch
from eigenp_utils.single_cell import plot_volcano_adata

def test_plot_volcano_adata():
    # Setup dummy AnnData
    n_obs = 100
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Setup mock rank_genes_groups results in adata.uns
    group_name = 'GroupA'
    rank_genes_key = 'rank_genes_groups'

    # Create some dummy stats
    # genes 0-9 are significant upregulated
    # genes 10-19 are significant downregulated
    # genes 20-49 are not significant

    names = np.array([f"gene_{i}" for i in range(n_vars)])

    # Significant Upregulated
    lfc_up = np.random.uniform(1.5, 3.0, 10)
    pvals_up = np.random.uniform(0, 0.01, 10)

    # Significant Downregulated
    lfc_down = np.random.uniform(-3.0, -1.5, 10)
    pvals_down = np.random.uniform(0, 0.01, 10)

    # Non-significant
    lfc_ns = np.random.uniform(-0.5, 0.5, 30)
    pvals_ns = np.random.uniform(0.1, 1.0, 30)

    logfoldchanges = np.concatenate([lfc_up, lfc_down, lfc_ns])
    pvals_adj = np.concatenate([pvals_up, pvals_down, pvals_ns])

    # Structure for single group comparison
    # rank_genes_groups usually stores structured arrays or dataframes per group
    # but the function accesses it as: adata.uns[key]['logfoldchanges'][group]

    adata.uns[rank_genes_key] = {
        'logfoldchanges': pd.DataFrame({group_name: logfoldchanges}, index=names).to_records(index=False), # actually scanpy structure is often structured array or recarray
        # But wait, the function does: comparison_uns['logfoldchanges'][group]
        # If it's a structured array, it works by field name.
        # If it's a dict of arrays, it works by key.
        # Let's mock it as a dict of arrays/recarrays which behaves like structured data

        # Simplified mock structure: dict of dicts/series/arrays?
        # Standard Scanpy: adata.uns['rank_genes_groups']['names'] is a structured array where fields are group names
    }

    # Let's mock the dictionary structure directly as the function expects
    # The function uses: comparison_uns['names'][group] -> returns array of names

    adata.uns[rank_genes_key] = {
        'logfoldchanges': {group_name: logfoldchanges},
        'pvals_adj': {group_name: pvals_adj},
        'names': {group_name: names}
    }

    # Mock adjust_text to avoid dependency requirement during test execution if not installed,
    # and to verify kwargs are passed.
    # However, the function imports adjust_text at module level.
    # If installed, it uses it. If not, it sets it to None.
    # We need to ensure `eigenp_utils.single_cell.adjust_text` is not None for the test to proceed.

    with patch('eigenp_utils.single_cell.adjust_text') as mock_adjust_text:
        # 1. Test basic execution
        ax = plot_volcano_adata(
            adata,
            rank_genes_key=rank_genes_key,
            group=group_name,
            show=False
        )
        assert isinstance(ax, plt.Axes)
        assert mock_adjust_text.called

        # 2. Test kwargs passing
        custom_kwargs = {'force_text': (1.0, 1.0), 'arrowprops': dict(color='blue')}
        plot_volcano_adata(
            adata,
            rank_genes_key=rank_genes_key,
            group=group_name,
            **custom_kwargs
        )
        # Check if called with custom kwargs
        # The function updates default kwargs with user kwargs.
        call_kwargs = mock_adjust_text.call_args[1]
        assert call_kwargs['force_text'] == (1.0, 1.0)
        assert call_kwargs['arrowprops']['color'] == 'blue'

        # 3. Test plot_positive_only
        ax_pos = plot_volcano_adata(
            adata,
            rank_genes_key=rank_genes_key,
            group=group_name,
            plot_positive_only=True
        )
        assert isinstance(ax_pos, plt.Axes)
        # Verify vertical lines: only one expected for positive only (plus one horizontal)
        # ax.lines contains the lines added by axhline/axvline
        # Standard: 1 hline, 2 vlines = 3 lines
        # Positive only: 1 hline, 1 vline = 2 lines
        # But ax.scatter adds collections, not lines.
        # axhline/axvline add Line2D objects to ax.lines
        assert len(ax_pos.lines) == 2

        # 4. Test missing key
        res = plot_volcano_adata(
            adata,
            rank_genes_key="wrong_key",
            group=group_name
        )
        assert res is None

def test_missing_adjust_text():
    # Simulate missing adjustText library
    with patch('eigenp_utils.single_cell.adjust_text', None):
        n_obs = 10
        n_vars = 10
        adata = AnnData(X=np.random.rand(n_obs, n_vars))

        with pytest.raises(ImportError) as excinfo:
            plot_volcano_adata(adata, "key", "group")

        assert "pip install adjustText" in str(excinfo.value)

if __name__ == "__main__":
    try:
        test_plot_volcano_adata()
        test_missing_adjust_text()
        print("All volcano plot tests passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed: {repr(e)}")
        exit(1)
