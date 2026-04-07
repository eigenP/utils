import pytest
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt

from eigenp_utils.single_cell import calculate_lineage_coupling, plot_coupling_heatmap

def test_calculate_lineage_coupling():
    # Create mock data
    n_cells = 100
    n_clones = 20
    n_types = 5

    obs = pd.DataFrame({
        'cell_type': np.random.choice([f'Type_{i}' for i in range(n_types)], n_cells),
        'CloneID': np.random.choice([f'Clone_{i}' for i in range(n_clones)], n_cells)
    })
    # Make some clones specific
    obs.loc[0:20, 'cell_type'] = 'Type_0'
    obs.loc[0:20, 'CloneID'] = 'Clone_0'
    # Ensure categorical to trigger warning if bug exists
    obs['cell_type'] = obs['cell_type'].astype('category')

    adata = anndata.AnnData(obs=obs)

    import warnings

    # Test function runs without warnings
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        obs_counts, z_scores, p_vals = calculate_lineage_coupling(
            adata,
            label_key='cell_type',
            clone_key='CloneID',
            n_permutations=10
        )

    # Check that no warnings related to shuffling categorical were raised
    for r in record:
        assert "shuffling a 'Categorical' object" not in str(r.message)

    assert isinstance(obs_counts, pd.DataFrame)
    assert isinstance(z_scores, pd.DataFrame)
    assert isinstance(p_vals, pd.DataFrame)

    assert obs_counts.shape == (n_types, n_types)
    assert z_scores.shape == (n_types, n_types)
    assert p_vals.shape == (n_types, n_types)


@pytest.mark.parametrize("title", ["Test Title 1", "Lineage Coupling"])
def test_plot_coupling_heatmap(title):
    n_types = 5

    obs_counts = pd.DataFrame(np.random.randint(0, 10, size=(n_types, n_types)))
    z_scores = pd.DataFrame(np.random.randn(n_types, n_types))
    p_vals = pd.DataFrame(np.random.rand(n_types, n_types))

    fig = plot_coupling_heatmap(obs_counts, z_scores, p_vals, title=title)

    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_title() == title
    plt.close(fig)
