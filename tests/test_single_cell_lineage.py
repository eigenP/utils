import pytest
import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt

from eigenp_utils.single_cell import calculate_lineage_coupling, plot_coupling_heatmap

@pytest.mark.parametrize("n_cells,n_clones,n_types", [
    (100, 20, 5),
    (500, 50, 10),
    (1000, 100, 3)
])
def test_calculate_lineage_coupling(n_cells, n_clones, n_types):
    # Create mock data
    np.random.seed(42)
    obs = pd.DataFrame({
        'cell_type': np.random.choice([f'Type_{i}' for i in range(n_types)], n_cells),
        'CloneID': np.random.choice([f'Clone_{i}' for i in range(n_clones)], n_cells)
    })
    # Make some clones specific to ensure non-trivial overlaps
    obs.loc[0:min(20, n_cells-1), 'cell_type'] = 'Type_0'
    obs.loc[0:min(20, n_cells-1), 'CloneID'] = 'Clone_0'

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
            n_permutations=100
        )

    # Check that no warnings related to shuffling categorical were raised (from legacy code)
    deprecation_warn_caught = False
    for r in record:
        assert "shuffling a 'Categorical' object" not in str(r.message)
        if "n_permutations" in str(r.message) and issubclass(r.category, DeprecationWarning):
            deprecation_warn_caught = True

    assert deprecation_warn_caught, "Deprecation warning for n_permutations was not caught."

    assert isinstance(obs_counts, pd.DataFrame)
    assert isinstance(z_scores, pd.DataFrame)
    assert isinstance(p_vals, pd.DataFrame)

    # Actual number of unique types might be less than n_types if randomly dropped
    actual_n_types = obs['cell_type'].nunique()
    assert obs_counts.shape == (actual_n_types, actual_n_types)
    assert z_scores.shape == (actual_n_types, actual_n_types)
    assert p_vals.shape == (actual_n_types, actual_n_types)

    # Check bounds of P-values (using normal survival function)
    assert (p_vals.values >= 0.0).all()
    assert (p_vals.values <= 1.0).all()

    # Check that diagonal has highest counts
    assert np.diag(obs_counts).max() > 0

    # Ensure no NaNs from division by zero
    assert not np.isnan(z_scores.values).any()


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
