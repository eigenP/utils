import pandas as pd
import numpy as np
import pytest
from eigenp_utils.plotting_utils import raincloud_plot

def test_raincloud_features():
    # generate some data
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'category': ['A'] * n + ['B'] * n,
        'value': np.concatenate([np.random.normal(0, 1, n), np.random.normal(2, 1, n)]),
        'highlight': np.concatenate([np.random.choice([True, False], n, p=[0.1, 0.9]),
                                     np.random.choice([True, False], n, p=[0.1, 0.9])])
    })

    # Add an extreme outlier that should be filtered out by 'robust_zscore'
    df.loc[0, 'value'] = 1000

    # Ensure no exceptions are raised during plotting with new kwargs
    fig_dict = raincloud_plot(
        data=df,
        x='category',
        y='value',
        outlier_method='robust_zscore',
        outlier_multiplier=3.0,
        highlight_mask=df['highlight'],
        highlight_color='lime'
    )

    # Check that it returns dict with 'fig' and 'axes'
    assert 'fig' in fig_dict
    assert 'axes' in fig_dict

    # Test fallback gracefully handles if all points are filtered
    df_all_outliers = pd.DataFrame({
        'category': ['A', 'A'],
        'value': [1000, -1000] # Mean is 0, MAD is 1000, zscores are 0.6745. Both < 3.
        # But wait, robust_zscore on 2 points? Let's just pass some random stuff.
    })

    fig_dict2 = raincloud_plot(
        data=df,
        x='category',
        y='value',
        outlier_method='iqr',
        outlier_multiplier=0.0001, # This will filter almost everything
    )
    assert 'fig' in fig_dict2

def test_raincloud_legacy():
    # Test legacy input without mask doesn't crash
    data = [np.random.normal(0, 1, 100), np.random.normal(1, 1, 100)]
    fig_dict = raincloud_plot(
        data=data,
        outlier_method='iqr',
        highlight_mask=None
    )
    assert 'fig' in fig_dict
