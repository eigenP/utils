
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eigenp_utils.plotting_utils import raincloud_plot

def test_raincloud_dodge_explicit():
    np.random.seed(42)
    n = 20
    groups = ['A', 'B']
    conditions = ['Control', 'Treatment']
    data = []
    for g in groups:
        for c in conditions:
            vals = np.random.normal(loc=10, scale=3, size=n)
            for v in vals:
                data.append({'Group': g, 'Condition': c, 'Value': v})

    df = pd.DataFrame(data)

    # Test dodge=True
    res_true = raincloud_plot(x='Group', y='Value', hue='Condition', data=df, dodge=True)
    assert res_true['fig'] is not None
    plt.close(res_true['fig'])

    # Test dodge=False
    res_false = raincloud_plot(x='Group', y='Value', hue='Condition', data=df, dodge=False)
    assert res_false['fig'] is not None
    plt.close(res_false['fig'])

def test_raincloud_palette_string():
    np.random.seed(42)
    df = pd.DataFrame({
        'Group': np.random.choice(['A', 'B'], 50),
        'Value': np.random.randn(50)
    })

    # Test valid colormap
    res_cmap = raincloud_plot(x='Group', y='Value', hue='Group', data=df, palette='viridis')
    assert res_cmap['fig'] is not None
    plt.close(res_cmap['fig'])

    # Test invalid colormap (fallback to single color, which then fails as invalid color)
    with pytest.raises(ValueError):
        raincloud_plot(x='Group', y='Value', hue='Group', data=df, palette='NotAColormap')

def test_raincloud_auto_dodge():
    # When hue == x, dodge should be False (no shifting)
    np.random.seed(42)
    df = pd.DataFrame({
        'Group': np.random.choice(['A', 'B'], 50),
        'Value': np.random.randn(50)
    })

    # We can inspect the plot elements to verify position?
    # Or just check it runs without error
    res = raincloud_plot(x='Group', y='Value', hue='Group', data=df)
    assert res['fig'] is not None
    plt.close(res['fig'])
