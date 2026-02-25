import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eigenp_utils.plotting_utils import raincloud_plot

def test_raincloud_hue_vertical():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'stage': np.random.choice(['Stage1', 'Stage2'], n),
        'condition': np.random.choice(['CondA', 'CondB'], n),
        'distances': np.random.exponential(scale=2.0, size=n)
    })

    res = raincloud_plot(data=df, x='stage', y='distances', hue='condition', title="Hue Plot")
    ax = res['axes']

    # Check xticks (should be 2: Stage1, Stage2)
    xticks = ax.get_xticks()
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]

    assert len(xticks) == 2
    # Sort to ensure order doesn't matter for this check
    assert sorted(xticklabels) == ['Stage1', 'Stage2']

    # Check labels
    assert ax.get_xlabel() == 'stage'
    assert ax.get_ylabel() == 'distances'

def test_raincloud_simple_xy():
    np.random.seed(42)
    df = pd.DataFrame({
        'stage': ['A', 'A', 'B', 'B'],
        'val': [1, 2, 3, 4]
    })
    res = raincloud_plot(data=df, x='stage', y='val')
    ax = res['axes']
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert sorted(xticklabels) == ['A', 'B']

def test_raincloud_horizontal_hue():
    np.random.seed(42)
    df = pd.DataFrame({
        'group': ['G1', 'G1', 'G2', 'G2'],
        'sub': ['S1', 'S2', 'S1', 'S2'],
        'val': [1, 2, 3, 4]
    })
    # Horizontal: y is category, x is value
    res = raincloud_plot(data=df, x='val', y='group', hue='sub', orientation='horizontal')
    ax = res['axes']

    # yticks should be G1, G2
    yticks = ax.get_yticks()
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(yticks) == 2
    assert sorted(yticklabels) == ['G1', 'G2']

    # xlabel should be 'val'
    assert ax.get_xlabel() == 'val'
    assert ax.get_ylabel() == 'group'
