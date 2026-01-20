
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg") # Use non-interactive backend
import matplotlib.pyplot as plt
from eigenp_utils.plotting_utils import raincloud_plot

def test_raincloud_plot_vertical_custom_labels():
    data = [np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]
    x_labels = ['Group A', 'Group B']

    # x_label as list
    res = raincloud_plot(data, x_label=x_labels, title="Vertical Plot")
    ax = res['axes']

    xticklabels = [l.get_text() for l in ax.get_xticklabels()]
    xlabel = ax.get_xlabel()

    assert xticklabels == x_labels
    assert xlabel == ""

def test_raincloud_plot_horizontal_custom_labels():
    data = [np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]
    y_labels = ['Group X', 'Group Y']

    # y_label as list, orientation horizontal
    res = raincloud_plot(data, y_label=y_labels, orientation='horizontal', title="Horizontal Plot")
    ax = res['axes']

    yticklabels = [l.get_text() for l in ax.get_yticklabels()]
    ylabel = ax.get_ylabel()

    assert yticklabels == y_labels
    assert ylabel == ""

def test_raincloud_plot_mismatch_warning(capsys):
    data = [np.random.normal(0, 1, 100), np.random.normal(2, 1, 100)]
    x_labels = ['Group A'] # Mismatch length

    res = raincloud_plot(data, x_label=x_labels, title="Mismatch Plot")
    ax = res['axes']

    xticklabels = [l.get_text() for l in ax.get_xticklabels()]
    xlabel = ax.get_xlabel()

    # Expect standard behavior: ticks are 0, 1. xlabel is "['Group A']"
    assert xticklabels == ['0', '1']
    assert xlabel == "['Group A']"

    # Check for warning print
    captured = capsys.readouterr()
    assert "Warning: x_label list length (1) does not match number of groups (2)" in captured.out
