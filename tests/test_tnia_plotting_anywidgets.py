import numpy as np
import pytest
from eigenp_utils.tnia_plotting_anywidgets import (
    show_xyz_max_slice_interactive,
    show_xyz_max_scatter_interactive,
    show_xyz_max_slice_interactive_point_annotator
)

def test_show_xyz_max_slice_interactive_vmax_default():
    np.random.seed(42)
    data = np.random.uniform(0, 100, (10, 10, 10))
    p995 = np.percentile(data, 99.5)

    # When vmax is None (default), it should calculate p99.5
    widget = show_xyz_max_slice_interactive(data, vmax=None)
    assert len(widget.vmax_list) == 1
    assert np.isclose(widget.vmax_list[0], p995)

    # When vmax is provided as a number
    widget2 = show_xyz_max_slice_interactive(data, vmax=50)
    assert widget2.vmax_list[0] == 50

    # Test with multiple channels
    data2 = np.random.uniform(0, 50, (10, 10, 10))
    p995_2 = np.percentile(data2, 99.5)

    widget_multi = show_xyz_max_slice_interactive([data, data2], vmax=[100, None])
    assert widget_multi.vmax_list[0] == 100
    assert np.isclose(widget_multi.vmax_list[1], p995_2)

def test_show_xyz_max_scatter_interactive_vmax_default():
    np.random.seed(42)
    X = np.random.uniform(0, 10, 100)
    Y = np.random.uniform(0, 10, 100)
    Z = np.random.uniform(0, 10, 100)

    cont_data = np.random.uniform(0, 100, 100)
    p995 = np.percentile(cont_data, 99.5)

    # single channel cont
    widget_single = show_xyz_max_scatter_interactive(X, Y, Z, channels=cont_data)
    assert np.isclose(widget_single.vmax, p995)

    # provided vmax
    widget_single_set = show_xyz_max_scatter_interactive(X, Y, Z, channels=cont_data, vmax=50)
    assert widget_single_set.vmax == 50

    # multi channel cont
    cont_data2 = np.random.uniform(0, 50, 100)
    p995_2 = np.percentile(cont_data2, 99.5)

    widget_multi = show_xyz_max_scatter_interactive(X, Y, Z, channels=[cont_data, cont_data2])
    assert isinstance(widget_multi.vmax, list)
    assert np.isclose(widget_multi.vmax[0], p995)
    assert np.isclose(widget_multi.vmax[1], p995_2)

    # mixed vmax
    widget_multi_mixed = show_xyz_max_scatter_interactive(X, Y, Z, channels=[cont_data, cont_data2], vmax=[80, None])
    assert widget_multi_mixed.vmax[0] == 80
    assert np.isclose(widget_multi_mixed.vmax[1], p995_2)

def test_show_xyz_max_slice_interactive_point_annotator_vmax_default():
    np.random.seed(42)
    data = np.random.uniform(0, 100, (10, 10, 10))
    p995 = np.percentile(data, 99.5)

    widget = show_xyz_max_slice_interactive_point_annotator(data, vmax=None)
    # The annotator widget appends a channel for annotations (vmax 255)
    assert len(widget.vmax_list) == 2
    assert np.isclose(widget.vmax_list[0], p995)
    assert widget.vmax_list[1] == 255
