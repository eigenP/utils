from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, show_zyx_max_slabs, show_zyx
import numpy as np
import pytest

def test_interactive_spacing_pixel_sizes_vs_sxy():
    """
    Test that when figsize is None, dynamically computed figsize parses
    pixel_sizes=(Z, Y, X) dicts correctly.
    """
    im = np.random.rand(10, 50, 50)

    # Using pixel_sizes dict
    w_dict = show_zyx_max_slice_interactive(im, pixel_sizes={'Z':2, 'Y':1, 'X':1})
    fig_dict = w_dict._render()
    size_dict = fig_dict.get_size_inches()

    # Using pixel_sizes tuple
    w_tuple = show_zyx_max_slice_interactive(im, pixel_sizes=(2, 1, 1))
    fig_tuple = w_tuple._render()
    size_tuple = fig_tuple.get_size_inches()

    np.testing.assert_allclose(size_dict, size_tuple)

    # Test the height ratios generated inside the gridspec
    gs_dict = fig_dict.axes[0].get_subplotspec().get_gridspec().get_height_ratios()
    gs_tuple = fig_tuple.axes[0].get_subplotspec().get_gridspec().get_height_ratios()

    assert gs_dict == gs_tuple

def test_xy_anisotropy():
    """
    Test that XY anisotropy is correctly respected in show_zyx spacing calculations.
    """
    im = np.random.rand(10, 50, 100) # Z, Y, X
    # Highly anisotropic XY pixels
    w = show_zyx_max_slice_interactive(im, pixel_sizes={'Z': 1.0, 'Y': 0.2, 'X': 1.0})
    fig = w._render()

    # Check gridspec ratios directly to see if physical scaling is applied
    gs = fig.axes[0].get_subplotspec().get_gridspec()
    width_ratios = gs.get_width_ratios()
    height_ratios = gs.get_height_ratios()

    # X physical = 100 * 1 = 100. Z physical = 10 * 1 = 10. Max width = 100.
    # Y physical = 50 * 0.2 = 10. Z physical = 10 * 1 = 10. Max height = 10.
    assert width_ratios[0] == 100
    assert height_ratios[1] == 10
