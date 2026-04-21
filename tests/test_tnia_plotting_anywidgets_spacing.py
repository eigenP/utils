from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, show_zyx_max_slabs, show_zyx
import numpy as np
import pytest

def test_interactive_spacing_pixel_sizes_vs_sxy():
    """
    Test that when figsize is None, dynamically computed figsize uses
    the same `z_xy_ratio` whether pixel_sizes=(Z, Y, X) or sxy=X, sz=Z is passed.
    This ensures that axis spacing remains identical across both signatures.
    """
    im = np.random.rand(10, 50, 50)

    # Using legacy sxy, sz
    w_legacy = show_zyx_max_slice_interactive(im, sxy=1, sz=2)
    fig_legacy = w_legacy._render()
    size_legacy = fig_legacy.get_size_inches()

    # Using pixel_sizes dict
    w_dict = show_zyx_max_slice_interactive(im, pixel_sizes={'Z':2, 'Y':1, 'X':1})
    fig_dict = w_dict._render()
    size_dict = fig_dict.get_size_inches()

    # Using pixel_sizes tuple
    w_tuple = show_zyx_max_slice_interactive(im, pixel_sizes=(2, 1, 1))
    fig_tuple = w_tuple._render()
    size_tuple = fig_tuple.get_size_inches()

    np.testing.assert_allclose(size_legacy, size_dict)
    np.testing.assert_allclose(size_legacy, size_tuple)

    # Test the height ratios generated inside the gridspec
    gs_legacy = fig_legacy.axes[0].get_subplotspec().get_gridspec().get_height_ratios()
    gs_dict = fig_dict.axes[0].get_subplotspec().get_gridspec().get_height_ratios()
    gs_tuple = fig_tuple.axes[0].get_subplotspec().get_gridspec().get_height_ratios()

    assert gs_legacy == gs_dict
    assert gs_legacy == gs_tuple
