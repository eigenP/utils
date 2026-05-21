import pytest
import numpy as np
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, show_zyx_max_scatter_interactive

def test_rotation_slice_interactive():
    im = np.random.rand(10, 20, 30)

    # Test no rotation
    w0 = show_zyx_max_slice_interactive(im)
    assert w0.rotate_view is None

    # Test float rotation
    w1 = show_zyx_max_slice_interactive(im, rotate_view=45.0)
    assert w1.rotate_view == 45.0

    # Test tuple rotation
    w2 = show_zyx_max_slice_interactive(im, rotate_view=(10, 20, 30))
    assert w2.rotate_view == (10, 20, 30)

    # Force a render with rotation to check for runtime errors
    w2._render_wrapper(None)
    assert w2.image_data is not None

def test_rotation_scatter_interactive():
    N = 100
    X = np.random.rand(N) * 30
    Y = np.random.rand(N) * 20
    Z = np.random.rand(N) * 10

    # Test no rotation
    w0 = show_zyx_max_scatter_interactive((Z, Y, X))
    assert w0.rotate_view is None

    # Test float rotation
    w1 = show_zyx_max_scatter_interactive((Z, Y, X), rotate_view=45.0)
    assert w1.rotate_view == 45.0

    # Test tuple rotation
    w2 = show_zyx_max_scatter_interactive((Z, Y, X), rotate_view=(10, 20, 30))
    assert w2.rotate_view == (10, 20, 30)

    # Force a render with points rotation to check for runtime errors
    w2.render = 'points'
    w2._render_wrapper(None)
    assert w2.image_data is not None

    # Force a render with density rotation to check for runtime errors
    w2.render = 'density'
    w2._render_wrapper(None)
    assert w2.image_data is not None
