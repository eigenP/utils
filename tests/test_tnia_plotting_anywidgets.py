import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive, TNIASliceWidget

def test_single_channel_instantiation():
    im = np.zeros((10, 20, 30))
    w = show_xyz_max_slice_interactive(im)
    assert isinstance(w, TNIASliceWidget)
    assert w.num_channels == 1
    assert len(w.channel_names) == 0
    # Render should produce image data (triggered by observer in init)
    assert w.image_data is not None and len(w.image_data) > 0

def test_multi_channel_instantiation():
    im = [np.zeros((10, 20, 30)) for _ in range(3)]
    w = show_xyz_max_slice_interactive(im)
    assert w.num_channels == 3
    assert len(w.channel_names) == 3
    assert w.channel_names == ["Channel 0", "Channel 1", "Channel 2"]
    assert w.channel_visible == [True, True, True]
    assert w.image_data is not None and len(w.image_data) > 0

def test_channel_visibility_update():
    # Use different values to ensure visual difference
    im = [np.zeros((10, 10, 10)), np.ones((10, 10, 10)) * 255]
    w = show_xyz_max_slice_interactive(im)

    initial_data = w.image_data
    assert initial_data

    # Hide channel 1 (the bright one)
    w.channel_visible = [True, False]

    # Check that image data changed (re-rendered)
    new_data = w.image_data
    assert new_data != initial_data

    # Hide all channels
    w.channel_visible = [False, False]
    empty_data = w.image_data
    assert empty_data != new_data
    assert empty_data != initial_data

def test_default_colors_resolution():
    im = [np.zeros((10, 10, 10)) for _ in range(2)]
    w = show_xyz_max_slice_interactive(im, colors=None)
    assert w.colors_resolved == ['magenta', 'cyan'] # Defaults

    w2 = show_xyz_max_slice_interactive(im, colors=['red', 'blue'])
    assert w2.colors_resolved == ['red', 'blue']
