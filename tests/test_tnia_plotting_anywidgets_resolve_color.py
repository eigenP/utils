import numpy as np
import pytest
import matplotlib.colors as mcolors
from eigenp_utils.tnia_plotting_anywidgets import resolve_color, show_zyx_max_slice_interactive
from eigenp_utils.plotting_utils import labels_cmap

def test_resolve_color():
    # Test hex colors
    assert resolve_color("#ff0000") == "#ff0000"

    # Test valid colormap names
    assert resolve_color("viridis") == "#fde725" # final color of viridis

    # Test colormap instances directly raise TypeError
    cmap = mcolors.LinearSegmentedColormap.from_list('test', ['black', 'white'])
    with pytest.raises(TypeError, match="Expected a registered colormap name"):
        resolve_color(cmap)

    # Test actual labels_cmap issue from prompt
    with pytest.raises(TypeError, match="Expected a registered colormap name"):
        resolve_color(labels_cmap)

def test_widget_rejects_colormap_instance():
    im = np.zeros((10, 20, 30))
    # This crashed previously due to channel_colors list expecting a unicode string but getting a Colormap instance
    with pytest.raises(TypeError, match="Expected a registered colormap name"):
        w = show_zyx_max_slice_interactive(im, colormap=labels_cmap)
