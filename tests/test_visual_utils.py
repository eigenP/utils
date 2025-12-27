import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib import colormaps as mpl_colormaps
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from eigenp_utils import color_coded_projection, hist_imshow, labels_cmap
import matplotlib.pyplot as plt

def test_color_coded_projection_basic():
    img = np.zeros((3, 2, 2), dtype=float)
    img[1] = 1.0
    result = color_coded_projection(img, color_map='plasma')
    assert result.shape == (2, 2, 3)
    expected_color = mpl_colormaps['plasma'](1 / (3 - 1))[:3]
    expected = np.stack([np.full((2, 2), expected_color[c]) for c in range(3)], axis=-1)
    assert np.allclose(result, expected)


def test_color_coded_projection_invalid_input():
    with pytest.raises(ValueError):
        color_coded_projection(np.zeros((2, 2)))


def test_hist_imshow_return_image_only():
    img = np.random.rand(3, 4, 5)
    slice_img = hist_imshow(img, return_image_only=True)
    assert slice_img.shape == (4, 5)
    assert np.allclose(slice_img, img[1])



def test_hist_imshow_axes_dict_contents():
    res = hist_imshow(np.random.rand(5, 5))
    axes = res["axes"]
    assert list(axes.keys()) == ["Image", "Histogram"]
    assert all(isinstance(ax, Axes) for ax in axes.values())

def test_labels_cmap():
    # Verify labels_cmap is a valid colormap and has correct structure
    assert labels_cmap.name == "labels_cmap"
    # It's a LinearSegmentedColormap, so we can check it returns black for 0
    assert labels_cmap(0.0) == (0.0, 0.0, 0.0, 1.0) # RGBA for black

    # Check that we can import it and it's not None
    assert labels_cmap is not None

def test_style_and_font_loaded():
    # Verify that the font 'Inter' is registered in font manager
    # Note: the exact name might vary ('Inter Regular', 'Inter') depending on how font_manager parses it
    # We check if any font in the manager has 'Inter' in its name

    # Check rcParams for font family being sans-serif
    assert plt.rcParams['font.family'] == ['sans-serif'] or plt.rcParams['font.family'] == 'sans-serif'

    # Check rcParams for sans-serif font being Inter
    current_sans = plt.rcParams['font.sans-serif']
    # It might be a list or a string
    if isinstance(current_sans, list):
        assert any('Inter' in f for f in current_sans)
    else:
        assert 'Inter' in current_sans
