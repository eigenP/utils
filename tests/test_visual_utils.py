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
    # It's a LinearSegmentedColormap, so we can check it returns transparent for 0
    assert labels_cmap(0.0) == (0.0, 0.0, 0.0, 0.0) # RGBA for transparent

    # Check that we can import it and it's not None
    assert labels_cmap is not None

    # Check it's registered globally
    assert plt.get_cmap("labels_cmap") is not None


def test_labels_cmap_diversity():
    # Verify that the colormap colors are perceptually diverse
    from skimage.color import rgb2lab, deltaE_ciede2000

    # Extract RGB values (skip index 0 which is transparent)
    # The colormap has 256 colors
    cmap_colors = labels_cmap(np.linspace(0, 1, 256))[1:]
    # Extract RGB only
    rgb_colors = cmap_colors[:, :3]

    lab_colors = rgb2lab(rgb_colors)

    # Calculate min overall distance
    min_overall_dist = float('inf')
    for i in range(len(lab_colors)):
        dists = deltaE_ciede2000(lab_colors[i:i+1], lab_colors)
        dists[i] = float('inf') # ignore self distance
        min_overall_dist = min(min_overall_dist, np.min(dists))

    # Calculate min adjacent distance
    min_adj_dist = float('inf')
    for i in range(len(lab_colors)-1):
        d = deltaE_ciede2000(lab_colors[i:i+1], lab_colors[i+1:i+2])[0]
        min_adj_dist = min(min_adj_dist, d)

    # Assert criteria
    # Glasbey farthest point guaranteed minimum distance of ~8 over 255 colors
    assert min_overall_dist > 5.0, f"Overall min distance is too small: {min_overall_dist}"

    # The simulated annealing ordered it so adjacent colors are far apart (>30)
    assert min_adj_dist > 25.0, f"Adjacent colors are too similar! Min adjacent distance: {min_adj_dist}"
