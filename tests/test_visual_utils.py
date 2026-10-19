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
    """Test that color coded projection basic works as expected."""
    img = np.zeros((3, 2, 2), dtype=float)
    img[1] = 1.0
    result = color_coded_projection(img, color_map='plasma')
    assert result.shape == (2, 2, 3)
    expected_color = mpl_colormaps['plasma'](1 / (3 - 1))[:3]
    expected = np.stack([np.full((2, 2), expected_color[c]) for c in range(3)], axis=-1)
    assert np.allclose(result, expected)


def test_color_coded_projection_invalid_input():
    """Test that color coded projection invalid input works as expected."""
    with pytest.raises(ValueError):
        color_coded_projection(np.zeros((2, 2)))


def test_hist_imshow_return_image_only():
    """Test that hist imshow return image only works as expected."""
    img = np.random.rand(3, 4, 5)
    slice_img = hist_imshow(img, return_image_only=True)
    assert slice_img.shape == (4, 5)
    assert np.allclose(slice_img, img[1])



def test_hist_imshow_axes_dict_contents():
    """Test that hist imshow axes dict contents works as expected."""
    res = hist_imshow(np.random.rand(5, 5))
    axes = res["axes"]
    assert list(axes.keys()) == ["Image", "Histogram"]
    assert all(isinstance(ax, Axes) for ax in axes.values())

def test_labels_cmap():
    """Test that labels cmap works as expected, including transparent index 0 and lightness boost."""
    from skimage.color import rgb2lab

    # Verify labels_cmap is a valid colormap and has correct structure
    assert labels_cmap.name == "labels_cmap"
    # It's a LinearSegmentedColormap, so we can check it returns transparent for 0
    assert labels_cmap(0.0) == (0.0, 0.0, 0.0, 0.0) # RGBA for transparent

    # Check that we can import it and it's not None
    assert labels_cmap is not None

    # Check that all non-transparent colors (1-255) have boosted lightness L* >= 40 for pop against black
    cmap_colors = labels_cmap(np.linspace(0, 1, 256))[1:, :3]
    lab_colors = rgb2lab(cmap_colors)
    min_lightness = lab_colors[:, 0].min()
    assert min_lightness >= 40.0, f"Label colors must have lightness L* >= 40 to pop on black background, got min {min_lightness:.2f}"


def test_labels_cmap_diversity():
    """Test that labels cmap diversity works as expected."""
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
    assert min_overall_dist > 0.4, f"Overall min distance is too small: {min_overall_dist}"

    # The simulated annealing ordered it so adjacent colors are far apart (>18)
    assert min_adj_dist > 18.0, f"Adjacent colors are too similar! Min adjacent distance: {min_adj_dist}"


# =========================================
# Source: scan low-severity regressions (plotting_utils)
# =========================================


def test_labels_cmap_low_ids_stay_opaque_with_many_labels():
    """With >256 labels the lowest ids must not fall on the transparent entry."""
    lab = np.zeros((4, 4), int)
    lab[0, 0] = 1
    lab[0, 1] = 1000
    fig, ax = plt.subplots()
    im = ax.imshow(lab, cmap=labels_cmap, interpolation="nearest")
    rgba = im.to_rgba(lab)
    plt.close(fig)
    assert rgba[1, 1, 3] == 0  # background stays transparent
    assert rgba[0, 0, 3] == 1  # label 1 is drawn
    assert rgba[0, 1, 3] == 1
    # entries are unchanged: labels_cmap(i / 255) is the i-th colour
    from eigenp_utils.plotting_utils import final_colors

    np.testing.assert_allclose(labels_cmap(np.arange(256) / 255), np.asarray(final_colors))


def test_hist_imshow_accepts_bool_lists_and_gamma_with_limits():
    res = hist_imshow(np.random.default_rng(0).random((16, 16)) > 0.5)
    assert "bool" in res["axes"]["Histogram"].get_xlabel()
    hist_imshow([[1, 2], [3, 4]])
    res = hist_imshow(np.random.default_rng(0).random((8, 8)), gamma=0.5, vmin=0.2, vmax=0.8)
    norm = res["axes"]["Image"].images[0].norm
    assert (norm.gamma, norm.vmin, norm.vmax) == (0.5, 0.2, 0.8)
    plt.close("all")


def test_color_coded_projection_stays_in_unit_range_and_handles_int16():
    rng = np.random.default_rng(0)
    img = rng.random((4, 8, 8)).astype(np.float32)
    img[2] = 1000.0  # constant frame
    assert color_coded_projection(img).max() <= 1.0

    # int16 values -20000..20000: the middle value is half way, not lost to overflow
    img = np.zeros((3, 1, 3), np.int16)
    img[:, 0, :] = [-20000, 0, 20000]
    out = color_coded_projection(img)
    ref = color_coded_projection(img.astype(np.float64))
    np.testing.assert_allclose(out, ref, atol=1e-6)
    assert out[0, 1].max() > 0.4


def test_color_coded_projection_single_frame():
    out = color_coded_projection(np.random.default_rng(0).random((1, 8, 8)))
    assert out.shape == (8, 8, 3) and np.isfinite(out).all()
