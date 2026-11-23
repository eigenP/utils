import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib import colormaps as mpl_colormaps
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from eigenp_utils import color_coded_projection, describe_image, hist_imshow, labels_cmap
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


def test_describe_image_return_image_only():
    """Test that hist imshow return image only works as expected."""
    img = np.random.rand(3, 4, 5)
    slice_img = describe_image(img, return_image_only=True)
    assert slice_img.shape == (4, 5)
    assert np.allclose(slice_img, img[1])



def test_describe_image_axes_dict_contents():
    """Test that hist imshow axes dict contents works as expected."""
    res = describe_image(np.random.rand(5, 5))
    axes = res["axes"]
    assert list(axes.keys()) == ["Image", "Histogram", "Info"]
    assert "shape: (5, 5)" in res["info"] and "dtype: float64" in res["info"]
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


def test_describe_image_accepts_bool_lists_and_gamma_with_limits():
    res = describe_image(np.random.default_rng(0).random((16, 16)) > 0.5)
    assert "bool" in res["axes"]["Histogram"].get_xlabel() and "dtype: bool" in res["info"]
    describe_image([[1, 2], [3, 4]])
    res = describe_image(np.random.default_rng(0).random((8, 8)), gamma=0.5, vmin=0.2, vmax=0.8)
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


def test_brightness_diagnostic_plotter_object_oriented():
    """The plotter draws on its own (or the given) axes, never on the current pyplot axes."""
    from eigenp_utils.plotting_utils import brightness_diagnostic_plotter

    diag = {'x_data': np.arange(5), 'y_data_norm': np.linspace(1, 0.5, 5),
            'y_fit_norm': np.linspace(1, 0.5, 5), 'gamma_fit_func': 'exponential'}
    current_fig, current_ax = plt.subplots()
    fig = brightness_diagnostic_plotter(diag)
    assert fig is not current_fig and len(current_ax.lines) == 0
    assert len(fig.axes[0].lines) == 2

    target_fig, target_ax = plt.subplots()
    assert brightness_diagnostic_plotter(diag, ax=target_ax) is target_fig
    assert len(target_ax.lines) == 2 and target_ax.get_title() == 'Z-axis Intensity Decay Fit'


def test_describe_image_dataarray_info_panel_not_clipped():
    xr = pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray, set_channel_info

    arr = np.random.default_rng(1).random((3, 4, 32, 40)).astype(np.float32)
    da = array_to_stczyx_xarray(arr, input_axes="CZYX", channel_names=["DAPI", "WGA", "TOMM20"],
                                pixel_sizes={"Z": 2.0, "Y": 0.108, "X": 0.108, "unit": "µm"})
    da = set_channel_info(da, {"DAPI": {"role": "dna"}, "WGA": {"role": "membrane"}, "TOMM20": {"role": "mito"}})
    res = describe_image(da)
    info = res["info"]
    assert "pixel size: Z=2 Y=0.108 X=0.108 µm" in info
    assert "C[2] TOMM20  role=mito" in info
    assert "shown: S=0, T=0, C=WGA, Z=2" in info
    # the rendered panel stays inside the figure (no clipping)
    fig = res["fig"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fb = fig.bbox
    for txt in res["axes"]["Info"].texts:
        bb = txt.get_window_extent(renderer)
        assert bb.x0 >= fb.x0 - 1 and bb.x1 <= fb.x1 + 1 and bb.y0 >= fb.y0 - 1
    plt.close("all")


def test_describe_image_layout_one_column_image_histogram_info():
    """Layout invariants of describe_image, for a square, a wide and a tall image.

    One column, top to bottom: the image (largest panel), a short histogram at
    least as wide as the image, and the info table spanning the full width.
    Every drawn artist (tick labels, axis labels, table text) must stay inside
    the figure, since the axes are placed in inches without a layout engine.
    """
    rng = np.random.default_rng(0)
    for shape in [(80, 80), (30, 200), (200, 30)]:
        res = describe_image(rng.random(shape), image_size=4.0)
        fig, axes = res["fig"], res["axes"]
        fig.draw_without_rendering()
        img = axes["Image"].get_window_extent()
        hist = axes["Histogram"].get_window_extent()
        info = axes["Info"].get_window_extent()
        assert hist.y1 < img.y0 and info.y1 < hist.y0  # stacked top to bottom
        assert hist.x0 <= img.x0 + 1 and hist.x1 >= img.x1 - 1  # as wide as the image
        assert hist.height < img.height or shape[0] < shape[1]
        assert info.x0 <= min(img.x0, hist.x0) and info.x1 >= max(img.x1, hist.x1)
        fb = fig.bbox
        for ax in axes.values():
            tb = ax.get_tightbbox()
            assert tb.x0 >= fb.x0 - 1 and tb.x1 <= fb.x1 + 1 and tb.y0 >= fb.y0 - 1 and tb.y1 <= fb.y1 + 1
        plt.close(fig)


def test_describe_image_histogram_shows_pixel_fractions_log_only_over_a_decade():
    """The histogram bars are fractions of the shown pixels, on a log axis for a
    skewed distribution and a linear one for a flat distribution (where a log
    axis would show no labelled major tick)."""
    rng = np.random.default_rng(0)
    skewed = describe_image(rng.exponential(100, (64, 64)))["axes"]["Histogram"]
    flat = describe_image(np.tile(np.arange(64.0), (64, 1)))["axes"]["Histogram"]
    assert skewed.get_yscale() == "log" and flat.get_yscale() == "linear"
    # 64 equally populated values in 64 bins: every bar is exactly 1/64 of the pixels
    bar_y = flat.patches[0].get_path().vertices[:, 1]
    assert bar_y.max() == pytest.approx(1 / 64)
    assert np.all((bar_y == 0) | np.isclose(bar_y, 1 / 64))
    plt.close("all")


def test_describe_image_marks_display_range_only_when_it_differs_from_data():
    """With autoscaled limits the histogram has no vmin/vmax marks and the info
    has no display row (it would repeat min/max). With vmax inside the data, a
    line marks it, the clipped intensities are shaded, and the info reports the
    exact fraction of clipped pixels; a vmax beyond the data widens the axis so
    the unused part of the display range is visible."""
    img = np.tile(np.arange(100.0), (10, 1))  # 0..99, each value 1 % of pixels
    auto = describe_image(img)
    assert not auto["axes"]["Histogram"].lines and "display" not in auto["info"]

    res = describe_image(img, vmax=79.5)
    ax = res["axes"]["Histogram"]
    assert [ln.get_xdata()[0] for ln in ax.lines] == [79.5]
    assert len([p for p in ax.patches if p.get_alpha() is not None and p.get_alpha() < 1]) == 1
    assert "display: vmin 0  vmax 79.5  (20% of pixels clipped)" in res["info"]

    wide = describe_image(img, vmax=200)["axes"]["Histogram"]
    assert wide.get_xlim()[1] == pytest.approx(200)
    plt.close("all")


def test_describe_image_channel_table_swatches_and_shown_row():
    """For a DataArray with channel metadata, the info table lists each channel
    with its metadata, draws a swatch in the channel's colour, and highlights the
    shown (middle) channel. Long source paths wrap at directory boundaries."""
    pytest.importorskip("xarray")
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Rectangle
    from eigenp_utils.image_io import array_to_stczyx_xarray, set_channel_info
    from eigenp_utils.plotting_utils import _wrap_value

    arr = np.random.default_rng(2).random((3, 16, 20)).astype(np.float32)
    da = array_to_stczyx_xarray(arr, input_axes="CYX", channel_names=["DAPI", "WGA", "TOMM20"])
    da = set_channel_info(da, {"DAPI": {"color": "blue"}, "WGA": {"color": "green"}, "TOMM20": {"color": "red"}})
    da.attrs["source"] = "/very/long/" + "directory_name/" * 12 + "file.ome.tif"
    res = describe_image(da)
    ax = res["axes"]["Info"]
    texts = {t.get_text(): t for t in ax.texts}
    assert {"DAPI", "WGA", "TOMM20"} <= set(texts)
    assert texts["WGA"].get_fontweight() == "bold" and texts["DAPI"].get_fontweight() == "normal"
    swatches = [tuple(p.get_facecolor()[:3]) for p in ax.patches
                if isinstance(p, Rectangle) and p.get_linewidth() > 0]
    for c in ("blue", "green", "red"):
        assert any(np.allclose(sw, to_rgb(c)) for sw in swatches)
    # the info string (plain text) keeps its line format
    assert "C[1] WGA  color=green" in res["info"] and "shown: S=0, T=0, C=WGA, Z=0" in res["info"]

    html = res["info_html"]
    assert isinstance(html, str) and html._repr_html_() == html
    assert "TOMM20" in html and "background:#008000" in html and "&#9664; shown" in html
    # metadata is escaped, not interpreted as markup
    da.attrs["source"] = "/data/<b>x</b>.tif"
    assert "/data/&lt;b&gt;x&lt;/b&gt;.tif" in describe_image(da)["info_html"]

    lines = _wrap_value("/a/" + "dir/" * 20 + "f.tif", 30)
    assert all(len(line) <= 30 for line in lines)
    assert all(line.endswith("/") for line in lines[:-1])
    assert "".join(line.strip() for line in lines) == "/a/" + "dir/" * 20 + "f.tif"
    plt.close("all")


def test_hist_imshow_is_a_deprecated_alias_of_describe_image():
    img = np.random.default_rng(0).random((8, 9))
    with pytest.warns(DeprecationWarning, match="describe_image"):
        res = hist_imshow(img, bins=16)
    ref = describe_image(img, bins=16)
    assert res["info"] == ref["info"] and res["info_html"] == ref["info_html"]
    assert list(res["axes"]) == list(ref["axes"])
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(hist_imshow(img, return_image_only=True), img)
    plt.close("all")
