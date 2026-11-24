from types import SimpleNamespace
import xml.etree.ElementTree as ET

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import colormaps as mpl_colormaps
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from eigenp_utils import (
    I_SERIES_COLORMAPS,
    color_coded_projection,
    describe_image,
    hist_imshow,
    i_blue,
    i_bordeaux,
    i_forest,
    i_gold,
    i_orange,
    i_purple,
    i_rose,
    i_slate,
    i_teal,
    labels_cmap,
)
from eigenp_utils.labels_cmap_data import LABELS_CMAP_COLORS
from eigenp_utils.plotting_utils import (
    adjust_colormap,
    brightness_diagnostic_plotter,
    colormap_maker,
    get_nice_number,
    plotly_scatter_3d,
    plotly_scatter_3d_from_adata_obsm,
    raincloud_plot,
    savefig_svg,
    set_plotting_style,
)

SVG_NAMESPACES = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "cc": "http://creativecommons.org/ns#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "svg": "http://www.w3.org/2000/svg",
}


def _scatter_points(ax, color):
    """Count scatter points in ``ax`` whose facecolor RGB matches ``color``."""
    rgb = mpl.colors.to_rgb(color)
    return sum(
        len(c.get_offsets())
        for c in ax.collections
        if isinstance(c, PathCollection)
        and len(c.get_facecolor())
        and np.allclose(c.get_facecolor()[0][:3], rgb)
    )


def _violin_collections(ax):
    """Return half-violin polygon collections on ``ax`` across Matplotlib versions."""
    return [c for c in ax.collections if isinstance(c, PolyCollection)]


def _iqr_collections(ax):
    """Return IQR vlines/hlines LineCollection artists on ``ax``."""
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def _median_collections(ax):
    """Return single-point white-filled median scatter markers on ``ax``."""
    return [
        c
        for c in ax.collections
        if isinstance(c, PathCollection)
        and len(c.get_offsets()) == 1
        and len(c.get_facecolor())
        and np.allclose(c.get_facecolor()[0][:3], (1.0, 1.0, 1.0))
    ]


def _rain_collections(ax):
    """Return rain scatter PathCollection artists (excluding white median markers)."""
    medians = set(_median_collections(ax))
    return [c for c in ax.collections if isinstance(c, PathCollection) and c not in medians]


# =========================================
# savefig_svg & set_plotting_style
# =========================================


@pytest.mark.parametrize("suptitle", ["My Awesome Title", None])
def test_savefig_svg_metadata_and_title_fallback(tmp_path, suptitle):
    """Verify savefig_svg writes SVG files with Dublin Core metadata and title fallback.

    Strategy: save a figure with and without a suptitle, parse the resulting SVG XML,
    and inspect the Dublin Core title, date, and creator fields.
    Expected: dc:title uses the figure suptitle when present and falls back to the
    filename string when absent; dc:creator is 'eigenp' and dc:date is non-empty.
    """
    fig, ax = plt.subplots()
    try:
        ax.plot([1, 2, 3], [1, 4, 9])
        if suptitle is not None:
            ax.set_title("Test Plot")
            fig.suptitle(suptitle)

        svg_stem = tmp_path / ("with_title" if suptitle else "no_title")
        savefig_svg(svg_stem, bgnd_color=(1, 0, 0, 0.5), pad_inches=0.2)

        svg_file = svg_stem.with_suffix(".svg")
        assert svg_file.exists()

        root = ET.parse(svg_file).getroot()
        metadata_elem = root.find("svg:metadata", SVG_NAMESPACES)
        assert metadata_elem is not None
        rdf_work = metadata_elem.find(".//cc:Work", SVG_NAMESPACES)
        assert rdf_work is not None

        dc_title = rdf_work.find(".//dc:title", SVG_NAMESPACES)
        assert dc_title is not None
        assert dc_title.text == (suptitle if suptitle is not None else str(svg_stem))

        dc_date = rdf_work.find(".//dc:date", SVG_NAMESPACES)
        assert dc_date is not None and len(dc_date.text) > 0

        dc_creator = rdf_work.find(".//dc:creator//cc:Agent//dc:title", SVG_NAMESPACES)
        assert dc_creator is not None and dc_creator.text == "eigenp"
    finally:
        plt.close(fig)


def test_savefig_svg_custom_metadata_extension_and_no_mutation(tmp_path):
    """Verify custom metadata reaches SVG XML, caller dict is not mutated, and '.svg' is not doubled.

    Strategy: pass a custom metadata dictionary and a filename already ending in '.svg' to
    savefig_svg, then inspect the filesystem path, XML fields, and original dict.
    Expected: no '.svg.svg' file is created, XML reflects custom Creator/Title, and the
    caller's metadata dictionary remains unchanged.
    """
    md = {"Creator": "alice", "Title": "Custom Title"}
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1])
        svg_path = tmp_path / "already_suffixed.svg"
        savefig_svg(svg_path, metadata=md)

        assert svg_path.exists()
        assert not (tmp_path / "already_suffixed.svg.svg").exists()
        assert md == {"Creator": "alice", "Title": "Custom Title"}

        root = ET.parse(svg_path).getroot()
        rdf_work = root.find("svg:metadata", SVG_NAMESPACES).find(".//cc:Work", SVG_NAMESPACES)
        assert rdf_work.find(".//dc:title", SVG_NAMESPACES).text == "Custom Title"
        assert rdf_work.find(".//dc:creator//cc:Agent//dc:title", SVG_NAMESPACES).text == "alice"
    finally:
        plt.close(fig)


def test_set_plotting_style():
    """Verify set_plotting_style toggles SVG text settings cleanly inside rc_context.

    Strategy: snapshot mpl.rcParams, apply set_plotting_style(editable_text=True) and
    then editable_text=False inside plt.rc_context(), and check restoration afterward.
    Expected: font family, svg.fonttype, and axes.unicode_minus update as specified and
    no rcParams leak out of the context.
    """
    before = dict(mpl.rcParams)

    with plt.rc_context():
        set_plotting_style()
        assert plt.rcParams["font.family"] == ["sans-serif"]
        assert plt.rcParams["svg.fonttype"] == "none"
        assert plt.rcParams["axes.unicode_minus"] is False

        set_plotting_style(editable_text=False)
        assert plt.rcParams["svg.fonttype"] == "path"
        assert plt.rcParams["axes.unicode_minus"] is True

    leaked = {k for k, v in before.items() if mpl.rcParams[k] != v}
    assert not leaked, f"rcParams leaked out of the test: {sorted(leaked)[:5]}"


def test_savefig_svg_editable_text(tmp_path):
    """Verify editable_text controls SVG <text> output and restores pre-call rcParams.

    Strategy: run inside an isolated rc_context({'svg.fonttype': 'path'}) so the test
    does not depend on global Matplotlib state; export once with default editable_text=True
    and once with editable_text=False.
    Expected: editable_text=True writes <text elements and restores svg.fonttype to 'path';
    editable_text=False under 'path' rcParams outlines glyphs without <text elements.
    """
    with plt.rc_context({"svg.fonttype": "path", "axes.unicode_minus": True}):
        fig, ax = plt.subplots()
        try:
            ax.text(0.5, 0.5, "Test Editable Text")

            svg_path_true = tmp_path / "editable_true"
            savefig_svg(svg_path_true)
            assert plt.rcParams["svg.fonttype"] == "path"
            assert plt.rcParams["axes.unicode_minus"] is True
            content_true = svg_path_true.with_suffix(".svg").read_text()
            assert "<text" in content_true

            svg_path_false = tmp_path / "editable_false"
            savefig_svg(svg_path_false, editable_text=False)
            assert plt.rcParams["svg.fonttype"] == "path"
            content_false = svg_path_false.with_suffix(".svg").read_text()
            assert "<text" not in content_false
        finally:
            plt.close(fig)


def test_savefig_svg_raster_threshold_and_file_size(tmp_path):
    """Verify scatter_raster_threshold rasterizes dense scatters, shrinks SVGs, and respects None.

    Strategy: first check rasterized state transitions on a 100-point scatter for threshold
    None (disabled), 500 (unmet), and 50 (met); then save a seeded 2000-point scatter as
    vector (threshold=3000) vs rasterized (threshold=1000) and compare file sizes on disk.
    Expected: PathCollection rasterization flips only when point count >= threshold,
    threshold=None never rasterizes, and rasterized SVG is strictly smaller than vector SVG.
    """
    fig, ax = plt.subplots()
    try:
        ax.scatter(range(100), range(100))
        scatter_col = [c for c in ax.collections if isinstance(c, PathCollection)][0]
        assert scatter_col.get_rasterized() is False

        out_none = tmp_path / "test_none_threshold.svg"
        savefig_svg(out_none, scatter_raster_threshold=None)
        assert scatter_col.get_rasterized() is False
        assert out_none.exists()

        out_no_raster = tmp_path / "test_no_raster.svg"
        savefig_svg(out_no_raster, scatter_raster_threshold=500)
        assert scatter_col.get_rasterized() is False
        assert out_no_raster.exists()

        out_raster = tmp_path / "test_raster.svg"
        savefig_svg(out_raster, scatter_raster_threshold=50)
        assert scatter_col.get_rasterized() is True
        assert out_raster.exists()
    finally:
        plt.close(fig)

    rng = np.random.default_rng(0)
    fig_size, ax_size = plt.subplots()
    try:
        ax_size.scatter(rng.random(2000), rng.random(2000))
        out_vector = tmp_path / "size_vector.svg"
        out_raster_size = tmp_path / "size_raster.svg"

        savefig_svg(out_vector, scatter_raster_threshold=3000, dpi=100)
        savefig_svg(out_raster_size, scatter_raster_threshold=1000, dpi=100)
        assert out_raster_size.stat().st_size < out_vector.stat().st_size
    finally:
        plt.close(fig_size)


# =========================================
# raincloud_plot
# =========================================


@pytest.mark.parametrize(
    "orientation, label_kwarg, tick_getter, axis_label_getter",
    [
        ("vertical", "x_label", "get_xticklabels", "get_xlabel"),
        ("horizontal", "y_label", "get_yticklabels", "get_ylabel"),
    ],
)
def test_raincloud_plot_custom_tick_labels(orientation, label_kwarg, tick_getter, axis_label_getter):
    """Verify passing a group-length label sequence sets categorical tick labels instead of axis label.

    Strategy: plot two groups in vertical and horizontal orientations passing a 2-element
    list via x_label (vertical) or y_label (horizontal).
    Expected: categorical tick labels match the list and the corresponding axis label stays empty.
    """
    rng = np.random.default_rng(1)
    data = [rng.normal(0, 1, 50), rng.normal(2, 1, 50)]
    custom_labels = ["Group A", "Group B"]

    res = raincloud_plot(data, orientation=orientation, title="Custom Labels", **{label_kwarg: custom_labels})
    try:
        ax = res["axes"]
        ticks = [t.get_text() for t in getattr(ax, tick_getter)()]
        assert ticks == custom_labels
        assert getattr(ax, axis_label_getter)() == ""
    finally:
        plt.close(res["fig"])


@pytest.mark.parametrize(
    "orientation, label_kwarg, tick_getter, axis_label_getter",
    [
        ("vertical", "x_label", "get_xticklabels", "get_xlabel"),
        ("horizontal", "y_label", "get_yticklabels", "get_ylabel"),
    ],
)
def test_raincloud_plot_label_mismatch_warning(capsys, orientation, label_kwarg, tick_getter, axis_label_getter):
    """Verify mismatched tick-label sequence length warns and falls back to default ticks.

    Strategy: supply a 1-element label list for 2 groups in vertical and horizontal modes.
    Expected: warning is printed naming the mismatched parameter, ticks remain ['0', '1'],
    and the stringified list is used as the axis label.
    """
    rng = np.random.default_rng(2)
    data = [rng.normal(0, 1, 40), rng.normal(2, 1, 40)]

    res = raincloud_plot(data, orientation=orientation, **{label_kwarg: ["Group A"]})
    try:
        ax = res["axes"]
        ticks = [t.get_text() for t in getattr(ax, tick_getter)()]
        assert ticks == ["0", "1"]
        assert getattr(ax, axis_label_getter)() == "['Group A']"
        captured = capsys.readouterr()
        assert f"Warning: {label_kwarg} list length (1) does not match number of groups (2)" in captured.out
    finally:
        plt.close(res["fig"])


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_raincloud_plot_half_violin_and_boxplot_geometry(orientation):
    """Verify half-violin clipping geometry, IQR boxplot extents, median marker, and rain scatter side.

    Strategy: draw two deterministic groups in both vertical and horizontal orientations,
    then inspect the PolyCollection vertices, LineCollection IQR segments, median marker
    coordinates, and rain scatter offsets directly.
    Expected: half-violins are clipped to one side of each group position, IQR bars span
    exact [Q1, Q3], median markers sit at exact (pos, median) / (median, pos), and rain
    points lie on the opposite side with exact data values.
    """
    g0 = np.linspace(0.0, 10.0, 31)
    g1 = np.linspace(20.0, 40.0, 31)
    groups = [g0, g1]

    res = raincloud_plot(groups, orientation=orientation)
    try:
        ax = res["axes"]
        violins = _violin_collections(ax)
        iqrs = _iqr_collections(ax)
        medians = _median_collections(ax)
        rains = _rain_collections(ax)

        assert len(violins) == len(iqrs) == len(medians) == len(rains) == 2

        for pos, vals, violin, iqr, med, rain in zip([0.0, 1.0], groups, violins, iqrs, medians, rains):
            q1, q50, q3 = np.percentile(vals, [25, 50, 75])
            verts = violin.get_paths()[0].vertices
            seg = iqr.get_segments()[0]
            med_xy = med.get_offsets()[0]
            rain_xy = rain.get_offsets()

            if orientation == "vertical":
                assert verts[:, 0].max() <= pos + 1e-6
                assert verts[:, 0].min() < pos - 1e-3
                np.testing.assert_allclose(seg[:, 0], [pos, pos])
                np.testing.assert_allclose(sorted(seg[:, 1]), [q1, q3])
                np.testing.assert_allclose(med_xy, [pos, q50])
                np.testing.assert_allclose(rain_xy[:, 1], vals)
                assert rain_xy[:, 0].mean() > pos + 0.1
            else:
                assert verts[:, 1].min() >= pos - 1e-6
                assert verts[:, 1].max() > pos + 1e-3
                np.testing.assert_allclose(sorted(seg[:, 0]), [q1, q3])
                np.testing.assert_allclose(seg[:, 1], [pos, pos])
                np.testing.assert_allclose(med_xy, [q50, pos])
                np.testing.assert_allclose(rain_xy[:, 0], vals)
                assert rain_xy[:, 1].mean() < pos - 0.1
    finally:
        plt.close(res["fig"])


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_raincloud_plot_hue_dodge_and_categorical_order(orientation):
    """Verify hue artist counts, dodge=True/False/auto positions, distinct colors, and CategoricalDtype order.

    Strategy: build a DataFrame with explicitly ordered CategoricalDtype columns ('Late'
    before 'Early', 'Treat' before 'Ctrl') and test dodge=True, dodge=False, auto-dodge
    with hue != cat, and auto-dodge with hue == cat across both orientations.
    Expected: categorical order is preserved on ticks, dodge=True symmetrically offsets hues
    by ±width/4 around each tick, dodge=False and hue==cat keep centers at integer ticks,
    and distinct hue levels receive distinct consistent colors.
    """
    rng = np.random.default_rng(10)
    n = 15
    df = pd.DataFrame(
        {
            "stage": pd.Categorical(
                np.repeat(["Late", "Early"], 2 * n), categories=["Late", "Early"], ordered=True
            ),
            "cond": pd.Categorical(
                np.tile(np.repeat(["Treat", "Ctrl"], n), 2), categories=["Treat", "Ctrl"], ordered=True
            ),
            "val": rng.normal(5.0, 1.0, size=4 * n),
        }
    )
    xy_kw = {"x": "stage", "y": "val"} if orientation == "vertical" else {"x": "val", "y": "stage"}
    cat_coord = 0 if orientation == "vertical" else 1

    # 1. dodge=True (and default auto-dodge when hue != cat)
    for dodge_val in (True, None):
        res = raincloud_plot(df, hue="cond", orientation=orientation, dodge=dodge_val, width_violin=0.5, **xy_kw)
        try:
            ax = res["axes"]
            ticks = [t.get_text() for t in (ax.get_xticklabels() if orientation == "vertical" else ax.get_yticklabels())]
            assert ticks == ["Late", "Early"]
            if orientation == "vertical":
                assert ax.get_xlabel() == "stage" and ax.get_ylabel() == "val"
            else:
                assert ax.get_xlabel() == "val" and ax.get_ylabel() == "stage"

            violins = _violin_collections(ax)
            medians = _median_collections(ax)
            assert len(violins) == 4 and len(medians) == 4

            positions = [m.get_offsets()[0][cat_coord] for m in medians]
            np.testing.assert_allclose(positions, [-0.125, 0.125, 0.875, 1.125])

            # Colors match by hue across categories and differ between hues
            colors = [v.get_facecolor()[0][:3] for v in violins]
            np.testing.assert_allclose(colors[0], colors[2])
            np.testing.assert_allclose(colors[1], colors[3])
            assert not np.allclose(colors[0], colors[1])
        finally:
            plt.close(res["fig"])

    # 2. dodge=False stacks hues on the category center
    res_nododge = raincloud_plot(df, hue="cond", orientation=orientation, dodge=False, **xy_kw)
    try:
        positions = [m.get_offsets()[0][cat_coord] for m in _median_collections(res_nododge["axes"])]
        np.testing.assert_allclose(positions, [0.0, 0.0, 1.0, 1.0])
    finally:
        plt.close(res_nododge["fig"])

    # 3. Auto-dodge turns off when hue == category column (centers stay at 0.0, 1.0)
    res_auto = raincloud_plot(df, hue="stage", orientation=orientation, dodge=None, **xy_kw)
    try:
        ax_auto = res_auto["axes"]
        violins = _violin_collections(ax_auto)
        medians = _median_collections(ax_auto)
        assert len(violins) == 2 and len(medians) == 2
        positions = [m.get_offsets()[0][cat_coord] for m in medians]
        np.testing.assert_allclose(positions, [0.0, 1.0])
        assert not np.allclose(violins[0].get_facecolor()[0][:3], violins[1].get_facecolor()[0][:3])
    finally:
        plt.close(res_auto["fig"])


def test_raincloud_plot_outlier_kde_exclusion_and_fallback():
    """Verify outlier_method trims the violin KDE while keeping all scatter points, plus all-outlier fallback.

    Strategy: append an extreme outlier (1000.0) to a normal group and compare violin y-extents
    and scatter points with outlier_method=None vs 'robust_zscore' and 'iqr'; then test a
    2-point group whose strict IQR multiplier removes every point.
    Expected: violin KDE max stays near the inliers (< 20) when filtering is enabled while
    the scatter still contains 1000.0; when all points are filtered, fallback still renders
    a valid violin spanning the full group.
    """
    rng = np.random.default_rng(42)
    vals = np.concatenate([rng.normal(0.0, 1.0, 40), [1000.0]])
    df = pd.DataFrame({"group": ["A"] * len(vals), "value": vals})

    res_raw = raincloud_plot(df, x="group", y="value", outlier_method=None)
    res_rz = raincloud_plot(df, x="group", y="value", outlier_method="robust_zscore", outlier_multiplier=3.0)
    res_iqr = raincloud_plot(df, x="group", y="value", outlier_method="iqr", outlier_multiplier=1.5)
    try:
        raw_ymax = _violin_collections(res_raw["axes"])[0].get_paths()[0].vertices[:, 1].max()
        assert raw_ymax >= 900.0

        for res_filt in (res_rz, res_iqr):
            ax = res_filt["axes"]
            filt_ymax = _violin_collections(ax)[0].get_paths()[0].vertices[:, 1].max()
            assert filt_ymax < 20.0
            rain_y = _rain_collections(ax)[0].get_offsets()[:, 1]
            assert len(rain_y) == len(vals)
            assert 1000.0 in rain_y
    finally:
        plt.close(res_raw["fig"])
        plt.close(res_rz["fig"])
        plt.close(res_iqr["fig"])

    # All-outliers fallback: two points at ±1000 are both outside 0.0001 * IQR
    df_all_out = pd.DataFrame({"group": ["A", "A"], "value": [1000.0, -1000.0]})
    res_fb = raincloud_plot(df_all_out, x="group", y="value", outlier_method="iqr", outlier_multiplier=0.0001)
    try:
        violins = _violin_collections(res_fb["axes"])
        assert len(violins) == 1
        y_verts = violins[0].get_paths()[0].vertices[:, 1]
        assert y_verts.min() <= -1000.0 + 1e-6 and y_verts.max() >= 1000.0 - 1e-6
    finally:
        plt.close(res_fb["fig"])


def test_raincloud_plot_styling_palettes_highlights_and_validation():
    """Verify styling kwargs, palette resolution, highlight masks across formats, and input validation.

    Strategy: inspect artist alpha/size/linewidth attributes; check colormap, dict, list, and
    invalid palette strings; verify highlight point counts for Seaborn DataFrame (Series and
    ndarray with NaNs), dict-of-masks, and list-of-masks; and trigger all ValueError paths.
    Expected: every styling property and palette color matches the request, highlight counts
    are exact after NaN filtering, and invalid inputs raise ValueError.
    """
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "group": ["A"] * 12 + ["B"] * 12,
            "value": np.r_[rng.normal(0, 1, 12), rng.normal(3, 1, 12)],
        }
    )

    # 1. Styling kwargs reach artists
    res_style = raincloud_plot(
        df,
        x="group",
        y="value",
        alpha_violin=0.35,
        alpha_scatter=0.25,
        size_scatter=18,
        size_median=75,
        linewidth_scatter=1.25,
        linewidth_boxplot=4.5,
        offset_scatter=0.15,
    )
    try:
        ax = res_style["axes"]
        for pc in _violin_collections(ax):
            assert pc.get_alpha() == pytest.approx(0.35)
        for rc in _rain_collections(ax):
            assert rc.get_alpha() == pytest.approx(0.25)
            assert rc.get_sizes()[0] == pytest.approx(18)
            assert rc.get_linewidths()[0] == pytest.approx(1.25)
        for mc in _median_collections(ax):
            assert mc.get_sizes()[0] == pytest.approx(75)
        for lc in _iqr_collections(ax):
            assert lc.get_linewidths()[0] == pytest.approx(4.5)
    finally:
        plt.close(res_style["fig"])

    # 2. Palette resolution: colormap string, dict, list, and invalid string ValueError
    res_cmap = raincloud_plot(df, x="group", y="value", hue="group", palette="viridis")
    try:
        vcols = [v.get_facecolor()[0][:3] for v in _violin_collections(res_cmap["axes"])]
        viridis = mpl_colormaps["viridis"]
        np.testing.assert_allclose(vcols[0], viridis(0.0)[:3], atol=1e-6)
        np.testing.assert_allclose(vcols[1], viridis(1.0)[:3], atol=1e-6)
    finally:
        plt.close(res_cmap["fig"])

    res_dict_pal = raincloud_plot(df, x="group", y="value", palette={"A": "#ff0000", "B": "#0000ff"})
    try:
        vcols = [v.get_facecolor()[0][:3] for v in _violin_collections(res_dict_pal["axes"])]
        np.testing.assert_allclose(vcols[0], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(vcols[1], (0.0, 0.0, 1.0))
    finally:
        plt.close(res_dict_pal["fig"])

    res_list_pal = raincloud_plot([df["value"][:12], df["value"][12:]], palette=["#00ff00", "#ff00ff"])
    try:
        vcols = [v.get_facecolor()[0][:3] for v in _violin_collections(res_list_pal["axes"])]
        np.testing.assert_allclose(vcols[0], (0.0, 1.0, 0.0))
        np.testing.assert_allclose(vcols[1], (1.0, 0.0, 1.0))
    finally:
        plt.close(res_list_pal["fig"])

    with pytest.raises(ValueError):
        raincloud_plot(df, x="group", y="value", hue="group", palette="NotAColormap")

    # 3. Highlight masks: Seaborn DataFrame (Series and ndarray with NaN), dict, and list of groups
    df_nan = df.copy()
    df_nan.loc[0, "value"] = np.nan
    mask_series = df_nan["value"] > 1.0
    expected_hi = int(mask_series.sum())
    for mask_arg in (mask_series, mask_series.to_numpy()):
        res_hi = raincloud_plot(df_nan, x="group", y="value", highlight_mask=mask_arg, highlight_color="lime")
        try:
            assert _scatter_points(res_hi["axes"], "lime") == expected_hi
        finally:
            plt.close(res_hi["fig"])

    g_a = rng.normal(0, 1, 15)
    g_b = rng.normal(0, 1, 12)
    m_a = g_a > 0.5
    m_b = g_b < -0.5
    res_dict_hi = raincloud_plot(
        {"a": g_a, "b": g_b}, highlight_mask={"a": m_a, "b": m_b}, highlight_color="magenta"
    )
    try:
        assert _scatter_points(res_dict_hi["axes"], "magenta") == int(m_a.sum() + m_b.sum())
    finally:
        plt.close(res_dict_hi["fig"])

    res_list_hi = raincloud_plot([g_a, g_b], highlight_mask=[m_a, m_b], highlight_color="cyan")
    try:
        assert _scatter_points(res_list_hi["axes"], "cyan") == int(m_a.sum() + m_b.sum())
    finally:
        plt.close(res_list_hi["fig"])

    # 4. Input validation errors
    with pytest.raises(ValueError, match="y must be specified"):
        raincloud_plot(df, x="group", orientation="vertical")
    with pytest.raises(ValueError, match="x must be specified"):
        raincloud_plot(df, y="group", orientation="horizontal")
    with pytest.raises(ValueError, match="Column nope not found"):
        raincloud_plot(df, x="nope", y="value")
    with pytest.raises(ValueError, match="Column nope not found"):
        raincloud_plot(df, x="group", y="nope")
    with pytest.raises(ValueError, match="Column nope not found"):
        raincloud_plot(df, x="group", y="value", hue="nope")
    with pytest.raises(ValueError, match="Unsupported data type"):
        raincloud_plot(42)


def test_raincloud_plot_raster_threshold():
    """Verify raincloud_plot rasterizes scatter points only when total count exceeds raster_threshold.

    Strategy: render a 10-point DataFrame and a 100-point DataFrame with raster_threshold=50
    and inspect get_rasterized() on the rain PathCollections.
    Expected: scatter collections stay unrasterized at 10 points and are rasterized at 100 points.
    """
    data_small = pd.DataFrame({"group": ["A"] * 5 + ["B"] * 5, "value": range(10)})
    res_small = raincloud_plot(data=data_small, x="group", y="value", raster_threshold=50)
    try:
        scatter_small = [
            c for c in res_small["axes"].collections if isinstance(c, PathCollection) and len(c.get_offsets()) == 5
        ]
        assert len(scatter_small) == 2
        assert all(c.get_rasterized() is False for c in scatter_small)
    finally:
        plt.close(res_small["fig"])

    data_large = pd.DataFrame({"group": ["A"] * 50 + ["B"] * 50, "value": range(100)})
    res_large = raincloud_plot(data=data_large, x="group", y="value", raster_threshold=50)
    try:
        scatter_large = [
            c for c in res_large["axes"].collections if isinstance(c, PathCollection) and len(c.get_offsets()) == 50
        ]
        assert len(scatter_large) == 2
        assert all(c.get_rasterized() is True for c in scatter_large)
    finally:
        plt.close(res_large["fig"])


def test_raincloud_accepts_series_and_tuple():
    """Verify raincloud_plot accepts a named pd.Series and a plain tuple of floats.

    Strategy: plot a named Series and a tuple of values and inspect the tick label and output.
    Expected: the Series name becomes the tick label and tuple input renders without error.
    """
    values = np.random.default_rng(0).normal(size=20)
    res_series = raincloud_plot(pd.Series(values, name="speed"))
    res_tuple = raincloud_plot(tuple(values))
    try:
        assert [t.get_text() for t in res_series["axes"].get_xticklabels()] == ["speed"]
        assert [t.get_text() for t in res_tuple["axes"].get_xticklabels()] == ["Data"]
    finally:
        plt.close(res_series["fig"])
        plt.close(res_tuple["fig"])


def test_raincloud_highlight_mask_for_arrays_and_dataframe_columns():
    """Verify highlight_mask stays aligned after NaN removal for arrays/DFs and warns on mismatch.

    Strategy: inject a NaN into a 1D array, pass a boolean mask for both array and wide
    DataFrame inputs, and pass an unmatched 1D mask for a list of two groups.
    Expected: highlighted point count equals non-NaN True count; mismatched mask emits UserWarning.
    """
    a = np.random.default_rng(0).normal(size=50)
    a[3] = np.nan
    expected = int(np.nansum(a > 1))

    res_arr = raincloud_plot(a, highlight_mask=a > 1, highlight_color="lime")
    res_df = raincloud_plot(pd.DataFrame({"v": a}), highlight_mask=a > 1, highlight_color="lime")
    try:
        assert _scatter_points(res_arr["axes"], "lime") == expected
        assert _scatter_points(res_df["axes"], "lime") == expected
        with pytest.warns(UserWarning, match="highlight_mask"):
            res_warn = raincloud_plot([a, a], highlight_mask=a > 1)
            plt.close(res_warn["fig"])
    finally:
        plt.close(res_arr["fig"])
        plt.close(res_df["fig"])


def test_raincloud_mad_outlier_method_matches_robust_zscore():
    """Verify documented outlier_method='mad' produces identical violins to 'robust_zscore'.

    Strategy: render two groups (one with an outlier at 25.0) using 'mad' and 'robust_zscore'
    and compare the resulting violin polygon vertices element-by-element.
    Expected: both calls produce two identical violin polygons.
    """
    rng = np.random.default_rng(0)
    data = [np.r_[rng.normal(size=30), 25.0], rng.normal(size=30)]
    res_mad = raincloud_plot(data, outlier_method="mad")
    res_rz = raincloud_plot(data, outlier_method="robust_zscore")
    try:
        v_mad = [c.get_paths()[0].vertices for c in _violin_collections(res_mad["axes"])]
        v_rz = [c.get_paths()[0].vertices for c in _violin_collections(res_rz["axes"])]
        assert len(v_mad) == 2
        for a, b in zip(v_mad, v_rz):
            np.testing.assert_allclose(a, b)
    finally:
        plt.close(res_mad["fig"])
        plt.close(res_rz["fig"])


def test_raincloud_empty_group_keeps_its_tick():
    """Verify an all-NaN group retains its tick position and label without crashing.

    Strategy: plot dict and wide DataFrame inputs where the second group is all-NaN.
    Expected: both tick labels ['a', 'b'] remain present on the x-axis.
    """
    res_dict = raincloud_plot({"a": [1.0, 2.0, 3.0], "b": [np.nan, np.nan]})
    res_df = raincloud_plot(pd.DataFrame({"a": np.arange(10.0), "b": [np.nan] * 10}))
    try:
        assert [t.get_text() for t in res_dict["axes"].get_xticklabels()] == ["a", "b"]
        assert [t.get_text() for t in res_df["axes"].get_xticklabels()] == ["a", "b"]
    finally:
        plt.close(res_dict["fig"])
        plt.close(res_df["fig"])


def test_raincloud_leaves_callers_figure_and_pyplot_alone():
    """Verify raincloud_plot with an explicit ax never spawns pyplot figures or alters layout engines.

    Strategy: draw onto an OO Figure's axes and onto a pyplot figure with constrained layout
    while a second pyplot figure is open.
    Expected: no stray pyplot figures are created and existing figure layout engines are untouched.
    """
    plt.close("all")
    data = [np.random.default_rng(0).normal(size=20)]
    fig = Figure()
    raincloud_plot(data, ax=fig.subplots())
    assert plt.get_fignums() == []

    fig1, ax1 = plt.subplots(layout="constrained")
    fig2, _ = plt.subplots()
    try:
        raincloud_plot(data, ax=ax1)
        assert type(fig1.get_layout_engine()).__name__ == "ConstrainedLayoutEngine"
        assert fig2.get_layout_engine() is None
    finally:
        plt.close(fig1)
        plt.close(fig2)


# =========================================
# Colormaps (colormap_maker, adjust_colormap, I-series, labels_cmap)
# =========================================


def test_colormap_maker_basic():
    """Verify colormap_maker creates an unregistered 256-entry custom colormap by default.

    Strategy: build a colormap from mixed color specifications without cmap_name.
    Expected: name defaults to 'custom_cmap', N == 256, and Matplotlib registry rejects lookup.
    """
    colors = ["k", "cyan", (1.0, 1.0, 1.0, 0.5)]
    cmap = colormap_maker(colors)

    assert isinstance(cmap, LinearSegmentedColormap)
    assert cmap.name == "custom_cmap"
    assert cmap.N == 256
    with pytest.raises(ValueError):
        plt.get_cmap("custom_cmap")


def test_colormap_maker_with_positions():
    """Verify colormap_maker maps non-uniform stop positions to exact RGBA endpoints.

    Strategy: create a red-green-blue colormap anchored at positions [0.0, 0.2, 1.0].
    Expected: evaluating the colormap at 0.0, 0.2, and 1.0 returns exact red, green, and blue.
    """
    cmap = colormap_maker(["red", "green", "blue"], positions=[0.0, 0.2, 1.0])

    assert cmap.name == "custom_cmap"
    assert cmap(0.0) == mpl.colors.to_rgba("red")
    assert cmap(0.2) == mpl.colors.to_rgba("green")
    assert cmap(1.0) == mpl.colors.to_rgba("blue")


def test_colormap_maker_registration():
    """Verify colormap_maker registers a named colormap usable by name in imshow, then cleans up.

    Strategy: register a multi-stop colormap under a unique name, look it up via plt.get_cmap,
    render an imshow using the string name, and unregister in a finally block to avoid leaks.
    Expected: registered colormap matches by name and ax.imshow binds the registered colormap.
    """
    colors = ["#08041c", "#390b5e", "#a2217c", "#f04e4c", "#fce205"]
    positions = [0.0, 0.2, 0.5, 0.8, 1.0]
    cmap_name = "synthwave_test_reg"
    try:
        cmap = colormap_maker(colors, positions=positions, cmap_name=cmap_name)
        assert cmap.name == cmap_name

        registered_cmap = plt.get_cmap(cmap_name)
        assert registered_cmap.name == cmap_name

        rng = np.random.default_rng(0)
        fig, ax = plt.subplots()
        try:
            im = ax.imshow(rng.random((8, 8)), cmap=cmap_name)
            assert im.cmap.name == cmap_name
        finally:
            plt.close(fig)
    finally:
        if hasattr(mpl.colormaps, "unregister") and cmap_name in mpl.colormaps:
            mpl.colormaps.unregister(cmap_name)


def test_colormap_maker_validation():
    """Verify colormap_maker rejects mismatched, unanchored, or non-monotonic position lists.

    Strategy: invoke colormap_maker with wrong length, non-zero start, non-one end, and
    decreasing interior positions.
    Expected: each invalid configuration raises ValueError with a descriptive message.
    """
    colors = ["black", "white"]

    with pytest.raises(ValueError, match="The number of positions must match"):
        colormap_maker(colors, positions=[0.0])
    with pytest.raises(ValueError, match="Positions must start with 0.0 and end with 1.0"):
        colormap_maker(colors, positions=[0.1, 1.0])
    with pytest.raises(ValueError, match="Positions must start with 0.0 and end with 1.0"):
        colormap_maker(colors, positions=[0.0, 0.9])
    with pytest.raises(ValueError, match="Positions must be strictly monotonically increasing"):
        colormap_maker(["red", "green", "blue", "yellow"], positions=[0.0, 0.6, 0.4, 1.0])


def test_adjust_colormap():
    """Verify adjust_colormap prepends start_color and slices the source colormap over [start_, end_].

    Strategy: adjust 'viridis' with default arguments and with custom start_/end_/start_color.
    Expected: returns a LinearSegmentedColormap named '<name>_adjusted' whose 0.0 stop equals
    start_color and whose 1.0 stop equals the source colormap evaluated at end_.
    """
    default_adj = adjust_colormap("viridis")
    orig = plt.get_cmap("viridis")
    assert isinstance(default_adj, LinearSegmentedColormap)
    assert default_adj.name == "viridis_adjusted"
    np.testing.assert_allclose(default_adj(0.0), [0.9, 0.9, 0.9, 1.0], atol=1e-6)
    np.testing.assert_allclose(default_adj(1.0), orig(1.0), atol=1e-6)

    custom_start = [0.1, 0.2, 0.3, 1.0]
    custom_adj = adjust_colormap("plasma", start_=0.3, end_=0.8, start_color=custom_start)
    plasma = plt.get_cmap("plasma")
    assert isinstance(custom_adj, LinearSegmentedColormap)
    assert custom_adj.name == "plasma_adjusted"
    np.testing.assert_allclose(custom_adj(0.0), custom_start, atol=1e-6)
    np.testing.assert_allclose(custom_adj(1.0), plasma(0.8), atol=1e-6)


def test_i_series_colormaps_registration():
    """Verify all I-series colormaps and snake_case aliases are registered and exported."""
    expected_names = [
        "I Blue",
        "I Bordeaux",
        "I Forest",
        "I Orange",
        "I Purple",
        "I Teal",
        "I Rose",
        "I Gold",
        "I Slate",
    ]
    assert set(I_SERIES_COLORMAPS.keys()) == set(expected_names)

    for name in expected_names:
        registered_cmap = plt.get_cmap(name)
        assert registered_cmap.name == name
        alias_cmap = plt.get_cmap(name.lower().replace(" ", "_"))
        assert alias_cmap.name == name

    exports = [i_blue, i_bordeaux, i_forest, i_orange, i_purple, i_teal, i_rose, i_gold, i_slate]
    assert [cm.name for cm in exports] == expected_names


def test_i_series_colormaps_properties():
    """Verify I-series colormaps start at pure white (0.0) and end at a deep chromatic shade (1.0)."""
    for name, cmap in I_SERIES_COLORMAPS.items():
        start_color = cmap(0.0)
        end_color = cmap(1.0)
        assert np.allclose(start_color[:3], (1.0, 1.0, 1.0), atol=1e-3), f"{name} must start at white for 0.0"
        assert not np.allclose(end_color[:3], (1.0, 1.0, 1.0), atol=1e-2), f"{name} must not end at white for 1.0"
        assert not np.allclose(end_color[:3], (0.0, 0.0, 0.0), atol=1e-2), f"{name} must not end at black for 1.0"


def test_i_series_colormaps_usage():
    """Verify registered I-series colormaps bind properly in describe_image and color_coded_projection."""
    rng = np.random.default_rng(0)
    res = describe_image(rng.random((16, 16)), cmap="I Blue")
    try:
        im = res["axes"]["Image"].images[0]
        assert im.cmap.name == "I Blue"
    finally:
        plt.close(res["fig"])

    projection = color_coded_projection(rng.random((3, 16, 16)), color_map="I Bordeaux")
    assert projection.shape == (16, 16, 3)
    assert np.isfinite(projection).all() and 0.0 <= projection.min() and projection.max() <= 1.0


def test_labels_cmap_schema_and_scalar_masked_handling():
    """Verify LABELS_CMAP_COLORS schema and _LabelsColormap scalar/masked lookup behavior.

    Strategy: validate the 255-color RGB table schema, check labels_cmap.colors alignment,
    then evaluate labels_cmap on scalar floats (0.0, small positive < 1/256, 1.0) and a
    masked float array containing both an unmasked and a masked sub-1/256 positive value.
    Expected: 255 RGB tuples in [0, 1], index 0 transparent, positive floats below 1/256
    map to opaque entry 1, and masked entries stay at the colormap's bad color.
    """
    assert len(LABELS_CMAP_COLORS) == 255
    for rgb in LABELS_CMAP_COLORS:
        assert isinstance(rgb, tuple) and len(rgb) == 3
        assert all(isinstance(ch, (int, float)) and 0.0 <= ch <= 1.0 for ch in rgb)

    assert len(labels_cmap.colors) == 256
    assert labels_cmap.colors[0] == (0, 0, 0, 0)
    expected_rgba = [(0, 0, 0, 0)] + [(r, g, b, 1.0) for r, g, b in LABELS_CMAP_COLORS]
    np.testing.assert_allclose(labels_cmap.colors, expected_rgba)

    # Scalar float inputs
    assert labels_cmap(0.0) == (0.0, 0.0, 0.0, 0.0)
    low_scalar = labels_cmap(0.001)  # 0 < 0.001 < 1/256 -> bumped to entry 1
    np.testing.assert_allclose(low_scalar, labels_cmap.colors[1])
    assert low_scalar[3] == 1.0
    np.testing.assert_allclose(labels_cmap(1.0), labels_cmap.colors[-1])

    # Masked array: unmasked 0.001 moves to entry 1; masked 0.001 stays bad (transparent)
    arr = np.ma.array([0.0, 0.001, 0.001], mask=[False, False, True])
    rgba = labels_cmap(arr)
    np.testing.assert_allclose(rgba[0], (0.0, 0.0, 0.0, 0.0))
    np.testing.assert_allclose(rgba[1], labels_cmap.colors[1])
    np.testing.assert_allclose(rgba[2], labels_cmap.get_bad())


def test_labels_cmap():
    """Verify labels_cmap has transparent index 0 and CIELAB lightness L* >= 40 for all labels."""
    skimage_color = pytest.importorskip("skimage.color")

    assert labels_cmap.name == "labels_cmap"
    assert labels_cmap(0.0) == (0.0, 0.0, 0.0, 0.0)

    # All non-transparent colors (1..255) must have lightness L* >= 40 to pop against black
    cmap_colors = np.asarray(labels_cmap.colors)[1:, :3]
    lab_colors = skimage_color.rgb2lab(cmap_colors)
    min_lightness = lab_colors[:, 0].min()
    assert min_lightness >= 40.0, f"Label colors must have L* >= 40 on black, got min {min_lightness:.2f}"


def test_labels_cmap_diversity():
    """Verify labels_cmap colors are perceptually well separated overall and between adjacent entries."""
    skimage_color = pytest.importorskip("skimage.color")

    rgb_colors = np.asarray(labels_cmap.colors)[1:, :3]
    lab_colors = skimage_color.rgb2lab(rgb_colors)

    min_overall_dist = float("inf")
    for i in range(len(lab_colors)):
        dists = skimage_color.deltaE_ciede2000(lab_colors[i : i + 1], lab_colors)
        dists[i] = float("inf")
        min_overall_dist = min(min_overall_dist, float(np.min(dists)))

    min_adj_dist = float("inf")
    for i in range(len(lab_colors) - 1):
        d = float(skimage_color.deltaE_ciede2000(lab_colors[i : i + 1], lab_colors[i + 1 : i + 2])[0])
        min_adj_dist = min(min_adj_dist, d)

    assert min_overall_dist > 0.4, f"Overall min CIEDE2000 distance too small: {min_overall_dist}"
    assert min_adj_dist > 18.0, f"Adjacent colors too similar! Min adjacent distance: {min_adj_dist}"


def test_labels_cmap_low_ids_stay_opaque_with_many_labels():
    """With >256 labels the lowest ids must not fall on the transparent entry."""
    lab = np.zeros((4, 4), int)
    lab[0, 0] = 1
    lab[0, 1] = 1000
    fig, ax = plt.subplots()
    try:
        im = ax.imshow(lab, cmap=labels_cmap, interpolation="nearest")
        rgba = im.to_rgba(lab)
    finally:
        plt.close(fig)

    assert rgba[1, 1, 3] == 0  # background stays transparent
    assert rgba[0, 0, 3] == 1  # label 1 stays opaque
    assert rgba[0, 1, 3] == 1
    # Entries match the public lookup table: labels_cmap(i / 255) == labels_cmap.colors[i]
    np.testing.assert_allclose(labels_cmap(np.arange(256) / 255), np.asarray(labels_cmap.colors))


# =========================================
# color_coded_projection
# =========================================


def test_color_coded_projection_basic():
    """Verify color_coded_projection maps a single active frame to its exact colormap RGB value."""
    img = np.zeros((3, 2, 2), dtype=float)
    img[1] = 1.0
    result = color_coded_projection(img, color_map="plasma")
    assert result.shape == (2, 2, 3)
    expected_color = mpl_colormaps["plasma"](1 / (3 - 1))[:3]
    expected = np.stack([np.full((2, 2), expected_color[c]) for c in range(3)], axis=-1)
    assert np.allclose(result, expected)


def test_color_coded_projection_multiframe_max_and_nan_frames():
    """Verify elementwise channel max across frames and finite output with partial/all-NaN frames.

    Strategy: activate distinct and overlapping pixels across 3 frames and verify per-channel
    maximum; then project a 3-frame stack containing one partially-NaN frame and one all-NaN frame.
    Expected: overlapping pixels receive elementwise max of contributing frame colors, silent
    pixels stay zero, and NaN pixels/frames produce finite outputs in [0, 1].
    """
    cmap = mpl_colormaps["plasma"]
    c0, c1, c2 = cmap(0.0)[:3], cmap(0.5)[:3], cmap(1.0)[:3]

    img = np.zeros((3, 2, 2), dtype=np.float32)
    img[0, 0, 0] = 1.0
    img[1, 0, 0] = 1.0
    img[1, 0, 1] = 1.0
    img[2, 1, 1] = 1.0

    out = color_coded_projection(img, color_map="plasma")
    np.testing.assert_allclose(out[0, 0], np.maximum(c0, c1), atol=1e-6)
    np.testing.assert_allclose(out[0, 1], c1, atol=1e-6)
    np.testing.assert_allclose(out[1, 1], c2, atol=1e-6)
    np.testing.assert_allclose(out[1, 0], [0.0, 0.0, 0.0], atol=1e-6)

    # Partial-NaN frame + all-NaN frame
    nan_stack = np.zeros((3, 2, 2), dtype=np.float32)
    nan_stack[0] = [[np.nan, 0.0], [1.0, 0.5]]
    nan_stack[1] = np.nan
    nan_stack[2] = [[0.0, 1.0], [0.0, 0.0]]
    nan_out = color_coded_projection(nan_stack, color_map="plasma")
    assert nan_out.shape == (2, 2, 3)
    assert np.isfinite(nan_out).all()
    assert nan_out.min() >= 0.0 and nan_out.max() <= 1.0
    np.testing.assert_allclose(nan_out[0, 0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(nan_out[1, 0], c0, atol=1e-6)
    np.testing.assert_allclose(nan_out[0, 1], c2, atol=1e-6)


@pytest.mark.parametrize("shape", [(4,), (2, 2), (2, 3, 4, 4)])
def test_color_coded_projection_invalid_dims(shape):
    """Verify color_coded_projection rejects non-3D arrays (1D, 2D, 4D) with ValueError."""
    with pytest.raises(ValueError, match="three dimensions"):
        color_coded_projection(np.zeros(shape))


def test_color_coded_projection_stays_in_unit_range_and_handles_int16():
    """Verify constant frames stay clipped to [0, 1] and signed int16 does not overflow."""
    rng = np.random.default_rng(0)
    img = rng.random((4, 8, 8)).astype(np.float32)
    img[2] = 1000.0  # constant frame
    assert color_coded_projection(img).max() <= 1.0

    # int16 values -20000..20000: middle value is halfway, not lost to integer overflow
    img_i16 = np.zeros((3, 1, 3), np.int16)
    img_i16[:, 0, :] = [-20000, 0, 20000]
    out = color_coded_projection(img_i16)
    ref = color_coded_projection(img_i16.astype(np.float64))
    np.testing.assert_allclose(out, ref, atol=1e-6)
    assert out[0, 1].max() > 0.4


def test_color_coded_projection_single_frame():
    """Verify a 1-frame 3D stack projects cleanly without dividing by zero."""
    out = color_coded_projection(np.random.default_rng(0).random((1, 8, 8)))
    assert out.shape == (8, 8, 3) and np.isfinite(out).all()


# =========================================
# describe_image & hist_imshow
# =========================================


def test_describe_image_return_image_only_3d_and_4d():
    """Verify describe_image extracts the middle slice for 3D and 4D arrays and reports it in info."""
    rng = np.random.default_rng(0)
    img3 = rng.random((3, 4, 5))
    slice3 = describe_image(img3, return_image_only=True)
    assert slice3.shape == (4, 5)
    np.testing.assert_allclose(slice3, img3[1])

    img4 = rng.random((3, 5, 6, 7))
    slice4 = describe_image(img4, return_image_only=True)
    assert slice4.shape == (6, 7)
    np.testing.assert_allclose(slice4, img4[1, 2])

    res4 = describe_image(img4)
    try:
        assert "shape: (3, 5, 6, 7)" in res4["info"]
        assert "shown: middle slice (1, 2)" in res4["info"]
        np.testing.assert_allclose(res4["axes"]["Image"].images[0].get_array(), img4[1, 2])
    finally:
        plt.close(res4["fig"])


def test_describe_image_axes_dict_contents():
    """Verify describe_image returns Image/Histogram/Info Axes and metadata summary text."""
    rng = np.random.default_rng(1)
    res = describe_image(rng.random((5, 5)))
    try:
        axes = res["axes"]
        assert list(axes.keys()) == ["Image", "Histogram", "Info"]
        assert "shape: (5, 5)" in res["info"] and "dtype: float64" in res["info"]
        assert all(isinstance(ax, Axes) for ax in axes.values())
    finally:
        plt.close(res["fig"])


def test_describe_image_accepts_bool_lists_and_gamma_with_limits():
    """Verify describe_image handles boolean arrays, Python lists, and gamma with vmin/vmax."""
    rng = np.random.default_rng(0)
    res_bool = describe_image(rng.random((16, 16)) > 0.5)
    res_list = describe_image([[1, 2], [3, 4]])
    res_gamma = describe_image(rng.random((8, 8)), gamma=0.5, vmin=0.2, vmax=0.8)
    try:
        assert "bool" in res_bool["axes"]["Histogram"].get_xlabel()
        assert "dtype: bool" in res_bool["info"]
        norm = res_gamma["axes"]["Image"].images[0].norm
        assert (norm.gamma, norm.vmin, norm.vmax) == (0.5, 0.2, 0.8)
    finally:
        plt.close(res_bool["fig"])
        plt.close(res_list["fig"])
        plt.close(res_gamma["fig"])


def test_describe_image_dataarray_info_panel_wrapping_and_return_image_only():
    """Verify DataArray info panel metadata, non-clipping, info_width wrapping, and return_image_only."""
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray, set_channel_info

    arr = np.random.default_rng(1).random((3, 4, 32, 40)).astype(np.float32)
    da = array_to_stczyx_xarray(
        arr,
        input_axes="CZYX",
        channel_names=["DAPI", "WGA", "TOMM20"],
        pixel_sizes={"Z": 2.0, "Y": 0.108, "X": 0.108, "unit": "µm"},
    )
    da = set_channel_info(da, {"DAPI": {"role": "dna"}, "WGA": {"role": "membrane"}, "TOMM20": {"role": "mito"}})

    # return_image_only extracts the 2D middle slice directly as an ndarray
    slice_only = describe_image(da, return_image_only=True)
    assert isinstance(slice_only, np.ndarray) and slice_only.shape == (32, 40)
    np.testing.assert_allclose(slice_only, arr[1, 2])

    res = describe_image(da)
    try:
        info = res["info"]
        assert "pixel size: Z=2 Y=0.108 X=0.108 µm" in info
        assert "C[2] TOMM20  role=mito" in info
        assert "shown: S=0, T=0, C=WGA, Z=2" in info

        fig = res["fig"]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fb = fig.bbox
        for txt in res["axes"]["Info"].texts:
            bb = txt.get_window_extent(renderer)
            assert bb.x0 >= fb.x0 - 1 and bb.x1 <= fb.x1 + 1 and bb.y0 >= fb.y0 - 1
    finally:
        plt.close(res["fig"])

    # info_width wraps long value lines in the info table to <= info_width characters
    da_long = da.copy()
    da_long.attrs["source"] = "/very/long/path/" + "segment_" * 10 + "image.tif"
    res_wrap = describe_image(da_long, info_width=24)
    try:
        rendered_lines = [t.get_text() for t in res_wrap["axes"]["Info"].texts]
        assert any("segment_" in line for line in rendered_lines)
        assert all(len(line) <= 24 for line in rendered_lines if "segment_" in line)
    finally:
        plt.close(res_wrap["fig"])


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
        try:
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
        finally:
            plt.close(fig)


def test_describe_image_histogram_shows_pixel_fractions_log_only_over_a_decade():
    """Verify histogram bars are pixel fractions, using log scale only when spanning > 1 decade."""
    rng = np.random.default_rng(0)
    skewed_res = describe_image(rng.exponential(100, (64, 64)))
    flat_res = describe_image(np.tile(np.arange(64.0), (64, 1)))
    try:
        skewed = skewed_res["axes"]["Histogram"]
        flat = flat_res["axes"]["Histogram"]
        assert skewed.get_yscale() == "log" and flat.get_yscale() == "linear"
        # 64 equally populated values in 64 bins: every bar is exactly 1/64 of the pixels
        bar_y = flat.patches[0].get_path().vertices[:, 1]
        assert bar_y.max() == pytest.approx(1 / 64)
        assert np.all((bar_y == 0) | np.isclose(bar_y, 1 / 64))
    finally:
        plt.close(skewed_res["fig"])
        plt.close(flat_res["fig"])


def test_describe_image_marks_display_range_only_when_it_differs_from_data():
    """Verify vmin/vmax markers, clipped shading, and display-row reporting on describe_image."""
    img = np.tile(np.arange(100.0), (10, 1))  # 0..99, each value 1% of pixels
    auto = describe_image(img)
    res = describe_image(img, vmax=79.5)
    wide = describe_image(img, vmax=200)
    try:
        assert not auto["axes"]["Histogram"].lines and "display" not in auto["info"]

        ax = res["axes"]["Histogram"]
        assert [ln.get_xdata()[0] for ln in ax.lines] == [79.5]
        assert len([p for p in ax.patches if p.get_alpha() is not None and p.get_alpha() < 1]) == 1
        assert "display: vmin 0  vmax 79.5  (20% of pixels clipped)" in res["info"]

        assert wide["axes"]["Histogram"].get_xlim()[1] == pytest.approx(200)
    finally:
        plt.close(auto["fig"])
        plt.close(res["fig"])
        plt.close(wide["fig"])


def test_describe_image_channel_table_swatches_and_shown_row():
    """Verify DataArray channel table swatches, bold shown channel, info_html escaping, and path wrapping."""
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
    try:
        ax = res["axes"]["Info"]
        texts = {t.get_text(): t for t in ax.texts}
        assert {"DAPI", "WGA", "TOMM20"} <= set(texts)
        assert texts["WGA"].get_fontweight() == "bold" and texts["DAPI"].get_fontweight() == "normal"
        swatches = [
            tuple(p.get_facecolor()[:3])
            for p in ax.patches
            if isinstance(p, Rectangle) and p.get_linewidth() > 0
        ]
        for c in ("blue", "green", "red"):
            assert any(np.allclose(sw, to_rgb(c)) for sw in swatches)
        assert "C[1] WGA  color=green" in res["info"] and "shown: S=0, T=0, C=WGA, Z=0" in res["info"]

        html = res["info_html"]
        assert isinstance(html, str) and html._repr_html_() == html
        assert "TOMM20" in html and "background:#008000" in html and "&#9664; shown" in html
    finally:
        plt.close(res["fig"])

    da.attrs["source"] = "/data/<b>x</b>.tif"
    res_esc = describe_image(da)
    try:
        assert "/data/&lt;b&gt;x&lt;/b&gt;.tif" in res_esc["info_html"]
    finally:
        plt.close(res_esc["fig"])

    lines = _wrap_value("/a/" + "dir/" * 20 + "f.tif", 30)
    assert all(len(line) <= 30 for line in lines)
    assert all(line.endswith("/") for line in lines[:-1])
    assert "".join(line.strip() for line in lines) == "/a/" + "dir/" * 20 + "f.tif"


def test_hist_imshow_is_a_deprecated_alias_of_describe_image():
    """Verify hist_imshow emits DeprecationWarning and delegates identically to describe_image."""
    img = np.random.default_rng(0).random((8, 9))
    with pytest.warns(DeprecationWarning, match="describe_image"):
        res = hist_imshow(img, bins=16)
    ref = describe_image(img, bins=16)
    try:
        assert res["info"] == ref["info"] and res["info_html"] == ref["info_html"]
        assert list(res["axes"]) == list(ref["axes"])
        with pytest.warns(DeprecationWarning):
            np.testing.assert_array_equal(hist_imshow(img, return_image_only=True), img)
    finally:
        plt.close(res["fig"])
        plt.close(ref["fig"])


# =========================================
# brightness_diagnostic_plotter
# =========================================


def test_brightness_diagnostic_plotter_object_oriented_and_none():
    """Verify brightness_diagnostic_plotter handles None, creates isolated figures, and respects ax.

    Strategy: call with None, with an unrelated active pyplot figure, and with an explicit ax.
    Expected: None returns None; default call never mutates the active pyplot axes; explicit ax
    receives both raw and fit curves with the expected title and returns its parent figure.
    """
    assert brightness_diagnostic_plotter(None) is None

    diag = {
        "x_data": np.arange(5),
        "y_data_norm": np.linspace(1, 0.5, 5),
        "y_fit_norm": np.linspace(1, 0.5, 5),
        "gamma_fit_func": "exponential",
    }
    current_fig, current_ax = plt.subplots()
    target_fig, target_ax = plt.subplots()
    fig = None
    try:
        fig = brightness_diagnostic_plotter(diag)
        assert fig is not current_fig and len(current_ax.lines) == 0
        assert len(fig.axes[0].lines) == 2

        assert brightness_diagnostic_plotter(diag, ax=target_ax) is target_fig
        assert len(target_ax.lines) == 2 and target_ax.get_title() == "Z-axis Intensity Decay Fit"
    finally:
        plt.close(current_fig)
        plt.close(target_fig)
        if fig is not None:
            plt.close(fig)


# =========================================
# plotly_scatter_3d & plotly_scatter_3d_from_adata_obsm
# =========================================


def test_plotly_scatter_3d_containers_options_and_validation():
    """Verify plotly_scatter_3d with DataFrame/dict/array inputs, colors, labels, camera, and errors.

    Strategy: test numeric and categorical color lists, DataFrame and dict column resolution,
    hover labels, initial_view camera settings, color=None, and missing-axis ValueErrors.
    Expected: numeric color values are preserved, categorical colors map to integer codes,
    labels/camera/titles are configured on the Plotly figure, color=None omits the colorbar,
    and omitting x, y, or z raises ValueError.
    """
    pytest.importorskip("plotly")

    # Numeric vs categorical list colors
    fig_num = plotly_scatter_3d(x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 2], color=[0.1, 0.5, 100.0])
    np.testing.assert_allclose(np.asarray(fig_num.data[0].marker.color, dtype=float), [0.1, 0.5, 100.0])
    assert fig_num.data[0].marker.showscale is True

    fig_cat = plotly_scatter_3d(x=[0, 1, 2], y=[0, 1, 2], z=[0, 1, 2], color=["b", "a", "b"])
    assert list(fig_cat.data[0].marker.color) == [1, 0, 1]

    # DataFrame + labels + initial_view
    df = pd.DataFrame(
        {"cx": [1.0, 2.0], "cy": [3.0, 4.0], "cz": [5.0, 6.0], "score": [0.2, 0.8], "name": ["p1", "p2"]}
    )
    cam = {"eye": {"x": 1.5, "y": 2.0, "z": 2.5}}
    fig_df = plotly_scatter_3d(data=df, x="cx", y="cy", z="cz", color="score", labels="name", initial_view=cam)
    trace = fig_df.data[0]
    np.testing.assert_allclose(trace.x, [1.0, 2.0])
    assert list(trace.text) == ["p1", "p2"]
    assert "<b>Label:</b> %{text}" in trace.hovertemplate
    assert "<b>Color:</b>" in trace.hovertemplate
    assert fig_df.layout.scene.xaxis.title.text == "cx"
    assert fig_df.layout.scene.yaxis.title.text == "cy"
    assert fig_df.layout.scene.zaxis.title.text == "cz"
    assert fig_df.layout.scene.camera.eye.x == pytest.approx(1.5)

    # Dict data + color=None omits colorbar and color hover field
    d = {"a": [0, 1], "b": [1, 2], "c": [2, 3], "lbl": ["u", "v"]}
    fig_dict = plotly_scatter_3d(data=d, x="a", y="b", z="c", labels="lbl", color=None)
    assert not fig_dict.data[0].marker.showscale
    assert "<b>Color:</b>" not in fig_dict.data[0].hovertemplate
    assert list(fig_dict.data[0].text) == ["u", "v"]

    # Missing coordinate raises ValueError
    with pytest.raises(ValueError, match="x, y, and z coordinates must be provided"):
        plotly_scatter_3d(x=[0, 1], y=[0, 1])
    with pytest.raises(ValueError, match="x, y, and z coordinates must be provided"):
        plotly_scatter_3d(x=[0, 1], z=[0, 1])
    with pytest.raises(ValueError, match="x, y, and z coordinates must be provided"):
        plotly_scatter_3d(y=[0, 1], z=[0, 1])


def test_plotly_scatter_3d_from_adata_obsm(capsys):
    """Verify plotly_scatter_3d_from_adata_obsm handles 3D, 2D zero-pad, obs keys, and errors.

    Strategy: construct a lightweight SimpleNamespace mimicking AnnData (.obsm and .obs),
    then exercise 3D coordinates with valid color_key/label_key, 2D coordinates (warning +
    zero Z padding), missing obs keys (warning), and missing obsm_key (ValueError).
    Expected: axis titles reflect obsm_key dimensions (or 'Zero' when padded), obs color and
    labels propagate when present, and missing keys warn or raise as documented.
    """
    pytest.importorskip("plotly")

    coords_3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    coords_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    obs = pd.DataFrame({"cluster": [0.1, 0.5, 0.9], "cell": ["c1", "c2", "c3"]})
    adata = SimpleNamespace(obsm={"X_umap": coords_3d, "X_pca2d": coords_2d}, obs=obs)

    # 3D obsm with valid color_key and label_key
    fig3 = plotly_scatter_3d_from_adata_obsm(adata, "X_umap", color_key="cluster", label_key="cell")
    np.testing.assert_allclose(fig3.data[0].x, coords_3d[:, 0])
    np.testing.assert_allclose(fig3.data[0].y, coords_3d[:, 1])
    np.testing.assert_allclose(fig3.data[0].z, coords_3d[:, 2])
    np.testing.assert_allclose(np.asarray(fig3.data[0].marker.color, dtype=float), [0.1, 0.5, 0.9])
    assert list(fig3.data[0].text) == ["c1", "c2", "c3"]
    assert fig3.layout.scene.xaxis.title.text == "X_umap_0"
    assert fig3.layout.scene.yaxis.title.text == "X_umap_1"
    assert fig3.layout.scene.zaxis.title.text == "X_umap_2"

    # 2D obsm pads Z with zeros and prints a warning
    fig2 = plotly_scatter_3d_from_adata_obsm(adata, "X_pca2d")
    np.testing.assert_array_equal(fig2.data[0].z, np.zeros(3))
    assert fig2.layout.scene.zaxis.title.text == "Zero"
    assert "Padding with zeros for Z-axis" in capsys.readouterr().out

    # Missing color_key and label_key warn gracefully without crashing
    fig_missing_obs = plotly_scatter_3d_from_adata_obsm(
        adata, "X_umap", color_key="no_color", label_key="no_label"
    )
    out = capsys.readouterr().out
    assert "'no_color' not found in adata.obs" in out
    assert "'no_label' not found in adata.obs" in out
    assert not fig_missing_obs.data[0].marker.showscale
    assert fig_missing_obs.data[0].text is None

    # Missing obsm_key raises ValueError
    with pytest.raises(ValueError, match="'X_missing' not found in adata.obsm"):
        plotly_scatter_3d_from_adata_obsm(adata, "X_missing")


# =========================================
# get_nice_number
# =========================================


@pytest.mark.parametrize(
    "value, expected, expected_type",
    [
        (0, 0, int),
        (1, 1, int),
        (10, 10, int),
        (100, 100, int),
        (999, 900, int),
        (3000, 3000, int),
        (4343, 4000, int),
        (10858, 10000, int),
        (21717, 20000, int),
        (-37, -30, int),
        (0.1, 0.1, float),
        (0.01, 0.01, float),
        (0.6, 0.6, float),
        (0.0042, 0.004, float),
        (-0.075, -0.07, float),
    ],
)
def test_get_nice_number_rounds_down_with_expected_type(value, expected, expected_type):
    """Verify get_nice_number floors to one significant digit with int (>=1 or 0) vs float (<1) types.

    Strategy: test zero, powers of 10, positive/negative integers, and positive/negative fractions.
    Expected: result equals the single-significant-digit floor, never exceeds |value|, and has
    exact return type int for 0 or |value| >= 1 and float for 0 < |value| < 1.
    """
    result = get_nice_number(value)
    assert type(result) is expected_type
    assert result == pytest.approx(expected)
    assert abs(result) <= abs(value)
