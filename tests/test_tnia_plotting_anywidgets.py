"""Tests for tnia_plotting_anywidgets (static ortho plots, interactive widgets, and IsoScatterWidget)."""

import ast
import base64
import importlib
import inspect
import io
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pytest
import traitlets
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

import eigenp_utils.tnia_plotting_anywidgets as tpa
from eigenp_utils.plotting_utils import labels_cmap
from eigenp_utils.tnia_plotting_anywidgets import (
    IsoScatterWidget,
    TNIASliceWidget,
    _attenuated_mip,
    _channel_colormaps,
    _get_rotated_line,
    _parse_zyx_tuple_or_dict,
    _rotate_expand,
    black_to,
    blend_colors,
    compute_histogram,
    create_multichannel_rgb,
    create_multichannel_rgb_cmap,
    get_gui_channel_color,
    infer_bit_depth,
    infer_histogram_range,
    resolve_color,
    show_iso_scatter,
    show_zyx,
    show_zyx_max,
    show_zyx_max_scatter_interactive,
    show_zyx_max_slabs,
    show_zyx_max_slice_interactive,
    show_zyx_max_slice_interactive_point_annotator,
    show_zyx_projection,
    show_zyx_slice,
    white_to,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _structured_volume(shape=(16, 48, 48)):
    """A small volume with an off-centre cube, so moving a slab changes the image."""
    vol = np.zeros(shape, dtype=np.uint8)
    z, y, x = shape
    vol[z // 4:z // 2, y // 4:y // 2, x // 4:x // 2] = 255
    return vol


def _count_renders(widget):
    """Swap widget._render for a counting wrapper; returns the mutable counter."""
    counter = {"n": 0}
    inner = widget._render

    def counting_render():
        counter["n"] += 1
        return inner()

    widget._render = counting_render
    return counter


def _fig_frac(fig, ax, x, y):
    """Data point on ``ax`` -> widget event coordinates (figure fraction, top-left origin)."""
    disp = ax.transData.transform((x, y))
    fx, fy = fig.transFigure.inverted().transform(disp)
    return float(fx), float(1.0 - fy)


def _rendered_centroid(ax):
    """Intensity-weighted centroid of the image on ``ax``, in its data (physical) units."""
    im = ax.images[0]
    arr = np.asarray(im.get_array(), dtype=float)
    if arr.ndim == 3:
        arr = arr[..., :3].sum(axis=-1)
    arr = np.nan_to_num(arr)
    x0, x1, y0, y1 = im.get_extent()  # (left, right, bottom, top) with top=0
    h, w = arr.shape
    xs = x0 + (np.arange(w) + 0.5) * (x1 - x0) / w
    ys = y1 + (np.arange(h) + 0.5) * (y0 - y1) / h
    tot = arr.sum()
    return float((arr.sum(axis=0) * xs).sum() / tot), float((arr.sum(axis=1) * ys).sum() / tot)


def _scale_bar_artists(fig):
    """(bar LineCollection or None, label Text or None) drawn on the XY panel."""
    bars = [c for c in fig.axXY.collections if isinstance(c, mcollections.LineCollection)]
    labels = [t for t in fig.axXY.texts
              if t.get_text().split(" ")[-1] in ("nm", "µm", "mm", "px") or "pixel_sizes" in t.get_text()]
    return (bars[-1] if bars else None), (labels[0] if labels else None)


def _panel_rgb(fig, name="axXY"):
    """Return the rendered array of a named panel axis as float."""
    return np.asarray(getattr(fig, name).images[0].get_array(), dtype=float)


def _two_channel_planes():
    """Channel 0 active in the left column, channel 1 active in the top row."""
    a = np.zeros((2, 2), dtype=np.float32)
    a[:, 0] = 1.0
    b = np.zeros((2, 2), dtype=np.float32)
    b[0, :] = 1.0
    return a, b


def _iso_points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(n) * 50, rng.random(n) * 40, rng.random(n) * 10


def _first_run(mask):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    brk = np.flatnonzero(np.diff(idx) > 1)
    return int(idx[0]), int(idx[brk[0]] if brk.size else idx[-1])


def _label_fontsize(fig):
    """Extract the rendered font size (pt) from the first channel label TextArea."""
    from matplotlib.offsetbox import AnchoredOffsetbox

    boxes = [a for a in fig.axLabels.artists if isinstance(a, AnchoredOffsetbox)]
    assert boxes, "expected an AnchoredOffsetbox on fig.axLabels"
    children = boxes[0].get_child().get_children()
    assert children, "expected at least one TextArea child"
    return children[0]._text.get_fontsize()


# ---------------------------------------------------------------------------
# Basic instantiation, channels, visibility, and 2D input
# ---------------------------------------------------------------------------

def test_single_channel_instantiation():
    """Verify single-channel slice viewer initializes defaults and renders an initial PNG.

    Strategy: pass a 3D zero volume to ``show_zyx_max_slice_interactive``.
    Expected outcome: returns a ``TNIASliceWidget`` with ``num_channels == 1``,
    one channel name, and non-empty ``image_data``.
    """
    im = np.zeros((10, 20, 30))
    w = show_zyx_max_slice_interactive(im)
    assert isinstance(w, TNIASliceWidget)
    assert w.num_channels == 1
    assert len(w.channel_names) == 1
    assert w.image_data is not None and len(w.image_data) > 0


def test_multi_channel_instantiation():
    """Verify multi-channel slice viewer configures channel names, opacities, and initial PNG.

    Strategy: pass a list of three 3D arrays to ``show_zyx_max_slice_interactive``.
    Expected outcome: ``num_channels == 3``, default names ``Channel 0..2``,
    full opacities ``[1.0, 1.0, 1.0]``, and non-empty ``image_data``.
    """
    im = [np.zeros((10, 20, 30)) for _ in range(3)]
    w = show_zyx_max_slice_interactive(im)
    assert w.num_channels == 3
    assert w.channel_names == ["Channel 0", "Channel 1", "Channel 2"]
    assert w.opacity_list == [1.0, 1.0, 1.0]
    assert w.image_data is not None and len(w.image_data) > 0


def test_channel_visibility_update():
    """Verify setting channel opacity to 0 hides channels and triggers distinct re-renders.

    Strategy: construct a two-channel viewer with one dark and one bright
    channel, hide channel 1, then hide both channels.
    Expected outcome: ``image_data`` changes at each visibility transition.
    """
    im = [np.zeros((10, 10, 10)), np.ones((10, 10, 10)) * 255]
    w = show_zyx_max_slice_interactive(im)

    initial_data = w.image_data
    assert initial_data

    w.opacity_list = [1.0, 0.0]
    new_data = w.image_data
    assert new_data != initial_data

    w.opacity_list = [0.0, 0.0]
    empty_data = w.image_data
    assert empty_data != new_data
    assert empty_data != initial_data


def test_2d_interactive_input_promoted_to_single_z_slice():
    """Verify 2D ``(Y, X)`` arrays work in interactive slice and annotator viewers.

    Strategy: pass a single 2D array, a list of two 2D arrays, and a 2D array
    for the point annotator to their public factory functions.
    Expected outcome: each widget promotes input to ``dims == (1, Y, X)`` with
    ``z_s == z_start == z_end == 0`` and produces a rendered ``image_data`` PNG.
    """
    rng = np.random.default_rng(0)
    im2d = rng.random((20, 30)).astype(np.float32)

    w1 = show_zyx_max_slice_interactive(im2d)
    assert w1.dims == (1, 20, 30)
    assert (w1.z_s, w1.z_start, w1.z_end) == (0, 0, 0)
    assert w1.num_channels == 1 and w1.image_data

    w2 = show_zyx_max_slice_interactive([im2d, im2d * 0.5])
    assert w2.dims == (1, 20, 30)
    assert w2.num_channels == 2 and w2.image_data

    wa = show_zyx_max_slice_interactive_point_annotator(im2d)
    assert wa.dims == (1, 20, 30)
    assert wa.num_channels == 2 and wa.channel_names[-1] == "Annotations"
    assert wa.image_data


def test_interactive_channel_labels():
    """Verify explicit ``channel_labels`` propagate to slice, annotator, and scatter viewers.

    Strategy: pass ``channel_labels=['DAPI', 'GFP']`` to all three interactive
    factory functions.
    Expected outcome: slice and scatter preserve the exact labels; the annotator
    appends ``'Annotations'`` to ``channel_names``.
    """
    rng = np.random.default_rng(1)
    im = np.zeros((10, 20, 30))
    labels = ["DAPI", "GFP"]

    w1 = show_zyx_max_slice_interactive([im, im], channel_labels=labels)
    assert w1.channel_labels_input == labels
    assert w1.channel_names == labels

    w2 = show_zyx_max_slice_interactive_point_annotator([im, im], channel_labels=labels)
    assert w2.channel_labels_input == labels
    assert w2.channel_names == labels + ["Annotations"]

    X = rng.random(10) * 10
    Y = rng.random(10) * 10
    Z = rng.random(10) * 10
    channels_multi = [rng.random(10), rng.random(10)]

    w3 = show_zyx_max_scatter_interactive(
        (Z, Y, X),
        channels=channels_multi,
        channel_labels=labels,
        render="points",
    )
    assert w3.channel_labels_input == labels
    assert w3.channel_names == labels


# ---------------------------------------------------------------------------
# Color / colormap resolution, GUI darkening, and padding
# ---------------------------------------------------------------------------

def test_resolve_color():
    """Verify ``resolve_color`` handles CSS colors, colormaps (including CSS-named ones), and Colormap instances.

    Strategy: resolve hex strings, plain CSS names, standard colormaps
    (``'viridis'``), and colormaps sharing CSS names (``'gray'``, ``'pink'``,
    ``'hot'``), plus registered and unregistered ``Colormap`` instances.
    Expected outcome: CSS-named colormaps evaluate ``cmap(1.0)`` (not the CSS
    color; 'pink' warns); plain colors pass through; registered instances
    (``labels_cmap``) resolve like their name; unregistered ones raise
    ``TypeError``. (Changed by D03: registered instances used to raise too.)
    """
    assert resolve_color("#ff0000") == "#ff0000"
    assert resolve_color("red") == "red"
    assert resolve_color("viridis") == "#fde725"

    for name in ("gray", "pink", "hot"):
        expected = mcolors.to_hex(matplotlib.colormaps[name](1.0)[:3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # 'pink' warns
            assert resolve_color(name) == expected
    # 'gray' and 'pink' are also CSS color names; colormap lookup must take precedence.
    assert resolve_color("gray") != mcolors.to_hex(to_rgb("gray"))
    with pytest.warns(UserWarning, match="'pink' is both a colormap"):
        assert resolve_color("pink") != mcolors.to_hex(to_rgb("pink"))

    cmap = mcolors.LinearSegmentedColormap.from_list("test", ["black", "white"])
    with pytest.raises(TypeError, match="Expected a registered colormap name"):
        resolve_color(cmap)
    assert resolve_color(labels_cmap) == resolve_color("labels_cmap")


def test_gui_channel_color_and_linear_colormaps():
    """Verify ``get_gui_channel_color`` darkens to 75% and ``black_to``/``white_to`` span exact endpoints.

    Strategy: test ``get_gui_channel_color`` on named color, hex, and colormap
    inputs, and evaluate ``black_to('red')`` / ``white_to('red')`` at 0.0, 0.5, 1.0.
    Expected outcome: GUI colors equal ``0.75 * rgb`` in hex; ``black_to`` ramps
    black -> red and ``white_to`` ramps white -> red.
    """
    assert get_gui_channel_color("white") == mcolors.to_hex((0.75, 0.75, 0.75))
    assert get_gui_channel_color("#ff0000") == mcolors.to_hex((0.75, 0.0, 0.0))
    vir_end = mcolors.to_rgb(matplotlib.colormaps["viridis"](1.0)[:3])
    assert get_gui_channel_color("viridis") == mcolors.to_hex(tuple(0.75 * v for v in vir_end))

    b_red = black_to("red")
    np.testing.assert_allclose(b_red(0.0)[:3], (0.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(b_red(0.5)[:3], (0.5, 0.0, 0.0), atol=1e-2)
    np.testing.assert_allclose(b_red(1.0)[:3], (1.0, 0.0, 0.0), atol=1e-6)

    w_red = white_to("red")
    np.testing.assert_allclose(w_red(0.0)[:3], (1.0, 1.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(w_red(0.5)[:3], (1.0, 0.5, 0.5), atol=1e-2)
    np.testing.assert_allclose(w_red(1.0)[:3], (1.0, 0.0, 0.0), atol=1e-6)


def test_widget_rejects_colormap_instance():
    """Verify widgets accept registered ``Colormap`` instances and reject unregistered ones.

    Strategy: pass ``labels_cmap`` (registered) and an unregistered
    ``LinearSegmentedColormap`` as ``colormap=`` to
    ``show_zyx_max_slice_interactive``, single channel and in a list, and the
    exported I-series instances to ``show_zyx_max``.
    Expected outcome: registered instances work and are stored by name (the
    colour trait is a string); an unregistered one raises ``TypeError``
    asking for registration. (Changed by D03: registered instances used to
    be rejected too.)
    """
    from eigenp_utils import i_blue, i_rose

    im = np.zeros((10, 20, 30))
    w = show_zyx_max_slice_interactive(im, colormap=labels_cmap)
    assert w.colors_resolved == ["labels_cmap"] and w.image_data
    w2 = show_zyx_max_slice_interactive([im, im], colormap=[labels_cmap, "lime"])
    assert w2.colors_resolved == ["labels_cmap", "lime"]
    plt.close(show_zyx_max([im, im], colormap=[i_blue, i_rose]))

    unregistered = mcolors.LinearSegmentedColormap.from_list("not_registered_xyz", ["black", "red"])
    with pytest.raises(TypeError, match="Expected a registered colormap name"):
        show_zyx_max_slice_interactive(im, colormap=unregistered)


@pytest.mark.parametrize(
    "n_channels, colormap, expected_prefix",
    [
        (2, None, ["white", "lime"]),
        (2, ["red", "blue"], ["red", "blue"]),
        (3, "magma", ["magma"]),
        (3, ["red", "lime"], ["red", "lime"]),
    ],
)
def test_slice_widget_colormap_resolution_and_padding(n_channels, colormap, expected_prefix):
    """Verify multichannel colormap defaults, explicit lists, single strings, and padding.

    Strategy: construct ``show_zyx_max_slice_interactive`` with ``None``, an
    exact list, a single colormap string (must not be split into characters),
    and a short list that must be padded with unused defaults.
    Expected outcome: ``colors_resolved`` and ``channel_colors`` have length
    ``n_channels``, preserve the given prefix, and contain no duplicates in padding.
    """
    rng = np.random.default_rng(9)
    chans = [rng.uniform(0, 1, (4, 16, 16)).astype(np.float32) for _ in range(n_channels)]
    w = show_zyx_max_slice_interactive(chans, colormap=colormap)
    assert len(w.colors_resolved) == n_channels
    assert len(w.channel_colors) == n_channels
    assert w.colors_resolved[:len(expected_prefix)] == expected_prefix
    assert len(set(w.colors_resolved)) == n_channels


# ---------------------------------------------------------------------------
# Deprecation warnings and parameter parsing / validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "caller",
    [
        "show_zyx",
        "show_zyx_max",
        "show_zyx_max_slabs",
        "show_zyx_max_slice_interactive",
        "show_zyx_max_slice_interactive_point_annotator",
        "show_zyx_max_scatter_interactive",
    ],
)
def test_deprecated_colors_warns_once_at_caller(caller):
    """Verify passing deprecated ``colors=`` emits exactly one ``FutureWarning`` pointing to the caller.

    Strategy: invoke each static and interactive entry point with ``colors='red'``
    while recording warnings.
    Expected outcome: exactly one ``FutureWarning`` mentioning ``'colors'`` and
    the removal version 0.1.0 is emitted, with ``filename == __file__``.
    (Changed by D08: it was a DeprecationWarning, hidden by default.)
    """
    rng = np.random.default_rng(0)
    im = rng.random((4, 8, 8)).astype(np.float32)
    pts = (rng.random(12) * 4, rng.random(12) * 8, rng.random(12) * 8)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        if caller == "show_zyx":
            fig = show_zyx(im[0], im[:, 0, :], im[:, :, 0], colors="red")
            plt.close(fig)
        elif caller == "show_zyx_max":
            fig = show_zyx_max(im, colors="red")
            plt.close(fig)
        elif caller == "show_zyx_max_slabs":
            fig = show_zyx_max_slabs(im, colors="red")
            plt.close(fig)
        elif caller == "show_zyx_max_slice_interactive":
            show_zyx_max_slice_interactive(im, colors="red")
        elif caller == "show_zyx_max_slice_interactive_point_annotator":
            show_zyx_max_slice_interactive_point_annotator(im, colors="red")
        else:
            show_zyx_max_scatter_interactive(pts, colors="red", render="points")

    dep = [r for r in rec if issubclass(r.category, FutureWarning) and "'colors'" in str(r.message)]
    assert len(dep) == 1, dep
    assert dep[0].filename == __file__
    assert "eigenp_utils 0.1.0" in str(dep[0].message)


@pytest.mark.parametrize(
    "kind",
    ["slice", "annotator", "scatter"],
)
def test_deprecation_warnings_interactive_legacy_kwargs(kind):
    """Verify legacy ``sxy``/``sz``, ``x_s``/``y_s``/``z_s``, and ``x_t``/``y_t``/``z_t`` kwargs emit ``FutureWarning``.

    Strategy: call each interactive factory function with each deprecated
    parameter group.
    Expected outcome: each call emits a ``FutureWarning`` pointing to the
    replacement parameter (``pixel_sizes``, ``slabs_position``, ``slabs_thickness``).
    (Changed by D08: it was a DeprecationWarning, hidden by default.)
    """
    rng = np.random.default_rng(2)
    im = np.zeros((10, 20, 30), dtype=np.float32)
    pts = (rng.random(10) * 10, rng.random(10) * 20, rng.random(10) * 30)

    def _call(**kw):
        if kind == "slice":
            return show_zyx_max_slice_interactive(im, **kw)
        if kind == "annotator":
            return show_zyx_max_slice_interactive_point_annotator(im, **kw)
        return show_zyx_max_scatter_interactive(pts, render="points", **kw)

    with pytest.warns(FutureWarning, match=r"'sxy'/'sz' is deprecated .*0\.1\.0.*'pixel_sizes'"):
        _call(sxy=0.5, sz=2.0)
    with pytest.warns(FutureWarning, match=r"'x_s'/'y_s'/'z_s' is deprecated .*'slabs_position'"):
        _call(x_s=5, y_s=10, z_s=5)
    with pytest.warns(FutureWarning, match=r"'x_t'/'y_t'/'z_t' is deprecated .*'slabs_thickness'"):
        _call(x_t=2, y_t=3, z_t=4)


def test_parse_zyx_tuple_or_dict_various_types():
    """Verify ``_parse_zyx_tuple_or_dict`` normalizes tuples, lists, dicts, and 0-d numpy scalars to floats.

    Strategy: pass native tuples/lists, dicts, 0-d numpy arrays, and extreme
    magnitudes.
    Expected outcome: all return plain 3-tuples of Python ``float`` values.
    """
    assert _parse_zyx_tuple_or_dict((1.5, 2.5, 3.5)) == (1.5, 2.5, 3.5)
    assert _parse_zyx_tuple_or_dict([1.5, 2.5, 3.5]) == (1.5, 2.5, 3.5)
    assert _parse_zyx_tuple_or_dict({"Z": 1.5, "Y": 2.5, "X": 3.5}) == (1.5, 2.5, 3.5)

    res = _parse_zyx_tuple_or_dict((np.array(1.5), np.array(2.5), np.array(3.5)))
    assert res == (1.5, 2.5, 3.5) and all(isinstance(v, float) for v in res)

    res_dict = _parse_zyx_tuple_or_dict({"Z": np.array(1.5), "Y": np.array(2.5), "X": np.array(3.5)})
    assert res_dict == (1.5, 2.5, 3.5) and all(isinstance(v, float) for v in res_dict)

    assert _parse_zyx_tuple_or_dict([1e-6, 1e6, 0.0]) == (1e-6, 1e6, 0.0)


def test_invalid_pixel_sizes_and_rotate_view_raise():
    """Verify invalid ``pixel_sizes`` and ``rotate_view`` inputs raise ``ValueError``.

    Strategy: pass wrong-length sequences, scalars, and non-numeric values to
    ``pixel_sizes`` and ``rotate_view`` via static and interactive entry points.
    Expected outcome: every invalid argument raises ``ValueError``.
    """
    im = np.zeros((4, 8, 8), dtype=np.float32)

    for bad_ps in ((1.0, 2.0), (1.0, 2.0, 3.0, 4.0), 2.5, ("bad", 1.0, 1.0)):
        with pytest.raises(ValueError):
            _parse_zyx_tuple_or_dict(bad_ps, default_val=1.0)
        with pytest.raises(ValueError):
            show_zyx_max(im, pixel_sizes=bad_ps)

    for bad_rot in ((10.0, 20.0), (1.0, 2.0, 3.0, 4.0), "90deg"):
        with pytest.raises(ValueError, match="rotate_view"):
            show_zyx(im[0], im[:, 0, :], im[:, :, 0], rotate_view=bad_rot)
        with pytest.raises(ValueError, match="rotate_view"):
            show_zyx_max_slice_interactive(im, rotate_view=bad_rot)


def test_pixel_size_dict_with_missing_axis_falls_back_to_one():
    """A missing Z with known X/Y takes the finest X/Y size (D01), with a single warning.

    Strategy: pass ``{'X': 0.5, 'Y': 0.5, 'Z': None, 'unit': 'micron'}`` to
    ``_parse_zyx_tuple_or_dict`` (the slab parser), ``show_zyx_max``, and both
    interactive wrappers (recording warnings on the slice wrapper to ensure it
    warns once).
    Expected outcome: the generic slab parser still fills ``default_val``;
    the viewers use Z = 0.5 (the finest X/Y size, isotropic voxels assumed)
    with a ``UserWarning`` naming Z; the interactive slice wrapper emits
    exactly one warning for the missing axis. (Changed by D01: Z used to
    become 1.0, i.e. one pixel next to µm-sized X/Y.)
    """
    ps = {"X": 0.5, "Y": 0.5, "Z": None, "unit": "micron"}
    with pytest.warns(UserWarning, match="axis Z"):
        assert _parse_zyx_tuple_or_dict(ps, default_val=1.0) == (1.0, 0.5, 0.5)
    # Slab defaults stay None without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _parse_zyx_tuple_or_dict({"Z": None, "Y": 2, "X": None}) == (None, 2.0, None)

    im = np.random.default_rng(0).random((6, 20, 30))
    with pytest.warns(UserWarning, match="axis Z"):
        fig = show_zyx_max(im, pixel_sizes=ps)
    assert abs(np.diff(fig.axXZ.get_ylim())[0]) == pytest.approx(6 * 0.5)
    plt.close(fig)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w = show_zyx_max_slice_interactive(im, pixel_sizes=ps)
    msgs = [str(r.message) for r in rec if "No value for axis Z" in str(r.message)]
    assert len(msgs) == 1
    assert w.sz == 0.5 and w.image_data

    pts = (np.array([0, 5.0]), np.array([0, 19.0]), np.array([0, 29.0]))
    with pytest.warns(UserWarning, match="axis Z"):
        w2 = show_zyx_max_scatter_interactive(pts, pixel_sizes=ps)
    assert w2.sz == 0.5 and w2.image_data


def test_deprecated_tnia_plotting_3d_warning():
    """Verify importing ``tnia_plotting_3d`` emits a ``FutureWarning`` and re-exports every name in ``__all__``.

    Strategy: evict ``eigenp_utils.tnia_plotting_3d`` from ``sys.modules``,
    re-import under ``warnings.catch_warnings``, and compare every symbol in
    ``__all__`` against ``eigenp_utils.tnia_plotting_anywidgets``.
    Expected outcome: emits one ``FutureWarning`` naming the removal version
    0.1.0, and every name in ``__all__`` resolves to the identical object on
    ``tnia_plotting_anywidgets``. (Changed by D08: it was a DeprecationWarning
    without a removal version.)
    """
    if "eigenp_utils.tnia_plotting_3d" in sys.modules:
        del sys.modules["eigenp_utils.tnia_plotting_3d"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import eigenp_utils.tnia_plotting_3d as tnia3d

        assert len(w) == 1
        assert issubclass(w[-1].category, FutureWarning)
        assert "deprecated" in str(w[-1].message) and "eigenp_utils 0.1.0" in str(w[-1].message)

    assert len(tnia3d.__all__) > 0
    for name in tnia3d.__all__:
        assert hasattr(tnia3d, name), f"tnia_plotting_3d missing {name!r}"
        assert hasattr(tpa, name), f"tnia_plotting_anywidgets missing {name!r}"
        assert getattr(tnia3d, name) is getattr(tpa, name)


# ---------------------------------------------------------------------------
# Static ortho views, projections, and multichannel blending
# ---------------------------------------------------------------------------

def test_show_zyx_slice_returns_correct_slices():
    """Verify ``show_zyx_slice`` extracts the exact XY, ZY, and XZ slices at distinct indices.

    Strategy: construct a deterministic ``(4, 3, 2)`` array with unique values
    and slice at ``x=0, y=2, z=3`` with ``use_plt=False``.
    Expected outcome: returns a Matplotlib ``Figure`` whose ``axXY``, ``axZY``,
    and ``axXZ`` images match the exact expected 2D slices.
    """
    arr = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    fig = show_zyx_slice(arr, x=0, y=2, z=3, use_plt=False)
    assert isinstance(fig, Figure)
    xy_expected = arr[3, :, :]
    xz_expected = arr[:, 2, :]
    zy_expected = np.flip(np.rot90(arr[:, :, 0], 1), 0)
    assert np.array_equal(fig.axXY.images[0].get_array(), xy_expected)
    assert np.array_equal(fig.axZY.images[0].get_array(), zy_expected)
    assert np.array_equal(fig.axXZ.images[0].get_array(), xz_expected)


def test_show_zyx_max_slabs_projection():
    """Verify ``show_zyx_max_slabs`` computes exact max projections over half-open slab intervals.

    Strategy: project a deterministic ``(4, 3, 2)`` array over ``x=[0, 1]``,
    ``y=[0, 2]``, ``z=[1, 4]``.
    Expected outcome: all three panel images match ``np.max`` over the
    corresponding slices.
    """
    arr = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    fig = show_zyx_max_slabs(arr, x=[0, 1], y=[0, 2], z=[1, 4])
    try:
        xy_expected = np.max(arr[1:4, :, :], axis=0)
        xz_expected = np.max(arr[:, 0:2, :], axis=1)
        zy_expected = np.flip(np.rot90(np.max(arr[:, :, 0:1], axis=2), 1), 0)
        assert np.array_equal(fig.axXY.images[0].get_array(), xy_expected)
        assert np.array_equal(fig.axXZ.images[0].get_array(), xz_expected)
        assert np.array_equal(fig.axZY.images[0].get_array(), zy_expected)
    finally:
        plt.close(fig)


def test_show_zyx_projection_custom_projector_median():
    """Verify ``show_zyx_projection`` applies a custom callable ``projector`` such as ``np.median``.

    Strategy: build a ``(5, 4, 3)`` volume with a single high outlier so median
    differs sharply from both max and mean, then call ``show_zyx_projection``
    with ``projector=np.median``.
    Expected outcome: all three ortho panels match ``np.median`` along their
    projection axes and differ from ``np.max``.
    """
    rng = np.random.default_rng(42)
    arr = rng.uniform(0.0, 2.0, (5, 4, 3)).astype(np.float32)
    arr[2, 1, 1] = 100.0

    fig = show_zyx_projection(arr, projector=np.median)
    try:
        xy_exp = np.median(arr, axis=0)
        xz_exp = np.median(arr, axis=1)
        zy_exp = np.flip(np.rot90(np.median(arr, axis=2), 1), 0)
        np.testing.assert_allclose(fig.axXY.images[0].get_array(), xy_exp, rtol=1e-6)
        np.testing.assert_allclose(fig.axXZ.images[0].get_array(), xz_exp, rtol=1e-6)
        np.testing.assert_allclose(fig.axZY.images[0].get_array(), zy_exp, rtol=1e-6)
        assert fig.axXY.images[0].get_array()[1, 1] < 10.0  # outlier rejected unlike MIP
    finally:
        plt.close(fig)


def test_show_zyx_max_slabs_zero_sized_slices():
    """Verify zero-width float slab intervals coerce to width-1 integer slices and 0-d numpy pixel sizes work.

    Strategy: pass identical float bounds ``x=[0.5, 0.5]``, ``y=[0.0, 0.0]``,
    ``z=[0, 0]`` to ``show_zyx_max_slabs``, and pass 0-d numpy scalar
    ``pixel_sizes`` to ``show_zyx_max_slice_interactive``.
    Expected outcome: static plot renders without raising ``ValueError``, and
    the interactive widget stores native Python float pixel sizes.
    """
    im = np.random.default_rng(3).random((10, 10, 10))
    fig = show_zyx_max_slabs(im, x=[0.5, 0.5], y=[0.0, 0.0], z=[0, 0])
    try:
        assert fig is not None
    finally:
        plt.close(fig)

    w = show_zyx_max_slice_interactive(im, pixel_sizes=(np.array(1.5), np.array(2.5), np.array(3.5)))
    assert (w.sz, w.sy, w.sx) == (1.5, 2.5, 3.5)
    assert isinstance(w.sz, float)


def test_create_multichannel_rgb_basic():
    """Verify ``create_multichannel_rgb`` maps independent channels onto their RGB base colors.

    Strategy: supply two channels active exclusively in XY and XZ respectively
    with ``colormap=['red', 'green']``.
    Expected outcome: XY is solid red, XZ is solid green, and ZY is black.
    """
    xy_list = [np.ones((2, 2)), np.zeros((2, 2))]
    xz_list = [np.zeros((2, 2)), np.ones((2, 2))]
    zy_list = [np.zeros((2, 2)), np.zeros((2, 2))]
    xy_rgb, xz_rgb, zy_rgb = create_multichannel_rgb(
        xy_list, xz_list, zy_list, colormap=["red", "green"]
    )
    red = np.asarray(to_rgb("red"))
    green = np.asarray(to_rgb("green"))
    assert np.allclose(xy_rgb, np.broadcast_to(red, (2, 2, 3)))
    assert np.allclose(xz_rgb, np.broadcast_to(green, (2, 2, 3)))
    assert np.allclose(zy_rgb, np.zeros((2, 2, 3)))


def test_screen_blending_and_soft_clip_control():
    """Verify ``'screen'`` blending formula and ``soft_clip=False`` hard-clipping in RGB compositors.

    Strategy: (a) compose two ``0.5``-intensity red channels with ``'screen'``
    blending in both ``create_multichannel_rgb`` and ``blend_colors`` and verify
    ``1 - (1 - 0.5) * (1 - 0.5) == 0.75``; (b) compose full-intensity ``'red'``
    ``(1, 0, 0)`` and ``'yellow'`` ``(1, 1, 0)`` (raw sum ``(2, 1, 0)``) under
    ``soft_clip=True`` vs ``soft_clip=False``.
    Expected outcome: screen gives ``(0.75, 0, 0)``; ``soft_clip=True`` preserves
    hue ratio at ``(1.0, 0.5, 0.0)`` while ``soft_clip=False`` hard-clips to
    ``(1.0, 1.0, 0.0)``.
    """
    half = np.full((2, 2), 0.5, dtype=np.float32)
    xy_scr, _, _ = create_multichannel_rgb(
        [half, half], [half, half], [half, half],
        vmin=0.0, vmax=1.0, colormap=["red", "red"], blend="screen",
    )
    np.testing.assert_allclose(xy_scr[0, 0], [0.75, 0.0, 0.0], atol=1e-6)

    pts_scr = blend_colors(
        np.array([[0.5, 0.5]]), ["red", "red"], vmin=0.0, vmax=1.0, blending="screen"
    )
    np.testing.assert_allclose(pts_scr[0], [0.75, 0.0, 0.0], atol=1e-6)

    ones = np.ones((2, 2), dtype=np.float32)
    xy_soft, _, _ = create_multichannel_rgb(
        [ones, ones], [ones, ones], [ones, ones],
        vmin=0.0, vmax=1.0, colormap=["red", "yellow"], blend="additive", soft_clip=True,
    )
    xy_hard, _, _ = create_multichannel_rgb(
        [ones, ones], [ones, ones], [ones, ones],
        vmin=0.0, vmax=1.0, colormap=["red", "yellow"], blend="additive", soft_clip=False,
    )
    np.testing.assert_allclose(xy_soft[0, 0], [1.0, 0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(xy_hard[0, 0], [1.0, 1.0, 0.0], atol=1e-6)

    pts_soft = blend_colors(
        np.array([[1.0, 1.0]]), ["red", "yellow"], vmin=0.0, vmax=1.0, soft_clip=True
    )
    pts_hard = blend_colors(
        np.array([[1.0, 1.0]]), ["red", "yellow"], vmin=0.0, vmax=1.0, soft_clip=False
    )
    np.testing.assert_allclose(pts_soft[0], [1.0, 0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(pts_hard[0], [1.0, 1.0, 0.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Interactive kwargs, slab sync, scatter modes, and annotator API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory_fn", [
    show_zyx_max_slice_interactive,
    show_zyx_max_slice_interactive_point_annotator,
])
def test_interactive_kwargs_images(factory_fn):
    """Verify physical ``slabs_position``, ``slabs_thickness``, and ``pixel_sizes`` convert to clamped voxel indices.

    Strategy: pass anisotropic ``pixel_sizes={'Z': 2.0, 'Y': 1.0, 'X': 0.5}``
    with physical position ``(5, 10, 15)`` and thickness ``(2, 3, 4)`` to both
    image viewer factories.
    Expected outcome: voxel half-thicknesses equal ``(1, 3, 8)``, centers equal
    ``(2, 10, 29)`` (clamped at X=29), and ``show_crosshair``/``sync_on_hover``
    flags match.
    """
    im = np.zeros((10, 20, 30))
    w = factory_fn(
        im,
        show_crosshair=False,
        sync_on_hover=True,
        slabs_thickness=(2, 3, 4),
        slabs_position=(5, 10, 15),
        pixel_sizes={"Z": 2.0, "Y": 1.0, "X": 0.5},
    )
    assert w.show_crosshair is False
    assert w.sync_on_hover is True
    assert w.z_t == 1 and w.y_t == 3 and w.x_t == 8
    assert w.z_s == 2 and w.y_s == 10 and w.x_s == 29
    assert w.sz == 2.0 and w.sy == 1.0 and w.sx == 0.5


def test_interactive_kwargs_scatter():
    """Verify ``show_zyx_max_scatter_interactive`` applies slab kwargs and ``pixel_sizes`` with ``(Z, Y, X)`` order.

    Strategy: pass seeded ``(Z, Y, X)`` point arrays in ``[0, 1]`` with explicit
    ``slabs_thickness``, ``slabs_position``, and ``pixel_sizes``.
    Expected outcome: the slabs are in physical units, as in the image
    viewers (D10): thickness ``(2, 3, 4)`` with pixel sizes ``(2, 1, 0.5)``
    gives index half-thicknesses ``(1, 3, 8)``; out-of-range positions clamp
    to ``1``, and pixel size traits match. (Changed by D10: the scatter took
    slabs in data units, so ``z_t`` was 2 and ``x_t`` 4.)
    """
    rng = np.random.default_rng(4)
    Z, Y, X = rng.random(10), rng.random(10), rng.random(10)
    w = show_zyx_max_scatter_interactive(
        (Z, Y, X),
        show_crosshair=False,
        sync_on_hover=True,
        slabs_thickness=(2, 3, 4),
        slabs_position=(5, 10, 15),
        pixel_sizes={"Z": 2.0, "Y": 1.0, "X": 0.5},
    )
    assert w.show_crosshair is False
    assert w.sync_on_hover is True
    assert w.z_t == 1 and w.y_t == 3 and w.x_t == 8
    assert w.z_s == 1 and w.y_s == 1 and w.x_s == 1
    assert w.sz == 2.0 and w.sy == 1.0 and w.sx == 0.5


def test_show_zyx_max_scatter_interactive_signature():
    """Verify ``show_zyx_max_scatter_interactive`` accepts tuples, lists, and ``(N, 3)`` arrays and rejects invalid shapes.

    Strategy: construct scatter viewers from ``(Z, Y, X)`` tuple, list, and
    ``(N, 3)`` stacked array, checking parsed coordinate arrays, then pass an
    ``(N, 2)`` array.
    Expected outcome: all three valid layouts preserve ``X_arr``, ``Y_arr``,
    ``Z_arr`` accurately; ``(N, 2)`` raises ``ValueError``.
    """
    rng = np.random.default_rng(5)
    X = rng.random(10) * 10
    Y = rng.random(10) * 10
    Z = rng.random(10) * 10
    channels = rng.random(10)

    w1 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels, render="points")
    w2 = show_zyx_max_scatter_interactive([Z, Y, X], channels=channels, render="points")
    points = np.stack([Z, Y, X], axis=1)
    w3 = show_zyx_max_scatter_interactive(points, channels=channels, render="points")

    for w in (w1, w2, w3):
        np.testing.assert_array_equal(w.X_arr, X)
        np.testing.assert_array_equal(w.Y_arr, Y)
        np.testing.assert_array_equal(w.Z_arr, Z)

    with pytest.raises(
        ValueError,
        match=r"points must be an array of shape .* representing \(Z, Y, X\) or a tuple/list of 3 arrays \(Z, Y, X\)\.",
    ):
        show_zyx_max_scatter_interactive(np.stack([Z, Y], axis=1), channels=channels)


@pytest.mark.parametrize("render", ["points", "density"])
def test_show_zyx_max_scatter_interactive_colormap(render):
    """Verify scatter viewers resolve and render single and multi-channel colormaps in both render modes.

    Strategy: render single-channel (``'viridis'``) and two-channel
    (``['viridis', 'plasma']``) scatter data in both ``'points'`` and
    ``'density'`` modes over the full slab.
    Expected outcome: ``colors_use`` and GUI ``channel_colors`` match the
    requested colormaps, and the rendered XY panel draws valid mapped colors
    (``viridis`` cmap / RGB facecolors in points mode; finite non-zero RGB
    image in density mode).
    """
    rng = np.random.default_rng(6)
    X = rng.uniform(1, 9, 40)
    Y = rng.uniform(1, 9, 40)
    Z = rng.uniform(1, 9, 40)
    ch1 = rng.random(40)
    ch_multi = [rng.random(40), rng.random(40)]

    w1 = show_zyx_max_scatter_interactive(
        (Z, Y, X), channels=ch1, colormap="viridis", render=render,
        slabs_thickness=(100, 100, 100), bins=32,
    )
    assert w1.colors_use == ["viridis"]
    assert w1.channel_colors == [get_gui_channel_color("viridis")]
    fig1 = w1._render()
    try:
        if render == "points":
            sc = [c for c in fig1.axXY.collections if isinstance(c, mcollections.PathCollection)]
            assert sc and sc[0].cmap.name == "viridis"
        else:
            rgb = np.asarray(fig1.axXY.images[0].get_array())
            assert rgb.ndim == 3 and rgb.shape[-1] == 3 and np.isfinite(rgb).all() and rgb.max() > 0
    finally:
        plt.close(fig1)

    w2 = show_zyx_max_scatter_interactive(
        (Z, Y, X), channels=ch_multi, colormap=["viridis", "plasma"], render=render,
        slabs_thickness=(100, 100, 100), bins=32,
    )
    assert w2.colors_use == ["viridis", "plasma"]
    assert w2.channel_colors == [get_gui_channel_color("viridis"), get_gui_channel_color("plasma")]
    fig2 = w2._render()
    try:
        if render == "points":
            sc = [c for c in fig2.axXY.collections if isinstance(c, mcollections.PathCollection)]
            assert sc and sc[0].get_facecolors().shape[0] == 40
            assert np.isfinite(sc[0].get_facecolors()).all()
        else:
            rgb = np.asarray(fig2.axXY.images[0].get_array())
            assert rgb.ndim == 3 and rgb.shape[-1] == 3 and np.isfinite(rgb).all() and rgb.max() > 0
    finally:
        plt.close(fig2)


def test_scatter_auto_render_threshold_and_idx_lists():
    """Verify ``render=None`` switches at 10,000 points and ``idx_lists`` channels render in both modes.

    Strategy: (a) construct scatter viewers with ``render=None`` for 50 points
    vs 10,000 points; (b) pass explicit index-list channels
    ``[np.array([0, 1, 2]), np.array([3, 4])]`` in both ``'points'`` and
    ``'density'`` modes over the full slab.
    Expected outcome: ``render=None`` selects ``'points'`` below 10,000 and
    ``'density'`` at 10,000; ``idx_lists`` sets ``mode == 'idx_lists'``,
    ``C == 2``, plots 3 and 2 points respectively in points mode, and renders
    both channels in density mode.
    """
    rng = np.random.default_rng(7)
    small = rng.uniform(0, 10, (50, 3))
    w_small = show_zyx_max_scatter_interactive(small, render=None)
    assert w_small.render == "points"

    large = rng.uniform(0, 10, (10_000, 3))
    w_large = show_zyx_max_scatter_interactive(large, render=None, bins=32)
    assert w_large.render == "density" and w_large.image_data

    pts = np.array([
        [1.0, 2.0, 2.0],
        [1.0, 2.0, 4.0],
        [1.0, 4.0, 2.0],
        [1.0, 8.0, 8.0],
        [1.0, 8.0, 9.0],
    ])
    idx_ch = [np.array([0, 1, 2]), np.array([3, 4])]

    w_pts = show_zyx_max_scatter_interactive(
        pts, channels=idx_ch, colormap=["red", "lime"], render="points",
        slabs_thickness=(100, 100, 100),
    )
    assert w_pts.mode == "idx_lists" and w_pts.C == 2
    fig_pts = w_pts._render()
    try:
        pcs = [c for c in fig_pts.axXY.collections if isinstance(c, mcollections.PathCollection)]
        assert [len(c.get_offsets()) for c in pcs] == [3, 2]
    finally:
        plt.close(fig_pts)

    w_den = show_zyx_max_scatter_interactive(
        pts, channels=idx_ch, colormap=["red", "lime"], render="density",
        slabs_thickness=(100, 100, 100), bins=16,
    )
    assert w_den.mode == "idx_lists" and w_den.C == 2
    fig_den = w_den._render()
    try:
        rgb = np.asarray(fig_den.axXY.images[0].get_array())
        assert rgb[..., 0].max() > 0.5 and rgb[..., 1].max() > 0.5
    finally:
        plt.close(fig_den)


def test_point_size_scaling():
    """Verify annotator marker ``point_size`` scales with ``point_size_scale`` times ``min(X, Y)``.

    Strategy: construct two annotators on a thin ``(1, 100, 100)`` volume with
    ``point_size_scale`` 0.1 and 0.5 via the public factory.
    Expected outcome: ``point_size`` equals 10 and 50 respectively.
    """
    im = np.zeros((1, 100, 100))
    w1 = show_zyx_max_slice_interactive_point_annotator(im, point_size_scale=0.1)
    w2 = show_zyx_max_slice_interactive_point_annotator(im, point_size_scale=0.5)
    assert w1.point_size == 10
    assert w2.point_size == 50
    assert w1.point_size < w2.point_size


def test_tnia_annotator_widget(tmp_path, monkeypatch):
    """Verify the annotator saves its points as a ``z,y,x`` CSV with ``$VAR`` expansion via ``save_csv_trigger``.

    Strategy: create an annotator via the public factory, add two points, set
    ``save_csv_filename`` using an environment variable, and bump ``save_csv_trigger``.
    Expected outcome: writes ``z,y,x\\n5,4,3\\n6,7,8\\n`` to the expanded path.
    """
    monkeypatch.setenv("TEST_DIR", str(tmp_path))
    im = np.random.default_rng(0).integers(0, 255, (10, 10, 10), dtype=np.uint8)
    w = show_zyx_max_slice_interactive_point_annotator(im)
    w.points = [[5, 4, 3], [6, 7, 8]]
    w.save_csv_filename = "$TEST_DIR/test_points.csv"
    w.save_csv_trigger += 1

    out = tmp_path / "test_points.csv"
    assert out.exists()
    assert out.read_text() == "z,y,x\n5,4,3\n6,7,8\n"


def test_annotator_annotate_kwarg_and_property():
    """Verify ``annotate`` kwarg default, explicit ``True``, and property getter/setter stay synced with ``annotation_mode``.

    Strategy: build default and ``annotate=True`` annotators via the public
    factory, then toggle ``w.annotate`` programmatically.
    Expected outcome: defaults to ``False``, respects ``annotate=True``, and
    mutating ``.annotate`` updates ``annotation_mode`` identically.
    """
    im = np.zeros((10, 20, 30))

    w_default = show_zyx_max_slice_interactive_point_annotator(im)
    assert w_default.annotation_mode is False
    assert w_default.annotate is False

    w_active = show_zyx_max_slice_interactive_point_annotator(im, annotate=True)
    assert w_active.annotation_mode is True
    assert w_active.annotate is True

    w_default.annotate = True
    assert w_default.annotation_mode is True and w_default.annotate is True
    w_default.annotate = False
    assert w_default.annotation_mode is False and w_default.annotate is False


def test_single_slider_start_end_sync_translate_and_clamp():
    """Verify start/end traits sync bidirectionally with center/thickness, translate, and clamp out-of-range writes.

    Strategy: initialize a ``(20, 40, 60)`` viewer with explicit position and
    thickness, update ``x_start``/``x_end`` directly, translate the range,
    update ``x_s``/``x_t``, and write an out-of-range ``x_end = 10**6``.
    Expected outcome: start/end and center/thickness stay paired through every
    operation, out-of-range writes clamp to ``X - 1`` with no clipping warning.
    """
    im = _structured_volume((20, 40, 60))
    w = show_zyx_max_slice_interactive(
        im, slabs_position=(10, 20, 30), slabs_thickness=(2, 4, 6), figsize=(4, 4)
    )

    assert (w.z_start, w.z_end) == (8, 12)
    assert (w.y_start, w.y_end) == (16, 24)
    assert (w.x_start, w.x_end) == (24, 36)

    w.x_start = 10
    w.x_end = 20
    assert (w.x_s, w.x_t) == (15, 5)

    # Range translation (+10)
    w.x_start = 20
    w.x_end = 30
    assert (w.x_s, w.x_t) == (25, 5)

    # Center/thickness -> start/end
    w.x_s = 40
    w.x_t = 10
    assert (w.x_start, w.x_end) == (30, 50)

    # Out-of-range write clamps cleanly to [0, X - 1]
    w.x_start = 0
    w.x_end = 10 ** 6
    assert w.x_start == 0
    assert w.x_end == 59
    assert w.x_s == (w.x_start + w.x_end) // 2
    assert w.x_t == (w.x_end - w.x_start) // 2
    assert w.warning_msg == ""


def test_save_trigger_exports_svg(tmp_path):
    """Verify bumping ``save_trigger`` writes a valid SVG file for slice and isometric scatter widgets.

    Strategy: configure ``save_filename`` inside ``tmp_path`` and increment
    ``save_trigger`` on both ``show_zyx_max_slice_interactive`` and
    ``show_iso_scatter``.
    Expected outcome: both target files are created, non-empty, and contain
    an ``<svg`` root element.
    """
    im = _structured_volume((6, 16, 16))
    w_slice = show_zyx_max_slice_interactive(im, figsize=(4, 4))
    slice_svg = tmp_path / "slice.svg"
    w_slice.save_filename = str(slice_svg)
    w_slice.save_trigger += 1
    assert slice_svg.exists() and "<svg" in slice_svg.read_text()

    X, Y, Z = _iso_points(n=30)
    w_iso = show_iso_scatter(X, Y, Z, figsize=(4, 4))
    iso_svg = tmp_path / "iso.svg"
    w_iso.save_filename = str(iso_svg)
    w_iso.save_trigger += 1
    assert iso_svg.exists() and "<svg" in iso_svg.read_text()


# ---------------------------------------------------------------------------
# Scale bar and layout / spacing / channel labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape, pixel_sizes, expected_text", [
    ((100, 200, 300), (1, 2, 3), "100 µm"),
    ((10, 50, 50), (0.5, 0.5, 0.5), "5 µm"),
    ((1, 5, 5), (10, 10, 10), "10 µm"),
    ((50, 100, 150), (2, 2, 2), "50 µm"),
    ((100, 100, 100), (1, 1, 1), "20 µm"),
    ((10, 512, 512), (10, 2, 2), "200 µm"),
])
def test_scale_bar_logic(shape, pixel_sizes, expected_text):
    """Verify scale-bar nice-length rounding matches physical width for both slice and scatter viewers.

    Strategy: render a zero ``uint8`` volume and a matching bounding point cloud
    at each ``(shape, pixel_sizes)`` pair and inspect the scale-bar text.
    Expected outcome: both slice and scatter viewers display ``expected_text``.
    """
    im = np.zeros(shape, dtype=np.uint8)
    w_slice = show_zyx_max_slice_interactive(im, pixel_sizes=pixel_sizes, figsize=(5, 5))
    fig_slice = w_slice._render()
    try:
        texts_slice = [txt.get_text() for ax in fig_slice.axes for txt in ax.texts]
        assert expected_text in texts_slice
    finally:
        plt.close(fig_slice)

    Z, Y, X = shape
    points = (np.array([0, Z - 1]), np.array([0, Y - 1]), np.array([0, X - 1]))
    w_scatter = show_zyx_max_scatter_interactive(points, pixel_sizes=pixel_sizes, figsize=(5, 5))
    fig_scatter = w_scatter._render()
    try:
        texts_scatter = [txt.get_text() for ax in fig_scatter.axes for txt in ax.texts]
        assert expected_text in texts_scatter
    finally:
        plt.close(fig_scatter)


def test_scale_bar_scales_with_figsize_scale():
    """Verify increasing ``figsize_scale`` enlarges both scale-bar font size and line width.

    Strategy: render a small ``(10, 50, 50)`` volume at ``figsize=(5, 5)`` with
    ``figsize_scale=1.0`` vs ``2.0``.
    Expected outcome: both the scale-bar label font size and bar linewidth are
    strictly larger at ``figsize_scale=2.0``.
    """
    im = np.zeros((10, 50, 50), dtype=np.uint8)
    w1 = show_zyx_max_slice_interactive(im, pixel_sizes=(0.5, 0.5, 0.5), figsize=(5, 5), figsize_scale=1.0)
    w2 = show_zyx_max_slice_interactive(im, pixel_sizes=(0.5, 0.5, 0.5), figsize=(5, 5), figsize_scale=2.0)
    fig1 = w1._render()
    fig2 = w2._render()
    try:
        bar1, lbl1 = _scale_bar_artists(fig1)
        bar2, lbl2 = _scale_bar_artists(fig2)
        assert bar1 is not None and lbl1 is not None
        assert bar2 is not None and lbl2 is not None
        assert lbl2.get_fontsize() > lbl1.get_fontsize()
        assert bar2.get_linewidth()[0] > bar1.get_linewidth()[0]
    finally:
        plt.close(fig1)
        plt.close(fig2)


@pytest.mark.parametrize("figsize", [(4, 4), (10, 10)])
def test_scale_bar_label_sits_centred_right_above_the_bar(figsize):
    """The scale-bar label is centred on the bar and just above it, at any figure size.

    It used to be centred in the empty bottom-right panel, away from the bar.
    Strategy: render, then compare display coordinates of the label's bounding
    box with the bar segment: horizontal centres agree within 1 px, the label's
    bottom is above the line but within about one font height of it, and the
    label is white on the dark theme.
    """
    im = np.random.default_rng(0).integers(0, 255, (10, 100, 100)).astype(np.uint8)
    fig = show_zyx_max_slabs(im, pixel_sizes=(1.0, 0.5, 0.5), figsize=figsize)
    try:
        bar, label = _scale_bar_artists(fig)
        assert bar is not None and label is not None
        renderer = fig.canvas.get_renderer()
        (x0, y0), (x1, _y1) = fig.axXY.transAxes.transform(np.asarray(bar.get_segments()[0]))
        bbox = label.get_window_extent(renderer)
        assert abs((bbox.x0 + bbox.x1) / 2 - (x0 + x1) / 2) < 1.0
        gap = bbox.y0 - y0
        font_px = label.get_fontsize() * fig.dpi / 72
        assert 0 < gap < font_px
        assert label.get_color() == "white"
    finally:
        plt.close(fig)


def test_scale_bar_show_false_hides_bar_and_label():
    """`scale_bar_show=False` removes both the bar and its label (static and widgets).

    Strategy: the static function and both widget types are rendered with the
    flag off (negative control: on), and the GUI trait can turn it back on.
    The copied parameters keep the setting so a paste reproduces the view.
    """
    im = np.random.default_rng(1).integers(0, 255, (10, 60, 60)).astype(np.uint8)
    for show in (True, False):
        fig = show_zyx_max_slabs(im, pixel_sizes=(1, 1, 1), scale_bar_show=show)
        try:
            bar, label = _scale_bar_artists(fig)
            assert (bar is not None, label is not None) == (show, show)
        finally:
            plt.close(fig)

    w = show_zyx_max_slice_interactive(im, pixel_sizes=(1, 1, 1), scale_bar_show=False)
    fig = w._render()
    try:
        assert _scale_bar_artists(fig) == (None, None)
    finally:
        plt.close(fig)
    w.copy_params_trigger += 1
    assert "scale_bar_show=False," in w.copy_params_string
    w.scale_bar_show = True
    fig = w._render()
    try:
        assert _scale_bar_artists(fig)[1] is not None
    finally:
        plt.close(fig)

    pts = (np.array([0.0, 9.0]), np.array([0.0, 59.0]), np.array([0.0, 59.0]))
    for render in ("points", "density"):
        ws = show_zyx_max_scatter_interactive(pts, pixel_sizes=(1, 1, 1), render=render, scale_bar_show=False)
        fig = ws._render()
        try:
            assert _scale_bar_artists(fig)[1] is None
        finally:
            plt.close(fig)


def test_scale_bar_fontsize_sets_label_and_bar_and_zero_hides_label_only():
    """`scale_bar_fontsize_pt` sizes label and bar together; 0 removes only the label.

    Strategy: render at 12 pt and 24 pt and check the label font size and that
    the bar's line width doubles with it (it follows the font). 0 must drop
    the label but keep the bar at the auto-size width (static function and
    widget trait); `scale_bar_show=False` removes both. The GUI's empty
    string means auto, and a negative size is rejected rather than ignored.
    """
    im = np.random.default_rng(5).integers(0, 255, (10, 60, 60)).astype(np.uint8)
    widths = {}
    for fs in (12, 24):
        fig = show_zyx_max_slabs(im, pixel_sizes=(1, 1, 1), scale_bar_fontsize_pt=fs)
        try:
            bar, label = _scale_bar_artists(fig)
            assert label.get_fontsize() == fs
            widths[fs] = bar.get_linewidth()[0]
        finally:
            plt.close(fig)
    assert widths[24] == pytest.approx(2 * widths[12])

    fig = show_zyx_max_slabs(im, pixel_sizes=(1, 1, 1))
    try:
        auto_width = _scale_bar_artists(fig)[0].get_linewidth()[0]
    finally:
        plt.close(fig)
    fig = show_zyx_max_slabs(im, pixel_sizes=(1, 1, 1), scale_bar_fontsize_pt=0)
    try:
        bar, label = _scale_bar_artists(fig)
        assert label is None and bar is not None
        assert bar.get_linewidth()[0] == pytest.approx(auto_width)
    finally:
        plt.close(fig)
    fig = show_zyx_max_slabs(im, pixel_sizes=(1, 1, 1), scale_bar_fontsize_pt=0, scale_bar_show=False)
    try:
        assert _scale_bar_artists(fig) == (None, None)
    finally:
        plt.close(fig)

    w = show_zyx_max_slice_interactive(im, pixel_sizes=(1, 1, 1), scale_bar_fontsize_pt=0, show_crosshair=False)
    fig = w._render()
    try:
        bar, label = _scale_bar_artists(fig)
        assert label is None and bar is not None
    finally:
        plt.close(fig)
    w.copy_params_trigger += 1
    assert "scale_bar_fontsize_pt=0.0," in w.copy_params_string
    w.scale_bar_fontsize_pt = ""  # GUI box cleared -> auto size
    fig = w._render()
    try:
        assert _scale_bar_artists(fig)[1] is not None
    finally:
        plt.close(fig)

    with pytest.raises(ValueError):
        show_zyx_max_slabs(im, pixel_sizes=(1, 1, 1), scale_bar_fontsize_pt=-1)


def test_scale_bar_and_label_have_faint_black_outline():
    """Bar and label carry a 30%-opacity black outline (visible over same-colour pixels).

    Strategy: inspect the path effects of both artists; the bar's stroke must
    be wider than the bar itself, otherwise it is hidden underneath.
    """
    im = np.random.default_rng(6).integers(0, 255, (10, 60, 60)).astype(np.uint8)
    fig = show_zyx_max_slabs(im, pixel_sizes=(1, 1, 1))
    try:
        bar, label = _scale_bar_artists(fig)
        for artist in (bar, label):
            strokes = [pe for pe in artist.get_path_effects() if isinstance(pe, patheffects.Stroke)]
            assert len(strokes) == 1
            gc = strokes[0]._gc
            assert gc["foreground"] == "black" and gc["alpha"] == 0.3
        bar_stroke = [pe for pe in bar.get_path_effects() if isinstance(pe, patheffects.Stroke)][0]
        assert bar_stroke._gc["linewidth"] > bar.get_linewidth()[0]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("unit, size, expected", [
    ("cm", 5e-5, "20 µm"),       # 256 px * 0.5 µm stored in cm
    ("nm", 500.0, "20 µm"),
    ("micron", 0.5, "20 µm"),
    ("\\u00B5m", 0.5, "20 µm"),  # ImageJ's escaped micro sign
    ("a.u.", 1.0, "50 a.u."),    # unknown unit: shown as given
    ("pixel", 1.0, "50 px"),     # ImageJ's uncalibrated unit: a pixel bar (D01 review fix)
])
def test_scale_bar_uses_pixel_size_unit(unit, size, expected):
    """Verify scale bars convert known length units to µm and display unknown units verbatim.

    A pixel unit ('pixel') means uncalibrated and gives the 'px' bar (it used to
    be shown verbatim as an unknown unit, '50 pixel').

    Strategy: pass pixel-size dicts with known length units and an unknown unit
    to ``show_zyx_max``, ``show_zyx_max_slice_interactive``, and
    ``show_zyx_max_scatter_interactive`` (points and density).
    Expected outcome: all four render the expected scale-bar label text.
    """
    im = np.zeros((4, 256, 256), dtype=np.uint8)
    ps = {"Z": size, "Y": size, "X": size, "unit": unit}
    fig = show_zyx_max(im, pixel_sizes=ps)
    try:
        assert expected in [t.get_text() for ax in fig.axes for t in ax.texts]
    finally:
        plt.close(fig)

    w = show_zyx_max_slice_interactive(im, pixel_sizes=ps)
    fig = w._render()
    try:
        assert expected in [t.get_text() for ax in fig.axes for t in ax.texts]
    finally:
        plt.close(fig)

    pts = (np.array([0, 3.0]), np.array([0, 255.0]), np.array([0, 255.0]))
    for render in ("points", "density"):
        ws = show_zyx_max_scatter_interactive(pts, pixel_sizes=ps, render=render)
        fig = ws._render()
        try:
            assert expected in [t.get_text() for ax in fig.axes for t in ax.texts], render
        finally:
            plt.close(fig)


@pytest.mark.parametrize("pixel_sizes, physical", [
    ({"Z": None, "Y": None, "X": None, "unit": None}, False),  # uncalibrated image_io dict
    ({"Z": 2.0, "Y": None, "X": None, "unit": "micron"}, False),  # only the Z step is known
    ({"Z": 2.0, "Y": 0.5, "X": None}, False),  # X, the axis the bar measures, is unknown
    ({"Z": None, "Y": 0.5, "X": None}, False),  # only the Y step is known
    ((2.0, None, None), False),
    ({"Z": None, "Y": 0.5, "X": 0.5, "unit": "micron"}, True),  # control: X and Y given
    (None, False),  # control: the uncalibrated label itself
])
def test_scale_bar_is_labelled_physically_only_with_x_and_y_pixel_sizes(pixel_sizes, physical):
    """The scale bar reads 'N µm' only when the X and Y pixel sizes were given, else 'N px'.

    Strategy: render a (10, 50, 50) volume with pixel sizes whose X and/or Y
    entry is None (as uncalibrated readers report them) through
    ``show_zyx_max``, the slice widget and the scatter widget (points and
    density). The None-axis fallback warnings are expected at construction;
    widget re-renders run with warnings as errors.

    Expected outcome: without X and Y the label is in pixels ('10 px', the
    same as for ``pixel_sizes=None``), not 'N µm', while the steps that
    were given still set the panel geometry (a Z step of 2 doubles the XZ
    panel height, a Y step of 0.5 halves the XY panel height), in the widget
    as in ``show_zyx_max``; re-rendering a widget does not repeat the
    warnings. With X and Y given the label stays in µm, and a missing Z takes
    the finest X/Y size. (Changed by D01: the uncalibrated label used to be
    '`pixel_sizes`', and a missing Z next to known X/Y used to be 1.0.)
    """
    expect_warning = pixel_sizes is not None and any(
        v is None for v in (pixel_sizes.values() if isinstance(pixel_sizes, dict) else pixel_sizes))
    raw = ((pixel_sizes.get("Z"), pixel_sizes.get("Y"), pixel_sizes.get("X")) if isinstance(pixel_sizes, dict)
           else (pixel_sizes or (None, None, None)))
    pz, py, px = (1.0 if v is None else v for v in raw)
    if raw[0] is None and raw[1] is not None and raw[2] is not None:
        pz = min(raw[1], raw[2])  # D01: missing Z with known X/Y

    def check(fig, geometry=False):
        try:
            label = _scale_bar_artists(fig)[1].get_text()
            assert ("µm" in label) if physical else label.endswith(" px"), label
            if geometry:
                for ax, size in ((fig.axXZ, 10 * pz), (fig.axXY, 50 * py)):
                    assert abs(np.diff(ax.get_ylim())[0]) == pytest.approx(size, rel=1e-6)
                assert abs(np.diff(fig.axXY.get_xlim())[0]) == pytest.approx(50 * px, rel=1e-6)
        finally:
            plt.close(fig)

    def build(fn, *args, **kwargs):
        if not expect_warning:
            return fn(*args, **kwargs)
        with pytest.warns(UserWarning, match="No value for axis"):
            return fn(*args, **kwargs)

    im = np.random.default_rng(0).random((10, 50, 50)).astype(np.float32)
    check(build(show_zyx_max, im, pixel_sizes=pixel_sizes), geometry=True)

    w = build(show_zyx_max_slice_interactive, im, pixel_sizes=pixel_sizes)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check(w._render(), geometry=True)

    pts = (np.array([0, 9.0]), np.array([0, 49.0]), np.array([0, 49.0]))
    for render in ("points", "density"):
        ws = build(show_zyx_max_scatter_interactive, pts, pixel_sizes=pixel_sizes, render=render)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            check(ws._render())


def test_tnia_figsize_scale():
    """Verify ``figsize_scale`` scales the auto-computed widget figure size linearly.

    Strategy: construct slice viewers on a cubic volume with ``figsize_scale``
    set to 1, 2, and 10.
    Expected outcome: ``figsize`` scales by exact factors of 2 and 10.
    """
    im = np.random.default_rng(8).random((20, 40, 40)).astype(np.float32)
    widget1 = show_zyx_max_slice_interactive(im, figsize_scale=1)
    widget2 = show_zyx_max_slice_interactive(im, figsize_scale=2)
    widget3 = show_zyx_max_slice_interactive(im, figsize_scale=10)
    assert np.allclose(np.array(widget2.figsize), 2 * np.array(widget1.figsize))
    assert np.allclose(np.array(widget3.figsize), 10 * np.array(widget1.figsize))


def test_interactive_spacing_pixel_sizes_vs_sxy():
    """Verify dict and tuple ``pixel_sizes`` produce identical auto figure sizes and XY axis geometry.

    Strategy: construct two viewers with ``{'Z': 2, 'Y': 1, 'X': 1}`` and
    ``(2, 1, 1)`` and compare rendered figure sizes and ``axXY`` positions.
    Expected outcome: figure sizes and ``axXY`` bounding boxes match closely.
    """
    im = np.random.default_rng(9).random((10, 50, 50)).astype(np.float32)
    w_dict = show_zyx_max_slice_interactive(im, pixel_sizes={"Z": 2, "Y": 1, "X": 1})
    w_tuple = show_zyx_max_slice_interactive(im, pixel_sizes=(2, 1, 1))
    fig_dict = w_dict._render()
    fig_tuple = w_tuple._render()
    try:
        np.testing.assert_allclose(fig_dict.get_size_inches(), fig_tuple.get_size_inches())
        np.testing.assert_allclose(fig_dict.axXY.get_position().bounds, fig_tuple.axXY.get_position().bounds)
    finally:
        plt.close(fig_dict)
        plt.close(fig_tuple)


def test_xy_anisotropy():
    """Verify anisotropic XY pixel sizes produce the expected physical width-to-height ratio on ``axXY``.

    Strategy: render a ``(10, 50, 100)`` volume with ``X=1.0, Y=0.2, Z=1.0`` so
    physical X = 100 and physical Y = 10.
    Expected outcome: ``axXY`` rendered width in inches is 10x its height.
    """
    im = np.random.default_rng(10).random((10, 50, 100)).astype(np.float32)
    w = show_zyx_max_slice_interactive(im, pixel_sizes={"Z": 1.0, "Y": 0.2, "X": 1.0})
    fig = w._render()
    try:
        pos_xy = fig.axXY.get_position()
        figW, figH = fig.get_size_inches()
        w_in = pos_xy.width * figW
        h_in = pos_xy.height * figH
        np.testing.assert_allclose(w_in / h_in, 10.0, rtol=1e-2)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "figsize, with_labels, gap_in",
    [
        ((8, 8), False, 1.0 / 16.0),
        ((8, 8), True, 1.0 / 16.0),
        ((12, 6), True, 1.0 / 16.0),
        ((6, 12), False, 1.0 / 8.0),
        ((12, 6), True, 1.0 / 8.0),
    ],
)
def test_fixed_physical_axes_gap(figsize, with_labels, gap_in):
    """Verify physical gap between adjacent ortho axes equals ``gap_in`` regardless of aspect ratio or labels.

    Strategy: render ``show_zyx`` across square, wide, and tall figure sizes
    with and without ``channel_labels`` and two ``gap_in`` values.
    Expected outcome: horizontal XY-ZY gap, vertical XZ-XY gap, and (when
    present) vertical XY-Labels gap all equal ``gap_in`` inches within ``1e-5``.
    """
    Z, Y, X = 20, 40, 120
    im = np.zeros((Z, Y, X), dtype=np.float32)
    labels = ["Ch0", "Ch1"] if with_labels else None

    fig = show_zyx(
        xy=im[10, :, :], xz=im[:, 20, :], zy=np.flip(np.rot90(im[:, :, 60], 1), 0),
        pixel_sizes={"X": 0.5, "Y": 0.5, "Z": 1.5},
        figsize=figsize,
        channel_labels=labels,
        gap_in=gap_in,
    )
    try:
        figW, figH = figsize
        pos_xy = fig.axXY.get_position()
        pos_zy = fig.axZY.get_position()
        pos_xz = fig.axXZ.get_position()

        np.testing.assert_allclose((pos_zy.x0 - pos_xy.x1) * figW, gap_in, atol=1e-5)
        np.testing.assert_allclose((pos_xy.y0 - pos_xz.y1) * figH, gap_in, atol=1e-5)

        if with_labels and fig.axLabels is not None:
            pos_labels = fig.axLabels.get_position()
            np.testing.assert_allclose((pos_labels.y0 - pos_xy.y1) * figH, gap_in, atol=1e-5)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("render_mode", ["points", "density"])
def test_scatter_widget_physical_axes_gap(render_mode):
    """Verify ``show_zyx_max_scatter_interactive`` maintains the exact 1/16-inch physical axis gap.

    Strategy: render seeded 3D scatter clouds in both ``'points'`` and
    ``'density'`` modes at ``figsize=(10, 8)``.
    Expected outcome: horizontal and vertical gaps between adjacent axes equal
    ``1/16`` inch within ``1e-5``.
    """
    rng = np.random.default_rng(11)
    X = rng.random(50) * 10
    Y = rng.random(50) * 20
    Z = rng.random(50) * 5

    w = show_zyx_max_scatter_interactive((Z, Y, X), figsize=(10, 8), render=render_mode)
    fig = w._render()
    try:
        figW, figH = (10, 8)
        pos_xy = fig.axXY.get_position()
        pos_zy = fig.axZY.get_position()
        pos_xz = fig.axXZ.get_position()

        np.testing.assert_allclose((pos_zy.x0 - pos_xy.x1) * figW, 1.0 / 16.0, atol=1e-5)
        np.testing.assert_allclose((pos_xy.y0 - pos_xz.y1) * figH, 1.0 / 16.0, atol=1e-5)
    finally:
        plt.close(fig)


def test_channel_label_height_and_fontsize_invariants():
    """Verify ``axLabels`` height and font size respond to figure width, explicit pt size, and stay anisotropy-invariant.

    Strategy: (a) compare auto label sizing on ``(4, 8)`` vs ``(12, 8)`` figures
    (changed by D12 / R1-23: the auto font follows the width, not the height);
    (b) compare explicit ``channel_label_fontsize_pt`` 12 vs 24; (c) compare a
    highly anisotropic ``(5, 10, 500)`` volume against an isotropic ``(5, 50, 50)``
    volume at the same ``figsize`` and font size; (d) verify propagation through
    interactive slice and scatter factories.
    Expected outcome: wider figures and larger font sizes strictly increase
    both rendered font size and ``axLabels`` physical height in inches; height is
    independent of image aspect ratio; interactive wrappers honour the kwarg.
    """
    im = np.zeros((10, 20, 20), dtype=np.float32)
    labels = ["Ch0", "Ch1"]

    fig_small = show_zyx(im[5], im[:, 10, :], im[:, :, 10], figsize=(4, 8), channel_labels=labels)
    fig_large = show_zyx(im[5], im[:, 10, :], im[:, :, 10], figsize=(12, 8), channel_labels=labels)
    try:
        h_small_in = fig_small.axLabels.get_position().height * 8.0
        h_large_in = fig_large.axLabels.get_position().height * 8.0
        fs_small = _label_fontsize(fig_small)
        fs_large = _label_fontsize(fig_large)
        assert 0 < h_small_in < h_large_in
        assert fs_small < fs_large
    finally:
        plt.close(fig_small)
        plt.close(fig_large)

    im_aniso = np.zeros((5, 10, 500), dtype=np.float32)
    im_iso = np.zeros((5, 50, 50), dtype=np.float32)
    fig_12 = show_zyx(
        im_aniso[2], im_aniso[:, 5, :], np.flip(np.rot90(im_aniso[:, :, 250], 1), 0),
        pixel_sizes=(1.0, 1.0, 1.0), figsize=(10, 8), channel_labels=labels, channel_label_fontsize_pt=12.0,
    )
    fig_24 = show_zyx(
        im_aniso[2], im_aniso[:, 5, :], np.flip(np.rot90(im_aniso[:, :, 250], 1), 0),
        pixel_sizes=(1.0, 1.0, 1.0), figsize=(10, 8), channel_labels=labels, channel_label_fontsize_pt=24.0,
    )
    fig_iso_24 = show_zyx(
        im_iso[2], im_iso[:, 25, :], np.flip(np.rot90(im_iso[:, :, 25], 1), 0),
        pixel_sizes=(1.0, 1.0, 1.0), figsize=(10, 8), channel_labels=labels, channel_label_fontsize_pt=24.0,
    )
    try:
        assert _label_fontsize(fig_12) == 12.0
        assert _label_fontsize(fig_24) == 24.0
        h12 = fig_12.axLabels.get_position().height * 8.0
        h24 = fig_24.axLabels.get_position().height * 8.0
        h24_iso = fig_iso_24.axLabels.get_position().height * 8.0
        assert h24 > h12
        np.testing.assert_allclose(h24, h24_iso, atol=1e-5)
    finally:
        plt.close(fig_12)
        plt.close(fig_24)
        plt.close(fig_iso_24)

    w_sl = show_zyx_max_slice_interactive(
        im_aniso, channel_labels=labels, channel_label_fontsize_pt=20.0, figsize=(10, 8)
    )
    assert w_sl.channel_label_fontsize_pt == 20.0
    fig_sl = w_sl._render()
    try:
        assert fig_sl.axLabels is not None and _label_fontsize(fig_sl) == 20.0
    finally:
        plt.close(fig_sl)

    rng = np.random.default_rng(12)
    pts_Z = rng.random(50) * 2
    pts_Y = rng.random(50) * 10
    pts_X = rng.random(50) * 100
    w_sc = show_zyx_max_scatter_interactive(
        (pts_Z, pts_Y, pts_X), channel_labels=labels, channel_label_fontsize_pt=14.0, figsize=(10, 8)
    )
    assert w_sc.channel_label_fontsize_pt == 14.0
    fig_sc = w_sc._render()
    try:
        assert fig_sc.axLabels is not None and _label_fontsize(fig_sc) == 14.0
    finally:
        plt.close(fig_sc)


@pytest.mark.parametrize("kind", ["slice", "scatter"])
def test_channel_label_bar_spans_exactly_the_xy_panel_pixels(kind):
    """The grey bar under the channel labels covers exactly the XY panel's pixel columns.

    It used to be a plotted line whose 'projecting' caps overshot both ends
    by half its width (a few px), and a snapped patch is still 1 px short of
    an imshow panel at some dpi (Agg places images at x0 + ceil(width)).
    Strategy: build one widget per kind, render its figure once, and rasterise
    at all 5 DPIs with and without ``bbox_inches='tight'``. The bar is the
    widest row of (0.85 grey) pixels with any alpha; its first and last column
    must equal those of the XY panel's opaque content.
    """
    rng = np.random.default_rng(0)
    if kind == "slice":
        w = show_zyx_max_slice_interactive(rng.random((8, 32, 48)).astype(np.float32), channel_labels=["BF"])
    else:
        w = show_zyx_max_scatter_interactive(rng.random((50, 3)) * [8, 32, 48], channel_labels=["A"])

    fig = w._render()
    try:
        for dpi in (72, 100, 150, 200, 300):
            for tight in (False, True):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight" if tight else None)
                a = plt.imread(io.BytesIO(buf.getvalue()))
                alpha = a[..., 3]
                grey = (alpha > 0.01) & (np.abs(a[..., :3] - 0.85).max(axis=2) < 0.02)
                runs = [(_first_run(grey[r]), r) for r in range(a.shape[0])]
                bar, bar_row = max(((run, r) for run, r in runs if run), key=lambda t: t[0][1] - t[0][0])
                img = _first_run(alpha[bar_row + int(0.2 * a.shape[0])] > 0.01)
                assert bar == img, (kind, dpi, tight, bar, img)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Axis bounds, annotation click/delete, and hover sync
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["slice", "scatter_points", "scatter_density"])
def test_widget_axis_bounds_structure(kind):
    """Verify construction populates well-formed ``axis_bounds`` for all three ortho planes.

    Strategy: construct a slice viewer and scatter viewers in both points and
    density modes (without calling ``_render_wrapper`` manually) and inspect
    every plane entry in ``w.axis_bounds``.
    Expected outcome: ``xy``, ``zy``, and ``xz`` are present with all required
    JS keys, ``x1 > x0``, ``y1_js > y0_js``, and 4-element ``bbox`` arrays.
    """
    rng = np.random.default_rng(13)
    if kind == "slice":
        w = show_zyx_max_slice_interactive(np.zeros((10, 20, 30), dtype=np.float32))
    else:
        pts = (rng.random(20) * 10, rng.random(20) * 10, rng.random(20) * 10)
        mode = "points" if kind == "scatter_points" else "density"
        w = show_zyx_max_scatter_interactive(pts, render=mode)

    expected_keys = {"x0", "x1", "y0", "y1", "y0_js", "y1_js", "bbox", "xlim", "ylim"}
    for plane in ("xy", "zy", "xz"):
        info = w.axis_bounds.get(plane)
        assert info is not None, f"{kind} missing {plane} axis_bounds"
        assert expected_keys.issubset(info.keys()), f"Missing keys in {kind}/{plane}"
        assert info["x1"] > info["x0"]
        assert info["y1_js"] > info["y0_js"]
        assert len(info["bbox"]) == 4


def test_axis_bounds_alignment():
    """Verify ``axis_bounds['xy']`` aligns with the Matplotlib ``axXY`` figure bounding box.

    Strategy: build an anisotropic slice viewer and compare ``axis_bounds['xy']``
    against ``fig.axXY.get_position()``.
    Expected outcome: ``x0`` and ``y0`` agree within ``1e-2``.
    """
    im = np.zeros((16, 64, 128), dtype=np.float32)
    pixel_sizes = {"X": 0.295, "Y": 1.0, "Z": 1.0}
    widget = show_zyx_max_slice_interactive(im, pixel_sizes=pixel_sizes)
    fig = widget._render()
    try:
        cell_bbox = fig.axXY.get_position()
        info = widget.axis_bounds["xy"]
        np.testing.assert_allclose(info["x0"], cell_bbox.x0, atol=1e-2)
        np.testing.assert_allclose(info["y0"], cell_bbox.y0, atol=1e-2)
    finally:
        plt.close(fig)


def test_ground_truth_annotation_all_planes():
    """Verify clicking on ``xy``, ``zy``, and ``xz`` panels registers exact anisotropic voxel coordinates.

    Strategy: transform known physical voxel centers through each panel's
    ``transData`` -> ``transFigure.inverted()`` into JS top-left figure
    fractions and assign ``widget.click_coords``.
    Expected outcome: each click appends the exact ``[z, y, x]`` target voxel.
    """
    Z, Y, X = 16, 64, 128
    im = np.zeros((Z, Y, X), dtype=np.float32)
    pixel_sizes = {"X": 0.295, "Y": 1.0, "Z": 1.0}

    widget = show_zyx_max_slice_interactive_point_annotator(
        im,
        pixel_sizes=pixel_sizes,
        channel_labels=["GRAY"],
        slabs_position=(8, 32, 64),
    )
    widget.annotation_mode = True
    widget.annotation_action = "add"

    targets = [
        ("xy", 8, 20, 45),
        ("zy", 12, 40, 64),
        ("xz", 5, 32, 100),
    ]

    for step, (plane, target_z, target_y, target_x) in enumerate(targets, start=1):
        widget.z_s = target_z
        widget.y_s = target_y
        widget.x_s = target_x

        fig = widget._render()
        try:
            ax = getattr(fig, f"ax{plane.upper()}")
            if plane == "xy":
                phys_a, phys_b = (target_x + 0.5) * widget.sx, (target_y + 0.5) * widget.sy
            elif plane == "zy":
                phys_a, phys_b = (target_z + 0.5) * widget.sz, (target_y + 0.5) * widget.sy
            else:
                phys_a, phys_b = (target_x + 0.5) * widget.sx, (target_z + 0.5) * widget.sz

            click_x, click_y = _fig_frac(fig, ax, phys_a, phys_b)
        finally:
            plt.close(fig)

        widget.click_coords = {"plane": plane, "x": click_x, "y": click_y, "t": step}
        assert [target_z, target_y, target_x] in widget.points, (
            f"Plane {plane} click at ({click_x:.3f}, {click_y:.3f}) mapped incorrectly."
        )


def test_annotation_deletion():
    """Verify click-to-delete across all three planes, ``add_point`` deduplication, and deletion guardrails.

    Strategy: add three points (testing ``add_point`` deduplication first), then
    exercise negative controls before deleting each point via ``_fig_frac`` +
    public ``widget.click_coords`` on ``xy``, ``zy``, and ``xz``:
      1. ``annotation_mode=False`` does not delete on click.
      2. Clicking ``> 20 px`` away (30 px offset) does not delete.
      3. Clicking when the point lies outside the active Z slab does not delete.
      4. Valid in-slab clicks on ``xy``, ``zy``, and ``xz`` each delete their point.
    Expected outcome: all negative controls leave ``widget.points`` intact, and
    the three valid plane clicks empty ``widget.points``.
    """
    im = np.zeros((16, 64, 64), dtype=np.float32)
    widget = show_zyx_max_slice_interactive_point_annotator(
        im, pixel_sizes=(2.0, 0.5, 0.5), slabs_thickness=(2.0, 2.0, 2.0)
    )

    widget.add_point(5, 10, 15)
    widget.add_point(5, 10, 15)  # duplicate must be ignored
    assert widget.points == [[5, 10, 15]]

    widget.add_point(10, 40, 20)
    widget.add_point(12, 25, 50)
    assert len(widget.points) == 3

    def _plane_click(plane, tz, ty, tx):
        fig = widget._render()
        try:
            ax = getattr(fig, f"ax{plane.upper()}")
            if plane == "xy":
                pa, pb = (tx + 0.5) * widget.sx, (ty + 0.5) * widget.sy
            elif plane == "zy":
                pa, pb = (tz + 0.5) * widget.sz, (ty + 0.5) * widget.sy
            else:
                pa, pb = (tx + 0.5) * widget.sx, (tz + 0.5) * widget.sz
            return _fig_frac(fig, ax, pa, pb)
        finally:
            plt.close(fig)

    widget.annotation_action = "delete"
    widget.z_s, widget.y_s, widget.x_s = 5, 10, 15

    # Control 1: annotation_mode=False ignores clicks
    widget.annotation_mode = False
    cx, cy = _plane_click("xy", 5, 10, 15)
    widget.click_coords = {"plane": "xy", "x": cx, "y": cy, "t": 1}
    assert [5, 10, 15] in widget.points

    # Control 2: click > 20 px away (30 px in x -> dist^2 = 900 > 400) does not delete
    widget.annotation_mode = True
    far_x, far_y = _plane_click("xy", 5, 10, 45)
    widget.click_coords = {"plane": "xy", "x": far_x, "y": far_y, "t": 2}
    assert [5, 10, 15] in widget.points

    # Control 3: wrong Z slab (z_s=0, z_t=1 -> slab [0, 1] excludes z=5) does not delete
    widget.z_s = 0
    cx, cy = _plane_click("xy", 5, 10, 15)
    widget.click_coords = {"plane": "xy", "x": cx, "y": cy, "t": 3}
    assert [5, 10, 15] in widget.points

    # Positive deletions on xy, zy, and xz
    for step, (plane, tz, ty, tx) in enumerate(
        [("xy", 5, 10, 15), ("zy", 10, 40, 20), ("xz", 12, 25, 50)], start=4
    ):
        widget.z_s, widget.y_s, widget.x_s = tz, ty, tx
        cx, cy = _plane_click(plane, tz, ty, tx)
        widget.click_coords = {"plane": plane, "x": cx, "y": cy, "t": step}
        assert [tz, ty, tx] not in widget.points

    assert widget.points == []


def test_hover_sync_all_planes():
    """Verify hover sync updates slider centers across ``xy``, ``zy``, and ``xz`` and respects ``sync_on_hover=False``.

    Strategy: construct a default anisotropic slice viewer (verifying
    ``sync_on_hover`` defaults to ``True``), compute figure fractions for target
    voxels on all three planes, dispatch ``hover_coords``, and then disable
    ``sync_on_hover`` as a negative control.
    Expected outcome: each plane updates its two corresponding slice indices;
    with ``sync_on_hover=False``, slider positions stay unchanged.
    """
    Z, Y, X = 16, 64, 128
    im = np.zeros((Z, Y, X), dtype=np.float32)
    pixel_sizes = {"X": 0.295, "Y": 1.0, "Z": 1.0}

    widget = show_zyx_max_slice_interactive(im, pixel_sizes=pixel_sizes)
    assert widget.sync_on_hover is True

    fig = widget._render()
    try:
        xy_fx, xy_fy = _fig_frac(fig, fig.axXY, (80 + 0.5) * widget.sx, (25 + 0.5) * widget.sy)
        zy_fx, zy_fy = _fig_frac(fig, fig.axZY, (12 + 0.5) * widget.sz, (40 + 0.5) * widget.sy)
        xz_fx, xz_fy = _fig_frac(fig, fig.axXZ, (30 + 0.5) * widget.sx, (4 + 0.5) * widget.sz)
    finally:
        plt.close(fig)

    widget.hover_coords = {"plane": "xy", "x": xy_fx, "y": xy_fy, "t": 1}
    assert (widget.x_s, widget.y_s) == (80, 25)

    widget.hover_coords = {"plane": "zy", "x": zy_fx, "y": zy_fy, "t": 2}
    assert (widget.z_s, widget.y_s) == (12, 40)

    widget.hover_coords = {"plane": "xz", "x": xz_fx, "y": xz_fy, "t": 3}
    assert (widget.x_s, widget.z_s) == (30, 4)

    # Negative control: sync_on_hover=False ignores hover_coords
    widget.sync_on_hover = False
    widget.hover_coords = {"plane": "xy", "x": xy_fx, "y": xy_fy, "t": 4}
    assert (widget.x_s, widget.y_s, widget.z_s) == (30, 40, 4)


def test_hover_sync_with_rotate_view():
    """Verify hover sync unrotates cursor coordinates accurately using an independent pixel-centroid oracle.

    Strategy: place a smooth Gaussian blob at voxel ``(10, 20, 48)`` inside a
    ``(32, 64, 64)`` volume, render with ``rotate_view=(30.0, 20.0, 45.0)`` over
    the full volume, locate the blob's rendered pixel centroid on each panel via
    ``_rendered_centroid`` (independent of internal rotation helpers), and send
    those figure coordinates to ``hover_coords``.
    Expected outcome: hovering ``xy``, ``xz``, and ``zy`` each snap their two
    corresponding slider indices to the exact blob voxel ``(10, 20, 48)``.
    """
    Z, Y, X = 32, 64, 64
    tz, ty, tx = 10, 20, 48
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    vol = np.exp(-((zz - tz) ** 2 + (yy - ty) ** 2 + (xx - tx) ** 2) / (2 * 2.5 ** 2)).astype(np.float32)

    widget = show_zyx_max_slice_interactive(
        vol,
        pixel_sizes=(1.0, 1.0, 1.0),
        sync_on_hover=True,
        rotate_view=(30.0, 20.0, 45.0),
        slabs_thickness=(100.0, 100.0, 100.0),
    )

    fig = widget._render()
    try:
        coords = {}
        for plane, ax in (("xy", fig.axXY), ("xz", fig.axXZ), ("zy", fig.axZY)):
            cx, cy = _rendered_centroid(ax)
            coords[plane] = _fig_frac(fig, ax, cx, cy)
    finally:
        plt.close(fig)

    widget.hover_coords = {"plane": "xy", "x": coords["xy"][0], "y": coords["xy"][1], "t": 1}
    assert (widget.x_s, widget.y_s) == (tx, ty)

    widget.x_s = 5
    widget.hover_coords = {"plane": "xz", "x": coords["xz"][0], "y": coords["xz"][1], "t": 2}
    assert (widget.x_s, widget.z_s) == (tx, tz)

    widget.y_s, widget.z_s = 5, 5
    widget.hover_coords = {"plane": "zy", "x": coords["zy"][0], "y": coords["zy"][1], "t": 3}
    assert (widget.z_s, widget.y_s) == (tz, ty)


# ---------------------------------------------------------------------------
# Render coalescing
# ---------------------------------------------------------------------------

def test_widget_construction_renders_figure_once(monkeypatch):
    """Building an interactive viewer must rasterize its figure exactly once.

    TNIAWidgetBase keeps two representations of every slab -- centre plus
    half-thickness (`x_s`/`x_t`) and start plus end (`x_start`/`x_end`) -- and
    observes `_render_wrapper` on all twelve traits. The initial synchronisation
    in `_init_observers` therefore cascades through most of them. Strategy:
    count calls to `_render` across construction. Rendering is by far the most
    expensive step (a full Matplotlib figure plus a PNG encode), so anything
    above one means the widget is paying for figures it immediately discards.
    """
    calls = []
    original = TNIASliceWidget._render

    def counting_render(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(TNIASliceWidget, "_render", counting_render)

    w = show_zyx_max_slice_interactive(_structured_volume(), figsize=(4, 4))

    assert len(calls) == 1
    assert w.image_data, "the single render must still populate image_data"


@pytest.mark.parametrize("trait, value", [
    ("x_start", 12),
    ("x_end", 40),
    ("z_s", 5),
    ("y_t", 4),
])
def test_slab_trait_change_renders_figure_once(trait, value):
    """Moving one slab control must rasterize the figure exactly once.

    Writing `x_start` makes the widget write back `x_s` and `x_t`, and both of
    those are observed by `_render_wrapper` too, so a naive implementation
    renders three times per slider step. Strategy: count `_render` calls for a
    single trait write, covering both directions of the synchronisation
    (start/end and centre/thickness). The image is also compared before and
    after, because coalescing the renders must not swallow the last one and
    leave a stale picture on screen.
    """
    w = show_zyx_max_slice_interactive(_structured_volume(), figsize=(4, 4))
    image_before = w.image_data
    counter = _count_renders(w)

    setattr(w, trait, value)

    assert counter["n"] == 1
    assert w.image_data != image_before


def test_one_render_per_multi_trait_update():
    """A middle-thumb drag (``x_start`` and ``x_end`` in one message) and a hover sync each render once.

    Strategy: wrap ``w._render`` to count invocations during a batched
    ``hold_trait_notifications`` update, a ``hover_coords`` sync, and a single
    ``vmax_list`` update.
    Expected outcome: each logical update invokes ``_render`` exactly once.
    """
    im = np.random.default_rng(0).random((10, 40, 50))
    w = show_zyx_max_slice_interactive(im)
    counter = _count_renders(w)

    with w.hold_trait_notifications():  # how ipywidgets applies one comm message
        w.x_start, w.x_end = w.x_start + 3, w.x_end + 3
    assert counter["n"] == 1

    counter["n"] = 0
    info = w.axis_bounds["xy"]
    w.hover_coords = {
        "plane": "xy",
        "x": info["x0"] + 0.3 * (info["x1"] - info["x0"]),
        "y": info["y0_js"] + 0.6 * (info["y1_js"] - info["y0_js"]),
        "t": 1,
    }
    assert counter["n"] == 1

    counter["n"] = 0
    w.vmax_list = [0.5]  # any real change still renders
    assert counter["n"] == 1


def test_scale_bar_toggle_triggers_one_rerender():
    """Toggling ``scale_bar_show`` re-renders the widget exactly once and updates ``image_data``.

    Strategy: attach a render counter to a slice viewer, flip ``scale_bar_show``
    from ``True`` to ``False``, and compare ``image_data`` before and after.
    Expected outcome: exactly one render occurs and the PNG changes.
    """
    im = np.random.default_rng(2).integers(0, 255, (10, 40, 40)).astype(np.uint8)
    w = show_zyx_max_slice_interactive(im, pixel_sizes=(1, 1, 1))
    before = w.image_data
    counter = _count_renders(w)
    w.scale_bar_show = False
    assert counter["n"] == 1
    assert w.image_data != before


def test_vmin_above_vmax_still_renders_with_vmax_vmin_plus_one():
    """A vmin set above vmax keeps every viewer rendering, with the compositor's vmax = vmin + 1 rule.

    Matplotlib's norms raise for vmin > vmax. The single-channel slice viewer
    passed the pair straight to them, so the render raised inside the
    observer (the error went to the kernel log, a pyplot figure leaked, and
    the image froze), while the multichannel compositor silently used
    vmax = vmin + 1. Typing vmin above vmax is easy when adjusting contrast
    one box at a time. Strategy: set ``vmin_list``/``vmax_list`` to
    [500]/[100] on a single-channel uint16 viewer.
    Expected outcome: no exception and no leaked figure; a new image whose norm
    spans (500, 501), the multichannel rule; a warning in ``warning_msg``; later
    changes (a slab move) still render. The same input on a two-channel viewer and
    on a continuous scatter channel renders with a warning too.
    Negative control: an ordered pair renders as given, without a warning.
    """
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 1000, (8, 32, 32)).astype(np.uint16)
    w = show_zyx_max_slice_interactive(vol, figsize=(4, 4))
    assert w.warning_msg == ""
    open_before = plt.get_fignums()
    before = w.image_data

    w.vmin_list = [500.0]
    w.vmax_list = [100.0]
    assert plt.get_fignums() == open_before
    assert w.image_data != before
    assert "vmin > vmax" in w.warning_msg
    fig = w._render()
    try:
        norm = fig.axXY.images[0].norm
        assert (norm.vmin, norm.vmax) == (500.0, 501.0)
    finally:
        plt.close(fig)

    inverted = w.image_data
    w.z_s = w.z_s + 2  # later renders still work (the crosshairs move)
    assert w.image_data != inverted

    w.vmax_list = [900.0]  # negative control: an ordered range is used as given
    assert w.warning_msg == ""
    fig = w._render()
    try:
        norm = fig.axXY.images[0].norm
        assert (norm.vmin, norm.vmax) == (500.0, 900.0)
    finally:
        plt.close(fig)

    multi = show_zyx_max_slice_interactive([vol, vol], figsize=(4, 4))
    before = multi.image_data
    multi.vmin_list = [500.0, 0.0]
    multi.vmax_list = [100.0, 900.0]
    assert multi.image_data != before
    assert "vmin > vmax" in multi.warning_msg and "channel 0" in multi.warning_msg

    pts = rng.random((200, 3)) * [8, 32, 32]
    scatter = show_zyx_max_scatter_interactive(pts, channels=rng.uniform(0, 1000, 200), render="points")
    before = scatter.image_data
    scatter.vmin_list = [500.0]
    scatter.vmax_list = [100.0]
    assert scatter.image_data != before
    assert "vmin > vmax" in scatter.warning_msg


def test_failed_render_keeps_last_frame_reports_warning_and_retries(monkeypatch):
    """A render that raises during an interactive change is reported in the GUI and retried.

    An exception raised from a trait observer only reaches the kernel log, so
    the image used to freeze without any message; the half-built pyplot figure
    was never closed; and the render state was recorded before rendering, so
    the second notification of the same comm message (``x_start`` and
    ``x_end`` of a middle-thumb drag) skipped the retry. Strategy: make
    ``_render`` open a pyplot figure and raise once, then move the slab with
    both traits in one ``hold_trait_notifications`` block.
    Expected outcome: no exception escapes, no figure is left open, the
    failure is shown in ``warning_msg`` and the second notification renders
    the new slab (the warning clears).
    Negative control: a failure while building the widget still raises.
    """
    w = show_zyx_max_slice_interactive(_structured_volume(), figsize=(4, 4))
    good = w.image_data
    inner = w._render
    calls = {"n": 0}

    def fail_once():
        calls["n"] += 1
        if calls["n"] == 1:
            plt.figure()
            raise RuntimeError("render exploded")
        return inner()

    seen = []
    w.observe(lambda c: seen.append(c["new"]), names="warning_msg")
    w._render = fail_once
    open_before = plt.get_fignums()
    with w.hold_trait_notifications():  # how ipywidgets applies one comm message
        w.x_start, w.x_end = w.x_start + 3, w.x_end + 3
    assert plt.get_fignums() == open_before
    assert any("render exploded" in m for m in seen)
    assert calls["n"] == 2
    assert w.image_data != good
    assert w.warning_msg == ""

    def broken_render(self):
        raise RuntimeError("bad input")

    monkeypatch.setattr(TNIASliceWidget, "_render", broken_render)
    with pytest.raises(RuntimeError, match="bad input"):
        show_zyx_max_slice_interactive(_structured_volume())


# ---------------------------------------------------------------------------
# Histograms, bit-depth inference, and robust default vmax
# ---------------------------------------------------------------------------

def test_compute_histogram_empty_and_all_nan_inputs():
    """Degenerate inputs yield empty payloads rather than raising.

    The histogram is computed once per channel at construction and shipped to
    the frontend, which skips channels whose counts are empty. Strategy: feed an
    empty array and an all-NaN float array -- the two ways a channel can carry
    no usable values -- and require the documented empty payload from both,
    since a raise here would break widget construction entirely.
    """
    assert compute_histogram(np.array([], dtype=np.float32)) == {"counts": [], "bin_edges": []}
    all_nan = np.full(10, np.nan, dtype=np.float32)
    assert compute_histogram(all_nan) == {"counts": [], "bin_edges": []}


@pytest.mark.parametrize("dtype, expected_range", [
    (np.uint8, (0.0, 255.0)),
    (np.uint16, (0.0, 65535.0)),
    (bool, (0.0, 1.0)),
])
def test_compute_histogram_spans_full_dtype_range(dtype, expected_range):
    """Integer and boolean channels bin over the whole dtype range, not the data range.

    The frontend draws the vmin/vmax tone curve against these bin edges, so the
    edges have to mean the same thing as the vmin/vmax boxes, which are bounded
    by dtype. Strategy: pass data occupying only a sliver of the dtype range and
    assert the edges still span the full range -- binning to the data range
    instead would silently misplace the curve for every dark image.
    """
    arr = np.ones((4, 4), dtype=dtype)
    result = compute_histogram(arr, bins=16)

    assert len(result["counts"]) == 16
    assert (result["bin_edges"][0], result["bin_edges"][-1]) == expected_range


def test_compute_histogram_counts_are_log_frequencies():
    """Counts are log1p-transformed so sparse tails stay visible.

    A fluorescence channel is dominated by background, and on a linear axis the
    signal bins are invisible next to it. Strategy: invert the transform with
    expm1 and check the recovered counts sum to the number of input values, and
    that a heavily populated bin is compressed relative to a sparse one -- this
    pins the transform itself rather than any particular bin layout.
    """
    arr = np.concatenate([np.zeros(1000, dtype=np.uint8), np.full(10, 200, dtype=np.uint8)])
    result = compute_histogram(arr, bins=16)
    counts = np.array(result["counts"])

    np.testing.assert_allclose(np.expm1(counts).sum(), arr.size, rtol=1e-6)
    background, signal = np.expm1(counts).max(), np.expm1(counts)[np.expm1(counts) > 0].min()
    assert background / signal == pytest.approx(100.0, rel=1e-6)
    assert counts.max() / counts[counts > 0].min() < 10.0


def test_compute_histogram_rescales_subsampled_counts():
    """Subsampling large channels preserves the overall count magnitude.

    Large volumes are strided down before binning to keep widget construction
    responsive, which would otherwise shrink every bar by the stride factor and
    change the shape of the log curve. Strategy: force subsampling with a small
    max_samples and check the recovered counts still total the full array size.
    """
    arr = np.arange(1000, dtype=np.float32)
    result = compute_histogram(arr, bins=10, max_samples=100)

    np.testing.assert_allclose(np.expm1(result["counts"]).sum(), arr.size, rtol=1e-6)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_compute_histogram_and_widget_exclude_nonfinite_values(nonfinite):
    """Non-finite values (NaN, +inf, -inf) are excluded from histograms and default ``vmax``.

    Strategy: (a) mix ``nonfinite`` into a 3-element finite float array and pass
    to ``compute_histogram``; (b) inject ``nonfinite`` into a ``[0, 1]`` float32
    volume and construct ``show_zyx_max_slice_interactive``.
    Expected outcome: histogram edges stay finite, recovered counts total only
    the 3 finite entries, and the widget renders with a finite ``vmax_list[0] <= 1.0``.
    """
    arr = np.array([0.0, 1.0, 2.0, nonfinite, nonfinite], dtype=np.float32)
    result = compute_histogram(arr, bins=4)
    assert np.all(np.isfinite(result["bin_edges"]))
    np.testing.assert_allclose(np.expm1(result["counts"]).sum(), 3, rtol=1e-6)

    vol = np.random.default_rng(8).uniform(0, 1, (4, 16, 16)).astype(np.float32)
    vol[0, 0, 0] = nonfinite
    w = show_zyx_max_slice_interactive(vol, pixel_sizes=(1, 1, 1))
    assert np.isfinite(w.vmax_list[0]) and w.vmax_list[0] <= 1.0
    assert np.all(np.isfinite(w.histograms_data[0]["bin_edges"]))
    assert len(w.histograms_data[0]["counts"]) == 128
    assert w.image_data


@pytest.mark.parametrize("values, dtype, expected", [
    ([0, 200], np.uint8, (8, False)),
    ([0, 3], np.uint16, (8, False)),          # never narrower than 8 bits
    ([0, 1000], np.uint16, (10, False)),
    ([0, 2531], np.uint16, (12, False)),      # 12-bit camera, dim channel
    ([0, 4095], np.uint16, (12, False)),      # boundary: 2**12 - 1 still fits
    ([0, 4096], np.uint16, (14, False)),      # boundary: one level more does not
    ([0, 65535], np.uint16, (16, False)),
    ([-2048, 2047], np.int16, (12, True)),    # sign bit counted
    ([-2049, 0], np.int16, (14, True)),
    ([0, 4095], np.float32, (12, False)),     # counts converted to float
    ([0, 70000], np.uint32, (24, False)),
])
def test_infer_bit_depth_snaps_to_common_detector_depths(values, dtype, expected):
    """The bit depth is the smallest common depth (8/10/12/14/16/24/...) holding the data.

    Strategy: place the maximum on both sides of each 2**b - 1 boundary, so an
    off-by-one in the bit count (2**b needs b + 1 bits) or a failure to snap up
    to the next common depth changes the result. Signed data must count the
    sign bit, and integer-valued floats must behave like their integer source.
    """
    assert infer_bit_depth(np.array(values, dtype=dtype)) == expected


@pytest.mark.parametrize("arr", [
    np.array([0.0, 0.5, 4095.5], dtype=np.float32),   # non-integer floats
    np.array([True, False]),
    np.array([], dtype=np.uint16),
])
def test_infer_bit_depth_returns_none_without_integer_levels(arr):
    """Data without integer intensity levels have no bit depth (negative control).

    Strategy: non-integer floats, boolean masks and empty arrays must return
    None rather than a guessed depth, so their histograms keep the data range.
    """
    assert infer_bit_depth(arr) is None


def test_slice_widget_histogram_spans_inferred_12bit_range():
    """12-bit data stored as uint16 get a 0..4095 histogram, not 0..65535.

    With the dtype range, 12-bit data occupy only the left 1/16 of the canvas.
    Strategy: build two uint16 channels from a 12-bit source (one dim, max 2531),
    and check that both histograms span the 4096 levels with integer levels
    centred in their bins (every bin covers exactly 4096 / 128 = 32 levels, so
    no comb artefact), that the recovered counts total every voxel, and that
    the inferred depth is reported for the frontend tooltip.
    """
    rng = np.random.default_rng(0)
    bright = rng.integers(0, 4096, (4, 16, 16)).astype(np.uint16)
    dim = rng.integers(0, 2532, (4, 16, 16)).astype(np.uint16)
    w = show_zyx_max_slice_interactive([bright, dim], pixel_sizes=(1, 1, 1))

    for h in w.histograms_data:
        edges = np.array(h["bin_edges"])
        assert h["bit_depth"] == 12
        assert (edges[0], edges[-1]) == (-0.5, 4095.5)
        np.testing.assert_allclose(np.diff(edges), 32.0)
        np.testing.assert_allclose(np.expm1(h["counts"]).sum(), bright.size, rtol=1e-6)


def test_slice_widget_bit_depth_kwarg_enforces_range_per_channel():
    """`bit_depth` overrides the inference: an int, 'dtype', 'data', or a per-channel list.

    Strategy: give the same 12-bit uint16 channel three times with a different
    setting each and check the bin edges: enforced 16 bits and 'dtype' span
    0..65535, 'data' spans exactly the data min..max. A scalar is broadcast.
    """
    rng = np.random.default_rng(1)
    ch = rng.integers(3, 2532, (4, 16, 16)).astype(np.uint16)
    lo, hi = int(ch.min()), int(ch.max())

    w = show_zyx_max_slice_interactive([ch, ch, ch], pixel_sizes=(1, 1, 1), bit_depth=[16, "dtype", "data"])
    ranges = [(h["bin_edges"][0], h["bin_edges"][-1]) for h in w.histograms_data]
    assert ranges == [(-0.5, 65535.5), (-0.5, 65535.5), (lo - 0.5, hi + 0.5)]
    assert [h["bit_depth"] for h in w.histograms_data] == [16, 16, None]

    w = show_zyx_max_slice_interactive([ch, ch], pixel_sizes=(1, 1, 1), bit_depth=14)
    assert all(h["bin_edges"][-1] == 16383.5 for h in w.histograms_data)


def test_annotator_annotation_channel_stays_8bit_under_enforced_bit_depth():
    """A scalar `bit_depth` applies to the image channels only; annotations stay 8-bit.

    The annotation channel is an internal uint8 mask drawn at 255. Broadcasting
    e.g. 12 bits to it would squeeze its histogram into the left 1/16.
    """
    rng = np.random.default_rng(2)
    ch = rng.integers(0, 4096, (4, 16, 16)).astype(np.uint16)
    w = show_zyx_max_slice_interactive_point_annotator([ch, ch], pixel_sizes=(1, 1, 1), bit_depth=16)
    assert [h["bit_depth"] for h in w.histograms_data] == [16, 16, 8]


@pytest.mark.parametrize("bad", [0, 65, "bits", 12.0, True])
def test_infer_histogram_range_rejects_invalid_bit_depth(bad):
    """Invalid `bit_depth` values raise instead of silently falling back to auto."""
    with pytest.raises(ValueError):
        infer_histogram_range(np.zeros(4, dtype=np.uint16), bad)


def test_infer_histogram_range_float_conventions():
    """Floats: normalised data keep [0, 1] / [-1, 1], integer-valued ones get a bit depth.

    Strategy: a float 0/1 mask must stay in [0, 1] (not be read as 8-bit counts),
    signed normalised data in [-1, 1], and generic floats return None so the
    histogram falls back to the data range (the previous behaviour).
    """
    assert infer_histogram_range(np.array([0.0, 1.0], dtype=np.float32)) == ((0.0, 1.0), None)
    assert infer_histogram_range(np.array([-0.3, 0.9])) == ((-1.0, 1.0), None)
    assert infer_histogram_range(np.array([0.0, 4095.0])) == ((-0.5, 4095.5), 12)
    assert infer_histogram_range(np.array([0.2, 7.5])) == (None, None)


def test_slice_widget_default_vmax_is_robust_to_hot_pixels():
    """Default vmax is the 99.99th percentile, so a few hot pixels do not set the range.

    Strategy: a 12-bit-like channel whose values stay <= 1000 except 5 hot
    pixels at 4095 (0.005% of the voxels, below the 0.01% tail). The plain
    maximum (4095) would map the real signal to < 25% brightness; the default
    must instead land within the bulk of the data (<= 1000) and be an integer
    for integer data. An explicit vmax must still win, and a channel whose
    signal covers fewer voxels than the 0.01% tail (here 5 bright voxels on a
    zero background, where the percentile is 0) falls back to the maximum, so
    it is not blank.
    """
    rng = np.random.default_rng(3)
    ch = rng.integers(0, 1001, (10, 100, 100)).astype(np.uint16)
    ch.flat[rng.choice(ch.size, 5, replace=False)] = 4095
    sparse = np.zeros((10, 100, 100), dtype=np.uint16)
    sparse.flat[:5] = 3000

    w = show_zyx_max_slice_interactive([ch, sparse], pixel_sizes=(1, 1, 1))
    vmax_ch, vmax_sparse = w.vmax_list
    assert 990 <= vmax_ch <= 1000 and float(vmax_ch).is_integer()
    assert vmax_sparse == 3000.0

    w = show_zyx_max_slice_interactive([ch, sparse], pixel_sizes=(1, 1, 1), vmax=[4095, None])
    assert w.vmax_list == [4095.0, 3000.0]


def test_slice_widget_default_vmax_keeps_sparse_signal_unsaturated():
    """A sparse channel (signal on ~0.2% of voxels) keeps its signal below the default vmax.

    This is the case that motivated q = 99.99: at the 99.9th percentile the
    default fell inside the signal and saturated ~40% of it. Strategy: 200
    signal voxels ramping 300..2000 on a 100-count background (0.2% of 1e5
    voxels). The default vmax must saturate at most 5% of the signal voxels;
    as a control, the 99.9th percentile of the same data saturates far more.
    """
    rng = np.random.default_rng(7)
    ch = rng.normal(100, 20, (10, 100, 100)).clip(0, 4095).astype(np.uint16)
    signal = np.linspace(300, 2000, 200).astype(np.uint16)
    ch.flat[rng.choice(ch.size, signal.size, replace=False)] = signal

    w = show_zyx_max_slice_interactive(ch, pixel_sizes=(1, 1, 1))
    assert (signal > w.vmax_list[0]).mean() <= 0.05
    assert (signal > np.percentile(ch, 99.9)).mean() > 0.3


def test_annotator_default_vmax_is_robust_and_annotations_stay_255():
    """The annotator uses the same robust default for image channels; annotations keep 255."""
    rng = np.random.default_rng(4)
    ch = rng.integers(0, 1001, (10, 100, 100)).astype(np.uint16)
    ch.flat[:5] = 4095  # 0.005% hot pixels, below the 0.01% tail
    w = show_zyx_max_slice_interactive_point_annotator(ch, pixel_sizes=(1, 1, 1))
    assert w.vmax_list[0] <= 1000 and w.vmax_list[-1] == 255.0


def test_default_vmax_of_all_nan_channel_does_not_warn():
    """Verify an all-NaN channel falls back to ``vmax=1.0`` without emitting ``RuntimeWarning``.

    Strategy: construct a two-channel viewer with one normal and one all-NaN
    channel while promoting ``RuntimeWarning`` to errors.
    Expected outcome: construction succeeds cleanly with ``vmax_list[1] == 1.0``.
    """
    im = np.random.default_rng(0).random((4, 8, 8))
    nan = np.full_like(im, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        w = show_zyx_max_slice_interactive([im, nan])
    assert w.vmax_list[1] == 1.0


def test_bool_mask_channels_open_in_slice_viewers():
    """Boolean masks (e.g. from ``extract_surface``) open in the slice viewer and the annotator.

    ``_robust_vmax`` (now ``_conventions._robust_max``) used to call ``np.percentile`` on the bool sample, which
    raises NumPy's "boolean subtract" TypeError, and every viewer computes it
    per channel at construction. Strategy: build the single-channel viewer, a
    two-channel viewer with a mask channel and the point annotator on a mask.
    Expected outcome: each renders; the mask channel's default vmax is 1.0,
    its dtype stays ``bool`` and its histogram keeps the 1-bit (0/1) range.
    Negative control: ``_robust_max`` of the mask equals that of the same
    mask as uint8, and an all-False mask gives 0.0 instead of raising.
    """
    mask = np.zeros((8, 32, 32), dtype=bool)
    mask[2:5, 10:20, 10:20] = True
    vol = np.random.default_rng(0).integers(0, 4000, mask.shape).astype(np.uint16)

    assert tpa._robust_max(mask) == tpa._robust_max(mask.astype(np.uint8)) == 1.0
    assert tpa._robust_max(np.zeros_like(mask)) == 0.0

    single = show_zyx_max_slice_interactive(mask, figsize=(4, 4))
    multi = show_zyx_max_slice_interactive([vol, mask], figsize=(4, 4))
    annot = show_zyx_max_slice_interactive_point_annotator(mask, figsize=(4, 4))
    for w, i in ((single, 0), (multi, 1), (annot, 0)):
        assert w.image_data
        assert w.vmax_list[i] == 1.0
        assert w.channel_dtypes[i] == "bool"
        assert w.histograms_data[i]["bit_depth"] == 1
        assert w.histograms_data[i]["bin_edges"][0] == -0.5
        assert w.histograms_data[i]["bin_edges"][-1] == 1.5


# ---------------------------------------------------------------------------
# Copy-parameters, copy-PNG, marimo app, and standalone execution
# ---------------------------------------------------------------------------

def test_copy_params_emits_reusable_physical_parameters():
    """The copy button emits parameters that reproduce the current view.

    Its whole purpose is to let a user tune a view interactively and paste the
    result into a script, so the emitted text has to be valid keyword arguments
    in physical units -- the widget works in voxel indices, but the plotting API
    takes micrometres. Strategy: drive the trigger, evaluate the emitted text as
    keyword arguments, feed them straight back into the factory, and require the
    rebuilt widget to land on the same voxel slabs.
    """
    vol = _structured_volume()
    pixel_sizes = {"Z": 2.0, "Y": 0.5, "X": 0.5}
    w = show_zyx_max_slice_interactive(vol, pixel_sizes=pixel_sizes, figsize=(4, 4))

    w.x_s, w.y_s, w.z_s = 20, 16, 6
    w.x_t, w.y_t, w.z_t = 4, 3, 2

    w.copy_params_trigger += 1
    params = eval(f"dict({w.copy_params_string})", {"__builtins__": {"dict": dict}}, {})

    # Positions and thicknesses are reported in physical units, (Z, Y, X) order.
    assert params["slabs_position"] == (6 * 2.0, 16 * 0.5, 20 * 0.5)
    assert params["slabs_thickness"] == (2 * 2.0, 3 * 0.5, 4 * 0.5)
    assert len(params["vmin"]) == len(params["vmax"]) == 1
    assert len(params["gamma"]) == len(params["opacity"]) == 1

    rebuilt = show_zyx_max_slice_interactive(vol, pixel_sizes=pixel_sizes, figsize=(4, 4), **params)

    assert (rebuilt.z_s, rebuilt.y_s, rebuilt.x_s) == (w.z_s, w.y_s, w.x_s)
    assert (rebuilt.z_t, rebuilt.y_t, rebuilt.x_t) == (w.z_t, w.y_t, w.x_t)


def test_scatter_copy_params_round_trip():
    """Verify scatter ``copy_params_trigger`` emits data-coordinate slab kwargs that reproduce the slab bounds.

    Strategy: build a scatter viewer on offset coordinates, adjust ``x`` and
    ``z`` slabs, bump ``copy_params_trigger``, parse the emitted ``slabs_*``
    kwargs, and rebuild the viewer.
    Expected outcome: all six start/end slab traits match the original widget.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(500, 600, 400)
    Y = rng.uniform(0, 50, 400)
    Z = rng.uniform(20, 40, 400)
    w = show_zyx_max_scatter_interactive((Z, Y, X), pixel_sizes=(2, 0.5, 0.5))
    w.x_start, w.x_end = 44, 56
    w.z_start, w.z_end = 6, 14
    w.copy_params_trigger += 1
    kwargs = {}
    for line in w.copy_params_string.splitlines():
        key, val = line.rstrip(",").split("=", 1)
        if key.startswith("slabs_"):
            kwargs[key] = eval(val)
    w2 = show_zyx_max_scatter_interactive((Z, Y, X), pixel_sizes=(2, 0.5, 0.5), **kwargs)
    for attr in ("x_start", "x_end", "y_start", "y_end", "z_start", "z_end"):
        assert getattr(w2, attr) == getattr(w, attr), attr


@pytest.mark.parametrize("kind", ["slice", "scatter"])
def test_copy_params_sends_the_final_string_once_per_click(kind):
    """Every "📋 kwargs" click sends the final kwargs string exactly once, even for an unchanged view.

    The browser copies on ``change:copy_params_string``, and traitlets does not
    notify an unchanged value, so a second click on an unchanged view copied
    nothing. The scatter viewer also assigned the string twice per click, so
    the browser first received slabs in index units. Strategy: observe the
    trait and bump ``copy_params_trigger`` twice without changing the view.
    Expected outcome: exactly two non-empty notifications, both equal to the
    final string (the trait is cleared in between, as for the PNG copy).
    """
    rng = np.random.default_rng(0)
    if kind == "slice":
        w = show_zyx_max_slice_interactive(_structured_volume(), figsize=(4, 4))
    else:
        pts = rng.random((50, 3)) * [8, 32, 48] + [0, 0, 500]
        w = show_zyx_max_scatter_interactive(pts)

    seen = []
    w.observe(lambda c: seen.append(c["new"]), names="copy_params_string")
    w.copy_params_trigger += 1
    w.copy_params_trigger += 1
    final = w.copy_params_string
    assert final.startswith("slabs_position=")
    assert [s for s in seen if s] == [final, final]


@pytest.mark.parametrize("kind", ["slice", "scatter"])
def test_copy_png_trigger_returns_tight_figure_png(kind):
    """Bumping `copy_png_trigger` puts a base64 PNG of the figure in `copy_png_data`.

    Strategy: decode the payload and check (a) PNG magic bytes, (b) it is the
    300-dpi tight-bbox render (larger than the 100-dpi on-screen image), and
    (c) a second identical request still fires a change (the trait is cleared
    first), which the JS listener relies on.
    """
    rng = np.random.default_rng(0)
    if kind == "slice":
        w = show_zyx_max_slice_interactive(rng.random((8, 32, 48)).astype(np.float32))
    else:
        pts = rng.random((50, 3)) * [8, 32, 48]
        w = show_zyx_max_scatter_interactive(pts)

    seen = []
    w.observe(lambda c: seen.append(c["new"]), names="copy_png_data")
    w.copy_png_trigger += 1
    data = base64.b64decode(w.copy_png_data)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"

    shown = plt.imread(io.BytesIO(base64.b64decode(w.image_data)))
    copied = plt.imread(io.BytesIO(data))
    assert copied.shape[0] > shown.shape[0] and copied.shape[1] > shown.shape[1]

    w.copy_png_trigger += 1
    assert len([s for s in seen if s]) == 2
    assert "" in seen


def test_widget_png_ignores_tight_savefig_rcparam():
    """axis_bounds are figure fractions, so the PNG must show the whole figure frame."""
    im = np.random.default_rng(0).random((10, 40, 50))
    sizes = {}
    for bbox in ("standard", "tight"):
        with matplotlib.rc_context({"savefig.bbox": bbox}):
            w = show_zyx_max_slice_interactive(im, figsize=(8, 6))
        sizes[bbox] = mpimg.imread(io.BytesIO(base64.b64decode(w.image_data))).shape[:2]
    assert sizes["tight"] == sizes["standard"]


def test_marimo_app_file_uses_existing_api():
    """Verify ``tests/marimo_app.py`` only imports existing ``eigenp_utils`` names and valid keywords.

    Strategy: parse ``tests/marimo_app.py`` with ``ast``, resolve every
    imported ``eigenp_utils`` symbol, and check every call's keyword arguments
    against the function signature.
    Expected outcome: all imported symbols exist, every keyword is accepted,
    and deprecated ``colors=`` is not used.
    """
    tree = ast.parse((Path(__file__).parent / "marimo_app.py").read_text())
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("eigenp_utils"):
            mod = importlib.import_module(node.module)
            for alias in node.names:
                assert hasattr(mod, alias.name), f"{node.module}.{alias.name} does not exist"
                imported[alias.asname or alias.name] = getattr(mod, alias.name)
    assert imported

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in imported:
            params = inspect.signature(imported[node.func.id]).parameters
            for kw in node.keywords:
                assert kw.arg in params, f"{node.func.id}() has no keyword {kw.arg!r}"
                assert kw.arg != "colors", "use colormap=, not the deprecated colors="


@pytest.mark.skipif(sys.platform == "emscripten", reason="no subprocesses in Pyodide; covered by tests/wasm")
def test_top_level_import_does_not_require_skimage():
    """``import eigenp_utils`` must work without scikit-image installed.

    scikit-image is an optional extra (``image-analysis``); the core plotting
    API, and in particular browser/WASM environments installing only core
    dependencies, must not need it. Strategy: in a fresh interpreter, make any
    ``skimage`` import raise, then import the package and its ortho-view API.

    Expected outcome: the subprocess exits cleanly and the core names exist.
    """
    code = textwrap.dedent("""
        import sys, importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "skimage" or name.startswith("skimage."):
                    raise ImportError("skimage blocked for test")
        sys.meta_path.insert(0, Block())
        import eigenp_utils
        from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive
        assert "show_zyx" in eigenp_utils.__all__
        assert not any(m == "skimage" or m.startswith("skimage.") for m in sys.modules)
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


@pytest.mark.skipif(sys.platform == "emscripten", reason="no subprocesses in Pyodide")
def test_widget_module_runs_without_installed_package(tmp_path):
    """The module (with its .js/.css and _conventions.py next to it) imports when eigenp_utils
    itself is not importable, e.g. `uv run tnia_plotting_anywidgets.py` with its PEP 723 deps.

    (D01-D04: the shared ``_conventions.py`` is now one of the sibling files.)"""
    src = Path(tpa.__file__).parent
    for name in (
        "tnia_plotting_anywidgets.py",
        "tnia_plotting_anywidgets.js",
        "tnia_plotting_anywidgets.css",
        "iso_scatter.js",
        "_conventions.py",
    ):
        shutil.copy(src / name, tmp_path / name)
    code = textwrap.dedent(f"""
        import sys, runpy, importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "eigenp_utils" or name.startswith("eigenp_utils."):
                    raise ModuleNotFoundError(name)
        sys.meta_path.insert(0, Block())
        ns = runpy.run_path({str(tmp_path / 'tnia_plotting_anywidgets.py')!r})
        assert "TNIASliceWidget" in ns and ns["TNIAWidgetBase"]._esm
        print("ok")
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg", "PATH": ""},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


# ---------------------------------------------------------------------------
# IsoScatterWidget
# ---------------------------------------------------------------------------

def test_show_iso_scatter_renders_and_responds_to_camera():
    """The isometric viewer renders on construction and re-renders when rotated.

    `elev` and `azim` are the widget's only interactive controls, and the whole
    projection is recomputed in Python, so a missed observer would leave the
    camera sliders visibly inert. Strategy: check an image exists after
    construction, then rotate and require the image to actually change.
    """
    X, Y, Z = _iso_points()
    w = show_iso_scatter(X, Y, Z, title="iso")

    assert isinstance(w, IsoScatterWidget)
    initial = w.image_data
    assert initial

    w.azim = w.azim + 45.0
    assert w.image_data != initial


@pytest.mark.parametrize("color, expect_continuous, expect_categorical", [
    (None, False, False),
    ("continuous", True, False),
    ("categorical", False, True),
])
def test_iso_scatter_color_modes(color, expect_continuous, expect_categorical):
    """Numeric colour arrays map through a colormap; non-numeric ones map per category.

    The two paths diverge early -- continuous data is handed to Matplotlib with
    a cmap, categorical data is pre-mapped to explicit RGBA -- and choosing
    wrongly either crashes on string input or renders labels as a meaningless
    gradient. Strategy: exercise all three inputs (none, numeric, string) and
    assert both the detected mode and that rendering still succeeds.
    """
    X, Y, Z = _iso_points(n=60)
    if color == "continuous":
        color_arg = np.linspace(0.0, 1.0, X.size)
    elif color == "categorical":
        color_arg = np.array(["a", "b", "c"] * (X.size // 3))
    else:
        color_arg = None

    w = show_iso_scatter(X, Y, Z, color=color_arg, figsize=(4, 4))

    assert w.is_continuous is expect_continuous
    assert w.is_categorical is expect_categorical
    assert w.image_data


def test_iso_scatter_subsamples_above_max_points():
    """Point counts above max_points are randomly subsampled before rendering.

    Rendering is a Matplotlib 3D scatter recomputed on every camera move, so the
    cap is what keeps large point clouds usable. Strategy: pass twice the cap and
    require the retained coordinates and their colours to be trimmed together --
    trimming positions without colours would silently mis-colour every point.
    """
    X, Y, Z = _iso_points(n=400)
    color = np.arange(X.size, dtype=float)

    w = show_iso_scatter(X, Y, Z, color=color, max_points=200, figsize=(4, 4))

    assert w.X_orig.size == 200
    assert w.Y_orig.size == 200
    assert w.Z_orig.size == 200
    assert w.color.size == 200


def test_iso_scatter_handles_empty_input():
    """An empty point cloud renders a placeholder instead of raising.

    Filtering upstream can legitimately leave nothing to plot, and the widget is
    often constructed from whatever a selection produced. Strategy: build from
    empty arrays and require a rendered image plus a figure carrying the
    placeholder text, since the centroid and radius maths would otherwise divide
    by an empty reduction.
    """
    empty = np.array([], dtype=float)
    w = show_iso_scatter(empty, empty, empty, figsize=(4, 4))

    assert w.image_data

    fig = w._render()
    try:
        assert any("No Data" in t.get_text() for t in fig.texts)
    finally:
        plt.close(fig)


def test_iso_scatter_uses_pixel_sizes_for_radius_and_y():
    """The isometric viewer keeps all points inside its axes with anisotropic pixels."""
    rng = np.random.default_rng(12)
    X, Y, Z = rng.uniform(0, 100, 300), rng.uniform(0, 100, 300), rng.uniform(0, 100, 300)
    w = show_iso_scatter(X, Y, Z, pixel_sizes=(5, 1, 1))
    assert w.max_radius >= 5 * (Z.max() - Z.min()) / 2

    w2 = show_iso_scatter(X, Y, Z, pixel_sizes=(1, 3, 1))
    w2.elev, w2.azim = 0, 0
    fig = w2._render()
    try:
        top = [ax for ax in fig.axes if ax.get_title() == "Top View (X-Y)"][0]
        pts = top.collections[0].get_offsets()
        assert np.ptp(pts[:, 1]) == pytest.approx(3 * np.ptp(pts[:, 0]), rel=0.05)
        assert w2.max_radius >= np.abs(pts).max()
    finally:
        plt.close(fig)


def test_iso_scatter_subsampling_is_seeded_and_leaves_global_rng_alone():
    """Verify ``show_iso_scatter`` subsampling is deterministic and does not mutate NumPy's global RNG.

    Strategy: seed global ``np.random``, construct two widgets on 20,000 points
    with default seed, and check both global RNG continuity and identical
    retained coordinates.
    Expected outcome: global ``np.random.rand()`` matches a fresh ``RandomState(42)``
    and both widgets keep identical ``X_orig`` subsets.
    """
    rng = np.random.default_rng(0)
    X, Y, Z = rng.random(20000), rng.random(20000), rng.random(20000)
    np.random.seed(42)
    w1 = show_iso_scatter(X, Y, Z)
    assert np.random.rand() == np.random.RandomState(42).rand()
    w2 = show_iso_scatter(X, Y, Z)
    assert np.array_equal(w1.X_orig, w2.X_orig)


def test_iso_scatter_drops_nonfinite_points_with_their_colours():
    """Points with a NaN/inf coordinate are dropped (with a warning) together with their colour values.

    One non-finite coordinate used to make the centroid and radius NaN, and
    ``set_xlim`` then raised, so tracking or segmentation output with a few
    missing positions could not be shown (``TNIAScatterWidget`` already drops
    them). Strategy: put a NaN in X and an inf in Z of a 60-point cloud with a
    per-point colour array.
    Expected outcome: a warning, 58 points left with the matching 58 colours,
    a finite radius and a rendered image.
    """
    X, Y, Z = _iso_points(n=60)
    X[3] = np.nan
    Z[7] = np.inf
    color = np.arange(60.0)
    with pytest.warns(UserWarning, match="non-finite"):
        w = show_iso_scatter(X, Y, Z, color=color, figsize=(4, 4))
    assert w.X_orig.size == w.Y_orig.size == w.Z_orig.size == w.color.size == 58
    assert 3.0 not in w.color and 7.0 not in w.color
    assert np.isfinite(w.max_radius) and w.max_radius > 0
    assert w.image_data


def test_iso_scatter_single_colour_and_single_point():
    """``color='red'`` draws every point red, and one point renders without singular-limit warnings.

    A colour name (the matplotlib idiom) became a 0-d array that the
    categorical path then iterated, raising TypeError. A single point (or
    identical points) gave a zero radius, so every axis got identical limits
    and Matplotlib warned nine times. Strategy: render a cloud with
    ``color='red'`` and inspect the Top View collection; render one point
    while recording warnings.
    Expected outcome: all face colours are red; the one-point widget has a
    positive radius, an image and no 'singular' warning.
    Negative control: ``color=None`` keeps the grey default.
    """
    X, Y, Z = _iso_points(n=30)
    for color, expected in (("red", "red"), (None, "gray")):
        w = show_iso_scatter(X, Y, Z, color=color, figsize=(4, 4))
        assert not w.is_continuous and not w.is_categorical
        fig = w._render()
        try:
            top = [ax for ax in fig.axes if ax.get_title() == "Top View (X-Y)"][0]
            fc = top.collections[0].get_facecolors()
            assert np.allclose(fc[:, :3], to_rgb(expected))
        finally:
            plt.close(fig)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w1 = show_iso_scatter([1.0], [2.0], [3.0], figsize=(4, 4))
    assert w1.max_radius > 0 and w1.image_data
    assert not [c for c in caught if "singular" in str(c.message)]


def test_iso_scatter_failed_savefig_does_not_leak_a_figure(monkeypatch):
    """A render whose PNG encoding fails still closes its pyplot figure.

    Strategy: make ``Figure.savefig`` raise, rotate the camera (which
    re-renders) and compare the open pyplot figures before and after.
    Expected outcome: the error propagates and no figure is left open.
    """
    w = show_iso_scatter(*_iso_points(n=20), figsize=(4, 4))
    open_before = plt.get_fignums()

    def failing_savefig(self, *args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(Figure, "savefig", failing_savefig)
    with pytest.raises(RuntimeError, match="disk full"):
        w.azim = w.azim + 10
    assert plt.get_fignums() == open_before


# ---------------------------------------------------------------------------
# Rotation geometry (_rotate_expand, physical-space rotation, scatter parity)
# ---------------------------------------------------------------------------

def test_rotate_expand_quarter_turns_are_exact():
    """Multiples of 90 degrees reduce to exact array permutations.

    At quarter turns every output pixel centre lands on an input pixel centre,
    and an order-3 spline interpolates its nodes exactly. Strategy: compare
    against ``np.rot90`` (counter-clockwise, matching skimage's sign convention)
    for a non-square 2D image and a channels-last RGB image.

    Expected outcome: identical shapes and values equal to ~1e-9; a sign error
    in the rotation would swap 90 for 270 and fail.
    """
    rng = np.random.default_rng(0)
    img = rng.random((7, 12))
    rgb = rng.random((7, 12, 3))
    for k, angle in [(1, 90), (2, 180), (3, 270)]:
        np.testing.assert_allclose(_rotate_expand(img, angle), np.rot90(img, k), atol=1e-9)
        np.testing.assert_allclose(_rotate_expand(rgb, angle), np.rot90(rgb, k, axes=(0, 1)), atol=1e-9)


def test_rotate_expand_canvas_and_range():
    """The canvas grows to hold the rotated image and spline overshoot is clipped.

    Strategy: rotate a constant image by 45 degrees. The expanded canvas must
    follow skimage's formula ``round(|(W-1) sin| + |(H-1) cos| + 1)`` (rows) and
    the analogous columns, the rotated interior must stay at the constant value,
    and no pixel may leave ``[0, value]`` even though cubic splines overshoot
    at the hard edge of the rotated square.
    """
    h, w, a = 20, 31, 45.0
    img = np.full((h, w), 2.0)
    out = _rotate_expand(img, a)
    c, s = np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))
    assert out.shape == (
        int(np.around(abs((w - 1) * s) + abs((h - 1) * c) + 1)),
        int(np.around(abs((w - 1) * c) + abs((h - 1) * s) + 1)),
    )
    assert out.min() >= 0.0 and out.max() <= 2.0
    cy, cx = (out.shape[0] - 1) // 2, (out.shape[1] - 1) // 2
    np.testing.assert_allclose(out[cy - 2:cy + 3, cx - 2:cx + 3], 2.0, atol=1e-3)


def test_rotate_expand_matches_skimage_geometry():
    """Output geometry is identical to skimage.transform.rotate(resize=True).

    Interpolation kernels differ (skimage's 2D fast path uses its own bicubic,
    scipy a cubic B-spline), so values are compared loosely while geometry is
    compared tightly. Strategy: rotate a smooth off-centre blob by assorted
    angles and require equal output shapes and centres of mass within 0.05 px.
    Negative control: rotating by the opposite angle must move the centroid by
    many pixels, proving the centroid check can detect direction errors.
    """
    skt = pytest.importorskip("skimage.transform")
    from scipy import ndimage as ndi

    for h, w in [(37, 64), (20, 101)]:
        img = np.zeros((h, w))
        img[h // 5:h // 2, w // 6:w // 3] = 1.0
        img = ndi.gaussian_filter(img, 1.0)
        for angle in [0.5, 15, 45, 133, -20, 271.3]:
            ref = skt.rotate(img, angle, resize=True, order=3, preserve_range=True)
            out = _rotate_expand(img, angle)
            assert out.shape == ref.shape
            assert np.abs(np.subtract(ndi.center_of_mass(ref), ndi.center_of_mass(out))).max() < 0.05
            assert np.abs(ref - out).max() < 0.1 * img.max()

        ref = skt.rotate(img, 30, resize=True, order=3, preserve_range=True)
        wrong = _rotate_expand(img, -30)
        assert np.abs(np.subtract(ndi.center_of_mass(ref), ndi.center_of_mass(wrong))).max() > 3.0


def test_rotate_expand_keeps_nan_local():
    """One NaN pixel no longer blanks the whole rotated panel.

    Expected outcome: only a few pixels around the NaN are NaN; everything
    else is finite.
    """
    img = np.random.default_rng(5).uniform(0, 1, (64, 64))
    img[30, 30] = np.nan
    out = _rotate_expand(img, 10)
    assert 0 < np.isnan(out).sum() <= 4
    img[30, 30] = 0.5
    np.testing.assert_allclose(np.isfinite(_rotate_expand(img, 10)).all(), True)


def test_rotation_sense_is_the_same_for_images_and_points():
    """rotate_view turns images and scatter points the same way.

    Strategy: an object 35 units right of the XY centre, rotate_view=(90, 0, 0).
    Locate it in the image widget (rendered centroid) and in the scatter widget
    in points and density mode. Then press 'C' on the points-mode marker.

    Expected outcome: all three put it above the centre (counter-clockwise on
    screen); hover sync recovers x=85, y=50. Points mode used to rotate the
    other way and its hover inverse disagreed with its own drawing.
    """
    vol = np.zeros((1, 101, 101), dtype=np.float32)
    vol[0, 50, 85] = 1.0
    w_img = show_zyx_max_slice_interactive(vol, rotate_view=(90, 0, 0))
    fig = w_img._render()
    try:
        _cx_img, cy_img = _rendered_centroid(fig.axXY)
        centre = np.mean(fig.axXY.get_ylim())
        assert cy_img < centre - 30
    finally:
        plt.close(fig)

    X = np.array([0.0, 100.0, 85.0])
    Y = np.array([0.0, 100.0, 50.0])
    Z = np.zeros(3)
    for render in ("points", "density"):
        w = show_zyx_max_scatter_interactive(
            (Z, Y, X), render=render, rotate_view=(90, 0, 0), bins=101,
            slabs_thickness=(1000, 1000, 1000),
        )
        fig = w._render()
        try:
            centre_y = np.mean(fig.axXY.get_ylim())
            if render == "points":
                pts = np.vstack([c.get_offsets() for c in fig.axXY.collections if len(c.get_offsets())])
                marker = pts[np.argmin(np.abs(pts[:, 0] - np.mean(fig.axXY.get_xlim())))]
                assert marker[1] < centre_y - 30, (marker, centre_y)
                hx, hy = _fig_frac(fig, fig.axXY, marker[0], marker[1])
            else:
                rgb = np.asarray(fig.axXY.images[0].get_array())[..., :3].sum(axis=-1)
                _x0e, _x1e, y0e, y1e = fig.axXY.images[0].get_extent()
                h, w_px = rgb.shape
                col = rgb[:, w_px // 2 - 3:w_px // 2 + 4].max(axis=1)
                row = int(np.argmax(col))
                y_marker = y1e + (row + 0.5) * (y0e - y1e) / h
                assert y_marker < centre_y - 30, (y_marker, centre_y)
        finally:
            plt.close(fig)
        if render == "points":
            w.hover_coords = {"plane": "xy", "x": hx, "y": hy}
            assert (w.x_s, w.y_s) == (85, 50)


@pytest.mark.parametrize("plane, angle, pixel_sizes", [
    ("xy", 45.0, (1, 1, 1)),
    ("xy", 180.0, (1, 1, 1)),
    ("xy", 30.0, (1, 1, 1)),
    ("xz", 30.0, (4, 1, 1)),
    ("zy", 60.0, (4, 1, 1)),
])
def test_rotated_click_maps_to_voxel_under_cursor(plane, angle, pixel_sizes):
    """A click on a rotated panel records the voxel drawn under the cursor.

    Strategy: a smooth blob centred on a known voxel. Render the annotator with
    rotate_view, take the blob's rendered centroid (independent of the
    crosshair/click transforms), click there.

    Expected outcome: the recorded point is the blob's voxel. It used to be off
    by a voxel for isotropic data and by tens of voxels for anisotropic panels.
    """
    Z, Y, X = 40, 64, 64
    pz, py, px = pixel_sizes
    tz, ty, tx = 30, 21, 45
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    r2 = ((zz - tz) * pz) ** 2 + ((yy - ty) * py) ** 2 + ((xx - tx) * px) ** 2
    vol = np.exp(-r2 / (2 * 3.0 ** 2)).astype(np.float32)

    rot = {"xy": (angle, 0, 0), "xz": (0, angle, 0), "zy": (0, 0, angle)}[plane]
    w = show_zyx_max_slice_interactive_point_annotator(
        vol,
        pixel_sizes=pixel_sizes,
        rotate_view=rot,
        slabs_position=(tz * pz, ty * py, tx * px),
        slabs_thickness=(pz, py, px),
    )
    w.annotation_mode = True
    fig = w._render()
    try:
        ax = {"xy": fig.axXY, "xz": fig.axXZ, "zy": fig.axZY}[plane]
        cx, cy = _rendered_centroid(ax)
        click_x, click_y = _fig_frac(fig, ax, cx, cy)
    finally:
        plt.close(fig)
    w.click_coords = {"plane": plane, "x": click_x, "y": click_y}
    assert w.points == [[tz, ty, tx]], w.points


def test_rotated_anisotropic_panel_keeps_physical_aspect():
    """Rotation happens in physical space, so anisotropic panels are not distorted.

    Strategy: a 40 x 40 um XZ plane (Z=10 at 4 um, X=40 at 1 um), rotated 90
    degrees, plus the matching crosshair transform.

    Expected outcome: the panel is still 40 x 40 um (it was drawn 10 x 160 um),
    and the rotated-line helper maps the panel corners onto the panel's edges.
    """
    vol = np.random.default_rng(4).uniform(0, 1, (10, 8, 40)).astype(np.float32)
    fig = show_zyx_max_slabs(vol, [0, 40], [0, 8], [0, 10], pixel_sizes=(4, 1, 1), rotate_view=(0, 90, 0))
    try:
        assert abs(np.diff(fig.axXZ.get_xlim())[0]) == pytest.approx(40, abs=1)
        assert abs(np.diff(fig.axXZ.get_ylim())[0]) == pytest.approx(40, abs=1)
        xs, ys = _get_rotated_line(0, 0, 40, 40, 90, 40, 10, 1, 4)
        np.testing.assert_allclose([xs[0], ys[0]], [0, 40], atol=1)
        np.testing.assert_allclose([xs[1], ys[1]], [40, 0], atol=1)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_rotated_integer_rgb_panel_keeps_its_colours(dtype):
    """Rotating a pre-composited integer RGB panel keeps imshow's 0..255 scale.

    Strategy: an RGB panel with red = 60 everywhere, shown by ``show_zyx``
    without rotation, rotated by 30 degrees, and rotated in the white theme
    (``blending='minimum'``).

    Expected outcome: unrotated, imshow gets the integer panel unchanged (red
    60, which it reads as 60/255). Rotated, the panel is float with red 60/255
    inside the image instead of being clipped to 1.0 (saturated), and the
    corners added by the rotation are black (white in the white theme).
    """
    rgb = np.zeros((20, 20, 3), dtype=dtype)
    rgb[..., 0] = 60
    fig = show_zyx(rgb, rgb[:5], rgb[:, :5])
    try:
        xy = np.asarray(fig.axXY.images[0].get_array())
        assert np.issubdtype(xy.dtype, np.integer)
        np.testing.assert_array_equal(xy[10, 10], [60, 0, 0])
    finally:
        plt.close(fig)

    for blending, corner in ((None, [0, 0, 0]), ("minimum", [1, 1, 1])):
        fig = show_zyx(rgb, rgb[:5], rgb[:, :5], rotate_view=30, blending=blending)
        try:
            xy = np.asarray(fig.axXY.images[0].get_array())
            assert np.issubdtype(xy.dtype, np.floating)
            h, w = xy.shape[:2]
            np.testing.assert_allclose(xy[h // 2, w // 2], [60 / 255, 0, 0], atol=1e-3)
            assert xy[..., 0].max() <= 60 / 255 + 1e-3 or blending == "minimum"
            np.testing.assert_allclose(xy[0, 0], corner, atol=1e-3)
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Slab indexing, scatter density/points geometry, and display range invariants
# ---------------------------------------------------------------------------

def test_slab_end_is_inclusive_so_last_plane_is_projected():
    """The range slider's end is inclusive, so the last plane is part of the slab.

    Strategy: put signal only in the last z-plane. A 'full' projection
    (thickness beyond the volume) and an explicit [0, Z-1] range must both show
    it in the XY panel; so must the one-plane range [Z-1, Z-1].

    Expected outcome: the XY panel's maximum is the signal value in all three
    cases (it used to be 0: the end index was treated as exclusive).
    """
    Z, Y, X = 6, 16, 16
    vol = np.zeros((Z, Y, X), dtype=np.float32)
    vol[-1, 8, 8] = 7.0

    w = show_zyx_max_slice_interactive(vol, slabs_thickness=(1000, 1000, 1000))
    for z_start, z_end in [(None, None), (0, Z - 1), (Z - 1, Z - 1)]:
        if z_start is not None:
            w.z_start, w.z_end = z_start, z_end
        fig = w._render()
        try:
            assert fig.axXY.images[0].get_array().max() == pytest.approx(7.0), (z_start, z_end)
        finally:
            plt.close(fig)


def test_scatter_density_ids_with_single_colormap_name():
    """Integer ids coloured by one colormap name render in density mode too.

    Strategy: 4 cluster ids with colormap='viridis' (points mode samples one
    colour per id from the colormap).

    Expected outcome: construction succeeds (it raised IndexError) and the four
    clusters are drawn in four different colours.
    """
    rng = np.random.default_rng(0)
    n = 4000
    ids = np.repeat(np.arange(4), n // 4)
    X = ids * 25 + rng.uniform(0, 10, n)
    Y = rng.uniform(0, 100, n)
    Z = rng.uniform(0, 10, n)
    w = show_zyx_max_scatter_interactive(
        (Z, Y, X), channels=ids, colormap="viridis", render="density",
        slabs_thickness=(1000, 1000, 1000),
    )
    fig = w._render()
    try:
        im = fig.axXY.images[0]
        rgb = np.asarray(im.get_array())[..., :3]
        w_px = rgb.shape[1]
        left, right = im.get_extent()[:2]
        col_of = lambda x: int((x - (w.xmin - 0.5) - left) / (right - left) * w_px)
        cols = [rgb[:, col_of(c * 25 + 5)].max(axis=0) for c in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                assert np.abs(cols[i] - cols[j]).max() > 0.05, (i, j, cols)
    finally:
        plt.close(fig)


def test_scatter_density_panels_scale_bar_and_hover_use_data_units():
    """Density panels are drawn in data units, like points mode.

    Strategy: X, Y in [0, 1000], Z in [0, 20], isotropic pixels. Check the
    physical extent of each panel, that the scale bar spans the length it is
    labelled with, and that hover sync ('C') at data (750, 250) on the XY panel
    selects that position.

    Expected outcome: XY spans ~1001 x 1001, XZ ~1001 x 21, ZY ~21 x 1001; the
    bar spans as many data units as its label says; hover gives x_s, y_s within one bin (2 units)
    of (750, 250). Before the fix every panel was a 512 x 512 bin square.
    """
    rng = np.random.default_rng(1)
    n = 20000
    X = rng.uniform(0, 1000, n)
    Y = rng.uniform(0, 1000, n)
    Z = rng.uniform(0, 20, n)
    X[0], Y[0], Z[0] = 0, 0, 0
    w = show_zyx_max_scatter_interactive((Z, Y, X), render="density", pixel_sizes=(1, 1, 1))
    fig = w._render()
    try:
        def span(ax):
            return abs(np.diff(ax.get_xlim())[0]), abs(np.diff(ax.get_ylim())[0])

        np.testing.assert_allclose(span(fig.axXY), (1001, 1001), rtol=1e-6)
        np.testing.assert_allclose(span(fig.axXZ), (1001, 21), rtol=1e-6)
        np.testing.assert_allclose(span(fig.axZY), (21, 1001), rtol=1e-6)

        label = [t.get_text() for t in fig.axXY.texts][0]
        bar = fig.axXY.collections[-1].get_segments()[0]
        bar_units = (bar[1][0] - bar[0][0]) * span(fig.axXY)[0]
        assert bar_units == pytest.approx(float(label.split()[0]), rel=1e-6)

        hx, hy = _fig_frac(fig, fig.axXY, 750 - w.xmin + 0.5, 250 - w.ymin + 0.5)
    finally:
        plt.close(fig)
    w.hover_coords = {"plane": "xy", "x": hx, "y": hy}
    assert abs(w.x_s - (750 - w.xmin)) <= 2 and abs(w.y_s - (250 - w.ymin)) <= 2, (w.x_s, w.y_s)


def test_scatter_density_bin_colour_encodes_value_not_sum():
    """A density bin shows the value of its points, not their sum.

    Strategy: 50 coincident points of value 0.1 and one point of value 1.0
    elsewhere (vmax=1). Points mode draws them at 10% and 100%.

    Expected outcome: the 0.1 bin is at ~10% brightness and the 1.0 bin at full
    brightness (both used to be white).
    """
    X = np.r_[np.full(50, 20.0), 80.0, 0.0, 99.0]
    Y = np.r_[np.full(50, 20.0), 80.0, 0.0, 99.0]
    Z = np.zeros_like(X)
    vals = np.r_[np.full(50, 0.1), 1.0, 0.0, 0.0]
    w = show_zyx_max_scatter_interactive(
        (Z, Y, X), channels=vals, render="density", bins=100, vmin=0.0, vmax=1.0
    )
    fig = w._render()
    try:
        rgb = np.asarray(fig.axXY.images[0].get_array())[..., :3]
        assert rgb[20, 20].max() == pytest.approx(0.1, abs=0.02)
        assert rgb[80, 80].max() == pytest.approx(1.0, abs=0.02)
    finally:
        plt.close(fig)


def test_opacity_of_one_channel_does_not_fade_composites():
    """Per-channel opacity is baked into composites, not applied globally.

    Strategy: (a) show_zyx with a pre-composited RGB image and a list opacity;
    (b) scatter density with two continuous channels where channel 0 is hidden
    (opacity 0) and channel 1 fully opaque.

    Expected outcome: no global alpha on the image (it was channel 0's value),
    and channel 1 is visible in the density composite.
    """
    rgb = np.random.default_rng(2).uniform(0, 1, (8, 8, 3)).astype(np.float32)
    fig = show_zyx(rgb, rgb, rgb, opacity=[0.2, 1.0])
    try:
        assert fig.axXY.images[0].get_alpha() is None
    finally:
        plt.close(fig)

    rng = np.random.default_rng(3)
    n = 2000
    X, Y, Z = rng.uniform(0, 50, n), rng.uniform(0, 50, n), rng.uniform(0, 5, n)
    a, b = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    w = show_zyx_max_scatter_interactive(
        (Z, Y, X), channels=[a, b], render="density",
        opacity=[0.0, 1.0], slabs_thickness=(1000, 1000, 1000),
    )
    fig = w._render()
    try:
        im = fig.axXY.images[0]
        assert im.get_alpha() in (None, 1.0)
        assert np.asarray(im.get_array())[..., 1].max() > 0.5  # lime channel present
    finally:
        plt.close(fig)


@pytest.mark.parametrize("gamma", [1, 0.5])
def test_single_channel_panels_share_one_intensity_scale(gamma):
    """All three ortho panels use one display range.

    Strategy: plane z=2 is dim (0-10), the rest bright (0-1000).

    Expected outcome: XY, XZ and ZY have the same (vmin, vmax), spanning the
    whole volume, for gamma 1 and gamma != 1.
    """
    rng = np.random.default_rng(6)
    vol = rng.uniform(0, 1000, (8, 32, 32)).astype(np.float32)
    vol[2] = rng.uniform(0, 10, (32, 32))
    fig = show_zyx_slice(vol, 16, 16, 2, gamma=gamma)
    try:
        clims = [ax.images[0].norm for ax in (fig.axXY, fig.axXZ, fig.axZY)]
        lims = {(round(n.vmin, 3), round(n.vmax, 3)) for n in clims}
        assert len(lims) == 1, lims
        (_lo, hi), = lims
        assert hi > 900
    finally:
        plt.close(fig)


def test_multichannel_composite_ignores_nonfinite_values():
    """A NaN or inf pixel neither saturates its channel nor leaks into the RGB output."""
    rng = np.random.default_rng(7)
    a = rng.uniform(0, 1000, (16, 16)).astype(np.float32)
    b = rng.uniform(0, 1000, (16, 16)).astype(np.float32)
    a[3, 3] = np.nan
    a[4, 4] = np.inf
    xy, _xz, _zy = create_multichannel_rgb([a, b], [a, b], [a, b], colormap=["red", "lime"])
    assert np.isfinite(xy).all()
    assert (xy[..., 0] > 0.99).mean() < 0.1


def test_blend_colors_scalar_gamma():
    """Verify ``blend_colors`` broadcasts a scalar ``gamma`` across channels.

    Strategy: blend a linear ``[0, 1]`` ramp with ``gamma=0.5`` on ``['white']``.
    Expected outcome: output equals ``sqrt(I)``.
    """
    I = np.linspace(0, 1, 5)[:, None]
    out = blend_colors(I, ["white"], vmin=0, vmax=1, gamma=0.5)
    np.testing.assert_allclose(out[:, 0], np.sqrt(I[:, 0]))


def test_points_mode_scales_y_by_sy():
    """Points mode uses the Y pixel size for Y (it used the X one).

    Strategy: pixel_sizes=(1, 2, 1); hover a marker at y=60 on the XY panel.

    Expected outcome: y_s == 60, and the XY panel spans twice as far in Y as
    in X for a square point cloud.
    """
    X = np.array([0.0, 99.0, 30.0])
    Y = np.array([0.0, 99.0, 60.0])
    Z = np.array([0.0, 9.0, 5.0])
    w = show_zyx_max_scatter_interactive((Z, Y, X), pixel_sizes=(1, 2, 1), slabs_thickness=(1000, 1000, 1000))
    fig = w._render()
    try:
        assert abs(np.diff(fig.axXY.get_ylim())[0]) == pytest.approx(2 * abs(np.diff(fig.axXY.get_xlim())[0]))
        hx, hy = _fig_frac(fig, fig.axXY, 30 * 1, 60 * 2)
    finally:
        plt.close(fig)
    w.hover_coords = {"plane": "xy", "x": hx, "y": hy}
    assert (w.x_s, w.y_s) == (30, 60)


def test_points_mode_scale_bar_matches_rotated_panel():
    """The points-mode scale bar spans the length it is labelled with when rotated."""
    rng = np.random.default_rng(10)
    X, Y = rng.uniform(0, 99, 500), rng.uniform(0, 99, 500)
    X[:2], Y[:2] = [0, 99], [0, 99]
    Z = np.zeros_like(X)
    w = show_zyx_max_scatter_interactive((Z, Y, X), pixel_sizes=(1, 1, 1), rotate_view=(45, 0, 0))
    fig = w._render()
    try:
        label = _scale_bar_artists(fig)[1].get_text()
        seg = [c for c in fig.axXY.collections if hasattr(c, "get_segments")][-1].get_segments()[0]
        bar_units = (seg[1][0] - seg[0][0]) * abs(np.diff(fig.axXY.get_xlim())[0])
        assert bar_units == pytest.approx(float(label.split()[0]), rel=1e-6)
    finally:
        plt.close(fig)


def test_scatter_one_plane_slab_shows_the_points_in_its_band():
    """A start == end slab shows the points inside the drawn one-unit band (centred on the index)."""
    rng = np.random.default_rng(11)
    n = 5000
    X, Y, Z = rng.uniform(0, 100, n), rng.uniform(0, 100, n), rng.uniform(0, 10, n)
    w = show_zyx_max_scatter_interactive((Z, Y, X))
    w.x_start, w.x_end = 50, 50
    fig = w._render()
    try:
        x0 = w.xmin + 50
        expected = int(((X >= x0 - 0.5) & (X < x0 + 0.5)).sum())
        shown = sum(len(c.get_offsets()) for c in fig.axZY.collections if hasattr(c, "get_offsets"))
        assert expected > 0 and shown == expected
    finally:
        plt.close(fig)


def test_multichannel_default_vmin_shows_negative_channels():
    """vmin=None is 0 for non-negative data and the data minimum for negative data."""
    rng = np.random.default_rng(0)
    neg = rng.uniform(-5, -1, (8, 8))
    pos = rng.uniform(0, 10, (8, 8))
    for fn in (create_multichannel_rgb, create_multichannel_rgb_cmap):
        xy, _, _ = fn([neg, pos], [neg, pos], [neg, pos], colormap=["red", "blue"], blend="max")
        red = xy[..., 0]
        assert red.max() > 0.5 and np.ptp(red) > 0.5
        assert xy[..., 2].max() == pytest.approx(1.0, abs=1e-2)
        with pytest.raises(ValueError, match="gamma"):
            fn([neg, pos], [neg, pos], [neg, pos], gamma=[1, 0])


def test_short_per_channel_lists_are_padded_and_long_ones_raise():
    """Per-channel lists shorter than the channel count take the default for the missing channels.

    Strategy: compose two channels with one-entry ``vmin``, ``vmax``, ``gamma``
    and ``opacity`` lists in ``create_multichannel_rgb`` and
    ``create_multichannel_rgb_cmap`` (and a one-entry ``gamma`` in
    ``blend_colors``), and compare with the lists padded by hand; pass three
    entries for two channels; render ``show_zyx_max`` with a short ``vmax``
    list, also with 'attenuated_mip', which reads vmin/vmax per channel.

    Expected outcome: a short list gives the same output as the list padded
    with the parameter's default (None for vmin/vmax, 1 for gamma and opacity)
    instead of an IndexError; a longer list raises a ValueError naming the
    parameter.
    """
    rng = np.random.default_rng(0)
    a, b = rng.uniform(0, 10, (2, 8, 8))
    cases = [("vmin", [1.0], [1.0, None]), ("vmax", [5.0], [5.0, None]),
             ("gamma", [0.5], [0.5, 1]), ("opacity", [0.5], [0.5, 1.0])]
    for fn in (create_multichannel_rgb, create_multichannel_rgb_cmap):
        for name, short, padded in cases:
            out = fn([a, b], [a, b], [a, b], colormap=["red", "lime"], **{name: short})
            ref = fn([a, b], [a, b], [a, b], colormap=["red", "lime"], **{name: padded})
            for o, r in zip(out, ref):
                np.testing.assert_array_equal(o, r)
            with pytest.raises(ValueError, match=f"{name} has 3 entries for 2 channels"):
                fn([a, b], [a, b], [a, b], **{name: padded + [1.0]})

    vals = np.column_stack([a.ravel(), b.ravel()])
    np.testing.assert_array_equal(
        blend_colors(vals, ["red", "lime"], gamma=[0.5]),
        blend_colors(vals, ["red", "lime"], gamma=[0.5, 1.0]),
    )
    with pytest.raises(ValueError, match="gamma has 3 entries for 2 channels"):
        blend_colors(vals, ["red", "lime"], gamma=[0.5, 1.0, 1.0])

    vol = rng.uniform(0, 10, (2, 4, 6, 6))
    for projection in (None, "attenuated_mip"):
        f0 = show_zyx_max([vol[0], vol[1]], vmin=[0.0], vmax=[5.0], projection=projection)
        f1 = show_zyx_max([vol[0], vol[1]], vmin=[0.0, None], vmax=[5.0, None], projection=projection)
        try:
            np.testing.assert_array_equal(_panel_rgb(f0), _panel_rgb(f1))
        finally:
            plt.close(f0)
            plt.close(f1)


def test_annotator_point_outside_volume_paints_nothing():
    """Verify adding points outside the volume bounds leaves the annotation mask unchanged.

    Strategy: add one interior point via ``show_zyx_max_slice_interactive_point_annotator``
    and record the painted voxel count, then add two out-of-bounds points.
    Expected outcome: painted voxel count in ``_annot_img`` does not change and
    the out-of-bounds points are dropped with a warning.
    """
    w = show_zyx_max_slice_interactive_point_annotator(np.zeros((20, 64, 64), dtype=np.uint8))
    w.add_point(10, 30, 30)
    inside = int((w._annot_img > 0).sum())
    assert inside > 0
    with pytest.warns(UserWarning, match="outside the volume"):
        w.add_point(-5, 30, 30)
    with pytest.warns(UserWarning, match="outside the volume"):
        w.add_point(10, 80, 30)
    assert int((w._annot_img > 0).sum()) == inside
    assert w.points == [[10, 30, 30]]


def test_annotator_points_trait_normalises_tuples_arrays_and_floats():
    """``points`` stores ``[z, y, x]`` lists of ints whatever row type it is given.

    The trait used to accept anything: points set as tuples (rows read back from
    a CSV) never matched the ``[z, y, x]`` lists that ``remove_point`` and
    ``add_point`` build, so they could not be deleted and were duplicated; an
    (N, 3) array was rejected; float rows (``np.loadtxt``) were stored and made
    the painting observer slice with floats, raising on that and every later
    change; points outside the volume were kept (and saved) but never drawn.
    Strategy: set tuples, an integral float array and invalid points.
    Expected outcome: tuples become int lists that remove/add match; the array
    is accepted and painted; a non-integral or two-coordinate point is refused
    with a TraitError (the old value stays); out-of-volume points are dropped
    with a warning. Negative control: int lists are stored as given.
    """
    w = show_zyx_max_slice_interactive_point_annotator(np.zeros((20, 64, 64), dtype=np.uint8))
    w.points = [[5, 10, 10], [6, 7, 8]]
    assert w.points == [[5, 10, 10], [6, 7, 8]]

    w.points = [(5, 10, 10), (5, 20, 20)]
    assert w.points == [[5, 10, 10], [5, 20, 20]]
    assert all(type(v) is int for p in w.points for v in p)
    w.remove_point(5, 10, 10)
    assert w.points == [[5, 20, 20]]
    w.add_point(5, 20, 20)
    assert w.points == [[5, 20, 20]]

    w.points = np.array([[3.0, 4.0, 5.0]])
    assert w.points == [[3, 4, 5]]
    assert w._annot_img[3, 4, 5] == 255

    with pytest.raises(traitlets.TraitError, match="integer"):
        w.points = [[3.5, 4, 5]]
    with pytest.raises(traitlets.TraitError, match="integer"):
        w.points = [[1, 2]]
    assert w.points == [[3, 4, 5]]

    with pytest.warns(UserWarning, match="outside the volume"):
        w.points = [[3, 4, 5], [-1, 4, 5], [3, 64, 5]]
    assert w.points == [[3, 4, 5]]


def test_annotator_click_in_rotated_padding_adds_no_point():
    """A click on the empty corner of a rotated panel adds nothing.

    ``rotate_view`` pads each panel with empty corners; a click there maps to
    indices outside the volume, which used to be clamped onto the border and
    added as a real point. Strategy: rotate the XY panel by 45 degrees and
    click near the top-left corner of its axes (padding).
    Expected outcome: no point. Negative control: a click at the panel centre
    (inside the image) still adds one point.
    """
    w = show_zyx_max_slice_interactive_point_annotator(
        np.zeros((10, 64, 64), dtype=np.uint8), rotate_view=(45, 0, 0), figsize=(4, 4)
    )
    w.annotation_mode = True
    info = w.axis_bounds["xy"]

    def at(fx, fy, t):
        return {
            "plane": "xy",
            "x": info["x0"] + fx * (info["x1"] - info["x0"]),
            "y": info["y0_js"] + fy * (info["y1_js"] - info["y0_js"]),
            "t": t,
        }

    w.click_coords = at(0.03, 0.03, 1)
    assert w.points == []
    w.click_coords = at(0.5, 0.5, 2)
    assert len(w.points) == 1


def test_scatter_widget_drops_nonfinite_points_and_accepts_empty_input():
    """Verify scatter viewers drop NaN/inf points with a warning, re-index ``idx_lists``, and handle empty clouds.

    Strategy: inject one ``nan`` and one ``inf`` into 30 points with both
    continuous and index-list channels, then test empty point clouds in both
    ``'points'`` and ``'density'`` modes.
    Expected outcome: emits a warning naming 2 dropped points, retains 28
    aligned values, re-indexes ``idx_lists``, and renders empty inputs cleanly.
    """
    rng = np.random.default_rng(0)
    X, Y, Z = rng.uniform(0, 50, 30), rng.uniform(0, 50, 30), rng.uniform(0, 10, 30)
    vals = rng.random(30)
    X[3] = np.nan
    Z[7] = np.inf
    with pytest.warns(UserWarning, match="Dropping 2 of 30"):
        w = show_zyx_max_scatter_interactive((Z, Y, X), channels=vals)
    assert w.N == 28 and len(w.cont_single) == 28
    assert w.cont_single[3] == vals[4] and w.image_data

    with pytest.warns(UserWarning):
        w = show_zyx_max_scatter_interactive((Z, Y, X), channels=[np.array([0, 3, 4]), np.array([7, 8])])
    assert [list(ix) for ix in w.idx_lists] == [[0, 3], [6]]

    for render in ("points", "density"):
        w = show_zyx_max_scatter_interactive((np.array([]), np.array([]), np.array([])), render=render)
        assert w.image_data


def test_scatter_widget_accepts_n_by_c_channel_array():
    """Verify ``show_zyx_max_scatter_interactive`` accepts an ``(N, C)`` channel array and rejects mismatched row counts.

    Strategy: pass a ``(150, 2)`` array for 150 points, then pass ``(149, 2)``.
    Expected outcome: valid input enters ``cont_multi`` mode with ``C == 2``
    and renders; mismatched rows raise ``ValueError``.
    """
    rng = np.random.default_rng(5)
    pts = rng.random((150, 3)) * [8, 20, 30]
    vals = rng.random((150, 2))
    w = show_zyx_max_scatter_interactive(pts, channels=vals, render="points")
    assert w.mode == "cont_multi" and w.C == 2
    plt.close(w._render())
    with pytest.raises(ValueError, match="n_points"):
        show_zyx_max_scatter_interactive(pts, channels=rng.random((149, 2)), render="points")


def test_show_zyx_max_accepts_a_channel_list():
    """Verify ``show_zyx_max`` accepts a list of 3D channel arrays and composites them to RGB.

    Strategy: pass two ``(4, 5, 6)`` arrays with ``colormap=['magenta', 'cyan']``.
    Expected outcome: XY panel has shape ``(5, 6, 3)``.
    """
    rng = np.random.default_rng(6)
    chans = [rng.random((4, 5, 6)).astype(np.float32) for _ in range(2)]
    fig = show_zyx_max(chans, colormap=["magenta", "cyan"])
    try:
        assert _panel_rgb(fig).shape == (5, 6, 3)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Blending modes and slab projections (keyword-only rendering options)
# ---------------------------------------------------------------------------

def test_minimum_blending_is_per_rgb_minimum_of_white_to_colour_channels():
    """Verify ``blend='minimum'`` maps each channel white -> color and takes the per-RGB minimum.

    Strategy: compose orthogonal magenta and cyan planes with full and partial
    opacities.
    Expected outcome: zero signal is white, individual channels show their exact
    colors, overlap is blue ``min(magenta, cyan)``, and opacity fades towards white.
    """
    a, b = _two_channel_planes()
    xy, _, _ = create_multichannel_rgb(
        [a, b], [a, b], [a, b], vmin=0, vmax=1, colormap=["magenta", "cyan"], blend="minimum"
    )
    np.testing.assert_allclose(xy[1, 1], [1, 1, 1])        # both off: white
    np.testing.assert_allclose(xy[1, 0], to_rgb("magenta"))  # only channel 0
    np.testing.assert_allclose(xy[0, 1], to_rgb("cyan"))     # only channel 1
    np.testing.assert_allclose(xy[0, 0], [0, 0, 1])        # min(magenta, cyan) = blue

    xy, _, _ = create_multichannel_rgb(
        [a, b], [a, b], [a, b], vmin=0, vmax=1,
        colormap=["magenta", "cyan"], opacity=[0.5, 0.0], blend="minimum",
    )
    np.testing.assert_allclose(xy[1, 0], [1, 0.5, 1])
    np.testing.assert_allclose(xy[0, 1], [1, 1, 1])


def test_minimum_blending_with_inverted_colormaps():
    """Verify ``create_multichannel_rgb_cmap(..., blend='minimum')`` composes inverted colormaps and plain colors.

    Strategy: compose ``['Blues', 'Reds']`` and ``['Blues', 'red']`` on
    orthogonal test planes and compare against 8-bit colormap lookups.
    Expected outcome: pixels match the per-channel RGB minimum of the lookup
    tables, with plain colors drawn white -> color.
    """
    a, b = _two_channel_planes()
    xy, _, _ = create_multichannel_rgb_cmap(
        [a, b], [a, b], [a, b], vmin=0, vmax=1, colormap=["Blues", "Reds"], blend="minimum"
    )
    blues, reds = matplotlib.colormaps["Blues"], matplotlib.colormaps["Reds"]
    q = lambda c, v: np.asarray(c(np.array([v]), bytes=True)[0, :3]) / 255.0
    np.testing.assert_allclose(xy[1, 1], np.minimum(q(blues, 0.0), q(reds, 0.0)), atol=1e-6)
    np.testing.assert_allclose(xy[0, 0], np.minimum(q(blues, 1.0), q(reds, 1.0)), atol=1e-6)

    xy, _, _ = create_multichannel_rgb_cmap(
        [a, b], [a, b], [a, b], vmin=0, vmax=1, colormap=["Blues", "red"], blend="minimum"
    )
    np.testing.assert_allclose(xy[1, 1], q(blues, 0.0), atol=1e-6)


def test_compositor_default_colours_follow_the_blend_mode():
    """Called directly with ``blend='minimum'``, the compositors use the inverted default colours.

    Strategy: compose channel 0 fully on and channel 1 off without a
    ``colormap`` through ``create_multichannel_rgb`` and
    ``create_multichannel_rgb_cmap``, with 'minimum' and 'additive' blending,
    and compare with ``show_zyx(..., blending='minimum')``.

    Expected outcome: with 'minimum', channel 0 is drawn in the first inverted
    default colour (black), exactly as ``show_zyx`` draws it, instead of
    white on white (invisible); with 'additive' the defaults are unchanged
    (channel 0 white).
    """
    on, off = np.ones((2, 2)), np.zeros((2, 2))
    fig = show_zyx([on, off], [on, off], [on, off], blending="minimum")
    try:
        via_show_zyx = _panel_rgb(fig)
    finally:
        plt.close(fig)
    for fn in (create_multichannel_rgb, create_multichannel_rgb_cmap):
        xy, _, _ = fn([on, off], [on, off], [on, off], blend="minimum")
        np.testing.assert_allclose(xy[0, 0], [0, 0, 0])
        np.testing.assert_allclose(xy, via_show_zyx, atol=1e-6)
        xy, _, _ = fn([on, off], [on, off], [on, off], blend="additive")
        np.testing.assert_allclose(xy[0, 0], [1, 1, 1])


def test_default_blending_is_unchanged():
    """Verify ``blending=None`` stays pixel-identical to additive (colors) and maximum (colormaps).

    Strategy: compare ``show_zyx`` with ``blending=None`` against explicit
    ``'additive'`` for color names and ``'maximum'`` for colormap names.
    Expected outcome: rendered XY arrays are bit-for-bit equal.
    """
    rng = np.random.default_rng(0)
    planes = [rng.random((6, 7)).astype(np.float32) for _ in range(2)]
    for cmap, mode in ((["magenta", "cyan"], "additive"), (["viridis", "magma"], "maximum")):
        f0 = show_zyx(planes, planes, planes, colormap=cmap)
        f1 = show_zyx(planes, planes, planes, colormap=cmap, blending=mode)
        try:
            np.testing.assert_array_equal(_panel_rgb(f0), _panel_rgb(f1))
        finally:
            plt.close(f0)
            plt.close(f1)


def test_minimum_blending_uses_a_white_theme():
    """Verify ``blending='minimum'`` switches figure/panel backgrounds, rotation padding, and scale bar to white theme.

    Strategy: render a rotated two-channel figure with ``blending='minimum'``
    and a default control figure.
    Expected outcome: minimum blending pads corners white, sets figure and XY
    facecolors to white, and draws a black scale bar; the default pads black
    with a white scale bar.
    """
    rng = np.random.default_rng(0)
    planes = [rng.random((20, 30)).astype(np.float32) for _ in range(2)]
    fig = show_zyx(planes, planes, planes, rotate_view=30, blending="minimum", pixel_sizes=(1, 1, 1))
    try:
        xy = _panel_rgb(fig)
        np.testing.assert_allclose(xy[0, 0], [1, 1, 1], atol=1e-6)
        assert mcolors.to_hex(fig.patch.get_facecolor()) == "#ffffff"
        assert mcolors.to_hex(fig.axXY.get_facecolor()) == "#ffffff"
        bar = fig.axXY.collections[0]
        assert mcolors.to_hex(bar.get_colors()[0]) == "#000000"
    finally:
        plt.close(fig)

    fig = show_zyx(planes, planes, planes, rotate_view=30, pixel_sizes=(1, 1, 1))
    try:
        np.testing.assert_allclose(_panel_rgb(fig)[0, 0], [0, 0, 0], atol=1e-6)
        assert mcolors.to_hex(fig.axXY.collections[0].get_colors()[0]) == "#ffffff"
    finally:
        plt.close(fig)


def test_minimum_blending_single_channel_and_default_colours():
    """Verify single-channel ``blending='minimum'`` draws black ink on white and uses inverted default colors.

    Strategy: render a single bright voxel with ``blending='minimum'`` and
    check ``_channel_colormaps(None, 3, 'minimum')`` vs default.
    Expected outcome: background maps to white, signal to black, and default
    palette is ``['black', 'magenta', 'lime']``.
    """
    img = np.zeros((4, 5, 6), dtype=np.float32)
    img[2, 2, 3] = 1.0
    fig = show_zyx_max(img, blending="minimum", vmin=0, vmax=1)
    try:
        rgba = fig.axXY.images[0].to_rgba(fig.axXY.images[0].get_array())
        np.testing.assert_allclose(rgba[0, 0, :3], [1, 1, 1], atol=1e-6)
        np.testing.assert_allclose(rgba[2, 3, :3], [0, 0, 0], atol=1e-6)
    finally:
        plt.close(fig)
    assert _channel_colormaps(None, 3, "minimum") == ["black", "magenta", "lime"]
    assert _channel_colormaps(None, 3) == ["white", "lime", "magenta"]


def test_blend_colors_minimum():
    """Verify ``blend_colors(..., blending='minimum')`` matches per-RGB minimum and default equals additive.

    Strategy: evaluate ``blend_colors`` on zero, single-channel, and overlap
    rows for ``['magenta', 'cyan']``.
    Expected outcome: minimum produces white, magenta, and blue; default matches
    explicit ``blending='additive'``.
    """
    vals = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    cols = blend_colors(vals, ["magenta", "cyan"], vmin=0, vmax=1, blending="minimum")
    np.testing.assert_allclose(cols, [[1, 1, 1], to_rgb("magenta"), [0, 0, 1]])
    np.testing.assert_allclose(
        blend_colors(vals, ["magenta", "cyan"], vmin=0, vmax=1),
        blend_colors(vals, ["magenta", "cyan"], vmin=0, vmax=1, blending="additive"),
    )


@pytest.mark.parametrize("colours", [["red", "lime"], ["viridis", "lime"]])
@pytest.mark.parametrize("blending", ["additive", "maximum", "screen", "minimum"])
def test_blend_colors_nan_in_one_channel_keeps_the_other_channels(blending, colours):
    """A NaN value of one channel counts as no signal instead of blackening the point.

    Strategy: blend three points where the second has no value (NaN) in
    channel 0 and the third none in channel 1, with the default range and with
    the same range given explicitly; then add an all-NaN third channel. Each
    is compared with the same points where the NaN is replaced by the
    channel's vmin. Warnings are errors.

    Expected outcome: the colours are finite and equal to the reference (the
    other channel stays visible), the default range is still each channel's
    finite min/max, and the all-NaN channel adds nothing and does not warn.
    """
    vals = np.array([[0.0, 1.0], [np.nan, 2.0], [5.0, np.nan]])
    no_signal = np.array([[0.0, 1.0], [0.0, 2.0], [5.0, 1.0]])
    rng = dict(vmin=[0.0, 1.0], vmax=[5.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ref = blend_colors(no_signal, colours, blending=blending, **rng)
        np.testing.assert_allclose(blend_colors(vals, colours, blending=blending, **rng), ref)
        np.testing.assert_allclose(blend_colors(vals, colours, blending=blending), ref)
        with_empty = np.column_stack([vals, np.full(3, np.nan)])
        np.testing.assert_allclose(blend_colors(with_empty, colours + ["blue"], blending=blending), ref)
    if colours == ["red", "lime"] and blending in ("additive", "maximum", "screen"):
        np.testing.assert_allclose(ref, [[0, 0, 0], [0, 1, 0], [1, 0, 0]])


def test_invalid_blending_and_projection_raise():
    """Verify invalid ``blending``, ``projection``, and negative ``attenuation`` raise ``ValueError``.

    Strategy: pass unsupported mode names and ``attenuation=-1`` to ``show_zyx_max``.
    Expected outcome: each call raises ``ValueError`` naming the offending argument.
    """
    img = np.zeros((3, 4, 5))
    with pytest.raises(ValueError, match="blending"):
        show_zyx_max(img, blending="multiply")
    with pytest.raises(ValueError, match="projection"):
        show_zyx_max(img, projection="iso")
    with pytest.raises(ValueError, match="attenuation"):
        show_zyx_max(img, projection="attenuated_mip", attenuation=-1)


@pytest.mark.parametrize("projection, ref", [("mip", np.max), ("minip", np.min), ("average", np.mean)])
def test_slab_projections_match_numpy(projection, ref):
    """Verify named slab projections ``mip``, ``minip``, and ``average`` match NumPy reductions.

    Strategy: project a seeded random volume with ``show_zyx_max_slabs`` over
    ``z=[3, 6]`` for each named projection mode.
    Expected outcome: XY panel equals ``ref(img[3:6], axis=0)`` within ``1e-6``.
    """
    rng = np.random.default_rng(1)
    img = rng.random((8, 9, 10)).astype(np.float32)
    fig = show_zyx_max_slabs(img, [2, 7], [1, 5], [3, 6], projection=projection, vmin=0, vmax=1)
    try:
        np.testing.assert_allclose(_panel_rgb(fig), ref(img[3:6], axis=0), rtol=1e-6)
    finally:
        plt.close(fig)


def test_projections_ignore_nan_voxels():
    """NaN voxels along a ray do not hide the valid signal behind them.

    Strategy: a zero volume with NaN over half of plane z=0, an all-NaN ray at
    (y, x) = (0, 0), and a bead of 100 at (z, y, x) = (5, 10, 5) behind the
    NaN plane. Project it with ``show_zyx_max`` (default MIP, alone and as a
    channel of a composite), with the named 'mip', 'minip', 'average' and
    'attenuated_mip' (attenuation 0) projections, and with ``np.max`` passed
    explicitly as ``projector``. Project a NaN-free random volume the same way,
    and average a float16 volume whose ray sum exceeds the float16 range.
    Warnings are errors.

    Expected outcome: the MIP keeps the bead (100) and is 0 on the other rays;
    MinIP is 0; 'average' is the mean of the finite samples (100 / 9 on the
    bead's ray); only the all-NaN ray is NaN; the composite shows the bead
    (white) instead of black; attenuation 0 equals the MIP wherever the MIP is
    finite. An explicitly passed projector is used as given (NaN propagates).
    Without NaN every projection equals ``np.max`` / ``np.min`` / ``np.mean``
    exactly, and the float16 average is accumulated like ``np.mean`` (finite).
    """
    v = np.zeros((10, 20, 20), dtype=np.float32)
    v[0, :, :10] = np.nan
    v[:, 0, 0] = np.nan
    v[5, 10, 5] = 100.0

    def expected(bead):
        out = np.zeros((20, 20), dtype=np.float32)
        out[10, 5] = bead
        out[0, 0] = np.nan
        return out

    def xy(*args, **kwargs):
        fig = show_zyx_max(*args, **kwargs)
        try:
            return _panel_rgb(fig)
        finally:
            plt.close(fig)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mip = xy(v, vmin=0, vmax=100)
        np.testing.assert_array_equal(mip, expected(100.0))
        np.testing.assert_array_equal(xy(v, vmin=0, vmax=100, projection="mip"), expected(100.0))
        np.testing.assert_array_equal(xy(v, vmin=0, vmax=100, projection="minip"), expected(0.0))
        np.testing.assert_allclose(xy(v, vmin=0, vmax=100, projection="average"), expected(100.0 / 9), rtol=1e-6)
        att = xy(v, vmin=0, vmax=100, projection="attenuated_mip", attenuation=0.0)
        finite = np.isfinite(mip)
        assert finite.sum() == 399
        np.testing.assert_allclose(att[finite], mip[finite], rtol=1e-6)
        np.testing.assert_allclose(xy([v, 0 * v])[10, 5], [1, 1, 1], atol=1e-6)

        clean = np.random.default_rng(0).standard_normal((10, 20, 20)).astype(np.float32)
        for projection, ref in ((None, np.max), ("mip", np.max), ("minip", np.min), ("average", np.mean)):
            np.testing.assert_array_equal(xy(clean, vmin=-1, vmax=1, projection=projection), ref(clean, axis=0))

        v16 = np.full((10, 2, 2), 10000, dtype=np.float16)
        v16[0, 0, 0] = np.nan
        np.testing.assert_array_equal(tpa._nan_mean_projection(v16, axis=0), np.full((2, 2), 10000, np.float16))

    fig = show_zyx_projection(v, projector=np.max, vmin=0, vmax=100)
    try:
        assert np.isnan(_panel_rgb(fig)[10, 5])
    finally:
        plt.close(fig)


def test_attenuated_mip_matches_napari_formula():
    """Verify ``_attenuated_mip`` implements napari's exact cumulative exponential attenuation formula.

    Strategy: evaluate a two-sample ray analytically with ``a=1.0`` and check
    that ``a=0.0`` reduces to a clipped MIP in data units.
    Expected outcome: matches the closed-form formula and clipped MIP within ``1e-5``.
    """
    a = 1.0
    ray = np.array([0.5, 1.0], dtype=np.float32).reshape(2, 1, 1)
    expected = max(0.5 * np.exp(-a * (0.5 - 1)), 1.0 * np.exp(-a * (1.5 - 1)))
    np.testing.assert_allclose(_attenuated_mip(ray, 0, 0.0, 1.0, a), expected, rtol=1e-6)

    vol = np.random.default_rng(2).random((5, 4, 3)).astype(np.float32) * 10
    np.testing.assert_allclose(
        _attenuated_mip(vol, 0, 2.0, 8.0, 0.0),
        np.clip(vol.max(axis=0), 2.0, 8.0),
        rtol=1e-5,
    )


def test_attenuated_mip_hides_structures_behind_bright_ones():
    """Verify ``_attenuated_mip`` dims deep structures behind haze while preserving shallow ones.

    Strategy: embed identical bright voxels at ``z=2`` (shallow) and ``z=25``
    (deep) inside uniform 0.2 haze, and compare ``attenuation=0.3`` vs ``0.0``.
    Expected outcome: shallow voxel stays bright, deep voxel is occluded below
    the front haze, and zero attenuation shows both at 1.0.
    """
    vol = np.full((30, 1, 2), 0.2, dtype=np.float32)
    vol[2, 0, 0] = 1.0
    vol[25, 0, 1] = 1.0
    out = _attenuated_mip(vol, 0, 0.0, 1.0, 0.3)
    np.testing.assert_allclose(out[0, 0], np.exp(-0.3 * 0.4), rtol=1e-5)
    np.testing.assert_allclose(out[0, 1], 0.2 * np.exp(0.3 * 0.8), rtol=1e-5)
    np.testing.assert_allclose(_attenuated_mip(vol, 0, 0.0, 1.0, 0.0), [[1.0, 1.0]])


def test_slice_widget_blending_and_projection_kwargs():
    """Verify keyword-only ``blending`` and ``projection`` options propagate without adding GUI traits.

    Strategy: construct slice and annotator widgets via public factories with
    ``blending='minimum'``, ``projection='attenuated_mip'``, and ``attenuation=0.2``,
    inspect traits, GUI colors, figure theme, and ``copy_params_string``.
    Expected outcome: options are not exposed as traitlets, dark ink lifts to
    ``#9e9e9e``, figure background is white, and ``copy_params`` emits all three kwargs.
    """
    rng = np.random.default_rng(3)
    chans = [rng.random((6, 12, 14)).astype(np.float32) for _ in range(2)]
    w = show_zyx_max_slice_interactive(
        chans, blending="minimum", projection="attenuated_mip", attenuation=0.2
    )
    assert not {"blending", "projection", "attenuation"} & set(w.trait_names())
    assert w.colors_resolved == ["black", "magenta"]
    assert w.channel_colors[0] == "#9e9e9e"  # dark ink lifted for the dark GUI
    fig = w._render()
    try:
        assert mcolors.to_hex(fig.patch.get_facecolor()) == "#ffffff"
    finally:
        plt.close(fig)
    w.copy_params_trigger += 1
    assert "blending='minimum'" in w.copy_params_string
    assert "projection='attenuated_mip'" in w.copy_params_string
    assert "attenuation=0.2" in w.copy_params_string

    w2 = show_zyx_max_slice_interactive(chans[0], blending="minimum", projection="minip")
    assert (w2.blending, w2.projection) == ("minimum", "minip")
    w3 = show_zyx_max_slice_interactive_point_annotator(chans, blending="minimum")
    assert w3.colors_resolved[:2] == ["black", "magenta"] and w3.blending == "minimum"


def test_scatter_widget_minimum_blending():
    """Verify ``blending='minimum'`` switches scatter viewers to a white background in both render modes.

    Strategy: construct points and density scatter viewers with
    ``blending='minimum'`` and a default control viewer.
    Expected outcome: minimum blending sets ``subplot_bg == 'white'`` and white
    ``axXY`` facecolor; default keeps ``subplot_bg == 'black'``.
    """
    rng = np.random.default_rng(4)
    pts = rng.random((200, 3)) * [8, 20, 30]
    for render in ("points", "density"):
        w = show_zyx_max_scatter_interactive(
            pts, channels=list(rng.random((2, 200))), render=render, blending="minimum"
        )
        assert w.subplot_bg == "white"
        fig = w._render()
        try:
            assert mcolors.to_hex(fig.axXY.get_facecolor()) == "#ffffff"
        finally:
            plt.close(fig)
    w = show_zyx_max_scatter_interactive(pts, render="points")
    assert w.subplot_bg == "black"


# ---------------------------------------------------------------------------
# xarray STCZYX DataArray integration (relocated from test_channel_info_xarray)
# ---------------------------------------------------------------------------


def _xarray_viewer_da(nz=3, ny=12, nx=10, ps=None):
    """Build a calibrated STCZYX DataArray with channel metadata coordinates."""
    xr = pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray, set_channel_info

    names = ["DAPI", "SOX2", "TBXT", "BF"]
    info = {
        "DAPI": {"color": "gray", "role": "dna"},
        "SOX2": {"color": "cyan", "role": "marker"},
        "TBXT": {"color": "yellow", "role": "marker"},
        "BF": {"color": "gray", "role": "brightfield"},
    }
    arr = np.arange(len(names) * nz * ny * nx, dtype=np.float32).reshape(len(names), nz, ny, nx)
    ps = ps or {"Z": 2.0, "Y": 0.3, "X": 0.3, "unit": "µm"}
    da = array_to_stczyx_xarray(arr, input_axes="CZYX", channel_names=names, pixel_sizes=ps)
    assert isinstance(da, xr.DataArray)
    return set_channel_info(da, info)


def test_viewer_accepts_stczyx_dataarray_and_explicit_overrides():
    """Verify ``show_zyx_max_slice_interactive`` unpacks xarray coords and lets explicit args win.

    Strategy: construct a calibrated STCZYX DataArray with ``color`` and ``role``
    coordinates, select two channels via ``.sel(C=...)``, and pass it to
    ``show_zyx_max_slice_interactive`` with and without explicit ``colormap`` and
    ``channel_labels`` overrides.
    Expected outcome: default call inherits channel names, colours, and physical
    pixel sizes from the DataArray; explicit arguments override them.
    """
    da = _xarray_viewer_da(ps={"Z": 2.0, "Y": 0.5, "X": 0.25, "unit": "µm"})
    sub = da.sel(C=["DAPI", "TBXT"])

    w = show_zyx_max_slice_interactive(sub)
    assert w.num_channels == 2
    assert w.channel_names == ["DAPI", "TBXT"]
    assert w.colors_resolved[1] == "yellow"
    assert (w.sz, w.sy, w.sx) == pytest.approx((2.0, 0.5, 0.25))

    w2 = show_zyx_max_slice_interactive(sub, colormap=["red", "blue"], channel_labels=["a", "b"])
    assert w2.colors_resolved == ["red", "blue"]
    assert w2.channel_names == ["a", "b"]


def test_viewer_dataarray_matches_list_input():
    """Verify passing an STCZYX DataArray produces the same viewer state as a plain channel list.

    Strategy: build a 4-channel calibrated DataArray without a ``color`` coord
    and compare ``show_zyx_max_slice_interactive(da)`` against passing the
    extracted NumPy arrays, channel names, and pixel sizes explicitly.
    Expected outcome: channel names, colours, pixel sizes, and rendered PNG
    payload match identically.
    """
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray

    names = ["DAPI", "SOX2", "TBXT", "BF"]
    ps = {"Z": 2.0, "Y": 0.3, "X": 0.3, "unit": "µm"}
    arr = np.arange(len(names) * 3 * 12 * 10, dtype=np.float32).reshape(len(names), 3, 12, 10)
    da = array_to_stczyx_xarray(arr, input_axes="CZYX", channel_names=names, pixel_sizes=ps)

    w_da = show_zyx_max_slice_interactive(da)
    w_list = show_zyx_max_slice_interactive(
        [arr[i] for i in range(len(names))],
        channel_labels=names,
        pixel_sizes=ps,
    )
    assert w_da.channel_names == w_list.channel_names == names
    assert w_da.colors_resolved == w_list.colors_resolved
    assert (w_da.sz, w_da.sy, w_da.sx) == pytest.approx((w_list.sz, w_list.sy, w_list.sx))
    assert w_da.image_data == w_list.image_data


def test_viewer_rejects_unselected_scene_or_time():
    """Verify ``show_zyx_max_slice_interactive`` rejects DataArrays with multi-entry S or T dims.

    Strategy: pass a 6D STCZYX DataArray with ``S=2`` directly to
    ``show_zyx_max_slice_interactive``.
    Expected outcome: raises ``ValueError`` instructing the caller to select one
    scene via ``isel(S=0)``.
    """
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray

    arr = np.zeros((2, 1, 1, 3, 4, 5), dtype=np.float32)
    da = array_to_stczyx_xarray(arr, input_axes="STCZYX")
    with pytest.raises(ValueError, match=r"isel\(S=0\)"):
        show_zyx_max_slice_interactive(da)


def test_viewer_deprecated_colors_arg_applies_to_dataarray():
    """Verify deprecated ``colors=`` still warns and overrides a DataArray's ``color`` coord.

    Strategy: pass a DataArray with a ``color`` coordinate along ``C`` together
    with deprecated ``colors=['red', 'blue']`` to ``show_zyx_max_slice_interactive``.
    Expected outcome: emits ``FutureWarning`` and applies ``['red', 'blue']``.
    (Changed by D08: it was a DeprecationWarning.)
    """
    da = _xarray_viewer_da().sel(C=["DAPI", "TBXT"])
    with pytest.warns(FutureWarning):
        w = show_zyx_max_slice_interactive(da, colors=["red", "blue"])
    assert w.colors_resolved == ["red", "blue"]


# ---------------------------------------------------------------------------
# Shared conventions (D01-D04): units, display range, colours, compositing
# ---------------------------------------------------------------------------

def _hot_pixel_volume():
    """uint16 Poisson(100) background, a +300 blob, and one 4000 hot pixel (fraction 1e-5)."""
    rng = np.random.default_rng(0)
    vol = rng.poisson(100, (10, 100, 100)).astype(np.uint16)
    vol[4:6, 40:60, 40:60] += 300
    vol[0, 0, 0] = 4000
    return vol


def _xy_clim(fig):
    return tuple(float(v) for v in fig.axXY.images[0].get_clim())


def test_same_data_gets_same_range_colours_and_label_in_every_viewer():
    """WP4 invariance: static single, static list, TNIASliceWidget and LabelAnnotator agree.

    Strategy: pass one calibrated uint16 volume (hot pixel, blob) through
    ``show_zyx_max(vol)``, ``show_zyx_max([vol])``, ``TNIASliceWidget(vol)`` and
    ``LabelAnnotator(vol)``; compare (vmin, vmax), channel colours and the
    scale-bar label (the annotator has no scale bar).
    Expected outcome: one (vmin, vmax) everywhere (the default display range
    of the volume); one grey/white channel colour; the same '10 µm' label in
    the three ortho viewers. Before, the four entry points used four ranges
    (panel min/max, 0/max, 0/p99.99, p0.1/p99.99) and the static single
    channel was viridis with a black label.
    """
    from eigenp_utils._conventions import default_display_range
    from eigenp_utils.label_annotator import LabelAnnotator

    vol = _hot_pixel_volume()
    ps = {"Z": 1.0, "Y": 0.5, "X": 0.5, "unit": "µm"}
    expected = default_display_range(vol)
    assert expected[0] == 0.0 and expected[1] < 1000  # the hot pixel does not set vmax

    single = show_zyx_max(vol, pixel_sizes=ps)
    lst = show_zyx_max([vol], pixel_sizes=ps)
    w = TNIASliceWidget(vol, pixel_sizes=ps)
    la = LabelAnnotator(vol, axes="ZYX")
    try:
        assert _xy_clim(single) == pytest.approx(expected)
        # the list composite equals the single channel through the same range
        grey = mcolors.to_rgb(single.axXY.images[0].to_rgba(np.asarray(single.axXY.images[0].get_array()))[5, 5])
        np.testing.assert_allclose(_panel_rgb(lst)[5, 5], grey, atol=1 / 255)
        assert (w.vmin_list[0], w.vmax_list[0]) == pytest.approx(expected)
        assert la._limits[0] == pytest.approx(expected)

        assert single.axXY.images[0].get_cmap()(1.0)[:3] == pytest.approx((1.0, 1.0, 1.0))
        assert w.colors_resolved == ["white"] and la.channel_colors == ["#ffffff"]

        wfig = w._render()
        labels = {_scale_bar_artists(f)[1].get_text() for f in (single, lst, wfig)}
        assert labels == {"10 µm"}
        plt.close(wfig)
    finally:
        plt.close(single)
        plt.close(lst)


def test_hot_pixel_does_not_change_the_blob_contrast_between_static_and_widget():
    """D02 (+) control: the blob's fraction of the display range is the same in the static and widget views.

    Strategy: the hot-pixel volume (hot pixel on 1e-5 of the voxels) through
    ``show_zyx_max`` and the slice widget; read the blob's normalised value.
    Expected outcome: equal to within 1% (both use the source volume's
    robust range), and the blob is shown bright (> 0.6), not at the ~7%
    the panel max (the hot pixel) gave.
    """
    vol = _hot_pixel_volume()
    fig = show_zyx_max(vol)
    w = show_zyx_max_slice_interactive(vol)
    try:
        lo, hi = _xy_clim(fig)
        static = (float(np.asarray(fig.axXY.images[0].get_array())[50, 50]) - lo) / (hi - lo)
        wlo, whi = w.vmin_list[0], w.vmax_list[0]
        widget = (float(vol[4:6, 50, 50].max()) - wlo) / (whi - wlo)
        assert abs(static - widget) <= 0.01 and static > 0.6
    finally:
        plt.close(fig)


def test_signed_data_are_not_clipped_at_zero():
    """D02 (+) control: signed data get vmin = data minimum in the widget and static views.

    Strategy: a float32 N(0, 1) volume through the slice widget, the
    multichannel widget, ``show_zyx_max`` and a cleared GUI vmin box.
    Expected outcome: vmin equals the data minimum (< -3), not 0.0; clearing
    the box restores that cached value (not the minimum of the current slab).
    """
    vol = np.random.default_rng(1).normal(0, 1, (8, 30, 30)).astype(np.float32)
    lo = float(vol.min())
    w = show_zyx_max_slice_interactive(vol)
    assert w.vmin_list[0] == pytest.approx(lo) and lo < -3
    wm = show_zyx_max_slice_interactive([vol, vol + 5])
    assert wm.vmin_list == pytest.approx([lo, 0.0])
    fig = show_zyx_max(vol)
    try:
        assert _xy_clim(fig)[0] == pytest.approx(lo)
    finally:
        plt.close(fig)

    seen = {}
    orig = tpa.show_zyx_max_slabs

    def spy(*args, **kwargs):
        seen.update(vmin=kwargs["vmin"])
        return orig(*args, **kwargs)

    tpa.show_zyx_max_slabs = spy
    try:
        w.vmin_list = [""]
        assert seen["vmin"] == pytest.approx(lo)
        wm.vmin_list = ["", 1.0]
        assert seen["vmin"] == pytest.approx([lo, 1.0])
    finally:
        tpa.show_zyx_max_slabs = orig


def test_outlier_free_data_keep_vmax_within_one_grey_level_of_the_max():
    """D02 (-) control: without outliers the robust vmax is the data maximum to within one grey level.

    Strategy: a uint8 ramp and a 12-bit ramp volume through ``show_zyx_max``
    and the slice widget.
    Expected outcome: |vmax - max| <= 1 in both.
    """
    for vol in (np.arange(16 * 16 * 16, dtype=np.uint16).reshape(16, 16, 16),
                (np.arange(20 * 20 * 20) % 256).astype(np.uint8).reshape(20, 20, 20)):
        fig = show_zyx_max(vol)
        try:
            assert abs(_xy_clim(fig)[1] - vol.max()) <= 1
        finally:
            plt.close(fig)
        assert abs(show_zyx_max_slice_interactive(vol).vmax_list[0] - vol.max()) <= 1


def test_attenuated_mip_uses_the_display_range():
    """R1-06: 'attenuated_mip' is normalised with the same (volume) range that is displayed.

    Strategy: render ``projection='attenuated_mip'`` and record the range the
    projector receives.
    Expected outcome: the projector's (lo, hi) equals the XY panel's clim.
    """
    vol = _hot_pixel_volume().astype(np.float32)
    seen = []
    orig = tpa._projector_for

    def spy(projection, attenuation, lo=None, hi=None):
        seen.append((lo, hi))
        return orig(projection, attenuation, lo, hi)

    tpa._projector_for = spy
    try:
        fig = show_zyx_max(vol, projection="attenuated_mip")
    finally:
        tpa._projector_for = orig
    try:
        assert seen and seen[0] == pytest.approx(_xy_clim(fig))
    finally:
        plt.close(fig)


def test_um_calibrated_output_is_identical_to_the_base_commit():
    """D01 (-) control: µm-calibrated renders are unchanged by the shared conventions.

    Strategy: render the fixture's figures (``tests/_tnia_um_reference.py``:
    single channel, two channels, slice widget; explicit display settings so
    only pixel-size handling can matter) and compare with
    ``tnia_um_reference.npz``, generated on the base commit af8d252.
    Expected outcome: identical panel arrays, extents, axes positions, clims
    and scale-bar labels; with the same matplotlib version also the same PNG
    bytes.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import _tnia_um_reference as ref
    finally:
        sys.path.pop(0)
    expected = np.load(ref.FIXTURE)
    got = ref.render_um_reference()
    same_mpl = str(expected["matplotlib_version"]) == matplotlib.__version__
    for key in expected.files:
        if key == "matplotlib_version" or (key.endswith("png_sha256") and not same_mpl):
            continue
        np.testing.assert_array_equal(got[key], expected[key], err_msg=key)


def test_cm_calibrated_tiff_round_trip_gives_um_and_the_right_aspect(tmp_path):
    """D01 (+) control (C-01): a TIFF calibrated in cm shows µm sizes and a correctly proportioned XY panel.

    Strategy: write a (20, 64, 64) stack with 1 µm pixels stored as cm
    resolution tags and no Z step; load it with ``load_image`` and view it
    with ``show_zyx_max`` and the slice widget.
    Expected outcome: pixel sizes 1 µm (Z filled from X/Y with a warning),
    the XY panel spans 64 x 64 µm and is as tall as the XZ panel is wide x 20/64
    (it used to collapse to a dot next to 1-cm slices), and the scale bar reads µm.
    """
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import load_image

    p = tmp_path / "cm.tif"
    tifffile.imwrite(p, np.random.default_rng(0).integers(0, 1000, (20, 64, 64)).astype(np.uint16),
                     resolution=(1e4, 1e4), resolutionunit="CENTIMETER")
    da = load_image(p, input_axes="ZYX")
    with pytest.warns(UserWarning, match="axis Z"):
        fig = show_zyx_max(da.isel(S=0, T=0, C=0).values, pixel_sizes=tpa._viewer_input_from_xarray(da, None, None, None)[3])
    try:
        assert abs(np.diff(fig.axXY.get_xlim())[0]) == pytest.approx(64.0)
        assert abs(np.diff(fig.axXZ.get_ylim())[0]) == pytest.approx(20.0)
        h_xy = fig.axXY.get_position().height
        h_xz = fig.axXZ.get_position().height
        assert h_xz / h_xy == pytest.approx(20 / 64, rel=1e-3)
        assert _scale_bar_artists(fig)[1].get_text().endswith("µm")
    finally:
        plt.close(fig)
    with pytest.warns(UserWarning, match="axis Z"):
        w = show_zyx_max_slice_interactive(da)
    assert (w.sz, w.sy, w.sx) == pytest.approx((1.0, 1.0, 1.0))


def test_uncalibrated_tiff_gets_a_pixel_scale_bar(tmp_path):
    """D01 (+) control (R8-01): an uncalibrated TIFF read by either reader shows a 'px' scale bar.

    Strategy: write a TIFF without calibration; view the
    ``get_tiff_voxel_size`` dict and the ``load_image`` DataArray.
    Expected outcome: labels end with ' px' (it was '20 µm' for the
    ``get_tiff_voxel_size`` dict, a physical bar on an uncalibrated image).
    """
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import get_tiff_voxel_size, load_image

    p = tmp_path / "plain.tif"
    vol = np.random.default_rng(0).integers(0, 255, (5, 40, 60)).astype(np.uint8)
    tifffile.imwrite(p, vol)
    with pytest.warns(UserWarning, match="No value for axis"):
        fig = show_zyx_max(vol, pixel_sizes=get_tiff_voxel_size(p))
    try:
        assert _scale_bar_artists(fig)[1].get_text() == "10 px"
    finally:
        plt.close(fig)
    w = show_zyx_max_slice_interactive(load_image(p, input_axes="ZYX"))
    fig = w._render()
    try:
        assert _scale_bar_artists(fig)[1].get_text() == "10 px"
    finally:
        plt.close(fig)


@pytest.mark.parametrize("size, unit, expected", [
    (0.002, "µm", "100 nm"),   # 256 px * 2 nm: EM scale
    (500.0, "nm", "20 µm"),
    (10.0, "µm", "500 µm"),
    (40.0, "µm", "2 mm"),      # 256 px * 40 µm: mm-scale tissue
    (0.04, "mm", "2 mm"),
])
def test_scale_bar_picks_nm_um_mm(size, unit, expected):
    """D01 (user override): the scale bar label picks nm / µm / mm by the bar's length.

    Strategy: render a 256-px-wide volume at EM, light-microscopy and
    tissue scales.
    Expected outcome: '100 nm', '20 µm', '500 µm', '2 mm' (it used to be
    '0.1 µm' and '2000 µm').
    """
    im = np.zeros((4, 256, 256), dtype=np.uint8)
    fig = show_zyx_max(im, pixel_sizes={"Z": size, "Y": size, "X": size, "unit": unit})
    try:
        assert _scale_bar_artists(fig)[1].get_text() == expected
    finally:
        plt.close(fig)


def test_static_single_channel_defaults_to_grey():
    """D03 (R1-02): a single channel defaults to the grey (white) ramp, as a one-channel list and the widgets.

    Strategy: render ``show_zyx_max(vol)`` with channel labels.
    Expected outcome: the colormap runs black -> white (not viridis) and the
    label is drawn in the GUI grey of 'white' (it used to be black).
    """
    vol = _structured_volume()
    fig = show_zyx_max_slabs(vol, [0, 48], [0, 48], [0, 16], channel_labels=["dna"])
    try:
        cmap = fig.axXY.images[0].get_cmap()
        assert cmap(0.0)[:3] == pytest.approx((0, 0, 0)) and cmap(1.0)[:3] == pytest.approx((1, 1, 1))
        texts = [c for c in fig.axLabels.findobj(matplotlib.text.Text) if c.get_text() == "dna"]
        assert mcolors.to_hex(texts[0].get_color()) == get_gui_channel_color("white")
    finally:
        plt.close(fig)


def test_white_start_colormaps_blend_with_minimum_by_default():
    """D04 (R5-13): channels coloured with the white-start I-series use 'minimum' blending by default.

    Strategy: two channels with squares, ``colormap=['I Blue', 'I Bordeaux']``
    and ``blending=None``, compared with explicit 'minimum'; a mix of a
    white-start and a dark-start colormap; dark colormaps (-).
    Expected outcome: identical to 'minimum' (objects coloured on white; the
    old default 'maximum' was white everywhere); the mix warns and keeps
    'maximum'; ['viridis', 'magma'] still equals 'maximum'.
    """
    a = np.zeros((4, 20, 20), np.float32)
    b = np.zeros((4, 20, 20), np.float32)
    a[:, 2:8, 2:8] = 1.0
    b[:, 12:18, 12:18] = 1.0
    f0 = show_zyx_max([a, b], colormap=["I Blue", "I Bordeaux"])
    f1 = show_zyx_max([a, b], colormap=["I Blue", "I Bordeaux"], blending="minimum")
    try:
        np.testing.assert_array_equal(_panel_rgb(f0), _panel_rgb(f1))
        assert _panel_rgb(f0)[5, 5].max() < 0.5  # the object is visible (dark ink on white)
        assert mcolors.to_hex(f0.patch.get_facecolor()) == "#ffffff"
    finally:
        plt.close(f0)
        plt.close(f1)
    with pytest.warns(UserWarning, match="start at white"):
        plt.close(show_zyx_max([a, b], colormap=["I Blue", "viridis"]))


def test_scatter_follows_the_blending_rule():
    """D04 (R1-12): the scatter viewer composites colormap channels like the slice viewer (maximum).

    Strategy: two continuous point channels coloured ['viridis', 'magma'] in
    points and density mode with ``blending=None`` vs explicit 'maximum';
    plain colours vs explicit 'additive'.
    Expected outcome: identical renders (density used to be additive with
    colormap names).
    """
    rng = np.random.default_rng(0)
    n = 300
    pts = (rng.uniform(0, 10, n), rng.uniform(0, 40, n), rng.uniform(0, 40, n))
    vals = rng.random((n, 2))
    for cmap, mode in ((["viridis", "magma"], "maximum"), (["red", "lime"], "additive")):
        for render in ("points", "density"):
            w0 = show_zyx_max_scatter_interactive(pts, channels=vals, colormap=cmap, render=render)
            w1 = show_zyx_max_scatter_interactive(pts, channels=vals, colormap=cmap, render=render, blending=mode)
            assert w0._blend == mode
            f0, f1 = w0._render(), w1._render()
            try:
                b0, b1 = io.BytesIO(), io.BytesIO()
                f0.savefig(b0, format="png", dpi=40)
                f1.savefig(b1, format="png", dpi=40)
                assert b0.getvalue() == b1.getvalue(), (cmap, render)
            finally:
                plt.close(f0)
                plt.close(f1)


def test_compositors_share_one_implementation():
    """D04 (R1-12): create_multichannel_rgb, create_multichannel_rgb_cmap and blend_colors agree.

    Strategy: compose the same two channels with plain colours (and with a
    colormap) through all three, with the same explicit range and blending.
    Expected outcome: identical colours (plain colours used to differ by up
    to 1/255 depending on whether another channel was a colormap name).
    """
    rng = np.random.default_rng(0)
    a, b = rng.random((2, 6, 7)).astype(np.float32)
    for cmap in (["red", "lime"], ["viridis", "lime"]):
        for blend in ("additive", "maximum", "screen", "minimum"):
            kw = dict(colormap=cmap, vmin=0, vmax=1, blend=blend)
            x1 = create_multichannel_rgb([a, b], [a, b], [a, b], **kw)[0]
            x2 = create_multichannel_rgb_cmap([a, b], [a, b], [a, b], **kw)[0]
            pts = blend_colors(np.column_stack([a.ravel(), b.ravel()]), cmap, vmin=0, vmax=1, blending=blend)
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_allclose(pts, x1.reshape(-1, 3), atol=1e-6)


def test_per_channel_lists_behave_the_same_in_every_viewer():
    """D04 (R1-13): a per-channel list longer than the channels raises in the static and widget viewers.

    Strategy: two channels with ``vmax=[1, 2, 3]`` (and ``gamma`` of 3
    entries) through ``show_zyx_max``, the slice widget and the scatter
    widget; then a short list with a None entry.
    Expected outcome: ValueError naming the parameter everywhere (the widgets
    used to drop the extra entry silently); the short list is padded with
    the default in the widgets (negative control).
    """
    a = np.random.default_rng(0).random((4, 5, 6)).astype(np.float32)
    pts = (np.array([0.0, 3.0]), np.array([0.0, 4.0]), np.array([0.0, 5.0]))
    chans = np.array([[0.2, 0.5], [0.7, 0.1]])
    for kw, msg in ((dict(vmax=[1, 2, 3]), "vmax has 3 entries"), (dict(gamma=[1, 1, 1]), "gamma has 3 entries")):
        with pytest.raises(ValueError, match=msg):
            plt.close(show_zyx_max([a, a], **kw))
        with pytest.raises(ValueError, match=msg):
            show_zyx_max_slice_interactive([a, a], **kw)
        with pytest.raises(ValueError, match=msg):
            show_zyx_max_scatter_interactive(pts, channels=chans, **kw)
    w = show_zyx_max_slice_interactive([a, a], vmax=[2.0, None], gamma=[0.5])
    assert w.vmax_list[0] == 2.0 and w.vmax_list[1] == w._auto_vmax[1] and w.gamma_list == [0.5, 1.0]


def test_annotator_and_xarray_colours_use_the_shared_palette():
    """D03 (R2-19, R8-07): the annotator pads like the slice viewer; tuple colours in a DataArray are used.

    Strategy: (1) annotator with ``colormap='white'`` on three channels and
    with six default channels (red is the 6th palette colour); (2) a
    DataArray whose ``color`` coordinate holds RGB tuples and an empty entry.
    Expected outcome: (1) image channels ['white', 'lime', 'magenta'] (it was
    white, white, lime) and an annotation colour not used by an image
    channel ('orange' when red is taken); (2) the tuples become hex colours,
    the empty entry takes a palette colour.
    """
    vol = _structured_volume()
    w = show_zyx_max_slice_interactive_point_annotator([vol] * 3, colormap="white")
    assert w.colors_resolved[:3] == ["white", "lime", "magenta"] and w.colors_resolved[3] == "red"
    w6 = show_zyx_max_slice_interactive_point_annotator([vol] * 6)
    assert w6.colors_resolved[5] == "red" and w6.colors_resolved[6] == "orange"

    cols = tpa._channel_colors_from_coord(np.array([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), ""], dtype=object))
    assert cols == ["#ff0000", "#00ff00", "white"]


def test_histogram_sample_does_not_alias_and_int8_range_is_capped():
    """R1-20: the histogram sample is random (no column aliasing); non-negative int8 spans 0..127.

    Strategy: a 1000 x 2000 uint16 image whose every other column is 1000,
    histogram with ``max_samples=1000``; a float image with 0.5 at every
    20th index for ``_is_integer_valued``; int8 data in [0, 100].
    Expected outcome: both value populations appear (a strided sample saw
    only one); the float image is not integer-valued; int8 gives (7, False)
    and a 0..127 range (it was 0..255).
    """
    img = np.zeros((1000, 2000), np.uint16)
    img[:, ::2] = 1000
    h = compute_histogram(img, max_samples=1000, value_range=(0, 1000))
    counts = np.asarray(h["counts"])
    assert counts[0] > 0 and counts[-1] > 0
    f = np.zeros(10_000_000, np.float32)
    f[1::20] = 0.5
    assert not tpa._is_integer_valued(f)
    assert infer_bit_depth(np.array([0, 100], np.int8)) == (7, False)
    assert infer_histogram_range(np.array([0, 100], np.int8))[0] == (-0.5, 127.5)


# ---------------------------------------------------------------------------
# Stage 2 (viewers): matplotlib OO interface (D05)
# ---------------------------------------------------------------------------

def test_no_pyplot_figures_after_rendering_every_viewer_with_defaults(tmp_path):
    """D05 control: no viewer registers a figure with pyplot (OO interface only).

    Strategy: close every pyplot figure, then build and use every static view
    and every widget type with the default arguments (including the "No
    Channels Visible" placeholder, copy-PNG, SVG export and IsoScatter's
    camera), and finally open a user figure before rendering a scatter in
    points mode (whose figure type used to depend on that global state).
    Expected outcome: ``plt.get_fignums()`` stays empty throughout, then holds
    only the user's figure; ``show_zyx`` returns a plain ``Figure`` whose
    canvas is not pyplot-managed. (Changed by D05: ``use_plt`` defaulted to
    True and every render created a pyplot figure.)
    """
    plt.close("all")
    rng = np.random.default_rng(0)
    vol = rng.random((6, 16, 20)).astype(np.float32)
    figs = [
        show_zyx(vol[3], vol[:, 8, :], np.flip(np.rot90(vol[:, :, 10], 1), 0)),
        show_zyx_slice(vol, 5, 5, 3),
        show_zyx_max(vol),
        show_zyx_projection(vol),
        show_zyx_max_slabs(vol),
        tpa.show_zyx_projection_slabs(vol, slice(None), slice(None), slice(None)),
    ]
    assert all(type(f) is Figure for f in figs)
    assert all(f.canvas.manager is None for f in figs)
    assert plt.get_fignums() == []

    w = show_zyx_max_slice_interactive(vol)
    w.opacity_list = [0.0]  # placeholder figure
    w.opacity_list = [1.0]
    w.copy_png_trigger += 1
    w.save_filename = str(tmp_path / "v.svg")
    w.save_trigger += 1
    multi = show_zyx_max_slice_interactive([vol, vol])
    multi.opacity_list = [0.0, 0.0]
    a = show_zyx_max_slice_interactive_point_annotator(vol)
    a.points = [[3, 8, 10]]
    pts = rng.random((50, 3)) * [6, 16, 20]
    show_zyx_max_scatter_interactive(pts, render="points")
    show_zyx_max_scatter_interactive(pts, render="density", bins=16)
    iso = show_iso_scatter(*_iso_points(n=30), figsize=(4, 4))
    iso.azim = iso.azim + 10
    assert plt.get_fignums() == []

    user_fig = plt.figure()
    try:
        show_zyx_max_scatter_interactive(pts, render="points")
        assert plt.get_fignums() == [user_fig.number]
    finally:
        plt.close("all")
    for f in figs:
        plt.close(f)


# ---------------------------------------------------------------------------
# Stage 2 (viewers): NaN / theme background, dead helpers (WP2)
# ---------------------------------------------------------------------------

def _rendered_rgb_at(fig, ax, fx, fy):
    """RGB (0..255) of the rendered figure at axes fraction ``(fx, fy)`` of ``ax``."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    x, y = ax.transAxes.transform((fx, fy))
    return buf[int(buf.shape[0] - y), int(x), :3].astype(int)


def test_nan_pixels_and_opacity_show_the_theme_background_not_white():
    """R1-03: single-channel NaN pixels and opacity fade to the panel background.

    Strategy: a volume of 10 whose top half is NaN, drawn grey on the dark
    theme, and with ``blending='minimum'`` (white theme); then a 0..1 ramp
    with ``opacity=0.5``.
    Expected outcome: NaN pixels render black on the dark theme (white on
    the white theme), like the multichannel composite; data pixels stay white
    (positive control); opacity fades value 0 to black, not to grey. (They
    used to show the white axes background: NaN looked like saturated signal.)
    """
    v = np.full((4, 16, 16), 10.0, dtype=np.float32)
    v[:, :8, :] = np.nan
    fig = show_zyx_max(v, colormap="gray", figsize=(4, 4))
    assert list(_rendered_rgb_at(fig, fig.axXY, 0.5, 0.75)) == [0, 0, 0]       # NaN (top half)
    assert list(_rendered_rgb_at(fig, fig.axXY, 0.5, 0.25)) == [255, 255, 255]  # data
    fig_multi = show_zyx_max([v, 0 * v], figsize=(4, 4))
    assert list(_rendered_rgb_at(fig_multi, fig_multi.axXY, 0.5, 0.75)) == [0, 0, 0]
    fig_min = show_zyx_max(v, colormap="red", blending="minimum", figsize=(4, 4))
    assert list(_rendered_rgb_at(fig_min, fig_min.axXY, 0.5, 0.75)) == [255, 255, 255]

    ramp = np.zeros((4, 16, 16), dtype=np.float32)
    ramp[:, :, 8:] = 1.0
    fig_op = show_zyx_max(ramp, colormap="gray", vmin=0, vmax=1, opacity=0.5, figsize=(4, 4))
    assert list(_rendered_rgb_at(fig_op, fig_op.axXY, 0.25, 0.5)) == [0, 0, 0]
    assert _rendered_rgb_at(fig_op, fig_op.axXY, 0.75, 0.5)[0] in range(120, 135)
    for f in (fig, fig_multi, fig_min, fig_op):
        plt.close(f)


def test_rotation_corners_show_the_background_not_value_zero():
    """R1-10: corners added by ``rotate_view`` are background, not the colour of 0.

    Strategy: signed data in [-1, 1] drawn grey (clim [-1, 1], so 0 is
    mid-grey) with the XY panel rotated by 45 degrees.
    Expected outcome: a corner of the XY panel renders black (the panel
    background) instead of grey 128, and the image centre is not black
    (positive control). The corners are NaN in the panel array.
    """
    rng = np.random.default_rng(0)
    xy = rng.uniform(-1, 1, (32, 32)).astype(np.float32)
    fig = show_zyx(xy, xy[:8], xy[:, :8], colormap="gray", vmin=-1, vmax=1,
                   rotate_view=(45, 0, 0), figsize=(4, 4))
    arr = np.asarray(fig.axXY.images[0].get_array(), dtype=float)
    assert np.isnan(arr[0, 0]) and np.isfinite(arr[arr.shape[0] // 2, arr.shape[1] // 2])
    assert list(_rendered_rgb_at(fig, fig.axXY, 0.02, 0.98)) == [0, 0, 0]
    plt.close(fig)


def test_dead_rotation_and_normalisation_helpers_are_gone():
    """R1-21: the unused private helpers ``_norm`` and ``_unrotate_2d`` are removed.

    ``_rotate_expand`` and ``_get_rotated_line`` stay as the documented
    skimage-equivalence test oracles.
    """
    assert not hasattr(tpa, "_norm") and not hasattr(tpa, "_unrotate_2d")
    assert callable(tpa._rotate_expand) and callable(tpa._get_rotated_line)


# ---------------------------------------------------------------------------
# Stage 2 (viewers): deprecations (D08)
# ---------------------------------------------------------------------------

def test_widget_classes_deprecate_legacy_kwargs_and_still_apply_them():
    """D08 / R8-09 / R2-14: the widget classes warn on legacy names and keep honouring them.

    Strategy: build ``TNIASliceWidget(colors=...)`` and
    ``TNIAScatterWidget(..., x_s=..., x_t=...)`` directly.
    Expected outcome: each emits a ``FutureWarning`` naming the replacement
    and removal version 0.1.0; the colour is applied and the scatter slab is
    placed (it used to be dropped silently).
    """
    vol = _structured_volume()
    with pytest.warns(FutureWarning, match=r"'colors'.*0\.1\.0.*'colormap'"):
        w = TNIASliceWidget(vol, colors="red")
    assert w.colors_resolved == ["red"]

    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 40, (200, 3))
    with pytest.warns(FutureWarning, match="'x_s'/'y_s'/'z_s'.*'slabs_position'"):
        ws = tpa.TNIAScatterWidget(pts[:, 2], pts[:, 1], pts[:, 0], x_s=10, y_s=20, z_s=20)
    assert ws.x_s == 10 - int(ws.xmin)
    with pytest.warns(FutureWarning, match="'x_t'/'y_t'/'z_t'.*'slabs_thickness'"):
        ws = tpa.TNIAScatterWidget(pts[:, 2], pts[:, 1], pts[:, 0], x_t=3, y_t=3, z_t=3)
    assert ws.x_t == 3


# ---------------------------------------------------------------------------
# Stage 2 (viewers): slab semantics (D10)
# ---------------------------------------------------------------------------

def _eval_copy_params(w):
    w.copy_params_trigger += 1
    return eval(f"dict({w.copy_params_string})", {"__builtins__": {"dict": dict}}, {})


_SLABS = ("x_start", "x_end", "y_start", "y_end", "z_start", "z_end")


@pytest.mark.parametrize("factory", [show_zyx_max_slice_interactive, show_zyx_max_slice_interactive_point_annotator])
def test_copy_kwargs_round_trip_even_and_odd_width_slabs(factory):
    """D10 controls: copy-kwargs of an even-width slab rebuild the identical view.

    Strategy: set an even-width X slab [10, 13] (4 voxels) and odd-width Y/Z
    slabs, copy the kwargs and rebuild the widget from them.
    Expected outcome (+): every inclusive [start, end] matches, so the drawn
    slabs are identical; the emitted position/thickness hold half-voxels
    (11.5 and 1.5 voxels). (−) The odd widths give the same integer
    centre/half-thickness as before. (The (s, t) form used to drop one voxel
    of an even width: [10, 13] came back as [10, 12].)
    """
    vol = _structured_volume((16, 48, 48))
    ps = {"Z": 2.0, "Y": 0.5, "X": 0.5}
    w = factory(vol, pixel_sizes=ps, figsize=(4, 4))
    w.x_start, w.x_end = 10, 13
    w.y_start, w.y_end = 20, 26
    w.z_start, w.z_end = 4, 8
    params = _eval_copy_params(w)
    assert params["slabs_position"] == (6 * 2.0, 23 * 0.5, 11.5 * 0.5)
    assert params["slabs_thickness"] == (2 * 2.0, 3 * 0.5, 1.5 * 0.5)
    rebuilt = factory(vol, pixel_sizes=ps, figsize=(4, 4),
                      slabs_position=params["slabs_position"], slabs_thickness=params["slabs_thickness"])
    assert [getattr(rebuilt, a) for a in _SLABS] == [10, 13, 20, 26, 4, 8]
    assert (rebuilt.y_s, rebuilt.y_t, rebuilt.z_s, rebuilt.z_t) == (23, 3, 6, 2)


def test_scatter_slabs_are_physical_like_the_image_viewer_and_round_trip():
    """D10 / R2-12 / R8-09: the same slab kwargs select the same slabs in the image and scatter viewers.

    Strategy: pixel sizes (2, 0.5, 0.5), position (20, 50, 50) and thickness
    (4, 5, 5) (physical), for a 40 x 200 x 200 image and for points covering
    the same index box (data coordinates 0..39, 0..199, 0..199); then an
    even-width scatter slab through copy-kwargs.
    Expected outcome: both viewers select X [90, 110] and Z [8, 12] (the
    scatter used to take the numbers as data units: X [45, 55], Z [16, 24]);
    the even-width scatter slab round-trips exactly.
    """
    ps = (2.0, 0.5, 0.5)
    kw = dict(pixel_sizes=ps, slabs_position=(20, 50, 50), slabs_thickness=(4, 5, 5))
    wi = show_zyx_max_slice_interactive(np.zeros((40, 200, 200), np.uint8), figsize=(4, 4), **kw)
    corners = np.array([[0, 0, 0], [39, 199, 199]], dtype=float)
    rng = np.random.default_rng(0)
    pts = np.vstack([corners, rng.uniform(0, 1, (100, 3)) * [39, 199, 199]])
    wsc = show_zyx_max_scatter_interactive(pts, render="points", figsize=(4, 4), **kw)
    for w in (wi, wsc):
        assert (w.x_start, w.x_end, w.z_start, w.z_end) == (90, 110, 8, 12)

    wsc.x_start, wsc.x_end = 41, 50
    params = _eval_copy_params(wsc)
    rebuilt = show_zyx_max_scatter_interactive(
        pts, render="points", figsize=(4, 4), pixel_sizes=ps,
        slabs_position=params["slabs_position"], slabs_thickness=params["slabs_thickness"])
    assert [getattr(rebuilt, a) for a in _SLABS] == [getattr(wsc, a) for a in _SLABS]


def test_hover_sync_keeps_an_even_slab_width():
    """D10 / R2-01: hover sync moves the drawn slab without changing its width.

    Strategy: an even-width X slab [10, 13] (4 voxels), then hover the XY
    panel over voxel x = 30.
    Expected outcome: the slab becomes [29, 32] (centre index 30, still 4
    voxels; the (s, t) path made it 3 voxels wide); Y keeps its width too.
    """
    w = show_zyx_max_slice_interactive(_structured_volume((16, 48, 48)), figsize=(4, 4))
    w.x_start, w.x_end = 10, 13
    y_width = w.y_end - w.y_start
    fig = w._render()
    fx, fy = _fig_frac(fig, fig.axXY, 30.5, 20.5)
    w.hover_coords = {"plane": "xy", "x": fx, "y": fy, "t": 1}
    assert (w.x_start, w.x_end) == (29, 32) and w.x_s == 30
    assert w.y_end - w.y_start == y_width and w.y_s == 20


def test_annotator_delete_uses_the_drawn_slab_and_a_screen_pixel_radius():
    """D10 / R2-01 / R2-06: deletion selects points in the drawn slab, within ~12 screen px.

    Strategy: 60 x 256 x 256 volume with pixel sizes (5, 0.2, 0.2) (the Z
    panels are wide on screen), even-width Z slab; (a) a point on the last
    plane of the even slab, clicked exactly; (b) a ZY click 19 z-slices
    (95 µm) from a point; (c) an XY click about 6 screen px from a point;
    (d) an XY click about 30 screen px away.
    Expected outcome: (a) deleted (the (s, t) form excluded that plane);
    (b) kept (it was deleted: the radius was 20 voxel indices); (c) deleted;
    (d) kept.
    """
    vol = np.zeros((60, 256, 256), np.uint8)
    w = show_zyx_max_slice_interactive_point_annotator(vol, pixel_sizes=(5, 0.2, 0.2), figsize=(8, 8))
    w.annotation_mode = True
    w.annotation_action = "delete"

    def click(plane, a, b, t):
        fig = w._render()
        ax = getattr(fig, f"ax{plane.upper()}")
        fx, fy = _fig_frac(fig, ax, a, b)
        w.click_coords = {"plane": plane, "x": fx, "y": fy, "t": t}

    w.z_start, w.z_end = 30, 33
    w.points = [[33, 128, 128]]
    click("xy", 128.5 * 0.2, 128.5 * 0.2, 1)
    assert w.points == []

    w.points = [[30, 128, 128]]
    w.x_start, w.x_end = 127, 129
    click("zy", (30 + 19 + 0.5) * 5, 128.5 * 0.2, 2)
    assert w.points == [[30, 128, 128]]

    fig = w._render()
    um_per_px = (fig.axXY.get_xlim()[1] - fig.axXY.get_xlim()[0]) / (
        w.axis_bounds["xy"]["x1"] - w.axis_bounds["xy"]["x0"]) / (fig.get_figwidth() * fig.dpi)
    w.z_start, w.z_end = 29, 31
    click("xy", 128.5 * 0.2 + 6 * um_per_px, 128.5 * 0.2, 3)
    assert w.points == []

    w.points = [[30, 128, 128]]
    click("xy", 128.5 * 0.2 + 30 * um_per_px, 128.5 * 0.2, 4)
    assert w.points == [[30, 128, 128]]


def test_half_voxel_slab_kwargs_select_even_widths():
    """D10: ``(s, t)`` accepts half-voxels, so even widths can be requested directly.

    Strategy: position 11.5 voxels and half-thickness 1.5 voxels on X (unit
    pixel sizes); negative control: integer position 11 with half-thickness 1.
    Expected outcome: X [10, 13] (4 voxels) and X [10, 12] (3 voxels).
    """
    vol = _structured_volume((16, 48, 48))
    w = show_zyx_max_slice_interactive(vol, slabs_position=(8, 24, 11.5), slabs_thickness=(1, 1, 1.5), figsize=(4, 4))
    assert (w.x_start, w.x_end) == (10, 13)
    w = show_zyx_max_slice_interactive(vol, slabs_position=(8, 24, 11), slabs_thickness=(1, 1, 1), figsize=(4, 4))
    assert (w.x_start, w.x_end) == (10, 12)


# ---------------------------------------------------------------------------
# Stage 2 (viewers): depth of annotated points (D11)
# ---------------------------------------------------------------------------

def _click_voxel(w, plane, z, y, x, t):
    fig = w._render()
    ax = getattr(fig, f"ax{plane.upper()}")
    a, b = {"xy": (x, y), "zy": (z, y), "xz": (x, z)}[plane]
    sa, sb = {"xy": (w.sx, w.sy), "zy": (w.sz, w.sy), "xz": (w.sx, w.sz)}[plane]
    fx, fy = _fig_frac(fig, ax, (a + 0.5) * sa, (b + 0.5) * sb)
    w.click_coords = {"plane": plane, "x": fx, "y": fy, "t": t}


@pytest.mark.parametrize("plane", ["xy", "zy", "xz"])
def test_annotation_on_a_projection_snaps_to_the_depth_of_the_clicked_object(plane):
    """D11 / R2-08: a click on a thick-slab MIP stores the depth of the bright voxel, not the slab centre.

    Strategy: a spot at (z, y, x) = (5, 30, 40) in a 40 x 64 x 64 volume,
    full-volume slabs, click the spot on each panel; then the same with
    ``snap_depth=False``, and a hidden (opacity 0) bright channel elsewhere
    on the same ray.
    Expected outcome: the stored point is (5, 30, 40) (the slab centre used
    to be stored: e.g. z = 20 on XY); with ``snap_depth=False`` the
    out-of-plane coordinate is the slab centre; the hidden channel is
    ignored.
    """
    vol = np.zeros((40, 64, 64), np.float32)
    vol[5, 30, 40] = 100.0
    other = np.zeros_like(vol)
    other[30, 30, 40] = other[25, 30, 40] = 1000.0
    other[5, 50, 40] = other[5, 30, 60] = 1000.0
    full = dict(slabs_thickness=(1000, 1000, 1000), figsize=(4, 4))

    w = show_zyx_max_slice_interactive_point_annotator([vol, other], opacity=[1, 0], annotate=True, **full)
    _click_voxel(w, plane, 5, 30, 40, 1)
    assert w.points == [[5, 30, 40]]

    w = show_zyx_max_slice_interactive_point_annotator(vol, annotate=True, snap_depth=False, **full)
    _click_voxel(w, plane, 5, 30, 40, 1)
    centre = {"xy": [20, 30, 40], "zy": [5, 30, 32], "xz": [5, 32, 40]}[plane]
    assert w.points == [centre]


def test_snap_depth_falls_back_to_the_centre_for_flat_profiles_and_average():
    """D11 negative controls: a flat ray and the 'average' projection keep the slab centre.

    Strategy: click an all-zero volume (flat profile), and a spot volume drawn
    with ``projection='average'`` (no single voxel is shown).
    Expected outcome: the stored depth is the slab centre in both cases.
    """
    vol = np.zeros((40, 64, 64), np.float32)
    w = show_zyx_max_slice_interactive_point_annotator(vol, annotate=True, slabs_thickness=(1000, 1000, 1000))
    _click_voxel(w, "xy", 5, 30, 40, 1)
    assert w.points == [[20, 30, 40]]
    vol[5, 30, 40] = 1.0
    w = show_zyx_max_slice_interactive_point_annotator(vol, annotate=True, projection="average",
                                                       slabs_thickness=(1000, 1000, 1000))
    _click_voxel(w, "xy", 5, 30, 40, 1)
    assert w.points == [[20, 30, 40]]


# ---------------------------------------------------------------------------
# Stage 2 (viewers): static wrapper API (D12)
# ---------------------------------------------------------------------------

def test_show_zyx_max_slabs_defaults_to_the_full_range_and_clamps_with_a_warning():
    """D12 / R1-17: ``show_zyx_max_slabs`` without ranges is the full MIP; ranges are clamped with a warning.

    Strategy: one bright voxel at (5, 15, 15) of a 10 x 30 x 30 volume;
    (a) no x/y/z; (b) x=[-1, 31] (beyond both ends); (c) x=[30, 31] and
    x=[40, 50] (entirely outside); (d) a valid range (negative control).
    Expected outcome: (a) every panel equals ``show_zyx_max`` (the XY max was
    0: only plane 0 was projected); (b) a UserWarning and the full X range;
    (c) a ValueError naming the empty range (it was 'zero-size array to
    reduction operation'); (d) no warning.
    """
    v = np.zeros((10, 30, 30), np.float32)
    v[5, 15, 15] = 1.0
    fig, ref = show_zyx_max_slabs(v), show_zyx_max(v)
    for name in ("axXY", "axXZ", "axZY"):
        np.testing.assert_array_equal(_panel_rgb(fig, name), _panel_rgb(ref, name))
    assert _panel_rgb(fig).max() == 1.0

    with pytest.warns(UserWarning, match="clipped"):
        fig = show_zyx_max_slabs(v, x=[-1, 31], y=[0, 30], z=[0, 10])
    np.testing.assert_array_equal(_panel_rgb(fig, "axXZ"), _panel_rgb(ref, "axXZ"))
    for bad in ([30, 31], [40, 50]):
        with pytest.raises(ValueError, match="x range"):
            show_zyx_max_slabs(v, x=bad, y=[0, 30], z=[0, 10])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        show_zyx_max_slabs(v, x=[10, 20], y=[0, 30], z=[0, 10])
    assert inspect.signature(show_zyx_max_slabs).parameters["x"].default is None


_DISPLAY_KWARGS = {
    "pixel_sizes", "figsize", "colormap", "vmin", "vmax", "gamma", "opacity", "blending",
    "rotate_view", "channel_labels", "channel_label_fontsize_pt", "scale_bar_show",
    "scale_bar_fontsize_pt", "subplot_bg", "use_plt", "channel_axis", "colors",
}


def test_every_static_wrapper_takes_the_same_display_kwargs_and_input_forms():
    """D12 / R1-16 / R8-08: one keyword set and one input normaliser for every static wrapper.

    Strategy: compare the signatures of show_zyx_slice / _max / _projection /
    _max_slabs / _projection_slabs; then draw the same two channels as a
    list, a tuple, a CZYX ndarray with ``channel_axis=0`` and a CZYX
    DataArray (with a ``color`` coordinate); then a 2D image.
    Expected outcome: every wrapper takes the shared display kwargs; all
    input forms give identical panels (the tuple, ndarray and DataArray
    raised cryptic TypeErrors); a 4D ndarray without ``channel_axis`` raises
    a ValueError asking for it (no guessing between CZYX and TZYX); a 2D
    image draws in every wrapper; ``show_zyx_slice`` takes a channel list.
    """
    wrappers = [show_zyx_slice, show_zyx_max, show_zyx_projection, show_zyx_max_slabs,
                tpa.show_zyx_projection_slabs]
    for f in wrappers:
        missing = _DISPLAY_KWARGS - set(inspect.signature(f).parameters)
        assert not missing, (f.__name__, missing)
    for f in wrappers[1:]:
        assert {"projection", "attenuation"} <= set(inspect.signature(f).parameters), f.__name__

    rng = np.random.default_rng(0)
    c0, c1 = (rng.random((4, 12, 10)).astype(np.float32) for _ in range(2))
    kw = dict(colormap=["cyan", "yellow"], figsize=(4, 4))
    ref = show_zyx_max([c0, c1], **kw)
    for im, extra in (((c0, c1), {}), (np.stack([c0, c1]), {"channel_axis": 0}),
                      (np.stack([c0, c1], axis=-1), {"channel_axis": -1})):
        fig = show_zyx_max(im, **kw, **extra)
        np.testing.assert_array_equal(_panel_rgb(fig), _panel_rgb(ref))
    with pytest.raises(ValueError, match="channel_axis"):
        show_zyx_max(np.stack([c0, c1]))

    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.stack([c0, c1]), dims=("C", "Z", "Y", "X"),
                      coords={"C": ["a", "b"], "color": ("C", ["cyan", "yellow"])})
    fig = show_zyx_max(da, figsize=(4, 4))
    np.testing.assert_array_equal(_panel_rgb(fig), _panel_rgb(ref))
    assert fig.axLabels is not None  # channel labels from the C coordinate
    w = TNIASliceWidget(da)
    assert w.channel_names == ["a", "b"] and w.colors_resolved == ["cyan", "yellow"]

    img2d = rng.random((12, 10)).astype(np.float32)
    for f, args in ((show_zyx_max, ()), (show_zyx_projection, ()), (show_zyx_max_slabs, ()),
                    (tpa.show_zyx_projection_slabs, (None, None, None)), (show_zyx_slice, (2, 3, 0))):
        assert _panel_rgb(f(img2d, *args, figsize=(4, 4))).shape == (12, 10)
    fig = show_zyx_slice([c0, c1], 2, 3, 1, colormap=["cyan", "yellow"], figsize=(4, 4))
    assert _panel_rgb(fig).shape == (12, 10, 3)


def _panel_block_fraction(fig):
    """Fraction of the figure height and width spanned by the ortho panels (and label row)."""
    axes = [fig.axXY, fig.axXZ, fig.axZY] + ([fig.axLabels] if fig.axLabels is not None else [])
    boxes = [a.get_position() for a in axes]
    return (max(b.y1 for b in boxes) - min(b.y0 for b in boxes),
            max(b.x1 for b in boxes) - min(b.x0 for b in boxes))


def test_auto_figsize_follows_the_physical_extents():
    """D12 / R1-18: ``figsize=None`` (now the default) sizes the figure from the data's physical extents.

    Strategy: a flat 20 x 100 x 1000 volume and a cube through ``show_zyx_max``
    (default figsize); the flat volume again with channel labels.
    Expected outcome: the panels fill the figure (> 90% of its height and
    width; they filled 12% of a fixed 10 x 10 figure); the figure fits a 10 in
    box; an explicit figsize is kept.
    """
    flat = np.zeros((20, 100, 1000), np.uint8)
    for im, labels in ((flat, None), (flat, ["a"]), (np.zeros((30, 30, 30), np.uint8), None)):
        fig = show_zyx_max([im] if labels else im, channel_labels=labels)
        fh, fw = _panel_block_fraction(fig)
        assert fh > 0.9 and fw > 0.9, (im.shape, labels, fh, fw)
        assert max(fig.get_size_inches()) == pytest.approx(10.0)
    assert tuple(show_zyx_max(flat, figsize=(6, 5)).get_size_inches()) == (6, 5)
    for f in (show_zyx_slice, show_zyx_max, show_zyx_projection, show_zyx_max_slabs, tpa.show_zyx_projection_slabs, show_zyx):
        assert inspect.signature(f).parameters["figsize"].default is None, f.__name__


def test_one_auto_font_size_tied_to_the_figure_width():
    """R1-23 / D12: channel labels and the scale bar share one auto font size, set by the figure width.

    Strategy: draw the same volume with channel labels at figsize (8, 3) and
    (8, 12), and at (4, 8) and (12, 8).
    Expected outcome: the label and the scale-bar text have the same size;
    it does not change with the height (thin volumes keep readable text) and
    grows with the width. (The two formulas differed, 10-24 pt from 3.5% vs
    8-24 pt from 3% of the height.)
    """
    vol = np.random.default_rng(0).random((10, 40, 40)).astype(np.float32)

    def sizes(figsize):
        fig = show_zyx_max([vol], channel_labels=["a"], pixel_sizes=(1, 1, 1), figsize=figsize)
        return _label_fontsize(fig), _scale_bar_artists(fig)[1].get_fontsize()

    a, b = sizes((8, 3)), sizes((8, 12))
    assert a[0] == a[1] and a == b
    narrow, wide = sizes((4, 8)), sizes((12, 8))
    assert narrow[0] == narrow[1] and narrow[0] < wide[0]


def test_widget_auto_figsize_uses_physical_extents_and_classes_match_factories():
    """D12 / R2-22 / R2-13: every viewer uses one auto-figsize rule, from physical extents.

    Strategy: points covering X 0..500 and Y 0..500 with pixel sizes
    (1, 4, 1) (Y physically 4x longer), through the scatter factory; an
    image of the same physical shape through the slice factory; the classes
    built directly without ``figsize``.
    Expected outcome: both factory figures are about 4x taller than wide
    (the scatter one ignored py and was square); a class built without
    figsize gets the same figure size as its factory (the classes used to
    draw at 10 x 10).
    """
    rng = np.random.default_rng(0)
    pts = np.vstack([[0, 0, 0], [10, 500, 500], rng.uniform(0, 1, (50, 3)) * [10, 500, 500]])
    ws = show_zyx_max_scatter_interactive(pts, pixel_sizes=(1, 4, 1), render="points")
    w_h = ws.figsize[1] / ws.figsize[0]
    assert 3.0 < w_h < 4.5, ws.figsize
    wi = show_zyx_max_slice_interactive(np.zeros((11, 501, 501), np.uint8), pixel_sizes=(1, 4, 1))
    assert 3.0 < wi.figsize[1] / wi.figsize[0] < 4.5, wi.figsize

    cls = tpa.TNIAScatterWidget(pts[:, 2], pts[:, 1], pts[:, 0], pixel_sizes=(1, 4, 1))
    np.testing.assert_allclose(cls.figsize, ws.figsize)
    vol = _structured_volume()
    np.testing.assert_allclose(TNIASliceWidget(vol).figsize, show_zyx_max_slice_interactive(vol).figsize)
