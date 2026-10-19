import sys
import textwrap
import warnings

from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pytest
import types

from eigenp_utils.plotting_utils import labels_cmap
from eigenp_utils.tnia_plotting_anywidgets import resolve_color, show_zyx_max_slice_interactive
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slabs, show_zyx_max_slice_interactive
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, TNIASliceWidget
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, show_zyx_max_scatter_interactive
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, show_zyx_max_slabs, show_zyx
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive_point_annotator, show_zyx_max_scatter_interactive
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_slice, show_zyx_max_slabs, create_multichannel_rgb
from eigenp_utils.tnia_plotting_anywidgets import compute_histogram, show_iso_scatter, IsoScatterWidget



# =========================================
# Source: test_tnia_plotting_anywidgets.py
# =========================================
matplotlib.use("Agg")

def test_single_channel_instantiation():
    """Test that single channel instantiation works as expected."""
    im = np.zeros((10, 20, 30))
    w = show_zyx_max_slice_interactive(im)
    assert isinstance(w, TNIASliceWidget)
    assert w.num_channels == 1
    assert len(w.channel_names) == 1
    # Render should produce image data (triggered by observer in init)
    assert w.image_data is not None and len(w.image_data) > 0

def test_multi_channel_instantiation():
    """Test that multi channel instantiation works as expected."""
    im = [np.zeros((10, 20, 30)) for _ in range(3)]
    w = show_zyx_max_slice_interactive(im)
    assert w.num_channels == 3
    assert len(w.channel_names) == 3
    assert w.channel_names == ["Channel 0", "Channel 1", "Channel 2"]
    assert w.opacity_list == [1.0, 1.0, 1.0]
    assert w.image_data is not None and len(w.image_data) > 0

def test_channel_visibility_update():
    """Test that channel visibility update works as expected."""
    # Use different values to ensure visual difference
    im = [np.zeros((10, 10, 10)), np.ones((10, 10, 10)) * 255]
    w = show_zyx_max_slice_interactive(im)

    initial_data = w.image_data
    assert initial_data

    # Hide channel 1 (the bright one)
    w.opacity_list = [1.0, 0.0]

    # Check that image data changed (re-rendered)
    new_data = w.image_data
    assert new_data != initial_data

    # Hide all channels
    w.opacity_list = [0.0, 0.0]
    empty_data = w.image_data
    assert empty_data != new_data
    assert empty_data != initial_data

def test_default_colors_resolution():
    """Test that default colors resolution works as expected."""
    im = [np.zeros((10, 10, 10)) for _ in range(2)]
    w = show_zyx_max_slice_interactive(im, colormap=None)
    assert w.colors_resolved == ['white', 'lime'] # Defaults

    w2 = show_zyx_max_slice_interactive(im, colormap=['red', 'blue'])
    assert w2.colors_resolved == ['red', 'blue']

def test_show_zyx_max_slice_interactive_point_annotator_args():
    """Test that show zyx max slice interactive point annotator args works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive_point_annotator, TNIAAnnotatorWidget
    im = [np.zeros((10, 10, 10)) for _ in range(2)]
    w = show_zyx_max_slice_interactive_point_annotator(
        im,
        pixel_sizes=(3, 2, 2),
        point_size_scale=0.05,
        colormap=['red', 'blue'],
        opacity=[0.5, 0.8]
    )
    assert isinstance(w, TNIAAnnotatorWidget)
    assert w.sx == 2
    assert w.sy == 2
    assert w.sz == 3

def test_point_size_scaling():
    """Test that point size scaling works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import TNIAAnnotatorWidget
    im = np.zeros((1, 100, 100)) # Thin Z dimension

    w1 = TNIAAnnotatorWidget(im, point_size_scale=0.1)
    w2 = TNIAAnnotatorWidget(im, point_size_scale=0.5)

    # Verify that the point size scales properly with the X/Y dimension (min(100, 100) = 100)
    # 0.1 * 100 = 10
    # 0.5 * 100 = 50
    assert w1.point_size == 10
    assert w2.point_size == 50
    assert w1.point_size < w2.point_size

def test_show_zyx_max_scatter_interactive_colormap():
    """Test that show zyx max scatter interactive colormap works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_scatter_interactive
    X = np.random.rand(10) * 10
    Y = np.random.rand(10) * 10
    Z = np.random.rand(10) * 10
    channels = np.random.rand(10)

    # Should not throw exception for invalid RGBA string
    w1 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels, colormap='viridis', render='points')
    w2 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels, colormap='viridis', render='density')

    channels_multi = [np.random.rand(10), np.random.rand(10)]
    w3 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels_multi, colormap=['viridis', 'plasma'], render='points')
    w4 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels_multi, colormap=['viridis', 'plasma'], render='density')

    assert w1 is not None
    assert w2 is not None
    assert w3 is not None
    assert w4 is not None

def test_show_zyx_max_scatter_interactive_signature():
    """Test that show zyx max scatter interactive signature works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_scatter_interactive
    X = np.random.rand(10) * 10
    Y = np.random.rand(10) * 10
    Z = np.random.rand(10) * 10
    channels = np.random.rand(10)

    # Test with tuple
    w1 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels, render='points')
    assert w1 is not None

    # Test with list
    w2 = show_zyx_max_scatter_interactive([Z, Y, X], channels=channels, render='points')
    assert w2 is not None

    # Test with (N, 3) array
    points = np.stack([Z, Y, X], axis=1)
    w3 = show_zyx_max_scatter_interactive(points, channels=channels, render='points')
    assert w3 is not None

    # Verify that the parsed data inside is correct
    np.testing.assert_array_equal(w3.X_arr, X)
    np.testing.assert_array_equal(w3.Y_arr, Y)
    np.testing.assert_array_equal(w3.Z_arr, Z)

    # Test invalid shape
    with pytest.raises(ValueError, match="points must be an array of shape .* representing \\(Z, Y, X\\) or a tuple/list of 3 arrays \\(Z, Y, X\\)."):
        invalid_points = np.stack([Z, Y], axis=1)
        show_zyx_max_scatter_interactive(invalid_points, channels=channels)

def test_colormap_list_multi_channel():
    """Test that colormap list multi channel works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive
    im = [np.zeros((10, 20, 30)) for _ in range(3)]
    # This should not raise any TypeError about unhashable lists
    w = show_zyx_max_slice_interactive(im, colormap=['red', 'blue', 'green'])
    assert w is not None
    assert w.colors_resolved == ['red', 'blue', 'green']

def test_colormap_list_scatter_multi_channel():
    """Test that colormap list scatter multi channel works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_scatter_interactive
    X = np.random.rand(10) * 10
    Y = np.random.rand(10) * 10
    Z = np.random.rand(10) * 10
    channels_multi = [np.random.rand(10), np.random.rand(10)]
    w = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels_multi, colormap=['viridis', 'plasma'], render='points')
    assert w is not None

def test_deprecation_warning_colors():
    """Test that deprecation warning colors works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive
    im = np.zeros((10, 20, 30))
    with pytest.warns(DeprecationWarning, match="The 'colors' parameter is deprecated and will be removed. Use 'colormap' instead."):
        w = show_zyx_max_slice_interactive(im, colors=['red'])
        assert w is not None


@pytest.mark.parametrize("factory_fn", [
    show_zyx_max_slice_interactive,
    show_zyx_max_slice_interactive_point_annotator,
])
def test_interactive_kwargs_images(factory_fn):
    """Test that interactive kwargs images works as expected."""
    im = np.zeros((10, 20, 30))
    w = factory_fn(im,
                   show_crosshair=False,
                   sync_on_hover=True,
                   slabs_thickness=(2, 3, 4),
                   slabs_position=(5, 10, 15),
                   pixel_sizes={'Z': 2.0, 'Y': 1.0, 'X': 0.5})

    assert w.show_crosshair is False
    assert w.sync_on_hover is True
    # Because pixel_sizes are set to Z=2.0, Y=1.0, X=0.5
    # slabs_thickness in physical units: (2, 3, 4)
    # The indices will be calculated as thickness // p => (2/2.0=1, 3/1.0=3, 4/0.5=8)
    assert w.z_t == 1 and w.y_t == 3 and w.x_t == 8

    # Note: slabs_position in physical units: (5, 10, 15)
    # The indices will be calculated as pos // p => (5/2.0=2.5->2, 10/1.0=10, 15/0.5=30)
    # clamped because coords map to dims [10, 20, 30] (max 9, 19, 29)
    assert w.z_s == 2 and w.y_s == 10 and w.x_s == 29
    assert w.sz == 2.0 and w.sy == 1.0 and w.sx == 0.5

def test_interactive_kwargs_scatter():
    """Test that interactive kwargs scatter works as expected."""
    X, Y, Z = np.random.rand(10), np.random.rand(10), np.random.rand(10)
    w = show_zyx_max_scatter_interactive((X, Y, Z),
                   show_crosshair=False,
                   sync_on_hover=True,
                   slabs_thickness=(2, 3, 4),
                   slabs_position=(5, 10, 15),
                   pixel_sizes={'Z': 2.0, 'Y': 1.0, 'X': 0.5})

    assert w.show_crosshair is False
    assert w.sync_on_hover is True
    assert w.z_t == 2 and w.y_t == 3 and w.x_t == 4
    # Note: slabs_position clamped because random coords are [0, 1] mapped to Dim=2
    assert w.z_s == 1 and w.y_s == 1 and w.x_s == 1
    assert w.sz == 2.0 and w.sy == 1.0 and w.sx == 0.5

@pytest.mark.parametrize("factory_fn", [
    show_zyx_max_slice_interactive,
    show_zyx_max_slice_interactive_point_annotator,
])
def test_deprecation_warnings_interactive(factory_fn):
    """Test that deprecation warnings interactive works as expected."""
    im = np.zeros((10, 20, 30))
    with pytest.warns(DeprecationWarning, match="The 'sxy' and 'sz' parameters are deprecated"):
        factory_fn(im, sxy=0.5, sz=2.0)
    with pytest.warns(DeprecationWarning, match="The 'x_s', 'y_s', 'z_s' parameters are deprecated"):
        factory_fn(im, x_s=5, y_s=10, z_s=5)
    with pytest.warns(DeprecationWarning, match="The 'x_t', 'y_t', 'z_t' parameters are deprecated"):
        factory_fn(im, x_t=2, y_t=3, z_t=4)

@pytest.mark.parametrize("shape, pixel_sizes, expected_text", [
    ((100, 200, 300), (1, 2, 3), '100 µm'),
    ((10, 50, 50), (0.5, 0.5, 0.5), '5 µm'),
    ((1, 5, 5), (10, 10, 10), '10 µm'),
    ((50, 100, 150), (2, 2, 2), '50 µm'),
    ((100, 100, 100), (1, 1, 1), '20 µm'),      # explicitly isotropic
    ((10, 512, 512), (10, 2, 2), '200 µm')      # user requested anisotropic
])
def test_scale_bar_logic(shape, pixel_sizes, expected_text):
    """Test that scale bar logic works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, show_zyx_max_scatter_interactive
    import numpy as np

    # Test slice interactive
    im = np.zeros(shape)
    w_slice = show_zyx_max_slice_interactive(im, pixel_sizes=pixel_sizes, figsize=(5,5))
    fig_slice = w_slice._render()

    # Extract text from scale bar
    texts_slice = [txt.get_text() for ax in fig_slice.axes for txt in ax.texts]
    assert expected_text in texts_slice

    # Extract fontsize of the scale bar
    font_size_unscaled = None
    for ax in fig_slice.axes:
        for txt in ax.texts:
            if expected_text in txt.get_text():
                font_size_unscaled = txt.get_fontsize()

    # Test scatter interactive
    Z, Y, X = shape
    # Make points such that Z_dim=Z, Y_dim=Y, X_dim=X
    points = (np.array([0, Z-1]), np.array([0, Y-1]), np.array([0, X-1]))
    w_scatter = show_zyx_max_scatter_interactive(points, pixel_sizes=pixel_sizes, figsize=(5,5))
    fig_scatter = w_scatter._render()

    texts_scatter = [txt.get_text() for ax in fig_scatter.axes for txt in ax.texts]
    assert expected_text in texts_scatter

    # Test with figsize_scale to make sure it scales font and lines correctly
    w_slice_scaled = show_zyx_max_slice_interactive(im, pixel_sizes=pixel_sizes, figsize=(5,5), figsize_scale=2.0)
    fig_slice_scaled = w_slice_scaled._render()

    def _bar_linewidth(fig):
        widths = [c.get_linewidth()[0] for ax in fig.axes for c in ax.collections
                  if len(c.get_linewidth())]
        return max(widths) if widths else None

    font_size_scaled = None
    for ax in fig_slice_scaled.axes:
        for txt in ax.texts:
            if expected_text in txt.get_text():
                font_size_scaled = txt.get_fontsize()
    linewidth_unscaled = _bar_linewidth(fig_slice)
    linewidth_scaled = _bar_linewidth(fig_slice_scaled)

    assert font_size_scaled is not None
    assert font_size_scaled > font_size_unscaled
    assert linewidth_unscaled is not None and linewidth_scaled is not None
    assert linewidth_scaled > linewidth_unscaled

def test_parse_zyx_tuple_or_dict_various_types():
    """Test that parse zyx tuple or dict various types works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import _parse_zyx_tuple_or_dict

    # 1. Native tuple of floats
    res = _parse_zyx_tuple_or_dict((1.5, 2.5, 3.5))
    assert res == (1.5, 2.5, 3.5)

    # 2. Native list of floats
    res = _parse_zyx_tuple_or_dict([1.5, 2.5, 3.5])
    assert res == (1.5, 2.5, 3.5)

    # 3. Dict with floats
    res = _parse_zyx_tuple_or_dict({'Z': 1.5, 'Y': 2.5, 'X': 3.5})
    assert res == (1.5, 2.5, 3.5)

    # 4. Tuple with np.ndarrays (0-d arrays / scalars)
    res = _parse_zyx_tuple_or_dict((np.array(1.5), np.array(2.5), np.array(3.5)))
    assert res == (1.5, 2.5, 3.5)
    assert isinstance(res[0], float)

    # 5. Dict with np.ndarrays
    res = _parse_zyx_tuple_or_dict({'Z': np.array(1.5), 'Y': np.array(2.5), 'X': np.array(3.5)})
    assert res == (1.5, 2.5, 3.5)
    assert isinstance(res[0], float)

    # 6. Very small/large values
    res = _parse_zyx_tuple_or_dict([1e-6, 1e6, 0.0])
    assert res == (1e-6, 1e6, 0.0)

def test_show_zyx_max_slabs_zero_sized_slices():
    """Test that show zyx max slabs zero sized slices works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slabs, show_zyx_max_slice_interactive
    im = np.random.rand(10, 10, 10)

    # Passing identical float intervals, should be coerced to integer intervals
    # of size 1 (i.e. x=[0, 1]) without raising a ValueError.
    fig = show_zyx_max_slabs(im, x=[0.5, 0.5], y=[0.0, 0.0], z=[0, 0])
    assert fig is not None

    # Additionally test the interactive wrapper passing 0-d np scalars
    w = show_zyx_max_slice_interactive(im, pixel_sizes=(np.array(1.5), np.array(2.5), np.array(3.5)))
    assert w.sz == 1.5
    assert w.sy == 2.5
    assert w.sx == 3.5
    assert isinstance(w.sz, float)

def test_interactive_channel_labels():
    """Test that interactive channel labels works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import (
        show_zyx_max_slice_interactive,
        show_zyx_max_slice_interactive_point_annotator,
        show_zyx_max_scatter_interactive
    )
    import numpy as np

    im = np.zeros((10, 20, 30))
    labels = ["DAPI", "GFP"]

    w1 = show_zyx_max_slice_interactive(
        [im, im],
        channel_labels=labels
    )
    assert w1.channel_labels_input == labels
    assert w1.channel_names == labels

    w2 = show_zyx_max_slice_interactive_point_annotator(
        [im, im],
        channel_labels=labels
    )
    assert w2.channel_labels_input == labels
    assert w2.channel_names == labels + ['Annotations']

    X = np.random.rand(10) * 10
    Y = np.random.rand(10) * 10
    Z = np.random.rand(10) * 10
    channels_multi = [np.random.rand(10), np.random.rand(10)]

    w3 = show_zyx_max_scatter_interactive(
        (Z, Y, X),
        channels=channels_multi,
        channel_labels=labels,
        render='points'
    )
    assert w3.channel_labels_input == labels

# =========================================
# Source: test_tnia_annotator_widget.py
# =========================================
def test_tnia_annotator_widget(tmp_path, monkeypatch):
    """The annotator writes its points as a z,y,x CSV (with $VAR expansion in the name)."""
    from eigenp_utils.tnia_plotting_anywidgets import TNIAAnnotatorWidget

    monkeypatch.setenv('TEST_DIR', str(tmp_path))

    im = np.random.randint(0, 255, (10, 10, 10), dtype=np.uint8)
    w = TNIAAnnotatorWidget(im)
    w.points = [[5, 4, 3], [6, 7, 8]]
    w.save_csv_filename = "$TEST_DIR/test_points.csv"
    w._save_csv(None)

    out = tmp_path / "test_points.csv"
    assert out.exists()
    assert out.read_text() == "z,y,x\n5,4,3\n6,7,8\n"


def test_sync_on_hover_defaults_to_true():
    """Test that sync_on_hover defaults to True for all interactive widgets and wrapper functions."""
    from eigenp_utils.tnia_plotting_anywidgets import (
        TNIASliceWidget,
        TNIAScatterWidget,
        TNIAAnnotatorWidget,
        show_zyx_max_slice_interactive,
        show_zyx_max_slice_interactive_point_annotator,
        show_zyx_max_scatter_interactive,
    )

    im = np.zeros((10, 20, 30))
    X, Y, Z = np.random.rand(10), np.random.rand(10), np.random.rand(10)

    # Class instantiations
    w_slice = TNIASliceWidget(im)
    assert w_slice.sync_on_hover is True

    w_scatter = TNIAScatterWidget(X, Y, Z)
    assert w_scatter.sync_on_hover is True

    w_annot = TNIAAnnotatorWidget(im)
    assert w_annot.sync_on_hover is True

    # Wrapper function instantiations
    w_fn_slice = show_zyx_max_slice_interactive(im)
    assert w_fn_slice.sync_on_hover is True

    w_fn_annot = show_zyx_max_slice_interactive_point_annotator(im)
    assert w_fn_annot.sync_on_hover is True

    w_fn_scatter = show_zyx_max_scatter_interactive((Z, Y, X))
    assert w_fn_scatter.sync_on_hover is True


def test_annotator_annotate_kwarg_and_property():
    """Test annotate kwarg default, explicit boolean settings, and property getter/setter on TNIAAnnotatorWidget."""
    from eigenp_utils.tnia_plotting_anywidgets import (
        TNIAAnnotatorWidget,
        show_zyx_max_slice_interactive_point_annotator,
    )

    im = np.zeros((10, 20, 30))

    # Default should be annotate=False (annotation_mode is False)
    w_default = show_zyx_max_slice_interactive_point_annotator(im)
    assert w_default.annotation_mode is False
    assert w_default.annotate is False

    # Explicit annotate=True during wrapper function creation
    w_active = show_zyx_max_slice_interactive_point_annotator(im, annotate=True)
    assert w_active.annotation_mode is True
    assert w_active.annotate is True

    # Direct class instantiation with annotate=True
    w_class = TNIAAnnotatorWidget(im, annotate=True)
    assert w_class.annotation_mode is True
    assert w_class.annotate is True

    # Test programmatically modifying via annotate property
    w_default.annotate = True
    assert w_default.annotation_mode is True
    assert w_default.annotate is True

    w_default.annotate = False
    assert w_default.annotation_mode is False
    assert w_default.annotate is False

# =========================================
# Source: test_tnia_plotting_anywidgets_resolve_color.py
# =========================================

def test_resolve_color():
    """Test that resolve color works as expected."""
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
    """Test that widget rejects colormap instance works as expected."""
    im = np.zeros((10, 20, 30))
    # This crashed previously due to channel_colors list expecting a unicode string but getting a Colormap instance
    with pytest.raises(TypeError, match="Expected a registered colormap name"):
        w = show_zyx_max_slice_interactive(im, colormap=labels_cmap)

# =========================================
# Source: test_tnia_plotting_rotation.py
# =========================================

def test_rotation_slice_interactive():
    """Test that rotation slice interactive works as expected."""
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
    """Test that rotation scatter interactive works as expected."""
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

# =========================================
# Source: test_marimo_update.py
# =========================================
def test_marimo_update(tmp_path):
    """Verify the reference marimo app snippet stays valid and up to date.

    Strategy: write the snippet to a pytest ``tmp_path`` scratch file (never the
    repository working tree), then compile it to check it is syntactically valid
    Python and inspect its source for the current keyword spelling.

    Expected outcome: the file is created under ``tmp_path``, compiles without a
    ``SyntaxError``, and uses the ``colormap`` keyword rather than the deprecated
    ``colors`` alias.
    """
    pytest.importorskip("marimo")

    app_code = textwrap.dedent("""
    import marimo as mo

    app = mo.App()

    @app.cell
    def __():
        import numpy as np
        from skimage.data import cells3d
        from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

        try:
            im = cells3d()
        except:
            from eigenp_utils.io import download_file
            url_to_fetch = "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif"
            download_file(url_to_fetch, "./cells3d.tif")
            from skimage.io import imread
            im = imread("./cells3d.tif")  # (Z, C, Y, X)
        membrane = im[:, 0, :, :]
        nuclei = im[:, 1, :, :]

        widget = show_zyx_max_slice_interactive(
            [membrane, nuclei],
            colormap=['magma', 'viridis']
        )
        return widget,

    if __name__ == "__main__":
        app.run()
    """)

    app_path = tmp_path / "marimo_app.py"
    app_path.write_text(app_code)

    written = app_path.read_text()
    assert written == app_code
    compile(written, str(app_path), "exec")
    assert "colormap=" in written
    assert "colors=" not in written

    # The snippet must reference a function that exists and accepts colormap=
    import inspect
    import eigenp_utils.tnia_plotting_anywidgets as tpa
    assert "colormap" in inspect.signature(tpa.show_zyx_max_slice_interactive).parameters


def test_marimo_app_file_uses_existing_api():
    """tests/marimo_app.py only imports existing eigenp_utils names and passes keywords
    those functions accept (no deprecated ``colors=``)."""
    import ast
    import importlib
    import inspect
    from pathlib import Path

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

# =========================================
# Source: test_tnia_figsize_scale.py
# =========================================
def test_tnia_figsize_scale():
    """Test that tnia figsize scale works as expected."""
    import numpy as np
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

    im = np.random.rand(100, 100, 100)
    widget1 = show_zyx_max_slice_interactive(im, figsize_scale=1)
    widget2 = show_zyx_max_slice_interactive(im, figsize_scale=2)
    widget3 = show_zyx_max_slice_interactive(im, figsize_scale=10)
    assert np.allclose(np.array(widget2.figsize), 2 * np.array(widget1.figsize))
    assert np.allclose(np.array(widget3.figsize), 10 * np.array(widget1.figsize))

# =========================================
# Source: test_tnia_plotting_anywidgets_3d_logic.py
# =========================================

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")



def test_show_zyx_slice_returns_correct_slices():
    """Test that show zyx slice returns correct slices works as expected."""
    arr = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    # x, y and z all differ, so swapping any two indices changes a panel
    fig = show_zyx_slice(arr, x=0, y=2, z=3, use_plt=False)
    assert isinstance(fig, Figure)
    xy_expected = arr[3, :, :]
    xz_expected = arr[:, 2, :]
    zy_expected = np.flip(np.rot90(arr[:, :, 0], 1), 0)
    xy_img = fig.axes[0].images[0].get_array()
    zy_img = fig.axes[1].images[0].get_array()
    xz_img = fig.axes[2].images[0].get_array()
    assert np.array_equal(xy_img, xy_expected)
    assert np.array_equal(zy_img, zy_expected)
    assert np.array_equal(xz_img, xz_expected)


def test_show_zyx_max_slabs_projection():
    """Test that show zyx max slabs projection works as expected."""
    arr = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    fig = show_zyx_max_slabs(arr, x=[0, 1], y=[0, 2], z=[1, 4])
    xy_expected = np.max(arr[1:4, :, :], axis=0)
    xz_expected = np.max(arr[:, 0:2, :], axis=1)
    zy_expected = np.flip(np.rot90(np.max(arr[:, :, 0:1], axis=2), 1), 0)
    xy_img = fig.axes[0].images[0].get_array()
    zy_img = fig.axes[1].images[0].get_array()
    xz_img = fig.axes[2].images[0].get_array()
    assert np.array_equal(xy_img, xy_expected)
    assert np.array_equal(xz_img, xz_expected)
    assert np.array_equal(zy_img, zy_expected)
    plt.close(fig)


def test_deprecated_tnia_plotting_3d_warning():
    """Test that deprecated tnia plotting 3d warning works as expected."""
    import warnings
    import importlib

    # Remove from sys.modules to ensure re-evaluation
    import sys
    if "eigenp_utils.tnia_plotting_3d" in sys.modules:
        del sys.modules["eigenp_utils.tnia_plotting_3d"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import eigenp_utils.tnia_plotting_3d as tnia3d

        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message)

    # verify re-export works
    assert hasattr(tnia3d, "show_zyx")


def test_create_multichannel_rgb_basic():
    """Test that create multichannel rgb basic works as expected."""
    xy_list = [np.ones((2, 2)), np.zeros((2, 2))]
    xz_list = [np.zeros((2, 2)), np.ones((2, 2))]
    zy_list = [np.zeros((2, 2)), np.zeros((2, 2))]
    xy_rgb, xz_rgb, zy_rgb = create_multichannel_rgb(
        xy_list, xz_list, zy_list, colormap=["red", "green"]
    )
    red = np.asarray(to_rgb("red"))
    green = np.asarray(to_rgb("green"))
    expected_xy = np.broadcast_to(red, (2, 2, 3))
    expected_xz = np.broadcast_to(green, (2, 2, 3))
    expected_zy = np.zeros((2, 2, 3))
    assert np.allclose(xy_rgb, expected_xy)
    assert np.allclose(xz_rgb, expected_xz)
    assert np.allclose(zy_rgb, expected_zy)

def test_show_zyx_max_scatter_interactive_colormap():
    """Test that show zyx max scatter interactive colormap works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_scatter_interactive
    X = np.random.rand(10) * 10
    Y = np.random.rand(10) * 10
    Z = np.random.rand(10) * 10
    channels = np.random.rand(10)

    # Should not throw exception for invalid RGBA string, and _render should not throw NameError
    w1 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels, colormap='viridis', render='points')
    w1._render() # Trigger render directly

    w2 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels, colormap='viridis', render='density')
    w2._render() # Trigger render directly

    channels_multi = [np.random.rand(10), np.random.rand(10)]
    w3 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels_multi, colormap=['viridis', 'plasma'], render='points')
    w3._render() # Trigger render directly

    w4 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels_multi, colormap=['viridis', 'plasma'], render='density')
    w4._render() # Trigger render directly

    assert w1 is not None
    assert w2 is not None
    assert w3 is not None
    assert w4 is not None

# =========================================
# Source: test_show_zyx_scale_bar.py
# =========================================

def test_show_zyx_max_slabs_scale_bar():
    """Test that show zyx max slabs scale bar works as expected."""
    im = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)

    # Using tuple
    fig2 = show_zyx_max_slabs(im, pixel_sizes=(1.5, 0.5, 0.5))

    # Using dictionary
    fig3 = show_zyx_max_slabs(im, pixel_sizes={'Z': 1.5, 'Y': 0.5, 'X': 0.5})

    def get_text_from_fig(fig):
        axBar = fig.axes[-1]
        for t in axBar.texts:
            if "µm" in t.get_text() or "pixel_sizes" in t.get_text() or "sxy" in t.get_text():
                return t.get_text()
        return None

    # Assert
    assert "µm" in get_text_from_fig(fig2)
    assert "µm" in get_text_from_fig(fig3)

def test_interactive_factory_passes_pixel_sizes():
    """Test that interactive factory passes pixel sizes works as expected."""
    im = np.random.randint(0, 255, (10, 100, 100), dtype=np.uint8)

    # Should not throw errors and initialize perfectly
    widget = show_zyx_max_slice_interactive(im, pixel_sizes=(1.5, 0.5, 0.5))
    assert widget._pixel_sizes_given is True
    assert widget.sx == 0.5
    assert widget.sy == 0.5
    assert widget.sz == 1.5

if __name__ == '__main__':
    test_show_zyx_max_slabs_scale_bar()
    test_interactive_factory_passes_pixel_sizes()
    print("Tests passed")

# =========================================
# Source: test_tnia_plotting_anywidgets_spacing.py
# =========================================

def test_interactive_spacing_pixel_sizes_vs_sxy():
    """
    Test that when figsize is None, dynamically computed figsize parses
    pixel_sizes=(Z, Y, X) dicts correctly.
    """
    im = np.random.rand(10, 50, 50)

    # Using pixel_sizes dict
    w_dict = show_zyx_max_slice_interactive(im, pixel_sizes={'Z':2, 'Y':1, 'X':1})
    fig_dict = w_dict._render()
    size_dict = fig_dict.get_size_inches()

    # Using pixel_sizes tuple
    w_tuple = show_zyx_max_slice_interactive(im, pixel_sizes=(2, 1, 1))
    fig_tuple = w_tuple._render()
    size_tuple = fig_tuple.get_size_inches()

    np.testing.assert_allclose(size_dict, size_tuple)

    # Verify positions of XY and XZ axes match between dict and tuple pixel_sizes
    pos_xy_dict = fig_dict.axXY.get_position()
    pos_xy_tuple = fig_tuple.axXY.get_position()
    np.testing.assert_allclose(pos_xy_dict.bounds, pos_xy_tuple.bounds)

def test_xy_anisotropy():
    """
    Test that XY anisotropy is correctly respected in show_zyx spacing calculations.
    """
    im = np.random.rand(10, 50, 100) # Z, Y, X
    # Highly anisotropic XY pixels
    w = show_zyx_max_slice_interactive(im, pixel_sizes={'Z': 1.0, 'Y': 0.2, 'X': 1.0})
    fig = w._render()

    # X physical = 100 * 1 = 100. Z physical = 10 * 1 = 10.
    # Y physical = 50 * 0.2 = 10. Z physical = 10 * 1 = 10.
    # So axXY width should be 10x its height
    pos_xy = fig.axXY.get_position()
    figW, figH = fig.get_size_inches()
    w_in = pos_xy.width * figW
    h_in = pos_xy.height * figH
    np.testing.assert_allclose(w_in / h_in, 10.0, rtol=1e-2)

def test_annotation_coordinate_registration():
    # Synthetic 3D image volume: Z=16, Y=64, X=128
    Z, Y, X = 16, 64, 128
    synthetic_im = np.zeros((Z, Y, X), dtype=np.float32)

    # Anisotropic voxel dimensions: sz=2.0 um, sy=0.5 um, sx=0.5 um
    pixel_sizes = (2.0, 0.5, 0.5)

    widget = show_zyx_max_slice_interactive_point_annotator(
        synthetic_im,
        pixel_sizes=pixel_sizes,
        slabs_position=(8 * 2.0, 32 * 0.5, 64 * 0.5), # Physical center
        slabs_thickness=(2 * 2.0, 4 * 0.5, 4 * 0.5)
    )

    widget.annotation_mode = True
    widget.annotation_action = 'add'

    # Target voxel to annotate: z=8, y=20, x=45
    target_z, target_y, target_x = 8, 20, 45
    widget.z_s = target_z

    # Calculate figure using widget's inner method to get the figure with active transforms
    fig = widget._render()

    px_target = (target_x + 0.5) * pixel_sizes[2]
    py_target = (target_y + 0.5) * pixel_sizes[1]

    # Transform physical coordinates to JS coordinates
    axXY = fig.axXY
    target_disp = axXY.transData.transform((px_target, py_target))
    target_fig = fig.transFigure.inverted().transform(target_disp)

    frac_x = float(target_fig[0])
    frac_y = float(1.0 - target_fig[1])  # JS top-left origin

    import matplotlib.pyplot as plt
    plt.close(fig)

    # Force render so axis_bounds is populated
    widget._render_wrapper(None)

    # Simulate user click
    widget._handle_click({'new': {'plane': 'xy', 'x': frac_x, 'y': frac_y}})

    # Assert point registered correctly
    assert [target_z, target_y, target_x] in widget.points, \
        f"Expected {[target_z, target_y, target_x]} in registered points, got {widget.points}"

def test_annotation_with_labels_and_anisotropy():
    Z, Y, X = 16, 64, 128
    im = np.zeros((Z, Y, X), dtype=np.float32)
    pixel_sizes = {'X': 0.295, 'Y': 1.0, 'Z': 1.0}

    widget = show_zyx_max_slice_interactive_point_annotator(
        im,
        pixel_sizes=pixel_sizes,
        channel_labels=['GRAY'],
        slabs_position=(8, 32, 64)
    )

    widget.annotation_mode = True
    widget.annotation_action = 'add'

    # Target voxel [Z, Y, X]
    target_z, target_y, target_x = 8, 20, 45
    widget.z_s = target_z

    # Calculate figure using widget's inner method to get the figure with active transforms
    fig = widget._render()

    px_target = (target_x + 0.5) * pixel_sizes['X']
    py_target = (target_y + 0.5) * pixel_sizes['Y']

    # Transform physical coordinates to JS coordinates
    axXY = fig.axXY
    target_disp = axXY.transData.transform((px_target, py_target))
    target_fig = fig.transFigure.inverted().transform(target_disp)

    frac_x = float(target_fig[0])
    frac_y = float(1.0 - target_fig[1])  # JS top-left origin

    import matplotlib.pyplot as plt
    plt.close(fig)

    # Force render so axis_bounds is populated
    widget._render_wrapper(None)

    # Simulate click
    click_dict = {'plane': 'xy', 'x': frac_x, 'y': frac_y}
    widget._handle_click({'new': click_dict})

    assert [target_z, target_y, target_x] in widget.points, \
        f"Expected {[target_z, target_y, target_x]} in {widget.points}"

def test_annotation_deletion():
    synthetic_im = np.zeros((10, 32, 32), dtype=np.float32)
    widget = show_zyx_max_slice_interactive_point_annotator(synthetic_im)

    widget.add_point(5, 10, 15)
    assert [5, 10, 15] in widget.points

    widget.annotation_mode = True
    widget.annotation_action = 'delete'
    widget.z_s = 5

    info = widget.axis_bounds['xy']
    b_x0, b_y0, b_w, b_h = info['bbox']
    xlim, ylim = info['xlim'], info['ylim']

    u = (15.5 - xlim[0]) / (xlim[1] - xlim[0])
    v = (10.5 - ylim[0]) / (ylim[1] - ylim[0])

    frac_x = b_x0 + u * b_w
    mpl_y_frac = b_y0 + v * b_h
    frac_y = 1.0 - mpl_y_frac

    widget._handle_click({'new': {'plane': 'xy', 'x': frac_x, 'y': frac_y}})
    assert [5, 10, 15] not in widget.points


def test_ground_truth_annotation_all_planes():
    Z, Y, X = 16, 64, 128
    im = np.zeros((Z, Y, X), dtype=np.float32)
    pixel_sizes = {'X': 0.295, 'Y': 1.0, 'Z': 1.0}

    widget = show_zyx_max_slice_interactive_point_annotator(
        im,
        pixel_sizes=pixel_sizes,
        channel_labels=['GRAY'],
        slabs_position=(8, 32, 64)
    )
    widget.annotation_mode = True
    widget.annotation_action = 'add'

    # Target voxels for XY, ZY, and XZ views
    targets = [
        ('xy', 8, 20, 45),
        ('zy', 12, 40, 64),
        ('xz', 5, 32, 100)
    ]

    for plane, target_z, target_y, target_x in targets:
        widget.z_s = target_z
        widget.y_s = target_y
        widget.x_s = target_x

        fig = widget._render()
        try:
            ax = getattr(fig, f"ax{plane.upper()}")
            if plane == 'xy':
                phys_a, phys_b = (target_x + 0.5) * widget.sx, (target_y + 0.5) * widget.sy
            elif plane == 'zy':
                phys_a, phys_b = (target_z + 0.5) * widget.sz, (target_y + 0.5) * widget.sy
            elif plane == 'xz':
                phys_a, phys_b = (target_x + 0.5) * widget.sx, (target_z + 0.5) * widget.sz

            display_pixel = ax.transData.transform((phys_a, phys_b))
            fig_norm = fig.transFigure.inverted().transform(display_pixel)
            click_x, click_y = float(fig_norm[0]), float(1.0 - fig_norm[1])
            
            # Execute the property assignment and validation before closing the figure
            widget.click_coords = {'plane': plane, 'x': click_x, 'y': click_y}
            assert [target_z, target_y, target_x] in widget.points, \
                f"Plane {plane} click at ({click_x:.3f}, {click_y:.3f}) mapped incorrectly."
        finally:
            plt.close(fig)


def test_hover_sync_all_planes():
    """
    Test that hover coordinates correctly sync back to the widget's physical state.
    We first call _render() explicitly to acquire the Matplotlib Figure and calculate
    accurate physical-to-display coordinate mappings for our simulated hover. We then
    close the figure, call _render_wrapper to populate internal state (like axis_bounds)
    as it would in production, and finally dispatch the hover coordinates.
    """
    Z, Y, X = 16, 64, 128
    im = np.zeros((Z, Y, X), dtype=np.float32)
    pixel_sizes = {'X': 0.295, 'Y': 1.0, 'Z': 1.0}

    widget = show_zyx_max_slice_interactive(
        im,
        pixel_sizes=pixel_sizes,
        sync_on_hover=True
    )

    target_z, target_y, target_x = 10, 25, 80

    fig = widget._render()
    try:
        ax = fig.axXY
        phys_x = (target_x + 0.5) * widget.sx
        phys_y = (target_y + 0.5) * widget.sy
        display_pixel = ax.transData.transform((phys_x, phys_y))
        fig_norm = fig.transFigure.inverted().transform(display_pixel)
        hover_x, hover_y = float(fig_norm[0]), float(1.0 - fig_norm[1])
    finally:
        plt.close(fig)

    widget._render_wrapper(None) # Forces bounds calculation as it would in reality

    # Execute assignment and assert prior to destruction
    widget.hover_coords = {'plane': 'xy', 'x': hover_x, 'y': hover_y}
    assert widget.x_s == target_x and widget.y_s == target_y, \
        f"Hover sync failed on XY plane. Got x={widget.x_s}, y={widget.y_s}"


def test_axis_bounds_alignment():
    Z, Y, X = 16, 64, 128
    im = np.zeros((Z, Y, X), dtype=np.float32)
    pixel_sizes = {'X': 0.295, 'Y': 1.0, 'Z': 1.0}

    widget = show_zyx_max_slice_interactive(im, pixel_sizes=pixel_sizes)
    fig = widget._render()
    try:
        ax_xy = fig.axXY
        cell_bbox = ax_xy.get_position()
        info = widget.axis_bounds['xy']

        # Verify that the image extent exactly fills the subplot cell bounding box without extra margins
        np.testing.assert_allclose(info['x0'], cell_bbox.x0, atol=1e-2)
        np.testing.assert_allclose(info['y0'], cell_bbox.y0, atol=1e-2)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("figsize", [(8, 8), (12, 6), (6, 12)])
@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("gap_in", [1.0 / 16.0, 1.0 / 8.0])
def test_fixed_physical_axes_gap(figsize, with_labels, gap_in):
    """
    Verify that the physical distance between adjacent axes is strictly equal to
    gap_in (in inches) regardless of figure size, aspect ratio, or channel_labels toggle.
    """
    Z, Y, X = 20, 40, 120
    im = np.zeros((Z, Y, X), dtype=np.float32)
    labels = ["Ch0", "Ch1"] if with_labels else None

    fig = show_zyx(
        xy=im[10, :, :], xz=im[:, 20, :], zy=np.flip(np.rot90(im[:, :, 60], 1), 0),
        pixel_sizes={'X': 0.5, 'Y': 0.5, 'Z': 1.5},
        figsize=figsize,
        channel_labels=labels,
        gap_in=gap_in
    )

    figW, figH = figsize
    pos_xy = fig.axXY.get_position()
    pos_zy = fig.axZY.get_position()
    pos_xz = fig.axXZ.get_position()

    # Horizontal gap between XY and ZY
    gap_h_in = (pos_zy.x0 - pos_xy.x1) * figW
    np.testing.assert_allclose(gap_h_in, gap_in, atol=1e-5)

    # Vertical gap between XZ and XY
    gap_v_in = (pos_xy.y0 - pos_xz.y1) * figH
    np.testing.assert_allclose(gap_v_in, gap_in, atol=1e-5)

    if with_labels and fig.axLabels is not None:
        pos_labels = fig.axLabels.get_position()
        # Vertical gap between XY and Labels
        gap_labels_in = (pos_labels.y0 - pos_xy.y1) * figH
        np.testing.assert_allclose(gap_labels_in, gap_in, atol=1e-5)

    plt.close(fig)


@pytest.mark.parametrize("render_mode", ["points", "density"])
def test_scatter_widget_physical_axes_gap(render_mode):
    """
    Verify that TNIAScatterWidget renders axes with exact 1/16th inch physical gap.
    """
    X = np.random.rand(50) * 10
    Y = np.random.rand(50) * 20
    Z = np.random.rand(50) * 5

    w = show_zyx_max_scatter_interactive((Z, Y, X), figsize=(10, 8), render=render_mode)
    fig = w._render()

    figW, figH = (10, 8)
    pos_xy = fig.axXY.get_position()
    pos_zy = fig.axZY.get_position()
    pos_xz = fig.axXZ.get_position()

    gap_h_in = (pos_zy.x0 - pos_xy.x1) * figW
    gap_v_in = (pos_xy.y0 - pos_xz.y1) * figH

    np.testing.assert_allclose(gap_h_in, 1.0 / 16.0, atol=1e-5)
    np.testing.assert_allclose(gap_v_in, 1.0 / 16.0, atol=1e-5)

    plt.close(fig)



def test_scatter_widget_axis_bounds_population():
    """
    Ensure scatter widget properly assigns axes to the figure, executes canvas.draw(),
    and computes the physical bounding boxes required by the JavaScript frontend.
    """
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_scatter_interactive
    
    X = np.random.rand(10) * 10
    Y = np.random.rand(10) * 10
    Z = np.random.rand(10) * 10

    # Test points rendering mode
    w_points = show_zyx_max_scatter_interactive((Z, Y, X), render='points')
    w_points._render_wrapper(None)  # Force full render pipeline
    
    assert 'xy' in w_points.axis_bounds, "Scatter (points) failed to populate 'xy' axis bounds."
    assert 'zy' in w_points.axis_bounds, "Scatter (points) failed to populate 'zy' axis bounds."
    assert 'xz' in w_points.axis_bounds, "Scatter (points) failed to populate 'xz' axis bounds."

    # Validate that canvas.draw() occurred by checking for non-zero, correctly oriented coordinates
    for plane in ['xy', 'zy', 'xz']:
        bbox = w_points.axis_bounds[plane]
        assert bbox['x1'] > bbox['x0'], f"Invalid x-coordinates for {plane} plane bounds."
        # Note: y1_js > y0_js because JS origin is top-left, while mpl origin is bottom-left
        assert bbox['y1_js'] > bbox['y0_js'], f"Invalid JS y-coordinates for {plane} plane bounds."
        assert len(bbox['bbox']) == 4, "JS bounding box array is improperly sized."

    # Test density rendering mode
    w_density = show_zyx_max_scatter_interactive((Z, Y, X), render='density')
    w_density._render_wrapper(None)
    assert 'xy' in w_density.axis_bounds, "Scatter (density) failed to populate axis bounds."

def test_slice_widget_axis_bounds_structure():
    """
    Verify the dictionary structure of axis_bounds exactly matches the expectations 
    of the JS Proxy traversal fix.
    """
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive
    im = np.zeros((10, 20, 30))
    w = show_zyx_max_slice_interactive(im)
    w._render_wrapper(None)
    
    bounds = w.axis_bounds
    
    for plane in ['xy', 'zy', 'xz']:
        plane_data = bounds.get(plane)
        assert plane_data is not None
        
        # The JS explicitly looks for x0, x1, y0_js, y1_js OR a 'bbox' array
        expected_keys = {'x0', 'x1', 'y0', 'y1', 'y0_js', 'y1_js', 'bbox', 'xlim', 'ylim'}
        assert expected_keys.issubset(plane_data.keys()), f"Missing required keys in {plane} bounds payload."


def test_hover_sync_with_rotate_view():
    """
    Verify that hover sync ('C' key) correctly unrotates cursor coordinates and sets
    the slice sliders (x_s, y_s, z_s) to the exact target voxel when rotate_view is active.
    """
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, _unrotate_2d, _get_rotated_line

    Z, Y, X = 20, 60, 80
    im = np.zeros((Z, Y, X), dtype=np.float32)
    pixel_sizes = {'X': 1.0, 'Y': 1.0, 'Z': 1.0}

    target_z, target_y, target_x = 10, 25, 35
    rotate_view = (30.0, 15.0, 45.0)  # (rot_z, rot_y, rot_x)

    widget = show_zyx_max_slice_interactive(
        im,
        pixel_sizes=pixel_sizes,
        sync_on_hover=True,
        rotate_view=rotate_view
    )

    # Force initial render to generate axis_bounds
    fig = widget._render()
    try:
        # Determine physical location of target in unrotated space
        orig_phys_x = (target_x + 0.5) * widget.sx
        orig_phys_y = (target_y + 0.5) * widget.sy

        # Rotate target point forward to display physical space
        xs, ys = _get_rotated_line(orig_phys_x, orig_phys_y, orig_phys_x, orig_phys_y, rotate_view[0], X, Y, widget.sx, widget.sy)
        rot_phys_x, rot_phys_y = xs[0], ys[0]

        # Convert rotated physical position to figure coordinates
        ax = fig.axXY
        display_pixel = ax.transData.transform((rot_phys_x, rot_phys_y))
        fig_norm = fig.transFigure.inverted().transform(display_pixel)
        hover_x, hover_y = float(fig_norm[0]), float(1.0 - fig_norm[1])
    finally:
        plt.close(fig)

    widget.hover_coords = {'plane': 'xy', 'x': hover_x, 'y': hover_y}

    assert widget.x_s == target_x and widget.y_s == target_y, \
        f"Hover sync with rotation failed on XY plane. Expected ({target_x}, {target_y}), got ({widget.x_s}, {widget.y_s})"


def test_annotator_click_with_rotate_view():
    """
    Verify that point annotation clicks on a rotated view accurately unrotate
    coordinates and record the exact target voxel in widget.points.
    """
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive_point_annotator, _get_rotated_line

    Z, Y, X = 20, 60, 80
    im = np.zeros((Z, Y, X), dtype=np.float32)
    rotate_view = 30.0

    widget = show_zyx_max_slice_interactive_point_annotator(
        im,
        rotate_view=rotate_view
    )
    widget.annotation_mode = True
    widget.annotation_action = 'add'

    target_z, target_y, target_x = 10, 20, 30
    widget.z_s = target_z

    fig = widget._render()
    try:
        orig_phys_x = (target_x + 0.5) * widget.sx
        orig_phys_y = (target_y + 0.5) * widget.sy

        xs, ys = _get_rotated_line(orig_phys_x, orig_phys_y, orig_phys_x, orig_phys_y, rotate_view, X, Y, widget.sx, widget.sy)
        rot_phys_x, rot_phys_y = xs[0], ys[0]

        ax = fig.axXY
        display_pixel = ax.transData.transform((rot_phys_x, rot_phys_y))
        fig_norm = fig.transFigure.inverted().transform(display_pixel)
        click_x, click_y = float(fig_norm[0]), float(1.0 - fig_norm[1])
    finally:
        plt.close(fig)

    widget.click_coords = {'plane': 'xy', 'x': click_x, 'y': click_y}

    assert [target_z, target_y, target_x] in widget.points, \
        f"Annotator click with rotation failed. Expected {[target_z, target_y, target_x]} in {widget.points}"


def test_channel_label_height_scaling_with_fontsize():
    """
    Test that the channel label axis height (axLabels) scales with fontsize_pt
    (figure height) and provides sufficient fake spacing so labels are not cropped.
    """
    im = np.zeros((10, 20, 20), dtype=np.float32)
    labels = ["Channel 0", "Channel 1"]

    # Small figure height -> smaller fontsize
    fig_small = show_zyx(
        xy=im[5, :, :], xz=im[:, 10, :], zy=im[:, :, 10],
        figsize=(8, 4),
        channel_labels=labels
    )
    h_small = fig_small.axLabels.get_position().height

    # Large figure height -> larger fontsize
    fig_large = show_zyx(
        xy=im[5, :, :], xz=im[:, 10, :], zy=im[:, :, 10],
        figsize=(8, 12),
        channel_labels=labels
    )
    h_large = fig_large.axLabels.get_position().height

    try:
        assert fig_small.axLabels is not None
        assert fig_large.axLabels is not None
        # Position height fraction should account for font size scaling
        pos_small = fig_small.axLabels.get_position()
        pos_large = fig_large.axLabels.get_position()
        assert pos_small.height > 0
        assert pos_large.height > 0
        # Label height in inches is 0.10 + fontsize/72, with fontsize growing with figH
        h_small_in, h_large_in = h_small * 4, h_large * 12
        for h_in, figH in ((h_small_in, 4), (h_large_in, 12)):
            fontsize = max(10, min(24, figH * 72 * 0.035))
            assert h_in == pytest.approx(0.10 + fontsize / 72.0, rel=1e-6)
        assert h_large_in > h_small_in
    finally:
        plt.close(fig_small)
        plt.close(fig_large)


def test_single_slider_start_end_sync_and_translate():
    """
    Test that start/end traits sync bidirectionally with center position (x_s, y_s, z_s)
    and thickness (x_t, y_t, z_t), and that range translation works as expected.
    """
    im = np.zeros((20, 40, 60))  # Z, Y, X
    w = show_zyx_max_slice_interactive(im, slabs_position=(10, 20, 30), slabs_thickness=(2, 4, 6))

    # Initial start / end expected values:
    # Z: z_s=10, z_t=2 => z_start=8, z_end=12
    # Y: y_s=20, y_t=4 => y_start=16, y_end=24
    # X: x_s=30, x_t=6 => x_start=24, x_end=36
    assert w.z_start == 8 and w.z_end == 12
    assert w.y_start == 16 and w.y_end == 24
    assert w.x_start == 24 and w.x_end == 36

    # Test updating x_start/x_end (e.g. user drags start and end thumbs in JS)
    w.x_start = 10
    w.x_end = 20
    # Center x_s should be (10+20)//2 = 15, x_t should be (20-10)//2 = 5
    assert w.x_s == 15
    assert w.x_t == 5

    # Test translating the range (dragging middle thumb in JS)
    # Move range by +10: start=20, end=30
    w.x_start = 20
    w.x_end = 30
    assert w.x_s == 25
    assert w.x_t == 5

    # Test programmatically updating x_s / x_t updates x_start / x_end
    w.x_s = 40
    w.x_t = 10
    assert w.x_start == 30
    assert w.x_end == 50


def test_channel_label_fontsize_pt_kwarg_and_anisotropic_height():
    """
    Verify that channel_label_fontsize_pt controls label font size and physical height of axLabels,
    and that axLabels physical height is independent of image anisotropy and aspect ratios.
    """
    # 1. Test custom channel_label_fontsize_pt in show_zyx
    im_aniso = np.zeros((5, 10, 500), dtype=np.float32) # highly anisotropic image
    figsize = (10, 8)
    labels = ["Ch0", "Ch1"]
    fontsize_pt = 18.0

    fig = show_zyx(
        xy=im_aniso[2, :, :], xz=im_aniso[:, 5, :], zy=np.flip(np.rot90(im_aniso[:, :, 250], 1), 0),
        pixel_sizes={'X': 1.0, 'Y': 1.0, 'Z': 1.0},
        figsize=figsize,
        channel_labels=labels,
        channel_label_fontsize_pt=fontsize_pt
    )

    try:
        assert fig.axLabels is not None
        pos_labels = fig.axLabels.get_position()
        labels_h_in = pos_labels.height * figsize[1]
        expected_hl_in = 0.10 + (fontsize_pt / 72.0) # 0.10 + 0.25 = 0.35 in
        np.testing.assert_allclose(labels_h_in, expected_hl_in, atol=1e-5)
    finally:
        plt.close(fig)

    # 2. Test propagation through interactive widgets and show_zyx_max_slice_interactive
    w = show_zyx_max_slice_interactive(
        im_aniso, channel_labels=labels, channel_label_fontsize_pt=20.0, figsize=(10, 8)
    )
    assert w.channel_label_fontsize_pt == 20.0
    fig_w = w._render()
    try:
        assert fig_w.axLabels is not None
        pos_labels_w = fig_w.axLabels.get_position()
        labels_h_in_w = pos_labels_w.height * 8.0
        expected_hl_in_w = 0.10 + (20.0 / 72.0)
        np.testing.assert_allclose(labels_h_in_w, expected_hl_in_w, atol=1e-5)
    finally:
        plt.close(fig_w)

    # 3. Test propagation through show_zyx_max_scatter_interactive
    pts_X = np.random.rand(50) * 100
    pts_Y = np.random.rand(50) * 10
    pts_Z = np.random.rand(50) * 2
    w_sc = show_zyx_max_scatter_interactive(
        (pts_X, pts_Y, pts_Z), channel_labels=labels, channel_label_fontsize_pt=14.0, figsize=(10, 8)
    )
    assert w_sc.channel_label_fontsize_pt == 14.0
    fig_sc = w_sc._render()
    try:
        assert fig_sc.axLabels is not None
        pos_labels_sc = fig_sc.axLabels.get_position()
        labels_h_in_sc = pos_labels_sc.height * 8.0
        expected_hl_in_sc = 0.10 + (14.0 / 72.0)
        np.testing.assert_allclose(labels_h_in_sc, expected_hl_in_sc, atol=1e-5)
    finally:
        plt.close(fig_sc)


# =========================================
# Render coalescing
# =========================================
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


def test_slab_traits_stay_paired_and_in_bounds():
    """Out-of-range slab writes are clamped, and both representations stay consistent.

    The two slab representations are kept in step by observers that clamp to the
    volume shape, so an interactive client can push any value without the
    renderer ever seeing coordinates outside the array. Strategy: write a wildly
    out-of-range end position and confirm it lands on the last valid index, that
    centre and half-thickness still describe the same interval, and that no
    clipping warning is surfaced -- the clamp is supposed to make the warning
    path unnecessary rather than merely report after the fact.
    """
    vol = _structured_volume()
    Z, Y, X = vol.shape
    w = show_zyx_max_slice_interactive(vol, figsize=(4, 4))

    w.x_start = 0
    w.x_end = 10 ** 6

    assert w.x_start == 0
    assert w.x_end == X - 1
    assert w.x_s == (w.x_start + w.x_end) // 2
    assert w.x_t == (w.x_end - w.x_start) // 2
    assert w.warning_msg == ""


# =========================================
# compute_histogram
# =========================================
def test_compute_histogram_empty_and_all_nan_inputs():
    """Degenerate inputs yield empty payloads rather than raising.

    The histogram is computed once per channel at construction and shipped to
    the frontend, which skips channels whose counts are empty. Strategy: feed an
    empty array and an all-NaN float array -- the two ways a channel can carry
    no usable values -- and require the documented empty payload from both,
    since a raise here would break widget construction entirely.
    """
    assert compute_histogram(np.array([], dtype=np.float32)) == {'counts': [], 'bin_edges': []}
    all_nan = np.full(10, np.nan, dtype=np.float32)
    assert compute_histogram(all_nan) == {'counts': [], 'bin_edges': []}


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

    assert len(result['counts']) == 16
    assert (result['bin_edges'][0], result['bin_edges'][-1]) == expected_range


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
    counts = np.array(result['counts'])

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

    np.testing.assert_allclose(np.expm1(result['counts']).sum(), arr.size, rtol=1e-6)


def test_compute_histogram_excludes_nans_from_float_channels():
    """NaNs are dropped from float channels instead of poisoning the bin edges.

    np.histogram propagates NaN into the automatic range, which would collapse
    every bin edge to NaN and blank the frontend canvas. Strategy: mix NaNs into
    a finite float channel and assert both the edges stay finite and the
    recovered counts total only the finite values.
    """
    arr = np.array([0.0, 1.0, 2.0, np.nan, np.nan], dtype=np.float32)
    result = compute_histogram(arr, bins=4)

    assert np.all(np.isfinite(result['bin_edges']))
    np.testing.assert_allclose(np.expm1(result['counts']).sum(), 3, rtol=1e-6)


# =========================================
# Copy-parameters round trip
# =========================================
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
    pixel_sizes = {'Z': 2.0, 'Y': 0.5, 'X': 0.5}
    w = show_zyx_max_slice_interactive(vol, pixel_sizes=pixel_sizes, figsize=(4, 4))

    w.x_s, w.y_s, w.z_s = 20, 16, 6
    w.x_t, w.y_t, w.z_t = 4, 3, 2

    w.copy_params_trigger += 1
    params = eval(f"dict({w.copy_params_string})", {"__builtins__": {"dict": dict}}, {})

    # Positions and thicknesses are reported in physical units, (Z, Y, X) order.
    assert params['slabs_position'] == (6 * 2.0, 16 * 0.5, 20 * 0.5)
    assert params['slabs_thickness'] == (2 * 2.0, 3 * 0.5, 4 * 0.5)
    assert len(params['vmin']) == len(params['vmax']) == 1
    assert len(params['gamma']) == len(params['opacity']) == 1

    rebuilt = show_zyx_max_slice_interactive(vol, pixel_sizes=pixel_sizes, figsize=(4, 4), **params)

    assert (rebuilt.z_s, rebuilt.y_s, rebuilt.x_s) == (w.z_s, w.y_s, w.x_s)
    assert (rebuilt.z_t, rebuilt.y_t, rebuilt.x_t) == (w.z_t, w.y_t, w.x_t)


# =========================================
# IsoScatterWidget
# =========================================
def _iso_points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(n) * 50, rng.random(n) * 40, rng.random(n) * 10


def test_show_iso_scatter_renders_and_responds_to_camera():
    """The isometric viewer renders on construction and re-renders when rotated.

    `elev` and `azim` are the widget's only interactive controls, and the whole
    projection is recomputed in Python, so a missed observer would leave the
    camera sliders visibly inert. Strategy: check an image exists after
    construction, then rotate and require the image to actually change.
    """
    X, Y, Z = _iso_points()
    # Default-sized figure: the 3-row layout plus a suptitle does not fit a
    # small one, and Matplotlib warns that tight_layout gave up.
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


# =========================================
# _rotate_expand (scipy replacement for skimage.transform.rotate(resize=True))
# =========================================
from eigenp_utils.tnia_plotting_anywidgets import _rotate_expand


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
    assert out.shape == (int(np.around(abs((w - 1) * s) + abs((h - 1) * c) + 1)),
                         int(np.around(abs((w - 1) * c) + abs((h - 1) * s) + 1)))
    assert out.min() >= 0.0 and out.max() <= 2.0
    cy, cx = (out.shape[0] - 1) // 2, (out.shape[1] - 1) // 2
    # The B-spline prefilter is global: edge ringing decays ~0.27**d, leaving
    # ~1e-5 residue at the centre of a 20 px image, hence the 1e-3 tolerance.
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


@pytest.mark.skipif(sys.platform == "emscripten", reason="no subprocesses in Pyodide; covered by tests/wasm")
def test_top_level_import_does_not_require_skimage():
    """``import eigenp_utils`` must work without scikit-image installed.

    scikit-image is an optional extra (``image-analysis``); the core plotting
    API, and in particular browser/WASM environments installing only core
    dependencies, must not need it. Strategy: in a fresh interpreter, make any
    ``skimage`` import raise, then import the package and its ortho-view API.

    Expected outcome: the subprocess exits cleanly and the core names exist.
    """
    import subprocess
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


# =========================================
# Bug-scan batch 4: slab indexing, density mode, one rotation convention
# =========================================

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
    H, W = arr.shape
    xs = x0 + (np.arange(W) + 0.5) * (x1 - x0) / W
    ys = y1 + (np.arange(H) + 0.5) * (y0 - y1) / H
    tot = arr.sum()
    return float((arr.sum(axis=0) * xs).sum() / tot), float((arr.sum(axis=1) * ys).sum() / tot)


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
    w = show_zyx_max_scatter_interactive((Z, Y, X), channels=ids, colormap='viridis', render='density',
                                         slabs_thickness=(1000, 1000, 1000))
    fig = w._render()
    try:
        im = fig.axXY.images[0]
        rgb = np.asarray(im.get_array())[..., :3]
        W = rgb.shape[1]
        left, right = im.get_extent()[:2]
        # cluster c occupies data x in [25c, 25c + 10]; the panel starts at xmin - 0.5
        col_of = lambda x: int((x - (w.xmin - 0.5) - left) / (right - left) * W)
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
    X = rng.uniform(0, 1000, n); Y = rng.uniform(0, 1000, n); Z = rng.uniform(0, 20, n)
    X[0], Y[0], Z[0] = 0, 0, 0
    w = show_zyx_max_scatter_interactive((Z, Y, X), render='density', pixel_sizes=(1, 1, 1))
    fig = w._render()
    try:
        def span(ax):
            return abs(np.diff(ax.get_xlim())[0]), abs(np.diff(ax.get_ylim())[0])
        np.testing.assert_allclose(span(fig.axXY), (1001, 1001), rtol=1e-6)
        np.testing.assert_allclose(span(fig.axXZ), (1001, 21), rtol=1e-6)
        np.testing.assert_allclose(span(fig.axZY), (21, 1001), rtol=1e-6)

        label = [t.get_text() for t in fig.axBar.texts][0]
        bar = fig.axXY.collections[-1].get_segments()[0]
        bar_units = (bar[1][0] - bar[0][0]) * span(fig.axXY)[0]
        assert bar_units == pytest.approx(float(label.split()[0]), rel=1e-6)

        # density panels start at the data box edge, min - 0.5
        hx, hy = _fig_frac(fig, fig.axXY, 750 - w.xmin + 0.5, 250 - w.ymin + 0.5)
    finally:
        plt.close(fig)
    w.hover_coords = {'plane': 'xy', 'x': hx, 'y': hy}
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
    w = show_zyx_max_scatter_interactive((Z, Y, X), channels=vals, render='density', bins=100,
                                         vmin=0.0, vmax=1.0)
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
    w = show_zyx_max_scatter_interactive((Z, Y, X), channels=[a, b], render='density',
                                         opacity=[0.0, 1.0], slabs_thickness=(1000, 1000, 1000))
    fig = w._render()
    try:
        im = fig.axXY.images[0]
        assert im.get_alpha() in (None, 1.0)
        assert np.asarray(im.get_array())[..., 1].max() > 0.5  # lime channel present
    finally:
        plt.close(fig)


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
        cx_img, cy_img = _rendered_centroid(fig.axXY)
        centre = np.mean(fig.axXY.get_ylim())
        assert cy_img < centre - 30
    finally:
        plt.close(fig)

    X = np.array([0.0, 100.0, 85.0]); Y = np.array([0.0, 100.0, 50.0]); Z = np.zeros(3)
    for render in ('points', 'density'):
        w = show_zyx_max_scatter_interactive((Z, Y, X), render=render, rotate_view=(90, 0, 0), bins=101,
                                             slabs_thickness=(1000, 1000, 1000))
        fig = w._render()
        try:
            centre_y = np.mean(fig.axXY.get_ylim())
            if render == 'points':
                pts = np.vstack([c.get_offsets() for c in fig.axXY.collections if len(c.get_offsets())])
                # The two box corners end up left/right of centre; the marker is
                # the point closest to the centre column.
                marker = pts[np.argmin(np.abs(pts[:, 0] - np.mean(fig.axXY.get_xlim())))]
                assert marker[1] < centre_y - 30, (marker, centre_y)
                hx, hy = _fig_frac(fig, fig.axXY, marker[0], marker[1])
            else:
                # Mask out the two corner points; the marker is the bright bin
                # nearest the centre column.
                rgb = np.asarray(fig.axXY.images[0].get_array())[..., :3].sum(axis=-1)
                x0e, x1e, y0e, y1e = fig.axXY.images[0].get_extent()
                H, W = rgb.shape
                col = rgb[:, W // 2 - 3:W // 2 + 4].max(axis=1)
                row = int(np.argmax(col))
                y_marker = y1e + (row + 0.5) * (y0e - y1e) / H
                assert y_marker < centre_y - 30, (y_marker, centre_y)
        finally:
            plt.close(fig)
        if render == 'points':
            w.hover_coords = {'plane': 'xy', 'x': hx, 'y': hy}
            assert (w.x_s, w.y_s) == (85, 50)


@pytest.mark.parametrize("plane, angle, pixel_sizes", [
    ('xy', 45.0, (1, 1, 1)),
    ('xy', 180.0, (1, 1, 1)),
    ('xy', 30.0, (1, 1, 1)),
    ('xz', 30.0, (4, 1, 1)),
    ('zy', 60.0, (4, 1, 1)),
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
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    r2 = ((zz - tz) * pz) ** 2 + ((yy - ty) * py) ** 2 + ((xx - tx) * px) ** 2
    vol = np.exp(-r2 / (2 * 3.0 ** 2)).astype(np.float32)

    rot = {'xy': (angle, 0, 0), 'xz': (0, angle, 0), 'zy': (0, 0, angle)}[plane]
    w = show_zyx_max_slice_interactive_point_annotator(vol, pixel_sizes=pixel_sizes, rotate_view=rot,
                                                       slabs_position=(tz * pz, ty * py, tx * px),
                                                       slabs_thickness=(pz, py, px))
    w.annotation_mode = True
    fig = w._render()
    try:
        ax = {'xy': fig.axXY, 'xz': fig.axXZ, 'zy': fig.axZY}[plane]
        cx, cy = _rendered_centroid(ax)
        click_x, click_y = _fig_frac(fig, ax, cx, cy)
    finally:
        plt.close(fig)
    w.click_coords = {'plane': plane, 'x': click_x, 'y': click_y}
    assert w.points == [[tz, ty, tx]], w.points


def test_rotated_anisotropic_panel_keeps_physical_aspect():
    """Rotation happens in physical space, so anisotropic panels are not distorted.

    Strategy: a 40 x 40 um XZ plane (Z=10 at 4 um, X=40 at 1 um), rotated 90
    degrees, plus the matching crosshair transform.

    Expected outcome: the panel is still 40 x 40 um (it was drawn 10 x 160 um),
    and the rotated-line helper maps the panel corners onto the panel's edges.
    """
    from eigenp_utils.tnia_plotting_anywidgets import _get_rotated_line

    vol = np.random.default_rng(4).uniform(0, 1, (10, 8, 40)).astype(np.float32)
    fig = show_zyx_max_slabs(vol, [0, 40], [0, 8], [0, 10], pixel_sizes=(4, 1, 1), rotate_view=(0, 90, 0))
    try:
        assert abs(np.diff(fig.axXZ.get_xlim())[0]) == pytest.approx(40, abs=1)
        assert abs(np.diff(fig.axXZ.get_ylim())[0]) == pytest.approx(40, abs=1)
        xs, ys = _get_rotated_line(0, 0, 40, 40, 90, 40, 10, 1, 4)
        # (0, 0) -> bottom-left corner and (40, 40) -> top-right corner after a CCW quarter turn
        np.testing.assert_allclose([xs[0], ys[0]], [0, 40], atol=1)
        np.testing.assert_allclose([xs[1], ys[1]], [40, 0], atol=1)
    finally:
        plt.close(fig)


def test_rotate_expand_keeps_nan_local():
    """One NaN pixel no longer blanks the whole rotated panel.

    Expected outcome: only a few pixels around the NaN are NaN; everything
    else is finite.
    """
    from eigenp_utils.tnia_plotting_anywidgets import _rotate_expand

    img = np.random.default_rng(5).uniform(0, 1, (64, 64))
    img[30, 30] = np.nan
    out = _rotate_expand(img, 10)
    assert 0 < np.isnan(out).sum() <= 4
    img[30, 30] = 0.5
    np.testing.assert_allclose(np.isfinite(_rotate_expand(img, 10)).all(), True)


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
        (lo, hi), = lims
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
    xy, xz, zy = create_multichannel_rgb([a, b], [a, b], [a, b], colormap=['red', 'lime'])
    assert np.isfinite(xy).all()
    # red is channel a only; with vmax from finite values most pixels are not saturated
    assert (xy[..., 0] > 0.99).mean() < 0.1


def test_blend_colors_scalar_gamma():
    from eigenp_utils.tnia_plotting_anywidgets import blend_colors

    I = np.linspace(0, 1, 5)[:, None]
    out = blend_colors(I, ['white'], vmin=0, vmax=1, gamma=0.5)
    np.testing.assert_allclose(out[:, 0], np.sqrt(I[:, 0]))


def test_histogram_and_widget_accept_inf_values():
    vol = np.random.default_rng(8).uniform(0, 1, (4, 16, 16)).astype(np.float32)
    vol[0, 0, 0] = np.inf
    h = compute_histogram(vol)
    assert np.isfinite(h['bin_edges']).all() and len(h['counts']) == 128
    show_zyx_max_slice_interactive(vol, pixel_sizes=(1, 1, 1))


@pytest.mark.parametrize("colormap", ['magma', ['red', 'lime']])
def test_slice_widget_pads_short_colormap_for_multichannel(colormap):
    """A single colormap string is not split into characters, and a short list is padded."""
    rng = np.random.default_rng(9)
    chans = [rng.uniform(0, 1, (4, 16, 16)).astype(np.float32) for _ in range(3)]
    w = show_zyx_max_slice_interactive(chans, colormap=colormap)
    assert len(w.channel_colors) == 3
    given = [colormap] if isinstance(colormap, str) else colormap
    assert w.colors_resolved[:len(given)] == given


def test_points_mode_scales_y_by_sy():
    """Points mode uses the Y pixel size for Y (it used the X one).

    Strategy: pixel_sizes=(1, 2, 1); hover a marker at y=60 on the XY panel.

    Expected outcome: y_s == 60, and the XY panel spans twice as far in Y as
    in X for a square point cloud.
    """
    X = np.array([0.0, 99.0, 30.0]); Y = np.array([0.0, 99.0, 60.0]); Z = np.array([0.0, 9.0, 5.0])
    w = show_zyx_max_scatter_interactive((Z, Y, X), pixel_sizes=(1, 2, 1), slabs_thickness=(1000, 1000, 1000))
    fig = w._render()
    try:
        assert abs(np.diff(fig.axXY.get_ylim())[0]) == pytest.approx(2 * abs(np.diff(fig.axXY.get_xlim())[0]))
        hx, hy = _fig_frac(fig, fig.axXY, 30 * 1, 60 * 2)
    finally:
        plt.close(fig)
    w.hover_coords = {'plane': 'xy', 'x': hx, 'y': hy}
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
        label = fig.axBar.texts[0].get_text()
        seg = [c for c in fig.axXY.collections if hasattr(c, 'get_segments')][-1].get_segments()[0]
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
        shown = sum(len(c.get_offsets()) for c in fig.axZY.collections if hasattr(c, 'get_offsets'))
        assert expected > 0 and shown == expected
    finally:
        plt.close(fig)


def test_iso_scatter_uses_pixel_sizes_for_radius_and_y():
    """The isometric viewer keeps all points inside its axes with anisotropic pixels."""
    rng = np.random.default_rng(12)
    X, Y, Z = rng.uniform(0, 100, 300), rng.uniform(0, 100, 300), rng.uniform(0, 100, 300)
    w = IsoScatterWidget(X, Y, Z, pixel_sizes=(5, 1, 1))
    assert w.max_radius >= 5 * (Z.max() - Z.min()) / 2

    w2 = IsoScatterWidget(X, Y, Z, pixel_sizes=(1, 3, 1))
    w2.elev, w2.azim = 0, 0
    fig = w2._render()
    try:
        top = [ax for ax in fig.axes if ax.get_title() == "Top View (X-Y)"][0]
        pts = top.collections[0].get_offsets()
        assert np.ptp(pts[:, 1]) == pytest.approx(3 * np.ptp(pts[:, 0]), rel=0.05)
        assert w2.max_radius >= np.abs(pts).max()
    finally:
        plt.close(fig)


# =========================================
# Source: remaining low-severity scan findings (widgets)
# =========================================


def test_pixel_size_dict_with_missing_axis_falls_back_to_one():
    """An uncalibrated axis (None, as image_io reports it) is treated as 1.0 with a warning."""
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max, _parse_zyx_tuple_or_dict

    ps = {'X': 0.5, 'Y': 0.5, 'Z': None, 'unit': 'micron'}
    with pytest.warns(UserWarning, match="axis Z"):
        assert _parse_zyx_tuple_or_dict(ps, default_val=1.0) == (1.0, 0.5, 0.5)
    # Slab defaults stay None without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _parse_zyx_tuple_or_dict({'Z': None, 'Y': 2, 'X': None}) == (None, 2.0, None)

    im = np.random.default_rng(0).random((6, 20, 30))
    with pytest.warns(UserWarning, match="axis Z"):
        fig = show_zyx_max(im, pixel_sizes=ps)
    plt.close(fig)
    with pytest.warns(UserWarning, match="axis Z"):
        w = show_zyx_max_slice_interactive(im, pixel_sizes=ps)
    assert w.sz == 1.0 and w.image_data
    pts = (np.array([0, 5.0]), np.array([0, 19.0]), np.array([0, 29.0]))
    with pytest.warns(UserWarning, match="axis Z"):
        w2 = show_zyx_max_scatter_interactive(pts, pixel_sizes=ps)
    assert w2.sz == 1.0 and w2.image_data


@pytest.mark.parametrize("fn_name", ["show_zyx_max", "show_zyx_max_slabs"])
def test_deprecated_colors_warns_once(fn_name):
    import eigenp_utils.tnia_plotting_anywidgets as tpa

    im = np.random.default_rng(0).random((4, 8, 8))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fig = getattr(tpa, fn_name)(im, colors='red')
    plt.close(fig)
    dep = [r for r in rec if issubclass(r.category, DeprecationWarning) and "'colors'" in str(r.message)]
    assert len(dep) == 1
    assert dep[0].filename == __file__  # points at the caller


@pytest.mark.parametrize("unit, size, expected", [
    ('cm', 5e-5, '20 µm'),       # 256 px * 0.5 µm stored in cm
    ('nm', 500.0, '20 µm'),
    ('micron', 0.5, '20 µm'),
    ('\\u00B5m', 0.5, '20 µm'),  # ImageJ's escaped micro sign
    ('pixel', 1.0, '50 pixel'),  # unknown unit: shown as given
])
def test_scale_bar_uses_pixel_size_unit(unit, size, expected):
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max

    im = np.zeros((4, 256, 256))
    ps = {'Z': size, 'Y': size, 'X': size, 'unit': unit}
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
    for render in ('points', 'density'):
        w = show_zyx_max_scatter_interactive(pts, pixel_sizes=ps, render=render)
        fig = w._render()
        try:
            assert expected in [t.get_text() for ax in fig.axes for t in ax.texts], render
        finally:
            plt.close(fig)


def test_multichannel_default_vmin_shows_negative_channels():
    """vmin=None is 0 for non-negative data and the data minimum for negative data."""
    from eigenp_utils.tnia_plotting_anywidgets import create_multichannel_rgb_cmap

    rng = np.random.default_rng(0)
    neg = rng.uniform(-5, -1, (8, 8))
    pos = rng.uniform(0, 10, (8, 8))
    for fn in (create_multichannel_rgb, create_multichannel_rgb_cmap):
        xy, _, _ = fn([neg, pos], [neg, pos], [neg, pos], colormap=['red', 'blue'], blend='max')
        red = xy[..., 0]
        assert red.max() > 0.5 and np.ptp(red) > 0.5
        # the non-negative channel keeps vmin = 0: its brightest pixel is full blue
        assert xy[..., 2].max() == pytest.approx(1.0, abs=1e-2)
        with pytest.raises(ValueError, match="gamma"):
            fn([neg, pos], [neg, pos], [neg, pos], gamma=[1, 0])


@pytest.mark.skipif(sys.platform == "emscripten", reason="no subprocesses in Pyodide")
def test_widget_module_runs_without_installed_package(tmp_path):
    """The module (with its .js/.css next to it) imports when eigenp_utils itself is
    not importable, e.g. `uv run tnia_plotting_anywidgets.py` with its PEP 723 deps."""
    import shutil
    import subprocess
    from pathlib import Path
    import eigenp_utils.tnia_plotting_anywidgets as tpa

    src = Path(tpa.__file__).parent
    for name in ("tnia_plotting_anywidgets.py", "tnia_plotting_anywidgets.js",
                 "tnia_plotting_anywidgets.css", "iso_scatter.js"):
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
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"MPLBACKEND": "Agg", "PATH": ""})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


def test_one_render_per_multi_trait_update():
    """A middle-thumb drag (x_start and x_end in one message) and a 'C' hover sync
    (x_s and y_s) each rasterise the figure once."""
    im = np.random.default_rng(0).random((10, 40, 50))
    w = show_zyx_max_slice_interactive(im)
    calls = []
    orig = w._render
    w._render = lambda: (calls.append(1), orig())[1]

    with w.hold_trait_notifications():  # how ipywidgets applies one comm message
        w.x_start, w.x_end = w.x_start + 3, w.x_end + 3
    assert len(calls) == 1

    calls.clear()
    info = w.axis_bounds['xy']
    w.hover_coords = {'plane': 'xy', 'x': info['x0'] + 0.3 * (info['x1'] - info['x0']),
                      'y': info['y0_js'] + 0.6 * (info['y1_js'] - info['y0_js']), 't': 1}
    assert len(calls) == 1

    calls.clear()
    w.vmax_list = [0.5]  # any real change still renders
    assert len(calls) == 1


def test_widget_png_ignores_tight_savefig_rcparam():
    """axis_bounds are figure fractions, so the PNG must show the whole figure frame."""
    import base64
    import io
    import matplotlib.image as mpimg

    im = np.random.default_rng(0).random((10, 40, 50))
    sizes = {}
    for bbox in ('standard', 'tight'):
        with matplotlib.rc_context({'savefig.bbox': bbox}):
            w = show_zyx_max_slice_interactive(im, figsize=(8, 6))
        sizes[bbox] = mpimg.imread(io.BytesIO(base64.b64decode(w.image_data))).shape[:2]
    assert sizes['tight'] == sizes['standard']


def test_scatter_copy_params_round_trip():
    rng = np.random.default_rng(0)
    X = rng.uniform(500, 600, 400)
    Y = rng.uniform(0, 50, 400)
    Z = rng.uniform(20, 40, 400)
    w = show_zyx_max_scatter_interactive((Z, Y, X), pixel_sizes=(2, 0.5, 0.5))
    w.x_start, w.x_end = 44, 56
    w.z_start, w.z_end = 6, 14
    w._copy_params(None)
    kwargs = {}
    for line in w.copy_params_string.splitlines():
        key, val = line.rstrip(',').split('=', 1)
        if key.startswith('slabs_'):
            kwargs[key] = eval(val)
    w2 = show_zyx_max_scatter_interactive((Z, Y, X), pixel_sizes=(2, 0.5, 0.5), **kwargs)
    for a in ('x_start', 'x_end', 'y_start', 'y_end', 'z_start', 'z_end'):
        assert getattr(w2, a) == getattr(w, a), a


def test_annotator_point_outside_volume_paints_nothing():
    from eigenp_utils.tnia_plotting_anywidgets import TNIAAnnotatorWidget

    w = TNIAAnnotatorWidget(np.zeros((20, 64, 64), dtype=np.uint8))
    w.add_point(10, 30, 30)
    inside = int((w._annot_img > 0).sum())
    assert inside > 0
    w.add_point(-5, 30, 30)
    w.add_point(10, 80, 30)
    assert int((w._annot_img > 0).sum()) == inside


def test_scatter_widget_drops_nonfinite_points_and_accepts_empty_input():
    rng = np.random.default_rng(0)
    X, Y, Z = rng.uniform(0, 50, 30), rng.uniform(0, 50, 30), rng.uniform(0, 10, 30)
    vals = rng.random(30)
    X[3] = np.nan
    Z[7] = np.inf
    with pytest.warns(UserWarning, match="Dropping 2 of 30"):
        w = show_zyx_max_scatter_interactive((Z, Y, X), channels=vals)
    assert w.N == 28 and len(w.cont_single) == 28
    assert w.cont_single[3] == vals[4] and w.image_data

    # index-list channels are re-indexed to the kept points
    with pytest.warns(UserWarning):
        w = show_zyx_max_scatter_interactive((Z, Y, X), channels=[np.array([0, 3, 4]), np.array([7, 8])])
    assert [list(ix) for ix in w.idx_lists] == [[0, 3], [6]]

    for render in ('points', 'density'):
        w = show_zyx_max_scatter_interactive((np.array([]), np.array([]), np.array([])), render=render)
        assert w.image_data


def test_default_vmax_of_all_nan_channel_does_not_warn():
    im = np.random.default_rng(0).random((4, 8, 8))
    nan = np.full_like(im, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        w = show_zyx_max_slice_interactive([im, nan])
    assert w.vmax_list[1] == 1.0


def test_iso_scatter_subsampling_is_seeded_and_leaves_global_rng_alone():
    rng = np.random.default_rng(0)
    X, Y, Z = rng.random(20000), rng.random(20000), rng.random(20000)
    np.random.seed(42)
    w1 = show_iso_scatter(X, Y, Z)
    assert np.random.rand() == np.random.RandomState(42).rand()
    w2 = show_iso_scatter(X, Y, Z)
    assert np.array_equal(w1.X_orig, w2.X_orig)



# =========================================
# Blending modes and slab projections (keyword-only rendering options)
# =========================================

import eigenp_utils.tnia_plotting_anywidgets as _tpa


def _panel_rgb(fig, name="axXY"):
    return np.asarray(getattr(fig, name).images[0].get_array(), dtype=float)


def _two_channel_planes():
    # channel 0 on in the left column, channel 1 on in the top row
    a = np.zeros((2, 2), dtype=np.float32); a[:, 0] = 1.0
    b = np.zeros((2, 2), dtype=np.float32); b[0, :] = 1.0
    return a, b


def test_minimum_blending_is_per_rgb_minimum_of_white_to_colour_channels():
    a, b = _two_channel_planes()
    xy, _, _ = create_multichannel_rgb([a, b], [a, b], [a, b], vmin=0, vmax=1,
                                       colormap=['magenta', 'cyan'], blend='minimum')
    np.testing.assert_allclose(xy[1, 1], [1, 1, 1])        # both off: white
    np.testing.assert_allclose(xy[1, 0], to_rgb('magenta'))  # only channel 0
    np.testing.assert_allclose(xy[0, 1], to_rgb('cyan'))     # only channel 1
    np.testing.assert_allclose(xy[0, 0], [0, 0, 1])        # min(magenta, cyan) = blue

    # opacity fades a channel towards white, the neutral value of the minimum
    xy, _, _ = create_multichannel_rgb([a, b], [a, b], [a, b], vmin=0, vmax=1,
                                       colormap=['magenta', 'cyan'], opacity=[0.5, 0.0], blend='minimum')
    np.testing.assert_allclose(xy[1, 0], [1, 0.5, 1])
    np.testing.assert_allclose(xy[0, 1], [1, 1, 1])


def test_minimum_blending_with_inverted_colormaps():
    a, b = _two_channel_planes()
    xy, _, _ = _tpa.create_multichannel_rgb_cmap([a, b], [a, b], [a, b], vmin=0, vmax=1,
                                                 colormap=['Blues', 'Reds'], blend='minimum')
    blues, reds = matplotlib.colormaps['Blues'], matplotlib.colormaps['Reds']
    q = lambda c, v: np.asarray(c(np.array([v]), bytes=True)[0, :3]) / 255.0  # 8-bit colormap lookup
    np.testing.assert_allclose(xy[1, 1], np.minimum(q(blues, 0.0), q(reds, 0.0)), atol=1e-6)
    np.testing.assert_allclose(xy[0, 0], np.minimum(q(blues, 1.0), q(reds, 1.0)), atol=1e-6)
    # a plain colour in the colormap path is drawn white -> colour as well
    xy, _, _ = _tpa.create_multichannel_rgb_cmap([a, b], [a, b], [a, b], vmin=0, vmax=1,
                                                 colormap=['Blues', 'red'], blend='minimum')
    np.testing.assert_allclose(xy[1, 1], q(blues, 0.0), atol=1e-6)


def test_default_blending_is_unchanged():
    rng = np.random.default_rng(0)
    planes = [rng.random((6, 7)).astype(np.float32) for _ in range(2)]
    for cmap, mode in ((['magenta', 'cyan'], 'additive'), (['viridis', 'magma'], 'maximum')):
        f0 = show_zyx(planes, planes, planes, colormap=cmap)
        f1 = show_zyx(planes, planes, planes, colormap=cmap, blending=mode)
        np.testing.assert_array_equal(_panel_rgb(f0), _panel_rgb(f1))
        plt.close('all')


def test_minimum_blending_uses_a_white_theme():
    rng = np.random.default_rng(0)
    planes = [rng.random((20, 30)).astype(np.float32) for _ in range(2)]
    fig = show_zyx(planes, planes, planes, rotate_view=30, blending='minimum', pixel_sizes=(1, 1, 1))
    xy = _panel_rgb(fig)
    np.testing.assert_allclose(xy[0, 0], [1, 1, 1], atol=1e-6)  # rotation corner padded white, not black
    assert mcolors.to_hex(fig.patch.get_facecolor()) == '#ffffff'
    assert mcolors.to_hex(fig.axXY.get_facecolor()) == '#ffffff'
    bar = fig.axXY.collections[0]
    assert mcolors.to_hex(bar.get_colors()[0]) == '#000000'  # scale bar visible on white
    plt.close(fig)

    # control: the default theme still pads with black and draws a white bar
    fig = show_zyx(planes, planes, planes, rotate_view=30, pixel_sizes=(1, 1, 1))
    np.testing.assert_allclose(_panel_rgb(fig)[0, 0], [0, 0, 0], atol=1e-6)
    assert mcolors.to_hex(fig.axXY.collections[0].get_colors()[0]) == '#ffffff'
    plt.close(fig)


def test_minimum_blending_single_channel_and_default_colours():
    img = np.zeros((4, 5, 6), dtype=np.float32); img[2, 2, 3] = 1.0
    fig = _tpa.show_zyx_max(img, blending='minimum', vmin=0, vmax=1)
    rgba = fig.axXY.images[0].to_rgba(fig.axXY.images[0].get_array())
    np.testing.assert_allclose(rgba[0, 0, :3], [1, 1, 1], atol=1e-6)  # background white
    np.testing.assert_allclose(rgba[2, 3, :3], [0, 0, 0], atol=1e-6)  # signal in the default ink (black)
    plt.close(fig)
    assert _tpa._channel_colormaps(None, 3, 'minimum') == ['black', 'magenta', 'lime']
    assert _tpa._channel_colormaps(None, 3) == ['white', 'lime', 'magenta']


def test_blend_colors_minimum():
    vals = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    cols = _tpa.blend_colors(vals, ['magenta', 'cyan'], vmin=0, vmax=1, blending='minimum')
    np.testing.assert_allclose(cols, [[1, 1, 1], to_rgb('magenta'), [0, 0, 1]])
    np.testing.assert_allclose(
        _tpa.blend_colors(vals, ['magenta', 'cyan'], vmin=0, vmax=1),
        _tpa.blend_colors(vals, ['magenta', 'cyan'], vmin=0, vmax=1, blending='additive'),
    )


def test_invalid_blending_and_projection_raise():
    img = np.zeros((3, 4, 5))
    with pytest.raises(ValueError, match="blending"):
        _tpa.show_zyx_max(img, blending='multiply')
    with pytest.raises(ValueError, match="projection"):
        _tpa.show_zyx_max(img, projection='iso')
    with pytest.raises(ValueError, match="attenuation"):
        _tpa.show_zyx_max(img, projection='attenuated_mip', attenuation=-1)


@pytest.mark.parametrize("projection, ref", [("mip", np.max), ("minip", np.min), ("average", np.mean)])
def test_slab_projections_match_numpy(projection, ref):
    rng = np.random.default_rng(1)
    img = rng.random((8, 9, 10)).astype(np.float32)
    fig = show_zyx_max_slabs(img, [2, 7], [1, 5], [3, 6], projection=projection, vmin=0, vmax=1)
    np.testing.assert_allclose(_panel_rgb(fig), ref(img[3:6], axis=0), rtol=1e-6)
    plt.close(fig)


def test_attenuated_mip_matches_napari_formula():
    # one ray of two samples, display range [0, 1]: scaled_i = v_i * exp(-a * (cumsum_i - 1))
    a = 1.0
    ray = np.array([0.5, 1.0], dtype=np.float32).reshape(2, 1, 1)
    expected = max(0.5 * np.exp(-a * (0.5 - 1)), 1.0 * np.exp(-a * (1.5 - 1)))
    np.testing.assert_allclose(_tpa._attenuated_mip(ray, 0, 0.0, 1.0, a), expected, rtol=1e-6)
    # zero attenuation is the plain MIP of the clipped data, returned in data units
    vol = np.random.default_rng(2).random((5, 4, 3)).astype(np.float32) * 10
    np.testing.assert_allclose(_tpa._attenuated_mip(vol, 0, 2.0, 8.0, 0.0),
                               np.clip(vol.max(axis=0), 2.0, 8.0), rtol=1e-5)


def test_attenuated_mip_hides_structures_behind_bright_ones():
    # A dim haze fills the slab; the same bright voxel is placed shallower or deeper
    vol = np.full((30, 1, 2), 0.2, dtype=np.float32)
    vol[2, 0, 0] = 1.0   # near the entry face (index 0)
    vol[25, 0, 1] = 1.0  # deep behind the haze
    # shallow: 1 * exp(-0.3 * (0.2 * 2 + 1 - 1)) = 0.89; deep: exp(-0.3 * 0.2 * 25) = 0.22,
    # below the first haze sample (0.2 * exp(0.3 * 0.8) = 0.25)
    out = _tpa._attenuated_mip(vol, 0, 0.0, 1.0, 0.3)
    np.testing.assert_allclose(out[0, 0], np.exp(-0.3 * 0.4), rtol=1e-5)
    np.testing.assert_allclose(out[0, 1], 0.2 * np.exp(0.3 * 0.8), rtol=1e-5)
    # control: without attenuation both are fully visible
    np.testing.assert_allclose(_tpa._attenuated_mip(vol, 0, 0.0, 1.0, 0.0), [[1.0, 1.0]])


def test_slice_widget_blending_and_projection_kwargs():
    rng = np.random.default_rng(3)
    chans = [rng.random((6, 12, 14)).astype(np.float32) for _ in range(2)]
    w = TNIASliceWidget(chans, blending='minimum', projection='attenuated_mip', attenuation=0.2)
    # keyword-only: no GUI traits were added
    assert not {'blending', 'projection', 'attenuation'} & set(w.trait_names())
    assert w.colors_resolved == ['black', 'magenta']
    assert w.channel_colors[0] == '#9e9e9e'  # dark ink lifted for the dark GUI
    fig = w._render()
    assert mcolors.to_hex(fig.patch.get_facecolor()) == '#ffffff'
    plt.close(fig)
    w.copy_params_trigger += 1
    assert "blending='minimum'" in w.copy_params_string
    assert "projection='attenuated_mip'" in w.copy_params_string
    assert "attenuation=0.2" in w.copy_params_string

    # interactive wrappers pass the options through
    w2 = show_zyx_max_slice_interactive(chans[0], blending='minimum', projection='minip')
    assert (w2.blending, w2.projection) == ('minimum', 'minip')
    w3 = show_zyx_max_slice_interactive_point_annotator(chans, blending='minimum')
    assert w3.colors_resolved[:2] == ['black', 'magenta'] and w3.blending == 'minimum'


def test_scatter_widget_minimum_blending():
    rng = np.random.default_rng(4)
    pts = rng.random((200, 3)) * [8, 20, 30]
    for render in ('points', 'density'):
        w = show_zyx_max_scatter_interactive(pts, channels=list(rng.random((2, 200))), render=render,
                                             blending='minimum')
        assert w.subplot_bg == 'white'
        fig = w._render()
        assert mcolors.to_hex(fig.axXY.get_facecolor()) == '#ffffff'
        plt.close(fig)
    # control: default background is still black
    w = show_zyx_max_scatter_interactive(pts, render='points')
    assert w.subplot_bg == 'black'


def test_scatter_widget_accepts_n_by_c_channel_array():
    rng = np.random.default_rng(5)
    pts = rng.random((150, 3)) * [8, 20, 30]
    vals = rng.random((150, 2))
    w = show_zyx_max_scatter_interactive(pts, channels=vals, render='points')
    assert w.mode == 'cont_multi' and w.C == 2
    plt.close(w._render())
    with pytest.raises(ValueError, match="n_points"):
        show_zyx_max_scatter_interactive(pts, channels=rng.random((149, 2)), render='points')


def test_show_zyx_max_accepts_a_channel_list():
    rng = np.random.default_rng(6)
    chans = [rng.random((4, 5, 6)).astype(np.float32) for _ in range(2)]
    fig = _tpa.show_zyx_max(chans, colormap=['magenta', 'cyan'])
    assert _panel_rgb(fig).shape == (5, 6, 3)
    plt.close(fig)


def test_interactive_wrapper_warns_once_for_a_missing_pixel_size_axis():
    img = np.random.default_rng(7).random((4, 8, 9)).astype(np.float32)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        show_zyx_max_slice_interactive(img, pixel_sizes={'Z': None, 'Y': 0.5, 'X': 0.5})
    msgs = [str(r.message) for r in rec if "No value for axis Z" in str(r.message)]
    assert len(msgs) == 1
