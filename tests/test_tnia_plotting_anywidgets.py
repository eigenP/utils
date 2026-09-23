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

    font_size_scaled = None
    linewidth_scaled = None
    for ax in fig_slice_scaled.axes:
        for txt in ax.texts:
            if expected_text in txt.get_text():
                font_size_scaled = txt.get_fontsize()
        for collection in ax.collections:
            if collection.get_linewidth():
                linewidth_scaled = collection.get_linewidth()[0]

    assert font_size_scaled is not None
    assert font_size_scaled > font_size_unscaled

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
def test_tnia_annotator_widget():
    """Test that tnia annotator widget works as expected."""
    from eigenp_utils.tnia_plotting_anywidgets import TNIAAnnotatorWidget
    import numpy as np
    import os

    os.environ['TEST_DIR'] = '/tmp'

    im = np.random.randint(0, 255, (10, 10, 10), dtype=np.uint8)
    w = TNIAAnnotatorWidget(im)
    w.points = [[5, 5, 5], [6, 6, 6]]
    w.save_csv_filename = "$TEST_DIR/test_points.csv"
    w._save_csv(None)

    with open("/tmp/test_points.csv", "r") as f:
        print(f.read())

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
        from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive

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

        widget = show_xyz_max_slice_interactive(
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
    print(widget1.figsize)
    print(widget2.figsize)
    print(widget3.figsize)

# =========================================
# Source: test_tnia_plotting_anywidgets_3d_logic.py
# =========================================

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")



def test_show_zyx_slice_returns_correct_slices():
    """Test that show zyx slice returns correct slices works as expected."""
    arr = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    fig = show_zyx_slice(arr, x=1, y=1, z=2, use_plt=False)
    assert isinstance(fig, Figure)
    xy_expected = arr[2, :, :]
    xz_expected = arr[:, 1, :]
    zy_expected = np.flip(np.rot90(arr[:, :, 1], 1), 0)
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
