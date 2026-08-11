import sys
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
def test_marimo_update():
    """Test that marimo update works as expected."""
    import marimo
    import pathlib

    app_code = """
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
    """
    with open("marimo_app.py", "w") as f:
        f.write(app_code)

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

    # Test the height ratios generated inside the gridspec
    gs_dict = fig_dict.axes[0].get_subplotspec().get_gridspec().get_height_ratios()
    gs_tuple = fig_tuple.axes[0].get_subplotspec().get_gridspec().get_height_ratios()

    assert gs_dict == gs_tuple

def test_xy_anisotropy():
    """
    Test that XY anisotropy is correctly respected in show_zyx spacing calculations.
    """
    im = np.random.rand(10, 50, 100) # Z, Y, X
    # Highly anisotropic XY pixels
    w = show_zyx_max_slice_interactive(im, pixel_sizes={'Z': 1.0, 'Y': 0.2, 'X': 1.0})
    fig = w._render()

    # Check gridspec ratios directly to see if physical scaling is applied
    gs = fig.axes[0].get_subplotspec().get_gridspec()
    width_ratios = gs.get_width_ratios()
    height_ratios = gs.get_height_ratios()

    # X physical = 100 * 1 = 100. Z physical = 10 * 1 = 10. Max width = 100.
    # Y physical = 50 * 0.2 = 10. Z physical = 10 * 1 = 10. Max height = 10.
    assert width_ratios[0] == 100
    assert height_ratios[1] == 10

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

    # Compute where this target voxel lands in physical space
    px_target = (target_x + 0.5) * pixel_sizes[2]
    py_target = (target_y + 0.5) * pixel_sizes[1]

    # Extract calculated axis bounds for 'xy' plane
    info = widget.axis_bounds['xy']
    b_x0, b_y0, b_w, b_h = info['bbox']
    xlim = info['xlim']
    ylim = info['ylim']

    # Map physical coords to normalized figure coords
    u = (px_target - xlim[0]) / (xlim[1] - xlim[0])
    v = (py_target - ylim[0]) / (ylim[1] - ylim[0])

    frac_x = b_x0 + u * b_w
    mpl_y_frac = b_y0 + v * b_h
    frac_y = 1.0 - mpl_y_frac  # Convert to JS top-down fraction

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

    # Fetch updated bounds for the XY plane
    widget._render_wrapper(None)
    info = widget.axis_bounds['xy']
    b_x0, b_y0, b_w, b_h = info['bbox']

    # Target voxel fraction within axXY
    u = (target_x + 0.5) / X
    v_from_top = (target_y + 0.5) / Y
    v_mpl = 1.0 - v_from_top

    frac_x = b_x0 + u * b_w
    mpl_y_frac = b_y0 + v_mpl * b_h
    frac_y = 1.0 - mpl_y_frac

    # Simulate click (trigger handle click manually to avoid traitlet identity check filtering)
    click_dict = {'plane': 'xy', 'x': frac_x, 'y': frac_y}
    widget._handle_click({'new': click_dict})

    assert [target_z, target_y, target_x] in widget.points, \
        f"Expected {[target_z, target_y, target_x]} in {widget.points}"

    # Verify point deletion
    widget.annotation_action = 'delete'
    widget._handle_click({'new': click_dict})

    assert [target_z, target_y, target_x] not in widget.points, \
        f"Point {[target_z, target_y, target_x]} was not deleted from {widget.points}"


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



def test_ground_truth_user_click_annotation():
    # 1. Setup anisotropic image volume
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

    # Target voxel to click on: [z, y, x]
    target_z, target_y, target_x = 8, 20, 45
    widget.z_s = target_z

    # 2. Render figure and locate axXY safely
    fig = widget._render()
    try:
        # Robustly locate ax_xy without assuming fig.axXY exists
        ax_xy = None
        if hasattr(fig, 'axXY'):
            ax_xy = fig.axXY
        else:
            # Match by image extent [0, X*px, Y*py, 0]
            target_extent = (0.0, X * widget.sx, Y * widget.sy, 0.0)
            for ax in fig.axes:
                images = ax.get_images()
                if images and np.allclose(images[0].get_extent(), target_extent):
                    ax_xy = ax
                    break

            # Fallback based on axes count if extent match is inconclusive
            if ax_xy is None:
                ax_xy = fig.axes[1] if len(fig.axes) == 5 else fig.axes[0]

        # Physical coordinates of voxel center
        phys_x = (target_x + 0.5) * widget.sx
        phys_y = (target_y + 0.5) * widget.sy

        # Matplotlib native transform pipeline:
        # Data coords (um) -> Display pixels -> Normalized Figure Coords [0.0 to 1.0]
        display_pixel = ax_xy.transData.transform((phys_x, phys_y))
        fig_norm = fig.transFigure.inverted().transform(display_pixel)

        # Convert Matplotlib bottom-left origin to JS top-left origin
        simulated_user_click_x = float(fig_norm[0])
        simulated_user_click_y = float(1.0 - fig_norm[1])

    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)

    # 3. Inject simulated raw browser click event into the widget
    # Use direct handle click bypass logic since anywidget identity cache might drop identical clicks
    widget._handle_click({'new': {
        'plane': 'xy',
        'x': simulated_user_click_x,
        'y': simulated_user_click_y
    }})

    # 4. Assert ground-truth mapping
    assert [target_z, target_y, target_x] in widget.points, \
        f"Click at ({simulated_user_click_x:.3f}, {simulated_user_click_y:.3f}) " \
        f"mapped incorrectly. Expected {[target_z, target_y, target_x]} in {widget.points}"
