import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive, TNIASliceWidget

def test_single_channel_instantiation():
    im = np.zeros((10, 20, 30))
    w = show_zyx_max_slice_interactive(im)
    assert isinstance(w, TNIASliceWidget)
    assert w.num_channels == 1
    assert len(w.channel_names) == 1
    # Render should produce image data (triggered by observer in init)
    assert w.image_data is not None and len(w.image_data) > 0

def test_multi_channel_instantiation():
    im = [np.zeros((10, 20, 30)) for _ in range(3)]
    w = show_zyx_max_slice_interactive(im)
    assert w.num_channels == 3
    assert len(w.channel_names) == 3
    assert w.channel_names == ["Channel 0", "Channel 1", "Channel 2"]
    assert w.opacity_list == [1.0, 1.0, 1.0]
    assert w.image_data is not None and len(w.image_data) > 0

def test_channel_visibility_update():
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
    im = [np.zeros((10, 10, 10)) for _ in range(2)]
    w = show_zyx_max_slice_interactive(im, colors=None)
    assert w.colors_resolved == ['magenta', 'cyan'] # Defaults

    w2 = show_zyx_max_slice_interactive(im, colors=['red', 'blue'])
    assert w2.colors_resolved == ['red', 'blue']

def test_show_zyx_max_slice_interactive_point_annotator_args():
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive_point_annotator, TNIAAnnotatorWidget
    im = [np.zeros((10, 10, 10)) for _ in range(2)]
    w = show_zyx_max_slice_interactive_point_annotator(
        im,
        sxy=2,
        sz=3,
        point_size_scale=0.05,
        colors=['red', 'blue'],
        opacity=[0.5, 0.8]
    )
    assert isinstance(w, TNIAAnnotatorWidget)
    assert w.sxy == 2
    assert w.sz == 3

def test_point_size_scaling():
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
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_scatter_interactive
    X = np.random.rand(10) * 10
    Y = np.random.rand(10) * 10
    Z = np.random.rand(10) * 10
    channels = np.random.rand(10)

    # Should not throw exception for invalid RGBA string
    w1 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels, colors='viridis', render='points')
    w2 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels, colors='viridis', render='density')

    channels_multi = [np.random.rand(10), np.random.rand(10)]
    w3 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels_multi, colors=['viridis', 'plasma'], render='points')
    w4 = show_zyx_max_scatter_interactive((Z, Y, X), channels=channels_multi, colors=['viridis', 'plasma'], render='density')

    assert w1 is not None
    assert w2 is not None
    assert w3 is not None
    assert w4 is not None

def test_show_zyx_max_scatter_interactive_signature():
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
