import sys
import types
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from eigenp_utils.tnia_plotting_anywidgets import (
    show_xyz_slice,
    show_xyz_max_slabs,
    create_multichannel_rgb,
)
from matplotlib.colors import to_rgb


def test_show_xyz_slice_returns_correct_slices():
    arr = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    fig = show_xyz_slice(arr, x=1, y=1, z=2, use_plt=False)
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


def test_show_xyz_max_slabs_projection():
    arr = np.arange(4 * 3 * 2).reshape(4, 3, 2)
    fig = show_xyz_max_slabs(arr, x=[0, 1], y=[0, 2], z=[1, 4])
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
    assert hasattr(tnia3d, "show_xyz")


def test_create_multichannel_rgb_basic():
    xy_list = [np.ones((2, 2)), np.zeros((2, 2))]
    xz_list = [np.zeros((2, 2)), np.ones((2, 2))]
    zy_list = [np.zeros((2, 2)), np.zeros((2, 2))]
    xy_rgb, xz_rgb, zy_rgb = create_multichannel_rgb(
        xy_list, xz_list, zy_list, colors=["red", "green"]
    )
    red = np.asarray(to_rgb("red"))
    green = np.asarray(to_rgb("green"))
    expected_xy = np.broadcast_to(red, (2, 2, 3))
    expected_xz = np.broadcast_to(green, (2, 2, 3))
    expected_zy = np.zeros((2, 2, 3))
    assert np.allclose(xy_rgb, expected_xy)
    assert np.allclose(xz_rgb, expected_xz)
    assert np.allclose(zy_rgb, expected_zy)

def test_show_xyz_max_scatter_interactive_colormap():
    from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_scatter_interactive
    X = np.random.rand(10) * 10
    Y = np.random.rand(10) * 10
    Z = np.random.rand(10) * 10
    channels = np.random.rand(10)

    # Should not throw exception for invalid RGBA string, and _render should not throw NameError
    w1 = show_xyz_max_scatter_interactive(X, Y, Z, channels=channels, colors='viridis', render='points')
    w1._render() # Trigger render directly

    w2 = show_xyz_max_scatter_interactive(X, Y, Z, channels=channels, colors='viridis', render='density')
    w2._render() # Trigger render directly

    channels_multi = [np.random.rand(10), np.random.rand(10)]
    w3 = show_xyz_max_scatter_interactive(X, Y, Z, channels=channels_multi, colors=['viridis', 'plasma'], render='points')
    w3._render() # Trigger render directly

    w4 = show_xyz_max_scatter_interactive(X, Y, Z, channels=channels_multi, colors=['viridis', 'plasma'], render='density')
    w4._render() # Trigger render directly

    assert w1 is not None
    assert w2 is not None
    assert w3 is not None
    assert w4 is not None
