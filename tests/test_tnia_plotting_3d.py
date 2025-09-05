import sys
import types
from pathlib import Path
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub ipywidgets if missing to satisfy imports
widgets = types.ModuleType("ipywidgets")
widgets.interact = lambda *args, **kwargs: None
widgets.IntSlider = type("IntSlider", (), {})
widgets.FloatRangeSlider = type("FloatRangeSlider", (), {})
widgets.Layout = type("Layout", (), {})
sys.modules.setdefault("ipywidgets", widgets)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from tnia_plotting_3d import show_xyz_slice, show_xyz_max_slabs, create_multichannel_rgb
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
