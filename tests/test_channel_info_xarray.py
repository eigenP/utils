"""Channel metadata as coordinates along C, and xarray input to the viewer / annotator."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
xr = pytest.importorskip("xarray")

from eigenp_utils.image_io import array_to_stczyx_xarray, channels_where, set_channel_info  # noqa: E402

NAMES = ["DAPI", "SOX2", "TBXT", "BF"]


def _da(nz=3, ny=12, nx=10, names=NAMES, ps=None):
    arr = np.arange(len(names) * nz * ny * nx, dtype=np.float32).reshape(len(names), nz, ny, nx)
    ps = ps or {"Z": 2.0, "Y": 0.3, "X": 0.3, "unit": "µm"}
    return array_to_stczyx_xarray(arr, input_axes="CZYX", channel_names=names, pixel_sizes=ps)


INFO = {
    "DAPI": {"color": "gray", "role": "dna"},
    "SOX2": {"color": "cyan", "role": "marker"},
    "TBXT": {"color": "yellow", "role": "marker"},
    "BF": {"color": "gray", "role": "brightfield"},
}


def test_set_channel_info_follows_selection_and_reordering():
    da = set_channel_info(_da(), INFO)
    assert list(da.coords["role"].values) == ["dna", "marker", "marker", "brightfield"]
    sub = da.sel(C=["TBXT", "DAPI"])  # reorder + subset: metadata travels with the channel
    assert list(sub.coords["color"].values) == ["yellow", "gray"]
    assert list(sub.coords["role"].values) == ["marker", "dna"]
    np.testing.assert_array_equal(sub.sel(C="DAPI").values, da.sel(C="DAPI").values)


def test_set_channel_info_strict_catches_typos_and_warns_on_missing():
    with pytest.raises(KeyError, match="DAPl"):
        set_channel_info(_da(), {"DAPl": {"role": "dna"}})
    with pytest.warns(UserWarning, match="no channel info"):
        da = set_channel_info(_da(), {"DAPI": {"role": "dna"}})
    assert da.coords["role"].values.tolist() == ["dna", "", "", ""]
    with pytest.warns(UserWarning):
        num = set_channel_info(_da(), {"DAPI": {"ex_nm": 405}})
    assert num.coords["ex_nm"].values[0] == 405 and np.isnan(num.coords["ex_nm"].values[1])
    with pytest.raises(ValueError, match="clashes"):
        set_channel_info(_da(), {"DAPI": {"Z": 1}})


def test_channels_where():
    da = set_channel_info(_da(), INFO)
    assert channels_where(da, role="dna") == ["DAPI"]
    assert channels_where(da, role=["marker", "dna"]) == ["DAPI", "SOX2", "TBXT"]  # C order, not query order
    assert channels_where(da, role="marker", color="cyan") == ["SOX2"]
    assert channels_where(da, color=lambda c: c != "gray") == ["SOX2", "TBXT"]
    with pytest.raises(KeyError):
        channels_where(da, organelle="mito")


def test_viewer_accepts_stczyx_dataarray():
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

    da = set_channel_info(_da(ps={"Z": 2.0, "Y": 0.5, "X": 0.25, "unit": "µm"}), INFO)
    w = show_zyx_max_slice_interactive(da.sel(C=["DAPI", "TBXT"]))
    assert w.num_channels == 2
    assert w.channel_names == ["DAPI", "TBXT"]
    assert w.colors_resolved[1] == "yellow"
    assert (w.sz, w.sy, w.sx) == pytest.approx((2.0, 0.5, 0.25))
    # explicit arguments win over the DataArray's coordinates
    w2 = show_zyx_max_slice_interactive(da.sel(C=["DAPI", "TBXT"]), colormap=["red", "blue"], channel_labels=["a", "b"])
    assert w2.colors_resolved == ["red", "blue"] and w2.channel_names == ["a", "b"]


def test_viewer_dataarray_matches_list_input():
    from eigenp_utils.tnia_plotting_anywidgets import _viewer_input_from_xarray

    da = _da()
    chans, cmap, labels, ps = _viewer_input_from_xarray(da, None, None, None)
    assert labels == NAMES and cmap is None
    for i, name in enumerate(NAMES):
        np.testing.assert_array_equal(chans[i], da.sel(C=name).isel(S=0, T=0).values)
    assert ps["Z"] == pytest.approx(2.0) and ps["X"] == pytest.approx(0.3)
    # plain inputs pass through unchanged
    arr = np.zeros((2, 3, 4))
    assert _viewer_input_from_xarray(arr, "x", "y", "z")[0] is arr


def test_viewer_rejects_unselected_scene_or_time():
    from eigenp_utils.tnia_plotting_anywidgets import _viewer_input_from_xarray

    arr = np.zeros((2, 1, 1, 3, 4, 5), dtype=np.float32)  # two scenes
    da = array_to_stczyx_xarray(arr, input_axes="STCZYX")
    with pytest.raises(ValueError, match=r"isel\(S=0\)"):
        _viewer_input_from_xarray(da, None, None, None)


def test_label_annotator_uses_channel_coords():
    pytest.importorskip("anywidget")
    from eigenp_utils.label_annotator import LabelAnnotator

    da = set_channel_info(_da(), INFO).sel(C=["SOX2", "DAPI"])
    w = LabelAnnotator(da)
    assert w.channel_names == ["SOX2", "DAPI"]
    assert w.channel_colors == ["#00ffff", "#ffffff"]  # 'gray' colormap -> its end colour
    w2 = LabelAnnotator(da, channel_colors=["#123456", "red"])
    assert w2.channel_colors == ["#123456", "#ff0000"]
