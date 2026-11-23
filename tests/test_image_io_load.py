"""image_io.load_image / channels_by (synthetic data: TIFFs, arrays, reader-style DataArrays)."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
image_io = pytest.importorskip("eigenp_utils.image_io")

from eigenp_utils import image_io as sio  # noqa: E402

NAMES = ["DAPI", "WGA", "TOMM20", "PMP70"]
INFO = {
    "DAPI": {"role": "dna", "color": "gray"},
    "WGA": {"role": "membrane", "color": "magenta"},
    "TOMM20": {"role": "mito", "color": "green"},
    "PMP70": {"role": "perox", "color": "yellow"},
}


def _reader_dataarray(nz=3, ny=16, nx=20, dz=2.0, dxy=0.108):
    """What a vendor reader's ``to_xarray(squeeze=False)`` typically returns: dims P,Z,C,Y,X,
    channel names in C, physical Z/Y/X coordinates."""
    data = np.random.default_rng(0).integers(0, 4096, size=(1, nz, len(NAMES), ny, nx), dtype=np.uint16)
    da = xr.DataArray(
        data,
        dims=("P", "Z", "C", "Y", "X"),
        coords={"P": ["XYPos:0"], "Z": np.arange(nz) * dz, "C": NAMES, "Y": np.arange(ny) * dxy, "X": np.arange(nx) * dxy},
    )
    return data, da


def test_reader_dataarray_channels_and_roles():
    data, src = _reader_dataarray()
    da = sio.load_image(src, channel_info=INFO)
    assert da.dims == ("S", "T", "C", "Z", "Y", "X")
    assert list(da.coords["C"].values) == NAMES  # names come from the source, not from a hand-written list
    ps = image_io.get_stczyx_pixel_sizes(da)
    assert ps["Z"] == pytest.approx(2.0) and ps["X"] == pytest.approx(0.108)
    assert da.dtype == np.uint16  # raw values, no dtype change
    np.testing.assert_array_equal(da.sel(C="TOMM20").values[0, 0], data[0, :, 2])
    assert sio.channels_by(da, "role", ["membrane", "dna"]) == ["WGA", "DAPI"]


def test_channel_info_validation_and_channels_by():
    _, src = _reader_dataarray()
    with pytest.raises(KeyError, match="DAPl"):
        sio.load_image(src, channel_info={"DAPl": {"role": "dna"}})
    dup = {**INFO, "PMP70": {"role": "mito"}}
    da = sio.load_image(src, channel_info=dup)
    with pytest.raises(ValueError, match="several channels"):
        sio.channels_by(da, "role", ["mito"])
    with pytest.raises(KeyError, match="lamin"):
        sio.channels_by(da, "role", ["lamin"])
    assert sio.channels_by(da, "role", ["lamin", "dna"], allow_missing=True) == ["DAPI"]


def test_tiff_imagej_roundtrip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    arr = np.random.default_rng(1).integers(0, 1000, size=(3, 2, 8, 9), dtype=np.uint16)  # ZCYX
    p = tmp_path / "img.tif"
    tifffile.imwrite(p, arr, imagej=True, resolution=(1 / 0.2, 1 / 0.2), metadata={"axes": "ZCYX", "spacing": 1.5, "unit": "um"})
    da = sio.load_image(p, channel_names=["DAPI", "WGA"], channel_info={k: INFO[k] for k in ("DAPI", "WGA")})
    assert da.dims == ("S", "T", "C", "Z", "Y", "X") and da.sizes["Z"] == 3
    np.testing.assert_array_equal(da.sel(C="WGA").values[0, 0], arr[:, 1])
    ps = image_io.get_stczyx_pixel_sizes(da)
    assert ps["Z"] == pytest.approx(1.5) and ps["X"] == pytest.approx(0.2)
    assert sio.channels_by(da, "role", "dna") == ["DAPI"]


def test_in_memory_array_and_describe():
    arr = np.zeros((2, 5, 6), dtype=np.float32)
    da = sio.load_image(arr, input_axes="CYX", channel_names=["DAPI", "WGA"], pixel_sizes={"Y": 0.1, "X": 0.1, "unit": "µm"})
    from eigenp_utils import hist_imshow

    text = hist_imshow(da)["info"]
    assert "DAPI" in text and "X=0.1" in text


def test_unsupported_suffix_points_to_reader(tmp_path):
    with pytest.raises(ValueError, match="own reader"):
        sio.load_image(tmp_path / "x.czi")


def test_describe_image_is_deprecated():
    da = sio.load_image(np.zeros((2, 4, 5)), input_axes="CYX", channel_names=["a", "b"])
    with pytest.warns(DeprecationWarning, match="hist_imshow"):
        assert "a" in sio.describe_image(da)


def test_channels_by_unique_false_and_scalar():
    _, src = _reader_dataarray()
    da = sio.load_image(src, channel_info={**INFO, "PMP70": {"role": "mito", "color": "yellow"}})
    assert sio.channels_by(da, "role", "mito", unique=False) == ["TOMM20", "PMP70"]
    assert sio.channels_by(da, "color", ["yellow", "gray"], unique=False) == ["PMP70", "DAPI"]


def test_tiff_samples_axis_is_read_as_channels(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    arr = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    p = tmp_path / "planar.tif"
    tifffile.imwrite(p, arr)  # default: planar 'SYX'
    with tifffile.TiffFile(p) as tif:
        assert "S" in tif.series[0].axes
    da = sio.load_image(p)
    assert da.sizes["S"] == 1 and da.sizes["C"] == 3
    np.testing.assert_array_equal(da.isel(S=0, T=0, Z=0).values, arr)
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[..., 1] = 7
    p2 = tmp_path / "rgb.tif"
    tifffile.imwrite(p2, rgb, photometric="rgb")
    da2 = sio.load_image(p2)
    assert da2.sizes["C"] == 3 and da2.sizes["S"] == 1
    assert int(da2.isel(S=0, T=0, Z=0, C=1).values.max()) == 7


def test_multiseries_ome_tiff_stacks_positions(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "multi.ome.tif"
    a0 = np.zeros((2, 8, 9), np.uint16)
    a1 = np.ones((2, 8, 9), np.uint16)
    with tifffile.TiffWriter(p, ome=True) as w:
        w.write(a0, metadata={"axes": "CYX"})
        w.write(a1, metadata={"axes": "CYX"})
    da = sio.load_image(p)
    assert da.sizes["S"] == 2 and da.sizes["C"] == 2
    assert int(da.isel(S=1).values.max()) == 1 and int(da.isel(S=0).values.max()) == 0
    da1 = sio.load_image(p, series=1)
    assert da1.sizes["S"] == 1 and int(da1.values.max()) == 1


def test_pixel_size_override_is_per_axis(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "ij.tif"
    tifffile.imwrite(p, np.zeros((4, 8, 9), np.uint16), imagej=True, resolution=(5, 5), metadata={"axes": "ZYX", "unit": "um"})
    ps = image_io.get_stczyx_pixel_sizes(sio.load_image(p, pixel_sizes={"Z": 0.5}))
    assert ps["Z"] == pytest.approx(0.5) and ps["X"] == pytest.approx(0.2) and ps["Y"] == pytest.approx(0.2)
    ps2 = image_io.get_stczyx_pixel_sizes(sio.load_image(p, pixel_sizes=(0.7, None, None)))
    assert ps2["Z"] == pytest.approx(0.7) and ps2["X"] == pytest.approx(0.2)


def test_integer_channel_labels_round_trip_through_sel(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "czyx.tif"
    tifffile.imwrite(p, np.arange(2 * 3 * 8 * 9, dtype=np.uint16).reshape(3, 2, 8, 9), imagej=True, metadata={"axes": "ZCYX"})
    da = sio.load_image(p, channel_info={0: {"role": "dna"}, 1: {"role": "membrane"}})
    names = sio.channels_by(da, "role", ["membrane", "dna"])
    assert names == [1, 0]
    assert da.sel(C=names).sizes["C"] == 2  # native labels work in sel


def test_channel_info_after_rename_gives_clear_error():
    src = xr.DataArray(np.zeros((2, 8, 9)), dims=("C", "Y", "X"), coords={"C": ["DAPI", "WGA"]})
    with pytest.raises(KeyError, match="key channel_info by the new names"):
        sio.load_image(src, channel_names=["dna_ch", "mem_ch"], channel_info={"DAPI": {"role": "dna"}, "WGA": {"role": "membrane"}})
    da = sio.load_image(src, channel_names=["dna_ch", "mem_ch"], channel_info={"dna_ch": {"role": "dna"}, "mem_ch": {"role": "membrane"}})
    assert sio.channels_by(da, "role", "dna") == ["dna_ch"]
