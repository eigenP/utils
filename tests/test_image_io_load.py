"""image_io.load_image / channels_by / describe_image (synthetic data; nd2 via a fake ND2File that mimics nd2 0.12 to_xarray)."""

import sys
import types

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


def _fake_nd2_module(monkeypatch, nz=3, ny=16, nx=20, dz=2.0, dxy=0.108, rgb=False):
    """Mimic nd2.ND2File.to_xarray(squeeze=False): dims P,Z,C,Y,X, C = channel names, physical coords."""
    data = np.random.default_rng(0).integers(0, 4096, size=(1, nz, len(NAMES), ny, nx), dtype=np.uint16)
    da = xr.DataArray(
        data,
        dims=("P", "Z", "C", "Y", "X"),
        coords={"P": ["XYPos:0"], "Z": np.arange(nz) * dz, "C": NAMES, "Y": np.arange(ny) * dxy, "X": np.arange(nx) * dxy},
    )

    class FakeND2:
        closed = False
        components_per_channel = 3 if rgb else 1

        def __init__(self, path):
            self.path = path

        def to_xarray(self, delayed=True, squeeze=True):
            assert squeeze is False, "loader must not squeeze (single-channel files would lose C)"
            return da

        def voxel_size(self):
            return types.SimpleNamespace(x=dxy, y=dxy, z=dz)

        def close(self):
            FakeND2.closed = True

    monkeypatch.setitem(sys.modules, "nd2", types.SimpleNamespace(ND2File=FakeND2))
    return data, FakeND2


def test_nd2_channels_from_file_and_roles(monkeypatch, tmp_path):
    data, fake = _fake_nd2_module(monkeypatch)
    path = tmp_path / "S1_002.nd2"
    da = sio.load_image(path, channel_info=INFO)
    assert da.dims == ("S", "T", "C", "Z", "Y", "X")
    assert list(da.coords["C"].values) == NAMES  # names come from the file, not from a hand-written list
    assert fake.closed  # eager load closes the file
    ps = image_io.get_stczyx_pixel_sizes(da)
    assert ps["Z"] == pytest.approx(2.0) and ps["X"] == pytest.approx(0.108)
    assert da.attrs["source"] == str(path)
    # raw values are untouched (no dtype change, no scaling)
    assert da.dtype == np.uint16
    np.testing.assert_array_equal(da.sel(C="TOMM20").values[0, 0], data[0, :, 2])
    # roles in the requested order (e.g. Cellpose input order), independent of file order
    assert sio.channels_by(da, "role", ["membrane", "dna"]) == ["WGA", "DAPI"]


def test_nd2_rgb_rejected(monkeypatch, tmp_path):
    _fake_nd2_module(monkeypatch, rgb=True)
    with pytest.raises(NotImplementedError):
        sio.load_image(tmp_path / "x.nd2")


def test_channel_info_validation_and_channels_by(monkeypatch, tmp_path):
    _fake_nd2_module(monkeypatch)
    with pytest.raises(KeyError, match="DAPl"):
        sio.load_image(tmp_path / "a.nd2", channel_info={"DAPl": {"role": "dna"}})
    dup = {**INFO, "PMP70": {"role": "mito"}}
    da = sio.load_image(tmp_path / "a.nd2", channel_info=dup)
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
    text = sio.describe_image(da)
    assert "DAPI" in text and "X=0.1" in text


def test_unsupported_suffix(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        sio.load_image(tmp_path / "x.czi")


def test_channels_by_unique_false_and_scalar(monkeypatch, tmp_path):
    _fake_nd2_module(monkeypatch)
    da = sio.load_image(tmp_path / "a.nd2", channel_info={**INFO, "PMP70": {"role": "mito", "color": "yellow"}})
    assert sio.channels_by(da, "role", "mito", unique=False) == ["TOMM20", "PMP70"]
    assert sio.channels_by(da, "color", ["yellow", "gray"], unique=False) == ["PMP70", "DAPI"]


def test_nd2_missing_package_message(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "nd2", None)  # import nd2 -> ImportError
    with pytest.raises(ImportError, match="eigenp_utils\\[nd2\\]"):
        sio.load_image(tmp_path / "x.nd2")
