"""Tests for ``eigenp_utils.image_io``: STCZYX conversion, pixel sizes, channel metadata, TIFF I/O, and ``load_image``."""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from eigenp_utils.image_io import (  # noqa: E402
    array_to_stczyx_xarray,
    channels_by,
    channels_where,
    get_stczyx_pixel_sizes,
    get_tiff_voxel_size,
    load_image,
    numpy_to_stczyx_xarray,
    set_channel_info,
)

_NAMES = ["DAPI", "WGA", "TOMM20", "PMP70"]
_INFO = {
    "DAPI": {"role": "dna", "color": "gray"},
    "WGA": {"role": "membrane", "color": "magenta"},
    "TOMM20": {"role": "mito", "color": "green"},
    "PMP70": {"role": "perox", "color": "yellow"},
}


def _reader_dataarray(nz=3, ny=16, nx=20, dz=2.0, dxy=0.108):
    """Build a vendor-reader-style DataArray (``P, Z, C, Y, X``) with physical coords."""
    data = np.random.default_rng(0).integers(0, 4096, size=(1, nz, len(_NAMES), ny, nx), dtype=np.uint16)
    da = xr.DataArray(
        data,
        dims=("P", "Z", "C", "Y", "X"),
        coords={
            "P": ["XYPos:0"],
            "Z": np.arange(nz) * dz,
            "C": _NAMES,
            "Y": np.arange(ny) * dxy,
            "X": np.arange(nx) * dxy,
        },
    )
    return data, da


def _calibrated_stczyx(nz=3, ny=12, nx=10, names=_NAMES, ps=None):
    """Build a calibrated STCZYX DataArray with distinct values per voxel."""
    arr = np.arange(len(names) * nz * ny * nx, dtype=np.float32).reshape(len(names), nz, ny, nx)
    ps = ps or {"Z": 2.0, "Y": 0.3, "X": 0.3, "unit": "µm"}
    return array_to_stczyx_xarray(arr, input_axes="CZYX", channel_names=names, pixel_sizes=ps)


def _imagej_stack(tmp_path, metadata, shape=(4, 8, 8), resolution=(2.0, 2.0), axes="ZYX"):
    """Write a real ImageJ TIFF with the given metadata and read its voxel sizes."""
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "ij.tif"
    tifffile.imwrite(
        path,
        np.zeros(shape, np.uint8),
        imagej=True,
        resolution=resolution,
        metadata={"axes": axes, **metadata},
    )
    return get_tiff_voxel_size(path)


# ---------------------------------------------------------------------------
# array_to_stczyx_xarray & get_stczyx_pixel_sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_axes, shape, expected_shape, perm_to_stczyx",
    [
        ("YX", (4, 5), (1, 1, 1, 1, 4, 5), None),
        ("ZCYX", (3, 2, 4, 5), (1, 1, 2, 3, 4, 5), (1, 0, 2, 3)),
        ("TCYX", (2, 3, 4, 5), (1, 2, 3, 1, 4, 5), (0, 1, 2, 3)),
        ("zyx", (3, 4, 5), (1, 1, 1, 3, 4, 5), None),
    ],
)
def test_array_to_stczyx_xarray_reorders_distinct_values_and_assigns_coords(
    input_axes, shape, expected_shape, perm_to_stczyx
):
    """Verify STCZYX expansion, axis transposition fidelity, default integer coords, and attrs.

    Strategy: feed distinct-valued arrays (including lowercase ``input_axes`` and
    plain Python lists) to ``array_to_stczyx_xarray``.
    Expected outcome: dims are strictly ``STCZYX``, shape matches expansion,
    every dimension has integer coords ``0..n-1``, ``attrs['axes'] == 'STCZYX'``,
    and voxel values follow their axes without transposition corruption.
    """
    arr = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    src = arr.tolist() if input_axes == "zyx" else arr
    da = array_to_stczyx_xarray(src, input_axes, attrs={"experiment": "e1"})

    assert da.dims == tuple("STCZYX")
    assert da.shape == expected_shape
    assert da.attrs == {"experiment": "e1", "axes": "STCZYX"}
    for dim in da.dims:
        np.testing.assert_array_equal(da.coords[dim].values, np.arange(da.sizes[dim]))

    squeezed = np.squeeze(da.values)
    expected_vals = arr.transpose(perm_to_stczyx) if perm_to_stczyx is not None else arr
    np.testing.assert_array_equal(squeezed, expected_vals)


def test_array_to_stczyx_xarray_idempotence():
    """Verify ``array_to_stczyx_xarray`` is idempotent on calibrated STCZYX DataArrays.

    Strategy: convert a 5D ``SCZYX`` array with exact binary-float pixel sizes,
    channel names, stage positions, custom attrs, and channel info, then run
    ``array_to_stczyx_xarray`` a second time on the output.
    Expected outcome: the second conversion is ``xr.testing.assert_identical`` to
    the first (preserving coordinates, units, stage positions, channel metadata,
    and attributes).
    """
    arr = np.arange(2 * 2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 2, 3, 4, 5)
    da = array_to_stczyx_xarray(
        arr,
        "SCZYX",
        channel_names=["DAPI", "GFP"],
        pixel_sizes={"Z": 2.0, "Y": 0.5, "X": 0.25, "unit": "µm"},
        stage_positions={"Y": [0.0, 100.0], "X": [0.0, 200.0]},
        attrs={"experiment": "idem"},
    )
    da = set_channel_info(da, {"DAPI": {"role": "dna"}, "GFP": {"role": "marker"}})

    xr.testing.assert_identical(array_to_stczyx_xarray(da), da)


def test_array_to_stczyx_xarray_does_not_modify_attrs():
    """Verify the caller's ``attrs`` dict is copied rather than mutated and ``axes`` is normalized.

    Strategy: pass a caller dict containing a stale ``axes`` entry and check
    both the returned DataArray and the original dict.
    Expected outcome: the caller's dict is unchanged; the output has
    ``axes == 'STCZYX'`` and retains the other metadata.
    """
    meta = {"source": "x.tif", "axes": "ZCYX"}
    da = array_to_stczyx_xarray(np.zeros((2, 1, 3, 3)), "ZCYX", attrs=meta)
    assert meta == {"source": "x.tif", "axes": "ZCYX"}
    assert da.attrs["axes"] == "STCZYX" and da.attrs["source"] == "x.tif"


def test_array_to_stczyx_xarray_from_dataarray_keeps_coords_and_attrs():
    """Verify a DataArray with physical coords keeps its coords, name, and attrs untouched.

    Strategy: pass a YX DataArray with calibrated float coords, a name, and
    existing attrs alongside a scalar string ``channel_names`` and extra attrs.
    Expected outcome: physical coords, name, and original attrs survive; new
    singleton dims receive integer coords; the input DataArray is not mutated.
    """
    y = np.arange(4) * 0.329
    x = np.arange(5) * 0.329
    src = xr.DataArray(
        np.arange(20, dtype=np.uint16).reshape(4, 5),
        dims=("Y", "X"),
        coords={"Y": y, "X": x},
        name="img",
        attrs={"metadata": "nd2-meta"},
    )
    da = array_to_stczyx_xarray(src, channel_names="BF", attrs={"experiment": "e1"})

    assert da.dims == tuple("STCZYX")
    assert da.shape == (1, 1, 1, 1, 4, 5)
    np.testing.assert_array_equal(da.coords["Y"].values, y)
    np.testing.assert_array_equal(da.coords["X"].values, x)
    assert da.coords["C"].values.tolist() == ["BF"]
    assert da.coords["S"].values.tolist() == [0]
    assert da.name == "img"
    assert da.attrs == {
        "metadata": "nd2-meta",
        "experiment": "e1",
        "axes": "STCZYX",
        "pixel_sizes": {"Z": None, "Y": pytest.approx(0.329), "X": pytest.approx(0.329), "unit": None},
    }
    np.testing.assert_array_equal(da.values[0, 0, 0, 0], src.values)
    assert src.dims == ("Y", "X") and src.attrs == {"metadata": "nd2-meta"}


def test_array_to_stczyx_xarray_dataarray_relabel_and_position_alias():
    """Verify ``P`` maps to ``S`` and positional relabelling handles permutations and custom dim names.

    Strategy: test (1) ``('P', 'C', 'Y', 'X')`` auto-aliasing ``P -> S``,
    (2) swapping ``('S', 'C', 'Y', 'X')`` to ``'CSYX'`` without name collision,
    and (3) relabelling multi-character dim ``('time', 'Y', 'X')`` to ``'TYX'``.
    Expected outcome: data values follow each relabelled axis accurately.
    """
    arr = np.random.default_rng(0).random((3, 2, 4, 5))
    src = xr.DataArray(arr, dims=("P", "C", "Y", "X"), coords={"C": ["DAPI", "GFP"]})
    da = array_to_stczyx_xarray(src)
    assert da.sizes["S"] == 3 and da.sizes["C"] == 2
    assert da.coords["C"].values.tolist() == ["DAPI", "GFP"]
    np.testing.assert_array_equal(da.values[:, 0, :, 0], arr)

    swapped = xr.DataArray(arr, dims=("S", "C", "Y", "X"))
    da2 = array_to_stczyx_xarray(swapped, input_axes="CSYX")
    assert da2.sizes["C"] == 3 and da2.sizes["S"] == 2
    np.testing.assert_array_equal(da2.values[:, 0, :, 0], arr.transpose(1, 0, 2, 3))

    named = xr.DataArray(np.arange(60).reshape(3, 4, 5), dims=("time", "Y", "X"))
    da3 = array_to_stczyx_xarray(named, input_axes="TYX")
    assert da3.sizes["T"] == 3
    np.testing.assert_array_equal(da3.values[0, :, 0, 0], named.values)


def test_array_to_stczyx_xarray_stays_lazy_with_dask():
    """Verify dask inputs (raw or wrapped in DataArray) stay lazy and match eager values.

    Strategy: wrap a small chunked dask array directly and inside an
    ``xr.DataArray``, convert both, and compare against eager NumPy conversion.
    Expected outcome: ``da.data`` remains a ``dask.array.Array`` and evaluates
    to identical values.
    """
    dask_array = pytest.importorskip("dask.array")
    arr = np.arange(2 * 6 * 7).reshape(2, 6, 7)
    lazy = dask_array.from_array(arr, chunks=(1, 6, 7))

    da = array_to_stczyx_xarray(lazy, "ZYX")
    assert isinstance(da.data, dask_array.Array)
    np.testing.assert_array_equal(da.values, array_to_stczyx_xarray(arr, "ZYX").values)

    da_x = array_to_stczyx_xarray(xr.DataArray(lazy, dims=("Z", "Y", "X")))
    assert isinstance(da_x.data, dask_array.Array)


@pytest.mark.parametrize(
    "image, kwargs, match",
    [
        (np.ones((10, 20)), {"input_axes": "ZYX"}, "Shape mismatch"),
        (np.ones((3, 10, 20)), {"input_axes": "CYX", "channel_names": ["C1", "C2"]}, "Provided 2 channel names"),
        (np.ones((3, 4, 5)), {"input_axes": "QYX"}, "allowed letters"),
        (np.ones((3, 4, 5)), {"input_axes": "YYX"}, "repeats"),
        (np.ones((3, 4, 5)), {}, "input_axes is required"),
        (xr.DataArray(np.ones((3, 4, 5)), dims=("time", "Y", "X")), {}, "Cannot map"),
        (np.ones((3, 4, 5)), {"input_axes": "ZYX", "pixel_sizes": (1.0, 2.0)}, "sequence"),
        (np.ones((3, 4, 5)), {"input_axes": "ZYX", "pixel_sizes": (1.0, 0.0, 1.0)}, "finite and positive"),
        (np.ones((3, 4, 5)), {"input_axes": "ZYX", "pixel_sizes": {"Y": -0.5}}, "finite and positive"),
        (np.ones((3, 4, 5)), {"input_axes": "ZYX", "pixel_sizes": {"X": np.nan}}, "finite and positive"),
        (np.ones((3, 4, 5)), {"input_axes": "ZYX", "pixel_sizes": {"Z": np.inf}}, "finite and positive"),
    ],
)
def test_array_to_stczyx_xarray_validation_errors(image, kwargs, match):
    """Verify ``array_to_stczyx_xarray`` rejects invalid axes, channel counts, and pixel sizes.

    Strategy: parametrize over invalid dimension counts, channel name lengths,
    unknown/repeated axis letters, missing ``input_axes``, unmappable DataArray
    dims, wrong-length pixel-size tuples, and non-positive/non-finite pixel sizes.
    Expected outcome: each case raises ``ValueError`` with an explanatory message.
    """
    with pytest.raises(ValueError, match=match):
        array_to_stczyx_xarray(image, **kwargs)


def test_numpy_to_stczyx_xarray_is_deprecated_alias():
    """Verify ``numpy_to_stczyx_xarray`` emits ``DeprecationWarning`` and matches the new API.

    Strategy: invoke both the deprecated alias and ``array_to_stczyx_xarray`` on
    identical inputs.
    Expected outcome: emits ``DeprecationWarning`` naming ``array_to_stczyx_xarray``
    and returns an identical ``DataArray``.
    """
    arr = np.arange(3 * 10 * 20).reshape(3, 10, 20)
    with pytest.warns(DeprecationWarning, match="array_to_stczyx_xarray"):
        old = numpy_to_stczyx_xarray(arr, "CYX", channel_names=["a", "b", "c"])
    new = array_to_stczyx_xarray(arr, "CYX", channel_names=["a", "b", "c"])
    xr.testing.assert_identical(old, new)


def test_array_to_stczyx_xarray_pixel_sizes_become_coords_and_attrs():
    """Verify explicit pixel sizes populate physical coords, coord units, and ``attrs['pixel_sizes']``.

    Strategy: pass pixel sizes as a case-insensitive dict with an uncalibrated
    ``None`` axis, as a ``(Z, Y, X)`` tuple, via ``attrs['pixel_sizes']``, and
    omitted altogether.
    Expected outcome: calibrated axes get float physical coordinates and ``units``
    attrs; uncalibrated axes stay integer indices; uncalibrated arrays omit
    ``attrs['pixel_sizes']``.
    """
    arr = np.zeros((3, 4, 5))
    da = array_to_stczyx_xarray(arr, "ZYX", pixel_sizes={"z": None, "Y": 0.5, "X": 0.25, "unit": "µm"})

    assert da.attrs["pixel_sizes"] == {"Z": None, "Y": 0.5, "X": 0.25, "unit": "µm"}
    np.testing.assert_allclose(da.coords["Y"].values, np.arange(4) * 0.5)
    np.testing.assert_allclose(da.coords["X"].values, np.arange(5) * 0.25)
    np.testing.assert_array_equal(da.coords["Z"].values, np.arange(3))
    assert da.coords["X"].attrs["units"] == "µm"
    assert "units" not in da.coords["Z"].attrs

    da_t = array_to_stczyx_xarray(arr, "ZYX", pixel_sizes=(2.0, 0.5, 0.5))
    assert da_t.attrs["pixel_sizes"] == {"Z": 2.0, "Y": 0.5, "X": 0.5, "unit": None}

    da_attr = array_to_stczyx_xarray(arr, "ZYX", attrs={"pixel_sizes": {"Z": 1.5, "Y": 0.5, "X": 0.5}})
    assert get_stczyx_pixel_sizes(da_attr) == {"Z": 1.5, "Y": 0.5, "X": 0.5, "unit": None}

    assert "pixel_sizes" not in array_to_stczyx_xarray(arr, "ZYX").attrs


def test_array_to_stczyx_xarray_pixel_sizes_conflicting_with_coords():
    """Verify explicit pixel sizes contradicting calibrated coords warn, win, and preserve origin.

    Strategy: construct a DataArray with float Y/X coords starting at ``Y=10.0``
    and pass a conflicting ``Y`` pixel size.
    Expected outcome: emits ``UserWarning``, rebuilds Y from origin ``10.0`` with
    the new step, and leaves the source DataArray untouched.
    """
    src = xr.DataArray(
        np.zeros((4, 5)),
        dims=("Y", "X"),
        coords={"Y": 10.0 + np.arange(4) * 0.3, "X": np.arange(5) * 0.3},
    )
    with pytest.warns(UserWarning, match="differs from the Y coordinate"):
        da = array_to_stczyx_xarray(src, pixel_sizes={"Y": 0.6, "X": 0.3, "unit": "um"})
    np.testing.assert_allclose(da.coords["Y"].values, 10.0 + np.arange(4) * 0.6)
    np.testing.assert_allclose(da.coords["X"].values, np.arange(5) * 0.3)
    assert da.attrs["pixel_sizes"] == {"Z": None, "Y": 0.6, "X": 0.3, "unit": "um"}
    np.testing.assert_allclose(src.coords["Y"].values, 10.0 + np.arange(4) * 0.3)
    assert src.coords["X"].attrs == {}


def test_get_stczyx_pixel_sizes_follows_downsampling_and_irregular_fallbacks():
    """Verify ``get_stczyx_pixel_sizes`` tracks slicing/coarsening and falls back for irregular coords.

    Strategy: inspect a calibrated STCZYX DataArray after (1) ``::2`` striding,
    (2) ``coarsen(Y=2, X=2).mean()``, (3) single-Z slicing, and (4) non-uniform,
    constant, or NaN float coordinates.
    Expected outcome: striding and ``coarsen`` report doubled Y/X steps; single-Z
    falls back to ``attrs['pixel_sizes']``; non-uniform/constant/NaN coords fall
    back to ``attrs['pixel_sizes']`` (or ``None`` when absent).
    """
    da = array_to_stczyx_xarray(
        np.zeros((3, 8, 8)),
        "ZYX",
        pixel_sizes={"Z": 2.0, "Y": 0.5, "X": 0.5, "unit": "µm"},
    )
    assert get_stczyx_pixel_sizes(da) == {"Z": 2.0, "Y": 0.5, "X": 0.5, "unit": "µm"}

    small = da[..., ::2, ::2]
    assert small.attrs["pixel_sizes"]["X"] == 0.5  # stale attrs snapshot
    assert get_stczyx_pixel_sizes(small) == {"Z": 2.0, "Y": 1.0, "X": 1.0, "unit": "µm"}
    assert array_to_stczyx_xarray(small).attrs["pixel_sizes"]["X"] == 1.0

    coarsened = da.coarsen(Y=2, X=2).mean()
    ps_coarse = get_stczyx_pixel_sizes(coarsened)
    assert ps_coarse["Z"] == pytest.approx(2.0)
    assert ps_coarse["Y"] == pytest.approx(1.0)
    assert ps_coarse["X"] == pytest.approx(1.0)

    assert get_stczyx_pixel_sizes(da.isel(Z=[0]))["Z"] == 2.0

    # Non-uniform, constant, and NaN float coordinates cannot yield a spacing -> fallback
    irreg = xr.DataArray(
        np.zeros((3, 3, 3)),
        dims=("Z", "Y", "X"),
        coords={
            "Z": [0.0, 1.0, 3.0],
            "Y": [1.0, 1.0, 1.0],
            "X": [0.0, np.nan, 2.0],
        },
        attrs={"pixel_sizes": {"Z": 4.0, "Y": None, "X": 0.25, "unit": "µm"}},
    )
    assert get_stczyx_pixel_sizes(irreg) == {"Z": 4.0, "Y": None, "X": 0.25, "unit": "µm"}


def test_array_to_stczyx_xarray_stage_positions_and_validation():
    """Verify stage positions become ``stage_*`` coords on ``S`` in all accepted forms and validate errors.

    Strategy: supply stage positions as dicts (with ``Y``/``X`` and ``stage_``
    prefixed keys), ``(S, 3)`` / ``(S, 2)`` arrays, a 1D array for ``S=1``, and
    existing ``P``-indexed coords (plus overriding them); also test invalid
    lengths, shapes, keys, and NaN entries.
    Expected outcome: valid inputs attach float coords on ``S`` that follow
    ``isel(S=...)`` and inherit physical units; invalid inputs raise ``ValueError``.
    """
    tiles = np.zeros((3, 4, 5))
    ys, xs = [0.0, 0.0, 900.0], [0.0, 1200.0, 0.0]
    da = array_to_stczyx_xarray(
        tiles,
        "SYX",
        pixel_sizes={"Y": 1.0, "X": 1.0, "unit": "µm"},
        stage_positions={"Y": ys, "X": xs},
    )
    assert da.coords["stage_y"].dims == ("S",)
    np.testing.assert_array_equal(da.coords["stage_x"].values, xs)
    np.testing.assert_array_equal(da.coords["stage_y"].values, ys)
    assert da.coords["stage_x"].attrs["units"] == "µm"
    assert "stage_z" not in da.coords
    np.testing.assert_array_equal(da.isel(S=[2]).coords["stage_y"].values, [900.0])

    # 'stage_' prefixed dict keys and (S, 3) / (S, 2) arrays
    da_pref = array_to_stczyx_xarray(tiles, "SYX", stage_positions={"stage_y": ys, "stage_x": xs})
    np.testing.assert_array_equal(da_pref.coords["stage_y"].values, ys)

    zyx = np.column_stack([[5.0, 5.0, 6.0], ys, xs])
    da3 = array_to_stczyx_xarray(tiles, "SYX", stage_positions=zyx)
    np.testing.assert_array_equal(da3.coords["stage_z"].values, [5.0, 5.0, 6.0])
    np.testing.assert_array_equal(da3.coords["stage_x"].values, xs)
    da2 = array_to_stczyx_xarray(tiles, "SYX", stage_positions=zyx[:, 1:])
    np.testing.assert_array_equal(da2.coords["stage_y"].values, ys)

    # 1D array accepted for a single scene (S == 1)
    da1 = array_to_stczyx_xarray(np.zeros((4, 5)), "YX", stage_positions=[10.0, 20.0])
    np.testing.assert_array_equal(da1.coords["stage_y"].values, [10.0])
    np.testing.assert_array_equal(da1.coords["stage_x"].values, [20.0])

    # Existing stage coords on 'P' move with relabel to 'S' and can be overridden
    src = xr.DataArray(
        np.zeros((2, 4, 5)),
        dims=("P", "Y", "X"),
        coords={"stage_x": ("P", [0.0, 1200.0]), "stage_y": ("P", [0.0, 0.0])},
    )
    da_kept = array_to_stczyx_xarray(src)
    assert da_kept.coords["stage_x"].dims == ("S",)
    np.testing.assert_array_equal(da_kept.coords["stage_x"].values, [0.0, 1200.0])
    da_over = array_to_stczyx_xarray(src, stage_positions={"Y": [1.0, 2.0], "X": [3.0, 4.0]})
    np.testing.assert_array_equal(da_over.coords["stage_x"].values, [3.0, 4.0])

    with pytest.raises(ValueError, match="S has size 3"):
        array_to_stczyx_xarray(tiles, "SYX", stage_positions={"Y": [0, 1], "X": [0, 1]})
    with pytest.raises(ValueError, match="shape"):
        array_to_stczyx_xarray(tiles, "SYX", stage_positions=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="keys"):
        array_to_stczyx_xarray(tiles, "SYX", stage_positions={"Q": [0, 1, 2]})
    with pytest.raises(ValueError, match="non-finite"):
        array_to_stczyx_xarray(tiles, "SYX", stage_positions={"Y": [0, np.nan, 1], "X": xs})


def test_array_to_stczyx_xarray_coordinate_edge_cases():
    """Verify subtle coordinate-rebuilding edge cases on DataArray inputs.

    Strategy: check (1) rebuilding descending float coords preserves direction
    and origin, (2) size-1 float coord preserves origin, (3) relabelling ``T``
    as ``Z`` drops the time coordinate while keeping ``T`` preserves it, and
    (4) integer coords with stored ``attrs['pixel_sizes']`` get physical coords
    so later striding updates ``get_stczyx_pixel_sizes``.
    Expected outcome: all four edge cases preserve physical meaning without
    silently misinterpreting stale coordinates.
    """
    desc = xr.DataArray(np.zeros((4, 3)), dims=("Y", "X"), coords={"Y": [1.5, 1.0, 0.5, 0.0]})
    with pytest.warns(UserWarning, match="differs from the Y coordinate"):
        da = array_to_stczyx_xarray(desc, pixel_sizes={"Y": 0.25})
    np.testing.assert_allclose(da.coords["Y"].values, [1.5, 1.25, 1.0, 0.75])

    one_z = xr.DataArray(np.zeros((1, 4, 3)), dims=("Z", "Y", "X"), coords={"Z": [3.5]})
    assert array_to_stczyx_xarray(one_z, pixel_sizes={"Z": 2.0}).coords["Z"].values.tolist() == [3.5]

    timed = xr.DataArray(np.zeros((3, 4, 5)), dims=("T", "Y", "X"), coords={"T": [0.0, 500.0, 1000.0]})
    as_z = array_to_stczyx_xarray(timed, input_axes="ZYX")
    assert "pixel_sizes" not in as_z.attrs
    np.testing.assert_array_equal(as_z.coords["Z"].values, [0, 1, 2])
    np.testing.assert_array_equal(array_to_stczyx_xarray(timed).coords["T"].values, [0.0, 500.0, 1000.0])

    stored = xr.DataArray(
        np.zeros((4, 8, 8)),
        dims=("Z", "Y", "X"),
        attrs={"pixel_sizes": {"Z": 2.0, "Y": 0.5, "X": 0.5, "unit": "µm"}},
    )
    da_s = array_to_stczyx_xarray(stored)
    np.testing.assert_allclose(da_s.coords["X"].values, np.arange(8) * 0.5)
    assert get_stczyx_pixel_sizes(da_s[..., ::2, ::2])["X"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_tiff_voxel_size
# ---------------------------------------------------------------------------


def test_get_tiff_voxel_size_imagej_variants(tmp_path):
    """Verify ImageJ TIFF voxel decoding on real files: asymmetric XY, spacing defaults, zunit, escapes.

    Strategy: write real ImageJ TIFFs exercising (1) explicit spacing +
    asymmetric resolution ``(2.0, 4.0)`` (catching X/Y swap and num/den inversion)
    with ``unit='micron'``, (2) multi-slice omitted spacing (defaults to 1.0),
    (3) single-slice omitted spacing (stays ``None``), (4) negative spacing
    (decoded as ``abs``), (5) ``zunit='nm'`` converted to ``µm``, and
    (6) ImageJ escaped ``\\u00B5m`` unit string.
    Expected outcome: all X, Y, Z sizes and harmonized units match ImageJ rules.
    """
    res = _imagej_stack(tmp_path, {"unit": "micron", "spacing": 0.5}, resolution=(2.0, 4.0))
    assert res == {"X": pytest.approx(0.5), "Y": pytest.approx(0.25), "Z": pytest.approx(0.5), "unit": "micron"}

    # ImageJ omits 'spacing' when the z-step is 1.0 on multi-slice stacks
    res_def = _imagej_stack(tmp_path, {"unit": "micron"})
    assert res_def == {"X": pytest.approx(0.5), "Y": pytest.approx(0.5), "Z": 1.0, "unit": "micron"}

    # Single-slice ImageJ image without spacing keeps Z=None
    res_2d = _imagej_stack(tmp_path, {"unit": "micron"}, shape=(8, 8), axes="YX")
    assert res_2d["Z"] is None and res_2d["X"] == pytest.approx(0.5)

    # Negative spacing is read as abs(spacing)
    res_neg = _imagej_stack(tmp_path, {"unit": "micron", "spacing": -2.0})
    assert res_neg["Z"] == pytest.approx(2.0)

    # Different zunit converts all axes to µm
    res_zu = _imagej_stack(tmp_path, {"unit": "micron", "spacing": 500.0, "zunit": "nm"})
    assert res_zu == {"X": pytest.approx(0.5), "Y": pytest.approx(0.5), "Z": pytest.approx(0.5), "unit": "µm"}

    # Escaped ImageJ micron spelling combines cleanly with nm zunit
    res_esc = _imagej_stack(tmp_path, {"unit": "\\u00B5m", "spacing": 1000.0, "zunit": "nm"})
    assert res_esc["Z"] == pytest.approx(1.0) and res_esc["unit"] == "µm"


def test_get_tiff_voxel_size_ome_and_standard_tags(tmp_path):
    """Verify real OME-TIFF (uniform & mixed units) and standard TIFF resolution-tag decoding.

    Strategy: write real TIFFs via ``tifffile.imwrite`` covering (1) OME-TIFF
    with uniform ``um`` physical sizes, (2) OME-TIFF with mixed ``nm``/``µm``
    units, and (3) standard TIFFs with ``CENTIMETER`` (3) and ``INCH`` (2)
    resolution tags and asymmetric X/Y resolutions.
    Expected outcome: OME physical sizes harmonize to ``µm`` when units differ;
    standard TIFF tags invert ``pixels/unit`` to ``units/pixel``, converted to
    µm for cm tags; inch-only tags (a print dpi) are uncalibrated (None) with a
    warning. (Changed by D01: these used to report raw 'cm' / 'inch' values.)
    """
    tifffile = pytest.importorskip("tifffile")

    p_ome = tmp_path / "uniform.ome.tif"
    tifffile.imwrite(
        p_ome,
        np.zeros((3, 8, 9), np.uint8),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 0.2,
            "PhysicalSizeY": 0.3,
            "PhysicalSizeZ": 0.5,
            "PhysicalSizeXUnit": "um",
            "PhysicalSizeYUnit": "um",
            "PhysicalSizeZUnit": "um",
        },
    )
    res_ome = get_tiff_voxel_size(p_ome)
    assert res_ome == {"X": pytest.approx(0.2), "Y": pytest.approx(0.3), "Z": pytest.approx(0.5), "unit": "um"}

    p_mix = tmp_path / "mixed.ome.tif"
    tifffile.imwrite(
        p_mix,
        np.zeros((3, 8, 9), np.uint8),
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": 200.0,
            "PhysicalSizeY": 250.0,
            "PhysicalSizeZ": 0.5,
            "PhysicalSizeXUnit": "nm",
            "PhysicalSizeYUnit": "nm",
            "PhysicalSizeZUnit": "µm",
        },
    )
    res_mix = get_tiff_voxel_size(p_mix)
    assert res_mix == {"X": pytest.approx(0.2), "Y": pytest.approx(0.25), "Z": pytest.approx(0.5), "unit": "µm"}

    # D01: lengths are reported in µm (0.005 cm = 50 µm), not in the tag's unit
    p_cm = tmp_path / "tags_cm.tif"
    tifffile.imwrite(p_cm, np.zeros((8, 9), np.uint8), resolution=(200.0, 100.0), resolutionunit=3)
    res_cm = get_tiff_voxel_size(p_cm)
    assert res_cm == {"X": pytest.approx(50.0), "Y": pytest.approx(100.0), "Z": None, "unit": "µm"}

    # D01: an inch resolution without ImageJ/OME units is a print dpi, i.e. uncalibrated
    p_in = tmp_path / "tags_in.tif"
    tifffile.imwrite(p_in, np.zeros((8, 9), np.uint8), resolution=(100.0, 50.0), resolutionunit=2)
    with pytest.warns(UserWarning, match="inches .* print resolution"):
        res_in = get_tiff_voxel_size(p_in)
    assert res_in == {"X": None, "Y": None, "Z": None, "unit": None}


@patch("eigenp_utils.image_io.tifffile.TiffFile")
def test_get_tiff_voxel_size_hybrid_and_unconvertible_units(mock_tifffile):
    """Verify OME-Z + standard-tag XY unit conversion and warning on unconvertible mixed units.

    Strategy: mock ``TiffFile`` for two hybrid metadata combinations not directly
    emitted by ``tifffile.imwrite``: (1) OME ``PhysicalSizeZ`` in ``µm`` with
    standard X/Y resolution tags in ``cm``; (2) X/Y in ``cm`` with unknown Z
    unit ``'frobs'``.
    Expected outcome: (1) converts all axes cleanly to ``µm``; (2) emits a
    ``UserWarning`` and keeps X's unit ``'cm'``.
    """
    res_tag = MagicMock(value=(10000, 1))  # 10000 px/cm -> 1e-4 cm = 1 µm
    unit_tag = MagicMock(value=3)

    mock_tif = MagicMock()
    mock_tif.is_imagej = False
    mock_tif.is_ome = True
    mock_tif.ome_metadata = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image>'
        '<Pixels PhysicalSizeZ="2.0" PhysicalSizeZUnit="µm" /></Image></OME>'
    )
    mock_tif.pages = [MagicMock()]
    mock_tif.pages[0].tags.get.side_effect = lambda k: {
        "XResolution": res_tag,
        "YResolution": res_tag,
        "ResolutionUnit": unit_tag,
    }.get(k)
    mock_tifffile.return_value.__enter__.return_value = mock_tif

    res = get_tiff_voxel_size("dummy.tif")
    assert res == {"X": pytest.approx(1.0), "Y": pytest.approx(1.0), "Z": pytest.approx(2.0), "unit": "µm"}

    mock_tif.ome_metadata = (
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image>'
        '<Pixels PhysicalSizeZ="2.0" PhysicalSizeZUnit="frobs" /></Image></OME>'
    )
    with pytest.warns(UserWarning, match="could not be converted"):
        res_bad = get_tiff_voxel_size("dummy.tif")
    assert res_bad["unit"] == "cm" and res_bad["Z"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Channel metadata: set_channel_info, channels_where, channels_by
# ---------------------------------------------------------------------------


def test_set_channel_info_follows_selection_reordering_and_complex_values():
    """Verify channel metadata travels with ``.sel(C=...)``, supports tuples, and matches int/str keys.

    Strategy: attach string, tuple (RGB), and integer-indexed channel metadata,
    then subset and reorder along ``C``.
    Expected outcome: auxiliary coordinates stay aligned with reordered channels,
    tuple values remain 1-D object entries, and string keys match integer labels.
    """
    da = set_channel_info(_calibrated_stczyx(), _INFO)
    assert list(da.coords["role"].values) == ["dna", "membrane", "mito", "perox"]

    sub = da.sel(C=["TOMM20", "DAPI"])
    assert list(sub.coords["color"].values) == ["green", "gray"]
    assert list(sub.coords["role"].values) == ["mito", "dna"]
    np.testing.assert_array_equal(sub.sel(C="DAPI").values, da.sel(C="DAPI").values)

    da_rgb = set_channel_info(_calibrated_stczyx(), {n: {"rgb": (i, 0, 0)} for i, n in enumerate(_NAMES)})
    assert da_rgb.coords["rgb"].values[2] == (2, 0, 0)

    # DataArray with 'C' dim but no 'C' coord auto-assigns 0..n-1 and accepts str key "0"
    bare = xr.DataArray(np.zeros((2, 4, 5)), dims=("C", "Y", "X"))
    annotated = set_channel_info(bare, {"0": {"role": "dna"}, 1: {"role": "membrane"}})
    assert annotated.coords["C"].values.tolist() == [0, 1]
    assert annotated.coords["role"].values.tolist() == ["dna", "membrane"]


def test_set_channel_info_validation_strictness_and_defaults():
    """Verify strict typo checking, ``strict=False``, missing-channel fills, and dimension validation.

    Strategy: probe (1) unknown channel typo with ``strict=True`` vs ``strict=False``,
    (2) missing channels filling string fields with ``""`` and numeric fields with
    ``NaN``, (3) field name colliding with a dimension name, and (4) DataArray
    without a ``'C'`` dimension.
    Expected outcome: strict mode raises ``KeyError``; non-strict skips unknown
    keys; missing entries warn and fill cleanly; dim clashes and missing ``C``
    raise ``ValueError``.
    """
    da = _calibrated_stczyx()
    with pytest.raises(KeyError, match="DAPl"):
        set_channel_info(da, {"DAPl": {"role": "dna"}})

    with pytest.warns(UserWarning, match="no channel info"):
        partial = set_channel_info(da, {"DAPI": {"role": "dna"}, "TYPO": {"role": "x"}}, strict=False)
    assert partial.coords["role"].values.tolist() == ["dna", "", "", ""]

    with pytest.warns(UserWarning):
        num = set_channel_info(da, {"DAPI": {"ex_nm": 405}})
    assert num.coords["ex_nm"].values[0] == 405 and np.isnan(num.coords["ex_nm"].values[1])

    with pytest.raises(ValueError, match="clashes"):
        set_channel_info(da, {"DAPI": {"Z": 1}})
    with pytest.raises(ValueError, match="no 'C' dimension"):
        set_channel_info(xr.DataArray(np.zeros((4, 5)), dims=("Y", "X")), {"0": {"role": "dna"}})


def test_channels_where_and_channels_by():
    """Verify ``channels_where`` filtering and ``channels_by`` ordered lookup with edge cases.

    Strategy: query by scalar, list (checking C-order vs requested-order), NumPy
    array, multiple AND criteria, callable predicate, empty criteria/values,
    non-unique duplicates (``unique=True`` vs ``False``), missing values
    (``allow_missing=False`` vs ``True``), and invalid/non-C coordinates.
    Expected outcome: ``channels_where`` preserves ``C`` order; ``channels_by``
    preserves requested order and enforces uniqueness/presence flags; invalid
    coordinates or missing ``C`` dim raise ``KeyError``/``ValueError``.
    """
    da = set_channel_info(_calibrated_stczyx(), _INFO)
    assert channels_where(da) == _NAMES
    assert channels_where(da, role="dna") == ["DAPI"]
    assert channels_where(da, role=["perox", "dna"]) == ["DAPI", "PMP70"]  # C order
    assert channels_where(da, color=np.array(["green", "magenta"])) == ["WGA", "TOMM20"]
    assert channels_where(da, role="mito", color="green") == ["TOMM20"]
    assert channels_where(da, color=lambda c: c != "gray") == ["WGA", "TOMM20", "PMP70"]

    with pytest.raises(KeyError):
        channels_where(da, organelle="mito")
    with pytest.raises(KeyError):
        channels_where(da, Y=0.0)
    with pytest.raises(ValueError, match="no 'C' dimension"):
        channels_where(xr.DataArray(np.zeros((4, 5)), dims=("Y", "X")), role="dna")

    assert channels_by(da, "role", ["membrane", "dna"]) == ["WGA", "DAPI"]
    assert channels_by(da, "role", "dna") == ["DAPI"]
    assert channels_by(da, "role", []) == []
    assert channels_by(da, "role", ["lamin", "dna"], allow_missing=True) == ["DAPI"]
    assert channels_by(da, "role", ["lamin"], allow_missing=True) == []
    with pytest.raises(KeyError, match="lamin"):
        channels_by(da, "role", ["lamin"])

    dup = set_channel_info(_calibrated_stczyx(), {**_INFO, "PMP70": {"role": "mito", "color": "yellow"}})
    with pytest.raises(ValueError, match="several channels"):
        channels_by(dup, "role", ["mito"])
    assert channels_by(dup, "role", "mito", unique=False) == ["TOMM20", "PMP70"]
    assert channels_by(dup, "color", ["yellow", "gray"], unique=False) == ["PMP70", "DAPI"]


# ---------------------------------------------------------------------------
# load_image (DataArrays, arrays, ImageJ/OME/planar/RGB/multi-series TIFFs)
# ---------------------------------------------------------------------------


def test_load_image_from_reader_dataarray_and_in_memory_array():
    """Verify ``load_image`` on reader-style DataArrays and plain NumPy arrays.

    Strategy: pass (1) a ``P, Z, C, Y, X`` uint16 DataArray with physical coords
    and ``channel_info``; (2) a float32 ``CYX`` NumPy array with explicit
    ``input_axes``, ``channel_names``, and ``pixel_sizes``.
    Expected outcome: both return strict STCZYX DataArrays preserving dtype,
    exact values, pixel sizes, and channel metadata, without setting
    ``attrs['source']`` (reserved for files).
    """
    data, src = _reader_dataarray()
    da = load_image(src, channel_info=_INFO)
    assert da.dims == ("S", "T", "C", "Z", "Y", "X")
    assert list(da.coords["C"].values) == _NAMES
    ps = get_stczyx_pixel_sizes(da)
    assert ps["Z"] == pytest.approx(2.0) and ps["X"] == pytest.approx(0.108)
    assert da.dtype == np.uint16
    assert "source" not in da.attrs
    np.testing.assert_array_equal(da.sel(C="TOMM20").values[0, 0], data[0, :, 2])
    assert channels_by(da, "role", ["membrane", "dna"]) == ["WGA", "DAPI"]

    arr = np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)
    da_arr = load_image(
        arr,
        input_axes="CYX",
        channel_names=["DAPI", "WGA"],
        channel_info={k: _INFO[k] for k in ("DAPI", "WGA")},
        pixel_sizes={"Y": 0.1, "X": 0.2, "unit": "µm"},
    )
    assert da_arr.shape == (1, 1, 2, 1, 5, 6)
    assert da_arr.dtype == np.float32
    assert "source" not in da_arr.attrs
    np.testing.assert_array_equal(da_arr.sel(C="WGA").values[0, 0, 0], arr[1])
    assert get_stczyx_pixel_sizes(da_arr) == {"Z": None, "Y": pytest.approx(0.1), "X": pytest.approx(0.2), "unit": "µm"}


def test_load_image_tiff_imagej_ome_and_uncalibrated(tmp_path):
    """Verify ``load_image`` round-trips ImageJ, OME-TIFF, and uncalibrated TIFFs accurately.

    Strategy: write (1) a ``ZCYX`` ImageJ TIFF with resolution & spacing,
    (2) an OME-TIFF with ``PhysicalSizeX/Y/Z``, and (3) a plain uncalibrated TIFF
    with no resolution metadata; load each via ``load_image``.
    Expected outcome: ImageJ and OME TIFFs preserve data, voxel sizes, and
    ``attrs['source']``; an uncalibrated TIFF's default 1.0/1.0 tags map to
    ``X=None, Y=None`` rather than fake 1.0 physical sizes.
    """
    tifffile = pytest.importorskip("tifffile")

    arr = np.random.default_rng(1).integers(0, 1000, size=(3, 2, 8, 9), dtype=np.uint16)
    p_ij = tmp_path / "img.tif"
    tifffile.imwrite(
        p_ij,
        arr,
        imagej=True,
        resolution=(1 / 0.2, 1 / 0.2),
        metadata={"axes": "ZCYX", "spacing": 1.5, "unit": "um"},
    )
    da_ij = load_image(p_ij, channel_names=["DAPI", "WGA"], channel_info={k: _INFO[k] for k in ("DAPI", "WGA")})
    assert da_ij.dims == ("S", "T", "C", "Z", "Y", "X") and da_ij.sizes["Z"] == 3
    assert da_ij.attrs["source"] == str(p_ij)
    np.testing.assert_array_equal(da_ij.sel(C="WGA").values[0, 0], arr[:, 1])
    ps_ij = get_stczyx_pixel_sizes(da_ij)
    assert ps_ij["Z"] == pytest.approx(1.5) and ps_ij["X"] == pytest.approx(0.2)
    assert channels_by(da_ij, "role", "dna") == ["DAPI"]

    p_ome = tmp_path / "vol.ome.tif"
    vol = np.arange(3 * 6 * 7, dtype=np.uint16).reshape(3, 6, 7)
    tifffile.imwrite(
        p_ome,
        vol,
        ome=True,
        metadata={"axes": "ZYX", "PhysicalSizeX": 0.25, "PhysicalSizeY": 0.25, "PhysicalSizeZ": 1.0},
    )
    da_ome = load_image(p_ome)
    np.testing.assert_array_equal(da_ome.values[0, 0, 0], vol)
    assert get_stczyx_pixel_sizes(da_ome) == {"Z": pytest.approx(1.0), "Y": pytest.approx(0.25), "X": pytest.approx(0.25), "unit": "µm"}

    # Uncalibrated TIFF without resolution unit: default 1.0/1.0 tags must become None
    p_uncal = tmp_path / "uncal.tif"
    tifffile.imwrite(p_uncal, np.zeros((6, 7), np.uint8))
    da_uncal = load_image(p_uncal)
    assert get_stczyx_pixel_sizes(da_uncal) == {"Z": None, "Y": None, "X": None, "unit": None}


def test_load_image_tiff_samples_axis_and_integer_channel_labels(tmp_path):
    """Verify tifffile's samples axis ``S`` maps to ``C`` and integer labels work in ``sel``.

    Strategy: save planar (``SYX``), RGB (``YXS``), and multi-channel ``ZCYX``
    TIFFs without explicit ``channel_names``.
    Expected outcome: planar and RGB samples land on ``C`` (not ``S``) with exact
    pixel values; integer channel labels work natively with ``channels_by`` and
    ``da.sel(C=...)``.
    """
    tifffile = pytest.importorskip("tifffile")

    arr = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
    p = tmp_path / "planar.tif"
    tifffile.imwrite(p, arr)  # default: planar 'SYX'
    with tifffile.TiffFile(p) as tif:
        assert "S" in tif.series[0].axes
    da = load_image(p)
    assert da.sizes["S"] == 1 and da.sizes["C"] == 3
    np.testing.assert_array_equal(da.isel(S=0, T=0, Z=0).values, arr)

    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[..., 1] = 7
    p2 = tmp_path / "rgb.tif"
    tifffile.imwrite(p2, rgb, photometric="rgb")
    da2 = load_image(p2)
    assert da2.sizes["C"] == 3 and da2.sizes["S"] == 1
    assert int(da2.isel(S=0, T=0, Z=0, C=1).values.max()) == 7

    p3 = tmp_path / "czyx.tif"
    tifffile.imwrite(p3, np.arange(2 * 3 * 8 * 9, dtype=np.uint16).reshape(3, 2, 8, 9), imagej=True, metadata={"axes": "ZCYX"})
    da3 = load_image(p3, channel_info={0: {"role": "dna"}, 1: {"role": "membrane"}})
    names = channels_by(da3, "role", ["membrane", "dna"])
    assert names == [1, 0]
    assert da3.sel(C=names).sizes["C"] == 2


def test_load_image_multiseries_homogeneous_and_heterogeneous(tmp_path):
    """Verify multi-series TIFF stacking for matching shapes and warning/selection for differing shapes.

    Strategy: write (1) a 2-series OME-TIFF with identical shapes and (2) a
    2-series OME-TIFF with different shapes; load with ``series=None`` and
    ``series=1``.
    Expected outcome: matching series stack along ``S``; differing series emit a
    ``UserWarning`` and load series 0 by default, while ``series=1`` selects the
    second series cleanly.
    """
    tifffile = pytest.importorskip("tifffile")

    p_same = tmp_path / "multi_same.ome.tif"
    a0 = np.zeros((2, 8, 9), np.uint16)
    a1 = np.ones((2, 8, 9), np.uint16)
    with tifffile.TiffWriter(p_same, ome=True) as w:
        w.write(a0, metadata={"axes": "CYX"})
        w.write(a1, metadata={"axes": "CYX"})
    da = load_image(p_same)
    assert da.sizes["S"] == 2 and da.sizes["C"] == 2
    assert int(da.isel(S=0).values.max()) == 0 and int(da.isel(S=1).values.max()) == 1
    da1 = load_image(p_same, series=1)
    assert da1.sizes["S"] == 1 and int(da1.values.max()) == 1

    p_diff = tmp_path / "multi_diff.ome.tif"
    b0 = np.full((6, 7), 3, np.uint16)
    b1 = np.full((10, 11), 9, np.uint16)
    with tifffile.TiffWriter(p_diff, ome=True) as w:
        w.write(b0, metadata={"axes": "YX"})
        w.write(b1, metadata={"axes": "YX"})
    with pytest.warns(UserWarning, match="different shapes"):
        da_d0 = load_image(p_diff)
    assert da_d0.shape == (1, 1, 1, 1, 6, 7) and int(da_d0.values.max()) == 3
    da_d1 = load_image(p_diff, series=1)
    assert da_d1.shape == (1, 1, 1, 1, 10, 11) and int(da_d1.values.max()) == 9


def test_load_image_pixel_size_override_and_channel_rename_errors(tmp_path):
    """Verify per-axis ``pixel_sizes`` overrides, rename/``channel_info`` guard, and format/axis errors.

    Strategy: (1) override only ``Z`` on an ImageJ TIFF via dict and tuple;
    (2) rename channels while passing old names in ``channel_info`` (strict vs
    non-strict); (3) pass an unsupported file suffix; (4) simulate TIFF axes with
    both ``S`` and ``C`` or unknown axis ``Q``.
    Expected outcome: per-axis overrides preserve untouched axes; old names under
    ``strict=True`` raise a guiding ``KeyError``; unsupported formats and
    ambiguous TIFF axes raise ``ValueError`` explaining how to proceed.
    """
    tifffile = pytest.importorskip("tifffile")

    p = tmp_path / "ij.tif"
    tifffile.imwrite(p, np.zeros((4, 8, 9), np.uint16), imagej=True, resolution=(5, 5), metadata={"axes": "ZYX", "unit": "um"})
    ps = get_stczyx_pixel_sizes(load_image(p, pixel_sizes={"Z": 0.5}))
    assert ps["Z"] == pytest.approx(0.5) and ps["X"] == pytest.approx(0.2) and ps["Y"] == pytest.approx(0.2)
    ps2 = get_stczyx_pixel_sizes(load_image(p, pixel_sizes=(0.7, None, None)))
    assert ps2["Z"] == pytest.approx(0.7) and ps2["X"] == pytest.approx(0.2)

    src = xr.DataArray(np.zeros((2, 8, 9)), dims=("C", "Y", "X"), coords={"C": ["DAPI", "WGA"]})
    with pytest.raises(KeyError, match="key channel_info by the new names"):
        load_image(src, channel_names=["dna_ch", "mem_ch"], channel_info={"DAPI": {"role": "dna"}, "WGA": {"role": "membrane"}})
    da_ren = load_image(src, channel_names=["dna_ch", "mem_ch"], channel_info={"dna_ch": {"role": "dna"}, "mem_ch": {"role": "membrane"}})
    assert channels_by(da_ren, "role", "dna") == ["dna_ch"]
    with pytest.warns(UserWarning, match="no channel info"):
        da_ns = load_image(src, channel_names=["dna_ch", "mem_ch"], channel_info={"DAPI": {"role": "dna"}}, strict=False)
    assert list(da_ns.coords["C"].values) == ["dna_ch", "mem_ch"]

    with pytest.raises(ValueError, match="own reader"):
        load_image(tmp_path / "x.czi")

    # Ambiguous TIFF axes ('SCYX' and unknown 'QYX') require explicit input_axes=
    for bad_axes, msg in [("SCYX", "both samples"), ("QYX", "outside STCZYX")]:
        series = MagicMock(axes=bad_axes, shape=(2, 2, 4, 5) if len(bad_axes) == 4 else (2, 4, 5), dtype=np.uint8)
        series.asarray.return_value = np.zeros(series.shape, np.uint8)
        mock_tif = MagicMock(series=[series])
        with patch("eigenp_utils.image_io.tifffile.TiffFile") as mock_cls:
            mock_cls.return_value.__enter__.return_value = mock_tif
            with pytest.raises(ValueError, match=msg):
                load_image(tmp_path / "ambig.tif")


# ---------------------------------------------------------------------------
# D01: readers report µm; uncalibrated is None
# ---------------------------------------------------------------------------


def test_cm_and_nm_tiffs_load_in_um_and_overrides_merge_in_um(tmp_path):
    """Verify cm / nm calibrated TIFFs round-trip to µm, and per-axis overrides are merged in µm (C-01, C-02, R8-02).

    Strategy: (+) write a 1 µm/px stack as cm resolution tags (no Z step) and
    an ImageJ stack in nm; read with ``get_tiff_voxel_size`` and
    ``load_image`` (with a Z override without unit, and with an override in
    nm). (-) a µm ImageJ stack keeps its values.
    Expected outcome: X/Y = 1 µm with unit 'µm' (not 1e-4 'cm'); the unitless
    Z override is read as µm next to them; an nm override is converted; the nm
    ImageJ stack loads as 0.1 / 0.3 µm; the µm control is unchanged.
    """
    tifffile = pytest.importorskip("tifffile")
    p_cm = tmp_path / "cm.tif"
    tifffile.imwrite(p_cm, np.zeros((5, 16, 16), np.uint16), resolution=(1e4, 1e4), resolutionunit="CENTIMETER")
    assert get_tiff_voxel_size(p_cm) == {"X": pytest.approx(1.0), "Y": pytest.approx(1.0), "Z": None, "unit": "µm"}
    ps = get_stczyx_pixel_sizes(load_image(p_cm, input_axes="ZYX", pixel_sizes={"Z": 2.0}))
    assert ps == {"Z": pytest.approx(2.0), "Y": pytest.approx(1.0), "X": pytest.approx(1.0), "unit": "µm"}
    ps_nm = get_stczyx_pixel_sizes(load_image(p_cm, input_axes="ZYX", pixel_sizes={"Z": 500.0, "unit": "nm"}))
    assert ps_nm["Z"] == pytest.approx(0.5) and ps_nm["X"] == pytest.approx(1.0) and ps_nm["unit"] == "µm"

    p_nm = tmp_path / "nm.tif"
    tifffile.imwrite(p_nm, np.zeros((4, 8, 8), np.uint16), imagej=True, resolution=(1 / 100.0, 1 / 100.0),
                     metadata={"axes": "ZYX", "unit": "nm", "spacing": 300.0})
    ps = get_stczyx_pixel_sizes(load_image(p_nm))
    assert ps == {"Z": pytest.approx(0.3), "Y": pytest.approx(0.1), "X": pytest.approx(0.1), "unit": "µm"}

    p_um = tmp_path / "um.tif"
    tifffile.imwrite(p_um, np.zeros((4, 8, 8), np.uint16), imagej=True, resolution=(2.0, 2.0),
                     metadata={"axes": "ZYX", "unit": "micron", "spacing": 1.5})
    assert get_tiff_voxel_size(p_um) == {"X": 0.5, "Y": 0.5, "Z": 1.5, "unit": "micron"}


def test_uncalibrated_and_dpi_only_tiffs_report_none(tmp_path):
    """Verify uncalibrated (1/1, no unit) and inch-only TIFFs give X = Y = None from both readers (R8-01).

    Strategy: write a TIFF with tifffile's default resolution, and a 72-dpi
    'screenshot' (inch tags); read both with ``get_tiff_voxel_size`` and
    ``load_image``.
    Expected outcome: both readers agree on None (no fake 1.0 or 352.8 µm
    pixels); the dpi file warns that the inch resolution is a print dpi.
    """
    tifffile = pytest.importorskip("tifffile")
    p = tmp_path / "plain.tif"
    tifffile.imwrite(p, np.zeros((6, 7), np.uint8))
    assert get_tiff_voxel_size(p) == {"X": None, "Y": None, "Z": None, "unit": None}

    p72 = tmp_path / "shot.tif"
    tifffile.imwrite(p72, np.zeros((64, 64), np.uint8), resolution=(72.0, 72.0), resolutionunit="INCH")
    with pytest.warns(UserWarning, match="72 dpi"):
        assert get_tiff_voxel_size(p72)["X"] is None
    with pytest.warns(UserWarning, match="72 dpi"):
        assert get_stczyx_pixel_sizes(load_image(p72)) == {"Z": None, "Y": None, "X": None, "unit": None}


def test_imagej_inch_and_pixel_units_report_none(tmp_path):
    """Verify ImageJ TIFFs whose unit is 'inch' or 'pixel' give unknown sizes, not calibrations.

    Fiji writes unit 'inch' when it re-saves a dpi-only screenshot, and
    'pixel' for an uncalibrated stack. Strategy: write both as ImageJ ZYX
    stacks with a spacing; read them with ``get_tiff_voxel_size`` and merge a
    µm Z override through ``load_image``.
    Expected outcome: 'inch' warns (print dpi) and gives X = Y = Z = None
    (it used to give 352.8 µm pixels silently); 'pixel' gives None without a
    warning, and a Z override merges (it used to raise ValueError).
    Negative control: the same file with unit 'micron' keeps its sizes.
    """
    tifffile = pytest.importorskip("tifffile")
    vol = np.zeros((3, 16, 16), np.uint8)
    p_in = tmp_path / "inch.tif"
    tifffile.imwrite(p_in, vol, imagej=True, resolution=(72, 72), resolutionunit=2,
                     metadata={"axes": "ZYX", "spacing": 2.0, "unit": "inch"})
    with pytest.warns(UserWarning, match="inches"):
        assert get_tiff_voxel_size(p_in) == {"X": None, "Y": None, "Z": None, "unit": None}

    p_px = tmp_path / "pixel.tif"
    tifffile.imwrite(p_px, vol, imagej=True, metadata={"axes": "ZYX", "spacing": 1.0, "unit": "pixel"})
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        assert get_tiff_voxel_size(p_px) == {"X": None, "Y": None, "Z": None, "unit": None}
        ps = get_stczyx_pixel_sizes(load_image(p_px, pixel_sizes={"Z": 2.0}))
    assert ps["Z"] == 2.0 and ps["X"] is None and ps["Y"] is None

    p_um = tmp_path / "micron.tif"
    tifffile.imwrite(p_um, vol, imagej=True, resolution=(2.0, 2.0),
                     metadata={"axes": "ZYX", "spacing": 1.5, "unit": "micron"})
    assert get_tiff_voxel_size(p_um) == {"X": 0.5, "Y": 0.5, "Z": 1.5, "unit": "micron"}
