"""LabelAnnotator: Python-side invariants (no browser needed; also runs in Pyodide)."""

import warnings

import numpy as np
import pytest

pytest.importorskip("anywidget")

from eigenp_utils.label_annotator import (  # noqa: E402
    LABELS_CMAP_COLORS,
    LabelAnnotator,
    _IMG_HDR,
    _LBL_HDR,
    _robust_limits,
    decode_label_slice,
    encode_label_slice,
)


def _img_header(w):
    return _IMG_HDR.unpack_from(bytes(w.image_to_js), 0)


def test_label_wire_roundtrip():
    """Verify ``encode_label_slice`` / ``decode_label_slice`` round-trip arbitrary uint16 labels.

    Strategy: encode a random ``(17, 23)`` ``uint32`` label slice with values in
    ``[0, 65534]`` at ``z=5`` and decode the resulting wire buffer.
    Expected outcome: decoded ``z`` equals 5, array dtype is ``uint16``, and
    every pixel matches the input exactly.
    """
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 65535, size=(17, 23)).astype(np.uint32)
    z, back = decode_label_slice(encode_label_slice(lab, 5))
    assert z == 5 and back.dtype == np.uint16
    np.testing.assert_array_equal(back, lab)


def test_label_wire_rejects_overflow():
    """Verify ``encode_label_slice`` rejects label ids exceeding the uint16 wire limit.

    Strategy: pass an array containing label id ``70_000`` (> 65535) to
    ``encode_label_slice``.
    Expected outcome: raises ``ValueError`` rather than silently truncating.
    """
    with pytest.raises(ValueError):
        encode_label_slice(np.full((2, 2), 70_000, dtype=np.uint32), 0)


def test_decode_label_slice_rejects_truncated_buffer():
    """Verify ``decode_label_slice`` raises ``ValueError`` on truncated label payloads.

    Strategy: encode a valid ``(4, 5)`` slice, strip bytes from the end of the
    body while keeping the 12-byte header intact, and pass it to
    ``decode_label_slice``.
    Expected outcome: raises ``ValueError`` citing the mismatched value count.
    """
    full = encode_label_slice(np.ones((4, 5), dtype=np.uint16), 1)
    assert len(full) == _LBL_HDR.size + 4 * 5 * 2
    truncated = full[:-4]
    with pytest.raises(ValueError, match="expected 20"):
        decode_label_slice(truncated)


@pytest.mark.parametrize(
    "axes, shape, spatial",
    [
        ("YX", (12, 9), (12, 9)),
        ("CYX", (3, 12, 9), (12, 9)),
        ("YXC", (12, 9, 3), (12, 9)),
        ("ZYX", (4, 12, 9), (4, 12, 9)),
        ("CZYX", (2, 4, 12, 9), (4, 12, 9)),
        ("ZCYX", (4, 2, 12, 9), (4, 12, 9)),
    ],
)
def test_axes_and_labels_shape(axes, shape, spatial):
    """Verify all supported axis permutations normalize to ``(C, Z, Y, X)`` and keep spatial label shape.

    Strategy: instantiate ``LabelAnnotator`` for 2D, 3D, and 4D arrays across
    all valid ``CZYX`` axis orderings.
    Expected outcome: ``w.labels`` has ``spatial`` shape with ``uint32`` dtype,
    the wire image header reports ``(c, y, x)``, and ``channel_names`` matches ``c``.
    """
    img = np.random.default_rng(1).random(shape).astype(np.float32)
    w = LabelAnnotator(img, axes=axes)
    assert w.labels.shape == spatial and w.labels.dtype == np.uint32
    z, c, y, x = _img_header(w)
    n_c = shape[axes.index("C")] if "C" in axes else 1
    assert (c, y, x) == (n_c, spatial[-2], spatial[-1])
    assert len(w.channel_names) == n_c


def test_axes_required_for_nd():
    """Verify ``LabelAnnotator`` requires ``axes=`` for NumPy arrays with ``ndim > 2``.

    Strategy: pass a 3D array without ``axes=``.
    Expected outcome: raises ``ValueError`` instead of guessing between ``CYX`` and ``ZYX``.
    """
    with pytest.raises(ValueError):
        LabelAnnotator(np.zeros((3, 8, 8)))


def test_image_slice_matches_selected_channel_data():
    """Channel order/transposition is preserved: the brightest channel stays brightest."""
    img = np.zeros((8, 10, 3), dtype=np.uint16)  # YXC
    img[..., 2] = np.arange(80).reshape(8, 10)  # only channel 2 has structure
    w = LabelAnnotator(img, axes="YXC")
    data = np.frombuffer(bytes(w.image_to_js), dtype=np.uint8, offset=_IMG_HDR.size).reshape(3, 8, 10)
    assert data[2].std() > 0 and data[0].std() == 0 and data[1].std() == 0
    # display scaling is monotone: brighter input never maps to darker output
    flat_in, flat_out = img[..., 2].ravel(), data[2].ravel()
    order = np.argsort(flat_in, kind="stable")
    assert np.all(np.diff(flat_out[order].astype(int)) >= 0)


def test_z_navigation_sends_requested_slice_and_its_labels():
    """Verify changing ``w.z`` ships the matching image and label slice and clamps out-of-range values.

    Strategy: construct a 5-slice volume with a known label at ``z=3``, navigate
    to ``z=3``, then attempt to set ``z=99``.
    Expected outcome: ``image_to_js`` and ``labels_to_js`` headers carry ``z=3``
    with the stored label intact, and ``z=99`` clamps to ``4``.
    """
    img = np.random.default_rng(2).random((5, 6, 7))
    lab = np.zeros((5, 6, 7), dtype=np.uint32)
    lab[3, 2, 2] = 9
    w = LabelAnnotator(img, axes="ZYX", labels=lab)
    w.z = 3
    assert _img_header(w)[0] == 3
    z, sl = decode_label_slice(w.labels_to_js)
    assert z == 3 and sl[2, 2] == 9
    w.z = 99  # clamped
    assert w.z == 4


def test_edits_from_frontend_update_volume_and_next_label():
    """Verify ``labels_from_js`` updates the target slice, advances instance id, and rejects bad shapes.

    Strategy: send a ``memoryview`` wire edit for ``z=2`` containing label 4,
    then send a shape-mismatched ``(2, 2)`` payload for ``z=0``.
    Expected outcome: slice 2 matches the edit, slices 0 and 1 remain zero,
    ``active_label`` advances to 5, and the malformed message is ignored with
    ``"ignored"`` recorded in ``w.status``.
    """
    img = np.random.default_rng(3).random((3, 6, 7))
    w = LabelAnnotator(img, axes="ZYX")
    assert w.active_label == 1
    edit = np.zeros((6, 7), dtype=np.uint16)
    edit[1:3, 1:3] = 4
    w.labels_from_js = memoryview(encode_label_slice(edit, 2))  # buffers may arrive as memoryview
    assert (w.labels[2] == edit).all() and w.labels[[0, 1]].max() == 0
    assert w.active_label == 5  # instance mode: next free id
    # malformed message is ignored, not applied
    w.labels_from_js = encode_label_slice(np.ones((2, 2), np.uint16), 0)
    assert w.labels[0].max() == 0 and "ignored" in w.status


def test_labels_from_js_is_race_safe_across_z_and_ignores_oob_z():
    """Verify edits apply to the slice encoded in their header even if ``w.z`` moved meanwhile.

    Strategy: navigate a 4-slice widget to ``w.z = 2``, deliver an edit whose
    wire header specifies ``z=0``, then deliver an out-of-range edit with
    ``z=99``.
    Expected outcome: the ``z=0`` edit writes exclusively to slice 0 (leaving
    slice 2 untouched), and ``z=99`` is ignored cleanly with ``"ignored"`` in
    ``w.status``.
    """
    img = np.random.default_rng(10).random((4, 6, 7)).astype(np.float32)
    w = LabelAnnotator(img, axes="ZYX")
    w.z = 2

    edit = np.zeros((6, 7), dtype=np.uint16)
    edit[1:4, 2:5] = 3
    w.labels_from_js = encode_label_slice(edit, 0)
    np.testing.assert_array_equal(w.labels[0], edit)
    assert w.labels[2].max() == 0

    w.labels_from_js = encode_label_slice(np.ones((6, 7), dtype=np.uint16), 99)
    assert "ignored" in w.status
    np.testing.assert_array_equal(w.labels[0], edit)
    assert w.labels[1:].max() == 0


def test_semantic_mode_does_not_auto_increment():
    """Verify semantic mode disables ``auto_fill`` and keeps ``active_label`` fixed after edits.

    Strategy: create a semantic annotator with 3 classes and apply an edit
    containing class id 2.
    Expected outcome: ``class_names`` and ``class_colors`` have length 3,
    ``auto_fill`` is ``False``, and ``active_label`` stays at 1 after the edit.
    """
    w = LabelAnnotator(np.random.default_rng(4).random((6, 7)), mode="semantic", classes=["mito", "perox", "cytosol"])
    assert w.class_names == ["mito", "perox", "cytosol"] and len(w.class_colors) == 3
    assert not w.auto_fill and w.active_label == 1
    edit = np.zeros((6, 7), dtype=np.uint16)
    edit[0, :3] = 2
    w.labels_from_js = encode_label_slice(edit, 0)
    assert w.active_label == 1 and (w.labels == edit).all()


def test_initial_labels_validation():
    """Verify initial ``labels=`` validates spatial shape, non-negativity, and uint16 max, and seeds ``active_label``.

    Strategy: pass mismatched shape, negative values, and values above 65535 to
    ``LabelAnnotator(..., labels=...)``, then pass a valid array with max id 3.
    Expected outcome: each invalid case raises ``ValueError``, while valid
    labels initialize ``active_label`` to ``4``.
    """
    img = np.zeros((6, 7))
    with pytest.raises(ValueError):
        LabelAnnotator(img, labels=np.zeros((5, 7)))
    with pytest.raises(ValueError):
        LabelAnnotator(img, labels=np.full((6, 7), -1, dtype=np.int32))
    with pytest.raises(ValueError):
        LabelAnnotator(img, labels=np.full((6, 7), 70_000))
    w = LabelAnnotator(img, labels=np.full((6, 7), 3))
    assert w.active_label == 4


def test_set_labels_updates_wire_state_active_label_and_validates():
    """Verify ``set_labels()`` updates internal labels, ``labels_to_js``, ``active_label``, and rejects bad inputs.

    Strategy: call ``set_labels`` on both an instance-mode 3D widget and a
    semantic-mode 2D widget, checking ``labels_to_js`` for the current ``z``,
    ``active_label`` progression vs preservation, and shape/range validation.
    Expected outcome: ``labels_to_js`` reflects the new slice, instance mode
    sets ``active_label = max + 1`` while semantic mode keeps ``active_label``
    unchanged, and invalid shapes, negative ids, or ids > 65535 raise ``ValueError``.
    """
    img3d = np.random.default_rng(11).random((3, 6, 7)).astype(np.float32)
    w_inst = LabelAnnotator(img3d, axes="ZYX", mode="instance")
    w_inst.z = 1

    new_lab = np.zeros((3, 6, 7), dtype=np.uint32)
    new_lab[1, 2:4, 3:5] = 8
    w_inst.set_labels(new_lab)
    np.testing.assert_array_equal(w_inst.labels, new_lab)
    z_wire, sl_wire = decode_label_slice(w_inst.labels_to_js)
    assert z_wire == 1
    np.testing.assert_array_equal(sl_wire, new_lab[1])
    assert w_inst.active_label == 9

    with pytest.raises(ValueError, match="expected labels of shape"):
        w_inst.set_labels(np.zeros((2, 6, 7), dtype=np.uint32))
    with pytest.raises(ValueError, match="labels must be in"):
        w_inst.set_labels(np.full((3, 6, 7), -1, dtype=np.int32))
    with pytest.raises(ValueError, match="labels must be in"):
        w_inst.set_labels(np.full((3, 6, 7), 70_000, dtype=np.uint32))

    w_sem = LabelAnnotator(np.zeros((6, 7), dtype=np.float32), mode="semantic", classes=["a", "b"])
    w_sem.active_label = 2
    sem_lab = np.ones((6, 7), dtype=np.uint32)
    w_sem.set_labels(sem_lab)
    assert w_sem.active_label == 2
    z_sem, sl_sem = decode_label_slice(w_sem.labels_to_js)
    assert z_sem == 0
    np.testing.assert_array_equal(sl_sem, sem_lab)


def test_annotated_slices_tracks_noncontiguous_and_cleared_slices():
    """Verify ``annotated_slices()`` reports sorted non-empty z indices and drops cleared slices.

    Strategy: inspect ``annotated_slices()`` on an empty 5-slice widget, after
    labelling slices 0 and 3, and after clearing slice 0 back to zeros.
    Expected outcome: returns ``[]``, then ``[0, 3]``, then ``[3]``.
    """
    w = LabelAnnotator(np.zeros((5, 6, 7), dtype=np.float32), axes="ZYX")
    assert w.annotated_slices() == []

    sl = np.zeros((6, 7), dtype=np.uint16)
    sl[2, 2] = 1
    w.labels_from_js = encode_label_slice(sl, 3)
    w.labels_from_js = encode_label_slice(sl, 0)
    assert w.annotated_slices() == [0, 3]

    w.labels_from_js = encode_label_slice(np.zeros((6, 7), dtype=np.uint16), 0)
    assert w.annotated_slices() == [3]


def test_labels_property_returns_defensive_copy():
    """Verify mutating ``w.labels`` does not alter the widget's internal label state.

    Strategy: read ``w.labels``, overwrite every entry in the returned array,
    and re-read ``w.labels``.
    Expected outcome: the widget's labels remain all zeros.
    """
    w = LabelAnnotator(np.zeros((3, 6, 7), dtype=np.float32), axes="ZYX")
    copy = w.labels
    copy[:] = 42
    assert w.labels.max() == 0


def test_new_label_trigger_sets_next_free_id():
    """Verify bumping ``new_label_trigger`` resets ``active_label`` to ``max(labels) + 1``.

    Strategy: paint label 5, manually lower ``active_label`` to 1, then
    increment ``new_label_trigger``.
    Expected outcome: ``active_label`` jumps to 6.
    """
    w = LabelAnnotator(np.zeros((6, 7), dtype=np.float32))
    sl = np.zeros((6, 7), dtype=np.uint16)
    sl[1, 1] = 5
    w.labels_from_js = encode_label_slice(sl, 0)
    w.active_label = 1
    w.new_label_trigger += 1
    assert w.active_label == 6


def test_save_writes_unscaled_pairs_for_annotated_slices(tmp_path):
    """Verify default ``save()`` writes unscaled image/mask TIFF pairs only for annotated slices.

    Strategy: annotate slice 1 of a 2-channel 3D ``uint16`` volume, save to
    ``tmp_path``, read back both TIFFs, and call ``save()`` a second time.
    Expected outcome: writes ``cellA_z001_img.tif`` (unscaled ``(C, Y, X)``)
    and ``cellA_z001_masks.tif``, and refuses to overwrite without ``overwrite=True``.
    """
    tifffile = pytest.importorskip("tifffile")
    img = np.random.default_rng(5).integers(0, 4000, size=(2, 3, 8, 9), dtype=np.uint16)  # CZYX
    w = LabelAnnotator(img, axes="CZYX")
    edit = np.zeros((8, 9), np.uint16)
    edit[2:4, 2:4] = 1
    w.labels_from_js = encode_label_slice(edit, 1)
    paths = w.save(tmp_path, "cellA")
    assert [p.name for p in paths] == ["cellA_z001_img.tif", "cellA_z001_masks.tif"]
    np.testing.assert_array_equal(tifffile.imread(paths[0]), img[:, 1])  # original data, not display-scaled
    np.testing.assert_array_equal(tifffile.imread(paths[1]), edit)
    with pytest.raises(FileExistsError):
        w.save(tmp_path, "cellA")


def test_save_modes_2d_stem_and_channel_shapes(tmp_path):
    """Verify ``save()`` supports ``slices='all'``/``'current'``, 2D stems, and single- vs multi-channel shapes.

    Strategy: (a) save a single-channel 3D volume with ``slices='all'`` and
    ``slices='current'``, checking filenames and ``(Y, X)`` TIFF shape; (b) save
    a 2D single-channel image and a 2D multi-channel image, checking that no
    ``_z000`` suffix is added and saved image shapes are ``(Y, X)`` vs ``(C, Y, X)``.
    Expected outcome: all filenames, slice selections, and saved array shapes match.
    """
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(12)

    vol = rng.integers(0, 500, size=(3, 8, 9), dtype=np.uint16)
    w3 = LabelAnnotator(vol, axes="ZYX")
    w3.z = 2

    all_paths = w3.save(tmp_path / "all", "stack", slices="all")
    assert [p.name for p in all_paths] == [
        "stack_z000_img.tif", "stack_z000_masks.tif",
        "stack_z001_img.tif", "stack_z001_masks.tif",
        "stack_z002_img.tif", "stack_z002_masks.tif",
    ]
    assert tifffile.imread(all_paths[0]).shape == (8, 9)

    cur_paths = w3.save(tmp_path / "cur", "stack", slices="current")
    assert [p.name for p in cur_paths] == ["stack_z002_img.tif", "stack_z002_masks.tif"]
    np.testing.assert_array_equal(tifffile.imread(cur_paths[0]), vol[2])

    img2d = rng.integers(0, 500, size=(8, 9), dtype=np.uint16)
    w2d = LabelAnnotator(img2d)
    p2d = w2d.save(tmp_path / "two_d", "plate", slices="current")
    assert [p.name for p in p2d] == ["plate_img.tif", "plate_masks.tif"]
    assert tifffile.imread(p2d[0]).shape == (8, 9)

    img2d_mc = rng.integers(0, 500, size=(3, 8, 9), dtype=np.uint16)
    w2d_mc = LabelAnnotator(img2d_mc, axes="CYX")
    p2d_mc = w2d_mc.save(tmp_path / "two_d_mc", "plate_mc", slices="current")
    assert [p.name for p in p2d_mc] == ["plate_mc_img.tif", "plate_mc_masks.tif"]
    assert tifffile.imread(p2d_mc[0]).shape == (3, 8, 9)


def test_input_validation_and_degenerate_channels():
    """Verify invalid constructor parameters raise ``ValueError`` and constant/all-NaN channels work safely.

    Strategy: pass invalid ``mode``, invalid ``axes`` strings, mismatched
    ``channel_names`` / ``channel_colors`` lengths, then construct widgets on a
    constant-valued channel (asserting zero ``RuntimeWarning``s) and an all-NaN
    channel (asserting ``_robust_limits`` is warning-free and widget construction
    succeeds without division-by-zero).
    Expected outcome: invalid options raise ``ValueError``; degenerate channels
    produce finite display limits ``hi > lo`` and valid wire buffers.
    """
    img2d = np.zeros((6, 7), dtype=np.float32)

    with pytest.raises(ValueError, match="mode"):
        LabelAnnotator(img2d, mode="panoptic")

    for bad_axes in ("YXZ", "TY", "YY", "CZ"):
        with pytest.raises(ValueError):
            LabelAnnotator(img2d, axes=bad_axes)

    with pytest.raises(ValueError, match="channel"):
        LabelAnnotator(img2d, channel_names=["a", "b"])
    with pytest.raises(ValueError, match="channel"):
        LabelAnnotator(img2d, channel_colors=["red", "blue"])

    const = np.full((6, 7), 5.0, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        w_const = LabelAnnotator(const)
    assert _img_header(w_const) == (0, 1, 6, 7)
    assert len(w_const.image_to_js) == _IMG_HDR.size + 6 * 7

    all_nan = np.full((6, 7), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _robust_limits(all_nan, 0.1, 99.99) == (0.0, 1.0)
    w_nan = LabelAnnotator(all_nan)
    assert _img_header(w_nan) == (0, 1, 6, 7)
    assert len(w_nan.image_to_js) == _IMG_HDR.size + 6 * 7


def test_xarray_stczyx_input():
    """Verify ``LabelAnnotator`` unpacks an STCZYX ``xarray.DataArray`` with singleton S and T.

    Strategy: convert a ``(2, 4, 10, 11)`` CZYX array via
    ``array_to_stczyx_xarray`` and pass the DataArray directly.
    Expected outcome: singleton S/T dimensions are squeezed and ``w.labels`` has
    spatial shape ``(4, 10, 11)``.
    """
    xr = pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray

    arr = np.random.default_rng(6).random((2, 4, 10, 11)).astype(np.float32)  # CZYX
    da = array_to_stczyx_xarray(arr, input_axes="CZYX")
    w = LabelAnnotator(da)
    assert isinstance(da, xr.DataArray) and w.labels.shape == (4, 10, 11)


def test_xarray_rejects_non_singleton_extra_dim():
    """Verify ``LabelAnnotator`` rejects DataArrays with a non-singleton extra dimension such as ``T=2``.

    Strategy: construct an STCZYX DataArray with ``T=2`` and pass it directly.
    Expected outcome: raises ``ValueError`` telling the caller to select a
    single index along ``T``.
    """
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray

    arr = np.zeros((1, 2, 1, 3, 6, 7), dtype=np.float32)
    da = array_to_stczyx_xarray(arr, input_axes="STCZYX")
    with pytest.raises(ValueError, match="dimension 'T' has size 2"):
        LabelAnnotator(da)


def test_instance_palette_is_labels_cmap():
    """Verify ``instance_palette`` mirrors ``LABELS_CMAP_COLORS`` in hex format.

    Strategy: instantiate ``LabelAnnotator`` on a 2D image and compare the
    length and first entry of ``instance_palette`` against ``LABELS_CMAP_COLORS``.
    Expected outcome: lengths match and entry 0 equals the rounded hex of
    ``LABELS_CMAP_COLORS[0]``.
    """
    w = LabelAnnotator(np.zeros((4, 4)))
    assert len(w.instance_palette) == len(LABELS_CMAP_COLORS)
    r, g, b = LABELS_CMAP_COLORS[0]
    assert w.instance_palette[0] == "#{:02x}{:02x}{:02x}".format(round(255 * r), round(255 * g), round(255 * b))


def test_xarray_uses_channel_coords():
    """Verify ``LabelAnnotator`` reads channel names and hex colours from xarray ``C`` coords.

    Strategy: attach ``color`` metadata to an STCZYX DataArray (including the
    colormap name ``'gray'``), reorder channels via ``.sel(C=...)``, and pass
    to ``LabelAnnotator`` with and without explicit ``channel_colors``.
    Expected outcome: inherits reordered channel names and converts ``'gray'``
    to ``'#ffffff'``; explicit ``channel_colors`` override the DataArray coord.
    """
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray, set_channel_info

    names = ["DAPI", "SOX2", "TBXT", "BF"]
    info = {
        "DAPI": {"color": "gray", "role": "dna"},
        "SOX2": {"color": "cyan", "role": "marker"},
        "TBXT": {"color": "yellow", "role": "marker"},
        "BF": {"color": "gray", "role": "brightfield"},
    }
    arr = np.arange(len(names) * 3 * 12 * 10, dtype=np.float32).reshape(len(names), 3, 12, 10)
    da = array_to_stczyx_xarray(arr, input_axes="CZYX", channel_names=names)
    da = set_channel_info(da, info).sel(C=["SOX2", "DAPI"])

    w = LabelAnnotator(da)
    assert w.channel_names == ["SOX2", "DAPI"]
    assert w.channel_colors == ["#00ffff", "#ffffff"]  # 'gray' colormap -> its end colour

    w2 = LabelAnnotator(da, channel_colors=["#123456", "red"])
    assert w2.channel_colors == ["#123456", "#ff0000"]


def test_save_refusal_writes_no_partial_files(tmp_path):
    """Verify ``LabelAnnotator.save`` checks all targets before writing anything.

    Strategy: annotate slice 0 and save; then also annotate slice 2 and call
    ``save`` again with ``overwrite=False``, followed by ``overwrite=True``.
    Expected outcome: ``overwrite=False`` raises ``FileExistsError`` without
    writing ``a_z002_img.tif``; ``overwrite=True`` writes all 4 files cleanly.
    """
    pytest.importorskip("tifffile")
    w = LabelAnnotator(np.random.default_rng(0).random((3, 6, 7)), axes="ZYX")
    edit = np.zeros((6, 7), np.uint16)
    edit[1, 1] = 1
    w.labels_from_js = encode_label_slice(edit, 0)
    w.save(tmp_path, "a")

    w.labels_from_js = encode_label_slice(edit, 2)
    with pytest.raises(FileExistsError):
        w.save(tmp_path, "a")
    assert not (tmp_path / "a_z002_img.tif").exists()
    assert len(w.save(tmp_path, "a", overwrite=True)) == 4
