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


def test_display_range_and_palette_follow_the_shared_conventions():
    """Verify LabelAnnotator uses the package display range and palette (D02, D03; R4-07, R8-20).

    Strategy: (1) sparse beads (0.002% of voxels) on zeros; (2) a signed
    channel; (3) explicit ``percentiles=(0.1, 99.99)`` on the sparse beads;
    (4) nine channels without colours, and two given colours for three
    channels including an RGB tuple.
    Expected outcome: (1) limits (0, bead max) as in the ortho viewers (the
    old p0.1/p99.99 default gave (0, 1): every bead saturated); (2) vmin is
    the data minimum; (3) explicit percentiles fall back to the maximum with
    a warning instead of lo + 1; (4) the palette cycles (white, lime,
    magenta, ...; it raised above 8 channels) and short lists are padded
    with unused palette colours.
    """
    from eigenp_utils._conventions import default_display_range

    beads = np.zeros((10, 100, 100), np.uint16)
    beads[5, 10, 10] = 3000
    beads[2, 50, 70] = 1200
    w = LabelAnnotator(beads, axes="ZYX")
    assert w._limits[0] == (0.0, 3000.0) == default_display_range(beads)

    signed = np.random.default_rng(0).normal(0, 1, (4, 20, 20)).astype(np.float32)
    assert LabelAnnotator(signed, axes="ZYX")._limits[0][0] == pytest.approx(float(signed.min()))

    with pytest.warns(UserWarning, match="percentiles"):
        w_pct = LabelAnnotator(beads, axes="ZYX", percentiles=(0.1, 99.99))
    assert w_pct._limits[0] == (0.0, 3000.0)

    w9 = LabelAnnotator(np.random.default_rng(1).random((9, 8, 8)), axes="CYX")
    assert w9.channel_colors[:3] == ["#ffffff", "#00ff00", "#ff00ff"]
    assert w9.channel_colors[7:] == w9.channel_colors[:2]
    w3 = LabelAnnotator(np.zeros((3, 8, 8)), axes="CYX", channel_colors=["lime", (1.0, 0.0, 0.0)])
    assert w3.channel_colors == ["#00ff00", "#ff0000", "#ffffff"]


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


# ---------------------------------------------------------------- z-linking
def _two_tracks():
    """6-slice volume: id 3 on z 0-3 (left), id 5 on z 2-5 (right), id 7 on z 4-5 (bottom left)."""
    lab = np.zeros((6, 10, 12), dtype=np.uint32)
    lab[0:4, 1:4, 1:4] = 3
    lab[2:6, 1:4, 7:10] = 5
    lab[4:6, 6:9, 1:4] = 7
    img = np.random.default_rng(7).random(lab.shape).astype(np.float32)
    return LabelAnnotator(img, axes="ZYX", labels=lab), lab


def _decode_or_none(buf):
    return decode_label_slice(buf) if len(bytes(buf)) else None


def test_merge_ids_relabels_every_slice_and_undo_restores():
    """Merging relabels the source id in all slices only; undo restores the exact volume.

    Strategy: merge id 5 into id 3 (``merge_ids(src=5, dst=3)``), then undo.
    Expected outcome: no voxel keeps id 5, every former id-5 voxel is 3, other
    voxels are unchanged, the extent of 3 becomes z 0-5, ``track_undo_ids`` has
    one entry, and ``undo_track_op`` restores the input bit-exactly.
    """
    w, lab = _two_tracks()
    assert w.merge_ids(5, 3)
    out = w.labels
    assert not (out == 5).any()
    np.testing.assert_array_equal(out[lab == 5], 3)
    np.testing.assert_array_equal(out[lab != 5], lab[lab != 5])
    assert w.label_extent(3) == [0, 1, 2, 3, 4, 5]
    assert "both on z 2, 3" in w.status  # 3 and 5 share slices 2-3: flagged
    assert len(w.track_undo_ids) == 1
    assert w.undo_track_op()
    np.testing.assert_array_equal(w.labels, lab)
    assert w.track_undo_ids == [] and not w.undo_track_op()


def test_merge_ids_refusals_leave_volume_unchanged():
    """Merges with background, identical ids or absent ids are refused without side effects."""
    w, lab = _two_tracks()
    for src, dst in [(5, 0), (0, 5), (5, 5), (42, 3)]:
        assert not w.merge_ids(src, dst)
        assert "refused" in w.status
    np.testing.assert_array_equal(w.labels, lab)
    assert w.track_undo_ids == []


def test_split_id_gives_new_id_from_z_and_undo():
    """Splitting id 3 at z=2 moves its voxels at z >= 2 to a fresh id ``max + 1``.

    Strategy: ``split_id(3, 2)`` on the two-track phantom, then undo.
    Expected outcome: returns 8 (max 7 + 1); id 3 keeps z 0-1, id 8 covers
    z 2-3 with exactly the former id-3 voxels there; other ids untouched;
    splitting at the first slice (pure rename) is refused; undo restores the input.
    """
    w, lab = _two_tracks()
    assert w.split_id(3, 0) is None and "refused" in w.status  # nothing below z=0
    assert w.split_id(3, 4) is None  # nothing at/after z=4
    new = w.split_id(3, 2)
    assert new == 8
    out = w.labels
    assert w.label_extent(3) == [0, 1] and w.label_extent(8) == [2, 3]
    np.testing.assert_array_equal(out[2:][lab[2:] == 3], 8)
    np.testing.assert_array_equal(out[:2], lab[:2])
    np.testing.assert_array_equal(out[lab != 3], lab[lab != 3])
    assert w.active_label != new, "the active id must not silently join the new track"
    assert w.undo_track_op()
    np.testing.assert_array_equal(w.labels, lab)


def test_track_ops_from_frontend_requests_and_undo_routing():
    """``track_request`` messages from JS run the same operations; undo needs the matching serial."""
    w, lab = _two_tracks()
    w.z = 3
    w.track_request = {"op": "merge", "src": 5, "dst": 3, "n": 1}
    assert not (w.labels == 5).any()
    # the current slice and its neighbours are re-sent after a volume edit
    z, cur = decode_label_slice(w.labels_to_js)
    assert z == 3 and (cur[lab[3] == 5] == 3).all()
    w.track_request = {"op": "split", "id": 3, "z": 4, "n": 2}
    assert w.label_extent(3) == [0, 1, 2, 3]
    assert len(w.track_undo_ids) == 2
    stale = w.track_undo_ids[0]
    w.track_request = {"op": "undo", "serial": stale, "n": 3}  # not the last op: refused
    assert "out of sync" in w.status and len(w.track_undo_ids) == 2
    w.track_request = {"op": "undo", "serial": w.track_undo_ids[-1], "n": 4}
    w.track_request = {"op": "undo", "serial": w.track_undo_ids[-1], "n": 5}
    np.testing.assert_array_equal(w.labels, lab)


def test_track_undo_stack_is_bounded():
    """Only the last ``_TRACK_UNDO_MAX`` track operations stay undoable."""
    from eigenp_utils.label_annotator import _TRACK_UNDO_MAX

    w, _ = _two_tracks()
    for _ in range(_TRACK_UNDO_MAX + 5):
        assert w.merge_ids(5, 3) and w.merge_ids(3, 5)
    assert len(w.track_undo_ids) == _TRACK_UNDO_MAX


def test_keep_label_does_not_advance_active_id():
    """``keep_label=True`` (``auto_increment=False``) keeps the active id across edits from JS.

    Strategy: paint id 1 on slice 0 and slice 1 via ``labels_from_js``.
    Expected outcome: ``active_label`` stays 1 (default mode would advance to
    2), and the extent readout reports both slices.
    """
    img = np.zeros((3, 6, 7), np.float32)
    w = LabelAnnotator(img, axes="ZYX", keep_label=True)
    assert w.auto_increment is False and w.active_label == 1
    for z in (0, 1):
        edit = np.zeros((6, 7), np.uint16)
        edit[2:4, 2:4] = 1
        w.labels_from_js = encode_label_slice(edit, z)
        assert w.active_label == 1
    assert w.active_label_extent == "label 1: z 0-1 (2 slices)"
    w2 = LabelAnnotator(img, axes="ZYX")
    assert w2.auto_increment is True
    w2.labels_from_js = encode_label_slice(edit, 0)
    assert w2.active_label == 2


def test_adjacent_slice_traits_follow_z_edges_and_edits():
    """``labels_prev_to_js`` / ``labels_next_to_js`` carry slices z-1 / z+1, empty out of range or when off."""
    w, lab = _two_tracks()
    assert w.show_adjacent is False
    assert bytes(w.labels_prev_to_js) == b"" and bytes(w.labels_next_to_js) == b""
    w.show_adjacent = True
    for z in range(6):
        w.z = z
        prev, nxt = _decode_or_none(w.labels_prev_to_js), _decode_or_none(w.labels_next_to_js)
        if z == 0:
            assert prev is None
        else:
            assert prev[0] == z - 1
            np.testing.assert_array_equal(prev[1], lab[z - 1])
        if z == 5:
            assert nxt is None
        else:
            assert nxt[0] == z + 1
            np.testing.assert_array_equal(nxt[1], lab[z + 1])
    w.z = 2
    w.merge_ids(7, 5)  # edits slices 4-5 only; neighbour z+1 = 3 unchanged, still sent
    w.z = 3
    _, nxt = decode_label_slice(w.labels_next_to_js)
    assert (nxt[lab[4] == 7] == 5).all(), "neighbour slice reflects the volume edit"
    w2 = LabelAnnotator(np.zeros((2, 4, 4)), axes="ZYX", show_adjacent=True)
    assert w2.z == 1 and decode_label_slice(w2.labels_prev_to_js)[0] == 0 and bytes(w2.labels_next_to_js) == b""


def test_active_label_extent_readout():
    """The extent string follows ``active_label`` and edits, and reports gaps."""
    w, lab = _two_tracks()
    w.active_label = 5
    assert w.active_label_extent == "label 5: z 2-5 (4 slices)"
    w.active_label = 42
    assert w.active_label_extent == "label 42: not present"
    w.auto_increment = False  # keep label: the readout stays on id 3 after the edit
    w.active_label = 3
    edit = lab[1].astype(np.uint16)
    edit[edit == 3] = 0
    w.labels_from_js = encode_label_slice(edit, 1)  # remove id 3 from z=1 -> gap
    assert w.active_label_extent == "label 3: z 0-3 (3 slices), missing z 1"
    w.active_label = 0
    assert "background" in w.active_label_extent
    sem = LabelAnnotator(np.zeros((4, 4)), mode="semantic")
    assert sem.active_label_extent == ""


def test_frontend_maps_alt_arrows_and_track_tools():
    """The JS binds Alt+ArrowLeft/Right to z-1/z+1 (preventing browser navigation) and has the new tools."""
    from eigenp_utils.label_annotator import LabelAnnotator as LA

    js = LA._esm
    assert 'ev.altKey && !ev.ctrlKey && !ev.metaKey && (k === "ArrowLeft" || k === "ArrowRight")' in js
    assert 'stepZ(k === "ArrowRight" ? 1 : -1)' in js and "ev.preventDefault()" in js
    for name in ('"merge"', '"split"', '"show_adjacent"', '"auto_increment", !keepCb.checked', '"track_request"'):
        assert name in js


def test_track_undo_keeps_voxels_edited_since_the_operation():
    """Undo of a merge restores only voxels that still hold the merged id.

    Strategy: merge 5 -> 3, then (as the browser would after a z change has
    dropped its slice undo history) erase part of the merged mask on z=4 and
    paint another part with a new id 9; then undo the merge.
    Expected outcome: untouched former id-5 voxels are 5 again, the erased
    voxels stay 0 and the repainted voxels stay 9 (the later edits are not
    silently reverted); ``status`` reports the skipped voxels.
    """
    w, lab = _two_tracks()
    assert w.merge_ids(5, 3)
    edit = w.labels[4].astype(np.uint16)
    ys, xs = np.nonzero(lab[4] == 5)
    edit[ys[0], xs[0]] = 0
    edit[ys[1], xs[1]] = 9
    w.labels_from_js = encode_label_slice(edit, 4)
    assert w.undo_track_op()
    out = w.labels
    assert out[4, ys[0], xs[0]] == 0 and out[4, ys[1], xs[1]] == 9
    expected = lab.copy()
    expected[4, ys[0], xs[0]] = 0
    expected[4, ys[1], xs[1]] = 9
    np.testing.assert_array_equal(out, expected)
    assert "2 voxels edited since" in w.status


def test_labels_rev_forces_reload_when_bytes_repeat():
    """Volume edits bump ``labels_rev`` even when ``labels_to_js`` bytes do not change.

    Strategy: the browser relabels mask 3 on z=1 as 6 with the fill tool
    (``labels_from_js``; Python does not echo it, so ``labels_to_js`` still
    holds the pre-fill slice), then ``merge_ids(6, 3)`` restores exactly that
    slice.
    Expected outcome: ``labels_to_js`` is byte-identical to before (hosts would
    not deliver it), but ``labels_rev`` increments, which makes the browser
    reload the slice instead of keeping (and later sending back) its stale ids.
    ``set_labels`` bumps it too.
    """
    w, lab = _two_tracks()
    w.z = 1
    before = bytes(w.labels_to_js)
    rev = w.labels_rev
    edit = lab[1].astype(np.uint16)
    edit[edit == 3] = 6
    w.labels_from_js = encode_label_slice(edit, 1)
    assert w.labels_rev == rev, "browser edits are not echoed"
    assert w.merge_ids(6, 3)
    assert bytes(w.labels_to_js) == before
    assert w.labels_rev == rev + 1
    w.set_labels(w.labels)
    assert w.labels_rev == rev + 2


def test_active_label_respects_wire_limit():
    """Ids above 65535 cannot become active (uint16 painting would wrap 65536 -> 0, 65537 -> 1).

    Strategy: set ``active_label`` to 65536; fill a volume up to id 65535 and
    ask for a new id.
    Expected outcome: 65536 raises ``TraitError`` and the active id is
    unchanged; with id 65535 present, the next free id is the smallest unused
    one instead of 65536.
    """
    import traitlets

    lab = np.zeros((1, 4, 4), np.uint32)
    lab[0, 0, 0] = 1
    lab[0, 0, 1] = 65535
    w = LabelAnnotator(np.zeros((4, 4), np.float32), labels=lab[0])
    assert w.active_label == 2  # max + 1 would be 65536
    with pytest.raises(traitlets.TraitError):
        w.active_label = 65536
    assert w.active_label == 2
    w.new_label_trigger += 1
    assert w.active_label == 2


# ------------------------------------------- sparse annotation: background, delete, counts
def _sparse_volume():
    """2 slices of 4x5 px: z0 = cell 1 (3 px), cell 2 (2 px), background (4 px);
    z1 = cell 2 (5 px), cell 9 (1 px), background (1 px). Everything else unannotated (0)."""
    from eigenp_utils.label_annotator import BACKGROUND_ID as BG

    lab = np.zeros((2, 4, 5), dtype=np.uint32)
    lab[0, 0, 0:3] = 1
    lab[0, 1, 0:2] = 2
    lab[0, 3, 1:5] = BG
    lab[1, 2, 0:5] = 2
    lab[1, 0, 4] = 9
    lab[1, 3, 0] = BG
    return lab


def test_background_id_is_reserved_and_never_allocated():
    """With ``background=True`` the id 65535 is annotated background, never a new cell id.

    Strategy: (a) volume with cells 1 and 4 plus background: the next id is 5
    (max over cells + 1, not over the raw max 65535); (b) cell 65534 present:
    the next id is the smallest unused (2), never 65535; (c) every id
    1..65534 used except 100: the next id is 100; (d) painting only background
    from the browser keeps ``active_label`` a cell id; (e) ``active_label``
    cannot be set to the background id.
    Expected outcome: exactly those ids; ``n`` / ``new_label_trigger`` agree.
    """
    import traitlets

    from eigenp_utils.label_annotator import BACKGROUND_ID

    assert BACKGROUND_ID == LabelAnnotator.BACKGROUND_ID == 65535
    lab = np.zeros((6, 7), np.uint32)
    lab[0, 0] = 1
    lab[1, 1] = 4
    lab[5, :] = BACKGROUND_ID
    img = np.zeros((6, 7), np.float32)
    w = LabelAnnotator(img, labels=lab, background=True)
    assert w.background is True and w.active_label == 5
    w.new_label_trigger += 1
    assert w.active_label == 5

    lab2 = lab.copy()
    lab2[2, 2] = 65534
    w2 = LabelAnnotator(img, labels=lab2, background=True)
    assert w2.active_label == 2

    full = np.arange(256 * 256, dtype=np.uint32).reshape(256, 256)  # 0..65535: every cell id + background
    full[full == 100] = 0
    w3 = LabelAnnotator(np.zeros((256, 256), np.float32), labels=full, background=True)
    assert w3.active_label == 100

    w4 = LabelAnnotator(img, labels=lab, background=True)
    edit = lab.astype(np.uint16)
    edit[4, :] = BACKGROUND_ID  # a background scribble from the browser
    w4.labels_from_js = encode_label_slice(edit, 0)
    assert w4.active_label == 5, "background strokes do not advance or become the active id"
    with pytest.raises(traitlets.TraitError, match="reserved"):
        w4.active_label = BACKGROUND_ID
    assert w4.active_label == 5


def test_background_requires_instance_mode():
    """``background=True`` with ``mode='semantic'`` raises (semantic mode uses a background class)."""
    with pytest.raises(ValueError, match="instance mode"):
        LabelAnnotator(np.zeros((4, 4)), mode="semantic", background=True)


def test_labels_signed_and_set_labels_round_trip_with_negatives():
    """Signed labels (-1 = background) round-trip through ``set_labels`` / ``labels_signed``.

    Strategy: build a signed 2D array with -1, 0 and cell ids; pass it to
    ``set_labels`` and to the constructor (``background=True``); also pass the
    unsigned form (65535).
    Expected outcome: ``labels`` stores 65535 where the input was -1 (uint32),
    ``labels_signed()`` is int32, has the input's spatial axes and equals the
    signed input exactly; the unsigned form gives the same volume; the wire
    slice carries 65535; any negative value maps to background.
    """
    from eigenp_utils.label_annotator import BACKGROUND_ID

    signed = np.zeros((6, 7), np.int32)
    signed[0, :3] = 1
    signed[2, 2:5] = 7
    signed[5, :] = -1
    img = np.zeros((6, 7), np.float32)
    w = LabelAnnotator(img, background=True)
    w.set_labels(signed)
    assert w.labels.dtype == np.uint32
    np.testing.assert_array_equal(w.labels == BACKGROUND_ID, signed == -1)
    out = w.labels_signed()
    assert out.dtype == np.int32 and out.shape == (6, 7)
    np.testing.assert_array_equal(out, signed)
    _, wire = decode_label_slice(w.labels_to_js)
    assert (wire[5] == BACKGROUND_ID).all()
    assert w.active_label == 8

    unsigned = np.where(signed < 0, BACKGROUND_ID, signed).astype(np.uint32)
    w.set_labels(unsigned)
    np.testing.assert_array_equal(w.labels_signed(), signed)
    w.set_labels(np.where(signed < 0, -5, signed))  # any negative value is background
    np.testing.assert_array_equal(w.labels_signed(), signed)

    w2 = LabelAnnotator(img, labels=signed, background=True)
    np.testing.assert_array_equal(w2.labels, unsigned)
    w3d = LabelAnnotator(np.zeros((2, 6, 7), np.float32), axes="ZYX", background=True)
    w3d.set_labels(np.stack([signed, -signed]))
    assert w3d.labels_signed().shape == (2, 6, 7)
    assert w3d.labels_signed()[1, 0, 0] == -1 and w3d.labels_signed()[1, 5, 0] == 1


def test_delete_id_on_one_slice_and_all_slices_with_undo():
    """``delete_id`` removes a whole mask on one slice or everywhere; both are undoable track ops.

    Strategy: on the two-track phantom delete id 3 on z=1, then id 5 in all
    slices; undo both (newest first); then the same through ``track_request``
    as the browser sends it (``z`` None for shift+click). Refusals: id 0, an
    absent id, an id absent from the given slice, an out-of-range z.
    Expected outcome: only the targeted voxels become 0; extents follow;
    each undo restores the previous volume exactly; refusals change nothing
    and say "refused".
    """
    w, lab = _two_tracks()
    assert w.delete_id(3, z=1)
    expected = lab.copy()
    expected[1][lab[1] == 3] = 0
    np.testing.assert_array_equal(w.labels, expected)
    assert w.label_extent(3) == [0, 2, 3]
    assert w.delete_id(5)
    expected[expected == 5] = 0
    np.testing.assert_array_equal(w.labels, expected)
    assert w.label_extent(5) == [] and "all slices (4 slices)" in w.status
    assert len(w.track_undo_ids) == 2
    assert w.undo_track_op()
    assert w.label_extent(5) == [2, 3, 4, 5]
    assert w.undo_track_op()
    np.testing.assert_array_equal(w.labels, lab)

    for args in [(0, None), (42, None), (7, 0), (3, 99)]:
        assert not w.delete_id(*args)
        assert "refused" in w.status
    np.testing.assert_array_equal(w.labels, lab)
    assert w.track_undo_ids == []

    w.z = 2
    w.track_request = {"op": "delete", "id": 3, "z": 2, "n": 1}
    assert w.label_extent(3) == [0, 1, 3]
    _, cur = decode_label_slice(w.labels_to_js)
    assert not (cur == 3).any(), "the current slice is re-sent after the delete"
    w.track_request = {"op": "delete", "id": 7, "z": None, "n": 2}
    assert w.label_extent(7) == []
    w.track_request = {"op": "undo", "serial": w.track_undo_ids[-1], "n": 3}
    w.track_request = {"op": "undo", "serial": w.track_undo_ids[-1], "n": 4}
    np.testing.assert_array_equal(w.labels, lab)


def test_delete_id_removes_background_scribbles():
    """The background id can be deleted like any mask (all slices), and the delete is undoable."""
    from eigenp_utils.label_annotator import BACKGROUND_ID

    lab = _sparse_volume()
    w = LabelAnnotator(np.zeros(lab.shape, np.float32), axes="ZYX", labels=lab, background=True)
    assert w.delete_id(BACKGROUND_ID)
    assert not (w.labels == BACKGROUND_ID).any()
    np.testing.assert_array_equal(w.labels[lab != BACKGROUND_ID], lab[lab != BACKGROUND_ID])
    assert w.counts()["background_px"] == 0
    assert w.undo_track_op()
    np.testing.assert_array_equal(w.labels, lab)


def test_counts_on_hand_built_volume_and_readout_updates():
    """``counts`` gives cells / background / annotated pixels per slice and for the volume.

    Strategy: hand-built 2x4x5 volume (see ``_sparse_volume``): z0 has cells
    1, 2 (5 px) and 4 background px; z1 has cells 2, 9 (6 px) and 1 background px.
    Then erase cell 9 from the browser and change z.
    Expected outcome: z0 -> 2 cells, 5/4/9 px, 9/20; z1 -> 2 cells, 6/1/7 px,
    7/20; volume -> 3 distinct cells (2 spans both slices), 11/5/16 px, 0.4.
    ``counts_text`` shows the slice and volume numbers and follows the edit
    and the z change. Without ``background`` 65535 counts as a cell.
    """
    from eigenp_utils.label_annotator import BACKGROUND_ID

    lab = _sparse_volume()
    w = LabelAnnotator(np.zeros(lab.shape, np.float32), axes="ZYX", labels=lab, background=True)
    keys = ("n_cells", "cell_px", "background_px", "annotated_px", "annotated_fraction")
    assert tuple(w.counts(0)) == keys
    assert w.counts(0) == {"n_cells": 2, "cell_px": 5, "background_px": 4, "annotated_px": 9, "annotated_fraction": 9 / 20}
    assert w.counts(1) == {"n_cells": 2, "cell_px": 6, "background_px": 1, "annotated_px": 7, "annotated_fraction": 7 / 20}
    assert w.counts() == {"n_cells": 3, "cell_px": 11, "background_px": 5, "annotated_px": 16, "annotated_fraction": 0.4}
    with pytest.raises(ValueError):
        w.counts(2)
    assert w.z == 1
    assert w.counts_text == (
        "z 1: 2 cells, 1 bg px, 7 annotated px (35.0%) | all z: 3 cells, 5 bg px, 16 annotated px (40.0%)"
    )
    edit = lab[1].astype(np.uint16)
    edit[edit == 9] = 0
    w.labels_from_js = encode_label_slice(edit, 1)
    assert w.counts() == {"n_cells": 2, "cell_px": 10, "background_px": 5, "annotated_px": 15, "annotated_fraction": 15 / 40}
    assert w.counts_text.startswith("z 1: 1 cell, 1 bg px, 6 annotated px (30.0%)")
    w.z = 0
    assert w.counts_text.startswith("z 0: 2 cells, 4 bg px, 9 annotated px (45.0%)")

    plain = LabelAnnotator(np.zeros(lab.shape, np.float32), axes="ZYX", labels=lab)
    assert plain.counts() == {"n_cells": 4, "cell_px": 16, "background_px": 0, "annotated_px": 16, "annotated_fraction": 0.4}
    assert "bg px" not in plain.counts_text
    flat = LabelAnnotator(np.zeros((4, 5), np.float32), labels=lab[0], background=True)
    assert flat.counts_text == "2 cells, 4 bg px, 9 annotated px (45.0%)"
    assert flat.counts() == flat.counts(0)
    sem = LabelAnnotator(np.zeros((4, 5), np.float32), mode="semantic", labels=np.where(lab[0] == BACKGROUND_ID, 2, lab[0]))
    assert sem.counts_text == "9 annotated px (45.0%)"


def test_merge_and_split_refuse_background_id():
    """Merge / split never touch the background id (it is not a cell); the volume is unchanged."""
    from eigenp_utils.label_annotator import BACKGROUND_ID

    lab = np.zeros((3, 6, 7), np.uint32)
    lab[0:3, 0:2, 0:2] = 4
    lab[0:3, 4:6, :] = BACKGROUND_ID
    w = LabelAnnotator(np.zeros(lab.shape, np.float32), axes="ZYX", labels=lab, background=True)
    assert not w.merge_ids(BACKGROUND_ID, 4)
    assert "annotated background" in w.status and "refused" in w.status
    assert not w.merge_ids(4, BACKGROUND_ID)
    assert "annotated background" in w.status
    assert w.split_id(BACKGROUND_ID, 1) is None
    assert "annotated background" in w.status and "refused" in w.status
    w.track_request = {"op": "merge", "src": BACKGROUND_ID, "dst": 4, "n": 1}
    np.testing.assert_array_equal(w.labels, lab)
    assert w.track_undo_ids == []
    assert w.split_id(4, 1) == 5, "cells still split; the new id is not the background id"


def test_save_signed_writes_int32_with_minus_one_background(tmp_path):
    """``save(signed=True)`` writes int32 masks with -1 background; the default stays uint32 / 65535."""
    tifffile = pytest.importorskip("tifffile")
    from eigenp_utils.label_annotator import BACKGROUND_ID

    lab = _sparse_volume()
    img = np.random.default_rng(3).integers(0, 999, size=lab.shape, dtype=np.uint16)
    w = LabelAnnotator(img, axes="ZYX", labels=lab, background=True)
    paths = w.save(tmp_path / "s", "a", signed=True)
    assert [p.name for p in paths] == ["a_z000_img.tif", "a_z000_masks.tif", "a_z001_img.tif", "a_z001_masks.tif"]
    for z, pm in ((0, paths[1]), (1, paths[3])):
        m = tifffile.imread(pm)
        assert m.dtype == np.int32
        np.testing.assert_array_equal(m, np.where(lab[z] == BACKGROUND_ID, -1, lab[z].astype(np.int64)))
    np.testing.assert_array_equal(tifffile.imread(paths[2]), img[1])
    plain = w.save(tmp_path / "u", "a")
    m = tifffile.imread(plain[1])
    assert m.dtype == np.uint32
    np.testing.assert_array_equal(m, lab[0])


def test_background_false_keeps_previous_behaviour():
    """Default ``background=False``: 65535 is an ordinary id, negatives are rejected, no background counts.

    Strategy: a volume with ids 1 and 65535 and no ``background`` argument.
    Expected outcome: ``background`` is False; 65535 can be active, merged and
    split; ``labels_signed`` keeps 65535; signed input still raises
    ``ValueError`` (with a hint); ``counts`` reports no background pixels.
    """
    lab = np.zeros((2, 6, 7), np.uint32)
    lab[:, 0, 0] = 1
    lab[:, 5, 5] = 65535
    w = LabelAnnotator(np.zeros(lab.shape, np.float32), axes="ZYX", labels=lab)
    assert w.background is False
    assert w.active_label == 2
    w.active_label = 65535
    assert w.active_label == 65535
    out = w.labels_signed()
    assert out.dtype == np.int32 and out[0, 5, 5] == 65535
    assert w.counts()["background_px"] == 0 and w.counts()["n_cells"] == 2
    with pytest.raises(ValueError, match="background=True"):
        w.set_labels(np.full(lab.shape, -1, np.int32))
    assert w.split_id(65535, 1) == 2
    assert w.merge_ids(2, 65535)
    np.testing.assert_array_equal(w.labels, lab)


def test_frontend_has_background_and_delete_tools():
    """The JS defines the background brush (``g``, open scribbles) and the delete tool (``x``, shift = all z)."""
    js = LabelAnnotator._esm
    assert "const BG_ID = 65535;" in js
    for needle in (
        '"background"', '"delete"', 'map.g = "background"', 'x: "delete"', 'op: "delete"',
        "ev.shiftKey ? null : state.lab.z", '"counts_text"', 'model.get("tool") === "brush" && model.get("auto_fill")',
    ):
        assert needle in js, needle


def test_display_defaults_and_outline_traits():
    """Display traits default to filled labels with a 2-px outline and 1-px border, and accept kwargs.

    Strategy: construct with defaults, switch ``display`` to ``"outlined"``,
    try an invalid value, and construct with ``display``/``outline_px``
    kwargs.
    Expected outcome: defaults ``"filled"``, 2 and 1; the legacy
    ``outline_only`` stays False; an invalid display value raises; the
    kwargs are kept (not overwritten by the constructor).
    """
    w = LabelAnnotator(np.zeros((8, 8), np.float32))
    assert w.display == "filled" and w.outline_px == 2 and w.outline_border_px == 1
    assert w.outline_only is False  # legacy trait unchanged
    w.display = "outlined"
    with pytest.raises(Exception):
        w.display = "dotted"
    assert LabelAnnotator(np.zeros((8, 8), np.float32), display="outlined", outline_px=3).outline_px == 3


def test_multiple_stacks_keep_separate_labels_and_state(tmp_path):
    """A list of stacks keeps labels, z and geometry per stack, and saves every stack.

    Strategy: two ZYX stacks of different shapes; edit stack 0, switch to
    stack 1 and edit it, switch back; read ``stack_labels``; save with
    ``stacks="all"``; set an out-of-range ``stack``; pass a wrong number of
    ``stack_names``.
    Expected outcome: each stack shows its own shape and labels, the next
    free id restarts per stack, z is restored on return; the saved file
    names carry the sanitised stack names and saving leaves the shown stack
    alone; ``stack`` is clamped; mismatched names raise ``ValueError``.
    """
    a =np.random.default_rng(0).random((3, 8, 9)).astype(np.float32)
    b = np.random.default_rng(1).random((2, 6, 7)).astype(np.float32)
    w = LabelAnnotator([a, b], axes="ZYX", stack_names=["cage A", "cage B"], brush_radius=0)
    assert w.stack == 0 and w.stack_names == ["cage A", "cage B"] and w.shape_zyx == [3, 8, 9]
    z0 = w.z
    w.labels_from_js = encode_label_slice(np.full((8, 9), 5, np.uint32), 0)
    w.stack = 1
    assert w.shape_zyx == [2, 6, 7] and w.labels.shape == (2, 6, 7) and not w.labels.any()
    assert w.active_label == 1  # new stack: next free id
    w.labels_from_js = encode_label_slice(np.full((6, 7), 2, np.uint32), 1)
    w.stack = 0
    assert w.z == z0 and (w.labels[0] == 5).all()
    labs = w.stack_labels
    assert (labs[0][0] == 5).all() and (labs[1][1] == 2).all()
    paths = w.save(tmp_path, "t", stacks="all")
    assert sorted(p.name for p in paths) == sorted(
        ["t_cage_A_z000_img.tif", "t_cage_A_z000_masks.tif", "t_cage_B_z001_img.tif", "t_cage_B_z001_masks.tif"])
    assert w.stack == 0
    w.stack = 7  # clamped
    assert w.stack == 1
    with pytest.raises(ValueError):
        LabelAnnotator([a, b], stack_names=["x"], axes="ZYX")


# ------------------------------------------------- review fixes (2026-11-25)
def test_undo_epoch_bumps_on_stack_switch_and_set_labels_only():
    """``undo_epoch`` tells the browser to drop its per-slice undo history (R4-01).

    Strategy: two same-shape 2D stacks; record ``undo_epoch`` before and after
    a browser stroke, a track op, its undo, a stack switch, ``set_labels`` and
    a z change (3D widget).
    Expected outcome: the epoch changes exactly on the stack switch and on
    ``set_labels`` (the browser's slice snapshots then belong to another stack
    or to replaced labels); strokes, track ops (whose browser undo markers must
    survive), track undo and z changes leave it alone.
    """
    a = np.zeros((6, 7), np.float32)
    w = LabelAnnotator([a, a.copy()], stack_names=["a", "b"])
    ep = w.undo_epoch
    edit = np.zeros((6, 7), np.uint16)
    edit[1:3, 1:3] = 4
    w.labels_from_js = encode_label_slice(edit, 0)
    assert w.undo_epoch == ep, "strokes keep the browser undo history"
    assert w.delete_id(4) and w.undo_track_op()
    assert w.undo_epoch == ep, "track ops and their undo keep it too"
    w.stack = 1
    assert w.undo_epoch == ep + 1
    w.set_labels(np.ones((6, 7), np.uint32))
    assert w.undo_epoch == ep + 2
    w3 = LabelAnnotator(np.zeros((3, 6, 7), np.float32), axes="ZYX")
    ep3 = w3.undo_epoch
    w3.z = 0
    assert w3.undo_epoch == ep3, "z changes are handled per slice in the browser"


def test_track_op_serials_are_unique_across_stacks():
    """Track-op serials never repeat across stacks (R4-02).

    Strategy: two 3D stacks with the same three tracks; merge + split on
    stack 0, switch to stack 1 and merge there, switch back; then send a stale
    undo request carrying stack 1's serial while stack 0 is shown.
    Expected outcome: stack 0 reports serials ``[1, 2]``, stack 1's first op
    gets serial 3 (not 1 again: the browser appends only serials it has not
    mirrored yet), stack 0's ids come back as ``[1, 2]``, and the stale
    request is refused without changing stack 0.
    """
    _, lab = _two_tracks()
    img = np.random.default_rng(7).random(lab.shape).astype(np.float32)
    w = LabelAnnotator([img, img.copy()], axes="ZYX", labels=[lab, lab.copy()])
    assert w.merge_ids(5, 3) and w.split_id(3, 2) is not None
    assert w.track_undo_ids == [1, 2]
    w.stack = 1
    assert w.track_undo_ids == []
    assert w.merge_ids(5, 3)
    assert w.track_undo_ids == [3]
    w.stack = 0
    assert w.track_undo_ids == [1, 2]
    before = w.labels
    w.track_request = {"op": "undo", "serial": 3, "n": 1}
    assert "out of sync" in w.status
    np.testing.assert_array_equal(w.labels, before)
    w.track_request = {"op": "undo", "serial": 2, "n": 2}
    assert w.track_undo_ids == [1]


def test_identical_label_messages_apply_after_python_changed_the_slice():
    """A label message byte-identical to the previous one is still applied (R4-04).

    Strategy: (a) one 2D widget: the browser clears the slice (zeros), Python
    ``set_labels`` fills it with 3, the browser clears it again with the same
    bytes; (b) two same-shape 2D stacks with masks: clear stack 0, switch to
    stack 1, clear stack 1 with the same bytes.
    Expected outcome: the second clear empties the slice in both cases
    (traitlets and the hosts drop equal values, so the widget must forget each
    message once applied). Negative control: the first message is applied as
    before.
    """
    clear = encode_label_slice(np.zeros((6, 7), np.uint16), 0)
    w = LabelAnnotator(np.zeros((6, 7), np.float32), labels=np.full((6, 7), 2))
    w.labels_from_js = clear
    assert not w.labels.any()
    w.set_labels(np.full((6, 7), 3))
    w.labels_from_js = clear
    assert not w.labels.any(), "second identical clear must be applied"

    lab = np.full((6, 7), 2, np.uint32)
    ws = LabelAnnotator([np.zeros((6, 7)), np.zeros((6, 7))], labels=[lab, lab.copy()])
    ws.labels_from_js = clear
    ws.stack = 1
    ws.labels_from_js = clear
    assert [int(x.max()) for x in ws.stack_labels] == [0, 0]


def test_float_labels_must_be_finite_integers():
    """Float label arrays are accepted only when every value is a finite integer (R4-09).

    Strategy: pass float labels with NaN, with non-integral values (0.999,
    1.7), integral floats, ``-1.0`` with ``background=True``, and an id above
    65535, to the constructor and to ``set_labels``.
    Expected outcome: NaN and non-integral values raise ``ValueError`` (NaN
    used to allocate 16 GiB in a bincount; 0.999 was silently floored to 0);
    integral floats load exactly as their integer ids, ``-1.0`` becomes the
    background id; the out-of-range message still says "labels must be in" and
    points to ``relabel_sequential``.
    """
    img = np.zeros((2, 3), np.float32)
    nan = np.array([[0, 1, 2], [np.nan, 0, 0]], np.float32)
    with pytest.raises(ValueError, match="finite"):
        LabelAnnotator(img, labels=nan)
    with pytest.raises(ValueError, match="integer"):
        LabelAnnotator(img, labels=np.array([[0, 1.7, 2.2], [0.999, 0, 0]]))
    w = LabelAnnotator(img, labels=np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 7.0]]))
    np.testing.assert_array_equal(w.labels, [[0, 1, 2], [0, 0, 7]])
    assert w.labels.dtype == np.uint32 and w.active_label == 8
    with pytest.raises(ValueError, match="finite"):
        w.set_labels(nan)
    np.testing.assert_array_equal(w.labels, [[0, 1, 2], [0, 0, 7]])
    wb = LabelAnnotator(img, labels=np.array([[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]), background=True)
    assert wb.labels[0, 0] == LabelAnnotator.BACKGROUND_ID
    with pytest.raises(ValueError, match="labels must be in.*relabel_sequential"):
        LabelAnnotator(img, labels=np.full((2, 3), 70_000.0))


def test_list_of_stacks_takes_labels_as_sequence_or_stacked_array():
    """With a list of stacks, ``labels`` is a sequence or an array stacked along axis 0 (R4-11).

    Strategy: two ``(2, 8, 9)`` stacks; pass labels as a list, a tuple, a
    stacked ``(2, 2, 8, 9)`` ndarray, None, a single ``(2, 8, 9)`` array (one
    stack's labels), a 3-entry stacked array, and a dict.
    Expected outcome: list, tuple and stacked array load each stack's ids
    (3 and 4); None starts empty; the single array raises ``ValueError``
    (its entries do not match the stacks' shapes), the 3-entry array raises
    ``ValueError`` and the dict raises ``TypeError``, instead of silently
    starting every stack empty.
    """
    imgs = [np.zeros((2, 8, 9), np.float32), np.zeros((2, 8, 9), np.float32)]
    lab = np.zeros((2, 2, 8, 9), np.uint32)
    lab[0] = 3
    lab[1] = 4
    for labels in (list(lab), tuple(lab), lab):
        w = LabelAnnotator(imgs, axes="ZYX", labels=labels)
        assert [int(x.max()) for x in w.stack_labels] == [3, 4]
    w = LabelAnnotator(imgs, axes="ZYX", labels=None)
    assert [int(x.max()) for x in w.stack_labels] == [0, 0]
    with pytest.raises(ValueError):
        LabelAnnotator(imgs, axes="ZYX", labels=lab[0])
    with pytest.raises(ValueError, match="2 stacks but 3"):
        LabelAnnotator(imgs, axes="ZYX", labels=np.zeros((3, 2, 8, 9), np.uint32))
    with pytest.raises(TypeError):
        LabelAnnotator(imgs, axes="ZYX", labels={0: lab[0], 1: lab[1]})


def test_save_all_stacks_refuses_colliding_names_and_leaves_the_view_alone(tmp_path):
    """``save(stacks="all")`` checks every stack before writing and does not drive the UI (R4-12).

    Strategy: two 2D stacks named "cage A" and "cage/A" (both sanitise to
    ``cage_A``) with masks 1 and 2; save with ``overwrite=True`` and with
    ``overwrite=False``. Then two distinct names, ``keep_label`` on and an
    active id set by hand, observing ``stack`` and ``labels_rev`` while saving.
    Expected outcome: the colliding saves raise ``ValueError`` and write no
    file (stack 0's masks were silently replaced before); with distinct names
    both stacks are written with their own masks, while ``stack``,
    ``active_label``, ``labels_rev`` and the shown image are untouched and no
    stack switch is notified.
    """
    tifffile = pytest.importorskip("tifffile")
    a = np.random.default_rng(0).random((6, 7)).astype(np.float32)
    labs = [np.full((6, 7), 1, np.uint32), np.full((6, 7), 2, np.uint32)]
    w = LabelAnnotator([a, a.copy()], labels=labs, stack_names=["cage A", "cage/A"])
    for ow in (True, False):
        with pytest.raises(ValueError, match="stack_names"):
            w.save(tmp_path, "t", overwrite=ow, stacks="all")
        assert list(tmp_path.iterdir()) == []

    w = LabelAnnotator([a, a.copy()], labels=labs, stack_names=["A", "B"], keep_label=True)
    w.active_label = 1
    seen = []
    w.observe(lambda ch: seen.append(ch["new"]), names="stack")
    rev, img = w.labels_rev, bytes(w.image_to_js)
    paths = w.save(tmp_path, "t", stacks="all")
    assert [p.name for p in paths] == ["t_A_img.tif", "t_A_masks.tif", "t_B_img.tif", "t_B_masks.tif"]
    assert int(tifffile.imread(paths[1]).max()) == 1 and int(tifffile.imread(paths[3]).max()) == 2
    assert seen == [] and w.stack == 0 and w.active_label == 1
    assert w.labels_rev == rev and bytes(w.image_to_js) == img


def test_save_validates_arguments_and_warns_when_nothing_is_written(tmp_path):
    """``save`` rejects unknown ``slices`` / ``stacks`` values and warns on an empty save (R4-22).

    Strategy: call ``save`` with ``slices="anotated"``, ``stacks="All"`` (on a
    two-stack widget), then an unannotated widget with the default
    ``slices="annotated"``.
    Expected outcome: the typos raise ``ValueError`` naming the allowed values
    (instead of a bare ``KeyError`` or silently saving only the current
    stack) and write nothing; the empty save returns ``[]`` with a
    ``UserWarning``.
    """
    w = LabelAnnotator(np.zeros((3, 6, 7), np.float32), axes="ZYX", labels=np.ones((3, 6, 7), np.uint32))
    with pytest.raises(ValueError, match="'annotated', 'all', 'current'"):
        w.save(tmp_path, "a", slices="anotated")
    two = LabelAnnotator([np.zeros((6, 7)), np.zeros((6, 7))], labels=[np.ones((6, 7)), np.ones((6, 7))])
    with pytest.raises(ValueError, match="'current', 'all'"):
        two.save(tmp_path, "a", stacks="All")
    assert list(tmp_path.iterdir()) == []
    empty = LabelAnnotator(np.zeros((3, 6, 7), np.float32), axes="ZYX")
    with pytest.warns(UserWarning, match="nothing"):
        assert empty.save(tmp_path, "e") == []


def test_nan_and_inf_image_pixels_render_without_warnings():
    """Non-finite image pixels are mapped before the uint8 cast, so no RuntimeWarning (R4-23).

    Strategy: a float ramp with one NaN, one +inf and one -inf pixel, and an
    all-NaN image, built and z-stepped with RuntimeWarnings turned into errors.
    Expected outcome: no warning; NaN and -inf render as 0, +inf as 255, and
    the finite pixels render exactly as before (monotone ramp up to 255).
    """
    img = np.linspace(0, 1, 2 * 6 * 7, dtype=np.float32).reshape(2, 6, 7)
    img[1, 0, 0], img[1, 0, 1], img[1, 0, 2] = np.nan, np.inf, -np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        w = LabelAnnotator(img, axes="ZYX")
        w.z = 1
        LabelAnnotator(np.full((6, 7), np.nan, np.float32))
    data = np.frombuffer(bytes(w.image_to_js), np.uint8, offset=_IMG_HDR.size).reshape(6, 7)
    assert data[0, 0] == 0 and data[0, 1] == 255 and data[0, 2] == 0
    assert data[-1, -1] == 255 and np.all(np.diff(data.ravel()[3:].astype(int)) >= 0)


def test_bool_image_is_accepted_and_saved_unchanged(tmp_path):
    """A boolean mask (e.g. from ``extract_surface``) can be the image (C-04).

    Strategy: a ``(3, 6, 7)`` bool volume as the image (axes ``ZYX``), then
    as one of two stacks; read the display slice and save an annotated slice.
    Expected outcome: no ``TypeError`` from the percentile of bool data; True
    pixels display as 255 and False as 0; ``save`` writes the original bool
    data, not a rescaled copy.
    """
    tifffile = pytest.importorskip("tifffile")
    mask = np.zeros((3, 6, 7), bool)
    mask[:, 2:4, 2:5] = True
    w = LabelAnnotator(mask, axes="ZYX", labels=np.ones((3, 6, 7), np.uint32))
    data = np.frombuffer(bytes(w.image_to_js), np.uint8, offset=_IMG_HDR.size).reshape(6, 7)
    np.testing.assert_array_equal(data, np.where(mask[1], 255, 0))
    paths = w.save(tmp_path, "m", slices="current")
    saved = tifffile.imread(paths[0])
    assert saved.dtype == bool
    np.testing.assert_array_equal(saved, mask[1])
    two = LabelAnnotator([mask, mask.copy()], axes="ZYX")
    two.stack = 1
    np.testing.assert_array_equal(np.frombuffer(bytes(two.image_to_js), np.uint8, offset=_IMG_HDR.size).reshape(6, 7), data)


def test_rendered_view_resync_resends_stale_slice_only():
    """A view rendered after strokes asks for the current slice; Python re-sends it if stale (R4-03).

    Strategy: a 3D widget at z=1; bump ``labels_resync`` (what a newly
    rendered view does) before any stroke, then after a browser stroke on
    z=1, then after a stroke on another slice.
    Expected outcome: with nothing stale no labels are re-sent (``labels_rev``
    unchanged; a view rendered while the kernel is busy must not have its
    queued strokes reloaded away); after the stroke on the shown slice
    ``labels_to_js`` carries it and ``labels_rev`` is bumped (so the new view
    does not start from, and later send back, the pre-stroke slice); a stroke
    on another slice does not make the shown slice stale.
    """
    w = LabelAnnotator(np.zeros((3, 6, 7), np.float32), axes="ZYX")
    rev = w.labels_rev
    w.labels_resync += 1
    assert w.labels_rev == rev
    edit = np.zeros((6, 7), np.uint16)
    edit[1:3, 1:3] = 4
    w.labels_from_js = encode_label_slice(edit, 1)
    w.labels_resync += 1
    assert w.labels_rev == rev + 1
    z, sl = decode_label_slice(w.labels_to_js)
    assert z == 1
    np.testing.assert_array_equal(sl, edit)
    w.labels_from_js = encode_label_slice(edit, 0)
    w.labels_resync += 1
    assert w.labels_rev == rev + 1


# ------------------------------------------------- stage 2: API, memory, metadata (D17/D18, R4-19, R8-10)
def test_labels_are_stored_as_uint16_and_returned_as_uint32(tmp_path):
    """Labels live in ``uint16`` internally (ids are at most 65535) but the public API stays ``uint32`` (R4-19).

    Strategy: a 2-stack widget with ids up to 65535; read ``_labels`` and the
    parked stack, then every public getter (``labels``, ``stack_labels``,
    ``save`` masks); run a track op and check its undo entry's dtypes.
    Expected outcome: internal arrays are ``uint16`` (half the memory of
    ``uint32``); ``labels``, ``stack_labels`` and saved masks are ``uint32``
    with the same values; track-op undo stores ``int32`` indices and
    ``uint16`` old values (not ``int64`` + ``uint32``: 6 instead of 12 bytes
    per voxel).
    """
    tifffile = pytest.importorskip("tifffile")
    lab = np.zeros((2, 6, 7), np.uint32)
    lab[0, :2] = 65535
    lab[1, 3:] = 7
    w = LabelAnnotator([np.zeros((2, 6, 7)), np.zeros((2, 6, 7))], axes="ZYX", labels=[lab, lab])
    assert w._labels.dtype == np.uint16 and w._parked[1]["labels"].dtype == np.uint16
    assert w.labels.dtype == np.uint32 and np.array_equal(w.labels, lab)
    assert all(s.dtype == np.uint32 and np.array_equal(s, lab) for s in w.stack_labels)
    w.labels_from_js = encode_label_slice(np.full((6, 7), 9, np.uint16), 1)
    assert w._labels.dtype == np.uint16 and (w.labels[1] == 9).all()
    paths = w.save(tmp_path, "t", slices="all")
    masks = tifffile.imread(paths[1])
    assert masks.dtype == np.uint32 and masks.max() == 65535
    assert w.merge_ids(9, 3)
    _, idx, old, *_ = w._track_undo[-1]
    assert idx.dtype == np.int32 and old.dtype == np.uint16
    assert w.undo_track_op() and (w.labels[1] == 9).all() and w._labels.dtype == np.uint16


def _u8(w):
    """The 8-bit image slice the widget last sent, as ``(C, Y, X)``."""
    z, c, y, x = _img_header(w)
    return np.frombuffer(bytes(w.image_to_js), dtype=np.uint8, offset=_IMG_HDR.size).reshape(c, y, x)


def test_trait_kwargs_are_honoured_and_validated():
    """Display and labelling kwargs given at construction are kept, not overwritten (D18, R4-08).

    Strategy: pass ``z``, ``auto_fill``, ``vmin``/``vmax``/``gamma`` (data
    units), ``active_label``, ``channel_visible``, ``class_colors``,
    ``auto_increment`` and ``stack``; then invalid values of each, a trait
    the widget computes (``shape_zyx``) and an unknown name.
    Expected outcome: every given value survives construction (before, z
    became nz // 2, vmin/vmax 0/255, gamma 1, active_label 1, auto_fill
    True); ``stack=1`` gives the geometry, labels and z of stack 1 (it used
    to keep stack 0's shape); invalid values raise ``ValueError``, computed
    traits and unknown names ``TypeError`` (unknown names were silently
    ignored).
    """
    v = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    w = LabelAnnotator(v, axes="ZYX", z=1, auto_fill=False, vmin=[50], vmax=[100], gamma=[0.5], active_label=7,
                       channel_visible=[False], opacity=0.3)
    assert (w.z, w.auto_fill, w.vmin, w.vmax, w.gamma, w.active_label) == (1, False, [50.0], [100.0], [0.5], 7)
    assert w.channel_visible == [False] and w.opacity == 0.3
    assert _img_header(w)[0] == 1, "the slice sent is the requested z"
    w = LabelAnnotator(v, axes="ZYX", vmin=10, vmax=20.5, gamma=2)  # scalars broadcast over channels
    assert (w.vmin, w.vmax, w.gamma) == ([10.0], [20.5], [2.0])
    w = LabelAnnotator(v, axes="ZYX", mode="semantic", classes=["a", "b"], class_colors=["red", "#00ff00"],
                       active_label=2, auto_fill=True)
    assert w.class_colors == ["#ff0000", "#00ff00"] and w.active_label == 2 and w.auto_fill is True
    w = LabelAnnotator(v, axes="ZYX", auto_increment=False)
    assert w.auto_increment is False

    a, b = np.zeros((3, 8, 9), np.float32), np.zeros((2, 6, 7), np.float32)
    lb = np.zeros((2, 6, 7), np.uint32)
    lb[1] = 4
    w = LabelAnnotator([a, b], axes="ZYX", labels=[None, lb], stack=1)
    assert w.stack == 1 and w.shape_zyx == [2, 6, 7] and w.labels.shape == (2, 6, 7) and w.labels.max() == 4
    assert _img_header(w)[2:] == (6, 7) and w.active_label == 5
    w.stack = 0
    assert w.shape_zyx == [3, 8, 9] and not w.labels.any()

    bad = [dict(z=5), dict(z=-1), dict(vmin=5, vmax=5), dict(vmin=[1, 2]), dict(gamma=0), dict(gamma=[np.nan]),
           dict(channel_visible=[True, True]), dict(stack=1),
           dict(mode="semantic", classes=["a"], class_colors=["red", "blue"]),
           dict(mode="semantic", classes=["a"], active_label=2), dict(keep_label=True, auto_increment=True)]
    for kw in bad:
        with pytest.raises(ValueError):
            LabelAnnotator(v, axes="ZYX", **kw)
    with pytest.raises(Exception, match="reserved"):
        LabelAnnotator(v, axes="ZYX", background=True, active_label=65535)
    for kw in (dict(shape_zyx=[1, 2, 3]), dict(image_to_js=b""), dict(colour="red")):
        with pytest.raises(TypeError):
            LabelAnnotator(v, axes="ZYX", **kw)


def test_mode_and_background_are_read_only_after_construction():
    """``mode`` and ``background`` cannot change after construction (D18, R4-08).

    Strategy: set ``background = True`` on a widget whose labels hold 65535
    and whose active id is 65535, and switch ``mode``.
    Expected outcome: both raise (``TraitError``) and leave the state alone;
    before, ``background = True`` silently turned cell 65535 into annotated
    background with the reserved id still active, and switching mode left
    no classes. Setting the current value again is allowed.
    """
    import traitlets

    lab = np.zeros((6, 7), np.uint32)
    lab[0, 0] = 65535
    w = LabelAnnotator(np.zeros((6, 7)), labels=lab)
    w.active_label = 65535
    with pytest.raises(traitlets.TraitError, match="construction"):
        w.background = True
    with pytest.raises(traitlets.TraitError, match="construction"):
        w.mode = "semantic"
    assert w.background is False and w.mode == "instance" and w.active_label == 65535
    w.mode = "instance"
    w.background = False


def test_tnia_style_aliases_colormap_and_channel_labels():
    """``colormap=`` and ``channel_labels=`` are aliases of ``channel_colors=`` / ``channel_names=`` (D18, R4-15).

    Strategy: build a 2-channel widget with the tnia names, then with both
    spellings of the same argument.
    Expected outcome: the aliases set the same traits as the original names;
    giving both spellings raises ``TypeError``.
    """
    img = np.zeros((2, 6, 7))
    w = LabelAnnotator(img, axes="CYX", colormap=["red", "gray"], channel_labels=["a", "b"])
    ref = LabelAnnotator(img, axes="CYX", channel_colors=["red", "gray"], channel_names=["a", "b"])
    assert w.channel_colors == ref.channel_colors == ["#ff0000", "#ffffff"]
    assert w.channel_names == ref.channel_names == ["a", "b"]
    with pytest.raises(TypeError, match="colormap"):
        LabelAnnotator(img, axes="CYX", colormap=["red", "gray"], channel_colors=["red", "gray"])
    with pytest.raises(TypeError, match="channel_labels"):
        LabelAnnotator(img, axes="CYX", channel_labels=["a", "b"], channel_names=["a", "b"])


def test_vmin_vmax_are_data_units_and_requantise_the_slice():
    """``vmin``/``vmax`` are in data units; changing them re-quantises the 8-bit slice in Python (D02, R4-07).

    Strategy: a 0..4095 ramp with 4 hot pixels at 60000 (12-bit camera).
    Read the default range, then set ``vmax = [900]`` from Python; then
    change only ``gamma``; then switch between two stacks with different
    ranges.
    Expected outcome: the defaults are the package's default display range
    in data units (``vmin`` 0, ``vmax`` the robust p99.99); after
    ``vmax = [900]`` the sent slice equals ``floor(255 * v / 900)`` clipped
    (+): the part of the ramp below 900 gets 255 levels, which the fixed
    8-bit pre-quantisation could not give (the GUI could only narrow it).
    Gamma is applied in the browser, so it does not change the bytes (−),
    and the labels are not re-sent. ``data_range`` spans the data so the
    sliders can also widen the range. Each stack keeps its own range.
    """
    from eigenp_utils._conventions import default_display_range

    ramp = np.tile(np.arange(4096, dtype=np.uint16), (4, 1)).reshape(1, 64, 256)
    ramp = np.concatenate([ramp, ramp])  # 2 z-slices
    ramp[1, 0, :4] = 60000
    w = LabelAnnotator(ramp, axes="ZYX")
    assert (w.vmin[0], w.vmax[0]) == default_display_range(ramp)
    lo, hi, step, auto_lo, auto_hi = w.data_range[0]
    assert (lo, hi) == (0.0, 60000.0) and step == 1.0 and (auto_lo, auto_hi) == (w.vmin[0], w.vmax[0])
    w.z = 0
    w.vmax = [900]
    expect = np.clip(np.floor(ramp[0].astype(np.float32) * np.float32(255.0 / 900.0)), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(_u8(w)[0], expect)
    assert np.unique(_u8(w)[0][ramp[0] < 900]).size == 255
    sent = bytes(w.image_to_js)
    labels_sent = bytes(w.labels_to_js)
    w.vmin = [1.0]
    w.vmin = [0.0]
    assert bytes(w.labels_to_js) == labels_sent, "a range change re-sends only the image"
    w.gamma = [0.5]
    assert bytes(w.image_to_js) == sent, "gamma is applied in the browser"

    dark = np.full((2, 6, 7), 10, np.uint16)
    dark[0, 0, 0] = 20
    w2 = LabelAnnotator([ramp, dark], axes="ZYX")
    first = (list(w2.vmin), list(w2.vmax))
    w2.vmax = [1234.0]
    w2.stack = 1
    assert (w2.vmin, w2.vmax) == ([0.0], [20.0])
    w2.stack = 0
    assert (w2.vmin, w2.vmax) == (first[0], [1234.0])


def test_list_of_dataarrays_keeps_names_colours_and_pixel_sizes():
    """xarray input goes through the viewers' unpacking, also inside a list of stacks (D18, R4-13).

    Strategy: a 2-channel DataArray with names, a ``color`` coordinate with
    one empty entry, and calibrated coordinates (Z 2 µm, Y 0.5 µm, X 0.25
    µm); pass it alone and as a list of two stacks.
    Expected outcome: in both cases the names come from the ``C``
    coordinate, the given colour is kept and only the empty entry takes a
    palette colour (a list used to fall back to ch0/ch1 and default colours,
    and one empty entry dropped all colours); ``pixel_sizes`` holds the
    calibrated sizes in µm (they used to be ignored). Explicit sizes win.
    """
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray, set_channel_info

    arr = np.random.default_rng(0).random((2, 3, 6, 7)).astype(np.float32)
    da = array_to_stczyx_xarray(arr, input_axes="CZYX", channel_names=["DAPI", "GFP"],
                                pixel_sizes={"Z": 2.0, "Y": 0.5, "X": 0.25, "unit": "µm"})
    da = set_channel_info(da, {"DAPI": {"color": "blue"}, "GFP": {"color": ""}}, strict=False)
    for w in (LabelAnnotator(da), LabelAnnotator([da, da])):
        assert w.channel_names == ["DAPI", "GFP"]
        assert w.channel_colors[0] == "#0000ff" and w.channel_colors[1] not in ("#0000ff", "")
        assert {k: w.pixel_sizes[k] for k in "ZYX"} == pytest.approx({"Z": 2.0, "Y": 0.5, "X": 0.25})
        assert w.pixel_sizes["unit"] == "µm"
    w = LabelAnnotator(da, pixel_sizes=(1.0, 0.3, 0.3))
    assert (w.pixel_sizes["Y"], w.pixel_sizes["X"]) == (0.3, 0.3)


def test_pixel_sizes_are_parsed_to_um_and_validated():
    """``pixel_sizes=`` uses the package parser: µm out, known units converted, bad input rejected (D01, D18).

    Strategy: give sizes as a (Z, Y, X) tuple, a dict in nm, a scalar, None
    and invalid values.
    Expected outcome: µm values (nm divided by 1000), None where unknown,
    ``ValueError`` for non-positive sizes or a wrong length.
    """
    img = np.zeros((3, 6, 7))
    assert LabelAnnotator(img, axes="ZYX", pixel_sizes=(2, 0.5, 0.25)).pixel_sizes == {
        "Z": 2.0, "Y": 0.5, "X": 0.25, "unit": "µm"}
    ps = LabelAnnotator(img, axes="ZYX", pixel_sizes={"Y": 500, "X": 250, "unit": "nm"}).pixel_sizes
    assert ps == {"Z": None, "Y": 0.5, "X": 0.25, "unit": "µm"}
    assert LabelAnnotator(img, axes="ZYX", pixel_sizes=0.2).pixel_sizes["X"] == 0.2
    assert LabelAnnotator(img, axes="ZYX").pixel_sizes == {"Z": None, "Y": None, "X": None, "unit": "µm"}
    for bad in (-1, (1, 2), {"X": 0}):
        with pytest.raises(ValueError):
            LabelAnnotator(img, axes="ZYX", pixel_sizes=bad)


def test_save_writes_axes_and_pixel_sizes_that_load_image_reads_back(tmp_path):
    """Saved training pairs carry axes and calibration, so ``image_io.load_image`` reads them back (R8-10).

    Strategy: save a 2-channel 3D widget with ``pixel_sizes`` (Y 0.5 µm,
    X 0.25 µm) and a 1-channel 2D widget without sizes; read every file
    with ``load_image`` and with plain ``tifffile.imread`` (what Cellpose
    uses).
    Expected outcome: images come back as C/Y/X with the original values,
    masks as Y/X; the calibrated pair reports Y 0.5 and X 0.25 µm, the
    uncalibrated one None (before, ``load_image`` raised on the files' 'QYX'
    axes). ``tifffile.imread`` still returns the plain arrays.
    """
    pytest.importorskip("xarray")
    tifffile = pytest.importorskip("tifffile")
    from eigenp_utils.image_io import get_stczyx_pixel_sizes, load_image

    img = np.random.default_rng(0).integers(0, 4000, (2, 3, 6, 7)).astype(np.uint16)
    lab = np.zeros((3, 6, 7), np.uint32)
    lab[1, 2:4, 2:5] = 3
    w = LabelAnnotator(img, axes="CZYX", labels=lab, pixel_sizes={"Z": 2, "Y": 0.5, "X": 0.25})
    pi, pm = w.save(tmp_path, "cal")
    da = load_image(pi)
    assert da.sizes["C"] == 2 and da.sizes["Y"] == 6 and da.sizes["X"] == 7
    np.testing.assert_array_equal(np.asarray(da.isel(S=0, T=0, Z=0).values), img[:, 1])
    ps = get_stczyx_pixel_sizes(da)
    assert ps["Y"] == pytest.approx(0.5, rel=1e-6) and ps["X"] == pytest.approx(0.25, rel=1e-6)
    dm = load_image(pm)
    np.testing.assert_array_equal(np.asarray(dm.squeeze().values), lab[1])
    assert get_stczyx_pixel_sizes(dm)["X"] == pytest.approx(0.25, rel=1e-6)
    np.testing.assert_array_equal(tifffile.imread(pi), img[:, 1])
    np.testing.assert_array_equal(tifffile.imread(pm), lab[1])

    w2 = LabelAnnotator(img[0, 0], labels=lab[1])
    pi2, pm2 = w2.save(tmp_path, "raw")
    d2 = load_image(pi2)
    assert d2.sizes["Y"] == 6 and get_stczyx_pixel_sizes(d2)["X"] is None
    np.testing.assert_array_equal(np.asarray(load_image(pm2).squeeze().values), lab[1])


def test_2d_image_takes_a_yx_pixel_size_pair():
    """A 2D image accepts ``pixel_sizes=(Y, X)``; a 3D one still needs Z, Y, X (D18, review fix).

    Strategy: give a (Y, X) pair to a CYX and a YX widget, a ZYX triple to
    the CYX widget, and the same pair to a CZYX widget.
    Expected outcome: (+) the 2D widgets report Y 0.5, X 0.25 µm and Z None
    (the pair used to raise 'expected 3 (Z, Y, X)'); a triple still works on
    2D; (−) the pair on a 3D image still raises ``ValueError``.
    """
    img = np.zeros((2, 6, 7), np.float32)
    want = {"Z": None, "Y": 0.5, "X": 0.25, "unit": "µm"}
    assert LabelAnnotator(img, axes="CYX", pixel_sizes=(0.5, 0.25)).pixel_sizes == want
    assert LabelAnnotator(img[0], axes="YX", pixel_sizes=[0.5, 0.25]).pixel_sizes == want
    ps = LabelAnnotator(img, axes="CYX", pixel_sizes=(2.0, 0.5, 0.25)).pixel_sizes
    assert (ps["Y"], ps["X"]) == (0.5, 0.25)
    with pytest.raises(ValueError, match="expected 3"):
        LabelAnnotator(np.zeros((2, 3, 6, 7)), axes="CZYX", pixel_sizes=(0.5, 0.25))


def test_malformed_trait_kwargs_are_rejected():
    """Constructor trait kwargs reject malformed values instead of coercing them (D18/R4-08, review fix).

    Strategy: pass an out-of-range ``opacity``, an unknown ``tool``, a string
    for ``channel_visible`` / ``instance_palette`` / ``class_colors``, a bool
    for ``vmin`` / ``vmax`` / ``gamma``, and ``auto_increment=True`` in
    semantic mode; then valid values of the same kwargs.
    Expected outcome: (−) each malformed value raises ``ValueError`` or
    ``TraitError`` (before, ``opacity=5`` and ``tool='nope'`` were kept,
    ``channel_visible='no'`` became all True, ``instance_palette='red'``
    failed on ``'e'``, ``vmin=True`` became 1.0 and ``auto_increment`` was
    silently dropped); (+) valid values are kept, and setting ``opacity``
    / ``tool`` later is validated the same way.
    """
    import traitlets

    v = np.zeros((2, 6, 7), np.float32)
    bad = [dict(opacity=5), dict(opacity=-0.1), dict(tool="nope"), dict(channel_visible="no"),
           dict(instance_palette="red"), dict(mode="semantic", classes=["red"], class_colors="red"),
           dict(vmin=True), dict(vmax=[np.bool_(True)]), dict(gamma=True),
           dict(mode="semantic", auto_increment=True)]
    for kw in bad:
        with pytest.raises((ValueError, traitlets.TraitError)):
            LabelAnnotator(v, axes="CYX", **kw)
    w = LabelAnnotator(v, axes="CYX", opacity=1.0, tool="fill", channel_visible=(True, False),
                       instance_palette=("red",), vmin=0, vmax=[2.5, 3])
    assert (w.opacity, w.tool, w.channel_visible, w.instance_palette) == (1.0, "fill", [True, False], ["#ff0000"])
    assert (w.vmin, w.vmax) == ([0.0, 0.0], [2.5, 3.0])
    assert LabelAnnotator(v, axes="CYX", mode="semantic", auto_increment=False).auto_increment is False
    for name, val in (("opacity", 2.0), ("tool", "pen")):
        with pytest.raises(traitlets.TraitError):
            setattr(w, name, val)


def test_narrow_window_on_large_offset_data_keeps_256_levels():
    """A narrow data-unit window far from zero is quantised without float32 rounding (R4-07, review fix).

    Strategy: an int64 ramp 20_000_000 + 0..255 shown with vmin/vmax at its
    ends (float32 spacing at 2e7 is 2); a small-offset ramp as control.
    Expected outcome: (+) both 8-bit slices equal ``0..255`` exactly, 256
    levels (the offset one had 129 levels).
    """
    ramp = np.arange(256, dtype=np.int64).reshape(16, 16)
    for off in (20_000_000, 0):
        w = LabelAnnotator(ramp + off, axes="YX", vmin=off, vmax=off + 255)
        np.testing.assert_array_equal(w._slice_u8(0)[0], ramp.astype(np.uint8))


def test_range_change_requantises_only_the_changed_channel():
    """A vmin/vmax change re-reads only the channels whose range changed (R4-07, review fix).

    Strategy: a 3-channel image whose array records which channels are read;
    change ``vmax`` of channel 1 only, then of every channel, then move z.
    Expected outcome: (+) the first change reads channel 1 only (all three
    were re-quantised before) and the slice equals a fresh full
    quantisation; (−) a change of every channel, or a new z, reads them all.
    """
    class Spy(np.ndarray):
        reads: list = []

        def __getitem__(self, key):
            if isinstance(key, tuple) and len(key) == 2:
                Spy.reads.append(int(key[0]))
            return np.asarray(super().__getitem__(key))

    img = np.random.default_rng(0).integers(0, 4000, (3, 4, 8, 9)).astype(np.uint16)
    w = LabelAnnotator(img, axes="CZYX")
    w._image = w._image.view(Spy)
    w._slice_u8(w.z)  # fill the cache for this image
    Spy.reads.clear()
    w.vmax = [w.vmax[0], 1000.0, w.vmax[2]]
    assert Spy.reads == [1]
    fresh = LabelAnnotator(img, axes="CZYX", vmax=w.vmax)._slice_u8(w.z)
    np.testing.assert_array_equal(w._slice_u8(w.z), fresh)
    Spy.reads.clear()
    w.vmax = [500.0, 600.0, 700.0]
    assert sorted(Spy.reads) == [0, 1, 2]
    Spy.reads.clear()
    w.z = (w.z + 1) % 4
    assert set(Spy.reads) == {0, 1, 2}
