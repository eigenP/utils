"""LabelAnnotator: Python-side invariants (no browser needed; also runs in Pyodide)."""

import numpy as np
import pytest

pytest.importorskip("anywidget")

from eigenp_utils.label_annotator import LabelAnnotator, decode_label_slice, encode_label_slice  # noqa: E402
from eigenp_utils.label_annotator import _IMG_HDR, LABELS_CMAP_COLORS  # noqa: E402


def _img_header(w):
    return _IMG_HDR.unpack_from(bytes(w.image_to_js), 0)


def test_label_wire_roundtrip():
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 65535, size=(17, 23)).astype(np.uint32)
    z, back = decode_label_slice(encode_label_slice(lab, 5))
    assert z == 5 and back.dtype == np.uint16
    np.testing.assert_array_equal(back, lab)


def test_label_wire_rejects_overflow():
    with pytest.raises(ValueError):
        encode_label_slice(np.full((2, 2), 70_000, dtype=np.uint32), 0)


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
    img = np.random.default_rng(1).random(shape).astype(np.float32)
    w = LabelAnnotator(img, axes=axes)
    assert w.labels.shape == spatial and w.labels.dtype == np.uint32
    z, c, y, x = _img_header(w)
    n_c = shape[axes.index("C")] if "C" in axes else 1
    assert (c, y, x) == (n_c, spatial[-2], spatial[-1])
    assert len(w.channel_names) == n_c


def test_axes_required_for_nd():
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


def test_semantic_mode_does_not_auto_increment():
    w = LabelAnnotator(np.random.default_rng(4).random((6, 7)), mode="semantic", classes=["mito", "perox", "cytosol"])
    assert w.class_names == ["mito", "perox", "cytosol"] and len(w.class_colors) == 3
    assert not w.auto_fill and w.active_label == 1
    edit = np.zeros((6, 7), dtype=np.uint16)
    edit[0, :3] = 2
    w.labels_from_js = encode_label_slice(edit, 0)
    assert w.active_label == 1 and (w.labels == edit).all()


def test_initial_labels_validation():
    img = np.zeros((6, 7))
    with pytest.raises(ValueError):
        LabelAnnotator(img, labels=np.zeros((5, 7)))
    with pytest.raises(ValueError):
        LabelAnnotator(img, labels=np.full((6, 7), 70_000))
    w = LabelAnnotator(img, labels=np.full((6, 7), 3))
    assert w.active_label == 4


def test_save_writes_unscaled_pairs_for_annotated_slices(tmp_path):
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


def test_xarray_stczyx_input():
    xr = pytest.importorskip("xarray")
    from eigenp_utils.image_io import array_to_stczyx_xarray

    arr = np.random.default_rng(6).random((2, 4, 10, 11)).astype(np.float32)  # CZYX
    da = array_to_stczyx_xarray(arr, input_axes="CZYX")
    w = LabelAnnotator(da)
    assert isinstance(da, xr.DataArray) and w.labels.shape == (4, 10, 11)


def test_instance_palette_is_labels_cmap():
    w = LabelAnnotator(np.zeros((4, 4)))
    assert len(w.instance_palette) == len(LABELS_CMAP_COLORS)
    r, g, b = LABELS_CMAP_COLORS[0]
    assert w.instance_palette[0] == "#{:02x}{:02x}{:02x}".format(round(255 * r), round(255 * g), round(255 * b))
