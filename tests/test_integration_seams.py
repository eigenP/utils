"""Seam tests (C-09): one module's output fed straight into another.

This file is an explicit exemption from the 1:1 test-mirroring rule
(docs/AGENTS.md): each test crosses at least two modules, e.g. ``image_io`` ->
``tnia_plotting_anywidgets`` or ``maxproj_registration`` -> itself through the
public chain. The per-module behaviour is tested in the mirrored files; these
tests only check that the hand-over (units, axis order, dtype, channel layout)
is consistent.

Pyodide-safe: no subprocesses; optional packages are guarded with
``pytest.importorskip``.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import pytest

from eigenp_utils.tnia_plotting_anywidgets import (
    TNIASliceWidget,
    show_zyx_max,
    show_zyx_max_scatter_interactive,
    show_zyx_max_slice_interactive,
    show_zyx_slice,
)

_UNITS = ("nm", "µm", "mm", "px")


def _scale_bar_label(fig):
    """Text of the scale-bar label on the XY panel (None when there is none)."""
    labels = [t.get_text() for t in fig.axXY.texts if t.get_text().split(" ")[-1] in _UNITS]
    return labels[0] if labels else None


def _write_cm_tiff(path, vol):
    """ImageJ TIFF calibrated in cm: 0.5 µm pixels (2e4 px/cm) and 2 µm Z steps."""
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(
        path, vol, imagej=True, resolution=(2e4, 2e4),
        metadata={"axes": "ZYX", "unit": "cm", "spacing": 2e-4},
    )


# ---------------------------------------------------------------------------
# image_io.load_image -> viewers
# ---------------------------------------------------------------------------

def test_cm_calibrated_tiff_through_load_image_into_static_and_slice_viewers(tmp_path):
    """A cm-calibrated TIFF reaches ``show_zyx_max`` and ``TNIASliceWidget`` in µm.

    Strategy: write a (10, 40, 60) ImageJ stack calibrated in cm (0.5 µm
    pixels, 2 µm Z steps), read it with ``image_io.load_image`` and pass the
    DataArray straight to ``show_zyx_max`` and ``TNIASliceWidget``.

    Expected outcome: ``load_image`` reports µm sizes; both viewers draw the XY
    panel 30 x 20 µm and the XZ panel 20 µm tall (aspect from the physical
    sizes, not the voxel counts); the widget's ``sz, sy, sx`` are the µm sizes;
    the scale bar is labelled in an automatically chosen length unit (µm
    here), not 'px' and not cm.
    """
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import get_stczyx_pixel_sizes, load_image

    vol = np.random.default_rng(0).integers(0, 1000, (10, 40, 60)).astype(np.uint16)
    p = tmp_path / "cm.tif"
    _write_cm_tiff(p, vol)
    da = load_image(p)
    ps = get_stczyx_pixel_sizes(da)
    assert ps["unit"] == "µm"
    assert (ps["Z"], ps["Y"], ps["X"]) == pytest.approx((2.0, 0.5, 0.5))

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # fully calibrated: no fallback warnings
        fig = show_zyx_max(da)
    try:
        assert abs(np.diff(fig.axXY.get_xlim())[0]) == pytest.approx(30.0)
        assert abs(np.diff(fig.axXY.get_ylim())[0]) == pytest.approx(20.0)
        assert abs(np.diff(fig.axXZ.get_ylim())[0]) == pytest.approx(20.0)
        assert fig.axXZ.get_position().height / fig.axXY.get_position().height == pytest.approx(1.0, rel=1e-3)
        assert _scale_bar_label(fig).endswith(" µm")
    finally:
        plt.close(fig)

    w = TNIASliceWidget(da)
    assert (w.sz, w.sy, w.sx) == pytest.approx((2.0, 0.5, 0.5))
    fig = w._render()
    try:
        assert abs(np.diff(fig.axXY.get_xlim())[0]) == pytest.approx(30.0)
        assert abs(np.diff(fig.axXZ.get_ylim())[0]) == pytest.approx(20.0)
        assert _scale_bar_label(fig).endswith(" µm")
    finally:
        plt.close(fig)


def test_scale_bar_unit_follows_magnitude_after_load_image(tmp_path):
    """The scale-bar unit is picked by magnitude for files at EM and tissue scales.

    Strategy: write (4, 256, 256) ImageJ stacks in nm (2 nm pixels) and in
    cm (40 µm pixels), load them with ``load_image`` and show them.
    Expected outcome: the EM file is labelled in nm and the tissue file in mm,
    while ``load_image`` reports both in µm (only the label changes).
    """
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import get_stczyx_pixel_sizes, load_image

    vol = np.zeros((4, 256, 256), np.uint8)
    vol[:, 100:150, 100:150] = 200
    cases = (("em.tif", 1 / 2e-7, "cm", 2e-7, "nm", 0.002),  # 2 nm pixels
             ("tissue.tif", 1 / 4e-3, "cm", 4e-3, "mm", 40.0))  # 40 µm pixels
    for name, res, unit, spacing, bar_unit, um in cases:
        p = tmp_path / name
        tifffile.imwrite(p, vol, imagej=True, resolution=(res, res),
                         metadata={"axes": "ZYX", "unit": unit, "spacing": spacing})
        da = load_image(p)
        assert get_stczyx_pixel_sizes(da)["X"] == pytest.approx(um, rel=1e-6)
        fig = show_zyx_max(da)
        try:
            assert _scale_bar_label(fig).endswith(" " + bar_unit), name
        finally:
            plt.close(fig)


def test_uncalibrated_tiff_through_load_image_gets_px_scale_bar(tmp_path):
    """An uncalibrated TIFF read by ``load_image`` gives a 'px' scale bar in every viewer.

    Strategy: write a plain (5, 40, 60) TIFF, load it, show it with
    ``show_zyx_max``, ``show_zyx_slice`` and ``TNIASliceWidget``.
    Expected outcome: every label ends with ' px'; the panels use 1.0 per voxel.
    """
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import get_stczyx_pixel_sizes, load_image

    p = tmp_path / "plain.tif"
    vol = np.random.default_rng(1).integers(0, 255, (5, 40, 60)).astype(np.uint8)
    tifffile.imwrite(p, vol)
    da = load_image(p, input_axes="ZYX")
    ps = get_stczyx_pixel_sizes(da)
    assert ps["X"] is None and ps["Y"] is None

    figs = [show_zyx_max(da), show_zyx_slice(da, 30, 20, 2), TNIASliceWidget(da)._render()]
    try:
        for fig in figs:
            assert _scale_bar_label(fig).endswith(" px"), _scale_bar_label(fig)
            assert abs(np.diff(fig.axXY.get_xlim())[0]) == pytest.approx(60.0)
    finally:
        for fig in figs:
            plt.close(fig)


# ---------------------------------------------------------------------------
# maxproj_registration: estimate_drift -> apply_subpixel_drift_correction
# ---------------------------------------------------------------------------

def _blob_scene(shape, n, seed):
    """Sparse Gaussian spots on a dark offset, like a fluorescence frame."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    a = np.zeros(shape)
    for _ in range(n):
        a[tuple(rng.integers(8, s - 8) for s in shape)] = rng.uniform(0.5, 1.0)
    return gaussian_filter(a, 2.5) * 1000 + 10


@pytest.mark.parametrize("ndim", [2, 3])
def test_estimate_drift_zyx_chains_into_apply_subpixel_drift_correction(ndim):
    """``estimate_drift(order='zyx')`` output is the shift ``apply_subpixel_drift_correction`` takes.

    Strategy: move a spot scene by 6 random subpixel motions (|d| <= 4 px; 2D
    and 3D), estimate the correction with ``order='zyx'``, apply it, and
    re-estimate the drift between the reference and the corrected frame.
    Expected outcome: the correction is the negative of the motion (no axis
    swap, no sign flip) and the RMS residual drift after applying it is
    <= 0.1 px. Negative control (2D): the legacy 'xy' order fed into the same
    chain swaps the axes and leaves a residual > 0.5 px.
    """
    from scipy.ndimage import shift as nd_shift

    from eigenp_utils.maxproj_registration import apply_subpixel_drift_correction, estimate_drift

    rng = np.random.default_rng(1)
    shape = (64, 64) if ndim == 2 else (32, 64, 64)
    residuals, wrong = [], []
    for k in range(6):
        motion = rng.uniform(-4, 4, ndim)
        ref = _blob_scene(shape, 12, k)
        moved = nd_shift(ref, motion, order=3, mode="nearest")
        corr = np.asarray(estimate_drift(ref, moved, order="zyx"))
        np.testing.assert_allclose(corr, -motion, atol=0.1)
        fixed = apply_subpixel_drift_correction(moved, corr)
        residuals.append(estimate_drift(ref, fixed, order="zyx"))
        if ndim == 2:
            swapped = apply_subpixel_drift_correction(moved, estimate_drift(ref, moved, order="xy"))
            wrong.append(estimate_drift(ref, swapped, order="zyx"))
    assert float(np.sqrt(np.mean(np.square(residuals)))) <= 0.1
    if ndim == 2:
        assert float(np.sqrt(np.mean(np.square(wrong)))) > 0.5


# ---------------------------------------------------------------------------
# plotting_utils: color_coded_projection -> describe_image
# ---------------------------------------------------------------------------

def test_color_coded_projection_is_shown_as_rgb_by_describe_image():
    """The RGB output of ``color_coded_projection`` is drawn as colour by ``describe_image``.

    Strategy: a (5, 32, 48) stack with one bright square per frame; project
    it and pass the result to ``describe_image``.
    Expected outcome: the projection is (32, 48, 3) float in [0, 1]; the shown
    image is the full RGB array (not a 3-pixel-wide slice of its leading
    axis), drawn with row 0 at the top; the info table reports the (32, 48, 3)
    shape; the histogram has one line per colour channel.
    """
    from eigenp_utils.plotting_utils import color_coded_projection, describe_image

    stack = np.zeros((5, 32, 48), np.float32)
    for t in range(5):
        stack[t, 2:8, 2 + 9 * t:8 + 9 * t] = 1.0
    rgb = color_coded_projection(stack)
    assert rgb.shape == (32, 48, 3)
    assert np.nanmin(rgb) >= 0 and np.nanmax(rgb) <= 1

    res = describe_image(rgb)
    try:
        shown = res["axes"]["Image"].images[0]
        np.testing.assert_allclose(np.asarray(shown.get_array(), dtype=float)[..., :3], rgb, atol=1e-6)
        assert shown.origin == "upper"
        assert "32" in res["info"] and "48" in res["info"]
        assert len(res["axes"]["Histogram"].lines) + len(res["axes"]["Histogram"].patches) >= 3
        # The squares sit in the top rows of the displayed image, with different colours
        top = np.asarray(shown.get_array(), dtype=float)[4, 4::9][:5, :3]
        assert len({tuple(np.round(c, 3)) for c in top}) == 5
    finally:
        plt.close(res["fig"])


# ---------------------------------------------------------------------------
# A bool mask through every viewer
# ---------------------------------------------------------------------------

def _bool_mask():
    m = np.zeros((6, 24, 32), dtype=bool)
    m[2:4, 8:16, 10:20] = True
    return m


def test_bool_mask_opens_in_every_viewer():
    """A bool (Z, Y, X) mask opens in the static, slice, scatter and LabelAnnotator viewers.

    Strategy: show the same mask with ``show_zyx_max``, ``show_zyx_slice``,
    ``show_zyx_max_slice_interactive``, ``show_zyx_max_scatter_interactive``
    (the mask's voxels as points, coloured by a bool per-point channel) and
    ``LabelAnnotator``.
    Expected outcome: nothing raises; the static and slice renderings are
    bright inside the mask and black outside; the scatter shows the mask's
    points (one per voxel, split by category); LabelAnnotator keeps the
    mask's shape.
    """
    mask = _bool_mask()
    figs = []
    try:
        figs.append(show_zyx_max(mask))
        figs.append(show_zyx_slice(mask, 15, 12, 3))
        w = show_zyx_max_slice_interactive(mask)
        figs.append(w._render())
        for fig in figs:
            xy = np.asarray(fig.axXY.images[0].get_array(), dtype=float)
            xy = xy[..., :3].max(axis=-1) if xy.ndim == 3 else xy
            assert xy.shape == mask.shape[1:]
            # Bright exactly where the mask is, black everywhere else
            # (a transposed or shifted rendering fails here).
            assert xy[8:16, 10:20].min() > 0
            outside = np.ones(xy.shape, dtype=bool)
            outside[8:16, 10:20] = False
            assert xy[outside].max() == 0

        z, y, x = np.nonzero(mask)
        inside = (x < 15)
        ws = show_zyx_max_scatter_interactive((z, y, x), channels=inside, channel_mode="categories")
        fig = ws._render()
        figs.append(fig)
        assert len(ws.category_labels) == 2
        # Every mask voxel is drawn as a point, split 80/80 between categories.
        counts = sorted(len(c.get_offsets()) for c in fig.axXY.collections
                        if type(c).__name__ == "PathCollection")
        assert counts == sorted([int(inside.sum()), int((~inside).sum())])
        assert sum(counts) == int(mask.sum())
    finally:
        for fig in figs:
            plt.close(fig)

    from eigenp_utils.label_annotator import LabelAnnotator

    la = LabelAnnotator(mask, axes="ZYX")
    assert la.labels.shape == mask.shape


# ---------------------------------------------------------------------------
# load_image (STCZYX) -> to_zyx -> processing
# ---------------------------------------------------------------------------

def test_load_image_to_zyx_feeds_processing_functions():
    """``load_image`` -> ``to_zyx`` gives the (Z, Y, X) array and pixel sizes processing takes.

    Strategy: build a 2-channel STCZYX DataArray with ``load_image`` from an
    in-memory CZYX array with µm pixel sizes; select one channel with
    ``to_zyx`` and run ``clahe_u8`` and ``extract_surface`` on it (passing
    the pixel sizes on). Negative control: the 6-D DataArray itself is
    rejected with a TypeError naming ``to_zyx``.
    Expected outcome: ``to_zyx`` returns a plain (Z, Y, X) ndarray and a µm
    pixel-size dict (unit None or 'µm'); ``clahe_u8`` returns uint8 of the same shape; and
    ``extract_surface`` returns a height map over (Y, X).
    """
    pytest.importorskip("xarray")
    pytest.importorskip("skimage")
    from eigenp_utils.clahe_equalize_adapthist import clahe_u8
    from eigenp_utils.image_io import load_image, to_zyx
    from eigenp_utils.surface_extraction import extract_surface

    rng = np.random.default_rng(5)
    zz = np.arange(16)[:, None, None]
    tissue = (zz >= 6).astype(np.float32) * np.ones((16, 32, 32), np.float32)
    data = np.stack([tissue * 200 + rng.random((16, 32, 32)) * 10,
                     rng.random((16, 32, 32)) * 50]).astype(np.uint16)
    da = load_image(data, input_axes="CZYX", channel_names=["mem", "noise"],
                    pixel_sizes={"Z": 2.0, "Y": 0.5, "X": 0.5})
    assert da.dims == ("S", "T", "C", "Z", "Y", "X")

    arr, ps = to_zyx(da, C="mem")
    assert isinstance(arr, np.ndarray) and arr.shape == (16, 32, 32)
    # Unit-less sizes stay unit None, which means µm by the package convention
    assert (ps["Z"], ps["Y"], ps["X"]) == (2.0, 0.5, 0.5) and ps["unit"] in (None, "µm")
    np.testing.assert_array_equal(arr, data[0])

    out = clahe_u8(arr)
    assert out.dtype == np.uint8 and out.shape == arr.shape

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        height = extract_surface(arr, downscale_factor=1, pixel_sizes=ps, gaussian_sigma_um=(2.0, 2.0, 2.0))
    height = height[0] if isinstance(height, tuple) else height
    assert np.asarray(height).shape[-2:] == (32, 32)

    with pytest.raises(TypeError, match="to_zyx"):
        clahe_u8(da)


# ---------------------------------------------------------------------------
# LabelAnnotator.save -> load_image
# ---------------------------------------------------------------------------

def test_label_annotator_save_round_trips_through_load_image(tmp_path):
    """Pairs written by ``LabelAnnotator.save`` load back with ``load_image`` with data and µm sizes.

    Strategy: open the STCZYX DataArray that ``load_image`` makes from a
    2-channel (C, Z, Y, X) array with 0.25 µm pixels in ``LabelAnnotator``, paint labels on slice 1 via
    ``set_labels``, save, and read both files back with ``load_image``.
    Expected outcome: the image file holds the original (unscaled) channel
    data of the slice, the mask file the labels (ids unchanged), and both
    carry 0.25 µm X/Y pixel sizes.
    """
    pytest.importorskip("tifffile")
    pytest.importorskip("xarray")
    from eigenp_utils.image_io import get_stczyx_pixel_sizes, load_image
    from eigenp_utils.label_annotator import LabelAnnotator

    rng = np.random.default_rng(7)
    data = rng.integers(0, 4000, (2, 3, 20, 24)).astype(np.uint16)
    src = load_image(data, input_axes="CZYX", pixel_sizes={"Z": 1.0, "Y": 0.25, "X": 0.25})
    la = LabelAnnotator(src)  # the STCZYX DataArray itself: dims and µm sizes from its coordinates
    assert la.pixel_sizes == {"Z": 1.0, "Y": 0.25, "X": 0.25, "unit": "µm"}
    lab = np.zeros((3, 20, 24), np.uint32)
    lab[1, 4:10, 5:12] = 1
    lab[1, 12:18, 14:20] = 300
    la.set_labels(lab)
    paths = la.save(tmp_path, "pair")
    img_p = [p for p in paths if str(p).endswith("_img.tif")][0]
    msk_p = [p for p in paths if str(p).endswith("_masks.tif")][0]

    di = load_image(img_p)
    dm = load_image(msk_p)
    np.testing.assert_array_equal(np.asarray(di.squeeze().values), data[:, 1])
    np.testing.assert_array_equal(np.asarray(dm.squeeze().values), lab[1])
    for d in (di, dm):
        ps = get_stczyx_pixel_sizes(d)
        assert (ps["Y"], ps["X"]) == pytest.approx((0.25, 0.25))
