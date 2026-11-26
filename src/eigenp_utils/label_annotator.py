# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anywidget",
#     "traitlets",
#     "numpy",
#     "matplotlib",  # colour names -> hex
#     "tifffile",  # only for LabelAnnotator.save
# ]
# ///
"""Lightweight in-notebook label annotator (anywidget; Jupyter and marimo).

Paint instance masks (e.g. Cellpose training masks) or semantic class
scribbles (e.g. pixel-classifier training labels) on 2D slices of a
multichannel image, without napari.

Design
------
- The browser does all compositing and painting. Python sends the current
  z-slice once as 8-bit channels (display only) plus its uint16 labels; the
  browser sends the edited slice labels back after every stroke. Python keeps
  the authoritative label volume (stored as ``uint16``, returned as ``uint32``).
- Wire format (little-endian):
  image : ``uint32[4] = (z, C, Y, X)`` header, then ``C*Y*X`` ``uint8``
  labels: ``uint32[3] = (z, Y, X)`` header, then ``Y*X`` ``uint16``
  Labels > 65535 cannot be edited in the widget (checked on input).
- Display scaling (8-bit conversion) is for viewing only; exported images are
  the original, unscaled array.
- z-linking (instance mode): one id across z = one 3D object. Python also
  sends the labels of slices z-1 / z+1 (``labels_prev_to_js`` /
  ``labels_next_to_js``, same encoding, empty when out of range or when
  ``show_adjacent`` is off) for the onion-skin outlines. Whole-track edits
  (``merge_ids``, ``split_id``) run in Python on the full volume, requested by
  the browser through ``track_request``; they keep their own undo stack
  (``track_undo_ids`` mirrors it so the browser's undo can route to it).
  Deleting a whole mask (``delete_id``) is a track operation too.
- Annotated background (``background=True``, instance mode): the reserved id
  ``BACKGROUND_ID`` (65535, the largest wire id) marks pixels annotated as
  "not a cell", so a label volume has three states: 0 unannotated, 65535
  background, 1..65534 cells. ``labels_signed`` / ``save(signed=True)`` give
  the signed convention (-1 background) used for sparse Cellpose training.

Only numpy, anywidget and traitlets are required (core dependencies; also
works in marimo WASM). ``tifffile`` is imported lazily by ``save``.
"""

from __future__ import annotations

import colorsys
import re
import struct
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import anywidget
import numpy as np
import traitlets

from ._conventions import (
    PALETTE,
    UM,
    _is_single_colour,
    _sample_and_extremes,
    channel_color_hex,
    default_colors,
    default_display_range,
    per_channel,
    resolve_pixel_sizes,
    robust_range,
)
from .labels_cmap_data import LABELS_CMAP_COLORS

_ASSETS = Path(__file__).parent
_IMG_HDR = struct.Struct("<4I")
_LBL_HDR = struct.Struct("<3I")
_MAX_WIRE_LABEL = np.iinfo(np.uint16).max
BACKGROUND_ID = _MAX_WIRE_LABEL  # reserved id for annotated background (background=True)
_TRACK_UNDO_MAX = 20  # whole-volume operations kept for undo
# ... and at most this many bytes of them (the newest one is always kept)
_TRACK_UNDO_MAX_BYTES = 512 * 2**20
# internal label storage: ids are at most 65535 (the uint16 wire), so uint16
# halves the memory of uint32; the public getters still return uint32 (R4-19)
_STORE = np.uint16

# The package palette (white, lime, magenta, ...), shared with the ortho viewers
# and cycled for more channels (D03). It used to start with magenta and fail
# above 8 channels.
_DEFAULT_CHANNEL_COLORS = PALETTE


def _class_palette(n: int) -> list[str]:
    """Distinct colours for semantic classes (golden-ratio hue steps)."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((0.11 + i * 0.618034) % 1.0, 0.75, 1.0)
        out.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return out


def _color_hex(c) -> str:
    """Hex colour for the browser from a colour or colormap name ('gray' -> its end colour, white).

    The package's one resolver (``_conventions.channel_color_hex``): colormap
    names win over CSS names, RGB(A) tuples and registered Colormap instances
    work, an unregistered instance raises TypeError, other values ValueError.
    """
    return channel_color_hex(c)


def _channel_colors(given, c: int) -> list[str]:
    """``c`` hex colours: ``given`` (None, or up to ``c`` entries) padded with unused palette colours, cycled."""
    if given is None:
        given = []
    elif _is_single_colour(given):
        given = [given]
    else:
        given = list(given)
    if len(given) > c:
        raise ValueError(f"need {c} channel names/colors, got {len(given)} colors")
    # the viewers' rule (_conventions.default_colors): unused palette colours, compared by resolved colour
    return [channel_color_hex(s) for s in default_colors(given, c)]


def _is_dataarray(x) -> bool:
    return hasattr(x, "dims") and hasattr(x, "isel") and hasattr(x, "coords")


def _unpack(image, axes, colors, names, pixel_sizes):
    """One stack as ``(CZYX view, input axes, colors, names, pixel_sizes)``.

    A DataArray goes through the ortho viewers' ``_viewer_input_from_xarray``
    (one rule for both: other dims must have size 1; names, colours and pixel
    sizes come from the coordinates unless given; a missing colour entry
    takes a palette colour without dropping the others).
    """
    if _is_dataarray(image):
        # imported lazily: only DataArray input needs it (xarray is then installed)
        from .tnia_plotting_anywidgets import _viewer_input_from_xarray

        out, colors, names, pixel_sizes = _viewer_input_from_xarray(image, colors, names, pixel_sizes)
        spatial = "".join(d for d in "ZYX" if d in image.dims)
        if isinstance(out, list):
            image, axes = np.stack(out), "C" + spatial
        else:
            image, axes = out, spatial
    czyx, input_axes = _to_czyx(image, axes)
    return czyx, input_axes, colors, names, pixel_sizes


def _parse_pixel_sizes(ps) -> dict:
    """``{'Z', 'Y', 'X', 'unit'}`` in µm (None where unknown), by the package's pixel-size parser."""
    if ps is None:
        return {"Z": None, "Y": None, "X": None, "unit": UM}
    values, _, unit = resolve_pixel_sizes(ps, 3, fill_z=False, warn=False, stacklevel=4)
    return {"Z": values[0], "Y": values[1], "X": values[2], "unit": unit}


def _channel_ranges(czyx: np.ndarray, percentiles, vmin, vmax):
    """Per channel: the automatic range, the start range and the slider range.

    Returns ``(auto, vmin, vmax, data_range)``: ``auto`` the default display
    limits (data units), ``vmin``/``vmax`` the given values (None entries take
    ``auto``), and ``data_range`` ``[lo, hi, step, auto_lo, auto_hi]`` for the
    browser's sliders (the finite data extent, widened to hold the range).
    """
    c = czyx.shape[0]
    auto = [_display_limits(czyx[i], percentiles) for i in range(c)]
    lo_in, hi_in = per_channel(vmin, c, "vmin"), per_channel(vmax, c, "vmax")
    vmin_out, vmax_out, ranges = [], [], []
    integer = czyx.dtype.kind in "iub"
    for i in range(c):
        a_lo, a_hi = auto[i]
        lo = a_lo if lo_in[i] is None else _finite(lo_in[i], "vmin")
        hi = a_hi if hi_in[i] is None else _finite(hi_in[i], "vmax")
        if hi <= lo:
            raise ValueError(f"channel {i}: vmax ({hi}) must exceed vmin ({lo})")
        _, mn, mx = _sample_and_extremes(czyx[i], n=1)
        vmin_out.append(lo)
        vmax_out.append(hi)
        ranges.append(_slider_range(mn, mx, lo, hi, a_lo, a_hi, integer))
    return auto, vmin_out, vmax_out, ranges


def _slider_range(mn, mx, lo, hi, a_lo, a_hi, integer):
    s_lo = min(v for v in (mn, lo, a_lo) if v is not None)
    s_hi = max(v for v in (mx, hi, a_hi) if v is not None)
    span = s_hi - s_lo
    step = 1.0 if integer and span <= 100_000 else (span / 1000.0 if span > 0 else 1.0)
    return [float(s_lo), float(s_hi), float(step), float(a_lo), float(a_hi)]


def _finite(v, name: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {v!r}") from None
    if not np.isfinite(f):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return f


def _plain_color_hex(c) -> str:
    """Hex of a plain matplotlib colour (class and instance colours: no colormaps); ValueError otherwise."""
    from matplotlib.colors import to_hex

    try:
        return to_hex(c)
    except (TypeError, ValueError):
        raise ValueError(f"{c!r} is not a colour") from None


def _rgb_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(255 * float(v))) for v in rgb[:3]))


def _to_czyx(image, axes: str | None) -> tuple[np.ndarray, str]:
    """Return (view with axes C, Z, Y, X; the input's axes string). DataArrays go through :func:`_unpack`."""
    arr = np.asarray(image)
    if axes is None:
        if arr.ndim == 2:
            axes = "YX"
        else:
            raise ValueError("pass axes= (e.g. 'CYX', 'ZYX', 'CZYX') for arrays with ndim > 2")
    axes = axes.upper()
    if len(axes) != arr.ndim or set(axes) - set("CZYX") or len(set(axes)) != len(axes):
        raise ValueError(f"axes={axes!r} does not match an array of shape {arr.shape}")
    if not {"Y", "X"} <= set(axes):
        raise ValueError("axes must contain Y and X")
    full, padded = arr, axes
    for a in "CZ":
        if a not in padded:
            full = full[np.newaxis]
            padded = a + padded
    order = [padded.index(a) for a in "CZYX"]
    return np.transpose(full, order), axes


def _restore_axes(labels_zyx: np.ndarray, spatial_axes: str) -> np.ndarray:
    """Squeeze a (Z, Y, X) label volume to YX when the input has no Z (labels are always (Z)YX)."""
    return labels_zyx[0] if "Z" not in spatial_axes else labels_zyx


def _robust_limits(arr: np.ndarray, lo: float, hi: float, max_samples: int = 2_000_000, seed: int = 0):
    """Display limits from explicit ``(lo, hi)`` percentiles (``percentiles=``), in data units.

    The shared ``_conventions.robust_range`` rule: a fixed-seed sample, and
    when the percentiles coincide on non-constant data (signal sparser than
    (100 - hi)%), the data maximum with a warning (it used to be ``lo + 1``,
    which rendered sparse channels binary). Constant data give ``(v, v + 1)``;
    no finite value gives (0, 1).
    """
    a, b = robust_range(arr, lo, hi, n=max_samples, seed=seed, stacklevel=3)
    if b <= a:
        b = a + 1.0
    return float(a), float(b)


def _display_limits(arr: np.ndarray, percentiles) -> tuple[float, float]:
    """Default display limits of a channel (D02): the package's default display range, or ``percentiles``."""
    if percentiles is None:
        return default_display_range(arr)
    return _robust_limits(arr, *percentiles)


def _coerce_label_values(lab: np.ndarray, background: bool) -> np.ndarray:
    """Validate label values and return them as ``uint16`` (a copy; the internal storage dtype).

    Accepts unsigned ids in ``[0, 65535]``; with ``background`` also negative
    values (signed convention), which become ``BACKGROUND_ID``. Float arrays
    must hold finite integer values (NaN would become id 2**31, and 0.999
    would be floored to 0).
    """
    if lab.size and lab.dtype.kind == "f":
        if not np.isfinite(lab).all():
            raise ValueError("labels must be finite integer ids; got NaN or inf")
        if (lab != np.round(lab)).any():
            raise ValueError("labels must be integer ids; got non-integer float values")
    if lab.size:
        lo, hi = lab.min(), lab.max()
        if hi > _MAX_WIRE_LABEL or (lo < 0 and not background):
            hint = " (negative values mean annotated background: pass background=True)" if lo < 0 else ""
            if hi > _MAX_WIRE_LABEL:
                hint += " (relabel sparse ids first, e.g. skimage.segmentation.relabel_sequential)"
            raise ValueError(f"labels must be in [0, {_MAX_WIRE_LABEL}]{hint}")
        if lo < 0:
            return np.where(lab < 0, BACKGROUND_ID, lab).astype(_STORE)
    return lab.astype(_STORE, copy=True)


def encode_image_slice(slice_cyx_u8: np.ndarray, z: int) -> bytes:
    c, y, x = slice_cyx_u8.shape
    return _IMG_HDR.pack(z, c, y, x) + np.ascontiguousarray(slice_cyx_u8, dtype=np.uint8).tobytes()


def encode_label_slice(labels_yx: np.ndarray, z: int) -> bytes:
    if labels_yx.size and int(labels_yx.max()) > _MAX_WIRE_LABEL:
        raise ValueError(f"labels above {_MAX_WIRE_LABEL} cannot be edited in the widget; relabel first")
    y, x = labels_yx.shape
    return _LBL_HDR.pack(z, y, x) + np.ascontiguousarray(labels_yx, dtype="<u2").tobytes()


def decode_label_slice(buf) -> tuple[int, np.ndarray]:
    data = bytes(buf)
    z, y, x = _LBL_HDR.unpack_from(data, 0)
    body = np.frombuffer(data, dtype="<u2", offset=_LBL_HDR.size)
    if body.size != y * x:
        raise ValueError(f"label message has {body.size} values, expected {y * x}")
    return int(z), body.reshape(y, x)


class LabelAnnotator(anywidget.AnyWidget):
    """Paint labels on slices of a (multichannel) image inside a notebook.

    Parameters
    ----------
    image : array-like, xarray.DataArray or list of these
        2D/3D/4D image. Axes given by ``axes`` (subset of ``"CZYX"`` in any
        order) or by DataArray dims (other dims must have size 1). A DataArray
        is unpacked exactly as by the ortho viewers
        (``tnia_plotting_anywidgets``): channel names default to the ``C``
        coordinate, colours to a ``color`` coordinate along ``C``
        (``image_io.set_channel_info``; a missing entry takes a palette
        colour) and ``pixel_sizes`` to the calibrated coordinates.
        A **list of stacks** (e.g. one 3D stack per cage / field of view; same
        channels, shapes may differ; arrays or DataArrays) shows one at a time
        with a stack selector (toolbar, ``stack`` trait, PageUp / PageDown).
        ``labels`` is then a matching sequence (or None, or an array stacked
        along axis 0), ``stack_names`` the names. Each stack keeps its own
        labels, z position, display range, pixel sizes and track-op undo
        history; ``labels`` returns the shown stack, :attr:`stack_labels` all
        of them, ``save(..., stacks="all")`` writes all. ``axes`` applies to
        every stack.
    axes : str, optional
        Axis letters of ``image`` (e.g. ``"CYX"``, ``"ZYX"``, ``"CZYX"``);
        required for ndarrays with ndim > 2.
    labels : array-like, optional
        Initial labels in **(Z)YX order** (``YX`` or ``ZYX``), whatever the
        order of the spatial letters in ``axes``: labels are always canonical
        (Z)YX, in and out (``labels``, ``stack_labels``, ``save``). For
        ``axes="XY"`` pass and receive ``(Y, X)`` labels.
    mode : {"instance", "semantic"}
        ``"instance"``: every new object gets a new id (Cellpose training
        masks); strokes are closed and filled by default.
        ``"semantic"``: paint class ids ``1..len(classes)`` (pixel-classifier
        scribbles); 0 means unlabelled.
    classes : list of str, optional
        Class names for semantic mode.
    channel_names, channel_colors : list of str, optional
        Colours may be any matplotlib colour (name, hex, RGB tuple) or
        colormap name or registered Colormap (a colormap such as ``"gray"``
        maps to its end colour). Missing colours take the package palette
        ('white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue'), cycled,
        as in the ortho viewers.
    colormap, channel_labels : optional, keyword only
        The ortho viewers' names for ``channel_colors`` and ``channel_names``
        (aliases; passing both spellings raises ``TypeError``).
    percentiles : (float, float), optional
        Automatic display range per channel over the whole volume (stack).
        None (default) uses the package's default display range, the same as
        the ortho viewers: 0 (or the data minimum if negative) to the robust
        99.99th percentile (the maximum for signal sparser than 0.01%).
        Explicit percentiles (e.g. ``(0.1, 99.99)``, the old default) fall
        back to the data maximum, with a warning, when they coincide.
    vmin, vmax : float or list of float, optional, keyword only
        Display range per channel in **data units** (a scalar applies to
        every channel; None entries take the automatic range). The browser
        shows an 8-bit slice that spans ``vmin..vmax``; changing the ``vmin``
        / ``vmax`` traits (Python or the channel sliders, which span the data
        range) re-quantises the slice in Python, so the range can be widened
        as well as narrowed. Double-click a slider to restore the automatic
        value.
    gamma : float or list of float, optional, keyword only
        Display gamma per channel (> 0, default 1), applied in the browser.
    pixel_sizes : dict, sequence or float, optional, keyword only
        ``{'Z', 'Y', 'X', 'unit'}``, ``(Z, Y, X)`` or an isotropic size, in
        µm unless a ``'unit'`` says otherwise (known length units are
        converted). Stored as :attr:`pixel_sizes` and written by :meth:`save`
        as TIFF resolution. Painting and display stay in pixels.
    z : int, optional, keyword only
        First z-slice shown (default: the middle one); of stack ``stack``.
    auto_fill : bool, optional, keyword only
        Close and fill each brush stroke (default True in instance mode,
        False in semantic mode).
    **kwargs
        Any other trait, e.g. ``opacity``, ``display``, ``outline_px``,
        ``tool``, ``active_label``, ``channel_visible``, ``class_colors``,
        ``instance_palette``, ``auto_increment``, ``stack``. They are applied
        after the defaults and validated (``ValueError``); traits the widget
        computes (``shape_zyx``, ``image_to_js``, ...) and unknown names raise
        ``TypeError``.

    Changed (D02/D03, 2026-11): the default range was (p0.1, p99.99) with
    ``lo + 1`` for sparse channels (rendered binary); the default colours
    were magenta, green, cyan, ... and more than 8 channels raised;
    RGB-tuple colours in a DataArray ``color`` coordinate were ignored.

    Changed (D18/R4-07, 2026-11): ``vmin``/``vmax`` are data units (they
    were 0..255 steps of a fixed 8-bit pre-quantisation, so the GUI could
    only narrow the range); trait kwargs such as ``z``, ``vmin``,
    ``active_label`` or ``stack`` are honoured (they were overwritten);
    unknown kwargs raise ``TypeError`` (they were ignored); ``mode`` and
    ``background`` are read-only after construction; a list of DataArrays
    keeps the coordinates' names, colours and pixel sizes; ``save`` writes
    axes and pixel sizes; labels are stored as uint16 internally (the
    getters still return uint32).
    brush_radius : int
        Brush radius in pixels.
    keep_label : bool
        Instance mode: keep the active id after each stroke or fill instead of
        advancing to the next free id (``auto_increment = not keep_label``).
        Use it to paint the same object on consecutive z-slices.
    show_adjacent : bool
        Show outlines of the labels on slices z-1 (dotted) and z+1 (faint) over
        the current slice, each in its id's colour ("onion skin").
    display : {"filled", "outlined"}
        ``"filled"`` (default): labels are filled at ``opacity``.
        ``"outlined"``: each label's boundary is drawn ``outline_px`` wide in its
        colour over an opaque black band ``outline_border_px`` wide on each side
        (readable on bright and dark images); the interior is only faintly tinted
        (``opacity`` / 4) and the outline fades with ``opacity`` below 0.45.
        Background (id 65535) stays hatched grey. Toolbar toggle or key ``o``.
        The legacy ``outline_only`` trait still draws thin 1-px outlines in the
        filled style and is superseded by ``display="outlined"``.
    outline_px, outline_border_px : int
        Widths in pixels of the coloured outline (default 2) and of the black
        border band around it (default 1) in the outlined display.
    background : bool
        Instance mode only: reserve ``BACKGROUND_ID`` (65535) for annotated
        background and add a ``background`` brush (key ``g``) that paints it
        as open scribbles (never closed / filled). The id is never handed out
        as a cell id, merge / split refuse it, and it is drawn in a fixed
        neutral grey (hatched) in every display mode. ``labels`` may then
        also be signed (negative values = background). Default False: 65535
        is an ordinary id and nothing changes. ``mode`` and ``background``
        cannot be changed after construction (``TraitError``).

    Attributes
    ----------
    labels : np.ndarray
        Current labels (``uint32``), (Z)YX order.
    pixel_sizes : dict
        ``{'Z', 'Y', 'X', 'unit'}`` of the shown stack, in µm.
    BACKGROUND_ID : int
        65535, the reserved annotated-background id (with ``background=True``).
    active_label_extent : str
        z extent of the active id, e.g. ``"label 37: z 12-19 (8 slices)"``.
    counts_text : str
        Readout of :meth:`counts` for the current slice and the volume.

    Notes
    -----
    z-linking ground truth: in instance mode an id is one 3D object, so the
    same id on several slices links those 2D masks. Tools for this:
    ``keep label`` (paint one id slice after slice), the ``z±1`` outlines,
    ``merge`` (click a mask: its id is replaced by the active id in all
    slices; :meth:`merge_ids`) and ``split`` (click a mask at slice z: its
    voxels at slices >= z get a new id; :meth:`split_id`). Both whole-track
    operations are undoable (ctrl+z / undo, or :meth:`undo_track_op`).

    Sparse training annotations (``background=True``): 0 = unannotated
    (ignored in training), ``BACKGROUND_ID`` = annotated background (-1 in
    :meth:`labels_signed` and ``save(signed=True)``), 1.. = cell instances.
    Paint background with the ``background`` brush; the eraser removes it like
    any id. The ``delete`` tool (and :meth:`delete_id`) removes a whole mask:
    click = this slice, shift+click = all slices; undoable like merge/split.
    The widget shows :meth:`counts` (cells, background and annotated pixels,
    annotated fraction) for the current slice and the volume. This is a
    labelling convention only: training code must map it to its own loss
    (e.g. mask out 0); Cellpose itself does not interpret -1.

    Keyboard (only while the image viewport has focus: click it first): o
    toggle filled/outlined display, b/e/f/p brush/eraser/fill/picker, g
    background brush (``background=True``), x delete mask (shift+click: all
    slices), m/s merge/split track, n new id, [ / ] brush radius,
    ArrowUp/ArrowDown or Alt+ArrowRight/Alt+ArrowLeft z+1/z-1,
    PageUp/PageDown previous/next stack, 0 fit, ctrl+z undo, space+drag or
    right/middle drag pan, 1-9 class (semantic mode). Keys the widget handles
    do not reach the notebook (no JupyterLab command-mode side effects); other
    keys and modified letters (ctrl+s, ...) do. Mouse wheel: zooms once the
    viewport was clicked (focused), otherwise scrolls the page; ctrl+wheel and
    trackpad pinch always zoom.

    Changed (D17, 2026-11): handled keys used to bubble to the host, and the
    plain wheel always zoomed (the page could not be scrolled past the widget).
    """

    BACKGROUND_ID = BACKGROUND_ID

    _esm = _ASSETS.joinpath("label_annotator.js").read_text()
    _css = _ASSETS.joinpath("label_annotator.css").read_text()

    # geometry / data (Python -> JS)
    shape_zyx = traitlets.List(traitlets.Int(), default_value=[1, 1, 1]).tag(sync=True)
    z = traitlets.Int(0).tag(sync=True)
    image_to_js = traitlets.Bytes(b"").tag(sync=True)
    labels_to_js = traitlets.Bytes(b"").tag(sync=True)
    # labels of slices z-1 / z+1 for the onion-skin outlines (b"" = none)
    labels_prev_to_js = traitlets.Bytes(b"").tag(sync=True)
    labels_next_to_js = traitlets.Bytes(b"").tag(sync=True)
    # bumped whenever Python rewrites labels the browser may have edited since
    # the last labels_to_js (track ops, set_labels): the browser then reloads
    # labels_to_js even if its bytes did not change (hosts drop equal values)
    labels_rev = traitlets.Int(0).tag(sync=True)
    # bumped when the browser's per-slice undo snapshots stop matching the shown
    # labels (stack switch, set_labels; not track ops, which keep their markers):
    # the browser then drops them and rebuilds its track-op markers
    undo_epoch = traitlets.Int(0).tag(sync=True)
    # edits (JS -> Python); Any because buffers may arrive as memoryview. Reset to
    # None once applied, so a byte-identical next message still notifies
    labels_from_js = traitlets.Any(None, allow_none=True).tag(sync=True)
    # bumped by every newly rendered view (JS -> Python): Python then re-sends the
    # current slice if labels_to_js is stale (strokes are never echoed)
    labels_resync = traitlets.Int(0).tag(sync=True)

    # display (composited in the browser; vmin / vmax re-quantise the 8-bit slice in Python)
    channel_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_visible = traitlets.List(traitlets.Bool()).tag(sync=True)
    # display range per channel in DATA units (the 8-bit slice spans vmin..vmax)
    vmin = traitlets.List(traitlets.Float()).tag(sync=True)
    vmax = traitlets.List(traitlets.Float()).tag(sync=True)
    gamma = traitlets.List(traitlets.Float()).tag(sync=True)
    # per channel [lo, hi, step, auto vmin, auto vmax] for the browser's sliders (Python -> JS)
    data_range = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
    opacity = traitlets.Float(0.45).tag(sync=True)
    # "filled" (default) | "outlined": per-label coloured boundary (outline_px wide)
    # over an opaque black border band (outline_border_px wide), faint interior
    display = traitlets.Enum(["filled", "outlined"], default_value="filled").tag(sync=True)
    outline_px = traitlets.Int(2).tag(sync=True)
    outline_border_px = traitlets.Int(1).tag(sync=True)
    # legacy: thin 1-px outlines in the filled style; kept for old code (prefer ``display``)
    outline_only = traitlets.Bool(False).tag(sync=True)
    show_adjacent = traitlets.Bool(False).tag(sync=True)

    # labelling
    mode = traitlets.Unicode("instance").tag(sync=True)
    class_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    class_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    # instance colours: labels_cmap palette, id k -> palette[(k - 1) % len]
    instance_palette = traitlets.List(traitlets.Unicode()).tag(sync=True)
    active_label = traitlets.Int(1).tag(sync=True)
    tool = traitlets.Unicode("brush").tag(sync=True)  # brush | eraser | fill | picker | merge | split | background | delete
    # reserve BACKGROUND_ID for annotated background (instance mode); see class docstring
    background = traitlets.Bool(False).tag(sync=True)
    counts_text = traitlets.Unicode("").tag(sync=True)
    # several stacks (list input): index of the shown stack and the stacks' names
    stack = traitlets.Int(0).tag(sync=True)
    stack_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    brush_radius = traitlets.Int(2).tag(sync=True)
    overwrite = traitlets.Bool(False).tag(sync=True)
    auto_fill = traitlets.Bool(True).tag(sync=True)
    auto_increment = traitlets.Bool(True).tag(sync=True)
    new_label_trigger = traitlets.Int(0).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    active_label_extent = traitlets.Unicode("").tag(sync=True)
    # whole-track operations: JS -> Python requests
    # {"op": "merge", "src", "dst"} | {"op": "split", "id", "z"} | {"op": "delete", "id", "z" (None: all)}
    # | {"op": "undo", "serial"},
    # plus a counter "n" so repeated identical requests still notify
    track_request = traitlets.Dict().tag(sync=True)
    # serials of the undoable track operations, oldest first (Python -> JS)
    track_undo_ids = traitlets.List(traitlets.Int()).tag(sync=True)

    # traits the widget computes: not constructor kwargs
    _COMPUTED = frozenset({
        "shape_zyx", "image_to_js", "labels_to_js", "labels_prev_to_js", "labels_next_to_js", "labels_rev",
        "undo_epoch", "labels_from_js", "labels_resync", "data_range", "class_names", "counts_text", "status",
        "active_label_extent", "track_request", "track_undo_ids", "new_label_trigger",
    })

    def __init__(
        self,
        image,
        axes: str | None = None,
        labels=None,
        mode: str = "instance",
        classes: list[str] | None = None,
        channel_names: list[str] | None = None,
        channel_colors: list[str] | None = None,
        percentiles: tuple[float, float] | None = None,
        brush_radius: int = 2,
        keep_label: bool = False,
        show_adjacent: bool = False,
        background: bool = False,
        stack_names: list[str] | None = None,
        *,
        colormap=None,
        channel_labels: list[str] | None = None,
        pixel_sizes=None,
        vmin=None,
        vmax=None,
        gamma=None,
        z: int | None = None,
        auto_fill: bool | None = None,
        **kwargs,
    ):
        if mode not in ("instance", "semantic"):
            raise ValueError("mode must be 'instance' or 'semantic'")
        background = bool(background)
        if background and mode != "instance":
            raise ValueError("background=True is for instance mode; in semantic mode add a background class")
        # tnia-style aliases (D18): one spelling per call
        if colormap is not None:
            if channel_colors is not None:
                raise TypeError("pass channel_colors or its alias colormap, not both")
            channel_colors = colormap
        if channel_labels is not None:
            if channel_names is not None:
                raise TypeError("pass channel_names or its alias channel_labels, not both")
            channel_names = channel_labels
        # trait kwargs: computed traits and unknown names raise (they used to be ignored);
        # the traits the constructor sets are applied after its defaults, validated
        traits = type(self).class_trait_names()
        for k in kwargs:
            if k in self._COMPUTED:
                raise TypeError(f"{k!r} is computed by LabelAnnotator and cannot be passed")
            if k not in traits:
                raise TypeError(f"LabelAnnotator got an unexpected keyword argument {k!r}")
        managed = {k: kwargs.pop(k) for k in
                   ("active_label", "class_colors", "stack", "channel_visible", "auto_increment", "instance_palette")
                   if k in kwargs}

        given_ps = pixel_sizes
        if isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("image: empty list of stacks")
            imgs = list(image)
            if labels is None:
                labs = [None] * len(imgs)
            elif isinstance(labels, np.ndarray) or (
                isinstance(labels, Sequence) and not isinstance(labels, (str, bytes))
            ):
                labs = list(labels)  # a sequence, or an array with one entry per stack along axis 0
            else:
                raise TypeError(
                    "labels for a list of stacks must be a list (one array or None per stack) or an array "
                    f"stacked along axis 0, not {type(labels).__name__}"
                )
            if len(labs) != len(imgs):
                raise ValueError(f"{len(imgs)} stacks but {len(labs)} label arrays")
            if stack_names is not None and len(stack_names) != len(imgs):
                raise ValueError(f"{len(imgs)} stacks but {len(stack_names)} stack_names")
            names_in = list(stack_names) if stack_names is not None else [f"stack {i}" for i in range(len(imgs))]
        else:
            imgs, labs, names_in = [image], [labels], []
        # every stack (a DataArray in a list too) is unpacked by the viewers' rule;
        # names / colours come from the first stack unless given, pixel sizes per stack
        stacks = []
        for k, im in enumerate(imgs):
            czyx, in_axes, channel_colors, channel_names, ps = _unpack(im, axes, channel_colors, channel_names, given_ps)
            if k == 0:
                input_axes = in_axes
            elif czyx.shape[0] != stacks[0][0].shape[0]:
                raise ValueError(f"all stacks need the same channels: {stacks[0][0].shape[0]} vs {czyx.shape[0]}")
            stacks.append((czyx, _parse_pixel_sizes(ps)))
        czyx = stacks[0][0]
        c = czyx.shape[0]

        names = list(channel_names) if channel_names else [f"ch{i}" for i in range(c)]
        if len(names) != c:
            raise ValueError(f"need {c} channel names/colors")
        colors = _channel_colors(channel_colors, c)
        classes = list(classes) if classes else (["foreground"] if mode == "semantic" else [])
        gamma = [_finite(g, "gamma") for g in per_channel(gamma, c, "gamma", default=1.0)]
        if any(g <= 0 for g in gamma):
            raise ValueError(f"gamma must be > 0, got {gamma}")
        visible = [bool(v) for v in per_channel(managed.pop("channel_visible", None), c, "channel_visible", True)]
        class_colors = managed.pop("class_colors", None)
        if class_colors is not None:
            class_colors = [_plain_color_hex(v) for v in class_colors]
            if len(class_colors) != len(classes):
                raise ValueError(f"{len(class_colors)} class_colors for {len(classes)} classes")
        palette = managed.pop("instance_palette", None)
        if palette is not None:
            palette = [_plain_color_hex(v) for v in palette]
            if not palette:
                raise ValueError("instance_palette must not be empty")
        auto_increment = managed.pop("auto_increment", None)
        if auto_increment is not None and keep_label and auto_increment:
            raise ValueError("keep_label=True means auto_increment=False; pass one of them")
        auto_increment = (not keep_label) if auto_increment is None else bool(auto_increment)
        stack = managed.pop("stack", None)
        if stack is not None and not 0 <= int(stack) < len(stacks):
            raise ValueError(f"stack={stack} out of range [0, {len(stacks) - 1}]")
        active = managed.pop("active_label", None)
        if active is not None and mode == "semantic" and not 0 <= int(active) <= len(classes):
            raise ValueError(f"active_label={active} out of range [0, {len(classes)}] (semantic classes)")

        # cached (nonzero ids, pixel counts) per slice: extent / counts readouts, next id
        # (serial, flat idx, old values, value written, what) per track op
        # one track serial for the whole widget (not per stack): serials never repeat, so
        # the browser can tell a new op from one it has mirrored already
        self._track_serial = 0
        self._ready = False
        self._spatial_axes = "".join(a for a in input_axes if a in "ZYX")
        # every stack's state; the shown one is unparked into self._image & co below
        self._stack_idx = 0
        self._parked: list[dict | None] = []
        for k, ((s_img, ps), s_lab) in enumerate(zip(stacks, labs)):
            sz, sy, sx = s_img.shape[1:]
            if s_lab is None:
                sl = np.zeros((sz, sy, sx), dtype=_STORE)
            else:
                sl = np.asarray(s_lab)
                sl = sl[np.newaxis] if sl.ndim == 2 else sl
                if sl.shape != (sz, sy, sx):
                    where = "image spatial shape" if len(stacks) == 1 else f"stack {k}: image"
                    raise ValueError(f"labels shape {np.asarray(s_lab).shape} does not match {where} {(sz, sy, sx)}")
                sl = _coerce_label_values(sl, background)
            auto, lo, hi, drange = _channel_ranges(s_img, percentiles, vmin, vmax)
            self._parked.append(dict(
                image=s_img, labels=sl, limits=auto, vmin=lo, vmax=hi, data_range=drange, ps=ps,
                hist=[None] * sz, undo=[], z=sz // 2,
            ))
        st = self._unpark(0)

        super().__init__(**kwargs)
        nz, ny, nx = self._labels.shape
        with self.hold_sync():
            self.stack_names = names_in
            self.shape_zyx = [nz, ny, nx]
            self.channel_names = names
            self.channel_colors = colors
            self.channel_visible = visible
            self.data_range = st["data_range"]
            self.vmin = st["vmin"]
            self.vmax = st["vmax"]
            self.gamma = gamma
            self.mode = mode
            self.class_names = classes
            self.class_colors = class_colors if class_colors is not None else _class_palette(len(classes))
            self.instance_palette = palette if palette is not None else [_rgb_hex(c) for c in LABELS_CMAP_COLORS]
            self.auto_fill = (mode == "instance") if auto_fill is None else bool(auto_fill)
            self.auto_increment = mode == "instance" and auto_increment
            self.brush_radius = brush_radius
            self.show_adjacent = bool(show_adjacent)
            self.background = background
            self.active_label = self._next_label() if mode == "instance" else 1
            self.z = st["z"]
            self._ready = True
            self._frozen = True  # mode / background are fixed from here on
            self._send_slice()
            self._update_extent()
            self._update_counts()
        # the user's choices, applied once the geometry is known (stack, then z, then the id)
        if stack is not None:
            self.stack = int(stack)
        if z is not None:
            nz = self._labels.shape[0]
            if not 0 <= int(z) < nz:
                raise ValueError(f"z={z} out of range [0, {nz - 1}]")
            self.z = int(z)
        if active is not None:
            self.active_label = int(active)

    # ------------------------------------------------------------------ sync
    def _slice_u8(self, z: int) -> np.ndarray:
        """The 8-bit display slice: each channel scaled from ``vmin`` (0) to ``vmax`` (255), in data units."""
        out = np.empty(self._image.shape[:1] + self._image.shape[2:], dtype=np.uint8)
        for i in range(out.shape[0]):
            lo, hi = float(self.vmin[i]), float(self.vmax[i])
            s = np.asarray(self._image[i, z], dtype=np.float32)
            if hi > lo:
                s = (s - lo) * (255.0 / (hi - lo))
            else:  # crossed sliders: a threshold
                s = np.where(s > lo, np.float32(255), np.float32(0))
            # NaN survives the clip and would warn in the uint8 cast: show it as 0
            np.nan_to_num(s, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
            np.clip(s, 0, 255, out=s)
            out[i] = s.astype(np.uint8)
        return out

    @traitlets.validate("vmin", "vmax", "gamma")
    def _valid_range(self, proposal):
        v = list(proposal["value"])
        name = proposal["trait"].name
        if getattr(self, "_ready", False) and len(v) != self._image.shape[0]:
            raise traitlets.TraitError(f"{name} needs {self._image.shape[0]} values (one per channel), got {len(v)}")
        if not all(np.isfinite(x) for x in v):
            raise traitlets.TraitError(f"{name} values must be finite, got {v}")
        if name == "gamma" and any(x <= 0 for x in v):
            raise traitlets.TraitError(f"gamma values must be > 0, got {v}")
        return v

    @traitlets.observe("vmin", "vmax")
    def _on_range(self, change):
        """Re-quantise the shown slice for the new range (labels are not re-sent)."""
        if not getattr(self, "_ready", False) or getattr(self, "_switching", False):
            return
        dr = [list(r) for r in self.data_range]
        changed = False
        for i, r in enumerate(dr):
            lo, hi = min(r[0], self.vmin[i], self.vmax[i]), max(r[1], self.vmin[i], self.vmax[i])
            if (lo, hi) != (r[0], r[1]):
                r[0], r[1], changed = lo, hi, True
        with self.hold_sync():
            if changed:  # the sliders must be able to show the new value
                self.data_range = dr
            self.image_to_js = encode_image_slice(self._slice_u8(self.z), self.z)

    @traitlets.validate("mode", "background")
    def _valid_fixed(self, proposal):
        name, v = proposal["trait"].name, proposal["value"]
        if getattr(self, "_frozen", False) and v != getattr(self, name):
            raise traitlets.TraitError(
                f"{name} is fixed at construction (labels and ids depend on it); create a new LabelAnnotator"
            )
        return v

    def _send_slice(self):
        z = self.z
        with self.hold_sync():
            self.image_to_js = encode_image_slice(self._slice_u8(z), z)
            self.labels_to_js = encode_label_slice(self._labels[z], z)
            self._send_adjacent()

    def _send_labels(self):
        """Re-send the current slice's labels and its neighbours (after a volume edit).

        ``labels_rev`` is bumped so the browser reloads the slice even when the
        encoded bytes equal the previous ``labels_to_js`` (e.g. a merge that
        restores what Python last sent, after browser strokes Python has
        received but never echoed); otherwise the browser would keep showing,
        and later send back, its stale slice.
        """
        with self.hold_sync():
            self.labels_to_js = encode_label_slice(self._labels[self.z], self.z)
            self._send_adjacent()
            self.labels_rev += 1

    def _send_adjacent(self):
        nz = self._labels.shape[0]

        def enc(k):
            return encode_label_slice(self._labels[k], k) if self.show_adjacent and 0 <= k < nz else b""

        self.labels_prev_to_js = enc(self.z - 1)
        self.labels_next_to_js = enc(self.z + 1)

    @traitlets.observe("show_adjacent")
    def _on_show_adjacent(self, change):
        if hasattr(self, "_image"):
            self._send_adjacent()

    def _park(self) -> dict:
        return dict(
            image=self._image, labels=self._labels, limits=self._limits, hist=self._slice_hist,
            undo=self._track_undo, z=int(self.z), ps=self._ps,
            vmin=list(self.vmin), vmax=list(self.vmax), data_range=[list(r) for r in self.data_range],
        )

    def _unpark(self, k: int) -> dict:
        """Make stack ``k`` the shown one (Python state only); returns its parked dict."""
        st = self._parked[k]
        self._parked[k] = None
        self._stack_idx = k
        self._image, self._labels, self._limits = st["image"], st["labels"], st["limits"]
        self._slice_hist, self._track_undo, self._ps = st["hist"], st["undo"], st["ps"]
        return st

    @property
    def pixel_sizes(self) -> dict:
        """Pixel sizes of the shown stack, ``{'Z', 'Y', 'X', 'unit'}`` in µm (None where unknown).

        From ``pixel_sizes=`` (any form the package accepts; known length units
        are converted to µm) or, for a DataArray, its calibrated coordinates.
        :meth:`save` writes Y/X as TIFF resolution. Display stays in pixels.
        """
        return dict(self._ps)

    @traitlets.validate("stack")
    def _valid_stack(self, proposal):
        n = max(1, len(self._parked)) if hasattr(self, "_parked") else 1
        return int(min(max(proposal["value"], 0), n - 1))

    @traitlets.observe("stack")
    def _on_stack(self, change):
        if not hasattr(self, "_parked") or change["new"] == self._stack_idx:
            return
        self._parked[self._stack_idx] = self._park()
        st = self._unpark(change["new"])
        with self.hold_sync():
            nz, ny, nx = self._labels.shape
            self.shape_zyx = [nz, ny, nx]
            # each stack keeps its own display range; the slice is sent once below
            self._switching = True
            try:
                self.data_range = st["data_range"]
                self.vmin, self.vmax = st["vmin"], st["vmax"]
            finally:
                self._switching = False
            self.track_undo_ids = [u[0] for u in self._track_undo]
            self.z = st["z"]  # triggers _send_slice when it differs; resend anyway below
            self._send_slice()
            self.labels_rev += 1
            self.undo_epoch += 1  # the browser's slice snapshots belong to the previous stack
            if self.mode == "instance":
                self.active_label = self._next_label()
            self._update_extent()
            self._update_counts()

    @property
    def stack_labels(self) -> list[np.ndarray]:
        """Labels of every stack (copies, ``uint32``, (Z)YX), in input order."""
        out = []
        for k, p in enumerate(self._parked):
            lab = self._labels if k == self._stack_idx else p["labels"]
            out.append(_restore_axes(lab.astype(np.uint32), self._spatial_axes))
        return out

    @traitlets.validate("z")
    def _valid_z(self, proposal):
        nz = self._image.shape[1] if hasattr(self, "_image") else 1
        return int(min(max(proposal["value"], 0), nz - 1))

    @traitlets.observe("z")
    def _on_z(self, change):
        if hasattr(self, "_image"):
            self._send_slice()
            self._update_counts()

    @traitlets.observe("background")
    def _on_background(self, change):
        if hasattr(self, "_slice_hist"):
            self._update_extent()
            self._update_counts()

    @traitlets.observe("labels_from_js")
    def _on_labels_from_js(self, change):
        buf = change["new"]
        if buf is None:
            return
        try:
            if len(bytes(buf)) == 0:
                return
            z, lab = decode_label_slice(buf)
            if lab.shape != self._labels.shape[1:] or not 0 <= z < self._labels.shape[0]:
                self.status = f"ignored label message for z={z} with shape {lab.shape}"
                return
            self._labels[z] = lab
            self._slice_hist[z] = None
            if self.mode == "instance" and self.auto_increment:
                self.active_label = self._next_label()
            self._update_extent()
            self._update_counts()
        finally:
            # traitlets (and the hosts) drop a value equal to the current one, so a
            # second identical message (e.g. "clear slice" again after set_labels or
            # a stack switch changed the slice) would be lost: forget this one
            self.labels_from_js = None

    @traitlets.observe("labels_resync")
    def _on_labels_resync(self, change):
        """A newly rendered view starts from ``labels_to_js``: re-send it if browser strokes made it stale."""
        if hasattr(self, "_image") and bytes(self.labels_to_js) != encode_label_slice(self._labels[self.z], self.z):
            self._send_labels()

    @traitlets.observe("new_label_trigger")
    def _on_new_label(self, change):
        self.active_label = self._next_label()

    def _max_cell_id(self) -> int:
        """Largest id usable for an object: 65535, or 65534 when 65535 is the background id."""
        return BACKGROUND_ID - 1 if self.background else _MAX_WIRE_LABEL

    def _cell_ids(self) -> np.ndarray:
        """Sorted ids present in the volume, excluding 0 and (with ``background``) ``BACKGROUND_ID``."""
        ids = [self._hist(z)[0] for z in range(self._labels.shape[0])]
        ids = np.unique(np.concatenate(ids)) if ids else np.zeros(0, np.intp)
        return ids[ids != BACKGROUND_ID] if self.background else ids

    def _next_label(self) -> int:
        """``max + 1`` over the object ids; once that exceeds the largest usable id
        (65535, or 65534 with ``background``), the smallest unused id. Never the background id."""
        top = self._max_cell_id()
        ids = self._cell_ids()
        nxt = int(ids[-1]) + 1 if ids.size else 1
        if nxt <= top:
            return nxt
        free = np.setdiff1d(np.arange(1, top + 1), ids, assume_unique=True)
        if free.size == 0:
            raise ValueError(f"all ids 1..{top} are in use")
        return int(free[0])

    @traitlets.validate("active_label")
    def _valid_active_label(self, proposal):
        v = int(proposal["value"])
        if not 0 <= v <= _MAX_WIRE_LABEL:
            # the browser paints uint16: a larger id would wrap (65536 -> 0, 65537 -> 1)
            raise traitlets.TraitError(f"active_label must be in [0, {_MAX_WIRE_LABEL}], got {v}")
        if v == BACKGROUND_ID and self.background:
            raise traitlets.TraitError(f"id {BACKGROUND_ID} is reserved for annotated background; use the background tool")
        return v

    @traitlets.observe("active_label")
    def _on_active_label(self, change):
        if hasattr(self, "_slice_hist"):
            self._update_extent()

    def _hist(self, z: int) -> tuple[np.ndarray, np.ndarray]:
        """(nonzero ids, their pixel counts) on slice ``z``; cached until the slice is edited."""
        h = self._slice_hist[z]
        if h is None:
            # ids <= 65535 (wire limit), so a bincount is a cheap O(n) unique
            bc = np.bincount(self._labels[z].ravel())
            ids = np.flatnonzero(bc)
            ids = ids[ids != 0]
            h = self._slice_hist[z] = (ids, bc[ids])
        return h

    def _ids_in_slice(self, z: int) -> np.ndarray:
        return self._hist(z)[0]

    def label_extent(self, label: int) -> list[int]:
        """z indices of the slices that contain ``label`` (sorted)."""
        return [z for z in range(self._labels.shape[0]) if bool((self._ids_in_slice(z) == label).any())]

    def _update_extent(self):
        a = int(self.active_label)
        if self.mode != "instance":
            self.active_label_extent = ""
            return
        if a == 0:
            self.active_label_extent = "label 0 (unannotated)" if self.background else "label 0 (background)"
            return
        zs = self.label_extent(a)
        if not zs:
            self.active_label_extent = f"label {a}: not present"
            return
        lo, hi = zs[0], zs[-1]
        txt = f"label {a}: z {lo}-{hi} ({len(zs)} slice{'s' if len(zs) != 1 else ''})"
        missing = sorted(set(range(lo, hi + 1)) - set(zs))
        if missing:
            txt += f", missing z {', '.join(map(str, missing))}"
        self.active_label_extent = txt

    def counts(self, z: int | None = None) -> dict:
        """Annotation counts for slice ``z`` or, with ``z=None``, the whole volume.

        Returns
        -------
        dict
            ``n_cells``: number of distinct object ids (excluding 0 and, with
            ``background``, ``BACKGROUND_ID``); ``cell_px``: pixels with such
            an id; ``background_px``: pixels with ``BACKGROUND_ID`` (0 unless
            ``background``); ``annotated_px``: ``cell_px + background_px``
            (all nonzero pixels); ``annotated_fraction``: ``annotated_px`` /
            pixels in the slice or volume. In semantic mode "cells" are the
            class ids.
        """
        nz = self._labels.shape[0]
        if z is None:
            zs = range(nz)
        else:
            z = int(z)
            if not 0 <= z < nz:
                raise ValueError(f"z={z} out of range [0, {nz - 1}]")
            zs = [z]
        cell_px = bg_px = 0
        ids = []
        for k in zs:
            i, c = self._hist(k)
            bg = (i == BACKGROUND_ID) if self.background else np.zeros(i.shape, bool)
            bg_px += int(c[bg].sum())
            cell_px += int(c[~bg].sum())
            ids.append(i[~bg])
        n_cells = int(np.unique(np.concatenate(ids)).size) if ids else 0
        total = len(zs) * self._labels.shape[1] * self._labels.shape[2]
        annotated = cell_px + bg_px
        return {
            "n_cells": n_cells,
            "cell_px": cell_px,
            "background_px": bg_px,
            "annotated_px": annotated,
            "annotated_fraction": annotated / total if total else 0.0,
        }

    def _update_counts(self) -> None:
        def fmt(c):
            frac = f"{c['annotated_px']} annotated px ({100 * c['annotated_fraction']:.1f}%)"
            if self.mode != "instance":
                return frac
            bg = f", {c['background_px']} bg px" if self.background else ""
            return f"{c['n_cells']} cell{'s' if c['n_cells'] != 1 else ''}{bg}, {frac}"

        if self._labels.shape[0] == 1:
            self.counts_text = fmt(self.counts(0))
        else:
            self.counts_text = f"z {self.z}: {fmt(self.counts(self.z))} | all z: {fmt(self.counts())}"

    # ------------------------------------------------------- track operations
    @traitlets.observe("track_request")
    def _on_track_request(self, change):
        req = change["new"] or {}
        op = req.get("op")
        if op == "merge":
            self.merge_ids(int(req["src"]), int(req["dst"]))
        elif op == "split":
            self.split_id(int(req["id"]), int(req["z"]))
        elif op == "delete":
            zr = req.get("z")
            self.delete_id(int(req["id"]), None if zr is None else int(zr))
        elif op == "undo":
            serial = req.get("serial")
            if self._track_undo and (serial is None or int(serial) == self._track_undo[-1][0]):
                self.undo_track_op()
            else:
                self.status = "track undo out of sync; nothing undone"
                self.track_undo_ids = [u[0] for u in self._track_undo]

    def _apply_track_op(self, idx: np.ndarray, new_values: int, what: str) -> None:
        """Write the id ``new_values`` at flat indices ``idx``, record the old values for undo, refresh."""
        flat = self._labels.reshape(-1)
        self._track_serial += 1
        # compact undo entry: int32 indices when they fit, uint16 old values (R4-19)
        if flat.size <= np.iinfo(np.int32).max:
            idx = idx.astype(np.int32, copy=False)
        self._track_undo.append((self._track_serial, idx, flat[idx].copy(), int(new_values), what))
        del self._track_undo[:-_TRACK_UNDO_MAX]
        while len(self._track_undo) > 1 and sum(u[1].nbytes + u[2].nbytes for u in self._track_undo) > _TRACK_UNDO_MAX_BYTES:
            del self._track_undo[0]
        flat[idx] = new_values
        self._after_volume_edit(idx)
        self.status = what

    def _after_volume_edit(self, idx: np.ndarray) -> None:
        plane = self._labels.shape[1] * self._labels.shape[2]
        for z in np.unique(idx // plane):
            self._slice_hist[int(z)] = None
        with self.hold_sync():
            self.track_undo_ids = [u[0] for u in self._track_undo]
            self._send_labels()
        self._update_extent()
        self._update_counts()

    def merge_ids(self, src: int, dst: int) -> bool:
        """Relabel ``src`` as ``dst`` in every slice (link two tracks into one object).

        Parameters
        ----------
        src : int
            Id to replace (e.g. the id of the clicked mask).
        dst : int
            Id it becomes (the active id in the widget).

        Returns
        -------
        bool
            True if voxels changed. Refused (False, reason in ``status``) if
            either id is 0 or the background id (``background=True``), they
            are equal, or ``src`` is absent. If both ids
            occur on a common slice the merge is applied and ``status`` warns,
            since that slice then holds one id on two masks.
        """
        src, dst = int(src), int(dst)
        if self.mode != "instance":
            self.status = "merge: instance mode only"
            return False
        if src == 0 or dst == 0:
            self.status = "merge refused: id 0 is background"
            return False
        if self.background and BACKGROUND_ID in (src, dst):
            self.status = f"merge refused: id {BACKGROUND_ID} is annotated background, not a cell"
            return False
        if src == dst:
            self.status = f"merge refused: clicked label {src} is already the active label"
            return False
        if dst > _MAX_WIRE_LABEL:
            self.status = f"merge refused: id {dst} > {_MAX_WIRE_LABEL}"
            return False
        idx = np.flatnonzero(self._labels.reshape(-1) == src)
        if idx.size == 0:
            self.status = f"merge refused: label {src} not present"
            return False
        both = sorted(set(self.label_extent(src)) & set(self.label_extent(dst)))
        what = f"merged label {src} into {dst} ({len(np.unique(idx // (self._labels.shape[1] * self._labels.shape[2])))} slices)"
        if both:
            what += f"; warning: both on z {', '.join(map(str, both))}"
        self._apply_track_op(idx, dst, what)
        return True

    def split_id(self, label: int, z_from: int) -> int | None:
        """Give the voxels of ``label`` at slices ``>= z_from`` a new id (``max + 1``).

        Parameters
        ----------
        label : int
            Id to split (e.g. the clicked mask's id).
        z_from : int
            First slice of the new object.

        Returns
        -------
        int or None
            The new id, or None if refused (reason in ``status``): ``label`` is
            0 or the background id (``background=True``), or it has no voxels below ``z_from`` or none at/above it (the
            split would only rename the track).
        """
        label, z_from = int(label), int(z_from)
        if self.mode != "instance":
            self.status = "split: instance mode only"
            return None
        if label == 0:
            self.status = "split refused: id 0 is background"
            return None
        if self.background and label == BACKGROUND_ID:
            self.status = f"split refused: id {BACKGROUND_ID} is annotated background, not a cell"
            return None
        zs = self.label_extent(label)
        if not any(z >= z_from for z in zs) or not any(z < z_from for z in zs):
            self.status = f"split refused: label {label} has no slices on both sides of z={z_from}"
            return None
        new = self._next_label()
        tail = self._labels[z_from:].reshape(-1)
        idx = np.flatnonzero(tail == label) + z_from * self._labels.shape[1] * self._labels.shape[2]
        self._apply_track_op(idx, new, f"split label {label} at z={z_from}: z>={z_from} -> label {new}")
        if self.mode == "instance" and (self.auto_increment or int(self.active_label) == new):
            self.active_label = self._next_label()  # the active id must not silently join the new track
        return new

    def delete_id(self, label: int, z: int | None = None) -> bool:
        """Remove a whole mask: set ``label`` to 0 on slice ``z`` (``None``: every slice).

        Undoable like merge / split (ctrl+z in the widget, :meth:`undo_track_op`).
        Works for any nonzero id, including ``BACKGROUND_ID``.

        Parameters
        ----------
        label : int
            Id to remove (e.g. the clicked mask's id).
        z : int, optional
            Slice to remove it from; None (default) removes it from all slices.

        Returns
        -------
        bool
            True if voxels changed; False (reason in ``status``) if ``label`` is
            0, ``z`` is out of range, or ``label`` is absent there.
        """
        label = int(label)
        if label == 0:
            self.status = "delete refused: id 0 is already unannotated"
            return False
        nz = self._labels.shape[0]
        if z is None:
            idx = np.flatnonzero(self._labels.reshape(-1) == label)
            where = "all slices"
        else:
            z = int(z)
            if not 0 <= z < nz:
                self.status = f"delete refused: z={z} out of range [0, {nz - 1}]"
                return False
            plane = self._labels.shape[1] * self._labels.shape[2]
            idx = np.flatnonzero(self._labels[z].reshape(-1) == label) + z * plane
            where = f"z={z}"
        if idx.size == 0:
            self.status = f"delete refused: label {label} not present on {where}"
            return False
        if z is None:
            n = len(np.unique(idx // (self._labels.shape[1] * self._labels.shape[2])))
            where = f"all slices ({n} slice{'s' if n != 1 else ''})"
        self._apply_track_op(idx, 0, f"deleted label {label} on {where} ({idx.size} px)")
        return True

    def undo_track_op(self) -> bool:
        """Undo the most recent :meth:`merge_ids` / :meth:`split_id` / :meth:`delete_id` (last few kept).

        Restores the voxels the operation changed that still hold the id it
        wrote; voxels edited since (e.g. erased or painted with another id on
        a slice whose browser undo history was dropped on a z change) are left
        alone and counted in ``status``. Returns False if there is nothing to
        undo.
        """
        if not self._track_undo:
            self.status = "no track operation to undo"
            return False
        _, idx, old, written, what = self._track_undo.pop()
        flat = self._labels.reshape(-1)
        keep = flat[idx] == written
        flat[idx[keep]] = old[keep]
        self._after_volume_edit(idx)
        skipped = int(keep.size - keep.sum())
        self.status = f"undone: {what}" + (f" ({skipped} voxels edited since were left as they are)" if skipped else "")
        return True

    # ------------------------------------------------------------------ API
    @property
    def labels(self) -> np.ndarray:
        """Copy of the labels (``uint32``) in (Z)YX order (YX without Z), whatever the input's axis order."""
        return _restore_axes(self._labels.astype(np.uint32), self._spatial_axes)

    def labels_signed(self) -> np.ndarray:
        """Copy of the labels as ``int32`` with ``BACKGROUND_ID`` -> -1 ((Z)YX).

        Signed convention for sparse training: 0 unannotated, -1 annotated
        background, 1.. cells. Without ``background`` there is no background
        id and this is ``labels`` as ``int32``.
        """
        return _restore_axes(self._signed(self._labels), self._spatial_axes)

    def _signed(self, lab: np.ndarray) -> np.ndarray:
        out = lab.astype(np.int32)
        if self.background:
            out[lab == BACKGROUND_ID] = -1
        return out

    def set_labels(self, labels) -> None:
        """Replace all labels ((Z)YX order) and refresh the view.

        Accepts unsigned ids in ``[0, 65535]`` and, with ``background=True``,
        the signed convention of :meth:`labels_signed` (negative values become
        ``BACKGROUND_ID``). Clears the track undo history and the browser's
        per-slice undo history (ctrl+z cannot bring back the replaced labels).
        """
        lab = np.asarray(labels)
        lab = lab[np.newaxis] if lab.ndim == 2 else lab
        if lab.shape != self._labels.shape:
            raise ValueError(f"expected labels of shape {self._labels.shape}, got {lab.shape}")
        self._labels = _coerce_label_values(lab, self.background)
        self._slice_hist = [None] * self._labels.shape[0]
        self._track_undo.clear()
        self.track_undo_ids = []
        with self.hold_sync():
            self._send_slice()
            self.labels_rev += 1  # reload even if the bytes equal the last labels_to_js
            self.undo_epoch += 1  # the browser's slice snapshots predate the new labels
        if self.mode == "instance":
            self.active_label = self._next_label()
        self._update_extent()
        self._update_counts()

    def annotated_slices(self) -> list[int]:
        """Indices of z-slices containing at least one labelled pixel (cells or annotated background)."""
        return [int(z) for z in np.flatnonzero(self._labels.reshape(self._labels.shape[0], -1).any(axis=1))]

    def save(
        self, out_dir, name: str, slices: str = "annotated", overwrite: bool = False, signed: bool = False,
        stacks: str = "current",
    ) -> list[Path]:
        """Write training pairs ``<name>[_z<k>]_img.tif`` / ``..._masks.tif``.

        Images are the original (unscaled) data as ``(C, Y, X)`` per slice;
        masks are ``uint32`` ``(Y, X)`` (``int32`` with ``signed=True``). This matches Cellpose's
        ``io.load_train_test_data(dir, image_filter="_img", mask_filter="_masks")``.
        Apply any training normalization afterwards (e.g. Cellpose with
        ``normalize={"normalize": False}`` on pre-normalized data).

        Parameters
        ----------
        slices : {"annotated", "all", "current"}
        signed : bool
            Write masks as ``int32`` in the signed convention of
            :meth:`labels_signed`: 0 unannotated, -1 annotated background
            (``BACKGROUND_ID``), 1.. cells. For sparse-training loaders that
            mask out 0; Cellpose's own loader does not interpret -1. Default
            False: ``uint32`` masks with the background id stored as 65535.
        stacks : {"current", "all"}
            With several stacks, ``"all"`` saves every stack as
            ``<name>_<stack name>...``. Stack names that map to the same file
            names raise ``ValueError`` before anything is written.

        Warns (and returns ``[]``) when no slice is selected, e.g. nothing is
        annotated yet.
        """
        import tifffile

        if slices not in ("annotated", "all", "current"):
            raise ValueError(f"slices must be one of 'annotated', 'all', 'current', got {slices!r}")
        if stacks not in ("current", "all"):
            raise ValueError(f"stacks must be one of 'current', 'all', got {stacks!r}")
        # (image, labels, current z, pixel sizes, file stem) per stack; parked stacks are
        # read in place, so saving never switches the shown stack
        jobs = [(self._image, self._labels, int(self.z), self._ps, name)]
        if stacks == "all" and len(self._parked) > 1:
            jobs = []
            for k, p in enumerate(self._parked):
                image, labels, zc, ps = (
                    (self._image, self._labels, int(self.z), self._ps) if k == self._stack_idx
                    else (p["image"], p["labels"], p["z"], p["ps"])
                )
                tag = re.sub(r"[^\w.-]+", "_", (self.stack_names[k] if k < len(self.stack_names) else f"stack{k}")).strip("_")
                jobs.append((image, labels, zc, ps, f"{name}_{tag or k}"))

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        targets = []
        for image, labels, zc, ps, stem0 in jobs:
            nz = labels.shape[0]
            if slices == "annotated":
                zs = [int(z) for z in np.flatnonzero(labels.reshape(nz, -1).any(axis=1))]
            else:
                zs = list(range(nz)) if slices == "all" else [zc]
            for z in zs:
                stem = stem0 if nz == 1 else f"{stem0}_z{z:03d}"
                targets.append((image, labels, z, ps, out / f"{stem}_img.tif", out / f"{stem}_masks.tif"))
        # check everything first so a refused save writes nothing
        paths = [p for *_, pi, pm in targets for p in (pi, pm)]
        dup = sorted(str(p) for p, n in Counter(paths).items() if n > 1)
        if dup:
            raise ValueError(f"several stacks would write {dup}: give the stacks distinct stack_names")
        if not overwrite:
            clash = [str(p) for p in paths if p.exists()]
            if clash:
                raise FileExistsError(f"{clash} exist (pass overwrite=True)")
        if not targets:
            warnings.warn(f"save: no {slices} slices to write; nothing written", UserWarning, stacklevel=2)
            return []
        written = []
        for image, labels, z, ps, pi, pm in targets:
            img = image[:, z]
            # axes (tifffile's shaped metadata) and the Y/X calibration (resolution tags in
            # pixels per cm), so image_io.load_image reads the pair back; Cellpose ignores both
            cal = {}
            if ps["unit"] == UM and ps["Y"] and ps["X"]:
                cal = dict(resolution=(1e4 / ps["X"], 1e4 / ps["Y"]), resolutionunit="CENTIMETER")
            one = img.shape[0] == 1
            # minisblack: 3 or 4 channels are channels, not an RGB(A) picture
            tifffile.imwrite(pi, img[0] if one else img, compression="zlib", photometric="minisblack",
                             metadata={"axes": "YX" if one else "CYX"}, **cal)
            tifffile.imwrite(pm, self._signed(labels[z]) if signed else labels[z].astype(np.uint32),
                             compression="zlib", metadata={"axes": "YX"}, **cal)
            written += [pi, pm]
        return written
