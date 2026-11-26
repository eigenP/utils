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

from ._conventions import PALETTE, _is_single_colour, channel_color_hex, channel_spec, default_colors, default_display_range, robust_range
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


def _colors_from_coord(values) -> list | None:
    """Channel colours from a DataArray ``color`` coordinate (names, hex or RGB tuples); unusable entries -> None."""
    out = []
    for v in list(values):
        try:
            out.append(channel_spec(v))
        except (TypeError, ValueError):
            if not (v is None or (isinstance(v, str) and not v) or (isinstance(v, float) and np.isnan(v))):
                warnings.warn(f"channel colour {v!r} is not a colour or colormap name; using the default.",
                              UserWarning, stacklevel=3)
            out.append(None)
    return None if all(v is None for v in out) else out


def _rgb_hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*(int(round(255 * float(v))) for v in rgb[:3]))


def _to_czyx(image, axes: str | None) -> tuple[np.ndarray, str]:
    """Return (view with axes C, Z, Y, X; the input's axes string)."""
    dims = getattr(image, "dims", None)
    if dims is not None:  # xarray.DataArray, e.g. STCZYX from eigenp_utils.image_io
        da = image
        for d in list(da.dims):
            if d not in ("C", "Z", "Y", "X"):
                if da.sizes[d] != 1:
                    raise ValueError(f"dimension {d!r} has size {da.sizes[d]}; select a single index first")
                da = da.isel({d: 0})
        axes = "".join(str(d) for d in da.dims)
        arr = np.asarray(da.values)
    else:
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
    """Squeeze a (Z, Y, X) label volume back to the input's spatial axes."""
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
        order) or by DataArray dims (other dims must have size 1). For a
        DataArray, channel names default to the ``C`` coordinate and colours to
        a ``color`` coordinate along ``C`` (``image_io.set_channel_info``).
        A **list of stacks** (e.g. one 3D stack per cage / field of view; same
        channels, shapes may differ) shows one at a time with a stack selector
        (toolbar, ``stack`` trait, PageUp / PageDown). ``labels`` is then a
        matching list (or None), ``stack_names`` the names. Each stack keeps its
        own labels, z position and track-op undo history; ``labels`` returns the
        shown stack, :attr:`stack_labels` all of them, ``save(..., stacks="all")``
        writes all. ``axes`` applies to every stack.
    axes : str, optional
        Axis letters of ``image`` (e.g. ``"CYX"``, ``"ZYX"``, ``"CZYX"``);
        required for ndarrays with ndim > 2.
    labels : array-like, optional
        Initial labels with the spatial axes of ``image`` (``YX`` or ``ZYX``).
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
    percentiles : (float, float), optional
        Display scaling to 8 bit, per channel over the whole volume. None
        (default) uses the package's default display range, the same as the
        ortho viewers: 0 (or the data minimum if negative) to the robust
        99.99th percentile (the maximum for signal sparser than 0.01%).
        Explicit percentiles (e.g. ``(0.1, 99.99)``, the old default) fall
        back to the data maximum, with a warning, when they coincide.

    Changed (D02/D03, 2026-11): the default range was (p0.1, p99.99) with
    ``lo + 1`` for sparse channels (rendered binary); the default colours
    were magenta, green, cyan, ... and more than 8 channels raised;
    RGB-tuple colours in a DataArray ``color`` coordinate were ignored.
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
        is an ordinary id and nothing changes.

    Attributes
    ----------
    labels : np.ndarray
        Current labels (``uint32``), spatial axes of the input.
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

    Keyboard (viewport focused): o toggle filled/outlined display, b/e/f/p brush/eraser/fill/picker, g
    background brush (``background=True``), x delete mask (shift+click: all
    slices), m/s merge/split track, n new id, [ / ] brush radius,
    ArrowUp/ArrowDown or Alt+ArrowRight/Alt+ArrowLeft z+1/z-1, 0 fit, ctrl+z
    undo, space+drag or right/middle drag pan, wheel zoom, 1-9 class
    (semantic mode).
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

    # display (client-side only)
    channel_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_visible = traitlets.List(traitlets.Bool()).tag(sync=True)
    vmin = traitlets.List(traitlets.Int()).tag(sync=True)
    vmax = traitlets.List(traitlets.Int()).tag(sync=True)
    gamma = traitlets.List(traitlets.Float()).tag(sync=True)
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
        **kwargs,
    ):
        if mode not in ("instance", "semantic"):
            raise ValueError("mode must be 'instance' or 'semantic'")
        background = bool(background)
        if background and mode != "instance":
            raise ValueError("background=True is for instance mode; in semantic mode add a background class")
        if getattr(image, "dims", None) is not None and "C" in image.dims:
            # channel names / colours from the DataArray (image_io.set_channel_info)
            if channel_names is None and "C" in image.coords:
                channel_names = [str(v) for v in image.coords["C"].values]
            if channel_colors is None and "color" in image.coords and image.coords["color"].dims == ("C",):
                channel_colors = _colors_from_coord(image.coords["color"].values)
        extra = []  # further stacks (list input): [(czyx, labels or None)]
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
            for im, lb in zip(imgs[1:], labs[1:]):
                extra.append((_to_czyx(im, axes)[0], lb))
            image, labels = imgs[0], labs[0]
        else:
            names_in = []
        czyx, input_axes = _to_czyx(image, axes)
        for e, _ in extra:
            if e.shape[0] != czyx.shape[0]:
                raise ValueError(f"all stacks need the same channels: {czyx.shape[0]} vs {e.shape[0]}")
        self._image = czyx
        self._spatial_axes = "".join(a for a in input_axes if a in "ZYX")
        c, nz, ny, nx = czyx.shape
        self._limits = [_display_limits(czyx[i], percentiles) for i in range(c)]

        if labels is None:
            self._labels = np.zeros((nz, ny, nx), dtype=_STORE)
        else:
            lab = np.asarray(labels)
            lab = lab[np.newaxis] if lab.ndim == 2 else lab
            if lab.shape != (nz, ny, nx):
                raise ValueError(f"labels shape {np.asarray(labels).shape} does not match image spatial shape {(nz, ny, nx)}")
            self._labels = _coerce_label_values(lab, background)

        names = list(channel_names) if channel_names else [f"ch{i}" for i in range(c)]
        if len(names) != c:
            raise ValueError(f"need {c} channel names/colors")
        colors = _channel_colors(channel_colors, c)
        classes = list(classes) if classes else (["foreground"] if mode == "semantic" else [])
        # cached (nonzero ids, pixel counts) per slice: extent / counts readouts, next id
        self._slice_hist: list[tuple[np.ndarray, np.ndarray] | None] = [None] * nz
        # (serial, flat idx, old values, value written, what)
        self._track_undo: list[tuple[int, np.ndarray, np.ndarray, int, str]] = []
        # one counter for the whole widget (not per stack): serials never repeat, so
        # the browser can tell a new op from one it has mirrored already
        self._track_serial = 0

        # inactive stacks are parked here; the active one lives in self._image & co
        self._stack_idx = 0
        self._parked: list[dict | None] = [None] * (1 + len(extra))
        for k, (e_img, e_lab) in enumerate(extra, start=1):
            ez, ey, ex = e_img.shape[1:]
            if e_lab is None:
                el = np.zeros((ez, ey, ex), dtype=_STORE)
            else:
                el = np.asarray(e_lab)
                el = el[np.newaxis] if el.ndim == 2 else el
                if el.shape != (ez, ey, ex):
                    raise ValueError(f"stack {k}: labels shape {el.shape} does not match image {(ez, ey, ex)}")
                el = _coerce_label_values(el, background)
            self._parked[k] = dict(
                image=e_img, labels=el, limits=[_display_limits(e_img[i], percentiles) for i in range(c)],
                hist=[None] * ez, undo=[], z=ez // 2,
            )

        super().__init__(**kwargs)
        with self.hold_sync():
            self.stack_names = names_in
            self.shape_zyx = [nz, ny, nx]
            self.channel_names = names
            self.channel_colors = colors
            self.channel_visible = [True] * c
            self.vmin = [0] * c
            self.vmax = [255] * c
            self.gamma = [1.0] * c
            self.mode = mode
            self.class_names = classes
            self.class_colors = _class_palette(len(classes))
            self.instance_palette = [_rgb_hex(c) for c in LABELS_CMAP_COLORS]
            self.auto_fill = mode == "instance"
            self.auto_increment = mode == "instance" and not keep_label
            self.brush_radius = brush_radius
            self.show_adjacent = bool(show_adjacent)
            self.background = background
            self.active_label = self._next_label() if mode == "instance" else 1
            self.z = nz // 2
            self._send_slice()
            self._update_extent()
            self._update_counts()

    # ------------------------------------------------------------------ sync
    def _slice_u8(self, z: int) -> np.ndarray:
        out = np.empty(self._image.shape[:1] + self._image.shape[2:], dtype=np.uint8)
        for i, (lo, hi) in enumerate(self._limits):
            s = (np.asarray(self._image[i, z], dtype=np.float32) - lo) * (255.0 / (hi - lo))
            # NaN survives the clip and would warn in the uint8 cast: show it as 0
            np.nan_to_num(s, copy=False, nan=0.0, posinf=255.0, neginf=0.0)
            np.clip(s, 0, 255, out=s)
            out[i] = s.astype(np.uint8)
        return out

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
            undo=self._track_undo, z=int(self.z),
        )

    @traitlets.validate("stack")
    def _valid_stack(self, proposal):
        n = max(1, len(self._parked)) if hasattr(self, "_parked") else 1
        return int(min(max(proposal["value"], 0), n - 1))

    @traitlets.observe("stack")
    def _on_stack(self, change):
        if not hasattr(self, "_parked") or change["new"] == self._stack_idx:
            return
        self._parked[self._stack_idx] = self._park()
        self._stack_idx = new = change["new"]
        st = self._parked[new]
        self._parked[new] = None
        self._image, self._labels, self._limits = st["image"], st["labels"], st["limits"]
        self._slice_hist, self._track_undo = st["hist"], st["undo"]
        with self.hold_sync():
            nz, ny, nx = self._labels.shape
            self.shape_zyx = [nz, ny, nx]
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
        """Labels of every stack (copies, ``uint32``, the input's spatial axes), in input order."""
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
        """Copy of the labels (``uint32``) with the input's spatial axes."""
        return _restore_axes(self._labels.astype(np.uint32), self._spatial_axes)

    def labels_signed(self) -> np.ndarray:
        """Copy of the labels as ``int32`` with ``BACKGROUND_ID`` -> -1 (input's spatial axes).

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
        """Replace all labels (spatial axes of the input) and refresh the view.

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
        # (image, labels, current z, file stem) per stack; parked stacks are read in
        # place, so saving never switches the shown stack
        jobs = [(self._image, self._labels, int(self.z), name)]
        if stacks == "all" and len(self._parked) > 1:
            jobs = []
            for k, p in enumerate(self._parked):
                image, labels, zc = (self._image, self._labels, int(self.z)) if k == self._stack_idx else (p["image"], p["labels"], p["z"])
                tag = re.sub(r"[^\w.-]+", "_", (self.stack_names[k] if k < len(self.stack_names) else f"stack{k}")).strip("_")
                jobs.append((image, labels, zc, f"{name}_{tag or k}"))

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        targets = []
        for image, labels, zc, stem0 in jobs:
            nz = labels.shape[0]
            if slices == "annotated":
                zs = [int(z) for z in np.flatnonzero(labels.reshape(nz, -1).any(axis=1))]
            else:
                zs = list(range(nz)) if slices == "all" else [zc]
            for z in zs:
                stem = stem0 if nz == 1 else f"{stem0}_z{z:03d}"
                targets.append((image, labels, z, out / f"{stem}_img.tif", out / f"{stem}_masks.tif"))
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
        for image, labels, z, pi, pm in targets:
            img = image[:, z]
            tifffile.imwrite(pi, img[0] if img.shape[0] == 1 else img, compression="zlib")
            tifffile.imwrite(pm, self._signed(labels[z]) if signed else labels[z].astype(np.uint32), compression="zlib")
            written += [pi, pm]
        return written
