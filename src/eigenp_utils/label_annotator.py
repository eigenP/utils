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
  the authoritative label volume (``uint32``).
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

Only numpy, anywidget and traitlets are required (core dependencies; also
works in marimo WASM). ``tifffile`` is imported lazily by ``save``.
"""

from __future__ import annotations

import colorsys
import struct
from pathlib import Path

import anywidget
import numpy as np
import traitlets

from .labels_cmap_data import LABELS_CMAP_COLORS

_ASSETS = Path(__file__).parent
_IMG_HDR = struct.Struct("<4I")
_LBL_HDR = struct.Struct("<3I")
_MAX_WIRE_LABEL = np.iinfo(np.uint16).max
_TRACK_UNDO_MAX = 20  # whole-volume operations kept for undo

_DEFAULT_CHANNEL_COLORS = ["#ff00ff", "#00ff00", "#00ffff", "#ff8000", "#ffff00", "#ffffff", "#ff0000", "#0080ff"]


def _class_palette(n: int) -> list[str]:
    """Distinct colours for semantic classes (golden-ratio hue steps)."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((0.11 + i * 0.618034) % 1.0, 0.75, 1.0)
        out.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return out


def _color_hex(c) -> str:
    """Hex colour for the browser from a colour or colormap name ('gray' -> its end colour, white)."""
    import matplotlib
    import matplotlib.colors as mcolors

    if isinstance(c, str):
        try:
            return mcolors.to_hex(matplotlib.colormaps[c](1.0)[:3])
        except (KeyError, ValueError):
            pass
    return mcolors.to_hex(c)


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
    flat = arr.reshape(-1)
    if flat.size > max_samples:
        rng = np.random.default_rng(seed)
        flat = flat[rng.integers(0, flat.size, max_samples)]
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(flat, [lo, hi])
    if b <= a:
        b = a + 1.0
    return float(a), float(b)


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
    image : array-like or xarray.DataArray
        2D/3D/4D image. Axes given by ``axes`` (subset of ``"CZYX"`` in any
        order) or by DataArray dims (other dims must have size 1). For a
        DataArray, channel names default to the ``C`` coordinate and colours to
        a ``color`` coordinate along ``C`` (``image_io.set_channel_info``).
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
        Colours may be any matplotlib colour or colormap name (a colormap such
        as ``"gray"`` maps to its end colour).
    percentiles : (float, float)
        Display scaling to 8 bit, per channel over the whole volume.
    brush_radius : int
        Brush radius in pixels.
    keep_label : bool
        Instance mode: keep the active id after each stroke or fill instead of
        advancing to the next free id (``auto_increment = not keep_label``).
        Use it to paint the same object on consecutive z-slices.
    show_adjacent : bool
        Show outlines of the labels on slices z-1 (dotted) and z+1 (faint) over
        the current slice, each in its id's colour ("onion skin").

    Attributes
    ----------
    labels : np.ndarray
        Current labels (``uint32``), spatial axes of the input.
    active_label_extent : str
        z extent of the active id, e.g. ``"label 37: z 12-19 (8 slices)"``.

    Notes
    -----
    z-linking ground truth: in instance mode an id is one 3D object, so the
    same id on several slices links those 2D masks. Tools for this:
    ``keep label`` (paint one id slice after slice), the ``z±1`` outlines,
    ``merge`` (click a mask: its id is replaced by the active id in all
    slices; :meth:`merge_ids`) and ``split`` (click a mask at slice z: its
    voxels at slices >= z get a new id; :meth:`split_id`). Both whole-track
    operations are undoable (ctrl+z / undo, or :meth:`undo_track_op`).

    Keyboard (viewport focused): b/e/f/p brush/eraser/fill/picker, m/s
    merge/split track, n new id, [ / ] brush radius, ArrowUp/ArrowDown or
    Alt+ArrowRight/Alt+ArrowLeft z+1/z-1, 0 fit, ctrl+z undo, space+drag or
    right/middle drag pan, wheel zoom, 1-9 class (semantic mode).
    """

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
    # edits (JS -> Python); Any because buffers may arrive as memoryview
    labels_from_js = traitlets.Any(None, allow_none=True).tag(sync=True)

    # display (client-side only)
    channel_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_visible = traitlets.List(traitlets.Bool()).tag(sync=True)
    vmin = traitlets.List(traitlets.Int()).tag(sync=True)
    vmax = traitlets.List(traitlets.Int()).tag(sync=True)
    gamma = traitlets.List(traitlets.Float()).tag(sync=True)
    opacity = traitlets.Float(0.45).tag(sync=True)
    outline_only = traitlets.Bool(False).tag(sync=True)
    show_adjacent = traitlets.Bool(False).tag(sync=True)

    # labelling
    mode = traitlets.Unicode("instance").tag(sync=True)
    class_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    class_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    # instance colours: labels_cmap palette, id k -> palette[(k - 1) % len]
    instance_palette = traitlets.List(traitlets.Unicode()).tag(sync=True)
    active_label = traitlets.Int(1).tag(sync=True)
    tool = traitlets.Unicode("brush").tag(sync=True)  # brush | eraser | fill | picker | merge | split
    brush_radius = traitlets.Int(2).tag(sync=True)
    overwrite = traitlets.Bool(False).tag(sync=True)
    auto_fill = traitlets.Bool(True).tag(sync=True)
    auto_increment = traitlets.Bool(True).tag(sync=True)
    new_label_trigger = traitlets.Int(0).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)
    active_label_extent = traitlets.Unicode("").tag(sync=True)
    # whole-track operations: JS -> Python requests
    # {"op": "merge", "src", "dst"} | {"op": "split", "id", "z"} | {"op": "undo", "serial"},
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
        percentiles: tuple[float, float] = (0.1, 99.99),
        brush_radius: int = 2,
        keep_label: bool = False,
        show_adjacent: bool = False,
        **kwargs,
    ):
        if mode not in ("instance", "semantic"):
            raise ValueError("mode must be 'instance' or 'semantic'")
        if getattr(image, "dims", None) is not None and "C" in image.dims:
            # channel names / colours from the DataArray (image_io.set_channel_info)
            if channel_names is None and "C" in image.coords:
                channel_names = [str(v) for v in image.coords["C"].values]
            if channel_colors is None and "color" in image.coords and image.coords["color"].dims == ("C",):
                cols = list(image.coords["color"].values)
                if all(isinstance(v, str) and v for v in cols):
                    channel_colors = cols
        czyx, input_axes = _to_czyx(image, axes)
        self._image = czyx
        self._spatial_axes = "".join(a for a in input_axes if a in "ZYX")
        c, nz, ny, nx = czyx.shape
        self._limits = [_robust_limits(czyx[i], *percentiles) for i in range(c)]

        if labels is None:
            self._labels = np.zeros((nz, ny, nx), dtype=np.uint32)
        else:
            lab = np.asarray(labels)
            lab = lab[np.newaxis] if lab.ndim == 2 else lab
            if lab.shape != (nz, ny, nx):
                raise ValueError(f"labels shape {np.asarray(labels).shape} does not match image spatial shape {(nz, ny, nx)}")
            if lab.size and (lab.min() < 0 or lab.max() > _MAX_WIRE_LABEL):
                raise ValueError(f"labels must be in [0, {_MAX_WIRE_LABEL}]")
            self._labels = lab.astype(np.uint32, copy=True)

        names = list(channel_names) if channel_names else [f"ch{i}" for i in range(c)]
        colors = [_color_hex(v) for v in channel_colors] if channel_colors else (["#ffffff"] if c == 1 else _DEFAULT_CHANNEL_COLORS[:c])
        if len(names) != c or len(colors) != c:
            raise ValueError(f"need {c} channel names/colors")
        classes = list(classes) if classes else (["foreground"] if mode == "semantic" else [])
        self._slice_ids: list[np.ndarray | None] = [None] * nz  # cached ids per slice (extent readout)
        # (serial, flat idx, old values, value written, what)
        self._track_undo: list[tuple[int, np.ndarray, np.ndarray, int, str]] = []
        self._track_serial = 0

        super().__init__(**kwargs)
        with self.hold_sync():
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
            self.active_label = self._next_label() if mode == "instance" else 1
            self.z = nz // 2
            self._send_slice()
            self._update_extent()

    # ------------------------------------------------------------------ sync
    def _slice_u8(self, z: int) -> np.ndarray:
        out = np.empty(self._image.shape[:1] + self._image.shape[2:], dtype=np.uint8)
        for i, (lo, hi) in enumerate(self._limits):
            s = (np.asarray(self._image[i, z], dtype=np.float32) - lo) * (255.0 / (hi - lo))
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

    @traitlets.validate("z")
    def _valid_z(self, proposal):
        nz = self._image.shape[1] if hasattr(self, "_image") else 1
        return int(min(max(proposal["value"], 0), nz - 1))

    @traitlets.observe("z")
    def _on_z(self, change):
        if hasattr(self, "_image"):
            self._send_slice()

    @traitlets.observe("labels_from_js")
    def _on_labels_from_js(self, change):
        buf = change["new"]
        if buf is None or len(bytes(buf)) == 0:
            return
        z, lab = decode_label_slice(buf)
        if lab.shape != self._labels.shape[1:] or not 0 <= z < self._labels.shape[0]:
            self.status = f"ignored label message for z={z} with shape {lab.shape}"
            return
        self._labels[z] = lab
        self._slice_ids[z] = None
        if self.mode == "instance" and self.auto_increment:
            self.active_label = self._next_label()
        self._update_extent()

    @traitlets.observe("new_label_trigger")
    def _on_new_label(self, change):
        self.active_label = self._next_label()

    def _next_label(self) -> int:
        """``max + 1``; once that exceeds the wire limit (65535), the smallest unused id."""
        nxt = int(self._labels.max()) + 1 if self._labels.size else 1
        if nxt <= _MAX_WIRE_LABEL:
            return nxt
        free = np.flatnonzero(np.bincount(self._labels.ravel(), minlength=_MAX_WIRE_LABEL + 1)[1:] == 0)
        if free.size == 0:
            raise ValueError(f"all ids 1..{_MAX_WIRE_LABEL} are in use")
        return int(free[0]) + 1

    @traitlets.validate("active_label")
    def _valid_active_label(self, proposal):
        v = int(proposal["value"])
        if not 0 <= v <= _MAX_WIRE_LABEL:
            # the browser paints uint16: a larger id would wrap (65536 -> 0, 65537 -> 1)
            raise traitlets.TraitError(f"active_label must be in [0, {_MAX_WIRE_LABEL}], got {v}")
        return v

    @traitlets.observe("active_label")
    def _on_active_label(self, change):
        if hasattr(self, "_slice_ids"):
            self._update_extent()

    def _ids_in_slice(self, z: int) -> np.ndarray:
        ids = self._slice_ids[z]
        if ids is None:
            # ids <= 65535 (wire limit), so a bincount is a cheap O(n) unique
            ids = self._slice_ids[z] = np.flatnonzero(np.bincount(self._labels[z].ravel()))
        return ids

    def label_extent(self, label: int) -> list[int]:
        """z indices of the slices that contain ``label`` (sorted)."""
        return [z for z in range(self._labels.shape[0]) if bool((self._ids_in_slice(z) == label).any())]

    def _update_extent(self):
        a = int(self.active_label)
        if self.mode != "instance":
            self.active_label_extent = ""
            return
        if a == 0:
            self.active_label_extent = "label 0 (background)"
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

    # ------------------------------------------------------- track operations
    @traitlets.observe("track_request")
    def _on_track_request(self, change):
        req = change["new"] or {}
        op = req.get("op")
        if op == "merge":
            self.merge_ids(int(req["src"]), int(req["dst"]))
        elif op == "split":
            self.split_id(int(req["id"]), int(req["z"]))
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
        self._track_undo.append((self._track_serial, idx, flat[idx].copy(), int(new_values), what))
        del self._track_undo[:-_TRACK_UNDO_MAX]
        flat[idx] = new_values
        self._after_volume_edit(idx)
        self.status = what

    def _after_volume_edit(self, idx: np.ndarray) -> None:
        plane = self._labels.shape[1] * self._labels.shape[2]
        for z in np.unique(idx // plane):
            self._slice_ids[int(z)] = None
        with self.hold_sync():
            self.track_undo_ids = [u[0] for u in self._track_undo]
            self._send_labels()
        self._update_extent()

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
            either id is 0, they are equal, or ``src`` is absent. If both ids
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
            0, or it has no voxels below ``z_from`` or none at/above it (the
            split would only rename the track).
        """
        label, z_from = int(label), int(z_from)
        if self.mode != "instance":
            self.status = "split: instance mode only"
            return None
        if label == 0:
            self.status = "split refused: id 0 is background"
            return None
        zs = self.label_extent(label)
        if not any(z >= z_from for z in zs) or not any(z < z_from for z in zs):
            self.status = f"split refused: label {label} has no slices on both sides of z={z_from}"
            return None
        new = self._next_label()
        if new > _MAX_WIRE_LABEL:
            self.status = f"split refused: new id {new} > {_MAX_WIRE_LABEL}"
            return None
        tail = self._labels[z_from:].reshape(-1)
        idx = np.flatnonzero(tail == label) + z_from * self._labels.shape[1] * self._labels.shape[2]
        self._apply_track_op(idx, new, f"split label {label} at z={z_from}: z>={z_from} -> label {new}")
        if self.mode == "instance" and (self.auto_increment or int(self.active_label) == new):
            self.active_label = self._next_label()  # the active id must not silently join the new track
        return new

    def undo_track_op(self) -> bool:
        """Undo the most recent :meth:`merge_ids` / :meth:`split_id` (last few kept).

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
        return _restore_axes(self._labels.copy(), self._spatial_axes)

    def set_labels(self, labels) -> None:
        """Replace all labels (spatial axes of the input) and refresh the view."""
        lab = np.asarray(labels)
        lab = lab[np.newaxis] if lab.ndim == 2 else lab
        if lab.shape != self._labels.shape:
            raise ValueError(f"expected labels of shape {self._labels.shape}, got {lab.shape}")
        if lab.size and (lab.min() < 0 or lab.max() > _MAX_WIRE_LABEL):
            raise ValueError(f"labels must be in [0, {_MAX_WIRE_LABEL}]")
        self._labels = lab.astype(np.uint32, copy=True)
        self._slice_ids = [None] * self._labels.shape[0]
        self._track_undo.clear()
        self.track_undo_ids = []
        with self.hold_sync():
            self._send_slice()
            self.labels_rev += 1  # reload even if the bytes equal the last labels_to_js
        if self.mode == "instance":
            self.active_label = self._next_label()
        self._update_extent()

    def annotated_slices(self) -> list[int]:
        """Indices of z-slices containing at least one labelled pixel."""
        return [int(z) for z in np.flatnonzero(self._labels.reshape(self._labels.shape[0], -1).any(axis=1))]

    def save(self, out_dir, name: str, slices: str = "annotated", overwrite: bool = False) -> list[Path]:
        """Write training pairs ``<name>[_z<k>]_img.tif`` / ``..._masks.tif``.

        Images are the original (unscaled) data as ``(C, Y, X)`` per slice;
        masks are ``uint32`` ``(Y, X)``. This matches Cellpose's
        ``io.load_train_test_data(dir, image_filter="_img", mask_filter="_masks")``.
        Apply any training normalization afterwards (e.g. Cellpose with
        ``normalize={"normalize": False}`` on pre-normalized data).

        Parameters
        ----------
        slices : {"annotated", "all", "current"}
        """
        import tifffile

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        zs = {"annotated": self.annotated_slices(), "all": list(range(self._labels.shape[0])), "current": [self.z]}[slices]
        targets = []
        for z in zs:
            stem = name if self._labels.shape[0] == 1 else f"{name}_z{z:03d}"
            targets.append((z, out / f"{stem}_img.tif", out / f"{stem}_masks.tif"))
        if not overwrite:  # check everything first so a refused save writes nothing
            clash = [str(p) for _, pi, pm in targets for p in (pi, pm) if p.exists()]
            if clash:
                raise FileExistsError(f"{clash} exist (pass overwrite=True)")
        written = []
        for z, pi, pm in targets:
            img = self._image[:, z]
            tifffile.imwrite(pi, img[0] if img.shape[0] == 1 else img, compression="zlib")
            tifffile.imwrite(pm, self._labels[z], compression="zlib")
            written += [pi, pm]
        return written
