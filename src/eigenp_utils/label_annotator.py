# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anywidget",
#     "traitlets",
#     "numpy",
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

_DEFAULT_CHANNEL_COLORS = ["#ff00ff", "#00ff00", "#00ffff", "#ff8000", "#ffff00", "#ffffff", "#ff0000", "#0080ff"]


def _class_palette(n: int) -> list[str]:
    """Distinct colours for semantic classes (golden-ratio hue steps)."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((0.11 + i * 0.618034) % 1.0, 0.75, 1.0)
        out.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return out


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
        order) or by DataArray dims (other dims must have size 1).
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
    percentiles : (float, float)
        Display scaling to 8 bit, per channel over the whole volume.

    Attributes
    ----------
    labels : np.ndarray
        Current labels (``uint32``), spatial axes of the input.
    """

    _esm = _ASSETS.joinpath("label_annotator.js").read_text()
    _css = _ASSETS.joinpath("label_annotator.css").read_text()

    # geometry / data (Python -> JS)
    shape_zyx = traitlets.List(traitlets.Int(), default_value=[1, 1, 1]).tag(sync=True)
    z = traitlets.Int(0).tag(sync=True)
    image_to_js = traitlets.Bytes(b"").tag(sync=True)
    labels_to_js = traitlets.Bytes(b"").tag(sync=True)
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

    # labelling
    mode = traitlets.Unicode("instance").tag(sync=True)
    class_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    class_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
    # instance colours: labels_cmap palette, id k -> palette[(k - 1) % len]
    instance_palette = traitlets.List(traitlets.Unicode()).tag(sync=True)
    active_label = traitlets.Int(1).tag(sync=True)
    tool = traitlets.Unicode("brush").tag(sync=True)  # brush | eraser | fill | picker
    brush_radius = traitlets.Int(2).tag(sync=True)
    overwrite = traitlets.Bool(False).tag(sync=True)
    auto_fill = traitlets.Bool(True).tag(sync=True)
    auto_increment = traitlets.Bool(True).tag(sync=True)
    new_label_trigger = traitlets.Int(0).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

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
        **kwargs,
    ):
        if mode not in ("instance", "semantic"):
            raise ValueError("mode must be 'instance' or 'semantic'")
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
        colors = list(channel_colors) if channel_colors else (["#ffffff"] if c == 1 else _DEFAULT_CHANNEL_COLORS[:c])
        if len(names) != c or len(colors) != c:
            raise ValueError(f"need {c} channel names/colors")
        classes = list(classes) if classes else (["foreground"] if mode == "semantic" else [])

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
            self.auto_increment = mode == "instance"
            self.brush_radius = brush_radius
            self.active_label = self._next_label() if mode == "instance" else 1
            self.z = nz // 2
            self._send_slice()

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
        if self.mode == "instance" and self.auto_increment:
            self.active_label = self._next_label()

    @traitlets.observe("new_label_trigger")
    def _on_new_label(self, change):
        self.active_label = self._next_label()

    def _next_label(self) -> int:
        return int(self._labels.max()) + 1 if self._labels.size else 1

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
        self._send_slice()
        if self.mode == "instance":
            self.active_label = self._next_label()

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
        written = []
        for z in zs:
            stem = name if self._labels.shape[0] == 1 else f"{name}_z{z:03d}"
            pi, pm = out / f"{stem}_img.tif", out / f"{stem}_masks.tif"
            for p in (pi, pm):
                if p.exists() and not overwrite:
                    raise FileExistsError(f"{p} exists (pass overwrite=True)")
            img = self._image[:, z]
            tifffile.imwrite(pi, img[0] if img.shape[0] == 1 else img, compression="zlib")
            tifffile.imwrite(pm, self._labels[z], compression="zlib")
            written += [pi, pm]
        return written
