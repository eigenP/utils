# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anywidget",
#     "traitlets",
#     "matplotlib",
#     "numpy",
#     "scipy",
# ]
# ///

import logging
import os
import pathlib
import warnings

_log = logging.getLogger(__name__)

def _safe_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        # Handle zero-dimensional numpy arrays like np.array(1.5)
        if hasattr(v, 'size') and v.size == 1 and hasattr(v, 'item'):
            return float(v.item())
        raise ValueError(f"Could not convert {v} to float")

def _parse_zyx_tuple_or_dict(val, default_val=None, warn=True):
    """
    Parse a (Z, Y, X) tuple/list or a {'Z', 'Y', 'X'} dict into three floats.

    Used for slab positions / thicknesses (0 is valid there, nothing is
    validated). Pixel sizes go through `_viewer_pixel_sizes` (the shared
    parser: units, validation, missing-Z rule).

    Missing or None entries (e.g. an uncalibrated axis from
    ``image_io.get_tiff_voxel_size``) become ``default_val``; a warning names
    the axis when a non-None default is substituted (unless ``warn`` is False,
    for callers that pass ``val`` on to a parser that will warn).
    """
    if val is None:
        return (default_val, default_val, default_val)
    if isinstance(val, dict):
        raw = (val.get('Z'), val.get('Y'), val.get('X'))
    else:
        try:
            raw = tuple(val)
        except TypeError:
            raw = ()
        if len(raw) != 3:
            raise ValueError(f"Expected dict with 'Z','Y','X' keys or a 3-element tuple/list, got {val}")

    out = []
    for axis, v in zip('ZYX', raw):
        if v is None:
            if default_val is not None and warn:
                warnings.warn(
                    f"No value for axis {axis} in {val!r}; using {default_val}.",
                    UserWarning, stacklevel=3,
                )
            out.append(default_val)
        else:
            out.append(_safe_float(v))
    return tuple(out)


try:
    from . import _conventions as _conv
except ImportError:  # run as a standalone script: load _conventions.py from next to this file
    import importlib.util as _ilu
    import sys as _sys

    _spec = _ilu.spec_from_file_location(
        "_eigenp_conventions", pathlib.Path(__file__).resolve().with_name("_conventions.py"))
    _conv = _ilu.module_from_spec(_spec)
    _sys.modules.setdefault("_eigenp_conventions", _conv)
    _spec.loader.exec_module(_conv)

PALETTE, PALETTE_MINIMUM, UM = _conv.PALETTE, _conv.PALETTE_MINIMUM, _conv.UM
channel_spec, channel_colormap, _conv_color_hex = _conv.channel_spec, _conv.channel_colormap, _conv.channel_color_hex
default_colors, default_display_range, per_channel = _conv.default_colors, _conv.default_display_range, _conv.per_channel
format_length, is_length_unit, is_pixel_unit, to_um = _conv.format_length, _conv.is_length_unit, _conv.is_pixel_unit, _conv.to_um
resolve_blending, resolve_pixel_sizes, sample_values = _conv.resolve_blending, _conv.resolve_pixel_sizes, _conv.sample_values
_robust_max, _check_blending = _conv._robust_max, _conv._check_blending
warn_deprecated = _conv.warn_deprecated


class _UncalibratedXYPixelSizes(dict):
    """Parsed pixel sizes whose X and Y fell back to 1.0 (a widget passes these on to keep a given Z step)."""


def _viewer_pixel_sizes(pixel_sizes, warn=True):
    """
    Pixel sizes of the ortho viewers: ``((pz, py, px), xy_calibrated, unit)``.

    Parsed by the package's shared parser (``_conventions.resolve_pixel_sizes``):
    a dict (keys case-insensitive, optional 'unit'), a (Z, Y, X) sequence, or
    None. Known length units are converted to µm, so ``unit`` is None (µm) or
    an unknown unit shown as given. A missing Z with known X and Y takes the
    finest of them (with a warning); other missing axes are drawn with size 1
    (pixel units) and warn, naming the axis. ``xy_calibrated`` is whether X and
    Y were given: only then is the scale bar labelled in physical units, else
    in pixels ('px'). A pixel unit ('pixel', 'px', ImageJ's uncalibrated
    unit) means uncalibrated: sizes 1 and a 'px' bar, without a warning.
    Scalars are rejected (the viewers take a Z, Y, X triple), as are
    non-finite and non-positive sizes.
    """
    if pixel_sizes is None:
        return (1.0, 1.0, 1.0), False, None
    if isinstance(pixel_sizes, _UncalibratedXYPixelSizes):
        return (float(pixel_sizes['Z']), float(pixel_sizes['Y']), float(pixel_sizes['X'])), False, None
    if not isinstance(pixel_sizes, dict) and np.ndim(pixel_sizes) == 0:
        raise ValueError(f"pixel_sizes must be a dict or a (Z, Y, X) sequence, got {pixel_sizes!r}")
    values, calibrated, unit = resolve_pixel_sizes(pixel_sizes, 3, warn=warn, stacklevel=3)
    missing = [a for a, v in zip('ZYX', values) if v is None]
    in_pixels = isinstance(pixel_sizes, dict) and any(
        str(k).upper() == 'UNIT' and is_pixel_unit(v) for k, v in pixel_sizes.items())
    if missing and warn and not in_pixels:
        warnings.warn(
            f"No value for axis {', '.join(missing)} in pixel_sizes; using 1.0 (pixel units).",
            UserWarning, stacklevel=3,
        )
    sizes = tuple(1.0 if v is None else float(v) for v in values)
    return sizes, bool(calibrated[1] and calibrated[2]), (None if unit == UM else unit)


def _xy_pixel_sizes_given(pixel_sizes):
    """Whether ``pixel_sizes`` gives the X and Y sizes, so the scale bar is labelled physically."""
    return _viewer_pixel_sizes(pixel_sizes, warn=False)[1]


def _length_unit(pixel_sizes):
    """The unit of the parsed sizes: None for µm (known units are converted), else an unknown unit."""
    return _viewer_pixel_sizes(pixel_sizes, warn=False)[2]

import anywidget
import traitlets
import matplotlib
import numpy as np
import sys
import io
import base64
from matplotlib import gridspec

from scipy import ndimage as ndi
from matplotlib.colors import PowerNorm, to_rgb, LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

def _new_figure(**kwargs):
    """A ``matplotlib.figure.Figure`` with an Agg canvas, not registered with pyplot (D05)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(**kwargs)
    FigureCanvasAgg(fig)
    return fig


def _pyplot_fignums():
    """Numbers of the open pyplot figures, without importing pyplot (empty if it was never imported)."""
    plt = sys.modules.get('matplotlib.pyplot')
    return set(plt.get_fignums()) if plt is not None else set()


def _close_pyplot_figures(nums):
    plt = sys.modules.get('matplotlib.pyplot')
    for num in nums if plt is not None else ():
        plt.close(num)


# The widget assets sit next to this module; resolving them from __file__ (not
# importlib.resources("eigenp_utils")) also works when the file runs standalone.
_ASSET_DIR = pathlib.Path(__file__).resolve().parent


class _PanelRotation:
    """
    Geometry of one rotated 2D panel, shared by the image resampling, the
    crosshairs and the click/hover inverse so all three agree exactly.

    A panel of ``shape`` (H, W) with pixel sizes ``(s_h, s_w)`` is displayed
    with pixel (r, c) centred at physical ``((c + 0.5) * s_w, (r + 0.5) * s_h)``.
    Rotation happens in physical space, counter-clockwise on screen (y down),
    about the centre of the panel. The rotated canvas has isotropic pixels of
    size ``p``; for isotropic input ``p`` equals the input pixel size and the
    result is identical to ``skimage.transform.rotate(resize=True)``.
    """

    def __init__(self, shape, angle_deg, s_h=1.0, s_w=1.0):
        H, W = int(shape[0]), int(shape[1])
        s_h, s_w = float(s_h), float(s_w)
        self.angle = float(angle_deg) if angle_deg else 0.0
        self.in_shape = (H, W)
        self.cx, self.cy = W * s_w / 2.0, H * s_h / 2.0
        if self.angle == 0.0:
            self.out_shape = (H, W)
            self.p_out = (s_h, s_w)
            return
        theta = np.deg2rad(self.angle)
        self.c, self.s = c, s = np.cos(theta), np.sin(theta)

        if s_h == s_w:
            # skimage's canvas: the rotated outermost pixel centres plus one pixel
            p = s_h
            hx, hy = (W - 1) / 2.0 * s_w, (H - 1) / 2.0 * s_h
            dx = np.array([-hx, -hx, hx, hx])
            dy = np.array([-hy, hy, hy, -hy])
            rx, ry = c * dx + s * dy, -s * dx + c * dy
            minx, miny = rx.min(), ry.min()
            self.out_shape = (int(np.around((ry.max() - miny) / p + 1)), int(np.around((rx.max() - minx) / p + 1)))
        else:
            # Anisotropic: resample onto the finer pixel size, capped at ~4x the
            # input pixel count, on a canvas holding the whole rotated physical box.
            ex = abs(c) * W * s_w + abs(s) * H * s_h
            ey = abs(s) * W * s_w + abs(c) * H * s_h
            p = max(min(s_h, s_w), np.sqrt(ex * ey / max(4 * H * W, 512 * 512)))
            out_w = max(1, int(np.ceil(ex / p - 1e-6)))
            out_h = max(1, int(np.ceil(ey / p - 1e-6)))
            self.out_shape = (out_h, out_w)
            minx, miny = -(out_w - 1) * p / 2.0, -(out_h - 1) * p / 2.0
        self.p_out = (p, p)
        # Output pixel (r, c) is centred at rotated offset (minx + c p, miny + r p)
        # from the panel centre, i.e. displayed at ((c + 0.5) p, (r + 0.5) p).
        self.offx, self.offy = 0.5 * p - minx, 0.5 * p - miny

        # Inverse map in (row, col) index order: input = M @ output + offset
        self.M = np.array([[c * p / s_h, s * p / s_h],
                           [-s * p / s_w, c * p / s_w]])
        self.offset = np.array([(s * minx + c * miny + self.cy) / s_h - 0.5,
                                (c * minx - s * miny + self.cx) / s_w - 0.5])

    def forward(self, x, y):
        """Unrotated physical (x, y) -> displayed physical (x, y)."""
        if self.angle == 0.0:
            return x, y
        dx, dy = np.asarray(x) - self.cx, np.asarray(y) - self.cy
        return (self.c * dx + self.s * dy + self.offx,
                -self.s * dx + self.c * dy + self.offy)

    def inverse(self, x, y):
        """Displayed physical (x, y) -> unrotated physical (x, y)."""
        if self.angle == 0.0:
            return x, y
        dx, dy = np.asarray(x) - self.offx, np.asarray(y) - self.offy
        return (self.c * dx - self.s * dy + self.cx,
                self.s * dx + self.c * dy + self.cy)

    def warp(self, image, order=3, cval=0.0):
        """Resample a 2D or channels-last image onto the rotated canvas.

        The corners outside the input are filled with ``cval``; ``cval=nan``
        marks them as missing (the viewers then show the panel background).
        """
        image = np.asarray(image)
        if self.angle == 0.0:
            return image
        if not np.issubdtype(image.dtype, np.floating):
            image = image.astype(np.float64)
        nan_fill = cval is not None and np.isnan(cval)
        if nan_fill:
            cval = 0.0

        def _warp(plane):
            finite = np.isfinite(plane)
            all_finite = finite.all()
            src = plane if all_finite else np.where(finite, plane, 0.0)
            out = ndi.affine_transform(src, self.M, offset=self.offset, output_shape=self.out_shape,
                                       order=order, mode='grid-constant', cval=cval)
            if order > 1 and src.size:
                lo, hi = (src.min(), src.max()) if nan_fill else (min(src.min(), cval), max(src.max(), cval))
                out = np.clip(out, lo, hi)
            if nan_fill:
                inside = ndi.affine_transform(np.ones(src.shape), self.M, offset=self.offset,
                                              output_shape=self.out_shape, order=1,
                                              mode='grid-constant', cval=0.0)
                out[inside < 0.5] = np.nan
            if not all_finite:
                # Keep non-finite pixels local instead of letting the spline
                # spread one NaN over the whole panel.
                bad = ndi.affine_transform((~finite).astype(np.float64), self.M, offset=self.offset,
                                           output_shape=self.out_shape, order=1,
                                           mode='grid-constant', cval=0.0)
                out[bad > 0.5] = np.nan
            return out

        if image.ndim == 2:
            return _warp(image)
        return np.stack([_warp(image[..., k]) for k in range(image.shape[-1])], axis=-1)


def _rotate_expand(image, angle_deg, order=3, pixel_size=None):
    """
    Rotate a 2D (H, W) or channels-last (H, W, C) image counter-clockwise by
    ``angle_deg`` about its centre, expanding the canvas to fit the result.

    Pure scipy replacement for
    ``skimage.transform.rotate(image, angle_deg, resize=True, order=order, preserve_range=True)``:
    same output shape, rotation centre ``((W - 1) / 2, (H - 1) / 2)``, zero fill
    and clipping of spline overshoot to the input range. Non-finite pixels stay
    local (NaN in the output) instead of spreading through the spline.

    ``pixel_size=(s_h, s_w)`` rotates in physical space instead; the output then
    has isotropic pixels (see ``_PanelRotation``).

    Kept as the skimage-equivalence test oracle (the viewers use
    ``_PanelRotation`` directly).
    """
    s_h, s_w = (1.0, 1.0) if pixel_size is None else pixel_size
    return _PanelRotation(np.shape(image)[:2], angle_deg, s_h, s_w).warp(image, order=order)


class _PointRotation:
    """Rotation of scatter points about (cx, cy); same interface as ``_PanelRotation``."""

    def __init__(self, angle_deg, cx, cy):
        self.angle, self.cx, self.cy = float(angle_deg or 0.0), cx, cy

    def forward(self, x, y):
        return _rotate_points_2d(x, y, self.angle, self.cx, self.cy)

    def inverse(self, x, y):
        return _rotate_points_2d(x, y, -self.angle, self.cx, self.cy)


def _rotate_points_2d(px, py, angle_deg, cx, cy):
    """Rotate points about (cx, cy) with the same on-screen sense as ``_rotate_expand``."""
    if angle_deg == 0:
        return px, py
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rx = cos_t * (px - cx) + sin_t * (py - cy) + cx
    ry = -sin_t * (px - cx) + cos_t * (py - cy) + cy
    return rx, ry


def _get_rotated_line(x0, y0, x1, y1, angle_deg, orig_dim_w, orig_dim_h, s_w, s_h):
    """
    Transforms a 2D line segment (x0,y0)->(x1,y1) in physical space through 2D rotation.

    Kept as a test oracle (``_PanelRotation.forward`` on two points); the
    viewers use ``_PanelRotation`` directly.
    """
    if angle_deg == 0.0 or angle_deg is None:
        return [x0, x1], [y0, y1]
    g = _PanelRotation((orig_dim_h, orig_dim_w), angle_deg, s_h, s_w)
    rx, ry = g.forward(np.array([x0, x1], dtype=float), np.array([y0, y1], dtype=float))
    return [float(rx[0]), float(rx[1])], [float(ry[0]), float(ry[1])]


def _parse_rotate_view(val):
    """rotate_view -> (rot_z, rot_y, rot_x) in degrees (XY, XZ and ZY panels)."""
    if val is None:
        return 0.0, 0.0, 0.0
    if isinstance(val, (int, float)):
        return float(val), 0.0, 0.0
    try:
        val_list = list(val)
        if len(val_list) == 3:
            return float(val_list[0]), float(val_list[1]), float(val_list[2])
    except TypeError:
        pass
    raise ValueError(f"rotate_view must be a float or a tuple of 3 floats (Z, Y, X), got {val}")


# The package palette (_conventions.PALETTE), shared with LabelAnnotator.
_DEFAULT_CHANNEL_COLORS = PALETTE
# With blending='minimum' each channel runs from white (low) to its colour (high),
# so the defaults are the complements of the additive defaults ("inks" on paper):
# the composite then looks like an inverted additive composite.
_DEFAULT_INVERTED_CHANNEL_COLORS = PALETTE_MINIMUM

# Channel compositing modes (napari names; 'add'/'max'/'min' accepted as aliases):
# _conventions.BLENDING_MODES, validated by _conventions._check_blending.
# Slab projections (napari's volume rendering modes, applied along the view axis).
_PROJECTIONS = ('mip', 'minip', 'average', 'attenuated_mip')


def _check_projection(projection, attenuation=0.05):
    """Canonical projection name ('mip' for None) after validating ``attenuation``."""
    p = 'mip' if projection is None else projection
    if p not in _PROJECTIONS:
        raise ValueError(f"projection must be one of {_PROJECTIONS} (or None), got {projection!r}")
    if not (np.isfinite(attenuation) and attenuation >= 0):
        raise ValueError(f"attenuation must be a finite number >= 0, got {attenuation!r}")
    return p


def _channel_colormaps(colormap, n, blending=None):
    """
    Normalise a per-channel colormap argument to a list of length ``n``.

    A single string or colour tuple applies to the first channel; missing
    entries are filled with the default channel colours not already used
    (the inverted defaults for ``blending='minimum'``).
    """
    return default_colors(colormap, n, blending)


def _channel_specs(colormap, n, blending=None):
    """`_channel_colormaps` as canonical strings (registered Colormap instances -> names, tuples -> hex)."""
    return [channel_spec(c) for c in _channel_colormaps(colormap, n, blending)]


#: Public name of `_conventions.is_colormap_name`: True if ``c`` is a string
#: naming a registered matplotlib colormap.
is_colormap = _conv.is_colormap_name

_GUI_MIN_LUMA = 0.45


def _gui_channel_color(c, blending=None):
    """
    Colour of a channel's row header, histogram and slider in the (dark) GUI.

    The channel's colour (a colormap's end colour) is mixed with white just
    enough to reach a luma of 0.45, which keeps its hue: blue stays blue and
    red stays red, black becomes a neutral grey. ``blending`` is accepted for
    compatibility; every mode uses the same rule.

    Changed (D14 / R3-10, 2026-11): colours were darkened to 75% for a light
    page, and under 'minimum' blending every dark ink (black, blue, red,
    navy, ...) became the same '#9e9e9e'.
    """
    col = resolve_color(c)
    try:
        rgb = np.asarray(mcolors.to_rgb(col), dtype=float)
    except (ValueError, TypeError):
        return col
    luma = float(rgb @ [0.2126, 0.7152, 0.0722])
    if luma < _GUI_MIN_LUMA:
        t = (_GUI_MIN_LUMA - luma) / (1.0 - luma)
        rgb = rgb + t * (1.0 - rgb)
    return mcolors.to_hex(np.clip(rgb, 0.0, 1.0))


def get_gui_channel_color(c):
    """
    Darkens a given color to improve contrast for GUI elements like channel labels and histograms.
    Converts color to RGB, multiplies by 0.75, and returns a hex string.
    """
    try:
        rgb = mcolors.to_rgb(resolve_color(c))
        darkened_rgb = tuple(val * 0.75 for val in rgb)
        return mcolors.to_hex(darkened_rgb)
    except (ValueError, TypeError):
        return resolve_color(c)

def resolve_color(c):
    """
    Attempts to resolve a string to a valid matplotlib color.
    If it is a valid colormap name, it returns the final color (at 1.0) of that colormap.

    Colormap names win over CSS colour names ('gray' gives white; 'pink'
    warns). A registered Colormap instance (e.g. ``eigenp_utils.i_blue``)
    gives its end colour; an unregistered one raises TypeError. Other strings
    and colour tuples are returned unchanged.

    Changed (D03, 2026-11): registered Colormap instances are accepted (they
    used to raise TypeError).
    """
    if isinstance(c, mcolors.Colormap):
        c = channel_spec(c)  # registered -> its name; unregistered -> TypeError

    if not isinstance(c, str):
        return c

    if is_colormap(c):
        return _conv_color_hex(c)  # end colour; warns for 'pink'

    try:
        # Check if it's already a valid color name or hex
        to_rgb(c)
        return c
    except ValueError:
        pass

    return c

def _join_docs(doc, extra):
    """A docstring followed by a shared section, both dedented (consistent on every Python version)."""
    import inspect
    return inspect.cleandoc(doc or '') + '\n\n' + inspect.cleandoc(extra) + '\n'


def _static_doc(func):
    """Append the input forms and display keywords shared by the static wrappers to a docstring."""
    func.__doc__ = _join_docs(func.__doc__, _STATIC_DOC)
    return func


_STATIC_DOC = """
    Input (every static wrapper): a ZYX (or YX) array, a list or tuple of
    them (one per channel), an ndarray with a channel axis named by
    ``channel_axis`` (e.g. ``channel_axis=0`` for CZYX; a 4D array without it
    raises ValueError rather than guessing between CZYX and TZYX), or an
    ``xarray.DataArray`` with C/Z/Y/X dims (other dims of size 1), whose
    ``C`` coordinate, ``color`` coordinate and calibrated coordinates give
    the default ``channel_labels``, ``colormap`` and ``pixel_sizes``.

    Display keyword arguments, shared by every static wrapper (see
    `show_zyx`): ``pixel_sizes`` ((Z, Y, X) or dict, converted to µm),
    ``figsize`` (inches; None, the default, sizes the figure from the
    physical extents to fit a 10-inch box), ``colormap`` (per channel), ``vmin`` / ``vmax`` /
    ``gamma`` / ``opacity`` (scalar or per channel; None takes the default
    display range of the whole volume), ``blending``, ``rotate_view``
    (degrees), ``channel_labels``, ``channel_label_fontsize_pt``,
    ``scale_bar_show``, ``scale_bar_fontsize_pt``, ``subplot_bg``,
    ``use_plt``; ``colors`` is the deprecated name of ``colormap``.
    Projecting wrappers also take ``projection`` / ``attenuation``.

    Changed (D12, 2026-11): every static wrapper takes the same display
    keywords and input forms (DataArrays, tuples, ``channel_axis``, 2D
    images); they used to differ per wrapper and reject DataArrays.
    ``figsize`` defaults to None (automatic from the physical extents; it
    was a fixed 10 x 10, mostly empty for elongated volumes), and the
    automatic font size of labels and scale bar is 3% of the figure width
    (two formulas used 3.5% and 3% of the height).
"""


# Copyright tnia 2021 - BSD License
@_static_doc
def show_zyx_slice(image_to_show, x, y, z, pixel_sizes=None, figsize=None, colormap=None, vmin = None, vmax=None, gamma = 1, use_plt=False, opacity=None, scale_bar_show=True, scale_bar_fontsize_pt=None,
                   *, colors=None, blending=None, rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None,
                   subplot_bg=None, channel_axis=None):
    """ extracts xy, xz, and zy slices at x, y, z of a 3D image and plots them

    Args:
        image_to_show: image to plot (see the input forms below)
        x (int): x index of the ZY slice
        y (int): y index of the XZ slice
        z (int): z index of the XY slice
        vmin, vmax (float, optional): display range; a bound left as None
            takes the default display range of the whole volume
            (`_conventions.default_display_range`: 0 or the negative data
            minimum, and the robust 99.99th percentile).

    Changed (D02, 2026-11): the default range is computed on the volume (it
    used to be the min/max of the three slices). Changed (D05, 2026-11):
    ``use_plt`` defaults to False (a plain Figure, not registered with pyplot).
    """
    colormap = _legacy_colors(colors, colormap)
    im, colormap, channel_labels, pixel_sizes = _ortho_input(
        image_to_show, colormap, channel_labels, pixel_sizes, channel_axis)

    def _slices(v):
        return v[z, :, :], v[:, y, :], np.flip(np.rot90(v[:, :, x], 1), 0)

    if isinstance(im, list):
        planes = [_slices(v) for v in im]
        slice_xy, slice_xz, slice_zy = ([p[k] for p in planes] for k in range(3))
        vmin, vmax = _channels_display_range(im, vmin, vmax)
    else:
        slice_xy, slice_xz, slice_zy = _slices(im)
        vmin, vmax = _volume_display_range(im, vmin, vmax)

    return show_zyx(slice_xy, slice_xz, slice_zy, pixel_sizes=pixel_sizes, figsize=figsize, colormap=colormap,
                    vmax=vmax, vmin=vmin, gamma=gamma, use_plt=use_plt, opacity=opacity, subplot_bg=subplot_bg,
                    rotate_view=rotate_view, channel_labels=channel_labels,
                    channel_label_fontsize_pt=channel_label_fontsize_pt, blending=blending,
                    scale_bar_show=scale_bar_show, scale_bar_fontsize_pt=scale_bar_fontsize_pt)


# Copyright tnia 2021 - BSD License
@_static_doc
def show_zyx_max(image_to_show, pixel_sizes=None, figsize=None, colormap=None, vmin = None, vmax=None, gamma = 1, colors = None, opacity = None, blending=None, projection=None, attenuation=0.05, scale_bar_show=True, scale_bar_fontsize_pt=None,
                 *, rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None, subplot_bg=None,
                 use_plt=False, channel_axis=None):
    """ plots max xy, xz, and zy projections of a 3D image (the whole volume)

    Args:
        image_to_show: image to plot (see the input forms below)
        projection, attenuation: see `show_zyx_projection_slabs`.
    """
    colormap = _legacy_colors(colors, colormap)
    return show_zyx_projection_slabs(
        image_to_show, None, None, None, pixel_sizes=pixel_sizes, figsize=figsize, projector=None,
        colormap=colormap, vmax=vmax, vmin=vmin, gamma=gamma, opacity=opacity, blending=blending,
        projection=projection, attenuation=attenuation, scale_bar_show=scale_bar_show,
        scale_bar_fontsize_pt=scale_bar_fontsize_pt, rotate_view=rotate_view, channel_labels=channel_labels,
        channel_label_fontsize_pt=channel_label_fontsize_pt, subplot_bg=subplot_bg, use_plt=use_plt,
        channel_axis=channel_axis)


@_static_doc
def show_zyx_projection(image_to_show, pixel_sizes=None, figsize=None, projector=None, colormap=None, vmin = None, vmax=None, gamma = 1, colors = None, opacity = None, blending=None, projection=None, attenuation=0.05, scale_bar_show=True, scale_bar_fontsize_pt=None,
                        *, rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None, subplot_bg=None,
                        use_plt=False, channel_axis=None):
    """ generates xy, xz, and zy projections of a 3D image (the whole volume) and plots them

    Args:
        image_to_show: image to plot (see the input forms below)
        projector: function ``f(volume, axis)`` to project with; None (default)
            is the maximum ignoring NaN voxels, like ``projection='mip'``.
        projection, attenuation: see `show_zyx_projection_slabs`.
    """
    colormap = _legacy_colors(colors, colormap)
    return show_zyx_projection_slabs(
        image_to_show, None, None, None, pixel_sizes=pixel_sizes, figsize=figsize, projector=projector,
        colormap=colormap, vmax=vmax, vmin=vmin, gamma=gamma, opacity=opacity, blending=blending,
        projection=projection, attenuation=attenuation, scale_bar_show=scale_bar_show,
        scale_bar_fontsize_pt=scale_bar_fontsize_pt, rotate_view=rotate_view, channel_labels=channel_labels,
        channel_label_fontsize_pt=channel_label_fontsize_pt, subplot_bg=subplot_bg, use_plt=use_plt,
        channel_axis=channel_axis)

# Copyright tnia 2021 - BSD License
def _draw_channel_label_bar(axLabels, fontsize_pt, as_image=False, x_ref=None):
    """Grey bar under the channel labels, exactly as wide as the XY panel.

    ``axLabels`` has the same x-position and width as the XY axes, so the bar
    spans axes x in [0, 1]. (A plotted line overshoots both ends by half its
    width because of matplotlib's default 'projecting' cap style.)

    The bar must also go through the same pixel rounding as what fills the XY
    panel: Agg places an image at ``x0 + ceil(width)`` but snaps a patch to the
    nearest pixel edges. ``as_image=True`` draws the bar as a 1x1 image (for
    panels filled by ``imshow``); otherwise it is a snapped rectangle (for
    panels whose frame is the axes background patch, e.g. the scatter viewer).

    ``x_ref`` (the XY axes) supplies the bar's x transform. Its final,
    aspect-adjusted position can differ from ``axLabels`` in the last bits, and
    when the panel edge falls on a half pixel (0.05 in at 150 dpi) that flips
    the rounding by a whole pixel on some platforms (Pyodide). Going through
    ``x_ref.transAxes`` gives x coordinates bit-identical to the panel's.
    """
    from matplotlib.transforms import blended_transform_factory
    x_ax = axLabels if x_ref is None else x_ref
    trans = blended_transform_factory(x_ax.transAxes, axLabels.transAxes)
    # An image bar spans the XY data limits through the XY data transform,
    # read at draw time, so both of its edges match an imshow panel whose
    # extent equals the limits bit for bit.
    data_trans = blended_transform_factory(x_ax.transData, axLabels.transAxes)
    thickness_in = max(1.0, fontsize_pt * 0.15) / 72.0
    ax_h_in = axLabels.get_position().height * axLabels.figure.get_figheight()
    h = thickness_in / ax_h_in  # thickness in axes fraction, centred on y = 0
    grey = (0.85, 0.85, 0.85)
    if as_image:
        from matplotlib.image import AxesImage

        class _BarImage(AxesImage):
            def get_extent(self):
                x0, x1 = x_ax.get_xlim()
                self._extent = (x0, x1, -h / 2.0, h / 2.0)
                return self._extent

            def get_window_extent(self, renderer=None):
                self.get_extent()
                return super().get_window_extent(renderer)

        im = _BarImage(axLabels, interpolation='nearest')
        im.get_extent()
        im.set_data(np.array([[grey]], dtype=float))
        im.set_transform(data_trans)
        im.set_clip_on(False)
        axLabels.add_artist(im)
    else:
        from matplotlib.patches import Rectangle
        axLabels.add_patch(Rectangle(
            (0.0, -h / 2.0), 1.0, h, transform=trans,
            facecolor=grey, edgecolor='none', linewidth=0,
            clip_on=False, snap=True,
        ))


_GAP_IN = 1.0 / 16.0   # gap between the ortho panels (inches)
_MARGIN_IN = 0.05      # figure margin (inches)


def _auto_fontsize_pt(fig_w_in):
    """
    The one automatic font size (pt) of channel labels and scale-bar labels:
    3% of the figure width (2.16 pt per inch), clipped to [8, 24] (R1-23).
    Width, not height, so thin volumes (short auto figures) keep readable text.
    """
    return float(max(8.0, min(24.0, float(fig_w_in) * 72 * 0.03)))


def _label_row_in(fontsize_pt):
    """Height (inches) of the channel-label row for a font size."""
    return 0.10 + fontsize_pt / 72.0


def _rotated_extent(w, h, angle_deg):
    """Bounding box (width, height) of a w x h panel rotated by ``angle_deg``."""
    if not angle_deg:
        return float(w), float(h)
    t = np.radians(angle_deg)
    c, s = abs(np.cos(t)), abs(np.sin(t))
    return float(w * c + h * s), float(w * s + h * c)


def _ortho_extents(wx, wy, wz, rotate_view=None):
    """Physical (column widths, row heights) of the ortho layout for extents X, Y, Z (after rotation)."""
    rz, ry, rx = _parse_rotate_view(rotate_view)
    xy = _rotated_extent(wx, wy, rz)
    xz = _rotated_extent(wx, wz, ry)
    zy = _rotated_extent(wz, wy, rx)
    return (max(xy[0], xz[0]), zy[0]), (max(xy[1], zy[1]), xz[1])


def _auto_figsize(cols, rows, has_labels, box=(10.0, 10.0), gap_in=_GAP_IN, label_fontsize_pt=None):
    """
    Automatic figure size (inches) for the ortho layout: the largest figure
    inside ``box`` whose panels keep their physical aspect and fill it, with
    the gaps, margins and (``has_labels``) the channel-label row (D12, R2-22).
    ``cols`` / ``rows`` are the physical (col1, col2) widths and (row1, row2)
    heights, e.g. from `_ortho_extents`.
    """
    W = max(float(cols[0]) + float(cols[1]), 1e-12)
    H = max(float(rows[0]) + float(rows[1]), 1e-12)
    fig_w = float(box[0])
    for _ in range(4):  # the label row depends on the font size, which depends on the width
        fs = label_fontsize_pt if label_fontsize_pt is not None else _auto_fontsize_pt(fig_w)
        extra_w = gap_in + 2.0 * _MARGIN_IN
        extra_h = (2 if has_labels else 1) * gap_in + 2.0 * _MARGIN_IN + (_label_row_in(fs) if has_labels else 0.0)
        scale = max(1e-6, min((box[0] - extra_w) / W, (box[1] - extra_h) / H))
        fig_w, fig_h = W * scale + extra_w, H * scale + extra_h
    return (float(fig_w), float(fig_h))


def show_zyx(xy, xz, zy, pixel_sizes=None, figsize=None, colormap=None, vmin=None, vmax=None, gamma=1, use_plt=False, colors=None, opacity=None, subplot_bg=None, rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None, gap_in=1.0/16.0, channel_label_colors=None, pixel_sizes_given=None, blending=None, scale_bar_show=True, scale_bar_fontsize_pt=None):
    """ shows pre-computed xy, xz and zy of a 3D image in a plot

    Args:
        xy (2d numpy array): xy projection
        xz (2d numpy array): xz projection
        zy (2d numpy array): zy projection
        figsize ((width, height) in inches, optional): None (default) fits
            the physical extents of the (rotated) panels, the gaps and the
            label row in a 10-inch box. Changed (D12, 2026-11): it was a
            fixed (10, 10).
        pixel_sizes (dict or (Z, Y, X) sequence, optional): converted to µm
            (a dict's 'unit'); a missing Z with known X/Y takes the finest of
            them; without X and Y sizes the scale bar is labelled in pixels.
        colormap: colour or colormap name (or registered Colormap) per channel;
            None gives grey for a single channel and the package palette
            ('white', 'lime', 'magenta', ...) for a list.
        vmin, vmax (float, optional): display range. A bound left as None is
            the default display range (`_conventions.default_display_range`:
            0, or the data minimum if negative, and the robust 99.99th
            percentile) computed once over all three panels, so the panels
            share one intensity scale. The volume viewers (`show_zyx_max`,
            the widgets) pass the range of the source volume instead.
        rotate_view (float or (Z, Y, X) tuple, optional): rotates the XY, XZ and
            ZY panels counter-clockwise (degrees) in physical space; rotated
            panels are resampled onto isotropic pixels.
        channel_label_colors (list, optional): label colours, overriding the
            ones derived from ``colormap`` (for pre-composited RGB input).
        pixel_sizes_given (bool, optional): whether the scale bar shows physical
            units; defaults to whether ``pixel_sizes`` gives the X and Y sizes
            (not None).
        blending (str, optional): how channels are composited (napari's names):
            'additive', 'maximum', 'screen' or 'minimum'. None (default) keeps
            the historical behaviour: additive for colour names, maximum when a
            channel is a colormap name; but 'minimum' when every colormap
            channel starts at white (the I-series, 'Blues', ...), whose maximum
            would be white everywhere (a mix of white- and dark-start
            colormaps warns). 'minimum' keeps the per-RGB minimum of
            the channels, each drawn from white (low) to its colour (high), or
            with an inverted colormap such as 'Blues' or 'Greys': it gives a
            multichannel image on a white background, and the panels, rotation
            padding and scale bar switch to a white theme. For a single channel
            or pre-composited RGB input only the white theme (and white->colour
            colormap) applies.
        scale_bar_show (bool, optional): draw the scale bar on the XY panel
            (default True). Its label sits centred right above the bar.
        scale_bar_fontsize_pt (float, optional): label font size in points; the
            bar's line width follows it. None sizes it from the figure height;
            0 hides the label but keeps the bar (at the auto width).
            ``scale_bar_show`` toggles the bar.
        subplot_bg (colour, optional): panel background. None (default) is
            the theme background: black, or white for ``blending='minimum'``.
            NaN pixels, rotation corners and faded (opacity < 1) channels
            show it.
        use_plt (bool, optional): create the figure with ``pyplot.figure``
            (registered with pyplot, shown by ``plt.show()``). Default False:
            a plain ``matplotlib.figure.Figure``, which Jupyter and marimo
            display when it is returned.
    Returns:
        matplotlib Figure with named axes ``axXY``, ``axZY``, ``axXZ``, ``axBar``.

    Changed (D05, 2026-11): ``use_plt`` defaults to False (the object-oriented
    interface; figures no longer pile up in pyplot), and the figure is no
    longer drawn before it is returned. Changed (WP2, 2026-11): the panels
    default to the theme background (they kept rcParams' white axes colour,
    so single-channel NaN pixels and opacity looked like saturated signal);
    NaN pixels take that colour, and the corners added by ``rotate_view`` to a
    single-channel panel are background, not the colour of value 0.

    Changed (D01-D04, 2026-11): a single channel defaults to grey, not
    viridis; the default vmin/vmax is the robust default display range, not
    the panels' min/max; pixel sizes are converted to µm and the scale bar
    reads 'N px' without X/Y sizes and picks nm / µm / mm by length;
    registered Colormap instances are accepted; all-white-start colormaps
    blend with 'minimum' by default. Missing channel colours skip palette
    colours already given in any spelling ('#00ff00' uses up 'lime', 'gray'
    uses up 'white'), and None entries in a colormap list take a palette
    colour, as in LabelAnnotator.
    """
    if colors is not None:
        warn_deprecated("The 'colors' parameter", "'colormap'")
        if colormap is None:
            colormap = colors

    blending = _check_blending(blending)
    if isinstance(xy, list):
        # One colour per channel, then the blending=None rule (D04)
        colormap, blending = _resolve_channels(colormap, len(xy), blending)
    white_theme = blending == 'minimum'
    if subplot_bg is None:
        # The theme background, never rcParams' (white) axes colour: NaN
        # pixels and opacity show it, so it must not look like signal (R1-03).
        subplot_bg = 'white' if white_theme else 'black'
    # Value used for the corners that rotation adds to a panel: the panel
    # background of a composite, missing (NaN, drawn as the background) for a
    # single channel (R1-10).
    pad_value = 0.0

    (pz, py, px), xy_given, length_unit = _viewer_pixel_sizes(pixel_sizes)
    both_given = xy_given if pixel_sizes_given is None else bool(pixel_sizes_given)

    # Preserve original colormap reference before resetting colormap=None for RGB arrays
    orig_colormap = colormap

    if isinstance(xy, list):
        xy, xz, zy = _composite_planes(
            xy, xz, zy, vmin=vmin, vmax=vmax, gamma=gamma, colormap=colormap, opacity=opacity,
            blending=blending,
        )

        colormap = None
        vmin, vmax, gamma = None, None, 1
        opacity = None
        if white_theme:
            pad_value = 1.0
    else:
        if isinstance(opacity, list):
            # Opacity of a pre-composited RGB(A) image is already baked in;
            # applying channel 0's value would fade every channel.
            opacity = None if np.ndim(xy) == 3 else opacity[0]
        if isinstance(vmin, list):
            vmin = vmin[0]
        if isinstance(vmax, list):
            vmax = vmax[0]
        if isinstance(gamma, list):
            gamma = gamma[0]

        if np.ndim(xy) == 2 and (vmin is None or vmax is None):
            # One intensity scale for all three panels (computed before
            # rotation, which pads the canvas with zeros). The volume viewers
            # pass the range of the source volume; here only the panels exist.
            vmin, vmax = _volume_display_range(_panel_values(xy, xz, zy), vmin, vmax)

        if isinstance(colormap, list) and len(colormap) == 1:
            colormap = colormap[0]

        # A plain colour is drawn black -> colour (white -> colour for minimum blending)
        to_cmap = white_to if white_theme else black_to
        if colormap is None and np.ndim(xy) == 2:
            # Grey ('white' ramp) like a one-channel list and the widgets (D03)
            colormap = PALETTE_MINIMUM[0] if white_theme else PALETTE[0]
            orig_colormap = colormap
        if colormap is not None:
            c = channel_spec(colormap)  # a registered Colormap instance -> its name
            orig_colormap = c
            colormap = matplotlib.colormaps[c] if is_colormap(c) else to_cmap(resolve_color(c))
            # NaN pixels take the panel background (matplotlib's 'bad' colour
            # is transparent, which showed whatever lay behind the panel)
            colormap = colormap.with_extremes(bad=subplot_bg)

        if np.ndim(xy) == 3:
            pad_value = 1.0 if white_theme else 0.0
        else:
            pad_value = np.nan

    rot_z, rot_y, rot_x = _parse_rotate_view(rotate_view)

    # Physical extent of each panel: (row pixel size, column pixel size).
    # Rotation happens in physical space, so a rotated panel has isotropic pixels.
    g_xy = _PanelRotation(np.shape(xy)[:2], rot_z, py, px)
    g_xz = _PanelRotation(np.shape(xz)[:2], rot_y, pz, px)
    g_zy = _PanelRotation(np.shape(zy)[:2], rot_x, py, pz)

    def _warp(g, panel):
        panel = np.asarray(panel)
        if g.angle != 0.0 and panel.ndim == 3 and np.issubdtype(panel.dtype, np.integer):
            # imshow reads integer RGB(A) on a 0..255 scale but the resampled
            # (float) panel on a 0..1 scale
            panel = panel.astype(np.float32) / 255.0
        return g.warp(panel, cval=pad_value)

    xy, xz, zy = _warp(g_xy, xy), _warp(g_xz, xz), _warp(g_zy, zy)
    (sxy_h, sxy_w), (sxz_h, sxz_w), (szy_h, szy_w) = g_xy.p_out, g_xz.p_out, g_zy.p_out
    if figsize is None:
        # From the physical extents of the (rotated) panels (D12)
        figsize = _auto_figsize(
            (max(xy.shape[1] * sxy_w, xz.shape[1] * sxz_w), zy.shape[1] * szy_w),
            (max(xy.shape[0] * sxy_h, zy.shape[0] * szy_h), xz.shape[0] * sxz_h),
            channel_labels is not None, gap_in=gap_in, label_fontsize_pt=channel_label_fontsize_pt)

    if use_plt:
        import matplotlib.pyplot as plt  # only on request: library figures stay out of pyplot (D05)
        fig = plt.figure(figsize=figsize, constrained_layout=False)
    else:
        fig = _new_figure(figsize=figsize, constrained_layout=False)

    figW, figH = figsize

    xy_w, xy_h = xy.shape[1] * sxy_w, xy.shape[0] * sxy_h
    xz_w, xz_h = xz.shape[1] * sxz_w, xz.shape[0] * sxz_h
    zy_w, zy_h = zy.shape[1] * szy_w, zy.shape[0] * szy_h

    col1_w = max(xy_w, xz_w)
    col2_w = zy_w
    row1_h = max(xy_h, zy_h)
    row2_h = xz_h
    if channel_label_fontsize_pt is None:
        channel_label_fontsize_pt = _auto_fontsize_pt(figW)

    hl_in = _label_row_in(channel_label_fontsize_pt) if channel_labels is not None else 0.0

    margin_in = _MARGIN_IN
    avail_w_in = figW - gap_in - 2.0 * margin_in
    num_v_gaps = 2 if channel_labels is not None else 1
    avail_h_in = figH - num_v_gaps * gap_in - 2.0 * margin_in - hl_in
    total_unscaled_h = row1_h + row2_h

    scale = min(avail_w_in / max(col1_w + col2_w, 1e-6), avail_h_in / max(total_unscaled_h, 1e-6))

    w1_in = col1_w * scale
    w2_in = col2_w * scale
    h1_in = row1_h * scale
    h2_in = row2_h * scale

    block_w_in = w1_in + gap_in + w2_in
    block_h_in = h1_in + gap_in + h2_in + (hl_in + gap_in if channel_labels is not None else 0.0)

    left_margin_in = (figW - block_w_in) / 2.0
    bottom_margin_in = (figH - block_h_in) / 2.0

    x0_frac = left_margin_in / figW
    w1_frac = w1_in / figW
    gap_w_frac = gap_in / figW
    w2_frac = w2_in / figW

    x_col1 = x0_frac
    x_col2 = x0_frac + w1_frac + gap_w_frac

    y0_frac = bottom_margin_in / figH
    h2_frac = h2_in / figH
    gap_h_frac = gap_in / figH
    h1_frac = h1_in / figH
    hl_frac = hl_in / figH

    y_row2 = y0_frac
    y_row1 = y0_frac + h2_frac + gap_h_frac

    axLabels = None
    if channel_labels is not None:
        y_labels = y_row1 + h1_frac + gap_h_frac
        axLabels = fig.add_axes([x_col1, y_labels, w1_frac, hl_frac])
        axXY = fig.add_axes([x_col1, y_row1, w1_frac, h1_frac])
        axZY = fig.add_axes([x_col2, y_row1, w2_frac, h1_frac])
        axXZ = fig.add_axes([x_col1, y_row2, w1_frac, h2_frac])
        axBar = fig.add_axes([x_col2, y_row2, w2_frac, h2_frac])

        axLabels.patch.set_visible(False)
        axLabels.set_xticks([])
        axLabels.set_yticks([])
        for spine in axLabels.spines.values():
            spine.set_visible(False)

        _draw_channel_label_bar(axLabels, channel_label_fontsize_pt, as_image=True, x_ref=axXY)

        n_labels = len(channel_labels)
        if n_labels > 0:
            def _get_color_str(c):
                if isinstance(c, mcolors.Colormap):
                    return mcolors.to_hex(c(1.0)[:3])
                return resolve_color(c)

            if channel_label_colors is not None:
                color_list = [_get_color_str(c) for c in channel_label_colors]
            elif orig_colormap is None:
                color_list = ['black']
            elif isinstance(orig_colormap, (str, mcolors.Colormap)) or (isinstance(orig_colormap, tuple) and mcolors.is_color_like(orig_colormap)):
                color_list = [_get_color_str(orig_colormap)]
            else:
                color_list = [_get_color_str(c) for c in orig_colormap]
            color_list = color_list or ['black']
            if len(color_list) < n_labels:
                color_list = color_list * (n_labels // len(color_list) + 1)

            from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox

            text_areas = []
            for i, label in enumerate(channel_labels):
                GUI_CHANNEL_COLOR = get_gui_channel_color(color_list[i])
                ta = TextArea(label, textprops=dict(color=GUI_CHANNEL_COLOR, fontsize=channel_label_fontsize_pt, fontweight='bold'))
                text_areas.append(ta)

            packer = HPacker(children=text_areas, align="center", pad=0, sep=10)
            anchored_box = AnchoredOffsetbox(
                loc='lower center', child=packer, pad=0.0, frameon=False, borderpad=0.0,
                bbox_to_anchor=(0.5, 0.2), bbox_transform=axLabels.transAxes
            )
            axLabels.add_artist(anchored_box)
    else:
        axXY = fig.add_axes([x_col1, y_row1, w1_frac, h1_frac])
        axZY = fig.add_axes([x_col2, y_row1, w2_frac, h1_frac])
        axXZ = fig.add_axes([x_col1, y_row2, w1_frac, h2_frac])
        axBar = fig.add_axes([x_col2, y_row2, w2_frac, h2_frac])

    if gamma == 1:
        axXY.imshow(xy, cmap=colormap, vmin=vmin, vmax=vmax, extent=[0, xy_w, xy_h, 0], interpolation='nearest', alpha=opacity)
        axZY.imshow(zy, cmap=colormap, vmin=vmin, vmax=vmax, extent=[0, zy_w, zy_h, 0], interpolation='nearest', alpha=opacity)
        axXZ.imshow(xz, cmap=colormap, vmin=vmin, vmax=vmax, extent=[0, xz_w, xz_h, 0], interpolation='nearest', alpha=opacity)
    else:
        norm = PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
        axXY.imshow(xy, cmap=colormap, norm=norm, extent=[0, xy_w, xy_h, 0], interpolation='nearest', alpha=opacity)
        axZY.imshow(zy, cmap=colormap, norm=norm, extent=[0, zy_w, zy_h, 0], interpolation='nearest', alpha=opacity)
        axXZ.imshow(xz, cmap=colormap, norm=norm, extent=[0, xz_w, xz_h, 0], interpolation='nearest', alpha=opacity)

    axXY.set_xlim([0, xy_w]); axXY.set_ylim([xy_h, 0])
    axZY.set_xlim([0, zy_w]); axZY.set_ylim([zy_h, 0])
    axXZ.set_xlim([0, xz_w]); axXZ.set_ylim([xz_h, 0])

    for i, ax in enumerate([axXY, axZY, axXZ, axBar]):
        if i < 3:
            ax.set_facecolor(subplot_bg)
        elif ax == axBar:
            ax.patch.set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if white_theme:
        fig.patch.set_facecolor('white')  # the gaps between panels belong to the white canvas
    else:
        fig.patch.set_alpha(0.0)  # set transparent bgnd

    if scale_bar_show:
        _add_scale_bar(axXY, xy_w, both_given, figsize, unit=length_unit,
                       color='black' if white_theme else 'white', fontsize_pt=scale_bar_fontsize_pt)

    fig.axXY = axXY
    fig.axZY = axZY
    fig.axXZ = axXZ
    fig.axBar = axBar
    fig.axLabels = axLabels

    # imshow's equal aspect may shrink a panel box; apply it now so transData
    # (used for the widgets' axis_bounds) is final without a full draw.
    for ax in (axXY, axZY, axXZ):
        ax.apply_aspect()
    return fig

def _scale_bar_fontsize(v):
    """Scale-bar font size from a kwarg or the GUI box: None/'' -> auto, else a float >= 0."""
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return None
    v = float(v)
    if not np.isfinite(v) or v < 0:
        raise ValueError(f"scale_bar_fontsize_pt must be >= 0 (0 hides the label), got {v}")
    return v


def _format_bar_value(v):
    """Scale-bar number: an integer when whole (20, 500), else 3 significant digits."""
    r = round(v)
    if r != 0 and abs(v - r) <= 1e-6 * abs(v):
        return str(int(r))
    return f"{v:.3g}"


def _add_scale_bar(ax_line, ax_physical_width_um, pixel_sizes_given, figsize, unit=None, color='white', fontsize_pt=None):
    """
    Draw the scale bar and its label on ``ax_line`` (the XY panel).

    Changed (D01, 2026-11): the label picks nm / µm / mm by the bar's length
    ('500 nm', '20 µm', '2 mm'; it used to be µm only), and without X/Y
    pixel sizes it reads 'N px' (it used to read '`pixel_sizes`'). Sizes in
    a pixel unit ('pixel', 'px') count as uncalibrated and also give 'N px'
    (they used to be shown as an unknown unit, '50 pixel').
    """
    # ``fontsize_pt`` None picks a size from the figure height; 0 hides the
    # label only (the bar keeps its auto-size width; ``scale_bar_show``
    # toggles the bar). The bar's line width follows the font size.
    fontsize_pt = _scale_bar_fontsize(fontsize_pt)
    show_label = fontsize_pt != 0
    if not show_label:
        fontsize_pt = None
    # ``ax_physical_width_um`` is in the units of the pixel sizes: ``unit``
    # (µm when None; the viewers convert known units to µm when parsing, a
    # known unit passed here is converted too). The label picks nm / µm / mm
    # by the bar's length (_conventions.format_length); an unknown unit is
    # shown as given. Without X/Y pixel sizes the width is in pixels ('px').
    unit_label = None
    if not pixel_sizes_given:
        unit_label = 'px'
    elif unit is not None:
        if is_length_unit(unit):
            ax_physical_width_um = to_um(ax_physical_width_um, unit)
        else:
            unit_label = unit

    # a small utility to pick the largest "nice" number ≤ target
    target = ax_physical_width_um * 0.2

    def nice_length(x):
        # get exponent
        if x <= 0: return 1
        exp = np.floor(np.log10(x))
        # candidates: 1, 2, 5 times 10^exp
        for m in [5, 2, 1]:
            val = m * 10**exp
            if val <= x:
                return val
        # if nothing smaller (very small x), just return x itself
        return x

    bar_um = nice_length(target)
    # Convert back to fraction of the full width of the current axes
    bar_frac = bar_um / ax_physical_width_um if ax_physical_width_um > 0 else 0

    # pick fontsize and line width to scale with figure size
    if fontsize_pt is None:
        fontsize_pt = _auto_fontsize_pt(figsize[0] if figsize is not None else 10)
    linewidth = max(1, fontsize_pt * 0.2 / (72 * 0.03))  # = 0.2 pt per inch of figure width at the auto size

    ### Draw
    # Put the line at the bottom right of ax_line
    x1 = 0.95
    x0 = x1 - bar_frac
    y_line = 0.05

    # A thin 30%-opacity black outline keeps bar and label visible over
    # pixels of their own colour (e.g. a white bar over saturated signal).
    from matplotlib import patheffects
    outline_pt = max(0.75, fontsize_pt * 0.06)

    def _outline(width):
        return [patheffects.withStroke(linewidth=width + 2 * outline_pt, foreground='black', alpha=0.3)]

    ax_line.hlines(y_line, x0, x1, transform=ax_line.transAxes, linewidth=linewidth, color=color, clip_on=False,
                   path_effects=_outline(linewidth))

    if not show_label:
        return

    if unit_label is None:
        value, unit_label = format_length(bar_um)
    else:
        value = bar_um
    text_label = f"{_format_bar_value(value)} {unit_label}"

    # Label centred right above the bar, in the bar's colour. The gap is in
    # points (half the line width plus a fraction of the font), so it does not
    # depend on the panel size.
    ax_line.annotate(
        text_label, xy=((x0 + x1) / 2, y_line), xycoords='axes fraction',
        xytext=(0, linewidth / 2 + 0.2 * fontsize_pt), textcoords='offset points',
        ha='center', va='bottom', color=color, fontsize=fontsize_pt, annotation_clip=False,
        path_effects=_outline(0),
    )

### New function
def _slab_slice(rng, n, name):
    """
    ``[start, stop)`` index range (rounded; a range of zero width widens to
    one plane) -> slice clamped to ``[0, n)``. None is the whole axis.
    Out-of-range bounds are clipped with a warning (negative values count
    from 0, not from the end); a range with nothing left raises ValueError.
    """
    if rng is None:
        return slice(None)
    lo, hi = (int(np.round(float(v))) for v in rng)
    if hi <= lo:
        hi = lo + 1
    clo, chi = max(0, lo), min(n, hi)
    if chi <= clo:
        raise ValueError(f"{name} range [{lo}, {hi}) lies outside the image (size {n} along {name.upper()}).")
    if (clo, chi) != (lo, hi):
        warnings.warn(f"{name} range [{lo}, {hi}) clipped to the image: [{clo}, {chi}).", UserWarning, stacklevel=3)
    return slice(clo, chi)


@_static_doc
def show_zyx_max_slabs(image_to_show, x=None, y=None, z=None, pixel_sizes=None, figsize=None, colormap=None, vmin=None, vmax=None, gamma=1, colors=None, opacity=None, rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None, blending=None, projection=None, attenuation=0.05, scale_bar_show=True, scale_bar_fontsize_pt=None,
                       *, subplot_bg=None, use_plt=False, channel_axis=None):
    """ plots max xy, xz, and zy projections of a 3D image SLABS (slice intervals)

    Author: PanosOik https://github.com/PanosOik

    Args:
        image_to_show: image to plot (see the input forms below)
        x, y, z: half-open index ranges ``[start, stop)`` of the slabs
            (voxel indices, rounded; a zero-width range widens to one
            plane). None (default) is the whole axis, i.e. the full
            projection. Bounds outside the image are clipped with a
            warning; a range entirely outside raises ValueError.
        projection, attenuation: see `show_zyx_projection_slabs`.

    Changed (D12, 2026-11): x, y and z default to None, the whole axis (the
    defaults were [0, 1], the first plane of each axis), and out-of-range
    bounds are clipped with a warning (they raised an opaque 'zero-size
    array' error).
    """
    colormap = _legacy_colors(colors, colormap)
    im, colormap, channel_labels, pixel_sizes = _ortho_input(
        image_to_show, colormap, channel_labels, pixel_sizes, channel_axis)
    Zn, Yn, Xn = np.shape(im[0] if isinstance(im, list) else im)
    return show_zyx_projection_slabs(
        im, _slab_slice(x, Xn, 'x'), _slab_slice(y, Yn, 'y'), _slab_slice(z, Zn, 'z'),
        pixel_sizes=pixel_sizes, figsize=figsize, projector=None, colormap=colormap, vmax=vmax, vmin=vmin,
        gamma=gamma, opacity=opacity, rotate_view=rotate_view, channel_labels=channel_labels,
        channel_label_fontsize_pt=channel_label_fontsize_pt, blending=blending, projection=projection,
        attenuation=attenuation, scale_bar_show=scale_bar_show, scale_bar_fontsize_pt=scale_bar_fontsize_pt,
        subplot_bg=subplot_bg, use_plt=use_plt)


@_static_doc
def show_zyx_projection_slabs(image_to_show, x_slices, y_slices, z_slices, pixel_sizes=None, figsize=None, projector=None, colormap=None, vmin = None, vmax=None, gamma = 1, colors = None, opacity = None, rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None, blending=None, projection=None, attenuation=0.05, scale_bar_show=True, scale_bar_fontsize_pt=None,
                              *, subplot_bg=None, use_plt=False, channel_axis=None):
    """ generates xy, xz, and zy projections of slabs of a 3D image and plots them

    Author: PanosOik https://github.com/PanosOik

    Args:
        image_to_show: image to plot (see the input forms below)
        x_slices, y_slices, z_slices: index slices (or None for the whole
            axis) of the slabs projected onto the ZY, XZ and XY panels.
        projector: function ``f(volume, axis)`` to project with (used when
            ``projection`` is None); None (default) is the maximum ignoring NaN
            voxels, like ``projection='mip'``.
        projection (str, optional): how each slab is projected along the view
            axis (napari's rendering names): 'mip' (maximum, the default),
            'minip' (minimum), 'average', or 'attenuated_mip'. None uses
            ``projector``. NaN voxels are ignored (a ray of only NaN stays NaN).
        attenuation (float, optional): for 'attenuated_mip', the attenuation per
            voxel along the ray (napari's default 0.05). Rays enter each slab at
            its lowest index (z = start for XY, y = start for XZ, x = start for
            ZY); samples are normalised with the display range (vmin/vmax, or
            the channel's default range), so bright structures hide dimmer ones
            behind them.

    A vmin/vmax left as None (per channel) is the default display range of
    the whole source volume (`_conventions.default_display_range`), the same
    range the widgets use; 'attenuated_mip' is normalised with it too.

    Changed (D02, 2026-11): the default range used to be the min/max of the
    projected panels (single channel) or 0 / the panels' max (channel list),
    so a single hot pixel darkened the whole figure.
    """
    colormap = _legacy_colors(colors, colormap)
    image_to_show, colormap, channel_labels, pixel_sizes = _ortho_input(
        image_to_show, colormap, channel_labels, pixel_sizes, channel_axis)
    x_slices, y_slices, z_slices = (slice(None) if v is None else v for v in (x_slices, y_slices, z_slices))
    if projection is not None:
        projection = _check_projection(projection, attenuation)

    # The default display range comes from the source volume (D02), not from
    # the projected panels, on which a single hot pixel covers far more than
    # 0.01% of the pixels. The same range normalises 'attenuated_mip'.
    if isinstance(image_to_show, list):
        vmin, vmax = _channels_display_range(image_to_show, vmin, vmax)
    else:
        vmin, vmax = _volume_display_range(image_to_show, vmin, vmax)

    def _projector(img, i):
        if projection is None:
            return _nan_max_projection if projector is None else projector
        lo = hi = None
        if projection == 'attenuated_mip':
            # Samples are normalised with the channel's display range
            lo = float(vmin[i] if isinstance(vmin, list) else vmin)
            hi = float(vmax[i] if isinstance(vmax, list) else vmax)
            if not hi > lo:
                hi = lo + 1.0
        return _projector_for(projection, attenuation, lo, hi)

    def _project(img, i):
        # Slicing creates a view. Some projectors might make copies. np.max over an axis
        # might allocate a temporary buffer if the memory is not contiguous in that axis.
        proj = _projector(img, i)
        return (proj(img[:, y_slices, :], axis=1),
                np.flip(np.rot90(proj(img[:, :, x_slices], axis=2), 1), 0),
                proj(img[z_slices, :, :], axis=0))

    if isinstance(image_to_show, list):
        projections = [_project(img, i) for i, img in enumerate(image_to_show)]
        projection_y = [p[0] for p in projections]
        projection_x = [p[1] for p in projections]
        projection_z = [p[2] for p in projections]
    else:
        projection_y, projection_x, projection_z = _project(image_to_show, 0)

    return show_zyx(projection_z, projection_y, projection_x, pixel_sizes=pixel_sizes, figsize=figsize,
                    colormap=colormap, vmax=vmax, vmin=vmin, gamma=gamma, opacity=opacity,
                    rotate_view=rotate_view, channel_labels=channel_labels,
                    channel_label_fontsize_pt=channel_label_fontsize_pt, blending=blending,
                    scale_bar_show=scale_bar_show, scale_bar_fontsize_pt=scale_bar_fontsize_pt,
                    subplot_bg=subplot_bg, use_plt=use_plt)


def _finite_min(*arrays):
    """Smallest finite value over several arrays (None if there is none)."""
    best = None
    for a in arrays:
        a = np.asarray(a)
        if a.size == 0:
            continue
        if np.issubdtype(a.dtype, np.floating):
            a = a[np.isfinite(a)]
            if a.size == 0:
                continue
        m = float(a.min())
        best = m if best is None else min(best, m)
    return best


def _check_gammas(gammas):
    for g in gammas:
        if g is not None and not float(g) > 0:
            raise ValueError(f"gamma must be > 0, got {g}")


def _per_channel(val, n, default, name):
    """``_conventions.per_channel`` with this module's argument order."""
    return per_channel(val, n, name, default)


def _panel_values(*arrays):
    """The values of several panels as one flat array (for a range over all of them)."""
    return np.concatenate([np.ravel(np.asarray(a)) for a in arrays])


def _volume_display_range(volume, vmin, vmax):
    """``(vmin, vmax)`` of one channel, bounds left as None taken from `default_display_range(volume)`.

    A list (one entry per channel, as the widgets pass) contributes its first
    entry. When only vmin is given and it is not below the default vmax, vmax
    becomes vmin + 1 (matplotlib's norms reject vmin > vmax).
    """
    if isinstance(vmin, (list, tuple)):
        vmin = vmin[0] if len(vmin) else None
    if isinstance(vmax, (list, tuple)):
        vmax = vmax[0] if len(vmax) else None
    if vmin is None or vmax is None:
        lo, hi = default_display_range(volume)
        if vmin is None:
            vmin = lo
        if vmax is None:
            vmax = hi if hi > float(vmin) else float(vmin) + 1.0
    return vmin, vmax


def _channels_display_range(channels, vmin, vmax):
    """Per-channel ``(vmin list, vmax list)``, None entries from each channel's whole volume."""
    n = len(channels)
    vmins = per_channel(vmin, n, 'vmin')
    vmaxs = per_channel(vmax, n, 'vmax')
    for i, ch in enumerate(channels):
        vmins[i], vmaxs[i] = _volume_display_range(ch, vmins[i], vmaxs[i])
    return vmins, vmaxs


def _resolve_channels(colormap, n, blending=None):
    """
    Channel colours (canonical strings) and the blending mode for ``n`` composited channels.

    Missing colours take the unused palette colours; ``blending=None`` follows
    `_conventions.resolve_blending` (additive for colours, maximum with a
    colormap name, minimum when every colormap starts at white). When the
    rule picks 'minimum', missing colours come from the minimum palette.
    """
    explicit = _check_blending(blending)
    specs = _channel_specs(colormap, n, explicit)
    mode = resolve_blending(specs, explicit, stacklevel=3)
    if mode == 'minimum' and explicit is None:
        specs = _channel_specs(colormap, n, mode)
    return specs, mode


def _finite_max(*arrays):
    """Largest finite value over several arrays (None if there is none)."""
    best = None
    for a in arrays:
        a = np.asarray(a)
        if a.size == 0:
            continue
        if np.issubdtype(a.dtype, np.floating):
            a = a[np.isfinite(a)]
            if a.size == 0:
                continue
        m = float(a.max())
        best = m if best is None else max(best, m)
    return best


def _norm_window(a, lo, hi, g, eps=1e-12):
    """Linear normalisation of ``a`` to [0, 1] over ``[lo, hi]`` (NaN/inf -> 0), then gamma."""
    out = (np.asarray(a).astype(np.float32, copy=False) - lo) / max(hi - lo, eps)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    out = np.clip(out, 0.0, 1.0, out=out)
    if g != 1:
        out = np.power(out, g, out=out)
    return out


def _composite(views, colormap, vmin=None, vmax=None, gamma=1, opacity=None, blending=None,
               soft_clip=True, eps=1e-12, default_range=None):
    """
    The package's one channel compositor (D04): per-channel windows, colours and blending.

    Parameters
    ----------
    views : list of lists of arrays
        ``views[k][i]`` is channel ``i`` in view ``k`` (e.g. the XY, XZ and ZY
        panels, or the (N,) values of a point cloud). Any shapes.
    colormap : per-channel colours or colormap names (padded with the palette).
    vmin, vmax, gamma, opacity : scalar or per-channel list (short lists are
        padded with the default; None entries take it).
    blending : 'additive', 'maximum', 'screen', 'minimum' (aliases accepted)
        or None for the package rule (`_conventions.resolve_blending`).
    default_range : callable ``i -> (vmin, vmax)`` for channel ``i``; default:
        `_conventions.default_display_range` over the channel's values in all
        views.

    Each channel is normalised to [0, 1] in its window (NaN and inf count as
    no signal), gamma-corrected, then coloured: a colormap name through its
    8-bit LUT (``bytes=True``), a plain colour as the exact ramp black ->
    colour (white -> colour for 'minimum'). Opacity scales a channel (fades it
    towards white for 'minimum'). 'additive' compresses highlights above 1
    when ``soft_clip``.

    Returns a list with one float32 RGB array (``shape + (3,)``) per view.
    """
    n = len(views[0])
    if any(len(v) != n for v in views):
        raise ValueError("every view must have the same number of channels")
    colors, blend = _resolve_channels(colormap, n, blending)

    gammas = _per_channel(gamma, n, 1, 'gamma')
    _check_gammas(gammas)
    opacities = _per_channel(opacity, n, 1.0, 'opacity')
    vmins = _per_channel(vmin, n, None, 'vmin')
    vmaxs = _per_channel(vmax, n, None, 'vmax')

    for i in range(n):
        if vmins[i] is None or vmaxs[i] is None:
            lo, hi = (default_range(i) if default_range is not None
                      else default_display_range(_panel_values(*(v[i] for v in views))))
            if vmins[i] is None:
                vmins[i] = lo
            if vmaxs[i] is None:
                vmaxs[i] = hi
        vmins[i], vmaxs[i] = float(vmins[i]), float(vmaxs[i])
        # Sanitize: ensure vmax > vmin
        if not np.isfinite(vmins[i]):
            vmins[i] = 0.0
        if not np.isfinite(vmaxs[i]) or vmaxs[i] <= vmins[i] + eps:
            vmaxs[i] = vmins[i] + 1.0  # avoid zero range

    shapes = [np.shape(v[0]) for v in views]
    start = 1.0 if blend in ('screen', 'minimum') else 0.0
    acc = [np.full(s + (3,), start, dtype=np.float32) for s in shapes]

    for i in range(n):
        c = colors[i]
        g, o, lo, hi = gammas[i], opacities[i], vmins[i], vmaxs[i]
        cmap = matplotlib.colormaps[c] if is_colormap(c) else None
        col = None if cmap is not None else np.asarray(to_rgb(resolve_color(c)), dtype=np.float32)
        for k, view in enumerate(views):
            norm = _norm_window(view[i], lo, hi, g, eps)
            if cmap is not None:
                rgb = cmap(norm, bytes=True)[..., :3].astype(np.float32) / 255.0
                if blend == 'minimum':
                    # opacity fades the channel towards white (the minimum's neutral value)
                    acc[k] = np.minimum(acc[k], 1.0 - o * (1.0 - rgb))
                    continue
                rgb *= o
            elif blend == 'minimum':
                # white -> colour, faded towards white by the opacity
                c_inv = (o * (1.0 - col)).astype(np.float32)
                acc[k] = np.minimum(acc[k], 1.0 - norm[..., None] * c_inv)
                continue
            else:
                rgb = norm[..., None] * (col * o).astype(np.float32)
            if blend == 'screen':
                acc[k] *= (1.0 - rgb)
            elif blend == 'maximum':
                acc[k] = np.maximum(acc[k], rgb)
            else:  # additive
                acc[k] += rgb

    out = []
    for a in acc:
        if blend == 'screen':
            a = 1.0 - a
        elif blend == 'additive':
            if soft_clip:
                # compress highlights smoothly instead of hard clipping
                a /= np.maximum(1.0, a.max(axis=-1, keepdims=True))
            a = np.clip(a, 0.0, 1.0)
        out.append(a)
    return out


def _composite_planes(xy_list, xz_list, zy_list, vmin=None, vmax=None, gamma=1, colormap=None,
                      opacity=None, blending=None, soft_clip=True, eps=1e-12):
    """`_composite` of the XY, XZ and ZY panels -> ``(xy_rgb, xz_rgb, zy_rgb)``."""
    assert isinstance(xy_list, list) and isinstance(xz_list, list) and isinstance(zy_list, list)
    n = len(xy_list)
    assert len(xz_list) == n and len(zy_list) == n, "xy/xz/zy must have same number of channels"
    return tuple(_composite([xy_list, xz_list, zy_list], colormap, vmin, vmax, gamma, opacity,
                            blending, soft_clip, eps))


def create_multichannel_rgb(
    xy_list, xz_list, zy_list,
    vmin=None, vmax=None, gamma=1, colormap=None, colors=None, opacity=None,
    blend='add',        # 'additive' | 'maximum' | 'screen' | 'minimum'
    soft_clip=True,     # only used for additive blending
    eps=1e-12,
    blending=None,
):
    """
    Compose multi-channel XY/XZ/ZY into RGB with per-channel normalization.

    Parameters
    ----------
    xy_list, xz_list, zy_list : list[np.ndarray]
        One 2D array per channel for each orientation.
    vmin, vmax : float | list[float] | None
        Per-channel input range. If None, the channel's default display range
        over its XY/XZ/ZY panels (`_conventions.default_display_range`): vmin
        is 0, or the data minimum when the channel has negative values; vmax
        is the robust 99.99th percentile (the maximum for sparse signal).
        Scalars are broadcast to all channels. (For vmin, vmax, gamma and
        opacity, a list shorter than the number of channels is padded with
        the default; a longer one raises ValueError.)
    gamma : float | list[float]
        Power gamma applied AFTER linear [0,1] normalization (1 = linear).
    colormap : str | tuple | Colormap | list of them
        One colour or colormap name (or registered Colormap) per channel
        (``colors`` is the deprecated name). Missing entries take the unused
        default colours 'white', 'lime', 'magenta', 'yellow', 'cyan', 'red',
        'blue' ('black', 'magenta', 'lime', 'blue', 'red', 'cyan', 'yellow'
        with ``blend='minimum'``).
    opacity : float | list[float] | None
        Opacity multiplier per channel (1 = full opacity).
    blend, blending : str
        How channels combine: 'additive' (default; alias 'add'), 'maximum'
        (alias 'max'), 'screen', or 'minimum' (alias 'min'). With 'minimum'
        each channel is drawn from white (low) to its colour (high) and the
        per-RGB minimum is kept, giving a multichannel image on a white
        background; opacity fades a channel towards white. ``blending`` (the
        viewers' name) wins over ``blend``; ``blend=None`` applies the
        viewers' rule (see `show_zyx`).
    soft_clip : bool
        For additive blending, compress values >1 instead of hard clipping.

    Plain colours are exact float ramps and colormap names use their 8-bit
    LUT; this is the same compositor as `create_multichannel_rgb_cmap` and
    `blend_colors`.

    Changed (D02/D04, 2026-11): the default vmax is the robust 99.99th
    percentile (it was the finite maximum). Colormap names now go through
    their full LUT (they used to be drawn as a black->end-colour ramp, e.g.
    'viridis' as black->yellow), so their output changes visibly; registered
    Colormap instances are accepted.
    """
    if colors is not None:
        warn_deprecated("The 'colors' parameter", "'colormap'")
        if colormap is None:
            colormap = colors
    return _composite_planes(xy_list, xz_list, zy_list, vmin=vmin, vmax=vmax, gamma=gamma,
                             colormap=colormap, opacity=opacity,
                             blending=blending if blending is not None else blend,
                             soft_clip=soft_clip, eps=eps)


def create_multichannel_rgb_cmap(xy_list, xz_list, zy_list, vmin=None, vmax=None, gamma=1, colormap=None, colors=None, opacity=None, blend='max', soft_clip=True, eps=1e-12, blending=None):
    """
    Like `create_multichannel_rgb`, but ``blend`` defaults to 'maximum'.

    Colormap names are applied as given; plain colours become black->colour
    ramps (white->colour for ``blend='minimum'``, so that inverted colormaps
    such as 'Blues' or 'Greys' and plain colours compose on a white background).
    Both functions are thin wrappers of the same compositor.

    Changed (D04, 2026-11): plain colours are exact float ramps, as in
    `create_multichannel_rgb` (they went through an 8-bit LUT here, up to
    1/255 different); the default vmax is the robust 99.99th percentile.
    """
    if colors is not None:
        warn_deprecated("The 'colors' parameter", "'colormap'")
        if colormap is None:
            colormap = colors
    return _composite_planes(xy_list, xz_list, zy_list, vmin=vmin, vmax=vmax, gamma=gamma,
                             colormap=colormap, opacity=opacity,
                             blending=blending if blending is not None else blend,
                             soft_clip=soft_clip, eps=eps)

def black_to(color):
    """Return a black→color LinearSegmentedColormap."""
    rgb = to_rgb(color)
    return LinearSegmentedColormap.from_list(f"black_to_{color}", [(0,0,0), rgb])


def white_to(color):
    """Return a white→color LinearSegmentedColormap (an inverted colormap, for minimum blending)."""
    rgb = to_rgb(color)
    return LinearSegmentedColormap.from_list(f"white_to_{color}", [(1, 1, 1), rgb])


def _attenuated_mip(volume, axis, lo, hi, attenuation):
    """
    Attenuated maximum intensity projection along ``axis`` (napari's ``attenuated_mip``).

    Rays enter the volume at index 0 of ``axis``. Each sample is normalised to
    [0, 1] with the display range (lo, hi) and scaled by
    ``exp(-attenuation * (cumulative normalised intensity - 1))``, so bright
    structures hide what lies behind them; the brightest scaled sample is kept.
    ``attenuation`` is per voxel along the ray (0 gives the plain MIP of the
    clipped data). The result is returned in data units (``lo + m * (hi - lo)``)
    so the usual display range, gamma and colormap apply to it.
    """
    span = max(float(hi) - float(lo), 1e-12)
    v = (np.asarray(volume, dtype=np.float32) - np.float32(lo)) / np.float32(span)
    v = np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    scaled = v * np.exp(-np.float32(attenuation) * (np.cumsum(v, axis=axis) - 1.0))
    m = np.clip(scaled.max(axis=axis), 0.0, 1.0)
    return float(lo) + m * span


def _is_float_ndarray(vol):
    """True for a float ndarray (other inputs, which hold no NaN, use the plain NumPy reductions)."""
    return isinstance(vol, np.ndarray) and np.issubdtype(vol.dtype, np.floating)


# np.fmax/np.fmin.reduce equal np.max/np.min bit for bit on NaN-free data and
# run as fast, so the default MIP needs no extra pass to look for NaN.
def _nan_max_projection(vol, axis):
    """Maximum along ``axis`` ignoring NaN voxels; NaN only where the whole ray is NaN (no warning)."""
    return np.fmax.reduce(vol, axis=axis) if _is_float_ndarray(vol) else np.max(vol, axis=axis)


def _nan_min_projection(vol, axis):
    """Minimum along ``axis`` ignoring NaN voxels; NaN only where the whole ray is NaN (no warning)."""
    return np.fmin.reduce(vol, axis=axis) if _is_float_ndarray(vol) else np.min(vol, axis=axis)


def _nan_mean_projection(vol, axis):
    """Mean along ``axis`` of the non-NaN voxels; NaN only where the whole ray is NaN (no warning)."""
    out = np.mean(vol, axis=axis)
    if not _is_float_ndarray(vol) or not np.isnan(out).any():
        return out  # a NaN voxel makes its ray's mean NaN, so only then is the mean redone
    nan = np.isnan(vol)
    count = vol.shape[axis] - nan.sum(axis=axis)
    # accumulate float16 in float32, as np.mean does
    total = np.where(nan, 0, vol).sum(axis=axis, dtype=np.float32 if vol.dtype == np.float16 else None)
    out = np.full(total.shape, np.nan, dtype=vol.dtype)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _projector_for(projection, attenuation, lo=None, hi=None):
    """A ``f(volume, axis)`` projector for a canonical projection name (NaN voxels are ignored)."""
    if projection == 'mip':
        return _nan_max_projection
    if projection == 'minip':
        return _nan_min_projection
    if projection == 'average':
        return _nan_mean_projection
    return lambda vol, axis: _attenuated_mip(vol, axis, lo, hi, attenuation)


def blend_colors(intensities, base_colors, vmin=None, vmax=None, gamma=1, soft_clip=True, blending=None,
                 opacity=None):
    """
    Blend multiple channels of intensities into RGB.

    intensities : (N, C) array
        Values per point per channel. NaN means no signal in that channel.
    base_colors : list of str or rgb tuples
        Colors (or colormap names) per channel; missing entries take the
        unused palette colours.
    vmin, vmax : float or list of float, optional
        Per-channel range; a one-entry list applies to every channel, a
        shorter list is padded with None, and a list longer than C raises
        ValueError. None (per channel) is the channel's min / max ignoring NaN.
    gamma : float or list of float
        Per-channel gamma; a list shorter than C is padded with 1 (a longer
        one raises ValueError).
    blending : str, optional
        'additive' (``soft_clip`` applies), 'maximum', 'screen' or 'minimum'
        (each channel from white to its colour, per-RGB minimum). None follows
        the viewers' rule: additive for colours, maximum when a channel is a
        colormap name, minimum when every colormap starts at white.
    opacity : float or list of float, optional
        Per-channel weight 0..1 of each channel's colour, as in the image
        viewers (0 removes the channel; None is 1 for every channel). New
        (R2-23, 2026-11): used by the scatter viewer's points mode.

    Notes:
        - The same compositor as the image viewers (`create_multichannel_rgb`):
          colormap names go through their 8-bit LUT, plain colours are exact
          ramps, NaN values add no colour.
        - The `colors` parameter in top-level functions is deprecated in favor of `colormap`. To support
          multi-channel arrays, `colormap` accepts a list of colormaps. To prevent `TypeError: unhashable type: 'list'`
          from Matplotlib's `imshow`, functions explicitly unset `colormap` after mapping channels into a pre-rendered RGB array.
        - The default colormap for interactive plotters (slice, scatter, annotator) when no colormap is provided
          is `['white']` for single-channel data, and cycles through `['white', 'lime', 'magenta', 'yellow', 'cyan', 'red', 'blue']`
          for multi-channel data.

    Changed (D04, 2026-11): ``blending=None`` follows the viewers' rule (it
    was always additive, also with colormap names), and colormap names use
    their 8-bit LUT (up to 1/255 different). A vmin / vmax / gamma list
    longer than C raises ValueError (extra entries used to be ignored).
    """
    intensities = np.asarray(intensities)
    N, C = intensities.shape

    def _broadcast(v):
        # a one-entry list applies to every channel (as before); else per channel
        v = v.tolist() if isinstance(v, np.ndarray) else v
        if isinstance(v, (list, tuple)) and len(v) == 1:
            return v[0]
        return v

    columns = [intensities[:, c].astype(float) for c in range(C)]

    def _data_range(c):
        valid = columns[c][~np.isnan(columns[c])]
        return (float(valid.min()), float(valid.max())) if valid.size else (0.0, 1.0)

    gamma = gamma.tolist() if isinstance(gamma, np.ndarray) else gamma
    opacity = opacity.tolist() if isinstance(opacity, np.ndarray) else opacity
    (colors,) = _composite([columns], list(base_colors), _broadcast(vmin), _broadcast(vmax),
                           gamma, opacity, blending, soft_clip, 1e-9, default_range=_data_range)
    return colors



def compute_histogram(arr, bins=128, max_samples=1_000_000, value_range=None):
    """
    Computes a 1D histogram for array values.

    To enhance the visibility of sparse/heavy-tailed intensity distributions in interactive
    widgets, the returned bin counts are transformed into log frequencies (using log1p).
    If the array size exceeds `max_samples`, it subsamples the array to improve performance
    without noticeably degrading the visual overlay.

    `value_range` ``(lo, hi)`` fixes the outer bin edges (see
    `infer_histogram_range`); by default uint8/uint16/bool channels span their
    dtype range and other channels their data range.

    Changed (D02, 2026-11): the subsample is the package's fixed-seed random
    sample (`_conventions.sample_values`); it was strided, which aliases with
    image rows or columns and could drop whole value populations.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        return {'counts': [], 'bin_edges': []}

    scale = 1.0
    if arr.size > max_samples:
        scale = arr.size / max_samples
    arr_clean = sample_values(arr, max_samples)  # finite values; bool -> uint8
    if arr_clean.size == 0:
        return {'counts': [], 'bin_edges': []}

    # Determine absolute range based on dtype
    range_val = None if value_range is None else (float(value_range[0]), float(value_range[1]))
    if range_val is not None:
        if arr.dtype == bool:
            arr_clean = arr_clean.astype(np.uint8)
    elif arr.dtype == np.uint8:
        range_val = (0, 255)
    elif arr.dtype == np.uint16:
        range_val = (0, 65535)
    elif arr.dtype == bool:
        range_val = (0, 1)
        # np.histogram would do this conversion itself, but warns about it.
        arr_clean = arr_clean.astype(np.uint8)

    if range_val is not None:
        counts, bin_edges = np.histogram(arr_clean, bins=bins, range=range_val)
    else:
        counts, bin_edges = np.histogram(arr_clean, bins=bins)

    if scale > 1:
        # Scale back up to match original magnitude
        counts = counts * scale

    # Use log frequencies to enhance visibility of tails in widgets
    counts = np.log1p(counts)

    return {'counts': counts.tolist(), 'bin_edges': bin_edges.tolist()}


# Bit depths of common detectors (8-bit cameras, 10/12/14/16-bit sCMOS/CCD/EMCCD)
# and of wider containers. Inferred depths snap up to the next of these.
_COMMON_BIT_DEPTHS = (8, 10, 12, 14, 16, 24, 32, 64)


def _snap_bit_depth(needed, cap=64):
    for b in _COMMON_BIT_DEPTHS:
        if b >= needed:
            return min(b, cap)
    return cap


def _bit_depth_range(bits, signed):
    """Inclusive integer range of a `bits`-bit signed/unsigned type."""
    if signed:
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2 ** bits - 1


def _is_integer_valued(arr, max_samples=100_000):
    """True if the finite values (a fixed-seed random sample of them) of a float array are all integers."""
    flat = sample_values(arr, max_samples)
    return flat.size > 0 and bool(np.all(flat == np.round(flat)))


def infer_bit_depth(arr):
    """
    Smallest common detector bit depth (8, 10, 12, 14, 16, 24, 32, 64) that
    holds every value of an integer-valued image.

    The depth is inferred from the data, not the container: 12-bit camera data
    stored as uint16 (max <= 4095) gives 12, a dim 16-bit acquisition whose
    maximum happens to stay below 4096 also gives 12 (the data cannot tell the
    two apart; pass ``bit_depth`` to the viewer to enforce one). Negative values
    count the sign bit (int16 data in [-2048, 2047] gives 12, signed). Float
    images whose finite values are all integers (e.g. camera counts converted
    to float32) are treated the same way. The result never exceeds the integer
    dtype's own width.

    Returns
    -------
    (bits, signed) : (int, bool), or None for bool, empty or non-integer data.
    """
    arr = np.asarray(arr)
    if arr.size == 0 or arr.dtype == bool:
        return None
    if np.issubdtype(arr.dtype, np.integer):
        lo, hi = int(arr.min()), int(arr.max())
        # Non-negative data in a signed dtype use at most its magnitude bits
        # (int8 [0, 100] -> 0..127, not 0..255)
        cap = arr.dtype.itemsize * 8 - (1 if np.issubdtype(arr.dtype, np.signedinteger) and lo >= 0 else 0)
    elif np.issubdtype(arr.dtype, np.floating):
        lo, hi = _finite_min(arr), _finite_max(arr)
        if lo is None or max(abs(lo), abs(hi)) >= 2.0 ** 53 or not _is_integer_valued(arr):
            return None
        cap = 64
        lo, hi = int(lo), int(hi)
    else:
        return None
    signed = lo < 0
    # Magnitude bits; a signed type of b bits holds [-2**(b-1), 2**(b-1) - 1].
    needed = max(hi, -lo - 1 if signed else 0).bit_length() + int(signed)
    return _snap_bit_depth(max(needed, 1), cap), signed


def infer_histogram_range(arr, bit_depth=None):
    """
    Value range ``(lo, hi)`` the channel histogram should span, and the bit
    depth it corresponds to.

    bit_depth : None | 'auto' | int | 'dtype' | 'data'
        None / 'auto' (default): floats in [0, 1] (or [-1, 1]) span that
        normalised range; other integer-valued data span the full range of
        their inferred bit depth (`infer_bit_depth`), e.g. 0..4095 for 12-bit
        data in a uint16 array; remaining floats span their data range.
        int: enforce that bit depth (signed if the data go below 0).
        'dtype': the full range of the integer dtype (uint16 -> 0..65535),
        the data range for floats. 'data': the data range.

    Returns
    -------
    (value_range, bits) : value_range is ``(lo, hi)`` for `compute_histogram`
        (integer levels are centred in their bins, so the edges sit half a
        level outside the range), or None to let `compute_histogram` choose;
        bits is the bit depth used, or None.
    """
    arr = np.asarray(arr)
    if bit_depth is None:
        bit_depth = 'auto'
    if arr.dtype == bool:
        return (-0.5, 1.5), 1

    def _levels(bits, signed):
        lo, hi = _bit_depth_range(bits, signed)
        return (lo - 0.5, hi + 0.5), bits

    if isinstance(bit_depth, str):
        mode = bit_depth.lower()
        if mode == 'data':
            lo, hi = _finite_min(arr), _finite_max(arr)
            if lo is None:
                return None, None
            if np.issubdtype(arr.dtype, np.integer):
                return (lo - 0.5, hi + 0.5), None
            return (None, None) if lo == hi else ((lo, hi), None)
        if mode == 'dtype':
            if np.issubdtype(arr.dtype, np.integer):
                info = np.iinfo(arr.dtype)
                return (info.min - 0.5, info.max + 0.5), info.bits
            return None, None
        if mode != 'auto':
            raise ValueError(f"bit_depth must be None, 'auto', 'dtype', 'data' or an int, got {bit_depth!r}")
        if np.issubdtype(arr.dtype, np.floating):
            # Normalised floats (incl. 0/1 masks) keep their [0, 1] / [-1, 1] range
            lo, hi = _finite_min(arr), _finite_max(arr)
            if lo is not None and hi <= 1.0:
                if lo >= 0.0:
                    return (0.0, 1.0), None
                if lo >= -1.0:
                    return (-1.0, 1.0), None
        inferred = infer_bit_depth(arr)
        if inferred is not None:
            return _levels(*inferred)
        return None, None

    if isinstance(bit_depth, (bool, np.bool_)) or not isinstance(bit_depth, (int, np.integer)):
        raise ValueError(f"bit_depth must be None, 'auto', 'dtype', 'data' or an int, got {bit_depth!r}")
    bits = int(bit_depth)
    if not 1 <= bits <= 64:
        raise ValueError(f"bit_depth must be between 1 and 64, got {bits}")
    lo = _finite_min(arr) if arr.size else None
    return _levels(bits, lo is not None and lo < 0)

def _draw_slab_crosshairs(axes, geoms, x_edges, y_edges, z_edges, widths, origin=(0.0, 0.0, 0.0)):
    """
    Draw the slab boundaries on the three ortho panels.

    ``*_edges`` are the (low, high) slab edges and ``widths`` the full
    (x, y, z) extents, all in unrotated physical units measured from
    ``origin``. ``geoms`` maps each plane to the forward transform of its
    panel (a ``_PanelRotation`` or any object with ``forward(x, y)``).
    """
    wx, wy, wz = widths
    ox, oy, oz = origin

    def line(plane, a0, b0, a1, b1):
        xs, ys = geoms[plane].forward(np.array([a0, a1], dtype=float), np.array([b0, b1], dtype=float))
        axes[plane].plot(xs, ys, color='r', ls=':', alpha=0.3)

    for e in x_edges:
        line('xy', e, oy, e, oy + wy)
        line('xz', e, oz, e, oz + wz)
    for e in y_edges:
        line('xy', ox, e, ox + wx, e)
        line('zy', oz, e, oz + wz, e)
    for e in z_edges:
        line('zy', e, oy, e, oy + wy)
        line('xz', ox, e, ox + wx, e)


def _unique_path(filename, default):
    """
    Absolute path for a new export: ``~`` and ``$VAR`` expanded, and a
    numbered suffix (``name_1.svg``, ``name_2.svg``, ...) when the file
    exists, so an export never overwrites a file (D15).
    """
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(filename or default)))
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(f"{stem}_{k}{ext}"):
        k += 1
    return f"{stem}_{k}{ext}"


# File extension -> matplotlib format; text stays editable in SVG / PDF
_EXPORT_RC = {'svg': {'svg.fonttype': 'none'}, 'pdf': {'pdf.fonttype': 42}}


def _export_figure(widget, render, filename, default="figure.svg"):
    """
    Save ``render()`` to ``filename`` for a widget's save button (D15).

    The format follows the extension (svg, pdf, png or any other format
    matplotlib writes; no extension adds '.svg'), at 300 dpi with a tight
    bounding box, with editable text in SVG and PDF. An existing file is
    never overwritten (`_unique_path`). The outcome goes to the widget's
    ``status_msg`` ('saved <absolute path>' or 'error: ...'); a failure also
    warns, a success is logged.
    """
    name = filename or default
    try:
        fig = render()
        if not fig:
            raise RuntimeError("nothing to save")
        if not os.path.splitext(name)[1]:
            name = name + '.svg'
        fmt = os.path.splitext(name)[1][1:].lower()
        if fmt not in fig.canvas.get_supported_filetypes():
            raise ValueError(f"unsupported file extension '.{fmt}' (use .svg, .pdf or .png)")
        path = _unique_path(name, default)
        with matplotlib.rc_context(_EXPORT_RC.get(fmt, {})):
            fig.savefig(path, format=fmt, dpi=300, bbox_inches='tight')
    except Exception as e:
        warnings.warn(f"Error saving {name}: {e}", UserWarning, stacklevel=3)
        widget.status_msg = f"error: could not save {name}: {e}"
        return None
    _log.info("Saved to %s", path)
    widget.status_msg = f"saved {path}"
    return path


class TNIAWidgetBase(anywidget.AnyWidget):
    _esm = _ASSET_DIR.joinpath("tnia_plotting_anywidgets.js").read_text()
    _css = _ASSET_DIR.joinpath("tnia_plotting_anywidgets.css").read_text()


    # Data traits
    image_data = traitlets.Unicode().tag(sync=True)

    # Slabs. The inclusive index range [x_start, x_end] is what the GUI
    # reads; the centre / half-thickness (x_s, x_t) are Python-only traits
    # (observable, not synced: the front end never reads them; R3-16).
    x_s = traitlets.Int(0)
    y_s = traitlets.Int(0)
    z_s = traitlets.Int(0)

    x_t = traitlets.Int(1)
    y_t = traitlets.Int(1)
    z_t = traitlets.Int(1)

    x_start = traitlets.Int(0).tag(sync=True)
    x_end = traitlets.Int(1).tag(sync=True)
    y_start = traitlets.Int(0).tag(sync=True)
    y_end = traitlets.Int(1).tag(sync=True)
    z_start = traitlets.Int(0).tag(sync=True)
    z_end = traitlets.Int(1).tag(sync=True)

    sx = traitlets.Float(1.0).tag(sync=True)
    sy = traitlets.Float(1.0).tag(sync=True)
    sz = traitlets.Float(1.0).tag(sync=True)

    # Last index of each axis (the range-slider tracks); set once in __init__.
    # Changed (R3-16, 2026-11): the never-read x/y/z_min_pos, x/y/z_thick_max
    # and min_thickness traits are gone.
    x_max_pos = traitlets.Int(100).tag(sync=True)
    y_max_pos = traitlets.Int(100).tag(sync=True)
    z_max_pos = traitlets.Int(100).tag(sync=True)

    # How the range boxes show an index i: ``(i + offset) * scale`` followed
    # by ``unit``, per axis ('x', 'y', 'z'), with integer steps when not
    # calibrated (R3-08). Set by the subclasses (`_set_range_display`).
    range_display = traitlets.Dict().tag(sync=True)

    # Resolution of the on-screen PNG in dots per inch, independent of
    # rcParams['figure.dpi'] (D14); exports keep their own 300 dpi.
    display_dpi = traitlets.Float(100.0)

    # Channels
    channel_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_dtypes = traitlets.List(traitlets.Unicode()).tag(sync=True)
    channel_colors = traitlets.List(traitlets.Unicode()).tag(sync=True)

    # Channel Parameters
    vmin_list = traitlets.List(traitlets.Any()).tag(sync=True)
    vmax_list = traitlets.List(traitlets.Any()).tag(sync=True)
    # The value a cleared ("") vmin / vmax box stands for, per channel
    # (R3-04): the box placeholder, and the slider / tone-curve position.
    vmin_auto = traitlets.List(traitlets.Float()).tag(sync=True)
    vmax_auto = traitlets.List(traitlets.Float()).tag(sync=True)
    # Controls that act on each channel: 'all', or 'opacity' where vmin /
    # vmax / gamma do nothing (categorical scatter points; R3-09).
    channel_controls = traitlets.List(traitlets.Unicode()).tag(sync=True)
    gamma_list = traitlets.List(traitlets.Float()).tag(sync=True)
    opacity_list = traitlets.List(traitlets.Float()).tag(sync=True)
    histograms_data = traitlets.List(traitlets.Dict()).tag(sync=True)

    # UI Toggles
    show_crosshair = traitlets.Bool(True).tag(sync=True)
    scale_bar_show = traitlets.Bool(True).tag(sync=True)
    scale_bar_fontsize_pt = traitlets.Any(None).tag(sync=True)  # None/'' auto, 0 hides the label only
    # 'C' pressed while the pointer is over a panel (no modifier, not typing
    # in a box) moves the other slabs to the voxel under the pointer.
    sync_on_hover = traitlets.Bool(True).tag(sync=True)

    warning_msg = traitlets.Unicode("").tag(sync=True)
    # Outcome of the last export ('saved <absolute path>' or 'error: ...'),
    # shown in the widget's status line (D15): print() from a comm callback
    # does not reach the notebook cell.
    status_msg = traitlets.Unicode("").tag(sync=True)

    # Communication
    hover_coords = traitlets.Dict().tag(sync=True) # {'plane': 'xy', 'x': 0.5, 'y': 0.5, 't': 123}
    axis_bounds = traitlets.Dict().tag(sync=True) # Bounding boxes of axes in figure coords

    # Save UI
    save_filename = traitlets.Unicode("filepath_save.svg").tag(sync=True)
    save_trigger = traitlets.Int(0).tag(sync=True)

    # Copy PNG UI: JS bumps the trigger, Python answers with a base64 PNG of
    # the figure (panels + channel labels + scale bar, no GUI controls).
    copy_png_trigger = traitlets.Int(0).tag(sync=True)
    copy_png_data = traitlets.Unicode("").tag(sync=True)

    # Copy Params UI
    copy_params_trigger = traitlets.Int(0).tag(sync=True)
    copy_params_string = traitlets.Unicode("").tag(sync=True)

    def __init__(self, X, Y, Z, show_crosshair=True, sync_on_hover=True, scale_bar_show=True, scale_bar_fontsize_pt=None, **kwargs):
        super().__init__(**kwargs)
        self.show_crosshair = show_crosshair
        self.scale_bar_show = bool(scale_bar_show)
        self.scale_bar_fontsize_pt = _scale_bar_fontsize(scale_bar_fontsize_pt)
        self.sync_on_hover = sync_on_hover
        self.dims = (Z, Y, X) # (Z, Y, X) convention from numpy shape
        # Last index per axis (the slab handlers clamp to the volume)
        self.x_max_pos = max(0, X - 1)
        self.y_max_pos = max(0, Y - 1)
        self.z_max_pos = max(0, Z - 1)

    def _set_range_display(self, offsets=(0.0, 0.0, 0.0)):
        """
        Fill ``range_display``: per axis, the range boxes show index i as
        ``(i + offset) * scale`` in ``unit`` (R3-08).

        Calibrated axes show physical positions in µm, or in nm / mm chosen by
        the magnitude of the axis (`_conventions.format_length`; an unknown
        unit is shown as given); uncalibrated ones show the index (plus the
        data offset of the scatter viewer) in 'px'. ``offsets`` is (z, y, x).
        """
        calibrated = bool(getattr(self, '_pixel_sizes_given', False))
        unit_in = getattr(self, '_length_unit', None)
        out = {}
        for axis, off, step in zip('zyx', offsets, (self.sz, self.sy, self.sx)):
            off = float(off)
            if not calibrated:
                out[axis] = {'offset': off, 'scale': 1.0, 'unit': 'px', 'calibrated': False}
                continue
            scale, unit = float(step), UM
            if unit_in is None:
                n = getattr(self, f'{axis}_max_pos')
                far = max(abs(off), abs(off + n)) * float(step)
                if far > 0 and np.isfinite(far):
                    value, unit = format_length(far)
                    scale = float(step) * (value / far)
            else:
                unit = str(unit_in)
            out[axis] = {'offset': off, 'scale': scale, 'unit': unit, 'calibrated': True}
        self.range_display = out

    def _init_observers(self):
        # Sync handlers between (x_s, x_t) and (x_start, x_end)
        self.observe(self._on_st_changed, names=['x_s', 'y_s', 'z_s', 'x_t', 'y_t', 'z_t'])
        self.observe(self._on_start_end_changed, names=['x_start', 'x_end', 'y_start', 'y_end', 'z_start', 'z_end'])

        # Observe all parameters to update plot
        self.observe(self._render_wrapper, names=['x_s', 'y_s', 'z_s', 'x_t', 'y_t', 'z_t', 'x_start', 'x_end', 'y_start', 'y_end', 'z_start', 'z_end'])

        # Observe channel parameters
        self.observe(self._render_wrapper, names=['vmin_list', 'vmax_list', 'gamma_list', 'opacity_list'])

        # Observe crosshair toggle
        self.observe(self._render_wrapper, names=['show_crosshair', 'scale_bar_show', 'scale_bar_fontsize_pt',
                                                  'display_dpi'])

        # Observe hover synchronization
        self.observe(self._handle_hover_sync, names=['hover_coords'])

        # Observe save trigger
        self.observe(self._save_svg, names='save_trigger')

        # Observe copy-PNG trigger
        self.observe(self._copy_png, names='copy_png_trigger')

        # Observe copy params trigger
        self.observe(self._copy_params, names='copy_params_trigger')

        # Traits that change the first frame (display_dpi) are set before it
        # is rendered, not by a second render in _apply_trait_kwargs.
        for k, v in getattr(self, '_early_trait_kwargs', {}).items():
            setattr(self, k, v)

        # Initial render
        self._render_wrapper(None)

    # Slabs (D10). The inclusive index range [start, end] of each axis is the
    # state that is drawn, exported and used for selection; the centre and
    # half-thickness (x_s, x_t) are derived from it ((start + end) // 2 and
    # (end - start) // 2), so an even width is never rounded away. Writing
    # x_s moves the slab and keeps its width; writing x_t sets [s - t, s + t].

    _AXIS_INDEX = {'x': 2, 'y': 1, 'z': 0}

    def _axis_size(self, axis):
        return self.dims[self._AXIS_INDEX[axis]]

    def _set_slab_traits(self, axis, start, end, s=None, t=None, keep_width=False):
        """
        Write one axis's [start, end] (clamped to the volume) and its centre/half-thickness.

        The requested width ``end - start`` (before clamping) is remembered as
        the slab's nominal width, which hover moves keep (``keep_width``), so a
        slab narrowed at a border widens again when it moves back.
        """
        if not keep_width:
            if not hasattr(self, '_slab_width'):
                self._slab_width = {}
            self._slab_width[axis] = max(0, int(end) - int(start))
        n = self._axis_size(axis)
        start = max(0, min(n - 1, int(start)))
        end = max(start, min(n - 1, int(end)))
        if s is None:
            s, t = (start + end) // 2, (end - start) // 2
        was = getattr(self, '_syncing_traits', False)
        self._syncing_traits = True
        try:
            with self.hold_trait_notifications():
                setattr(self, f'{axis}_start', start)
                setattr(self, f'{axis}_end', end)
                setattr(self, f'{axis}_s', int(s))
                setattr(self, f'{axis}_t', int(t))
        finally:
            self._syncing_traits = was

    def _init_slab(self, axis, centre, half):
        """
        Place a slab from a centre and half-thickness in index units (floats).

        ``centre`` is clamped into the volume first. When ``centre -/+ half``
        are whole indices (half-voxel values, as copy-kwargs emits for even
        widths) they are the inclusive range exactly; otherwise the centre and
        half-thickness are rounded (to the nearest index, as before) and the
        range ``[s - t, s + t]`` is clipped to the volume.
        """
        n = self._axis_size(axis)
        c = min(max(float(centre), 0.0), n - 1.0)
        h = max(0.0, float(half))
        lo, hi = c - h, c + h
        if abs(lo - round(lo)) < 1e-6 and abs(hi - round(hi)) < 1e-6:
            start, end = int(round(lo)), int(round(hi))
            s, t = None, None
            if c == round(c) and h == round(h):
                s, t = int(round(c)), int(round(h))  # an odd width keeps its centre when clipped
        else:
            s, t = int(round(c)), int(round(h))
            start, end = s - t, s + t
        self._set_slab_traits(axis, start, end, s, t)

    def _translate_slab(self, axis, centre):
        """
        Centre the slab on index ``centre``, keeping its nominal width.

        For a width w (``end - start``) the slab is ``[c - w // 2, c - w // 2 + w]``,
        so ``(start + end) // 2 == c`` for odd and even widths; at a border it
        is clipped (``x_s`` stays ``c``) and regains its width when moved back.
        """
        widths = getattr(self, '_slab_width', {})
        width = widths.get(axis, getattr(self, f'{axis}_end') - getattr(self, f'{axis}_start'))
        c = int(centre)
        start = c - width // 2
        self._set_slab_traits(axis, start, start + width, c, width // 2, keep_width=True)

    def _on_st_changed(self, change):
        if getattr(self, '_syncing_traits', False):
            return
        axis, kind = change.name[0], change.name[2]
        if kind == 's':
            self._translate_slab(axis, change.new)
        else:
            s = getattr(self, f'{axis}_s')
            self._set_slab_traits(axis, s - change.new, s + change.new, s, change.new)

    def _on_start_end_changed(self, change):
        if getattr(self, '_syncing_traits', False):
            return
        axis = change.name[0]
        self._set_slab_traits(axis, getattr(self, f'{axis}_start'), getattr(self, f'{axis}_end'))

    def _render_wrapper(self, change):
        # Skip renders requested from inside a trait-synchronization pass.
        # Moving one slider writes the paired representation too (e.g. x_start
        # -> x_s + x_t), and every one of those traits is observed here, so a
        # single interaction would otherwise rasterize the figure three times.
        # The trait that started the sync is itself observed by this handler,
        # so the outer notification still renders exactly once after the sync
        # has finished. _init_observers() renders explicitly after its initial
        # sync for the same reason.
        if getattr(self, '_syncing_traits', False):
            return

        # Traits set in one message (a middle-thumb drag sets x_start and x_end;
        # hover sync sets x_s and y_s) notify one at a time after all of them
        # are set, so the second notification would render the same state again.
        state = self._render_state()
        if change is not None and getattr(change, 'name', None) in self._RENDER_STATE_TRAITS:
            if state == getattr(self, '_last_render_state', None):
                return

        open_before = _pyplot_fignums()
        fig = None
        try:
            fig = self._render()
            if fig:
                if hasattr(fig, 'axXY') and hasattr(fig, 'axZY') and hasattr(fig, 'axXZ'):
                    def get_axis_info(ax):
                        xlim = ax.get_xlim()
                        ylim = ax.get_ylim()
                        p_top_left = fig.transFigure.inverted().transform(ax.transData.transform((xlim[0], ylim[1])))
                        p_bot_right = fig.transFigure.inverted().transform(ax.transData.transform((xlim[1], ylim[0])))
                        x0 = float(p_top_left[0])
                        x1 = float(p_bot_right[0])
                        y1_mpl = float(p_top_left[1])
                        y0_mpl = float(p_bot_right[1])
                        # Map Matplotlib figure coordinates (bottom-left) to JS top-left origin coordinates
                        y0_js = float(1.0 - y1_mpl)
                        y1_js = float(1.0 - y0_mpl)
                        return {
                            'x0': x0,
                            'x1': x1,
                            'y0': y0_mpl,
                            'y1': y1_mpl,
                            'y0_js': y0_js,
                            'y1_js': y1_js,
                            'bbox': [x0, y0_js, x1 - x0, y1_js - y0_js],
                            'bbox_mpl': [x0, y0_mpl, x1 - x0, y1_mpl - y0_mpl],
                            'xlim': [float(xlim[0]), float(xlim[1])],
                            'ylim': [float(ylim[0]), float(ylim[1])]
                        }

                    self.axis_bounds = {
                        'xy': get_axis_info(fig.axXY),
                        'zy': get_axis_info(fig.axZY),
                        'xz': get_axis_info(fig.axXZ)
                    }

                buf = io.BytesIO()
                # The PNG must show the whole figure frame that axis_bounds were
                # computed in, whatever the user's savefig rcParams (e.g. 'tight').
                # At display_dpi (D14): the frame no longer follows
                # rcParams['figure.dpi'] / ['savefig.dpi'].
                dpi = float(self.display_dpi)
                with matplotlib.rc_context({'savefig.bbox': 'standard'}):
                    fig.savefig(buf, format='png', dpi=dpi)
                # Size of the displayed PNG, for distances in screen pixels
                self._png_px = tuple(float(v) for v in fig.get_size_inches() * dpi)
                self.image_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            # Recorded only after a successful render, so a state whose render
            # failed is retried by the next notification.
            self._last_render_state = state
        except Exception as e:
            # A pyplot figure opened by a render that failed half-way (e.g. by a
            # user projector) was never returned.
            _close_pyplot_figures(_pyplot_fignums() - open_before)
            if change is None:
                raise  # building the widget: report to the caller
            # From an interactive change the exception would only reach the
            # kernel log while the image silently stays stale: keep the last
            # frame and say why in the widget.
            self.warning_msg = f"⚠️ Render failed: {type(e).__name__}: {e}"
    # Traits whose values fully describe what the slab/channel observers render.
    _RENDER_STATE_TRAITS = (
        'x_s', 'y_s', 'z_s', 'x_t', 'y_t', 'z_t',
        'x_start', 'x_end', 'y_start', 'y_end', 'z_start', 'z_end',
        'vmin_list', 'vmax_list', 'gamma_list', 'opacity_list', 'show_crosshair',
        'scale_bar_show', 'scale_bar_fontsize_pt', 'display_dpi',
    )

    def _render_state(self):
        return tuple(repr(getattr(self, name)) for name in self._RENDER_STATE_TRAITS)

    def _warn_inverted_ranges(self, channels):
        """Tell the GUI that these channels' vmin > vmax is drawn with vmax = vmin + 1."""
        if not channels:
            return
        which = (f"channel {channels[0]}" if len(channels) == 1
                 else "channels " + ", ".join(str(c) for c in channels))
        msg = f"⚠️ vmin > vmax for {which}: using vmax = vmin + 1"
        self.warning_msg = f"{self.warning_msg}  {msg}" if self.warning_msg else msg

    def _handle_hover_sync(self, change):
        if not self.sync_on_hover:
            return

        coords = change.new if hasattr(change, 'new') else change.get('new', {})
        if not coords:
            return

        plane = coords.get('plane')
        frac_x = coords.get('x')
        frac_y = coords.get('y')

        info = self.axis_bounds.get(plane)
        if not info or not isinstance(info, dict):
            return

        x0, x1 = info.get('x0'), info.get('x1')
        y0_js, y1_js = info.get('y0_js'), info.get('y1_js')
        if x0 is None or x1 is None or y0_js is None or y1_js is None:
            return

        mpl_x = frac_x
        mpl_y_js = frac_y

        if not (x0 <= mpl_x <= x1 and y0_js <= mpl_y_js <= y1_js):
            return

        u = (mpl_x - x0) / (x1 - x0) if (x1 - x0) > 0 else 0.0
        v = (mpl_y_js - y0_js) / (y1_js - y0_js) if (y1_js - y0_js) > 0 else 0.0

        xlim = info.get('xlim')
        ylim = info.get('ylim')

        if not xlim or not ylim:
            return

        phys_x = xlim[0] + u * (xlim[1] - xlim[0])
        phys_y = ylim[1] + v * (ylim[0] - ylim[1])

        idx = self._phys_to_index(plane, phys_x, phys_y)
        if idx is None:
            return
        with self.hold_trait_notifications():
            # Move each slab under the cursor, keeping its width (D10)
            for axis, i in idx.items():
                self._translate_slab(axis, max(0, min(self._axis_size(axis) - 1, i)))

    def _panel_geometry(self, plane):
        """
        ``_PanelRotation`` of a displayed panel; the same object maps the
        image, the crosshairs and click/hover positions.
        """
        Z, Y, X = self.dims
        rot_z, rot_y, rot_x = _parse_rotate_view(getattr(self, 'rotate_view', None))
        if plane == 'xy':
            return _PanelRotation((Y, X), rot_z, self.sy, self.sx)
        if plane == 'zy':
            return _PanelRotation((Y, Z), rot_x, self.sy, self.sz)
        if plane == 'xz':
            return _PanelRotation((Z, X), rot_y, self.sz, self.sx)
        return None

    def _phys_to_index(self, plane, phys_x, phys_y):
        """
        Displayed physical position on a panel -> voxel indices, e.g.
        ``{'x': 3, 'y': 7}`` for the XY panel. Voxel i spans
        ``[i * s, (i + 1) * s)`` before rotation.
        """
        g = self._panel_geometry(plane)
        if g is None:
            return None
        u, v = g.inverse(phys_x, phys_y)
        if plane == 'xy':
            return {'x': int(np.floor(u / self.sx)), 'y': int(np.floor(v / self.sy))}
        if plane == 'zy':
            return {'z': int(np.floor(u / self.sz)), 'y': int(np.floor(v / self.sy))}
        return {'x': int(np.floor(u / self.sx)), 'z': int(np.floor(v / self.sz))}

    def _render(self):
        raise NotImplementedError

    def _check_trait_kwargs(self, kwargs):
        """Reject keyword arguments that are neither parameters nor traits (D12: no silent typos)."""
        traits = set(type(self).class_trait_names())
        unknown = [k for k in kwargs if k not in traits]
        if unknown:
            raise TypeError(f"{type(self).__name__}() got unexpected keyword argument(s): "
                            + ", ".join(repr(k) for k in unknown))
        # Applied before the first render (see _init_observers)
        self._early_trait_kwargs = {k: kwargs.pop(k) for k in self._EARLY_TRAITS if k in kwargs}

    _EARLY_TRAITS = ('display_dpi',)

    def _apply_trait_kwargs(self, kwargs):
        """Set traits passed as keyword arguments (after the widget is built, so observers see them)."""
        if kwargs:
            with self.hold_trait_notifications():
                for k, v in kwargs.items():
                    setattr(self, k, v)

    def _slab_origin(self):
        """Data coordinate (z, y, x) of widget index 0 (copy-kwargs positions are measured from 0)."""
        return (0, 0, 0)

    def _save_svg(self, change):
        """The figure to ``save_filename`` (see `_export_figure`); outcome in ``status_msg``."""
        _export_figure(self, self._render, self.save_filename)

    def _png_bytes(self, dpi=300):
        """The figure as PNG bytes, cropped like the SVG export (bbox 'tight')."""
        fig = self._render()
        if not fig:
            return None
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        return buf.getvalue()

    def _copy_png(self, change):
        # Clear first so an identical image still fires a change event in JS.
        self.copy_png_data = ""
        try:
            data = self._png_bytes()
        except Exception as e:
            warnings.warn(f"Error rendering PNG: {e}", UserWarning, stacklevel=2)
            data = None
        self.copy_png_data = base64.b64encode(data).decode('utf-8') if data else "error"

    def _copy_params(self, change):
        s = "\n".join(self._copy_params_lines())
        # Clear first so an unchanged view still fires a change event in JS
        # (which copies on change), and assign the final string only once.
        self.copy_params_string = ""
        self.copy_params_string = s

    def _copy_params_lines(self):
        """The current view as keyword-argument lines (one ``key=value,`` each)."""
        s = []

        # Physical centre and half-thickness of the drawn [start, end] slabs
        # (half-voxel values for even widths, so the paste rebuilds the same
        # slabs; D10). The scatter viewer's index 0 is its data minimum.
        oz, oy, ox = self._slab_origin()
        pos, half = [], []
        for axis, o, step in (('z', oz, self.sz), ('y', oy, self.sy), ('x', ox, self.sx)):
            a, b = getattr(self, f'{axis}_start'), getattr(self, f'{axis}_end')
            pos.append(float(((a + b) / 2.0 + o) * step))
            half.append(float((b - a) / 2.0 * step))
        s.append(f"slabs_position=({pos[0]}, {pos[1]}, {pos[2]}),")
        s.append(f"slabs_thickness=({half[0]}, {half[1]}, {half[2]}),")

        # Handle channels
        if hasattr(self, 'num_channels'):
            n = self.num_channels
        else:
            n = getattr(self, 'C', 1)

        vmin = [None if v == "" else float(v) for v in self.vmin_list][:n]
        vmax = [None if v == "" else float(v) for v in self.vmax_list][:n]
        gamma = [float(g) for g in self.gamma_list][:n]
        opacity = [float(o) for o in self.opacity_list][:n]

        # In case n==1 and it's a list, the user can still copy it as a list,
        # but single channel functions accept list of length 1 seamlessly.
        s.append(f"vmin={vmin},")
        s.append(f"vmax={vmax},")
        s.append(f"gamma={gamma},")
        s.append(f"opacity={opacity},")

        if hasattr(self, 'rotate_view') and self.rotate_view is not None:
            s.append(f"rotate_view={self.rotate_view},")
        if not self.scale_bar_show:
            s.append("scale_bar_show=False,")
        if _scale_bar_fontsize(self.scale_bar_fontsize_pt) is not None:
            s.append(f"scale_bar_fontsize_pt={_scale_bar_fontsize(self.scale_bar_fontsize_pt)},")
        # Keyword-only rendering options (not in the GUI), kept so a paste reproduces the view
        if getattr(self, 'blending', None) is not None:
            s.append(f"blending={self.blending!r},")
        if getattr(self, 'projection', None) is not None:
            s.append(f"projection={self.projection!r},")
            if self.projection == 'attenuated_mip':
                s.append(f"attenuation={self.attenuation},")

        return s

class TNIASliceWidget(TNIAWidgetBase):
    """
    Interactive ortho-slab viewer of a 3D image (or a list of channels).

    Besides the GUI-controlled display settings, two keyword-only rendering
    options are available (not shown in the GUI):

    - ``blending``: how channels are composited, 'additive', 'maximum',
      'screen' or 'minimum' (see `show_zyx`). None keeps the default
      (additive; maximum when a channel is a colormap name). 'minimum' draws
      each channel from white to its colour and gives a white canvas; with no
      ``colormap`` the channels get the inverted default colours.
    - ``projection`` / ``attenuation``: how each slab is projected, 'mip'
      (default), 'minip', 'average' or 'attenuated_mip' (see
      `show_zyx_projection_slabs`).
    - ``bit_depth``: range of the channel histograms. None / 'auto' infers
      the detector bit depth from the data (8, 10, 12, 14, 16, ... bits, e.g.
      0..4095 for 12-bit data stored as uint16); an int enforces one; 'dtype'
      uses the full dtype range; 'data' the data range. A list sets it per
      channel. See `infer_histogram_range`.

    Display defaults follow the package conventions (`_conventions`): per
    channel, vmin / vmax default to the channel's default display range on
    its whole volume (0, or the data minimum if negative; the robust 99.99th
    percentile), and a cleared GUI box restores that cached value. Pixel
    sizes are converted to µm; without X/Y sizes the scale bar reads 'px'.

    Changed (D01-D04, 2026-11): vmin used to default to 0.0, which clipped
    signed data, and a cleared vmin box used the current slab's minimum;
    registered Colormap instances are accepted. Per-channel vmin / vmax /
    gamma / opacity lists longer than the number of channels raise
    ValueError, as in the static viewers (extra entries used to be dropped).

    Slabs: ``slabs_position`` is the physical centre and ``slabs_thickness``
    the physical half-thickness of each slab, (Z, Y, X). The drawn slab is
    the inclusive index range ``[x_start, x_end]``; ``x_s`` / ``x_t`` are its
    centre and half-thickness in indices. Changed (D10, 2026-11): half-voxel
    values select even widths (centre 11.5, half 1.5 voxels -> [10, 13]), and
    copy-kwargs, hover sync and the annotator use the drawn range; the
    (x_s, x_t) form used to drop one voxel of an even width.

    Changed (D12, 2026-11): ``figsize=None`` is an automatic size that fits
    the physical extents (rotation included) in an 8-inch box (it was
    10 x 10 inches); unknown keyword arguments raise TypeError (they were
    ignored) and widget trait kwargs are applied; ``im`` may also be an
    xarray DataArray, a tuple of channels or an ndarray with an explicit
    ``channel_axis`` (a 4D ndarray without it raises ValueError); the
    automatic label and scale-bar font size is 3% of the figure width,
    clipped to 8-24 pt. Changed (D08): ``colors=`` is accepted with a
    FutureWarning (use ``colormap=``).

    Changed (D14 / D15 / R3-16, 2026-11): the on-screen PNG is rendered at
    ``display_dpi`` (default 100; it followed rcParams['figure.dpi']); the
    save button follows the file extension, never overwrites (numbered
    suffix) and reports in ``status_msg``; ``x_s`` ... ``z_t`` (and the
    annotator's ``points``) are no longer synced to the browser, and the
    never-read ``*_min_pos``, ``*_thick_max`` and ``min_thickness`` traits
    are gone. New synced traits for the GUI: ``vmin_auto`` / ``vmax_auto``,
    ``range_display``, ``channel_controls`` and ``status_msg``.
    """
    def __init__(self, im, pixel_sizes=None, figsize=None, colormap=None, vmin=None, vmax=None, gamma=1,
                 show_crosshair=True, sync_on_hover=True, scale_bar_show=True, scale_bar_fontsize_pt=None, slabs_position=None, slabs_thickness=None, opacity=None, rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None,
                 blending=None, projection=None, attenuation=0.05, bit_depth=None, colors=None, channel_axis=None,
                 **kwargs):
        self._check_trait_kwargs(kwargs)
        if colors is not None:
            warn_deprecated("The 'colors' parameter", "'colormap'")
            if colormap is None:
                colormap = colors
        # DataArrays, tuples, channel_axis and 2D images, as in every viewer (D12)
        im, colormap, channel_labels, pixel_sizes = _ortho_input(
            im, colormap, channel_labels, pixel_sizes, channel_axis)

        self.blending = _check_blending(blending)
        self.projection = None if projection is None else _check_projection(projection, attenuation)
        self.attenuation = float(attenuation)
        self.rotate_view = rotate_view
        self.channel_labels_input = channel_labels
        self.channel_label_fontsize_pt = channel_label_fontsize_pt
        # Handle 2D images gracefully by adding a Z dimension of 1
        if isinstance(im, list):
            if im[0].ndim == 2:
                im = [img[np.newaxis, ...] for img in im]
        elif im.ndim == 2:
            im = im[np.newaxis, ...]

        # Pixel sizes in µm (D01); X/Y unknown -> drawn as 1 with a 'px' scale bar
        (pz, py, px), self._pixel_sizes_given, self._length_unit = _viewer_pixel_sizes(pixel_sizes)
        z_t, y_t, x_t = _parse_zyx_tuple_or_dict(slabs_thickness, default_val=None)
        z_s, y_s, x_s = _parse_zyx_tuple_or_dict(slabs_position, default_val=None)

        # Determine dimensions
        im_shape = (im[0].shape if isinstance(im, list) else im.shape)
        Z, Y, X = im_shape

        super().__init__(X, Y, Z, show_crosshair=show_crosshair, sync_on_hover=sync_on_hover, scale_bar_show=scale_bar_show, scale_bar_fontsize_pt=scale_bar_fontsize_pt)

        # Override Base Trait defaults with physical scaling properly
        self.sx = px
        self.sy = py
        self.sz = pz
        self.channel_labels_input = channel_labels
        self.X_arr_phys = X * px
        self.Y_arr_phys = Y * py
        self.Z_arr_phys = Z * pz


        self.im_orig = im

        if figsize is None:
            # The widgets' automatic size: physical extents in an 8-inch box,
            # with the channel-label row the slice viewers always draw (D12)
            figsize = _auto_figsize(_ortho_extents(X * px, Y * py, Z * pz, rotate_view)[0],
                                    _ortho_extents(X * px, Y * py, Z * pz, rotate_view)[1],
                                    True, box=(8.0, 8.0), label_fontsize_pt=channel_label_fontsize_pt)
        self.figsize = figsize
        self.colormap = colormap
        self.colors_orig = colormap

        # Helper to ensure args are lists of length num_channels
        def _to_list(val, n, default):
            if val is None:
                return [default] * n
            elif isinstance(val, (list, tuple)):
                if len(val) >= n:
                    return list(val[:n])
                else:
                    return list(val) + [default] * (n - len(val))
            else:
                return [val] * n

        # Initialize Channel info
        if isinstance(im, list):
            self.num_channels = len(im)
            if channel_labels is not None:
                self.channel_names = _to_list(channel_labels, self.num_channels, "Channel")
            else:
                self.channel_names = [f"Channel {i}" for i in range(self.num_channels)]
            self.channel_dtypes = [img.dtype.name for img in im]

            # Resolve default colors to ensure stability when toggling. A single
            # colormap string must not be split into characters, and missing
            # entries get the default channel colours. Registered Colormap
            # instances become their names (the colours are a string trait).
            self.colors_resolved, blending_eff = _resolve_channels(colormap, self.num_channels, self.blending)
        else:
            self.num_channels = 1
            if channel_labels is not None:
                self.channel_names = _to_list(channel_labels, 1, "Channel")
            else:
                self.channel_names = ["Channel 0"]
            self.channel_dtypes = [im.dtype.name]
            if colormap is None:
                self.colors_resolved = _channel_specs(None, 1, self.blending)
            elif isinstance(colormap, (list, tuple)) and not (isinstance(colormap, tuple) and mcolors.is_color_like(colormap)):
                self.colors_resolved = [channel_spec(c) for c in colormap]
            else:
                self.colors_resolved = [channel_spec(colormap)]
            blending_eff = self.blending

        # Use resolve_color for channel_colors (which is passed to JS)
        GUI_CHANNEL_COLORS = [_gui_channel_color(c, blending_eff) for c in self.colors_resolved]
        self.channel_colors = GUI_CHANNEL_COLORS

        # Default display range per channel (D02): computed once on the whole
        # channel volume, the same rule as the static viewers and
        # LabelAnnotator (0, or the data minimum if negative, so signed data are
        # not clipped; robust 99.99th percentile). Also used when a vmin / vmax
        # box is cleared in the GUI.
        channels = im if isinstance(im, list) else [im]
        ranges = [default_display_range(c) for c in channels]
        self._auto_vmin = [r[0] for r in ranges]
        self._auto_vmax = [r[1] for r in ranges]
        self.vmin_list = [self._auto_vmin[i] if v is None else float(v)
                          for i, v in enumerate(per_channel(vmin, self.num_channels, 'vmin'))]
        self.vmax_list = [self._auto_vmax[i] if v is None else float(v)
                          for i, v in enumerate(per_channel(vmax, self.num_channels, 'vmax'))]
        # What a cleared box means, shown by the GUI (R3-04)
        self.vmin_auto = [float(v) for v in self._auto_vmin]
        self.vmax_auto = [float(v) for v in self._auto_vmax]
        self.channel_controls = ['all'] * self.num_channels
        self._set_range_display()
        # the static viewers' per-channel rule: a list longer than the channels raises
        self.gamma_list = per_channel(gamma, self.num_channels, 'gamma', 1.0)
        self.opacity_list = per_channel(opacity, self.num_channels, 'opacity', 1.0)

        # Physical position / half-thickness -> index slabs (D10: half-voxel
        # values select even widths exactly; see TNIAWidgetBase._init_slab)
        def _default_t(n): return max(1, n // 64)
        for axis, n, pos, half, step in (('x', X, x_s, x_t, px), ('y', Y, y_s, y_t, py), ('z', Z, z_s, z_t, pz)):
            h = half / step if half is not None else _default_t(n)
            c = pos / step if pos is not None else n // 2
            if round(h) >= n:
                # Thicker than the volume: the whole axis
                c, h = n // 2, max(1, n // 2)
            self._init_slab(axis, c, h)

        # Compute histograms over each channel's (inferred or enforced) bit-depth range
        hists = []
        channels = im if isinstance(im, list) else [im]
        for img, bd in zip(channels, _to_list(bit_depth, len(channels), None)):
            value_range, bits = infer_histogram_range(img, bd)
            h = compute_histogram(img, value_range=value_range)
            h['bit_depth'] = bits
            hists.append(h)
        self.histograms_data = hists

        # Initialize observers and render
        self._init_observers()
        self._apply_trait_kwargs(kwargs)

    def _render(self):
        Z, Y, X = self.dims

        # The slab [start, end] is inclusive, as shown by the range sliders.
        x0 = max(0, self.x_start)
        x1 = max(x0, min(X - 1, self.x_end))
        y0 = max(0, self.y_start)
        y1 = max(y0, min(Y - 1, self.y_end))
        z0 = max(0, self.z_start)
        z1 = max(z0, min(Z - 1, self.z_end))

        # Half-open index ranges for slicing
        x_lims = [x0, x1 + 1]
        y_lims = [y0, y1 + 1]
        z_lims = [z0, z1 + 1]

        clipped = False
        if x0 > self.x_start or x1 < self.x_end: clipped = True
        if y0 > self.y_start or y1 < self.y_end: clipped = True
        if z0 > self.z_start or z1 < self.z_end: clipped = True

        if clipped:
            self.warning_msg = "⚠️ Projection clipped to image boundaries"
        else:
            self.warning_msg = ""

        # Prepare arguments based on opacity
        # Channels with opacity == 0 are considered hidden and filtered out
        if isinstance(self.im_orig, list):
            # A cleared box ("") restores the channel's cached automatic value
            vmin_resolved = [self._auto_vmin[i] if v == "" else float(v) for i, v in enumerate(self.vmin_list)]
            vmax_resolved = [self._auto_vmax[i] if v == "" else float(v) for i, v in enumerate(self.vmax_list)]
            gamma_resolved = [float(g) for g in self.gamma_list]
            opacity_resolved = [float(o) for o in self.opacity_list]

            # Indices of visible channels
            visible_indices = [i for i, op in enumerate(opacity_resolved) if op > 0]

            if not visible_indices:
                # Create a placeholder figure
                fig = _new_figure(figsize=self.figsize if self.figsize else (8, 8))
                fig.text(0.5, 0.5, "No Channels Visible", ha='center', va='center', color='white')
                fig.patch.set_facecolor('black')
                return fig

            im_curr = [self.im_orig[i] for i in visible_indices]

            if isinstance(self.colors_resolved, list) and len(self.colors_resolved) == self.num_channels:
                 colors_curr = [self.colors_resolved[i] for i in visible_indices]
            else:
                 colors_curr = self.colors_resolved

            vmin_curr = [vmin_resolved[i] for i in visible_indices]
            vmax_curr = [vmax_resolved[i] for i in visible_indices]
            gamma_curr = [gamma_resolved[i] for i in visible_indices]
            opacity_curr = [opacity_resolved[i] for i in visible_indices]
            # The compositor draws vmin > vmax with vmax = vmin + 1; say so in the GUI.
            self._warn_inverted_ranges([i for i in visible_indices
                                        if vmin_resolved[i] is not None and vmin_resolved[i] > vmax_resolved[i]])
            if self.channel_labels_input is not None:
                channel_labels_curr = [self.channel_names[i] for i in visible_indices]
            else:
                import matplotlib.colors as mcolors
                def _get_name(c):
                    return c.name if isinstance(c, mcolors.Colormap) else str(c)
                channel_labels_curr = [_get_name(c) for c in colors_curr] if isinstance(colors_curr, list) else _get_name(colors_curr)

        else:
            # A cleared box ("") restores the cached automatic value
            vmin_val = self._auto_vmin[0] if self.vmin_list[0] == "" else float(self.vmin_list[0])
            vmax_val = None if self.vmax_list[0] == "" else float(self.vmax_list[0])

            # Robust default if vmax is not explicitly set
            if vmax_val is None:
                vmax_val = self._auto_vmax[0]

            if vmin_val is not None and vmin_val > vmax_val:
                # Matplotlib's norms raise for vmin > vmax; use the multichannel
                # compositor's rule (create_multichannel_rgb) so both paths agree.
                vmax_val = vmin_val + 1.0
                self._warn_inverted_ranges([0])

            gamma_val = float(self.gamma_list[0])
            opacity_val = float(self.opacity_list[0])

            if opacity_val <= 0:
                fig = _new_figure(figsize=self.figsize if self.figsize else (8, 8))
                fig.text(0.5, 0.5, "No Channels Visible", ha='center', va='center', color='white')
                fig.patch.set_facecolor('black')
                return fig

            im_curr = self.im_orig
            colors_curr = self.colors_resolved
            vmin_curr = vmin_val
            vmax_curr = vmax_val
            gamma_curr = gamma_val
            opacity_curr = opacity_val

            if self.channel_labels_input is not None:
                channel_labels_curr = self.channel_names
            else:
                import matplotlib.colors as mcolors
                def _get_name(c):
                    return c.name if isinstance(c, mcolors.Colormap) else str(c)
                channel_labels_curr = [_get_name(c) for c in colors_curr] if isinstance(colors_curr, list) else [_get_name(colors_curr)]

        # Pass pixel_sizes directly to trigger scale bar logic
        pass_pixel_sizes = None
        if getattr(self, '_pixel_sizes_given', False):
            pass_pixel_sizes = {'Z': self.sz, 'Y': self.sy, 'X': self.sx,
                                'unit': getattr(self, '_length_unit', None)}
        elif (getattr(self, 'sz', 1.0), getattr(self, 'sy', 1.0), getattr(self, 'sx', 1.0)) != (1.0, 1.0, 1.0):
            # X or Y is unknown: keep the steps that were given (e.g. the Z
            # step), but label the bar as for pixel_sizes=None
            pass_pixel_sizes = _UncalibratedXYPixelSizes(Z=self.sz, Y=self.sy, X=self.sx)

        # show_zyx_max_slabs returns a Figure
        fig = show_zyx_max_slabs(
            im_curr, x_lims, y_lims, z_lims,
            pixel_sizes=pass_pixel_sizes, figsize=self.figsize, colormap=colors_curr,
            vmin=vmin_curr, vmax=vmax_curr, gamma=gamma_curr, opacity=opacity_curr,
            rotate_view=self.rotate_view, channel_labels=channel_labels_curr,
            channel_label_fontsize_pt=self.channel_label_fontsize_pt,
            blending=self.blending, projection=self.projection, attenuation=self.attenuation,
            scale_bar_show=self.scale_bar_show,
            scale_bar_fontsize_pt=_scale_bar_fontsize(self.scale_bar_fontsize_pt),
        )

        if self.show_crosshair and fig:
            axXY = getattr(fig, 'axXY', None)
            axZY = getattr(fig, 'axZY', None)
            axXZ = getattr(fig, 'axXZ', None)

            if axXY and axZY and axXZ:
                # Slab edges sit on voxel boundaries: the projected range
                # [start, end] covers [start * s, (end + 1) * s).
                _draw_slab_crosshairs(
                    {'xy': axXY, 'zy': axZY, 'xz': axXZ},
                    {plane: self._panel_geometry(plane) for plane in ('xy', 'zy', 'xz')},
                    x_edges=(x_lims[0] * self.sx, x_lims[1] * self.sx),
                    y_edges=(y_lims[0] * self.sy, y_lims[1] * self.sy),
                    z_edges=(z_lims[0] * self.sz, z_lims[1] * self.sz),
                    widths=(X * self.sx, Y * self.sy, Z * self.sz),
                )

        return fig


# Colour of the annotator's point channel: the first not used by an image channel
_ANNOTATION_COLORS = ['red', 'orange', 'cyan', 'lime', 'magenta', 'yellow']


class _PointList(traitlets.List):
    """List trait that also takes a tuple or an (N, 3) array of points."""
    def validate(self, obj, value):
        if isinstance(value, (tuple, np.ndarray)):
            value = list(value)
        return super().validate(obj, value)


class TNIAAnnotatorWidget(TNIASliceWidget):
    """
    Subclass of TNIASliceWidget that supports interactive point annotation.

    ``points`` are voxel indices ``[z, y, x]`` (not physical coordinates).
    In delete mode a click removes the nearest point drawn within about 12
    screen pixels (at least the marker's size) among the points inside the
    panel's drawn slab. Clicks on the padding of a rotated panel are ignored.

    ``snap_depth`` (default True): a click on a projection stores, as the
    out-of-plane coordinate, the depth of the voxel the projection shows
    there (argmax of the summed visible channels inside the slab, argmin for
    'minip'), falling back to the slab centre for a flat profile or the
    'average' projection. False stores the slab centre.

    Changed (D11, 2026-11): the slab centre was always stored, up to half
    the slab thickness away from the clicked object.
    Changed (D10, 2026-11): the delete radius was 20 voxel indices mixed
    across axes, and the visible slab was the (x_s, x_t) form, which drops
    one plane of an even-width slab.
    """
    # UI Toggles
    annotation_mode = traitlets.Bool(False).tag(sync=True)
    annotation_action = traitlets.Unicode('add').tag(sync=True) # 'add' or 'delete'

    # Save CSV UI
    save_csv_filename = traitlets.Unicode("points.csv").tag(sync=True)
    save_csv_trigger = traitlets.Int(0).tag(sync=True)

    # Communication
    click_coords = traitlets.Dict().tag(sync=True) # {'plane': 'xy', 'x': 0.5, 'y': 0.5, 't': 123}

    # Data
    # List of [z, y, x] index lists. Python-only (the front end never reads
    # it; R3-16): observe it from Python as before.
    points = _PointList()

    @traitlets.validate('points')
    def _validate_points(self, proposal):
        """
        Store every point as a ``[z, y, x]`` list of ints, so tuples or integral
        float rows (e.g. from ``np.loadtxt``) match the points that clicks,
        ``add_point`` and ``remove_point`` build. Non-integral or malformed
        points are refused; points outside the volume are dropped (never drawn).
        """
        dims = getattr(self, 'dims', None)
        out = []
        n_outside = 0
        for p in proposal['value']:
            try:
                v = np.asarray(p, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                v = None
            if v is None or v.size != 3 or not np.all(np.isfinite(v)) or not np.all(v == np.round(v)):
                raise traitlets.TraitError(
                    f"each point must be three integer voxel indices (z, y, x), got {p!r}")
            pt = [int(c) for c in v]
            if dims is not None and not all(0 <= c < n for c, n in zip(pt, dims)):
                n_outside += 1
                continue
            out.append(pt)
        if n_outside:
            warnings.warn(f"Dropped {n_outside} point(s) outside the volume (Z, Y, X) = {tuple(dims)}.",
                          UserWarning, stacklevel=2)
        return out

    @property
    def annotate(self):
        return self.annotation_mode

    @annotate.setter
    def annotate(self, value):
        self.annotation_mode = bool(value)

    def __init__(self, im, colormap=None, colors=None, opacity=None, point_size_scale=0.01, annotate=False,
                 *args, snap_depth=True, channel_axis=None, **kwargs):
        if colors is not None:
            warn_deprecated("The 'colors' parameter", "'colormap'")
            if colormap is None:
                colormap = colors
        # DataArrays, tuples, channel_axis and 2D images, as in every viewer (D12)
        im, colormap, labels, ps = _ortho_input(
            im, colormap, kwargs.get('channel_labels'), kwargs.get('pixel_sizes'), channel_axis)
        if labels is not None:
            kwargs['channel_labels'] = labels
        if ps is not None:
            kwargs['pixel_sizes'] = ps
        # Normalize input to list
        if not isinstance(im, list):
            im_list = [im]
        else:
            im_list = list(im)

        # One colour per image channel, padded like the slice viewer (unused
        # palette colours; inverted defaults for minimum blending)
        colors_list, _ = _resolve_channels(colormap, len(im_list), kwargs.get('blending'))

        if opacity is None:
            opacity_list = [1.0] * len(im_list)
        elif isinstance(opacity, (int, float)):
            opacity_list = [opacity] * len(im_list)
        else:
            opacity_list = list(opacity)
            while len(opacity_list) < len(im_list):
                opacity_list.append(1.0)
            opacity_list = opacity_list[:len(im_list)]

        # Gracefully handle 2D
        if im_list[0].ndim == 2:
            im_list = [img[np.newaxis, ...] for img in im_list]

        # Get shape
        im_shape = im_list[0].shape
        Z, Y, X = im_shape
        min_dim = min(X, Y)
        self.point_size = max(3, int(np.ceil(point_size_scale * min_dim)))

        # Create persistent annotation channel
        self._annot_img = np.zeros((Z, Y, X), dtype=np.uint8)

        # Append annotation channel, in a colour no image channel uses (R2-19)
        im_list.append(self._annot_img)
        used = {_conv_color_hex(c, strict=False) for c in colors_list}
        colors_list.append(next((c for c in _ANNOTATION_COLORS if _conv_color_hex(c) not in used),
                                _ANNOTATION_COLORS[0]))
        opacity_list.append(1.0)

        # Ensure vmax is padded correctly for the annotation channel
        vmax = kwargs.get('vmax', None)

        if vmax is None:
            vmax_list = [None] * (len(im_list) - 1)
        elif isinstance(vmax, (list, tuple)):
            vmax_list = (list(vmax) + [None] * len(im_list))[:len(im_list) - 1]
        else:
            vmax_list = [vmax] * (len(im_list) - 1)
        # None entries get the superclass's robust default; annotations are 0/255.
        kwargs['vmax'] = vmax_list + [255.0]

        # The annotation channel is always 8-bit, whatever is set for the image channels
        bit_depth = kwargs.get('bit_depth', None)
        n_img = len(im_list) - 1
        if isinstance(bit_depth, (list, tuple)):
            bd_list = (list(bit_depth) + [None] * n_img)[:n_img]
        else:
            bd_list = [bit_depth] * n_img
        kwargs['bit_depth'] = bd_list + [8]

        self.snap_depth = bool(snap_depth)

        # Initialize superclass
        super().__init__(im_list, colormap=colors_list, opacity=opacity_list, *args, **kwargs)

        self.annotation_mode = bool(annotate)

        # Override the last channel name
        self.channel_names = self.channel_names[:-1] + ['Annotations']

    def _init_observers(self):
        super()._init_observers()
        self.observe(self._handle_click, names=['click_coords'])
        self.observe(self._on_points_changed, names=['points'])
        # annotation_mode changes no pixel (the JS sets the cursor), so it
        # does not re-render (R3-16).
        self.observe(self._save_csv, names='save_csv_trigger')

    def _handle_click(self, change):
        if not self.annotation_mode:
            return

        coords = change.new if hasattr(change, 'new') else change.get('new', {})
        if not coords:
            return

        plane = coords.get('plane')
        frac_x = coords.get('x')
        frac_y = coords.get('y')

        info = self.axis_bounds.get(plane)
        if not info or not isinstance(info, dict):
            return

        x0, x1 = info.get('x0'), info.get('x1')
        y0_js, y1_js = info.get('y0_js'), info.get('y1_js')
        if x0 is None or x1 is None or y0_js is None or y1_js is None:
            return

        mpl_x = frac_x
        mpl_y_js = frac_y

        if not (x0 <= mpl_x <= x1 and y0_js <= mpl_y_js <= y1_js):
            return

        u = (mpl_x - x0) / (x1 - x0) if (x1 - x0) > 0 else 0.0
        v = (mpl_y_js - y0_js) / (y1_js - y0_js) if (y1_js - y0_js) > 0 else 0.0

        xlim = info.get('xlim')
        ylim = info.get('ylim')

        if not xlim or not ylim:
            return

        phys_x = xlim[0] + u * (xlim[1] - xlim[0])
        phys_y = ylim[1] + v * (ylim[0] - ylim[1])

        Z, Y, X = self.dims

        idx = self._phys_to_index(plane, phys_x, phys_y)
        if idx is None:
            return
        # A click on the padding of a rotated panel maps outside the volume;
        # it must not be clamped onto the border and added as a real point.
        extent = {'x': X, 'y': Y, 'z': Z}
        outside = any(not 0 <= i < extent[axis] for axis, i in idx.items())
        # The out-of-plane coordinate: the depth of what the projection shows
        # at the clicked pixel (D11), else the slab centre
        normal = next(a for a in 'xyz' if a not in idx)
        if not outside:
            idx[normal] = self._snap_depth(normal, idx)
        data_x = idx.get('x', self.x_s)
        data_y = idx.get('y', self.y_s)
        data_z = idx.get('z', self.z_s)

        data_x = max(0, min(X - 1, data_x))
        data_y = max(0, min(Y - 1, data_y))
        data_z = max(0, min(Z - 1, data_z))

        if outside:
            return  # padding of a rotated panel: neither add nor delete
        if self.annotation_action == 'add':
            self.add_point(data_z, data_y, data_x)
        elif self.annotation_action == 'delete':
            self._delete_near(plane, frac_x, frac_y, info)

    def _snap_depth(self, axis, idx):
        """
        Index along ``axis`` (the panel's normal) of the voxel the projection
        shows at the in-plane indices ``idx`` (D11).

        With ``snap_depth`` the visible image channels (opacity > 0, the
        annotation channel excluded), each normalised by its display range,
        are summed along the ray inside the drawn slab; the argmax ('mip') or
        argmin ('minip') is returned. For 'attenuated_mip' each channel's
        profile is first scaled exactly as the projection scales it
        (``v * exp(-attenuation * (cumsum(v) - 1))`` on the clipped
        normalised samples, rays entering at the slab start), so the voxel
        that wins the display wins the snap. NaN voxels are ignored, as the
        projections ignore them. A flat profile (all-NaN included), the
        'average' projection or ``snap_depth=False`` give the slab centre.
        """
        centre = getattr(self, f'{axis}_s')
        if not self.snap_depth or self.projection == 'average':
            return centre
        lo, hi = getattr(self, f'{axis}_start'), getattr(self, f'{axis}_end')
        if hi <= lo:
            return centre
        sl = {a: idx.get(a) for a in 'xyz'}
        sl[axis] = slice(lo, hi + 1)
        minip = self.projection == 'minip'
        profile = None
        n_img = self.num_channels - 1  # the last channel holds the annotations
        for i in range(n_img):
            if float(self.opacity_list[i]) <= 0:
                continue
            vmin = self._auto_vmin[i] if self.vmin_list[i] == "" else float(self.vmin_list[i])
            vmax = self._auto_vmax[i] if self.vmax_list[i] == "" else float(self.vmax_list[i])
            ray = np.asarray(self.im_orig[i][sl['z'], sl['y'], sl['x']], dtype=np.float64)
            ray = (ray - vmin) / max(vmax - vmin, 1e-12)
            if self.projection == 'attenuated_mip':
                v = np.clip(np.nan_to_num(ray, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                ray = v * np.exp(-float(self.attenuation) * (np.cumsum(v) - 1.0))
            else:
                # NaN never wins (the projections skip it); infinities are
                # bounded so that summing channels stays finite.
                ray = np.clip(ray, -1e12, 1e12)
                ray[np.isnan(ray)] = 1e15 if minip else -1e15
            profile = ray if profile is None else profile + ray
        if profile is None or profile.size == 0:
            return centre
        shown = np.abs(profile) < 1e14  # voxels with no NaN in any visible channel
        if not shown.any() or np.all(profile[shown] == profile[shown][0]):
            return centre
        k = np.argmin(profile) if minip else np.argmax(profile)
        return lo + int(k)

    # Delete radius in rendered (PNG) pixels (D10): fixed on screen, whatever
    # the voxel size or the panel's zoom.
    _DELETE_RADIUS_PX = 12.0

    def _delete_near(self, plane, frac_x, frac_y, info):
        """Remove the visible point drawn nearest to a click, if within ~12 screen px.

        Visible means inside the drawn slab ``[start, end]`` of the panel's
        normal axis. Distances are measured on the rendered image between the
        click and each point's drawn position (voxel centre, rotation applied).
        """
        if not self.points:
            return
        pts = np.array(self.points)
        normal = {'xy': ('z', 0), 'zy': ('x', 2), 'xz': ('y', 1)}[plane]
        lo, hi = getattr(self, f'{normal[0]}_start'), getattr(self, f'{normal[0]}_end')
        visible = pts[(pts[:, normal[1]] >= lo) & (pts[:, normal[1]] <= hi)]
        if visible.size == 0:
            return
        # Panel coordinates (horizontal, vertical) of each point's voxel centre
        steps = {0: self.sz, 1: self.sy, 2: self.sx}
        col_ax, row_ax = {'xy': (2, 1), 'zy': (0, 1), 'xz': (2, 0)}[plane]
        a = (visible[:, col_ax] + 0.5) * steps[col_ax]
        b = (visible[:, row_ax] + 0.5) * steps[row_ax]
        a, b = self._panel_geometry(plane).forward(a, b)

        (x0, x1), (y0, y1) = (info['x0'], info['x1']), (info['y0_js'], info['y1_js'])
        (xl0, xl1), (yl0, yl1) = info['xlim'], info['ylim']
        w_px, h_px = getattr(self, '_png_px', (1.0, 1.0))
        fx = x0 + (np.asarray(a) - xl0) / (xl1 - xl0) * (x1 - x0)
        fy = y0 + (np.asarray(b) - yl1) / (yl0 - yl1) * (y1 - y0)
        dist = np.hypot((fx - frac_x) * w_px, (fy - frac_y) * h_px)

        # At least the drawn marker's half-size on screen
        marker_um = max(1, self.point_size // 2) * min(self.sx, self.sz)
        marker_px = marker_um / abs(xl1 - xl0) * (x1 - x0) * w_px
        k = int(np.argmin(dist))
        if dist[k] <= max(self._DELETE_RADIUS_PX, marker_px):
            self.remove_point(*visible[k])

    def add_point(self, z, y, x):
        """Programmatically add a point"""
        pt = [int(z), int(y), int(x)]
        if pt not in self.points:
            self.points = self.points + [pt]

    def remove_point(self, z, y, x):
        """Programmatically remove a point"""
        pt = [int(z), int(y), int(x)]
        new_points = []
        deleted = False
        for p in self.points:
            if not deleted and p == pt:
                deleted = True
                continue
            new_points.append(p)
        if deleted:
            self.points = new_points

    def _save_csv(self, change):
        """
        Write ``points`` to ``save_csv_filename`` (D15).

        Columns ``z,y,x`` are voxel indices, followed by the physical
        positions ``z_um,y_um,x_um`` (index x pixel size; ``*_px`` when the
        X/Y pixel sizes are unknown, in the panels' units). The file is
        always written (header only without points, with a warning), never
        overwrites an existing file (``points_1.csv``, ...), and the outcome
        goes to ``status_msg``.

        Changed (D15, 2026-11): the CSV had only ``z,y,x``, overwrote the
        file, and nothing was written without points, so a file with
        deleted points stayed the latest.
        """
        filename = None
        try:
            filename = _unique_path(self.save_csv_filename, "points.csv")
            suffix = 'um' if self._pixel_sizes_given else 'px'
            steps = (self.sz, self.sy, self.sx)
            with open(filename, 'w') as f:
                f.write(f"z,y,x,z_{suffix},y_{suffix},x_{suffix}\n")
                for p in self.points:
                    phys = [float(i) * float(s) for i, s in zip(p, steps)]
                    f.write(f"{p[0]},{p[1]},{p[2]},{phys[0]!r},{phys[1]!r},{phys[2]!r}\n")
        except Exception as e:
            name = filename or self.save_csv_filename
            warnings.warn(f"Error saving CSV {name}: {e}", UserWarning, stacklevel=2)
            self.status_msg = f"error: could not save {name}: {e}"
            return
        if not self.points:
            warnings.warn(f"No points to save: wrote a CSV with only the header to {filename}.",
                          UserWarning, stacklevel=2)
        _log.info("Saved %d points to %s", len(self.points), filename)
        self.status_msg = f"saved {len(self.points)} points to {filename}"


    def _on_points_changed(self, change):
        # Update annotation mask efficiently
        self._annot_img.fill(0)
        Z, Y, X = self.dims
        s = max(1, self.point_size // 2)
        R = s * min(self.sx, self.sz)
        s_xy = max(1, int(np.round(R / self.sx)))
        s_z = max(1, int(np.round(R / self.sz)))
        for p in self.points:
            pz, py, px = p
            z0 = max(0, pz - s_z)
            z1 = min(Z, pz + s_z + 1)
            y0 = max(0, py - s_xy)
            y1 = min(Y, py + s_xy + 1)
            x0 = max(0, px - s_xy)
            x1 = min(X, px + s_xy + 1)
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                continue  # marker box entirely outside the volume
            self._annot_img[z0:z1, y0:y1, x0:x1] = 255

        self._render_wrapper(change)

def _int_bounds(a):
    """(floor(min), ceil(max)) of a coordinate array; (0, 0) when it is empty."""
    if len(a) == 0:
        return 0.0, 0.0
    return float(np.floor(np.min(a))), float(np.ceil(np.max(a)))


def _scatter_auto_figsize(xb, yb, zb, sizes, rotate_view, has_labels, label_fontsize_pt=None):
    """Automatic scatter figure size: the data box ``[min - 0.5, max + 0.5)`` per axis, in physical units."""
    pz, py, px = sizes
    (x0, x1), (y0, y1), (z0, z1) = xb, yb, zb
    cols, rows = _ortho_extents((x1 + 1 - x0) * px, (y1 + 1 - y0) * py, (z1 + 1 - z0) * pz, rotate_view)
    return _auto_figsize(cols, rows, has_labels, box=(8.0, 8.0), label_fontsize_pt=label_fontsize_pt)


_CHANNEL_MODES = ('auto', 'values', 'categories', 'index_lists')
# 'auto' reads one integer label per point as categories up to this many
# distinct labels; more are ambiguous (intensities?) and need channel_mode.
_MAX_AUTO_CATEGORIES = 64


def _is_index_list(a, n):
    """True if ``a`` could be indices of distinct points among ``n`` (integers in [0, n), no repeats)."""
    a = np.asarray(a)
    if a.size and not (np.issubdtype(a.dtype, np.integer) and a.min() >= 0 and a.max() < n):
        return False
    return len(np.unique(a)) == a.size


def _classify_scatter_channels(channels, n, channel_mode='auto'):
    """
    Interpret the scatter viewer's ``channels`` for ``n`` points (D13).

    Returns ``(mode, data)``: 'single' (None), 'cont_single' ((n,) floats),
    'cont_multi' ((n, C) floats), 'ids' (``(codes, labels)`` from
    ``np.unique``, so sparse and negative labels work), or 'idx_lists'
    (list of integer index arrays).

    ``channel_mode='auto'`` infers, and raises ValueError when the input is
    ambiguous: a list of equal-length (n) arrays is values unless every one
    is a valid index list (then ambiguous); lists of other lengths are index
    lists; an (n, C) array or one float array is values; one integer/bool
    array is categories up to 64 distinct labels (ambiguous above); strings
    are categories. 'values', 'categories' and 'index_lists' force a reading.
    """
    if channel_mode not in _CHANNEL_MODES:
        raise ValueError(f"channel_mode must be one of {_CHANNEL_MODES}, got {channel_mode!r}")
    if channels is None:
        return 'single', None
    hint = "pass channel_mode='values', 'categories' or 'index_lists'"
    if isinstance(channels, (list, tuple)):
        if len(channels) == 0:
            # No channel at all: zero index lists (as before D13), not an ambiguity
            if channel_mode not in ('auto', 'index_lists'):
                raise ValueError(f"channels is an empty list; channel_mode={channel_mode!r} needs data")
            return 'idx_lists', []
        arrs = [np.asarray(a).reshape(-1) for a in channels]
        per_point = all(len(a) == n for a in arrs)
        mode = channel_mode
        if mode == 'auto':
            if not per_point:
                mode = 'index_lists'
            elif any(not np.issubdtype(a.dtype, np.integer) for a in arrs) or \
                    not all(_is_index_list(a, n) for a in arrs):
                mode = 'values'
            else:
                raise ValueError(f"channels is a list of {len(arrs)} integer arrays of one entry per point that "
                                 f"are also valid point-index lists; {hint}.")
        if mode == 'values':
            if not per_point:
                raise ValueError(f"channel_mode='values' needs one value per point ({n}) in every channel, "
                                 f"got lengths {[len(a) for a in arrs]}")
            return 'cont_multi', np.stack([a.astype(float) for a in arrs], axis=1)
        if mode == 'index_lists':
            lists = []
            for a in arrs:
                if a.size and not (np.issubdtype(a.dtype, np.integer) or np.all(a == np.round(a))):
                    raise ValueError("index lists must hold integer point indices")
                a = a.astype(int)
                if a.size and (a.min() < 0 or a.max() >= n):
                    raise ValueError(f"index lists must hold point indices in [0, {n}), got [{a.min()}, {a.max()}]")
                lists.append(a)
            return 'idx_lists', lists
        raise ValueError("channel_mode='categories' takes one label per point (an array), not a list")

    a = np.asarray(channels)
    if a.ndim == 2:
        if channel_mode not in ('auto', 'values'):
            raise ValueError(f"an (n_points, n_channels) array holds values; got channel_mode={channel_mode!r}")
        if a.shape[0] != n:
            raise ValueError(
                f"A 2D channels array must have shape (n_points, n_channels) = ({n}, C), got {a.shape}")
        return 'cont_multi', a.astype(float)
    a = a.reshape(-1)
    if len(a) != n:
        if channel_mode in ('auto', 'index_lists'):
            return _classify_scatter_channels([a], n, 'index_lists')
        raise ValueError(f"channels has {len(a)} entries for {n} points")
    numeric = np.issubdtype(a.dtype, np.number) or a.dtype == bool
    mode = channel_mode
    if mode == 'auto':
        if not numeric:
            mode = 'categories'
        elif np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
            if len(np.unique(a)) > _MAX_AUTO_CATEGORIES:
                raise ValueError(f"channels has {len(np.unique(a))} distinct integer values: categories or "
                                 f"values (e.g. intensities)? {hint}.")
            mode = 'categories'
        else:
            mode = 'values'
    if mode == 'values':
        if not numeric:
            raise ValueError("channel_mode='values' needs numeric channels")
        return 'cont_single', a.astype(float)
    if mode == 'categories':
        labels, codes = np.unique(a, return_inverse=True)
        return 'ids', (codes.reshape(-1).astype(int), labels)
    return _classify_scatter_channels([a], n, 'index_lists')


def _filter_channels(mode, data, keep):
    """Keep the channel data of the points where ``keep`` (index lists are re-indexed)."""
    if keep is None or mode == 'single':
        return data
    if mode == 'idx_lists':
        new_index = np.cumsum(keep) - 1
        return [new_index[ix[keep[ix]]] for ix in data]
    if mode == 'ids':
        return data[0][keep], data[1]
    return data[keep]


def _finite_mask(X, Y, Z, warn=True):
    """Coordinates as 1D arrays, and the mask of points with finite coordinates (None if all are)."""
    X, Y, Z = (np.asarray(a).reshape(-1) for a in (X, Y, Z))
    if not (len(X) == len(Y) == len(Z)):
        raise ValueError(f"X, Y and Z must have the same length, got {len(X)}, {len(Y)}, {len(Z)}")
    keep = np.isfinite(X.astype(float)) & np.isfinite(Y.astype(float)) & np.isfinite(Z.astype(float))
    if keep.all():
        return X, Y, Z, None
    if warn:
        warnings.warn(f"Dropping {int((~keep).sum())} of {len(X)} points with non-finite coordinates.",
                      UserWarning, stacklevel=4)
    return X, Y, Z, keep


def _finite_points(X, Y, Z, channels=None):
    """
    Drop points with a non-finite coordinate (with a warning), together with
    their per-point channel values; index-list channels are re-indexed.
    """
    X, Y, Z = (np.asarray(a).reshape(-1) for a in (X, Y, Z))
    if not (len(X) == len(Y) == len(Z)):
        raise ValueError(f"X, Y and Z must have the same length, got {len(X)}, {len(Y)}, {len(Z)}")
    keep = np.isfinite(X.astype(float)) & np.isfinite(Y.astype(float)) & np.isfinite(Z.astype(float))
    if keep.all():
        return X, Y, Z, channels
    n_drop = int((~keep).sum())
    warnings.warn(f"Dropping {n_drop} of {len(X)} points with non-finite coordinates.",
                  UserWarning, stacklevel=3)
    N = len(X)
    new_index = np.cumsum(keep) - 1

    if channels is not None:
        if isinstance(channels, (list, tuple)):
            arrs = [np.asarray(a) for a in channels]
            if all(len(a) == N for a in arrs) and any(np.issubdtype(a.dtype, np.floating) for a in arrs):
                channels = [a[keep] for a in arrs]  # one value per point and channel
            else:
                channels = [new_index[ix[keep[ix]]] for ix in (np.asarray(i, dtype=int) for i in arrs)]
        else:
            channels = np.asarray(channels)[keep]
    return X[keep], Y[keep], Z[keep], channels


class TNIAScatterWidget(TNIAWidgetBase):
    """
    Interactive ortho-slab viewer of 3D points (see
    `show_zyx_max_scatter_interactive`, whose docstring lists every
    parameter). ``X_arr``, ``Y_arr``, ``Z_arr`` are coordinates in data
    units; ``pixel_sizes`` (Z, Y, X) converts them to µm.

    Changed (D10, 2026-11): ``slabs_position`` / ``slabs_thickness`` are
    physical (data units x pixel size), like the image viewer: the slab
    index is ``physical / pixel size - data minimum``; copy-kwargs emits the
    same physical values. The legacy ``x_s`` ... ``z_t`` kwargs (FutureWarning,
    D08) are now applied; they used to be ignored by the class.

    Changed (D13, 2026-11): new ``channel_mode='auto'|'values'|'categories'|
    'index_lists'``. In 'auto', ambiguous integer channels raise ValueError
    (one integer array with more than 64 distinct values, or equal-length
    integer lists that are also valid index lists); categories go through
    ``np.unique`` (negative and sparse labels kept, channels named after the
    labels, see ``category_labels``).

    Changed (D12, 2026-11): unknown keyword arguments raise TypeError and
    trait kwargs are applied; ``figsize=None`` is an automatic size from the
    physical extents in an 8-inch box (it was 10 x 10 inches).

    Changed (D14 / R3-09 / R2-23 / R3-04, 2026-11): in points mode the
    per-channel opacity multiplies ``alpha`` (0 hides the channel) and gamma
    applies to a single continuous channel (both were ignored); categorical
    channels (one colour per channel, points mode) expose only opacity in
    the GUI (``channel_controls``) and 'ids' has no histogram of ID values;
    a cleared vmin / vmax box means ``vmin_auto`` / ``vmax_auto`` (0 and the
    data maximum), not the data minimum or the panel range. See also
    `TNIASliceWidget` (display_dpi, exports, synced traits).
    """
    def __init__(self, X_arr, Y_arr, Z_arr, channels=None, pixel_sizes=None, render='points', bins=512,
                 point_size=4, alpha=0.6, colormap=None, colors=None, opacity=None, gamma=1, vmin=None, vmax=None, figsize=None,
                 show_crosshair=True, sync_on_hover=True, scale_bar_show=True, scale_bar_fontsize_pt=None, subplot_bg=None, slabs_position=None, x_s=None, y_s=None, z_s=None, slabs_thickness=None, x_t=None, y_t=None, z_t=None, rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None,
                 blending=None, channel_mode='auto', **kwargs):
        self._check_trait_kwargs(kwargs)
        # Keyword-only channel compositing (not in the GUI); see `show_zyx`.
        # 'minimum' also switches to a white canvas and inverted default colours.
        self.blending = _check_blending(blending)
        self.rotate_view = rotate_view
        self.channel_label_fontsize_pt = channel_label_fontsize_pt
        if colors is not None:
            warn_deprecated("The 'colors' parameter", "'colormap'")
            if colormap is None:
                colormap = colors

        # Legacy per-axis slab kwargs (the factories already route theirs)
        if any(v is not None for v in (x_s, y_s, z_s)):
            warn_deprecated("Passing 'x_s'/'y_s'/'z_s'", "'slabs_position'")
            if slabs_position is None:
                slabs_position = (z_s, y_s, x_s)
        if any(v is not None for v in (x_t, y_t, z_t)):
            warn_deprecated("Passing 'x_t'/'y_t'/'z_t'", "'slabs_thickness'")
            if slabs_thickness is None:
                slabs_thickness = (z_t, y_t, x_t)

        if channel_mode is None:
            channel_mode = 'auto'
        X_arr, Y_arr, Z_arr, keep = _finite_mask(X_arr, Y_arr, Z_arr)
        mode, ch_data = _classify_scatter_channels(channels, len(X_arr), channel_mode)
        if keep is not None:
            X_arr, Y_arr, Z_arr = X_arr[keep], Y_arr[keep], Z_arr[keep]
            ch_data = _filter_channels(mode, ch_data, keep)
        self.X_arr = X_arr
        self.Y_arr = Y_arr
        self.Z_arr = Z_arr

        # Compute bounds for init
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = (_int_bounds(a) for a in (X_arr, Y_arr, Z_arr))

        X_dim = int(np.ceil(xmax - xmin + 1))
        Y_dim = int(np.ceil(ymax - ymin + 1))
        Z_dim = int(np.ceil(zmax - zmin + 1))

        super().__init__(X_dim, Y_dim, Z_dim, show_crosshair=show_crosshair, sync_on_hover=sync_on_hover, scale_bar_show=scale_bar_show, scale_bar_fontsize_pt=scale_bar_fontsize_pt)

        self.channels = channels
        # Pixel sizes in µm (D01); X/Y unknown -> drawn as 1 with a 'px' scale bar
        (pz, py, px), self._pixel_sizes_given, self._length_unit = _viewer_pixel_sizes(pixel_sizes)
        self.sx = px
        self.sy = py
        self.sz = pz
        self.channel_labels_input = channel_labels
        self.render = render
        self.bins = bins
        self.point_size = point_size
        self.alpha = alpha
        self.colors = colors
        if figsize is None:
            figsize = _scatter_auto_figsize((xmin, xmax), (ymin, ymax), (zmin, zmax), (pz, py, px),
                                            rotate_view, channel_labels is not None, channel_label_fontsize_pt)
        self.figsize = figsize

        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        self.subplot_bg = subplot_bg

        # Precompute Density helpers
        def _resolve_bins(B):
            if isinstance(B, (tuple, list)) and len(B) == 2:
                bx, by = int(B[0]), int(B[1])
            else:
                bx = by = int(B)
            bx = max(1, bx); by = max(1, by)
            return bx, by

        self.BX, self.BY = _resolve_bins(bins)

        # Widget index i covers data [min + i - 0.5, min + i + 0.5), so points on
        # integer coordinates sit in the middle of their index; the data box is
        # [min - 0.5, max + 0.5) per axis.
        self._box_size = (xmax + 1 - xmin, ymax + 1 - ymin, zmax + 1 - zmin)
        Wx, Wy, Wz = self._box_size
        self._box_lo = (xmin - 0.5, ymin - 0.5, zmin - 0.5)
        self._box_centre_phys = ((xmin + xmax) / 2.0 * px, (ymin + ymax) / 2.0 * py, (zmin + zmax) / 2.0 * pz)
        # Density bins: BX x BY in XY; Z gets bins of about the same physical
        # size, so each panel is drawn with its true physical aspect.
        bxs, bys = Wx / self.BX * px, Wy / self.BY * py
        self.BZ = int(np.clip(np.round(Wz * pz / min(bxs, bys)), 1, 4 * max(self.BX, self.BY)))
        self._bin_data = (Wx / self.BX, Wy / self.BY, Wz / self.BZ)
        self._bin_phys = (bxs, bys, Wz / self.BZ * pz)

        # Analyze channels
        self.N = len(self.X_arr)

        # Channel interpretation (D13): see _classify_scatter_channels
        self.mode = mode
        self.ch_ids = self.category_labels = None
        self.cont_single = self.cont_multi = self.idx_lists = None
        if mode == 'ids':
            self.ch_ids, self.category_labels = ch_data
        elif mode == 'cont_single':
            self.cont_single = ch_data
        elif mode == 'cont_multi':
            self.cont_multi = ch_data
        elif mode == 'idx_lists':
            self.idx_lists = ch_data

        if self.mode in ('single',):
            self.C = 1
        elif self.mode == 'ids':
            self.C = max(1, len(self.category_labels))
        elif self.mode == 'idx_lists':
            self.C = len(self.idx_lists)
        elif self.mode == 'cont_single':
            self.C = 1
        else:
            self.C = self.cont_multi.shape[1]

        if colormap is None:
            defaults = (_DEFAULT_INVERTED_CHANNEL_COLORS if self.blending == 'minimum'
                        else _DEFAULT_CHANNEL_COLORS)
            default_cols = list(defaults[:1] if self.C == 1 else defaults)
            while len(default_cols) < self.C:
                 default_cols += default_cols
            self.colors_use = default_cols[:max(1, self.C)]
        else:
            if isinstance(colormap, str) or (isinstance(colormap, tuple) and mcolors.is_color_like(colormap)):
                self.colors_use = [colormap]
            elif isinstance(colormap, (list, tuple)):
                self.colors_use = list(colormap)
            else:
                self.colors_use = [colormap]
            # registered Colormap instances -> names, colour tuples -> hex
            self.colors_use = [channel_spec(c) for c in self.colors_use]

        if self.mode == 'ids' and len(self.colors_use) == 1 and is_colormap(self.colors_use[0]):
            cmap = matplotlib.colormaps[self.colors_use[0]]
            self.colors_rgb = [cmap(i / max(1, self.C - 1))[:3] for i in range(self.C)]
        else:
            self.colors_rgb = [matplotlib.colors.to_rgb(resolve_color(c)) for c in self.colors_use]

        # The blending=None rule of every viewer (D04), applied to the colours
        # that are composited (as in density mode); 'minimum' also gives the
        # white theme. ``self.blending`` keeps what was asked (copy-params).
        self._blend = resolve_blending(self._composite_colors(), self.blending, stacklevel=3)
        if subplot_bg is None:
            subplot_bg = 'white' if self._blend == 'minimum' else 'black'
        self.subplot_bg = subplot_bg


        # Set traitlets lists for interactive parameters
        def _to_list(val, n, default):
            if val is None:
                return [default] * n
            elif isinstance(val, (list, tuple)):
                if len(val) >= n:
                    return list(val[:n])
                else:
                    return list(val) + [default] * (n - len(val))
            else:
                return [val] * n

        def _resolve_vmin(val, n):
            lst = per_channel(val, n, 'vmin')
            res = []
            for x in lst:
                res.append(0.0 if x is None else float(x))
            return res

        def _resolve_vmax(val, n, mode, cont_single, cont_multi):
            lst = per_channel(val, n, 'vmax')
            res = []
            for i, x in enumerate(lst):
                if x is not None:
                    res.append(float(x))
                else:
                    if mode == 'cont_single':
                        m = _finite_max(cont_single)
                    elif mode == 'cont_multi':
                        m = _finite_max(cont_multi[:, i])
                    else:
                        m = 1.0
                    res.append(1.0 if m is None else m)
            return res

        self.vmin_list = _resolve_vmin(vmin, self.C)
        self.vmax_list = _resolve_vmax(vmax, self.C, self.mode, self.cont_single, self.cont_multi)
        # A cleared box ("") restores these defaults (R3-04): it used to mean
        # the data minimum / maximum in points mode and the panel range in
        # density mode.
        self.vmin_auto = _resolve_vmin(None, self.C)
        self.vmax_auto = _resolve_vmax(None, self.C, self.mode, self.cont_single, self.cont_multi)
        self.gamma_list = per_channel(gamma, self.C, 'gamma', 1.0)
        self.opacity_list = per_channel(opacity, self.C, 'opacity', 1.0)
        # Categorical points (one colour per channel) only have an opacity (R3-09)
        categorical = self.mode in ('single', 'ids', 'idx_lists')
        self.channel_controls = (['opacity' if categorical and render != 'density' else 'all'] * self.C)

        if channel_labels is not None:
            self.channel_names = _to_list(channel_labels, self.C, "Channel")
        else:
            self.channel_names = ([str(v) for v in self.category_labels] if self.mode == 'ids'
                                  else [f"Channel {i}" for i in range(self.C)])
        self.channel_dtypes = ["float"] * self.C
        # The slice viewer's GUI colour rule (R3-10)
        self.channel_colors = [_gui_channel_color(c, self._blend) for c in self._label_colors()]

        z_t_in, y_t_in, x_t_in = _parse_zyx_tuple_or_dict(slabs_thickness, default_val=None)
        z_s_in, y_s_in, x_s_in = _parse_zyx_tuple_or_dict(slabs_position, default_val=None)

        # Slabs in physical units, as in the image viewers (D10): data
        # coordinate = physical / pixel size, widget index = data - minimum.
        for axis, n, pos, half, step, lo, hi in (
                ('x', X_dim, x_s_in, x_t_in, px, xmin, xmax),
                ('y', Y_dim, y_s_in, y_t_in, py, ymin, ymax),
                ('z', Z_dim, z_s_in, z_t_in, pz, zmin, zmax)):
            h = half / step if half is not None else max(1, n // 64)
            c = pos / step - lo if pos is not None else int((hi - lo) / 2)
            self._init_slab(axis, c, h)

        self._set_range_display((self.zmin, self.ymin, self.xmin))

        # Compute histograms. Categorical channels have no intensities: no
        # histogram (Changed, R3-09: 'ids' used to show a histogram of the
        # ID values in channel 0's row).
        hists = []
        if self.mode in ('single', 'ids', 'idx_lists'):
            pass
        elif self.mode == 'cont_single':
            hists.append(compute_histogram(self.cont_single))
        elif self.mode == 'cont_multi':
            for c in range(self.C):
                hists.append(compute_histogram(self.cont_multi[:, c]))

        # pad to C
        while len(hists) < self.C:
            hists.append({'counts': [], 'bin_edges': []})
        self.histograms_data = hists

        # Initialize observers and render
        self._init_observers()
        self._apply_trait_kwargs(kwargs)

    # Widget index 0 is the data minimum of each axis (``_copy_params_lines``)
    def _slab_origin(self):
        return (self.zmin, self.ymin, self.xmin)

    def _label_colors(self):
        """Colour of each channel as drawn (used for the channel labels)."""
        return [matplotlib.colors.to_hex(self.colors_rgb[i % len(self.colors_rgb)]) for i in range(self.C)]

    def _composite_colors(self):
        """One colour or colormap name per channel, as the compositor receives them.

        With fewer entries than channels (e.g. integer ids coloured by one
        colormap name) the per-channel colours sampled in __init__ are used,
        as points mode does.
        """
        if len(self.colors_use) >= self.C:
            return list(self.colors_use[:self.C])
        return self._label_colors()

    def _panel_geometry(self, plane):
        rot_z, rot_y, rot_x = _parse_rotate_view(self.rotate_view)
        if self.render == 'density':
            # Histogram panels, displayed from 0 (= data minimum) in physical units
            bxs, bys, bzs = self._bin_phys
            if plane == 'xy':
                return _PanelRotation((self.BY, self.BX), rot_z, bys, bxs)
            if plane == 'zy':
                return _PanelRotation((self.BY, self.BZ), rot_x, bys, bzs)
            if plane == 'xz':
                return _PanelRotation((self.BZ, self.BX), rot_y, bzs, bxs)
            return None
        # Points, displayed at their physical coordinates and rotated about the
        # centre of the data box
        cx, cy, cz = self._box_centre_phys
        if plane == 'xy':
            return _PointRotation(rot_z, cx, cy)
        if plane == 'zy':
            return _PointRotation(rot_x, cz, cy)
        if plane == 'xz':
            return _PointRotation(rot_y, cx, cz)
        return None

    def _phys_to_index(self, plane, phys_x, phys_y):
        # Widget index i covers data coordinates [min + i - 0.5, min + i + 0.5).
        g = self._panel_geometry(plane)
        if g is None:
            return None
        u, v = g.inverse(phys_x, phys_y)
        if self.render == 'density':
            ox, oy, oz = 0.0, 0.0, 0.0
        else:
            bx0, by0, bz0 = self._box_lo
            ox, oy, oz = bx0 * self.sx, by0 * self.sy, bz0 * self.sz
        if plane == 'xy':
            return {'x': int(np.floor((u - ox) / self.sx)), 'y': int(np.floor((v - oy) / self.sy))}
        if plane == 'zy':
            return {'z': int(np.floor((u - oz) / self.sz)), 'y': int(np.floor((v - oy) / self.sy))}
        return {'x': int(np.floor((u - ox) / self.sx)), 'z': int(np.floor((v - oz) / self.sz))}

    def _render(self):
        # A cleared box ("") is the channel's automatic value (R3-04)
        vmin_resolved = [self.vmin_auto[i] if v == "" else float(v) for i, v in enumerate(self.vmin_list)]
        vmax_resolved = [self.vmax_auto[i] if v == "" else float(v) for i, v in enumerate(self.vmax_list)]
        gamma_resolved = [float(g) for g in self.gamma_list]
        opacity_resolved = [float(o) for o in self.opacity_list]

        # Translate widget relative coordinates (0..Dim) to data coordinates (min..max).
        # The slab [x0, x1] is inclusive in widget indices; index i covers data
        # [xmin + i - 0.5, xmin + i + 0.5), so points are selected with
        # x0 - 0.5 <= X < x1 + 0.5, the band the crosshairs draw.
        x0 = max(self.xmin, self.xmin + self.x_start)
        x1 = min(self.xmax, self.xmin + self.x_end)
        y0 = max(self.ymin, self.ymin + self.y_start)
        y1 = min(self.ymax, self.ymin + self.y_end)
        z0 = max(self.zmin, self.zmin + self.z_start)
        z1 = min(self.zmax, self.zmin + self.z_end)

        x_lims = (x0, x1)
        y_lims = (y0, y1)
        z_lims = (z0, z1)

        clipped = False
        if x0 > self.xmin + self.x_start or x1 < self.xmin + self.x_end: clipped = True
        if y0 > self.ymin + self.y_start or y1 < self.ymin + self.y_end: clipped = True
        if z0 > self.zmin + self.z_start or z1 < self.zmin + self.z_end: clipped = True

        if clipped:
            self.warning_msg = "⚠️ Projection clipped to data boundaries"
        else:
            self.warning_msg = ""

        def _slab_masks(Xi, Yi, Zi):
            mZ = (Zi >= z_lims[0] - 0.5) & (Zi < z_lims[1] + 0.5)
            mY = (Yi >= y_lims[0] - 0.5) & (Yi < y_lims[1] + 0.5)
            mX = (Xi >= x_lims[0] - 0.5) & (Xi < x_lims[1] + 0.5)
            return mZ, mY, mX

        rot_z, rot_y, rot_x = _parse_rotate_view(self.rotate_view)
        channel_labels = self.channel_names if self.channel_labels_input is not None else None
        sx, sy, sz = self.sx, self.sy, self.sz
        Wx, Wy, Wz = self._box_size
        widths = (Wx * sx, Wy * sy, Wz * sz)
        bx0, by0, bz0 = self._box_lo

        # Density Mode Logic
        if self.render == 'density':
            bx, by, bz = self._bin_data

            def _bin2d(cols, rows, col_lo, col_w, n_c, row_lo, row_w, n_r, w=None):
                # Counts per bin, or the per-bin maximum of w (NaN where empty),
                # so a bin's colour encodes the value as it does in points mode.
                ci = np.clip(np.floor((cols - col_lo) / col_w).astype(np.intp), 0, n_c - 1)
                ri = np.clip(np.floor((rows - row_lo) / row_w).astype(np.intp), 0, n_r - 1)
                flat = ri * n_c + ci
                if w is None:
                    out = np.bincount(flat, minlength=n_r * n_c).astype(float)
                else:
                    ok = np.isfinite(w)
                    out = np.full(n_r * n_c, -np.inf)
                    np.maximum.at(out, flat[ok], w[ok])
                    out[np.isneginf(out)] = np.nan
                return out.reshape(n_r, n_c)

            def _planes(Xi, Yi, Zi, w=None):
                mZ, mY, mX = _slab_masks(Xi, Yi, Zi)
                wz = None if w is None else w[mZ]
                wy = None if w is None else w[mY]
                wx = None if w is None else w[mX]
                Hxy = _bin2d(Xi[mZ], Yi[mZ], bx0, bx, self.BX, by0, by, self.BY, wz)
                Hxz = _bin2d(Xi[mY], Zi[mY], bx0, bx, self.BX, bz0, bz, self.BZ, wy)
                Hzy = _bin2d(Zi[mX], Yi[mX], bz0, bz, self.BZ, by0, by, self.BY, wx)
                return Hxy, Hxz, Hzy

            xy_list, xz_list, zy_list = [], [], []

            if self.mode in ('single', 'ids', 'idx_lists'):
                if self.mode == 'single':
                    idx_lists_local = [np.arange(self.N)]
                elif self.mode == 'ids':
                    idx_lists_local = [np.nonzero(self.ch_ids == c)[0] for c in range(self.C)]
                else:
                    idx_lists_local = self.idx_lists

                for idxs in idx_lists_local:
                    Hxy, Hxz, Hzy = _planes(self.X_arr[idxs], self.Y_arr[idxs], self.Z_arr[idxs])
                    xy_list.append(Hxy); xz_list.append(Hxz); zy_list.append(Hzy)

            elif self.mode == 'cont_single':
                Hxy, Hxz, Hzy = _planes(self.X_arr, self.Y_arr, self.Z_arr, self.cont_single)
                xy_list = [Hxy]; xz_list = [Hxz]; zy_list = [Hzy]

            else: # cont_multi
                for c in range(self.C):
                    Hxy, Hxz, Hzy = _planes(self.X_arr, self.Y_arr, self.Z_arr, self.cont_multi[:, c])
                    xy_list.append(Hxy); xz_list.append(Hxz); zy_list.append(Hzy)

            # One colour (or colormap) per channel, composited with the
            # viewers' blending rule (D04: it used to be additive here even
            # with colormap names, unlike the slice viewer).
            xy_rgb, xz_rgb, zy_rgb = _composite_planes(
                xy_list, xz_list, zy_list,
                vmin=vmin_resolved, vmax=vmax_resolved, gamma=gamma_resolved,
                colormap=self._composite_colors(), opacity=opacity_resolved, blending=self._blend,
                soft_clip=True,
            )

            # Each histogram bin is drawn with its physical size, so panels,
            # scale bar, crosshairs and hover all share the data's units.
            # Opacity is already baked into the composite.
            bxs, bys, bzs = self._bin_phys
            fig = show_zyx(
                xy_rgb, xz_rgb, zy_rgb,
                pixel_sizes={'Z': bzs, 'Y': bys, 'X': bxs, 'unit': self._length_unit},
                pixel_sizes_given=self._pixel_sizes_given,
                figsize=self.figsize, colormap=None,
                vmin=None, vmax=None, gamma=1, colors=None, opacity=None, subplot_bg=self.subplot_bg,
                rotate_view=self.rotate_view, channel_labels=channel_labels,
                channel_label_fontsize_pt=self.channel_label_fontsize_pt,
                channel_label_colors=self._label_colors(),
                blending=self._blend, scale_bar_show=self.scale_bar_show,
                scale_bar_fontsize_pt=_scale_bar_fontsize(self.scale_bar_fontsize_pt),
            )

            if self._blend != 'minimum':  # the white theme keeps its white canvas
                fig.patch.set_alpha(0.0) # transparent figure bg
                fig.patch.set_facecolor("none")
            # Subplot facecolor is handled by show_zyx if subplot_bg is passed

            if self.show_crosshair:
                _draw_slab_crosshairs(
                    {'xy': fig.axXY, 'zy': fig.axZY, 'xz': fig.axXZ},
                    {plane: self._panel_geometry(plane) for plane in ('xy', 'zy', 'xz')},
                    x_edges=((x_lims[0] - self.xmin) * sx, (x_lims[1] + 1 - self.xmin) * sx),
                    y_edges=((y_lims[0] - self.ymin) * sy, (y_lims[1] + 1 - self.ymin) * sy),
                    z_edges=((z_lims[0] - self.zmin) * sz, (z_lims[1] + 1 - self.zmin) * sz),
                    widths=widths,
                )

            return fig

        else:
            # Points mode: points sit at their physical coordinates and are
            # rotated about the centre of the data box, in the same sense as
            # the image panels.
            geoms = {plane: self._panel_geometry(plane) for plane in ('xy', 'zy', 'xz')}
            cx, cy, cz = self._box_centre_phys

            def _rot_bounds(w1, w2, angle):
                if angle == 0: return w1, w2
                rad = np.radians(angle)
                cos_a, sin_a = abs(np.cos(rad)), abs(np.sin(rad))
                return w1 * cos_a + w2 * sin_a, w1 * sin_a + w2 * cos_a

            # Physical extents of the rotated panels
            w_x_xy, w_y_xy = _rot_bounds(widths[0], widths[1], rot_z)
            w_x_xz, w_z_xz = _rot_bounds(widths[0], widths[2], rot_y)
            w_z_zy, w_y_zy = _rot_bounds(widths[2], widths[1], rot_x)

            col1_w = max(w_x_xy, w_x_xz)
            col2_w = w_z_zy
            row1_h = max(w_y_xy, w_y_zy)
            row2_h = w_z_xz

            figW, figH = self.figsize
            gap_in = _GAP_IN
            margin_in = _MARGIN_IN

            if self.channel_label_fontsize_pt is not None:
                channel_label_fontsize_pt = self.channel_label_fontsize_pt
            else:
                channel_label_fontsize_pt = _auto_fontsize_pt(figW)

            hl_in = _label_row_in(channel_label_fontsize_pt) if channel_labels is not None else 0.0
            num_v_gaps = 2 if channel_labels is not None else 1

            avail_w_in = figW - gap_in - 2.0 * margin_in
            avail_h_in = figH - num_v_gaps * gap_in - 2.0 * margin_in - hl_in

            scale = min(avail_w_in / max(col1_w + col2_w, 1e-6), avail_h_in / max(row1_h + row2_h, 1e-6))

            w1_in = col1_w * scale
            w2_in = col2_w * scale
            h1_in = row1_h * scale
            h2_in = row2_h * scale

            block_w_in = w1_in + gap_in + w2_in
            block_h_in = h1_in + gap_in + h2_in + (hl_in + gap_in if channel_labels is not None else 0.0)

            left_margin_in = (figW - block_w_in) / 2.0
            bottom_margin_in = (figH - block_h_in) / 2.0

            x0_frac = left_margin_in / figW
            w1_frac = w1_in / figW
            gap_w_frac = gap_in / figW
            w2_frac = w2_in / figW

            x_col1 = x0_frac
            x_col2 = x0_frac + w1_frac + gap_w_frac

            y0_frac = bottom_margin_in / figH
            h2_frac = h2_in / figH
            gap_h_frac = gap_in / figH
            h1_frac = h1_in / figH
            hl_frac = hl_in / figH

            y_row2 = y0_frac
            y_row1 = y0_frac + h2_frac + gap_h_frac

            # Create the figure at the size the layout was computed for
            fig_bg = 'white' if self._blend == 'minimum' else 'none'
            fig = _new_figure(figsize=(figW, figH), facecolor=fig_bg)

            axLabels = None
            if channel_labels is not None:
                y_labels = y_row1 + h1_frac + gap_h_frac
                axLabels = fig.add_axes([x_col1, y_labels, w1_frac, hl_frac])
                axXY = fig.add_axes([x_col1, y_row1, w1_frac, h1_frac])
                axZY = fig.add_axes([x_col2, y_row1, w2_frac, h1_frac])
                axXZ = fig.add_axes([x_col1, y_row2, w1_frac, h2_frac])
                axBar = fig.add_axes([x_col2, y_row2, w2_frac, h2_frac])

                axLabels.patch.set_visible(False)
                axLabels.set_xticks([])
                axLabels.set_yticks([])
                for spine in axLabels.spines.values():
                    spine.set_visible(False)

                _draw_channel_label_bar(axLabels, channel_label_fontsize_pt, x_ref=axXY)

                n_labels = len(channel_labels)
                if n_labels > 0:
                    color_list = self._label_colors()
                    if len(color_list) < n_labels:
                        color_list = color_list * (n_labels // len(color_list) + 1)

                    from matplotlib.offsetbox import TextArea, HPacker, AnchoredOffsetbox

                    text_areas = []
                    for i, label in enumerate(channel_labels):
                        GUI_CHANNEL_COLOR = get_gui_channel_color(color_list[i])
                        ta = TextArea(label, textprops=dict(color=GUI_CHANNEL_COLOR, fontsize=channel_label_fontsize_pt, fontweight='bold'))
                        text_areas.append(ta)

                    packer = HPacker(children=text_areas, align="center", pad=0, sep=10)
                    anchored_box = AnchoredOffsetbox(
                        loc='lower center', child=packer, pad=0.0, frameon=False, borderpad=0.0,
                        bbox_to_anchor=(0.5, 0.2), bbox_transform=axLabels.transAxes
                    )
                    axLabels.add_artist(anchored_box)
            else:
                axXY = fig.add_axes([x_col1, y_row1, w1_frac, h1_frac])
                axZY = fig.add_axes([x_col2, y_row1, w2_frac, h1_frac])
                axXZ = fig.add_axes([x_col1, y_row2, w1_frac, h2_frac])
                axBar = fig.add_axes([x_col2, y_row2, w2_frac, h2_frac])

            fig.axXY = axXY
            fig.axZY = axZY
            fig.axXZ = axXZ
            fig.axBar = axBar
            fig.axLabels = axLabels

            for ax in (axXY, axZY, axXZ, axBar):
                if ax is not axBar and self.subplot_bg is not None:
                    ax.set_facecolor(self.subplot_bg)
                else:
                    ax.patch.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            def _scatter_planes(Xi, Yi, Zi, opacity=1.0, **kw):
                # Per-channel opacity multiplies the marker alpha; 0 hides
                # the channel, as in the image viewers (R3-09 / R2-23).
                alpha = self.alpha * float(opacity)
                if alpha <= 0:
                    return
                mZ, mY, mX = _slab_masks(Xi, Yi, Zi)

                def _sub(m):
                    return {k: (v[m] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == len(Xi) else v) for k, v in kw.items()}

                if mZ.any():
                    u, v = geoms['xy'].forward(Xi[mZ] * sx, Yi[mZ] * sy)
                    axXY.scatter(u, v, s=self.point_size, alpha=alpha, linewidths=0, **_sub(mZ))
                if mY.any():
                    u, v = geoms['xz'].forward(Xi[mY] * sx, Zi[mY] * sz)
                    axXZ.scatter(u, v, s=self.point_size, alpha=alpha, linewidths=0, **_sub(mY))
                if mX.any():
                    u, v = geoms['zy'].forward(Zi[mX] * sz, Yi[mX] * sy)
                    axZY.scatter(u, v, s=self.point_size, alpha=alpha, linewidths=0, **_sub(mX))

            if self.mode == 'cont_single':
                vals = self.cont_single
                c_use = self.colors_use[0]
                if is_colormap(c_use):
                    cmap = matplotlib.colormaps[c_use]
                elif self._blend == 'minimum':
                    cmap = white_to(resolve_color(c_use))
                else:
                    cmap = black_to(resolve_color(c_use))

                vmin_c, vmax_c = vmin_resolved[0], vmax_resolved[0]
                if vmin_c > vmax_c:
                    # Normalize raises for vmin > vmax (same rule as the slice viewer)
                    vmax_c = vmin_c + 1.0
                    self._warn_inverted_ranges([0])
                # gamma after the linear window, as in the image viewers
                # (Changed, R2-23: points mode ignored gamma)
                g = gamma_resolved[0]
                norm = (matplotlib.colors.Normalize(vmin=vmin_c, vmax=vmax_c) if g == 1.0
                        else PowerNorm(gamma=g, vmin=vmin_c, vmax=vmax_c, clip=True))
                _scatter_planes(self.X_arr, self.Y_arr, self.Z_arr, opacity=opacity_resolved[0],
                                c=vals, cmap=cmap, norm=norm)

            elif self.mode == 'cont_multi':
                if any(o > 0 for o in opacity_resolved):
                    cols = blend_colors(self.cont_multi, self.colors_use, vmin=vmin_resolved, vmax=vmax_resolved,
                                        gamma=gamma_resolved, soft_clip=True, blending=self._blend,
                                        opacity=opacity_resolved)
                    _scatter_planes(self.X_arr, self.Y_arr, self.Z_arr, c=cols)

            else:
                if self.mode == 'single':
                    idx_lists_local = [np.arange(self.N)]
                elif self.mode == 'ids':
                    idx_lists_local = [np.nonzero(self.ch_ids == c)[0] for c in range(self.C)]
                else:
                    idx_lists_local = self.idx_lists

                for c, idxs in enumerate(idx_lists_local):
                    if idxs.size == 0: continue
                    col = [self.colors_rgb[c % len(self.colors_rgb)]]
                    _scatter_planes(self.X_arr[idxs], self.Y_arr[idxs], self.Z_arr[idxs],
                                    opacity=opacity_resolved[c] if c < len(opacity_resolved) else 1.0, c=col)

            # Axis limits: the rotated data box, centred on the rotation centre
            axXY.set_xlim([cx - w_x_xy / 2, cx + w_x_xy / 2])
            axXY.set_ylim([cy + w_y_xy / 2, cy - w_y_xy / 2])

            axXZ.set_xlim([cx - w_x_xz / 2, cx + w_x_xz / 2])
            axXZ.set_ylim([cz + w_z_xz / 2, cz - w_z_xz / 2])

            axZY.set_xlim([cz - w_z_zy / 2, cz + w_z_zy / 2])
            axZY.set_ylim([cy + w_y_zy / 2, cy - w_y_zy / 2])

            if self.show_crosshair:
                _draw_slab_crosshairs(
                    {'xy': axXY, 'zy': axZY, 'xz': axXZ}, geoms,
                    x_edges=((x_lims[0] - 0.5) * sx, (x_lims[1] + 0.5) * sx),
                    y_edges=((y_lims[0] - 0.5) * sy, (y_lims[1] + 0.5) * sy),
                    z_edges=((z_lims[0] - 0.5) * sz, (z_lims[1] + 0.5) * sz),
                    widths=widths,
                    origin=(bx0 * sx, by0 * sy, bz0 * sz),
                )

            # Scale bar (kept opaque), measured against the drawn XY width
            fig.patch.set_alpha(1.0)
            if self.scale_bar_show:
                _add_scale_bar(axXY, w_x_xy, self._pixel_sizes_given, (figW, figH), unit=self._length_unit,
                               color='black' if self._blend == 'minimum' else 'white',
                               fontsize_pt=_scale_bar_fontsize(self.scale_bar_fontsize_pt))
            return fig


def _channel_colors_from_coord(values, stacklevel=3):
    """Channel colours from a ``color`` coordinate (names, hex or RGB tuples), or None.

    Entries that are neither a colour nor a colormap name (e.g. '' or NaN for
    channels without a colour) take the default palette colour, with a
    warning when the entry is not empty. Changed (D03, 2026-11): RGB tuples
    are used (they were silently ignored, as was the whole coordinate when
    one entry was unusable).
    """
    specs = []
    for v in list(values):
        try:
            specs.append(channel_spec(v))
        except (TypeError, ValueError):
            if not (v is None or (isinstance(v, str) and not v) or (isinstance(v, float) and np.isnan(v))):
                warnings.warn(f"channel colour {v!r} is not a colour or colormap name; using the default.",
                              UserWarning, stacklevel=stacklevel)
            specs.append(None)
    if all(s is None for s in specs):
        return None
    defaults = [c for c in PALETTE if c not in specs] or PALETTE
    fill = iter(defaults * (len(specs) // len(defaults) + 1))
    return [s if s is not None else next(fill) for s in specs]


def _viewer_input_from_xarray(im, colormap, channel_labels, pixel_sizes):
    """Unpack an xarray.DataArray (e.g. STCZYX from ``image_io``) for the ortho viewers.

    Dims other than C, Z, Y, X must have size 1 (select first, e.g.
    ``da.isel(S=0, T=0)``). Channel labels default to the ``C`` coordinate,
    colours to a ``color`` coordinate along ``C`` (``image_io.set_channel_info``),
    and pixel sizes to the calibrated coordinates (``get_stczyx_pixel_sizes``).
    Explicit arguments always win. Non-xarray inputs are returned unchanged.
    """
    if not (hasattr(im, "dims") and hasattr(im, "isel") and hasattr(im, "coords")):
        return im, colormap, channel_labels, pixel_sizes
    da = im
    for d in list(da.dims):
        if d not in ("C", "Z", "Y", "X"):
            if da.sizes[d] != 1:
                raise ValueError(
                    f"dimension {d!r} has size {da.sizes[d]}; select one index first, e.g. da.isel({d}=0)"
                )
            da = da.isel({d: 0})
    if not {"Y", "X"} <= set(da.dims):
        raise ValueError(f"DataArray needs Y and X dimensions, got {tuple(da.dims)}")
    if pixel_sizes is None:
        from .image_io import get_stczyx_pixel_sizes
        ps = get_stczyx_pixel_sizes(da)
        if any(ps.get(a) is not None for a in ("Z", "Y", "X")):
            pixel_sizes = {a: ps[a] for a in ("Z", "Y", "X") if ps.get(a) is not None}
            if ps.get("unit"):
                pixel_sizes["unit"] = ps["unit"]
    spatial = [d for d in ("Z", "Y", "X") if d in da.dims]
    if "C" not in da.dims:
        return np.asarray(da.transpose(*spatial).values), colormap, channel_labels, pixel_sizes
    da = da.transpose("C", *spatial)
    if channel_labels is None and "C" in da.coords:
        channel_labels = [str(c) for c in da.coords["C"].values]
    if colormap is None and "color" in da.coords and da.coords["color"].dims == ("C",):
        colormap = _channel_colors_from_coord(da.coords["color"].values, stacklevel=4)
    channels = [np.asarray(da.isel(C=i).values) for i in range(da.sizes["C"])]
    return channels, colormap, channel_labels, pixel_sizes


def _legacy_colors(colors, colormap, stacklevel=4):
    """The deprecated ``colors=`` (FutureWarning, D08), used when ``colormap`` is None."""
    if colors is None:
        return colormap
    warn_deprecated("The 'colors' parameter", "'colormap'", stacklevel=stacklevel)
    return colormap if colormap is not None else colors


def _ortho_input(im, colormap=None, channel_labels=None, pixel_sizes=None, channel_axis=None):
    """
    One input normaliser for every ortho viewer (D12).

    Returns ``(image, colormap, channel_labels, pixel_sizes)`` where image is
    a 3D (Z, Y, X) array or a list of them (channels):

    - an ``xarray.DataArray`` goes through `_viewer_input_from_xarray`
      (channel labels, colours and pixel sizes from its coordinates;
      explicit arguments win);
    - a tuple of channels becomes a list;
    - an ndarray with ``channel_axis`` is split along that axis; a 4D
      ndarray without it raises ValueError (CZYX or TZYX cannot be told
      apart, so the channel axis is never guessed);
    - 2D (YX) images get a Z axis of length 1.
    """
    if hasattr(im, "dims") and hasattr(im, "isel") and hasattr(im, "coords"):
        im, colormap, channel_labels, pixel_sizes = _viewer_input_from_xarray(
            im, colormap, channel_labels, pixel_sizes)
    elif isinstance(im, tuple):
        im = list(im)
    if not isinstance(im, list):
        im = np.asarray(im)
        if channel_axis is not None:
            im = [np.asarray(c) for c in np.moveaxis(im, channel_axis, 0)]
        elif im.ndim > 3:
            raise ValueError(
                f"got a {im.ndim}D array {im.shape}: pass channel_axis= to split channels "
                "(e.g. channel_axis=0 for CZYX), or a list of 3D arrays.")
    if isinstance(im, list):
        im = [np.asarray(c) for c in im]
        if any(c.ndim not in (2, 3) for c in im):
            raise ValueError(f"each channel must be a 2D or 3D array, got shapes {[c.shape for c in im]}")
        im = [c[np.newaxis] if c.ndim == 2 else c for c in im]
    elif im.ndim == 2:
        im = im[np.newaxis]
    elif im.ndim != 3:
        raise ValueError(f"expected a 2D or 3D image, got shape {im.shape}")
    return im, colormap, channel_labels, pixel_sizes


def _legacy_viewer_kwargs(colors=None, colormap=None, sxy=None, sz=None, pixel_sizes=None,
                         x_s=None, y_s=None, z_s=None, slabs_position=None,
                         x_t=None, y_t=None, z_t=None, slabs_thickness=None, stacklevel=5):
    """
    Resolve the deprecated viewer names (D08, FutureWarning): ``colors`` ->
    ``colormap``, ``sxy``/``sz`` -> ``pixel_sizes``, ``x_s..z_s`` ->
    ``slabs_position``, ``x_t..z_t`` -> ``slabs_thickness``. The new names
    win when both are given. ``stacklevel`` is counted from
    ``warn_deprecated`` up to the user's call.
    """
    colormap = _legacy_colors(colors, colormap, stacklevel=stacklevel)
    if sxy is not None or sz is not None:
        warn_deprecated("Passing 'sxy'/'sz'", "'pixel_sizes'", stacklevel=stacklevel - 1)
        if pixel_sizes is None:
            pixel_sizes = (1.0 if sz is None else sz, 1.0 if sxy is None else sxy, 1.0 if sxy is None else sxy)
    if any(v is not None for v in (x_s, y_s, z_s)):
        warn_deprecated("Passing 'x_s'/'y_s'/'z_s'", "'slabs_position'", stacklevel=stacklevel - 1)
        if slabs_position is None:
            slabs_position = (z_s, y_s, x_s)
    if any(v is not None for v in (x_t, y_t, z_t)):
        warn_deprecated("Passing 'x_t'/'y_t'/'z_t'", "'slabs_thickness'", stacklevel=stacklevel - 1)
        if slabs_thickness is None:
            slabs_thickness = (z_t, y_t, x_t)
    return colormap, pixel_sizes, slabs_position, slabs_thickness


def _prepare_viewer_args(im, pixel_sizes=None, sxy=None, sz=None, colormap=None, colors=None,
                         channel_labels=None, channel_axis=None, figsize=None, figsize_scale=1,
                         rotate_view=None, channel_label_fontsize_pt=None,
                         slabs_position=None, x_s=None, y_s=None, z_s=None,
                         slabs_thickness=None, x_t=None, y_t=None, z_t=None, _stacklevel=6):
    """
    The argument preparation shared by the image-viewer factories (D12).

    Resolves the deprecated names, normalises the input (`_ortho_input`:
    DataArray, tuple, ``channel_axis``, 2D), and computes the automatic
    figure size (physical extents in an 8-inch box, the rule the classes
    use) times ``figsize_scale``. Returns a dict of keyword arguments for
    ``TNIASliceWidget`` / ``TNIAAnnotatorWidget`` (slab defaults are the
    classes'): im, pixel_sizes, colormap, channel_labels, figsize,
    slabs_position, slabs_thickness.
    """
    colormap, pixel_sizes, slabs_position, slabs_thickness = _legacy_viewer_kwargs(
        colors, colormap, sxy, sz, pixel_sizes, x_s, y_s, z_s, slabs_position,
        x_t, y_t, z_t, slabs_thickness, stacklevel=_stacklevel)
    # after resolving the deprecated names, so they still apply to DataArray input
    im, colormap, channel_labels, pixel_sizes = _ortho_input(im, colormap, channel_labels, pixel_sizes, channel_axis)
    if figsize is None:
        # The widget parses pixel_sizes again and warns about missing axes once
        (pz, py, px), _, _ = _viewer_pixel_sizes(pixel_sizes, warn=False)
        Z, Y, X = np.shape(im[0] if isinstance(im, list) else im)
        cols, rows = _ortho_extents(X * px, Y * py, Z * pz, rotate_view)
        figsize = _auto_figsize(cols, rows, True, box=(8.0, 8.0), label_fontsize_pt=channel_label_fontsize_pt)
    if figsize_scale != 1:
        figsize = (figsize[0] * figsize_scale, figsize[1] * figsize_scale)
    return dict(im=im, pixel_sizes=pixel_sizes, colormap=colormap, channel_labels=channel_labels,
                figsize=figsize, slabs_position=slabs_position, slabs_thickness=slabs_thickness)


def _viewer_doc(func):
    """Append the parameters shared by the interactive viewers to a factory docstring."""
    func.__doc__ = _join_docs(func.__doc__, _VIEWER_DOC)
    return func


_VIEWER_DOC = """
Parameters shared by the interactive viewers
--------------------------------------------
pixel_sizes : dict or (Z, Y, X) sequence, optional
    Voxel size per axis. A dict may carry a 'unit' (known length units are
    converted to µm); without one the numbers are taken as µm. Missing X/Y
    sizes draw the panels in pixels with a 'px' scale bar; a missing Z with
    known X/Y takes the finest of them (with a warning).
figsize : (width, height) in inches, optional
    None (default): fit the physical extents of the (rotated) panels and the
    channel-label row in an 8-inch box. ``figsize_scale`` multiplies it.
colormap : colour, colormap name, registered Colormap, or a list of them
    One per channel; None takes the package palette ('white', 'lime',
    'magenta', ...). ``colors`` is the deprecated name (FutureWarning).
vmin, vmax : float or per-channel list, optional (data units)
    Display range. In the image viewers None (per channel) is the default
    display range of the whole channel volume (0, or the data minimum if
    negative; robust 99.99th percentile), which a box cleared in the GUI
    restores; for the scatter viewer see its own entry.
gamma, opacity : float or per-channel list
    Gamma after the linear window (1 = linear); opacity 0..1 (0 hides).
slabs_position : (Z, Y, X) or dict, optional (physical units, µm)
    Centre of the slab of each axis; None centres it. Half-voxel values are
    allowed (copy-kwargs emits them for even widths).
slabs_thickness : (Z, Y, X) or dict, optional (physical units, µm)
    Half-thickness of each slab: the slab spans centre +/- half-thickness
    (inclusive voxel range). None is 1/64 of the axis (at least one voxel);
    a value beyond the volume shows the whole axis. ``x_s, y_s, z_s`` /
    ``x_t, y_t, z_t`` are deprecated per-axis aliases (FutureWarning).
sxy, sz : float, deprecated
    Use ``pixel_sizes=(sz, sxy, sxy)`` (FutureWarning).
rotate_view : float or (Z, Y, X) degrees, optional
    Rotates the XY panel (a float) or the XY, XZ and ZY panels
    counter-clockwise for display only; slabs stay along the array axes.
channel_labels : list of str, optional
    Names drawn above the panels and in the GUI rows.
channel_label_fontsize_pt : float, optional
    Label font size in points; None is 3% of the figure width (8..24 pt).
blending : str, optional
    'additive', 'maximum', 'screen' or 'minimum' (white canvas); None picks
    additive for colours, maximum with a colormap name, minimum when every
    colormap starts at white. Keyword only; not in the GUI.
show_crosshair : bool
    Draw the slab boundaries on the panels.
sync_on_hover : bool
    Enables hover sync. It is not continuous: pressing 'C' while the
    pointer is over a panel moves the other slabs to the voxel under it
    (keeping their widths). The key is ignored with Ctrl / Cmd / Alt held
    and while typing in a text box (Changed, D17 / R3-02: it used to fire on
    Ctrl+C and while typing anywhere on the page).
display_dpi : float
    Resolution of the on-screen image (default 100 dots per inch), whatever
    rcParams['figure.dpi'] says; the image shrinks to the available width
    and the controls wrap below it. Exports use 300 dpi. New (D14, 2026-11):
    the image used to follow rcParams['figure.dpi'].
scale_bar_show : bool
    Draw the scale bar (also a GUI checkbox).
scale_bar_fontsize_pt : float, optional
    Scale-bar label size in points (None = auto, as the channel labels; 0
    hides the label but keeps the bar). Also a GUI box (empty = auto).

Exports: the save button writes the figure to the file name in the box,
in the format of its extension (.svg with editable text, .pdf, .png; none
adds .svg), at 300 dpi; an existing file is never overwritten (a numbered
suffix is added: ``name_1.svg``) and the outcome (absolute path or error)
is shown in the widget's status line (``status_msg``). The PNG button
copies the figure to the clipboard; "kwargs" copies the current view as
keyword arguments (physical slab centre / half-thickness, display ranges)
that rebuild the same view. Changed (D15, 2026-11): saving always wrote
SVG, overwrote the file and reported with print().

GUI: dark theme; range boxes show physical positions with their unit
(index x pixel size, plus the data minimum for the scatter viewer;
indices in 'px' when uncalibrated); a cleared vmin / vmax box shows and
uses the automatic value (``vmin_auto`` / ``vmax_auto``). Sliders take
Arrow / PageUp / PageDown keys once focused.

Changed (D12, 2026-11): factories and classes share one argument
preparation (`_prepare_viewer_args`) and one automatic figsize rule; the
classes raise TypeError for unknown keyword arguments.
"""


@_viewer_doc
def show_zyx_max_slice_interactive(
    im,
    pixel_sizes=None, sxy=None, sz=None,
    figsize=None, colormap=None,
    vmin=None, vmax=None,
    gamma=1, figsize_scale=1,
    show_crosshair=True, sync_on_hover=True, scale_bar_show=True, scale_bar_fontsize_pt=None,
    colors=None, opacity=None,
    slabs_position=None, x_s=None, y_s=None, z_s=None,
    slabs_thickness=None, x_t=None, y_t=None, z_t=None,
    rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None,
    blending=None, projection=None, attenuation=0.05, bit_depth=None,
    channel_axis=None, display_dpi=100,
):
    """
    Interactive ortho-slab viewer of a 3D image (anywidget): XY, XZ and ZY
    projections of movable slabs, per-channel display controls, histograms.

    Returns a `TNIASliceWidget`.

    Parameters
    ----------
    im : ndarray, list/tuple of ndarrays, or xarray.DataArray
        ZYX (or YX) array, a list of them (one per channel), an ndarray with
        a channel axis given by ``channel_axis`` (e.g. 0 for CZYX; a 4D array
        without it raises), or a DataArray with C/Z/Y/X dims (other dims of
        size 1, e.g. STCZYX from ``image_io``). For a DataArray,
        ``channel_labels`` default to the ``C`` coordinate, ``colormap`` to a
        ``color`` coordinate along ``C`` and ``pixel_sizes`` to the calibrated
        coordinates; explicit arguments win.
    channel_axis : int, optional
        Axis of ``im`` holding the channels (ndarray input).
    projection : str, optional
        Slab projection: 'mip' (default), 'minip', 'average' or
        'attenuated_mip' (napari's names). Keyword only; not in the GUI.
    attenuation : float, optional
        Attenuation per voxel along the ray for 'attenuated_mip' (default 0.05).
    bit_depth : None, 'auto', int, 'dtype', 'data' or list, optional
        Range of the channel histograms: by default the detector bit depth
        inferred from the data (e.g. 0..4095 for 12-bit data stored as
        uint16); an int enforces one, 'dtype' the dtype range, 'data' the
        data range; a list sets it per channel.
    figsize_scale : float
        Multiplies ``figsize`` (automatic or given).

    Histograms: 128 bins per channel (NaN excluded), log counts, drawn under
    each channel's controls with the current vmin / vmax / gamma curve.
    """
    a = _prepare_viewer_args(
        im, pixel_sizes, sxy, sz, colormap, colors, channel_labels, channel_axis, figsize, figsize_scale,
        rotate_view, channel_label_fontsize_pt, slabs_position, x_s, y_s, z_s, slabs_thickness, x_t, y_t, z_t)
    return TNIASliceWidget(
        a['im'], pixel_sizes=a['pixel_sizes'], figsize=a['figsize'], colormap=a['colormap'],
        vmin=vmin, vmax=vmax, gamma=gamma, opacity=opacity, show_crosshair=show_crosshair,
        sync_on_hover=sync_on_hover, scale_bar_show=scale_bar_show, scale_bar_fontsize_pt=scale_bar_fontsize_pt,
        slabs_position=a['slabs_position'], slabs_thickness=a['slabs_thickness'], rotate_view=rotate_view,
        channel_labels=a['channel_labels'], channel_label_fontsize_pt=channel_label_fontsize_pt,
        blending=blending, projection=projection, attenuation=attenuation, bit_depth=bit_depth,
        display_dpi=display_dpi,
    )


@_viewer_doc
def show_zyx_max_slice_interactive_point_annotator(
    im,
    pixel_sizes=None, sxy=None, sz=None,
    figsize=None, colormap=None,
    vmin=None, vmax=None,
    gamma=1, figsize_scale=1,
    show_crosshair=True, sync_on_hover=True, scale_bar_show=True, scale_bar_fontsize_pt=None,
    colors=None, opacity=None,
    point_size_scale=0.01,
    annotate=False,
    snap_depth=True,
    slabs_position=None, x_s=None, y_s=None, z_s=None,
    slabs_thickness=None, x_t=None, y_t=None, z_t=None,
    rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None,
    blending=None, projection=None, attenuation=0.05, bit_depth=None,
    channel_axis=None, display_dpi=100,
):
    """
    Interactive ortho-slab viewer with point annotation (anywidget).

    The same viewer as `show_zyx_max_slice_interactive` (same ``im``,
    ``channel_axis``, ``projection``, ``attenuation`` and ``bit_depth``),
    plus an annotation toggle: a left click on a panel adds a point, or in
    delete mode removes the nearest point drawn within about 12 screen
    pixels among the points inside the panel's slab. Returns a
    `TNIAAnnotatorWidget`; ``w.points`` is the list of points as voxel
    indices ``[z, y, x]`` (not physical coordinates), and the CSV button
    writes them as ``z,y,x`` rows.

    Parameters
    ----------
    point_size_scale : float
        Marker size as a fraction of min(X, Y) in voxels (at least 3).
    annotate : bool
        Start with annotation mode on.
    snap_depth : bool
        A click on a projection stores the depth of the voxel the projection
        shows there (argmax of the visible channels inside the slab, argmin
        for 'minip'; the slab centre for flat profiles or 'average'). False
        stores the slab centre.
    """
    a = _prepare_viewer_args(
        im, pixel_sizes, sxy, sz, colormap, colors, channel_labels, channel_axis, figsize, figsize_scale,
        rotate_view, channel_label_fontsize_pt, slabs_position, x_s, y_s, z_s, slabs_thickness, x_t, y_t, z_t)
    return TNIAAnnotatorWidget(
        a['im'], pixel_sizes=a['pixel_sizes'], figsize=a['figsize'], colormap=a['colormap'],
        vmin=vmin, vmax=vmax, gamma=gamma, opacity=opacity, show_crosshair=show_crosshair,
        sync_on_hover=sync_on_hover, scale_bar_show=scale_bar_show, scale_bar_fontsize_pt=scale_bar_fontsize_pt,
        point_size_scale=point_size_scale, annotate=annotate, snap_depth=snap_depth,
        slabs_position=a['slabs_position'], slabs_thickness=a['slabs_thickness'], rotate_view=rotate_view,
        channel_labels=a['channel_labels'], channel_label_fontsize_pt=channel_label_fontsize_pt,
        blending=blending, projection=projection, attenuation=attenuation, bit_depth=bit_depth,
        display_dpi=display_dpi,
    )


@_viewer_doc
def show_zyx_max_scatter_interactive(
    points,
    channels=None,
    pixel_sizes=None, sxy=None, sz=None,
    render=None,
    bins=512,
    point_size=4, alpha=0.6,
    colormap=None, colors=None, opacity=None,
    gamma=1, vmin=None, vmax=None,
    figsize=None, figsize_scale=1.0,
    show_crosshair=True, sync_on_hover=True, scale_bar_show=True, scale_bar_fontsize_pt=None,
    subplot_bg=None,
    slabs_position=None, x_s=None, y_s=None, z_s=None,
    slabs_thickness=None, x_t=None, y_t=None, z_t=None,
    rotate_view=None, channel_labels=None, channel_label_fontsize_pt=None,
    blending=None, channel_mode='auto', display_dpi=100,
):
    """
    Interactive ortho-slab viewer of a 3D point cloud (anywidget): the
    points inside each slab, projected onto the XY, XZ and ZY panels.

    Returns a `TNIAScatterWidget`.

    Parameters
    ----------
    points : (N, 3) or (3, N) array, or a tuple/list of 3 arrays
        Coordinates in (Z, Y, X) order, in data units: one unit is one voxel,
        so physical position = coordinate x pixel size. Points with a
        non-finite coordinate are dropped (with a warning).
    channels : array or list, optional
        How points are coloured (see ``channel_mode``): one value per point
        (a continuous channel), an (N, C) array or a list of C per-point
        arrays (C continuous channels), one label per point (categories:
        one channel per distinct label, named after it; -1 and sparse labels
        work), or a list of index arrays (index lists: channel c is the
        points ``channels[c]``).
    channel_mode : 'auto', 'values', 'categories' or 'index_lists'
        How ``channels`` is read. 'auto' (default) infers it and raises
        ValueError when the input is ambiguous: integer arrays of one entry
        per point that are also valid index lists, or one integer array with
        more than 64 distinct values (categories or intensities?). Lists of
        other lengths are index lists.

    Changed (D13, 2026-11): integer channels were guessed silently: a list
    of integer per-point values became index lists, an integer array
    became max+1 category channels (label -1 was never drawn); categories
    now go through ``np.unique`` and ambiguous input raises.
    render : 'points', 'density' or None
        Scatter markers, or 2D histograms of the slab (``bins``); None picks
        'points' below 10,000 points.
    bins : int or (int, int)
        Density bins along X and Y (Z gets bins of about the same physical size).
    point_size : float
        Marker area in points^2 (matplotlib ``s``).
    alpha : float
        Marker transparency in points mode.
    subplot_bg : colour, optional
        Panel background; black, or white with blending='minimum'.
    vmin, vmax : float or per-channel list (values of the channels)
        For continuous channels; vmin defaults to 0 and vmax to the largest
        finite value.

    Changed (D10, 2026-11): `slabs_position` / `slabs_thickness` are physical
    (data coordinate x pixel size), centre and half-thickness, as in the image
    viewers; they used to be data units, so the same kwargs selected
    different slabs in the two viewers. Copy-kwargs emits physical values.

    Changed (D04, 2026-11): per-channel `vmin`, `vmax`, `gamma` and `opacity`
    follow the static viewers' rule (`_conventions.per_channel`): a list
    longer than the number of channels raises ValueError (extra entries used
    to be dropped silently); None entries take the default.
    """
    colormap, pixel_sizes, slabs_position, slabs_thickness = _legacy_viewer_kwargs(
        colors, colormap, sxy, sz, pixel_sizes, x_s, y_s, z_s, slabs_position,
        x_t, y_t, z_t, slabs_thickness, stacklevel=5)
    # The widget parses pixel_sizes again and warns about missing axes once
    (pz, py, px), _, _ = _viewer_pixel_sizes(pixel_sizes, warn=False)

    if isinstance(points, (tuple, list)) and len(points) == 3:
        # points is a tuple or list of 3 arrays: (Z, Y, X)
        Z, Y, X = points[0], points[1], points[2]
        X = np.asarray(X)
        Y = np.asarray(Y)
        Z = np.asarray(Z)
    else:
        points = np.asarray(points)
        if points.ndim == 2 and points.shape[1] == 3:
            # points shape is (N, 3), assume Z, Y, X
            Z, Y, X = points[:, 0], points[:, 1], points[:, 2]
        elif points.ndim == 2 and points.shape[0] == 3:
            # points shape is (3, N), assume Z, Y, X
            Z, Y, X = points[0], points[1], points[2]
        elif points.ndim == 1 and len(points) == 3:
            # array of 3 list-like objects
            Z, Y, X = points[0], points[1], points[2]
            X = np.asarray(X)
            Y = np.asarray(Y)
            Z = np.asarray(Z)
        else:
            raise ValueError("points must be an array of shape (N, 3) representing (Z, Y, X) or a tuple/list of 3 arrays (Z, Y, X).")

    # Points with a non-finite coordinate are dropped (with a warning) by the widget
    Xf, Yf, Zf, keep = _finite_mask(X, Y, Z, warn=False)
    if keep is not None:
        Xf, Yf, Zf = Xf[keep], Yf[keep], Zf[keep]

    if render is None:
        render = 'points' if len(Xf) < 10000 else 'density'

    if figsize is None and figsize_scale != 1.0:
        # The class's automatic size, from the physical extents of the data box
        figsize = _scatter_auto_figsize(_int_bounds(Xf), _int_bounds(Yf), _int_bounds(Zf), (pz, py, px),
                                        rotate_view, channel_labels is not None, channel_label_fontsize_pt)
    if figsize is not None and figsize_scale != 1.0:
        figsize = (figsize[0] * figsize_scale, figsize[1] * figsize_scale)

    w = TNIAScatterWidget(
        X, Y, Z,
        channels=channels, pixel_sizes=pixel_sizes, render=render, bins=bins,
        point_size=point_size, alpha=alpha, colormap=colormap, opacity=opacity,
        gamma=gamma, vmin=vmin, vmax=vmax, figsize=figsize, show_crosshair=show_crosshair, sync_on_hover=sync_on_hover, scale_bar_show=scale_bar_show, scale_bar_fontsize_pt=scale_bar_fontsize_pt,
        subplot_bg=subplot_bg,
        slabs_position=slabs_position, slabs_thickness=slabs_thickness,
        rotate_view=rotate_view, channel_labels=channel_labels, channel_label_fontsize_pt=channel_label_fontsize_pt,
        blending=blending, channel_mode=channel_mode, display_dpi=display_dpi,
    )
    w.X_arr_phys = w.X_arr * px
    w.Y_arr_phys = w.Y_arr * py
    w.Z_arr_phys = w.Z_arr * pz
    w.x_min_phys = w.xmin * px
    w.x_max_phys = w.xmax * px
    w.y_min_phys = w.ymin * py
    w.y_max_phys = w.ymax * py
    w.z_min_phys = w.zmin * pz
    w.z_max_phys = w.zmax * pz
    return w

class IsoScatterWidget(anywidget.AnyWidget):
    _esm = _ASSET_DIR.joinpath("iso_scatter.js").read_text()
    # The dark viewer theme shared with the ortho viewers (D14)
    _css = _ASSET_DIR.joinpath("tnia_plotting_anywidgets.css").read_text()

    image_data = traitlets.Unicode().tag(sync=True)
    elev = traitlets.Float(30).tag(sync=True)
    azim = traitlets.Float(-60).tag(sync=True)
    save_filename = traitlets.Unicode("iso_scatter.svg").tag(sync=True)
    save_trigger = traitlets.Int(0).tag(sync=True)
    # Outcome of the last export, shown in the widget (D15)
    status_msg = traitlets.Unicode("").tag(sync=True)

    def __init__(self, X, Y, Z, color=None, sxy=None, sz=None, figsize=(10, 8),
                 point_size=5, alpha=0.6, cmap='viridis', max_points=10000,
                 title=None, pixel_sizes=None, channel_labels=None, seed=0):
        """
        Isometric scatter widget of X, Y, Z coordinates (see `show_iso_scatter`).

        Changed (D16, 2026-11): integer and bool ``color`` arrays are
        categories (they were a continuous gradient); ``sxy``/``sz`` are
        deprecated (FutureWarning; use ``pixel_sizes``) and now also apply to
        empty input; ``figsize`` defaults to (10, 8) like ``show_iso_scatter``
        (it was (12, 10)). Integer colours with more than 64 distinct values
        stay continuous, with a UserWarning. Subsampling to ``max_points``
        emits a UserWarning (it was silent). ``channel_labels`` (stored but
        unused before) names the legend entries: a dict ``{label: name}`` or
        one name per category in sorted label order; for continuous colours
        one string, the colorbar label.
        """
        super().__init__()
        if sxy is not None or sz is not None:
            warn_deprecated("Passing 'sxy'/'sz'", "'pixel_sizes'")
            if pixel_sizes is None:
                pixel_sizes = (1.0 if sz is None else sz, 1.0 if sxy is None else sxy, 1.0 if sxy is None else sxy)

        # Store ORIGINAL Data
        self.X_orig = np.asarray(X)
        self.Y_orig = np.asarray(Y)
        self.Z_orig = np.asarray(Z)

        # Determine color mode and convert to numpy array immediately
        self.is_continuous = False
        self.is_categorical = False

        # A single colour (color='red', the matplotlib idiom) colours every point
        self.fixed_color = None
        if color is not None and np.ndim(color) == 0 and matplotlib.colors.is_color_like(color):
            self.fixed_color = color
            color = None

        # Ensure color is numpy array
        if color is not None:
             self.color = np.asarray(color)
        else:
             self.color = None

        # Points with a non-finite coordinate cannot be placed (and would make
        # the centroid and radius NaN); drop them with their colour values.
        self.X_orig, self.Y_orig, self.Z_orig, self.color = _finite_points(
            self.X_orig, self.Y_orig, self.Z_orig, self.color)

        # Handle empty data
        if len(self.X_orig) == 0:
             self.cx, self.cy, self.cz = 0, 0, 0
             self.max_radius = 1.0
             (pz, py, px), _, _ = _viewer_pixel_sizes(pixel_sizes)  # sxy/sz apply here too
             self.sx = px
             self.sy = py
             self.sz = pz
             self.figsize = figsize
             self.point_size = point_size
             self.alpha = alpha
             self.cmap = cmap
             self.title = title
             self.legend_names = None
             # Initialize rendering observers even if empty
             self.observe(self._render_wrapper, names=['elev', 'azim'])
             self.observe(self._save_svg, names='save_trigger')
             self._render_wrapper(None)
             return

        # Subsample if needed
        if len(self.X_orig) > max_points:
            # Subsampling changes what is drawn: say so (D16)
            warnings.warn(f"IsoScatter: showing a random subset of {max_points} of {len(self.X_orig)} points "
                          f"(max_points={max_points}); raise max_points to show more.",
                          UserWarning, stacklevel=2)
            # Local generator: reproducible, and the caller's global stream is untouched
            idx = np.random.default_rng(seed).choice(len(self.X_orig), max_points, replace=False)
            self.X_orig = self.X_orig[idx]
            self.Y_orig = self.Y_orig[idx]
            self.Z_orig = self.Z_orig[idx]
            if self.color is not None:
                 self.color = self.color[idx]

        (pz, py, px), _, _ = _viewer_pixel_sizes(pixel_sizes)
        self.sx = px
        self.sy = py
        self.sz = pz
        self.channel_labels_input = channel_labels
        self.figsize = figsize
        self.point_size = point_size
        self.alpha = alpha
        self.cmap = cmap
        self.title = title

        # Calculate Centroid and Fixed Bounds for Rotation
        self.cx = (self.X_orig.min() + self.X_orig.max()) / 2
        self.cy = (self.Y_orig.min() + self.Y_orig.max()) / 2
        self.cz = (self.Z_orig.min() + self.Z_orig.max()) / 2

        # Calculate max radius from centroid (in the same physical units as _render)
        dx = (self.X_orig - self.cx) * self.sx
        dy = (self.Y_orig - self.cy) * self.sy
        dz = (self.Z_orig - self.cz) * self.sz
        dists = np.sqrt(dx**2 + dy**2 + dz**2)
        self.max_radius = dists.max() * 1.05 # 5% buffer
        if not self.max_radius > 0:
            # One point (or identical points): avoid identical axis limits
            self.max_radius = max(self.sx, self.sy, self.sz)

        # Analyze color types (after subsampling)
        if self.color is not None:
            # Floats are continuous; integer labels (cluster ids, -1 noise),
            # bools and strings are categories (D16), integers up to
            # _MAX_AUTO_CATEGORIES distinct values like the ortho scatter.
            n_unique = len(np.unique(self.color)) if self.color.dtype.kind in 'iub' else 0
            if n_unique > _MAX_AUTO_CATEGORIES:
                 warnings.warn(f"IsoScatter: color has {n_unique} distinct integer values (more than "
                               f"{_MAX_AUTO_CATEGORIES}); drawing it as continuous values. Pass float values "
                               f"to silence this.", UserWarning, stacklevel=2)
                 self.color = self.color.astype(float)
            if np.issubdtype(self.color.dtype, np.floating):
                 self.is_continuous = True
            else:
                 self.is_categorical = True
                 # Map categories to colors
                 self.unique_cats = np.unique(self.color)
                 n_cats = len(self.unique_cats)

                 # Use a distinct colormap for categorical
                 # We'll generate a color mapping dictionary
                 # Use matplotlib 'tab10' or 'tab20' or 'viridis' if too many
                 base_cmap = matplotlib.colormaps['tab20' if n_cats > 10 else 'tab10']
                 self.cat_color_map = {cat: base_cmap(i % base_cmap.N) for i, cat in enumerate(self.unique_cats)}
                 self.c_mapped = [self.cat_color_map[c] for c in self.color]
        self.legend_names = self._legend_names(channel_labels)

        self.observe(self._render_wrapper, names=['elev', 'azim'])
        self.observe(self._save_svg, names='save_trigger')

        # Initial render
        self._render_wrapper(None)

    def _legend_names(self, channel_labels):
        """Legend entry names per category, or the colorbar label (``channel_labels``, D16)."""
        if self.is_continuous:
            if channel_labels is None:
                return 'Value'
            if not isinstance(channel_labels, str):
                raise ValueError("channel_labels for continuous colours must be one string (the colorbar label)")
            return channel_labels
        if not self.is_categorical:
            return None
        cats = list(self.unique_cats)
        if channel_labels is None:
            return {cat: str(cat) for cat in cats}
        if isinstance(channel_labels, dict):
            return {cat: str(channel_labels.get(cat, cat)) for cat in cats}
        if isinstance(channel_labels, str) or len(channel_labels) != len(cats):
            raise ValueError(f"channel_labels must be a dict {{label: name}} or a sequence of {len(cats)} names "
                             f"(one per category, in sorted label order {cats})")
        return {cat: str(name) for cat, name in zip(cats, channel_labels)}

    def _render_wrapper(self, change):
        # The figure is a plain matplotlib Figure (never registered with
        # pyplot), so a failed encode leaks nothing.
        fig = self._render()
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            self.image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    def _save_svg(self, change):
        """The figure to ``save_filename`` (see `_export_figure`); outcome in ``status_msg``."""
        _export_figure(self, self._render, self.save_filename)

    def _get_rotation_matrix(self, elev_deg, azim_deg):
        # Convert to radians
        elev = np.radians(elev_deg)
        azim = np.radians(azim_deg)

        # Rotation around Z (Azimuth)
        Rz = np.array([
            [np.cos(azim), -np.sin(azim), 0],
            [np.sin(azim),  np.cos(azim), 0],
            [0,             0,            1]
        ])

        # Rotation around X (Elevation) - or Y?
        # Usually Elevation is tilting up/down.
        # If Z is up, X is right, Y is forward (or similar).
        # Rotation around X tilts Y and Z.
        Rx = np.array([
            [1, 0,             0],
            [0, np.cos(elev), -np.sin(elev)],
            [0, np.sin(elev),  np.cos(elev)]
        ])

        # Combined Rotation
        return Rx @ Rz

    def _render(self):
        fig = _new_figure(figsize=self.figsize, facecolor='white', layout='tight')

        # Layout: 3 Rows, 2 Columns
        # Row 0 (Span 2): 3D Scatter
        # Row 1 (0): Side View (Y-Z)
        # Row 1 (1): Front View (X-Z)
        # Row 2 (0): Top View (X-Y)
        # Row 2 (1): Legend
        gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1.5, 1, 1], figure=fig)

        # Handle Empty Data
        if self.X_orig.size == 0:
            fig.text(0.5, 0.5, "No Data", ha='center', va='center')
            return fig

        # 1. Apply Rotation to Data
        # We need to center the data first to rotate around centroid
        X_c = (self.X_orig - self.cx) * self.sx
        Y_c = (self.Y_orig - self.cy) * self.sy
        Z_c = (self.Z_orig - self.cz) * self.sz

        # Stack
        points = np.stack([X_c, Y_c, Z_c], axis=1) # N x 3

        # Get Rotation Matrix
        R = self._get_rotation_matrix(self.elev, self.azim)

        points_rot = points @ R.T # (3x3 @ 3xN).T -> N x 3

        Xs, Ys, Zs = points_rot[:, 0], points_rot[:, 1], points_rot[:, 2]

        # Common scatter args
        scatter_kwargs = {
            's': self.point_size,
            'alpha': self.alpha,
        }

        if self.is_continuous:
            scatter_kwargs['c'] = self.color
            scatter_kwargs['cmap'] = self.cmap
        elif self.is_categorical:
            scatter_kwargs['c'] = self.c_mapped
        else:
            scatter_kwargs['c'] = 'gray' if self.fixed_color is None else self.fixed_color

        # Global Limits
        lim = self.max_radius
        # Limits: [-lim, lim]

        # --- 3D Plot (Row 0, Span 2) ---
        ax3d = fig.add_subplot(gs[0, :], projection='3d')
        p3d = ax3d.scatter(Xs, Ys, Zs, **scatter_kwargs)

        # Fix Camera
        ax3d.view_init(elev=20, azim=-45)

        ax3d.set_xlim([-lim, lim])
        ax3d.set_ylim([-lim, lim])
        ax3d.set_zlim([-lim, lim])
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')

        # --- Side Projection (Row 1, Left) ---
        # Z vs Y (Y on horizontal, Z on vertical)
        ax_yz = fig.add_subplot(gs[1, 0])
        ax_yz.scatter(Ys, Zs, **scatter_kwargs)
        ax_yz.set_aspect('equal')
        ax_yz.set_xlim([-lim, lim])
        ax_yz.set_ylim([-lim, lim])
        ax_yz.set_xlabel('Y')
        ax_yz.set_ylabel('Z')
        ax_yz.set_title("Side View (Y-Z)")
        ax_yz.grid(True, linestyle=':', alpha=0.6)

        # --- Front Projection (Row 1, Right) ---
        # Z vs X (X on horizontal, Z on vertical)
        ax_xz = fig.add_subplot(gs[1, 1])
        ax_xz.scatter(Xs, Zs, **scatter_kwargs)
        ax_xz.set_aspect('equal')
        ax_xz.set_xlim([-lim, lim])
        ax_xz.set_ylim([-lim, lim])
        ax_xz.set_xlabel('X')
        ax_xz.set_ylabel('Z')
        ax_xz.set_title("Front View (X-Z)")
        ax_xz.grid(True, linestyle=':', alpha=0.6)

        # --- Top Projection (Row 2, Left) ---
        # Y vs X (X horizontal, Y vertical)
        ax_yx = fig.add_subplot(gs[2, 0])
        ax_yx.scatter(Xs, Ys, **scatter_kwargs)
        ax_yx.set_aspect('equal')
        ax_yx.set_xlim([-lim, lim])
        ax_yx.set_ylim([-lim, lim])
        ax_yx.set_xlabel('X')
        ax_yx.set_ylabel('Y')
        ax_yx.set_title("Top View (X-Y)")
        ax_yx.grid(True, linestyle=':', alpha=0.6)

        # --- Legend (Row 2, Right) ---
        ax_leg = fig.add_subplot(gs[2, 1])
        ax_leg.axis('off')

        if self.is_continuous:
            # Safely add colorbar only if collection exists
            if hasattr(p3d, 'cmap'):
                 cbar = fig.colorbar(p3d, ax=ax_leg, fraction=0.8, pad=0.05, aspect=20)
                 cbar.set_label(self.legend_names)
        elif self.is_categorical:
            handles = [
                matplotlib.lines.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=self.cat_color_map[cat],
                                        markersize=10, label=self.legend_names[cat])
                for cat in self.unique_cats
            ]
            ax_leg.legend(handles=handles, loc='center', title="Categories", frameon=False)

        if self.title:
            fig.suptitle(self.title, fontsize=16)

        return fig

def show_iso_scatter(
    X, Y, Z,
    color=None,
    sxy=None, sz=None,
    figsize=(10, 8),
    point_size=5,
    alpha=0.6,
    cmap='viridis',
    max_points=10000,
    title=None,
    pixel_sizes=None,
    seed=0,
    channel_labels=None,
):
    """
    Displays an interactive isometric scatter plot widget.

    Parameters
    ----------
    X, Y, Z : array-like
        Coordinates of points. Points with a non-finite coordinate are
        dropped (with a warning), together with their ``color`` values.
    color : array-like or color, optional
        One value per point: floats are mapped through ``cmap`` (continuous);
        integers (up to 64 distinct values), bools and strings are categories
        (one colour per distinct label, with a legend); integers with more
        distinct values are drawn as continuous, with a UserWarning. Or a
        single matplotlib colour (e.g. ``'red'``) for every point.
    sxy, sz : float, deprecated
        XY and Z scaling; use ``pixel_sizes=(sz, sxy, sxy)`` (FutureWarning).
    pixel_sizes : tuple or dict, optional
        (Z, Y, X) pixel sizes, or a dict with 'Z', 'Y', 'X' keys. Takes
        precedence over ``sxy``/``sz``.
    figsize : tuple
        Size of the rendered figure.
    point_size : int
        Size of scatter points.
    alpha : float
        Transparency of points.
    cmap : str
        Colormap for continuous data.
    max_points : int
        Maximum number of points to render (random subsampling, with a
        UserWarning, if exceeded).
    title : str, optional
        Plot title.
    seed : int or None
        Seed of the subsampling (default 0, reproducible; None for a fresh draw).
    channel_labels : dict, sequence or str, optional
        Legend names of the categories: ``{label: name}`` or one name per
        category in sorted label order. For continuous colours, one string:
        the colorbar label (default 'Value').

    Changed (D16, 2026-11): integer and bool colours are categories (they
    were drawn as a continuous gradient), up to 64 distinct values;
    ``sxy``/``sz`` are deprecated; subsampling warns; new ``channel_labels``.
    """
    if sxy is not None or sz is not None:
        warn_deprecated("Passing 'sxy'/'sz'", "'pixel_sizes'")
        if pixel_sizes is None:
            pixel_sizes = (1.0 if sz is None else sz, 1.0 if sxy is None else sxy, 1.0 if sxy is None else sxy)
    return IsoScatterWidget(
        X, Y, Z,
        color=color,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        cmap=cmap,
        max_points=max_points,
        title=title,
        pixel_sizes=pixel_sizes,
        seed=seed,
        channel_labels=channel_labels,
    )
