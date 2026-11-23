# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
#     "pandas",
# ]
# ///
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.colors as mcolors
import matplotlib.ticker as mpl_ticker
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, PowerNorm
from matplotlib import colormaps as mpl_colormaps
from pathlib import Path
import pandas as pd
import itertools
import math
import datetime
import warnings
from .stats import remove_outliers

# --- Initialization: Load Font and Style ---
ROOT_DIR = Path(__file__).parent
font_path = ROOT_DIR / 'Inter-Regular.ttf'
style_path = ROOT_DIR / 'scientific.mplstyle'

# Load Style

def set_plotting_style(editable_text=True):
    # Load Style
    try:
        if style_path.exists():
            plt.style.use(str(style_path))
        else:
            print(f"Warning: Style file not found at {style_path}")
    except Exception as e:
        print(f"Warning: Failed to load style from {style_path}: {e}")
    
    # Load Font
    try:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = prop.get_name()
        else:
            # Fallback if file not found, though it should be there
            print(f"Warning: Font file not found at {font_path}")
    except Exception as e:
        print(f"Warning: Failed to load font from {font_path}: {e}")

    # Set SVG export settings for editable text globally if requested
    if editable_text:
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['svg.fonttype'] = 'path'
        plt.rcParams['axes.unicode_minus'] = True



# --- Initialization: Create labels_cmap ---
from .labels_cmap_data import LABELS_CMAP_COLORS

# Start with a transparent black at index 0
final_colors = [(0, 0, 0, 0)]

# Add the 255 pre-calculated Glasbey-style maximally distant colors
for r, g, b in LABELS_CMAP_COLORS:
    final_colors.append((r, g, b, 1.0))


class _LabelsColormap(ListedColormap):
    """Lookup table whose transparent entry 0 is reserved for background (value 0).

    A normalised value x picks entry ``int(x * N)``, so with more than 256 labels
    the lowest label ids (x < 1/N) would land on the transparent entry. Any
    positive value is therefore moved to entry 1 at least.
    """

    def __call__(self, X, alpha=None, bytes=False):
        Xa = np.ma.asarray(X)
        if Xa.dtype.kind == "f":
            # norm() returns masked arrays: masked (bad) values are left alone
            low = np.ma.filled((Xa > 0) & (Xa < 1.0 / self.N), False)
            if np.any(low):
                Xa = Xa.copy()
                Xa[low] = 1.0 / self.N
                X = Xa if np.ndim(X) else Xa.item()
        return super().__call__(X, alpha=alpha, bytes=bytes)


# Create the new colormap (entry i is final_colors[i])
labels_cmap = _LabelsColormap(final_colors, name="labels_cmap")

try:
    mpl_colormaps.register(labels_cmap, name="labels_cmap", force=True)
except AttributeError:
    # Fallback for older Matplotlib versions
    plt.register_cmap(name="labels_cmap", cmap=labels_cmap)

# Print a hint for users who might need the background to be black
print("Hint: labels_cmap background is transparent by default. To set it to black, run:")
print("labels_cmap._init(); labels_cmap._lut[0] = [0, 0, 0, 1]")


# --- Color Stops (White -> Pastel -> Pure -> Deep Saturated) ---
# Each ramp starts at #FFFFFF and terminates in a deep, high-chroma shade.
I_SERIES_STOPS = {
    "I Blue": ["#FFFFFF", "#66B2FF", "#0066FF", "#003366"],
    "I Bordeaux": ["#FFFFFF", "#C4526E", "#800020", "#4A0014"],
    "I Forest": ["#FFFFFF", "#72CB72", "#228B22", "#0A3B18"],
    # For orange, #7A2500 retains better chromatic identity than #591C00 (which borders on muddy brown)
    "I Orange": ["#FFFFFF", "#FFC266", "#FF7F00", "#7A2500"],
    "I Purple": ["#FFFFFF", "#D187D1", "#800080", "#3B003B"],
    # Broadband subtractive primaries & counterstains
    "I Teal": ["#FFFFFF", "#66E0D0", "#00A896", "#004D40"],     # Cyan-like (Absorbs R)
    "I Rose": ["#FFFFFF", "#FF80BF", "#E6007A", "#59002E"],     # Magenta-like (Absorbs G)
    "I Gold": ["#FFFFFF", "#FFE066", "#D49B00", "#664400"],     # Yellow-like (Absorbs B, dark anchor)
    "I Slate": ["#FFFFFF", "#94A3B8", "#475569", "#0F172A"],    # Neutral DAPI/Histology counterstain
}

# --- Instantiate and Register ---
I_SERIES_COLORMAPS = {
    name: LinearSegmentedColormap.from_list(name, stops, N=256)
    for name, stops in I_SERIES_STOPS.items()
}

i_blue = I_SERIES_COLORMAPS["I Blue"]
i_bordeaux = I_SERIES_COLORMAPS["I Bordeaux"]
i_forest = I_SERIES_COLORMAPS["I Forest"]
i_orange = I_SERIES_COLORMAPS["I Orange"]
i_purple = I_SERIES_COLORMAPS["I Purple"]
i_teal = I_SERIES_COLORMAPS["I Teal"]
i_rose = I_SERIES_COLORMAPS["I Rose"]
i_gold = I_SERIES_COLORMAPS["I Gold"]
i_slate = I_SERIES_COLORMAPS["I Slate"]

for name, cmap in I_SERIES_COLORMAPS.items():
    alias_name = name.lower().replace(" ", "_")

    # Matplotlib >= 3.7 registry pattern with backward-compatibility fallback
    for reg_name in (name, alias_name):
        try:
            mpl_colormaps.register(cmap, name=reg_name, force=True)
        except AttributeError:
            plt.register_cmap(name=reg_name, cmap=cmap)


def _image_info(image, shown=None):
    """Structured description of an array or an (STCZYX) xarray.DataArray.

    Returns ``(rows, channels)``: ``rows`` is a list of ``(key, value)`` strings,
    ``channels`` is ``None`` or ``(columns, cells)`` with one row of strings per
    channel (index, label, then one entry per auxiliary ``C`` coordinate).
    """
    rows, channels = [], None
    if hasattr(image, "dims") and hasattr(image, "coords"):
        src = image.attrs.get("source") if hasattr(image, "attrs") else None
        if src:
            rows.append(("source", str(src)))
        rows.append(("dims", " ".join(f"{d}={n}" for d, n in image.sizes.items())))
        rows.append(("dtype", str(image.dtype)))
        try:
            from .image_io import get_stczyx_pixel_sizes

            ps = get_stczyx_pixel_sizes(image)
            if any(ps.get(a) is not None for a in ("Z", "Y", "X")):
                rows.append((
                    "pixel size",
                    " ".join(f"{a}={ps[a]:.4g}" if ps.get(a) is not None else f"{a}=?" for a in ("Z", "Y", "X"))
                    + (f" {ps['unit']}" if ps.get("unit") else ""),
                ))
        except ImportError:  # image_io needs xarray + tifffile
            pass
        if "C" in image.dims:
            labels = list(image.coords["C"].values) if "C" in image.coords else list(range(image.sizes["C"]))
            meta = [k for k, c in image.coords.items() if c.dims == ("C",) and k != "C"]
            cells = []
            for i, lab in enumerate(labels):
                vals = [image.coords[k].values[i] for k in meta]
                cells.append([str(i), str(lab)] + ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in vals])
            channels = (["C", "name"] + [str(k) for k in meta], cells)
    else:
        rows.append(("shape", str(tuple(np.shape(image)))))
        rows.append(("dtype", str(np.asarray(image).dtype)))
    if shown:
        rows.append(("shown", shown))
    return rows, channels


def _image_info_lines(image, shown=None) -> list:
    """Text lines describing an array or an (STCZYX) xarray.DataArray."""
    rows, channels = _image_info(image, shown)
    lines = [f"{k}: {v}" for k, v in rows if k != "shown"]
    if channels is not None:
        columns, cells = channels
        for row in cells:
            extra = " ".join(f"{k}={v}" for k, v in zip(columns[2:], row[2:]))
            lines.append(f"C[{row[0]}] {row[1]}" + (f"  {extra}" if extra else ""))
    if shown:
        lines.append(f"shown: {shown}")
    return lines


def _swatch_color(value):
    """RGB of a channel colour entry (a colormap name gives its last colour), else None."""
    if not isinstance(value, str) or not value:
        return None
    try:  # colormap names first: 'gray', 'pink', 'hot' are also CSS colours
        return mpl_colormaps[value](1.0)[:3]
    except (KeyError, ValueError):
        pass
    return mcolors.to_rgb(value) if mcolors.is_color_like(value) else None


def _wrap_value(text, width):
    """Greedy wrap that may also break after '/' or ',', so long paths wrap at directories."""
    import re

    width = max(int(width), 8)
    tokens = [t for t in re.split(r"(?<=[/,])|(?= )", text) if t]
    lines, cur = [], ""
    for tok in tokens:
        if len(cur) + len(tok) <= width:
            cur += tok
            continue
        if cur.strip():
            lines.append(cur.rstrip())
        tok = tok.lstrip() if cur else tok
        while len(tok) > width:  # a single token longer than the line
            lines.append(tok[:width])
            tok = tok[width:]
        cur = "  " + tok if lines else tok  # hanging indent for continuation lines
    if cur.strip() or not lines:
        lines.append(cur.rstrip())
    return lines


def _draw_info_table(ax, rows, channels, width_pt, fontsize, shown_channel=None, max_chars=None):
    """Draw a key/value table (and a channel table) on ``ax`` in point units.

    The axes data coordinates are points with y pointing down; the caller sets
    the axes position to ``width_pt`` x returned height. Values and cells are
    monospace, so their widths are known without a renderer.
    """
    from matplotlib.patches import Rectangle

    fs = fontsize
    cw = 0.61 * fs  # monospace advance (DejaVu Sans Mono: 0.602 em), small safety margin
    kw = 0.66 * fs  # conservative average advance of bold proportional text
    pad = 0.5 * fs
    ink, muted, rule, zebra, highlight = "#2B2B2B", "#6B6B6B", "#C8C8C8", "#F3F3F3", "#E1ECFA"

    def row_height(n_lines, size):
        return (1.2 * n_lines + 0.5) * size

    def text(x, y, s, size, **kw_):
        ax.text(x, y + 0.25 * size, s, fontsize=size, va="top", ha="left", linespacing=1.2,
                clip_on=True, **kw_)

    y = 0.0
    key_w = max(len(k) for k, _ in rows) * kw + 2 * pad
    val_chars = int((width_pt - key_w - pad) / cw)
    if max_chars:
        val_chars = min(val_chars, int(max_chars))
    for i, (key, value) in enumerate(rows):
        lines = _wrap_value(value, val_chars)
        h = row_height(len(lines), fs)
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, y), width_pt, h, fc=zebra, ec="none"))
        text(pad, y, key, fs, color=muted, fontweight="bold")
        text(key_w, y, "\n".join(lines), fs, color=ink, family="monospace")
        y += h

    if channels is not None:
        columns, cells = channels
        y += 0.8 * fs
        color_col = columns.index("color") if "color" in columns else None

        def widths(size):
            out = []
            for j, name in enumerate(columns):
                w = max([len(name) * kw / fs * size] + [len(r[j]) * cw / fs * size for r in cells])
                if j == color_col:
                    w += 1.3 * size
                out.append(w + 2 * pad / fs * size)
            return out

        cfs = fs
        if sum(widths(fs)) > width_pt:  # shrink the channel table to the figure width
            cfs = max(6.0, fs * width_pt / sum(widths(fs)))
        col_w = widths(cfs)
        xs = np.concatenate([[0.0], np.cumsum(col_w)[:-1]])
        cpad = pad / fs * cfs

        h = row_height(1, cfs)
        for x, name in zip(xs, columns):
            text(x + cpad, y, name, cfs, color=muted, fontweight="bold")
        y += h
        ax.plot([0, min(width_pt, sum(col_w))], [y, y], color=rule, lw=0.8, solid_capstyle="butt", clip_on=True)
        for i, row in enumerate(cells):
            is_shown = shown_channel is not None and i == shown_channel
            if is_shown:
                ax.add_patch(Rectangle((0, y), width_pt, h, fc=highlight, ec="none"))
            for j, (x, cell) in enumerate(zip(xs, row)):
                x0 = x + cpad
                if j == color_col:
                    rgb = _swatch_color(cell)
                    if rgb is not None:
                        s = 0.85 * cfs
                        ax.add_patch(Rectangle((x0, y + (h - s) / 2), s, s, fc=rgb, ec=muted, lw=0.5))
                    x0 += 1.3 * cfs
                text(x0, y, cell, cfs, color=ink, family="monospace",
                     fontweight="bold" if is_shown else "normal")
            y += h
        if shown_channel is not None:
            text(sum(col_w) + cpad, y - (len(cells) - shown_channel) * h, "◀ shown", cfs, color=muted)

    ax.set_xlim(0, width_pt)
    ax.set_ylim(y, 0)
    ax.axis("off")
    return y


class _HTML(str):
    """A ``str`` of HTML that notebooks (Jupyter, marimo) render as HTML."""

    def _repr_html_(self):
        return str(self)


def _info_html(rows, channels, shown_channel=None):
    """The info table as self-contained HTML (inline styles, no dependency), in
    the spirit of the dask / xarray notebook reprs. Backgrounds are translucent
    so the table reads on light and dark notebook themes."""
    from html import escape

    cell = "padding:2px 10px 2px 6px;text-align:left;vertical-align:top;border:none;color:inherit;"
    mono = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;"
    key = cell + "font-weight:600;opacity:0.7;white-space:nowrap;"
    zebra = "background:rgba(128,128,128,0.10);"
    out = ['<div style="font-size:12px;line-height:1.35;">',
           '<table style="border-collapse:collapse;margin:0 0 8px 0;color:inherit;font-size:inherit;">']
    for i, (k, v) in enumerate(rows):
        bg = zebra if i % 2 == 0 else ""
        out.append(f'<tr style="{bg}"><th style="{key}">{escape(k)}</th>'
                   f'<td style="{cell}{mono}white-space:pre-wrap;overflow-wrap:anywhere;">{escape(v)}</td></tr>')
    out.append("</table>")
    if channels is not None:
        columns, cells = channels
        out.append('<table style="border-collapse:collapse;color:inherit;font-size:inherit;">')
        out.append("<tr>" + "".join(
            f'<th style="{key}border-bottom:1px solid rgba(128,128,128,0.5);">{escape(c)}</th>' for c in columns
        ) + "</tr>")
        for i, row in enumerate(cells):
            shown = shown_channel is not None and i == shown_channel
            bg = "background:rgba(66,133,244,0.18);font-weight:600;" if shown else ""
            tds = []
            for c, v in zip(columns, row):
                sw = ""
                rgb = _swatch_color(v) if c == "color" else None
                if rgb is not None:
                    sw = ('<span style="display:inline-block;width:0.85em;height:0.85em;margin-right:4px;'
                          f'vertical-align:-0.1em;border:1px solid rgba(128,128,128,0.8);background:{mcolors.to_hex(rgb)};"></span>')
                tds.append(f'<td style="{cell}{mono}white-space:pre;">{sw}{escape(v)}</td>')
            mark = f'<td style="{cell}opacity:0.7;">&#9664; shown</td>' if shown else ""
            out.append(f'<tr style="{bg}">' + "".join(tds) + mark + "</tr>")
        out.append("</table>")
    out.append("</div>")
    return _HTML("".join(out))


def describe_image(image, bins=64, gamma=1, return_image_only=False, info_width=None, image_size=4.0,
                   **imshow_kwargs):
    """
    Displays an image, its histogram and an info table, stacked in one column.

    Top to bottom: the image (longer side ``image_size`` inches); a short
    histogram as wide as the image (fraction of pixels per bin, on a log scale
    when it spans more than a decade), with the display range (``vmin`` /
    ``vmax`` of the image's norm) marked by dashed lines and the clipped
    intensities shaded when it differs from the data range; and a
    full-width info table.

    For inputs with more than two dimensions the middle slice of every extra
    dimension is shown. The info table lists the shape and dtype, or for an
    ``xarray.DataArray`` (e.g. STCZYX from ``image_io``) the source, dims,
    pixel sizes and which slice is shown, plus min / max / mean of the shown
    slice and, when set (``vmin``/``vmax``/``gamma``), the display range with
    the fraction of clipped pixels; channels and their metadata coordinates
    (``set_channel_info``) get their own table, with a swatch for a ``color``
    entry and the shown channel highlighted.

    Parameters
    ----------
    image : array-like or xarray.DataArray
        The input image.
    bins : int, optional
        The number of bins to use for the histogram. Default is 64.
    gamma : float, optional
        Display gamma (``PowerNorm``); ``vmin``/``vmax`` passed in ``imshow_kwargs``
        are applied to that norm.
    return_image_only : bool, optional
        Return only the 2D slice that would be shown.
    info_width : int, optional
        Maximum characters per value line in the info table (default: as many
        as fit the figure width).
    image_size : float, optional
        Length in inches of the image's longer side.
    **imshow_kwargs
        Passed to ``imshow`` (``cmap``, ``vmin``, ``vmax``, ...).

    Returns
    -------
    dict
        ``"fig"`` (:class:`matplotlib.figure.Figure`), ``"axes"`` (mapping
        ``"Image"``, ``"Histogram"``, ``"Info"`` to axes), ``"info"`` (plain
        text) and ``"info_html"`` (the same table as HTML; a ``str`` that
        notebooks render when it is a cell's output). When
        ``return_image_only`` is ``True`` only the 2D slice is returned.

    Examples
    --------
    >>> res = describe_image(np.random.rand(100, 100), vmax=0.8)
    >>> print(res["info"])
    >>> res["info_html"]  # last expression in a notebook cell: rendered table

    Formerly ``hist_imshow`` (deprecated alias).
    """
    ### SET DEFAULT KWARGS
    imshow_kwargs.setdefault('origin', 'lower')
    imshow_kwargs.setdefault('cmap', 'gray')

    shown = None
    shown_channel = None
    if hasattr(image, "dims") and hasattr(image, "isel"):
        da = image
        extra = [d for d in da.dims[:-2]]
        sel = {d: da.sizes[d] // 2 for d in extra}
        if sel:
            parts = []
            for d, i in sel.items():
                lab = da.coords[d].values[i] if d in da.coords and d == "C" else i
                parts.append(f"{d}={lab}")
            shown = ", ".join(parts)
        shown_channel = sel.get("C")
        info_src = da
        image = np.asarray(da.isel(sel).values) if sel else np.asarray(da.values)
    else:
        image = np.asarray(image)
        im_shape = image.shape
        if len(im_shape) > 2:
            middle_indices = [s // 2 for s in im_shape[:-2]]
            shown = "middle slice " + str(tuple(middle_indices)) + " of the leading dims"
            image = image[tuple(middle_indices + [slice(None), slice(None)])]
        info_src = np.empty(im_shape, dtype=image.dtype) if len(im_shape) > 2 else image

    if return_image_only:
        return image

    info_lines = _image_info_lines(info_src, shown)
    rows, channels = _image_info(info_src, shown)

    dtype = image.dtype
    if image.dtype == bool:
        # numpy cannot subtract booleans (needed by hist and by the stats below)
        image = image.astype(np.uint8)

    finite = image[np.isfinite(image)] if image.dtype.kind in "fc" else image.ravel()
    if finite.size:
        stats = f"min {finite.min():.4g}  max {finite.max():.4g}  mean {finite.mean():.4g}"
    else:
        stats = "no finite values"

    # --- figure: one column (image, histogram, info), axes placed in inches ---
    H, W = image.shape[:2]
    w_img = image_size * W / max(H, W)
    h_img = image_size * H / max(H, W)
    w_hist, h_hist = max(w_img, 3.0), 0.9
    col_w = max(w_img, w_hist)

    fig = plt.figure(figsize=(col_w + 2.0, h_img + h_hist + 4.0))
    ax_img = fig.add_axes([0.1, 0.6, 0.4, 0.3])
    ax_hist = fig.add_axes([0.1, 0.4, 0.4, 0.1])
    ax_info = fig.add_axes([0.0, 0.0, 1.0, 0.3])

    norm = None
    if gamma != 1:
        # matplotlib refuses a norm together with vmin/vmax, so hand them to the norm
        norm = PowerNorm(gamma=gamma, vmin=imshow_kwargs.pop('vmin', None),
                         vmax=imshow_kwargs.pop('vmax', None))
    im = ax_img.imshow(image, norm=norm, interpolation='nearest', **imshow_kwargs)
    vmin, vmax = im.norm.vmin, im.norm.vmax

    frac = np.zeros(0)
    if finite.size:
        # fraction of pixels per bin: unit-free, unlike a density in 1/intensity
        frac, _, _ = ax_hist.hist(finite.ravel(), bins=bins, histtype='stepfilled',
                                  weights=np.full(finite.size, 1.0 / finite.size))
    nonzero = frac[frac > 0]
    if nonzero.size and nonzero.max() > 10 * nonzero.min():
        # log fraction only pays off over more than a decade (else no major tick is labelled)
        ax_hist.set_yscale('log')
        ax_hist.yaxis.set_major_locator(mpl_ticker.LogLocator(numticks=3))
        ax_hist.yaxis.set_minor_formatter(mpl_ticker.NullFormatter())
    else:
        ax_hist.yaxis.set_major_locator(mpl_ticker.MaxNLocator(2))
    ax_hist.set_xlabel(f'Pixel intensity ({dtype})')
    ax_hist.set_ylabel('Fraction')

    # display range: dashed lines at vmin / vmax, intensities the display clips shaded
    display = None
    if vmin is not None and vmax is not None and np.isfinite([vmin, vmax]).all():
        lo = min(float(finite.min()), vmin) if finite.size else vmin
        hi = max(float(finite.max()), vmax) if finite.size else vmax
        if hi <= lo:
            hi = lo + 1
        ax_hist.set_xlim(lo, hi)
        dmin, dmax = (float(finite.min()), float(finite.max())) if finite.size else (None, None)
        for v in (vmin, vmax):
            if v not in (dmin, dmax):  # autoscaled limits sit on the data range: nothing to mark
                ax_hist.axvline(v, color="#6B6B6B", ls=(0, (3, 2)), lw=1)
        if vmin > lo:
            ax_hist.axvspan(lo, vmin, color="#808080", alpha=0.18, lw=0)
        if vmax < hi:
            ax_hist.axvspan(vmax, hi, color="#808080", alpha=0.18, lw=0)
        if (vmin, vmax) != (dmin, dmax) or gamma != 1:  # else it repeats the intensity row
            display = f"vmin {vmin:.4g}  vmax {vmax:.4g}" + (f"  gamma {gamma:g}" if gamma != 1 else "")
            if finite.size:
                clipped = np.mean((finite < vmin) | (finite > vmax))
                if clipped > 0:
                    display += f"  ({100 * clipped:.2g}% of pixels clipped)"

    info_lines.append(stats)
    rows.append(("intensity", stats))
    if display:
        info_lines.append(f"display: {display}")
        rows.append(("display", display))
    info = "\n".join(info_lines)

    def overhang(ax):
        """Inches by which ticks/labels stick out of the axes box: left, right, bottom, top."""
        fig.draw_without_rendering()
        tb, bb = ax.get_tightbbox(), ax.get_window_extent()
        d = fig.dpi
        return (max(bb.x0 - tb.x0, 0) / d, max(tb.x1 - bb.x1, 0) / d,
                max(bb.y0 - tb.y0, 0) / d, max(tb.y1 - bb.y1, 0) / d)

    pad, gap = 0.12, 0.15
    fw, fh = fig.get_size_inches()
    ax_img.set_position([0, 0, w_img / fw, h_img / fh])
    ax_hist.set_position([0, 0, w_hist / fw, h_hist / fh])
    oi, oh = overhang(ax_img), overhang(ax_hist)

    dx_img, dx_hist = (col_w - w_img) / 2, (col_w - w_hist) / 2  # both centred in the column
    left = pad + max(oi[0] - dx_img, oh[0] - dx_hist, 0)
    right = pad + max(oi[1] - dx_img, oh[1] - dx_hist, 0)
    fig_w = max(left + col_w + right, 4.5)
    x0 = left + (fig_w - left - col_w - right) / 2

    info_w_pt = (fig_w - 2 * pad) * 72
    info_fs = float(np.clip(0.65 * plt.rcParams['font.size'], 7, 10))
    info_h = _draw_info_table(ax_info, rows, channels, info_w_pt, info_fs,
                              shown_channel=shown_channel, max_chars=info_width) / 72

    top = pad + oi[3]
    gap_ih = gap + oi[2] + oh[3]  # image tick labels above, histogram offset text below
    below = oh[2]
    fig_h = top + h_img + gap_ih + h_hist + below + gap + info_h + pad

    fig.set_size_inches(fig_w, fig_h)
    y_img = fig_h - top - h_img
    y_hist = y_img - gap_ih - h_hist
    ax_img.set_position([(x0 + dx_img) / fig_w, y_img / fig_h, w_img / fig_w, h_img / fig_h])
    ax_hist.set_position([(x0 + dx_hist) / fig_w, y_hist / fig_h, w_hist / fig_w, h_hist / fig_h])
    ax_info.set_position([pad / fig_w, pad / fig_h, info_w_pt / 72 / fig_w, info_h / fig_h])

    axes = {"Image": ax_img, "Histogram": ax_hist, "Info": ax_info}
    return {"fig": fig, "axes": axes, "info": info,
            "info_html": _info_html(rows, channels, shown_channel)}


def hist_imshow(*args, **kwargs):
    """Deprecated alias of :func:`describe_image` (same arguments and return value)."""
    import warnings

    warnings.warn(
        "hist_imshow is deprecated; use eigenp_utils.describe_image (same arguments and return value)",
        DeprecationWarning,
        stacklevel=2,
    )
    return describe_image(*args, **kwargs)


def color_coded_projection(image: np.ndarray, color_map='plasma') -> np.ndarray:
    """
    Pseudocolors each frame of a 3D image (time, y, x) using the specified color map.

    :param image: 3D array (time, y, x) representing the image
    :param color_map: Name of the color map to use (default is 'plasma')
    :return: RGB image with dimensions (y, x, channel)
    """
    # Ensure the input image has three dimensions (time, y, x)
    if image.ndim != 3:
        raise ValueError("Input image must have three dimensions (time, y, x)")

    # Get the colormap
    cmap = mpl_colormaps[color_map]

    # Get the time dimension
    time_dimension = image.shape[0]

    # Initialize the result array (y, x, 3) for the maximum intensity projection
    rgb_image = np.zeros((image.shape[1], image.shape[2], 3), dtype=np.float32)

    # Loop through the time dimension and apply the colormap
    for t in range(time_dimension):
        # Normalize the current frame to the range [0, 1], in float: integer
        # differences overflow (int16 -20000..20000), and NaN pixels count as 0
        frame = image[t].astype(np.float32)
        frame_min = np.nanmin(frame) if np.isfinite(frame).any() else 0.0
        frame_max = np.nanmax(frame) if np.isfinite(frame).any() else 0.0
        if frame_max > frame_min:
            frame_normalized = np.nan_to_num((frame - frame_min) / (frame_max - frame_min))
        else:
            # A constant frame has no contrast to normalise: keep its value, but
            # clipped to [0, 1] (a frame of 1000 must not give RGB values of 1000)
            frame_normalized = np.clip(np.nan_to_num(frame), 0.0, 1.0)

        # Get the corresponding color value from the colormap based on the current time index
        color_value = cmap(t / max(time_dimension - 1, 1))[:3]

        # Modulate the color value by the intensity factor and update max projection
        # Optimization: We update rgb_image iteratively to avoid allocating the large (time, y, x, 3) array.
        # We also loop over channels to avoid allocating a full (y, x, 3) temporary for the current frame.
        for c in range(3):  # Loop through the RGB channels
            # Compute this channel's contribution for the current time point
            # This allocates a temporary (y, x) array which is much smaller
            channel_component = frame_normalized * color_value[c]

            # Update the max projection in-place
            np.maximum(rgb_image[:, :, c], channel_component, out=rgb_image[:, :, c])

    return rgb_image


def raincloud_plot(data,
                   x_label=None,
                   y_label=None,
                   title=None,
                   ax=None,
                   orientation='vertical',
                   palette=None,
                   figsize=(4, 4),
                   x=None, y=None, hue=None, dodge=None,
                   size_scatter=40,
                   size_median=150,
                   alpha_scatter=0.50,
                   alpha_violin=0.8,
                   linewidth_scatter=0.5,
                   linewidth_boxplot=8,
                   offset_scatter=0.20,
                   width_violin=0.5,
                   raster_threshold=1e3,
                   outlier_method=None,
                   outlier_multiplier=3.0,
                   highlight_mask=None,
                   highlight_color='lime'):
    """
    Creates a raincloud plot (half-violin + boxplot + jittered scatter).

    Parameters
    ----------
    data : array-like, dict, or pd.DataFrame
        Input data. Can be:
        - Single array/list/tuple or pd.Series.
        - List of arrays/lists.
        - Dictionary {label: array}.
        - DataFrame (columns are groups).
    x_label : str, optional
        Label for x-axis.
    y_label : str, optional
        Label for y-axis.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    orientation : {'vertical', 'horizontal'}, default 'vertical'
        Orientation of the plot.
    palette : list of colors or str, optional
        Colors to use.
    figsize : tuple, default (4, 4)
        Figure size if creating a new figure.
    x : str, optional
        If data is a DataFrame, column name for x-axis variable.
    y : str, optional
        If data is a DataFrame, column name for y-axis variable.
    hue : str, optional
        If data is a DataFrame, column name for grouping variable.
    dodge : bool, optional
        When hue is nested, whether to draw elements at different positions.
        If None, defaults to True unless hue matches x or y variable.
    raster_threshold : int, optional
        If the total number of scatter points across all groups exceeds this threshold,
        the jittered scatter points will be rasterized. This is useful for keeping SVG
        file sizes small when there are many points.
    outlier_method : str, optional
        Method to use for identifying outliers to exclude from the violin KDE estimation
        (e.g., 'mad', 'iqr', 'zscore', 'robust_zscore'). If set, outliers are excluded
        from the violin plot but still shown in the scatter plot.
    outlier_multiplier : float, default 3.0
        The threshold multiplier for the outlier method (e.g., number of MADs or IQRs).
    highlight_mask : array-like or pd.Series, optional
        A boolean mask matching the shape/index of the input data. Scatter points where
        this mask is True will be colored with `highlight_color`. For a DataFrame the
        mask is per row; for a dict, give a dict of masks; for a list of groups, a list
        with one mask per group. A mask that cannot be matched raises a warning.
    highlight_color : str, default 'lime'
        The color to use for highlighted scatter points.

    Returns
    -------
    dict
        {'fig': figure, 'axes': ax}
    """

    # --- 1. Normalize Input Data ---
    # We will convert all inputs into a list of "PlotItems".
    # Each PlotItem is a dict: {'values': array, 'position': float, 'color': Any, 'width': float, 'label': str}

    plot_items = []
    xtick_labels = []
    xtick_positions = []

    vert = (orientation == 'vertical')

    # Branch A: DataFrame with x/y specified (Seaborn-style)
    if isinstance(data, pd.DataFrame) and (x is not None or y is not None):
        if vert:
            cat_col = x
            val_col = y
            if val_col is None: raise ValueError("For vertical plot, y must be specified (as value).")
        else:
            cat_col = y
            val_col = x
            if val_col is None: raise ValueError("For horizontal plot, x must be specified (as value).")

        # Check columns exist
        if val_col and val_col not in data.columns: raise ValueError(f"Column {val_col} not found.")
        if cat_col and cat_col not in data.columns: raise ValueError(f"Column {cat_col} not found.")
        if hue and hue not in data.columns: raise ValueError(f"Column {hue} not found.")

        # Get Data and Drop NaNs
        cols_to_check = [c for c in [val_col, cat_col, hue] if c]
        df_clean = data.dropna(subset=cols_to_check)

        # Get Categories
        if cat_col:
            cats = df_clean[cat_col].unique()
            # Try to respect pandas categorical order if present
            if isinstance(df_clean[cat_col].dtype, pd.CategoricalDtype):
                cats = df_clean[cat_col].cat.categories
                cats = [c for c in cats if c in df_clean[cat_col].values]
            else:
                try:
                    cats = np.sort(cats)
                except:
                    pass
        else:
            cats = ["Data"]

        # Get Hues
        if hue:
            hues = df_clean[hue].unique()
            if isinstance(df_clean[hue].dtype, pd.CategoricalDtype):
                hues = df_clean[hue].cat.categories
                hues = [h for h in hues if h in df_clean[hue].values]
            else:
                try:
                    hues = np.sort(hues)
                except:
                    pass
        else:
            hues = [None]

        # Auto-detect dodge
        if dodge is None:
            if hue and cat_col and (hue == cat_col):
                dodge = False
            else:
                dodge = True

        # Resolve Palette
        if hue:
            if isinstance(palette, dict):
                color_map = palette
            else:
                if palette is None:
                    pal_list = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B", "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC"]
                    pal_iter = itertools.cycle(pal_list)
                elif isinstance(palette, str):
                    # Check if it's a colormap
                    try:
                        cmap = mpl_colormaps[palette]
                        # Sample colors from colormap
                        if len(hues) > 1:
                            colors = cmap(np.linspace(0, 1, len(hues)))
                        else:
                            colors = [cmap(0.5)]
                        pal_iter = itertools.cycle(colors)
                    except (KeyError, ValueError):
                        # Not a colormap, treat as single color
                        pal_iter = itertools.cycle([palette])
                else:
                    pal_iter = itertools.cycle(palette)

                color_map = {h: next(pal_iter) for h in hues}
        else:
             # Legacy default: all blue if no palette, or cycle if palette
             pass

        # Build PlotItems
        total_width = width_violin
        n_hues = len(hues)
        if dodge:
            slot_width = total_width / n_hues
        else:
            slot_width = total_width

        for i_cat, cat in enumerate(cats):
            xtick_positions.append(i_cat)
            xtick_labels.append(str(cat))

            if cat_col:
                sub_df = df_clean[df_clean[cat_col] == cat]
            else:
                sub_df = df_clean

            for i_hue, h in enumerate(hues):
                if hue:
                    group_data = sub_df[sub_df[hue] == h]
                    label = f"{cat} - {h}"

                    if dodge:
                        offset = (i_hue - (n_hues - 1) / 2) * slot_width
                    else:
                        offset = 0

                    pos = i_cat + offset
                    col = color_map.get(h, "#4C78A8")
                else:
                    group_data = sub_df
                    label = str(cat)
                    pos = i_cat
                    offset = 0

                    if palette is None:
                        col = "#4C78A8"
                    elif isinstance(palette, str):
                        col = palette
                    elif isinstance(palette, dict):
                         col = palette.get(cat, "#4C78A8")
                    else:
                        pal_cycle = itertools.cycle(palette)
                        col = next(itertools.islice(pal_cycle, i_cat, i_cat + 1))

                vals = group_data[val_col].values
                if highlight_mask is not None:
                    if isinstance(highlight_mask, pd.Series):
                        # Extract by index
                        mask_subset = highlight_mask.loc[group_data.index].values
                    else:
                        # Assume array-like of same length as data, we need the indices
                        if isinstance(data, pd.DataFrame):
                            # group_data is a sub-dataframe, we can get its integer positions from the original data
                            # But group_data has the original index if it's from df_clean
                            # Wait, df_clean dropped NaNs. Let's just index the mask by the original index
                            try:
                                mask_series = pd.Series(highlight_mask, index=data.index)
                                mask_subset = mask_series.loc[group_data.index].values
                            except Exception:
                                mask_subset = None
                        else:
                            mask_subset = None
                else:
                    mask_subset = None

                if len(vals) == 0: continue

                plot_items.append({
                    'values': vals,
                    'mask': mask_subset,
                    'position': pos,
                    'width': slot_width,
                    'color': col,
                    'label': label
                })

        if x_label is None: x_label = cat_col if vert else val_col
        if y_label is None: y_label = val_col if vert else cat_col

    else:
        # Branch B: Legacy Input Processing
        raw_plot_data = [] # List of (label, values)
        raw_masks = []  # highlight mask per group (None where it cannot be applied)

        def _group_mask(mask, keep, index=None):
            """Restrict a per-value mask to the values kept after NaN removal."""
            if mask is None:
                return None
            if isinstance(mask, pd.Series) and index is not None:
                mask = mask.reindex(index).fillna(False)
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != keep.shape:
                return None
            return mask[keep]

        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                keep = data[col].notna().to_numpy()
                vals = data[col].to_numpy()[keep]
                raw_plot_data.append((str(col), vals))
                raw_masks.append(_group_mask(highlight_mask, keep, data.index))
        elif isinstance(data, pd.Series):
            keep = data.notna().to_numpy()
            vals = data.to_numpy(dtype=float)[keep]
            label = str(data.name) if data.name is not None else "Data"
            raw_plot_data.append((label, vals))
            raw_masks.append(_group_mask(highlight_mask, keep, data.index))
        elif isinstance(data, dict):
            mask_dict = highlight_mask if isinstance(highlight_mask, dict) else {}
            for label, values in data.items():
                vals = np.asarray(values, dtype=float)
                keep = ~np.isnan(vals)
                raw_plot_data.append((str(label), vals[keep]))
                raw_masks.append(_group_mask(mask_dict.get(label), keep))
        elif isinstance(data, (list, tuple, np.ndarray)):
            is_multiple = False
            try:
                numeric_arr = np.asarray(data, dtype=float)
                if numeric_arr.ndim > 1:
                    is_multiple = True
                else:
                    is_multiple = False
            except (ValueError, TypeError):
                is_multiple = True

            if is_multiple:
                # One mask per group, given as a list/tuple with one entry per group
                group_masks = (
                    list(highlight_mask)
                    if isinstance(highlight_mask, (list, tuple)) and len(highlight_mask) == len(data)
                    else [None] * len(data)
                )
                for i, d in enumerate(data):
                    vals = np.asarray(d, dtype=float)
                    keep = ~np.isnan(vals)
                    raw_plot_data.append((str(i), vals[keep]))
                    raw_masks.append(_group_mask(group_masks[i], keep))
            else:
                vals = np.asarray(data, dtype=float)
                keep = ~np.isnan(vals)
                raw_plot_data.append(("Data", vals[keep]))
                raw_masks.append(_group_mask(highlight_mask, keep))
        else:
            raise ValueError("Unsupported data type")

        if highlight_mask is not None and all(m is None for m in raw_masks):
            warnings.warn(
                "highlight_mask could not be matched to the data and is ignored. Give one "
                "boolean per value (per row for a DataFrame), a dict of masks for dict data, "
                "or a list with one mask per group for a list of groups.",
                stacklevel=2,
            )

        # Colors
        n_groups = len(raw_plot_data)

        # Determine effective labels for color lookup if palette is a dict
        effective_labels = [item[0] for item in raw_plot_data]

        # Check if user provided custom labels via x_label (vertical) or y_label (horizontal)
        labels_override = None
        if vert:
             if x_label is not None and isinstance(x_label, (list, tuple, np.ndarray)) and not isinstance(x_label, str):
                 if len(x_label) == n_groups:
                     labels_override = x_label
        else:
             if y_label is not None and isinstance(y_label, (list, tuple, np.ndarray)) and not isinstance(y_label, str):
                 if len(y_label) == n_groups:
                     labels_override = y_label

        if labels_override is not None:
            effective_labels = labels_override

        if palette is None:
            colors = ["#4C78A8"] * n_groups
        elif isinstance(palette, dict):
            colors = []
            for lbl in effective_labels:
                # Try exact match first, then string match
                if lbl in palette:
                    colors.append(palette[lbl])
                elif str(lbl) in palette:
                    colors.append(palette[str(lbl)])
                else:
                    colors.append("#4C78A8") # Fallback color
        elif isinstance(palette, str):
             # If single color string
            colors = [palette] * n_groups
        else:
            # List of colors
            colors = list(itertools.islice(itertools.cycle(palette), n_groups))

        for i, (label, vals) in enumerate(raw_plot_data):
            plot_items.append({
                'values': vals,
                'mask': raw_masks[i],
                'position': float(i),
                'width': width_violin,
                'color': colors[i],
                'label': label
            })
            xtick_positions.append(i)
            xtick_labels.append(label)

    # --- 2. Calculate Total Points for Rasterization ---
    total_points = sum(len(item['values']) for item in plot_items)
    rasterize_scatter = False
    if raster_threshold is not None and total_points > raster_threshold:
        rasterize_scatter = True

    # --- 3. Setup Figure/Axes ---
    # Only a figure created here is laid out; a caller's figure keeps its own layout
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='tight')
    else:
        fig = ax.figure

    # 'mad' (documented) is the median/MAD robust z-score of remove_outliers
    if outlier_method == 'mad':
        outlier_method = 'robust_zscore'

    # --- 4. Plotting Loop ---
    for item in plot_items:
        vals = item['values']
        pos = item['position']
        col = item['color']
        width = item['width']

        if len(vals) == 0:
            # Empty or all-NaN group: keep its tick, draw nothing
            continue

        # A. Violin
        # Filter outliers for the KDE estimation if requested
        if outlier_method is not None:
            violin_vals = remove_outliers(vals, method=outlier_method, threshold=outlier_multiplier)
            if len(violin_vals) == 0:
                # Fallback if everything is filtered (should be rare)
                violin_vals = vals
        else:
            violin_vals = vals

        parts = ax.violinplot(violin_vals, positions=[pos], widths=width,
                              showextrema=False, orientation='vertical' if vert else 'horizontal')

        for pc in parts["bodies"]:
            pc.set_facecolor(col)
            pc.set_edgecolor(col)
            pc.set_alpha(alpha_violin)

            # Clipping to half
            # get_paths()[0].vertices is (N, 2) array of (x, y)
            verts = pc.get_paths()[0].vertices

            if vert:
                # Vertical: x is dim 0. Center is 'pos'.
                mean_val = verts[:, 0].mean()
                verts[:, 0] = np.clip(verts[:, 0], None, mean_val)
            else:
                # Horizontal: y is dim 1. Center is 'pos'.
                mean_val = verts[:, 1].mean()
                verts[:, 1] = np.clip(verts[:, 1], mean_val, None)

        # B. Scatter (Rain)
        rng = np.random.default_rng(0)
        # Scale jitter and offset by width ratio relative to 0.5 (default)
        scale_factor = width / 0.5
        jitter_vals = rng.normal(loc=0, scale=0.04 * scale_factor, size=len(vals))
        offset = offset_scatter * scale_factor

        if vert:
            x_scatter = pos + offset + jitter_vals
            y_scatter = vals
        else:
            x_scatter = vals
            y_scatter = pos - offset + jitter_vals

        mask = item.get('mask', None)
        if mask is not None and len(mask) == len(vals):
            mask = np.asarray(mask, dtype=bool)
            # Plot non-highlighted points
            ax.scatter(x_scatter[~mask], y_scatter[~mask], color=col, alpha=alpha_scatter,
                       edgecolor="black", linewidth=linewidth_scatter, s=size_scatter,
                       rasterized=rasterize_scatter)
            # Plot highlighted points on top
            ax.scatter(x_scatter[mask], y_scatter[mask], color=highlight_color, alpha=alpha_scatter,
                       edgecolor="black", linewidth=linewidth_scatter, s=size_scatter,
                       rasterized=rasterize_scatter, zorder=3)
        else:
            ax.scatter(x_scatter, y_scatter, color=col, alpha=alpha_scatter,
                       edgecolor="black", linewidth=linewidth_scatter, s=size_scatter,
                       rasterized=rasterize_scatter)

        # C. Boxplot Elements (Median + IQR)
        q1, med, q3 = np.percentile(vals, [25, 50, 75])

        if vert:
            ax.vlines(pos, q1, q3, color="k", linewidth=linewidth_boxplot, zorder=2)
            ax.scatter(pos, med, color="white", edgecolor="k", linewidth=2, s=size_median, zorder=3)
        else:
            ax.hlines(pos, q1, q3, color="k", linewidth=linewidth_boxplot, zorder=2)
            ax.scatter(med, pos, color="white", edgecolor="k", linewidth=2, s=size_median, zorder=3)

    # --- 5. Cosmetics ---
    if vert:
        if x_label is not None and isinstance(x_label, (list, tuple, np.ndarray)) and not isinstance(x_label, str):
            if len(x_label) == len(xtick_labels):
                xtick_labels = x_label
                x_label = None
            else:
                print(f"Warning: x_label list length ({len(x_label)}) does not match number of groups ({len(xtick_labels)}). Using as axis label.")
    else:
        if y_label is not None and isinstance(y_label, (list, tuple, np.ndarray)) and not isinstance(y_label, str):
            if len(y_label) == len(xtick_labels):
                xtick_labels = y_label
                y_label = None
            else:
                print(f"Warning: y_label list length ({len(y_label)}) does not match number of groups ({len(xtick_labels)}). Using as axis label.")

    if vert:
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)
    else:
        ax.set_yticks(xtick_positions)
        ax.set_yticklabels(xtick_labels)
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    return {"fig": fig, "axes": ax}


# Function to adjust colormaps
def colormap_maker(colors, positions=None, cmap_name=None):
    """
    Creates a new custom colormap from a list of colors and optional positions.

    If `cmap_name` is provided, the colormap is registered with Matplotlib and
    can be accessed via `plt.get_cmap(cmap_name)` or `cmap=cmap_name` in plotting functions.

    Parameters
    ----------
    colors : list
        A list of colors in any format accepted by Matplotlib (e.g., 'k', 'cyan',
        (1.0, 1.0, 1.0, 0.5), '#FF5733').
    positions : tuple or list of floats, optional
        A sequence of floats from 0.0 to 1.0 indicating the position of each color.
        If not provided, the colors will be equally spaced.
    cmap_name : str, optional
        The name to register the colormap under in Matplotlib. If None, the colormap
        is not registered.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The generated colormap object.

    Examples
    --------
    >>> # Create and register a 'cyberpunk' colormap
    >>> cmap = colormap_maker(['#08041c', '#390b5e', '#a2217c', '#f04e4c', '#fce205'], cmap_name='cyberpunk')
    >>> plt.imshow(data, cmap='cyberpunk')
    """
    if positions is not None:
        if len(positions) != len(colors):
            raise ValueError("The number of positions must match the number of colors.")
        if positions[0] != 0.0 or positions[-1] != 1.0:
            raise ValueError("Positions must start with 0.0 and end with 1.0.")
        if not all(positions[i] < positions[i+1] for i in range(len(positions)-1)):
            raise ValueError("Positions must be strictly monotonically increasing.")

        # LinearSegmentedColormap.from_list accepts a list of (value, color) tuples
        color_data = list(zip(positions, colors))
    else:
        color_data = colors

    # Generate a default name for internal creation if None is provided
    internal_name = cmap_name if cmap_name else "custom_cmap"

    cmap = LinearSegmentedColormap.from_list(internal_name, color_data)

    if cmap_name is not None:
        # Register the colormap
        # In newer Matplotlib versions, plt.colormaps.register or mpl.colormaps.register is preferred
        # but plt.register_cmap is the classic way. Let's use mpl_colormaps for compatibility.
        try:
            mpl_colormaps.register(cmap, name=cmap_name, force=True)
        except AttributeError:
            # Fallback for older Matplotlib versions
            plt.register_cmap(name=cmap_name, cmap=cmap)

    return cmap

def adjust_colormap(cmap_name, start_ = 0.25, end_ = 1.0, start_color = [0.9, 0.9, 0.9, 1]):
    '''
        # Use colormap from 25% to 100% (just the color part)
        # start_color = [0.9, 0.9, 0.9, 1] # gray
    '''
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(start_, end_, 256))  # Use colormap from 25% to 100% (just the color part)
    colors = np.vstack((start_color, colors))  # Adding faint gray for underflow
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f"{cmap_name}_adjusted", colors)
    return new_cmap

def plotly_scatter_3d(data=None, x=None, y=None, z=None, color=None, labels=None,
                      marker_size=5, opacity=0.8, colorscale='Purples',
                      initial_view=None, **kwargs):
    """
    Creates an interactive 3D scatter plot using Plotly.

    Parameters
    ----------
    data : pd.DataFrame or dict, optional
        Data source. If provided, `x`, `y`, `z`, `color`, and `labels` can be column names/keys.
    x, y, z : str or array-like
        Coordinates. Can be column names (if `data` is provided) or arrays.
    color : str or array-like, optional
        Color values. Can be a column name or an array.
    labels : str or array-like, optional
        Labels for hover text. Can be a column name or an array.
    marker_size : float or array-like, default 5
        Size of the markers.
    opacity : float, default 0.8
        Opacity of the markers.
    colorscale : str, default 'Purples'
        Name of the colorscale to use.
    initial_view : dict, optional
        Dictionary to set the initial camera view (e.g., `dict(eye=dict(x=1.5, y=1.5, z=1.5))`).
    **kwargs :
        Additional keyword arguments passed to `go.Scatter3d`.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError("Plotly is required for this function. Install it with `pip install .[plotting]`") from e

    # Helper to resolve input to array
    def resolve_input(val, data_obj):
        if val is None:
            return None
        if isinstance(val, str) and data_obj is not None:
            if isinstance(data_obj, pd.DataFrame):
                return data_obj[val]
            elif isinstance(data_obj, dict):
                return data_obj[val]
        return val

    # Resolve inputs
    x_data = resolve_input(x, data)
    y_data = resolve_input(y, data)
    z_data = resolve_input(z, data)
    c_data = resolve_input(color, data)
    l_data = resolve_input(labels, data)

    if x_data is None or y_data is None or z_data is None:
        raise ValueError("x, y, and z coordinates must be provided.")

    # Prepare marker dict
    marker = dict(
        size=marker_size,
        opacity=opacity,
        colorscale=colorscale,
    )

    # Handle color
    if c_data is not None:
        if isinstance(c_data, (list, tuple)):
            # is_numeric_dtype(list) is always False: test the converted values
            c_data = pd.Series(c_data)
        if not pd.api.types.is_numeric_dtype(c_data):
             try:
                # Ensure it's a Series or convert
                c_series = pd.Series(c_data)
                c_codes = c_series.astype('category').cat.codes
                marker['color'] = c_codes
             except Exception as e:
                print(f"Warning: Failed to convert non-numeric color data to codes: {e}")
                marker['color'] = c_data
        else:
            marker['color'] = c_data

        marker['showscale'] = True
        marker['colorbar'] = dict(title='Value')

    # Construct hover template
    hovertemplate = (
        '<b>X:</b> %{x:.2f}<br>'
        '<b>Y:</b> %{y:.2f}<br>'
        '<b>Z:</b> %{z:.2f}<br>'
    )
    if l_data is not None:
        hovertemplate = '<b>Label:</b> %{text}<br>' + hovertemplate

    if c_data is not None:
        hovertemplate += '<b>Color:</b> %{marker.color:.2f}<br>'

    hovertemplate += '<extra></extra>'

    # Create Scatter3d trace
    trace = go.Scatter3d(
        x=x_data,
        y=y_data,
        z=z_data,
        mode='markers',
        marker=marker,
        text=l_data,
        hovertemplate=hovertemplate,
        **kwargs
    )

    fig = go.Figure(data=[trace])

    # Update layout
    scene = dict(
        xaxis_title=x if isinstance(x, str) else 'X',
        yaxis_title=y if isinstance(y, str) else 'Y',
        zaxis_title=z if isinstance(z, str) else 'Z',
        aspectmode='data'
    )

    if initial_view:
        scene['camera'] = initial_view

    fig.update_layout(
        scene=scene,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def plotly_scatter_3d_from_adata_obsm(adata, obsm_key, color_key=None, label_key=None, **kwargs):
    """
    Creates an interactive 3D scatter plot from an AnnData object's obsm slot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obsm_key : str
        Key for `adata.obsm` to use as coordinates (e.g., 'X_umap', 'X_pca').
        Expects at least 3 dimensions.
    color_key : str, optional
        Key for `adata.obs` to use for coloring.
    label_key : str, optional
        Key for `adata.obs` to use for labels.
    **kwargs :
        Additional keyword arguments passed to `plotly_scatter_3d`.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """
    if obsm_key not in adata.obsm:
        raise ValueError(f"'{obsm_key}' not found in adata.obsm.")

    coords = adata.obsm[obsm_key]

    if coords.shape[1] < 3:
        # Warn or handle gracefully? User asked for 3D.
        # We'll pad with zeros if 2D, but warn.
        print(f"Warning: '{obsm_key}' has only {coords.shape[1]} dimensions. Padding with zeros for Z-axis.")
        z = np.zeros(coords.shape[0])
        x = coords[:, 0]
        y = coords[:, 1]
    else:
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

    # Extract color and labels
    color = None
    if color_key:
        if color_key in adata.obs:
            color = adata.obs[color_key]
        else:
            print(f"Warning: '{color_key}' not found in adata.obs.")

    labels = None
    if label_key:
        if label_key in adata.obs:
            labels = adata.obs[label_key]
        else:
            print(f"Warning: '{label_key}' not found in adata.obs.")

    # Set default axis titles if not provided in kwargs (via passing x,y,z names)
    # We pass arrays directly, so plotly_scatter_3d won't see names unless we pass them or update layout.
    # To get nice axis titles, we can manually update the figure afterwards or update the scene dict inside.
    # But plotly_scatter_3d infers titles from string args. Here we pass arrays.

    fig = plotly_scatter_3d(x=x, y=y, z=z, color=color, labels=labels, **kwargs)

    # Update axis titles to reflect the keys
    fig.update_layout(scene=dict(
        xaxis_title=f"{obsm_key}_0",
        yaxis_title=f"{obsm_key}_1",
        zaxis_title=f"{obsm_key}_2" if coords.shape[1] >= 3 else "Zero",
    ))

    return fig

def get_nice_number(value):
    """
    Rounds a number down to one significant digit, a 'nice' round number for legends.
    Examples: 4343 -> 4000, 10858 -> 10000, 21717 -> 20000, 0.6 -> 0.6

    Never rounds up, so the result does not exceed `value`. Values of 1 or more
    return an int, smaller values a float.
    """
    if value == 0: return 0

    sign = -1 if value < 0 else 1
    magnitude = abs(value)

    # Calculate magnitude
    exponent = math.floor(math.log10(magnitude))
    # The small tolerance absorbs binary rounding (0.6 / 0.1 == 5.999...)
    leading_digit = math.floor(magnitude / (10.0 ** exponent) * (1 + 1e-12))
    if leading_digit >= 10:
        leading_digit, exponent = 1, exponent + 1
    elif leading_digit == 0:
        exponent -= 1
        leading_digit = math.floor(magnitude / (10.0 ** exponent) * (1 + 1e-12))

    if exponent >= 0:
        return sign * int(leading_digit * (10 ** exponent))
    return sign * round(leading_digit * 10.0 ** exponent, -exponent)


def savefig_svg(filename, bgnd_color=(1, 1, 1, 0.8), bbox_inches='tight', dpi=300, pad_inches=0.1, scatter_raster_threshold=1e3, editable_text=True, **kwargs):
    """
    Saves the currently active matplotlib figure as an SVG file.

    Parameters
    ----------
    filename : str or Path
        The path where the SVG will be saved. The '.svg' extension is appended if missing.
    bgnd_color : color, default (1, 1, 1, 0.8)
        The background color (facecolor) of the saved figure.
    bbox_inches : str or Bbox, default 'tight'
        Bounding box in inches: only the given portion of the figure is saved.
    dpi : float, default 300
        The resolution in dots per inch for rasterized elements.
    pad_inches : float, default 0.1
        Amount of padding around the figure when bbox_inches is 'tight'.
    scatter_raster_threshold : int, optional
        If provided, any scatter plot (PathCollection) in the figure containing
        this number of points or more will be rasterized before saving. This
        helps keep SVG file sizes manageable for large datasets.
    editable_text : bool, default True
        If True, exports SVGs with editable text by temporarily setting
        `svg.fonttype` to 'none' and `axes.unicode_minus` to False.
    **kwargs :
        Additional keyword arguments passed directly to `plt.savefig`.
    """
    fig = plt.gcf()

    # Extract figure title for metadata
    title = fig._suptitle.get_text() if fig._suptitle else str(filename)

    # Initialize metadata with predefined options
    # Copy: the caller's dict must not keep this figure's title/date
    metadata = dict(kwargs.pop('metadata', None) or {})
    metadata.setdefault('Creator', 'eigenp')
    metadata.setdefault('Date', datetime.datetime.now().isoformat())
    metadata.setdefault('Title', title)

    # Rasterize scatter collections if threshold is met
    if scatter_raster_threshold is not None:
        from matplotlib.collections import PathCollection
        for ax in fig.axes:
            for collection in ax.collections:
                if isinstance(collection, PathCollection):
                    # Check the number of offsets (points) in the scatter
                    if len(collection.get_offsets()) >= scatter_raster_threshold:
                        collection.set_rasterized(True)

    # Make sure we add '.svg' if it's not present
    filename_str = str(filename)
    if not filename_str.lower().endswith('.svg'):
        filename_str += '.svg'

    # Context manager for editable text settings
    context_rc = {}
    if editable_text:
        context_rc['svg.fonttype'] = 'none'
        context_rc['axes.unicode_minus'] = False

    # Call savefig on the current figure with the context
    with plt.rc_context(context_rc):
        fig.savefig(
            filename_str,
            format='svg',
            facecolor=bgnd_color,
            bbox_inches=bbox_inches,
            dpi=dpi,
            pad_inches=pad_inches,
            metadata=metadata,
            **kwargs
        )
    print(f"Saved figure to {filename_str}")

def brightness_diagnostic_plotter(diagnostic_data, ax=None):
    """
    Plots the diagnostic data returned by `adjust_brightness_per_slice` when `return_diagnostic=True`.

    Args:
        diagnostic_data (dict): The diagnostic data containing `x_data`, `y_data_norm`, `y_fit_norm`, and `gamma_fit_func`.
        ax (matplotlib.axes.Axes, optional): Axes to draw on. If None, a new figure is created.

    Returns:
        matplotlib.figure.Figure: The figure containing the plot.
    """
    if diagnostic_data is None:
        return None

    x_data = diagnostic_data['x_data']
    y_data_norm = diagnostic_data['y_data_norm']
    y_fit_norm = diagnostic_data['y_fit_norm']
    gamma_fit_func = diagnostic_data['gamma_fit_func']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), layout='tight')
    else:
        fig = ax.figure
    ax.plot(x_data, y_data_norm, 'o', label='Raw 99th Percentile (Normalized)', alpha=0.7)
    ax.plot(x_data, y_fit_norm, '-', label=f'Fitted Curve ({gamma_fit_func})', linewidth=2)
    ax.set_xlabel('Z-slice index')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Z-axis Intensity Decay Fit')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    return fig
