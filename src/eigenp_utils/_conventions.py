# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Shared conventions of eigenp_utils: units, pixel sizes, display ranges, channel colours.

One place for the rules every viewer, reader and processing function follows
(decisions D01-D04 of the 2026-11 review):

- **Units (D01).** µm is the canonical length unit. Pixel sizes are parsed by
  :func:`resolve_pixel_sizes` (dict / sequence / scalar / None; "unknown" is
  None) and converted to µm with :func:`to_um`. Scale-bar labels pick nm, µm or
  mm with :func:`format_length`.
- **Display range (D02).** :func:`default_display_range` is the default
  ``(vmin, vmax)`` of every viewer, computed on the source volume:
  vmin = 0 (or the data minimum if negative), vmax = the 99.99th percentile of
  a fixed-seed sample (:func:`sample_values`), the maximum when that is
  degenerate. Processing functions use :func:`robust_range` (default
  (0.1, 99.9) percentiles) with the same degenerate fallback.
- **Colours (D03).** One default :data:`PALETTE`, cycled. One resolver,
  :func:`channel_color_hex`, and :func:`channel_colormap` for compositors.
- **Compositing (D04).** :func:`resolve_blending` is the single rule for
  ``blending=None``.
- :func:`per_channel` broadcasts per-channel parameters;
  :func:`warn_deprecated` emits the package's deprecation warnings.

WASM-safe: imports only numpy and the standard library at module level;
matplotlib is imported inside the colour functions.
"""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np

__all__ = [
    "UM",
    "PALETTE",
    "PALETTE_MINIMUM",
    "BLENDING_MODES",
    "to_um",
    "is_length_unit",
    "format_length",
    "resolve_pixel_sizes",
    "sample_values",
    "default_display_range",
    "robust_range",
    "is_colormap_name",
    "channel_spec",
    "channel_color_hex",
    "channel_colormap",
    "starts_at_white",
    "default_colors",
    "resolve_blending",
    "per_channel",
    "warn_deprecated",
]

# ---------------------------------------------------------------------------
# Units (D01)
# ---------------------------------------------------------------------------

UM = "µm"

# Length units -> factor to µm, keyed by the lower-case spellings ImageJ, OME,
# TIFF and users write. The single unit table of the package.
_UNIT_TO_UM = {
    "µm": 1.0, "μm": 1.0, "um": 1.0, "micron": 1.0, "microns": 1.0,
    "micrometer": 1.0, "micrometre": 1.0, "micrometers": 1.0, "micrometres": 1.0,
    "nm": 1e-3, "nanometer": 1e-3, "nanometre": 1e-3, "nanometers": 1e-3, "nanometres": 1e-3,
    "å": 1e-4, "angstrom": 1e-4, "angstroms": 1e-4,
    "mm": 1e3, "millimeter": 1e3, "millimetre": 1e3, "millimeters": 1e3, "millimetres": 1e3,
    "cm": 1e4, "centimeter": 1e4, "centimetre": 1e4, "centimeters": 1e4, "centimetres": 1e4,
    "m": 1e6, "meter": 1e6, "metre": 1e6, "meters": 1e6, "metres": 1e6,
    "inch": 25400.0, "in": 25400.0, "inches": 25400.0,
}
_INCH_SPELLINGS = ("inch", "in", "inches")


def _unit_key(unit):
    """Lower-case lookup key of a unit name (ImageJ's escaped micro sign unescaped), or None."""
    if unit is None:
        return None
    key = str(unit).strip().replace("\\u00B5", "µ").replace("\\u00b5", "µ")
    # 'Mm' (megametre) is not a microscopy unit; 'MM' etc. are millimetres
    return key.lower() or None


def is_length_unit(unit) -> bool:
    """True if ``unit`` is a known length unit (any spelling of nm, µm, mm, cm, m, Å, inch)."""
    return _unit_key(unit) in _UNIT_TO_UM


def _is_um_spelling(unit) -> bool:
    return _UNIT_TO_UM.get(_unit_key(unit)) == 1.0


def to_um(value, unit, *, stacklevel=2):
    """Convert a length (scalar or array) given in ``unit`` to µm.

    ``unit`` None means µm (the package convention). Known units: nm, µm (um,
    micron, ...), mm, cm, m, Å, and inch, which warns because inch values in
    image files usually come from a print resolution (dpi), not a calibration.
    An unknown unit raises ValueError.
    """
    if unit is None:
        return value
    key = _unit_key(unit)
    factor = _UNIT_TO_UM.get(key)
    if factor is None:
        raise ValueError(f"unknown length unit {unit!r}; expected nm, µm, mm, cm, m or inch")
    if key in _INCH_SPELLINGS:
        warnings.warn(
            "pixel size given in inches; inch values usually come from a print resolution (dpi), "
            "not a microscope calibration. Converted to µm; pass pixel_sizes= to override.",
            UserWarning, stacklevel=stacklevel + 1,
        )
    if factor == 1.0:
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(None if v is None else v * factor for v in value)
    return value * factor


def format_length(value_um):
    """A length in µm as ``(value, unit)`` for a label, with nm / µm / mm picked by magnitude.

    Below 1 µm the value is given in nm, from 1000 µm on in mm, else in µm
    (e.g. 0.5 -> (500.0, 'nm'), 20 -> (20.0, 'µm'), 2000 -> (2.0, 'mm')). The
    underlying numbers stay in µm; only the label changes.
    """
    v = float(value_um)
    if not math.isfinite(v) or v == 0:
        return v, UM
    if abs(v) < 1.0:
        return v * 1e3, "nm"
    if abs(v) >= 1e3:
        return v / 1e3, "mm"
    return v, UM


# ---------------------------------------------------------------------------
# Pixel sizes (D01)
# ---------------------------------------------------------------------------

def _as_float(v, axis):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        if hasattr(v, "size") and getattr(v, "size", 0) == 1 and hasattr(v, "item"):
            return float(v.item())
        raise ValueError(f"pixel_sizes[{axis!r}] must be a number or None, got {v!r}") from None


def resolve_pixel_sizes(ps, ndim=3, *, axes="zyx", fill_z=True, warn=True, stacklevel=2):
    """Parse pixel sizes into ``(values, calibrated, unit)``: the package's one pixel-size parser.

    Parameters
    ----------
    ps : dict, sequence, scalar or None
        ``{'Z', 'Y', 'X', 'unit'}`` dict (keys case-insensitive; other keys are
        ignored; missing or None entries mean unknown), a sequence with one
        entry per axis (a full ``axes`` sequence, e.g. (Z, Y, X), is also
        accepted for fewer dimensions: the trailing entries are used), a
        scalar (isotropic), or None (nothing known).
    ndim : int
        Number of trailing letters of ``axes`` to return.
    axes : str
        Axis letters of a full entry (default 'zyx').
    fill_z : bool
        A missing Z while every in-plane (Y, X) size is known takes the finest
        of them (isotropic voxels assumed), with a UserWarning naming axis Z.
        False leaves it None.
    warn : bool
        Emit the fill warning (and the inch warning of :func:`to_um`).

    Returns
    -------
    values : tuple of float or None
        Sizes in µm (None where unknown), in the order of ``axes[-ndim:]``.
    calibrated : tuple of bool
        Whether each size was given (False for unknown or filled-in axes).
    unit : str
        'µm' for known length units and for no unit; an unknown unit (e.g.
        'pixel') is returned as given, with the values unconverted.

    Raises
    ------
    ValueError
        Wrong number of entries, a non-numeric entry, or a size that is not
        finite and > 0.
    """
    letters = axes.upper()
    if not 1 <= ndim <= len(letters):
        raise ValueError(f"ndim must be between 1 and {len(letters)} for axes {axes!r}, got {ndim}")
    out_axes = letters[-ndim:]
    unit = None
    if ps is None:
        raw = [None] * ndim
    elif isinstance(ps, dict):
        keyed = {str(k).upper(): v for k, v in ps.items()}
        unit = keyed.get("UNIT")
        raw = [keyed.get(a) for a in out_axes]
    elif isinstance(ps, (str, bytes)):
        raise ValueError(f"pixel_sizes must be a dict, a sequence or a number, got {ps!r}")
    elif isinstance(ps, numbers.Number) or (hasattr(ps, "ndim") and getattr(ps, "ndim") == 0):
        raw = [ps] * ndim
    else:
        try:
            entries = list(ps)
        except TypeError:
            raise ValueError(f"pixel_sizes must be a dict, a sequence or a number, got {ps!r}") from None
        if len(entries) == len(letters) and len(entries) != ndim:
            entries = entries[-ndim:]
        if len(entries) != ndim:
            raise ValueError(
                f"pixel_sizes sequence has {len(entries)} entries, expected {ndim} ({', '.join(out_axes)})."
            )
        raw = entries
    values = [_as_float(v, a) for a, v in zip(out_axes, raw)]
    for a, v in zip(out_axes, values):
        if v is not None and not (math.isfinite(v) and v > 0):
            raise ValueError(f"pixel_sizes[{a!r}] must be finite and positive, got {v!r}.")
    calibrated = [v is not None for v in values]

    out_unit = UM
    if isinstance(unit, str) and not unit.strip():
        unit = None
    if unit is not None:
        if is_length_unit(unit):
            if warn and any(v is not None for v in values):
                to_um(1.0, unit, stacklevel=stacklevel + 1)  # the inch warning, once
            factor = _UNIT_TO_UM[_unit_key(unit)]
            values = [None if v is None else v * factor for v in values]
        else:
            out_unit = str(unit).strip()

    if fill_z and "Z" in out_axes:
        iz = out_axes.index("Z")
        plane = [i for i, a in enumerate(out_axes) if a in "YX"]
        if values[iz] is None and plane and all(values[i] is not None for i in plane):
            values[iz] = min(values[i] for i in plane)
            if warn:
                shown = f"{values[iz]:g} {out_unit}"
                warnings.warn(
                    f"No value for axis Z in pixel_sizes; using the finest X/Y pixel size {shown} "
                    "(isotropic voxels assumed).",
                    UserWarning, stacklevel=stacklevel + 1,
                )
    return tuple(values), tuple(calibrated), out_unit


# ---------------------------------------------------------------------------
# Display ranges (D02)
# ---------------------------------------------------------------------------

def _unwrap(a):
    """The array behind an xarray.DataArray (lazy dask data stays lazy)."""
    if hasattr(a, "dims") and hasattr(a, "data") and hasattr(a, "coords"):
        return a.data
    return a


def _is_lazy(a):
    """A dask-like chunked array (has ``blocks`` and ``numblocks``)."""
    return hasattr(a, "blocks") and hasattr(a, "numblocks") and hasattr(a, "compute")


def _finite(flat):
    if np.issubdtype(flat.dtype, np.floating) or np.issubdtype(flat.dtype, np.complexfloating):
        flat = flat[np.isfinite(flat)]
    return flat


def _lazy_sample(a, n, rng):
    """Values of randomly chosen chunks of a dask array, about ``n`` of them (no full compute)."""
    nb = tuple(a.numblocks)
    total = int(np.prod(nb)) if nb else 1
    per_block = max(1, int(a.size // max(total, 1)))
    want = max(1, min(total, -(-n // per_block)))
    picks = rng.permutation(total)[:want]
    parts = [np.asarray(a.blocks[np.unravel_index(int(p), nb)].compute()) for p in picks]
    flat = np.concatenate([np.asarray(p).reshape(-1) for p in parts]) if parts else np.empty(0)
    if flat.size > n:
        flat = flat[rng.integers(0, flat.size, n)]
    return flat


def sample_values(a, n=1_000_000, seed=0):
    """A fixed-seed random sample of at most ``n`` finite values of ``a`` (1D array).

    Arrays with at most ``n`` values are returned whole (finite values only),
    so small images give exact statistics. Larger ones are sampled at random
    indices with ``numpy.random.default_rng(seed)`` (deterministic; a strided
    sample can alias with image rows or columns). Booleans become uint8.
    An xarray.DataArray is unwrapped; dask arrays are sampled chunk-wise
    (a fixed-seed choice of whole chunks, then of values) without computing the
    whole array.
    """
    a = _unwrap(a)
    rng = np.random.default_rng(seed)
    if _is_lazy(a):
        flat = _lazy_sample(a, n, rng)
    else:
        arr = np.asarray(a)
        if arr.size > n:
            idx = rng.integers(0, arr.size, n)
            flat = arr[np.unravel_index(idx, arr.shape)] if arr.ndim > 1 else arr.reshape(-1)[idx]
        else:
            flat = arr.reshape(-1)
    flat = np.asarray(flat)
    if flat.dtype == bool:
        flat = flat.astype(np.uint8)
    return _finite(flat)


def _finite_extremes(a, sample):
    """(min, max) of the finite values: of the whole array for in-memory data, else of the sample."""
    a = _unwrap(a)
    if not _is_lazy(a):
        arr = np.asarray(a)
        if arr.dtype == bool:
            arr = arr.view(np.uint8)
        if arr.size == 0:
            return None, None
        if np.issubdtype(arr.dtype, np.floating):
            finite = np.isfinite(arr)
            if finite.all():
                return float(arr.min()), float(arr.max())
            arr = arr[finite]
            if arr.size == 0:
                return None, None
        return float(arr.min()), float(arr.max())
    if sample.size == 0:
        return None, None
    return float(sample.min()), float(sample.max())


def _is_integer_dtype(a):
    dt = getattr(_unwrap(a), "dtype", None)
    return dt is not None and np.issubdtype(dt, np.integer)


def _robust_max(a, q=99.99, n=1_000_000, seed=0):
    """The ``q``-th percentile of the finite values (fixed-seed sample), the maximum when degenerate.

    Degenerate: the percentile does not exceed the sample minimum (signal on
    fewer than (100 - q)% of the values). Integer data give an integer
    (rounded up). None when there is no finite value.
    """
    sample = sample_values(a, n, seed)
    _, hi = _finite_extremes(a, sample)
    if hi is None:
        return None
    if sample.size == 0:
        return hi
    v = float(np.percentile(sample, q))
    if not v > float(sample.min()):
        return hi
    if _is_integer_dtype(a):
        v = float(np.ceil(v))
    return min(v, hi)


def default_display_range(volume, q=99.99, n=1_000_000, seed=0):
    """Default display range ``(vmin, vmax)`` of a channel, computed on its whole (source) volume.

    - vmin = 0 (fluorescence convention), or the data minimum if the data go
      below 0, so signed data are not clipped.
    - vmax = the 99.99th percentile of a fixed-seed sample of at most ``n``
      finite values (:func:`sample_values`), rounded up for integer data; the
      finite maximum when the percentile does not exceed the sample minimum
      (signal on fewer than 0.01% of the voxels). q = 99.99 tolerates hot
      pixels up to 0.01% of the voxels while saturating little of a sparse
      channel.
    - Always vmax > vmin: constant data give vmax = vmin + 1; no finite value
      gives (0.0, 1.0); boolean data give (0.0, 1.0).

    Pass the source volume, not projected panels: on a maximum projection a
    single hot pixel covers far more than 0.01% of the pixels.
    """
    a = _unwrap(volume)
    if getattr(a, "dtype", None) == bool:
        return 0.0, 1.0
    sample = sample_values(a, n, seed)
    lo, _ = _finite_extremes(a, sample)
    if lo is None:
        return 0.0, 1.0
    vmin = min(0.0, lo)
    vmax = _robust_max(a, q, n, seed)
    if vmax is None or not vmax > vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def robust_range(a, pmin=0.1, pmax=99.9, *, n=1_000_000, seed=0, stacklevel=2):
    """``(lo, hi)`` percentiles of the finite values, for processing (e.g. rescaling to 8 bits).

    Percentiles of a fixed-seed sample of at most ``n`` values
    (:func:`sample_values`; exact for smaller arrays). When the range is
    degenerate (``hi <= lo``) although the data are not constant, which happens
    when the signal covers fewer than (100 - pmax)% of the values, ``hi``
    falls back to the finite maximum (and ``lo`` to the finite minimum if that
    is still degenerate), with a UserWarning, instead of blanking the image.
    Constant data return ``(v, v)`` without a warning; no finite value gives
    (0.0, 1.0).
    """
    sample = sample_values(a, n, seed)
    if sample.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(sample, [pmin, pmax]))
    if hi <= lo:
        mn, mx = _finite_extremes(a, sample)
        lo, hi = _degenerate_fallback(lo, hi, mn, mx, pmin, pmax, stacklevel=stacklevel + 1)
    return lo, hi


def _degenerate_fallback(lo, hi, mn, mx, pmin, pmax, stacklevel=2):
    """Shared rule for a degenerate percentile range ``hi <= lo`` of non-constant data.

    ``hi`` becomes the finite maximum ``mx`` (``lo`` the minimum ``mn`` if
    that is still degenerate), with a UserWarning. Constant data (``mx <=
    mn``) are returned unchanged and silently. Values may be data units or
    integer codes; they are returned as given.
    """
    if hi > lo or mx is None or not mx > mn:
        return lo, hi
    if mx > lo:
        hi = mx
    else:
        lo, hi = mn, mx
    warnings.warn(
        f"The ({pmin}, {pmax}) percentiles of the data are equal (signal on fewer than "
        f"{100 - pmax:g}% of the values); using the range [{lo:g}, {hi:g}] instead.",
        UserWarning, stacklevel=stacklevel + 1,
    )
    return lo, hi


# ---------------------------------------------------------------------------
# Colours (D03) and compositing rule (D04)
# ---------------------------------------------------------------------------

#: Default channel colours of every viewer, cycled for more channels.
PALETTE = ["white", "lime", "magenta", "yellow", "cyan", "red", "blue"]
#: Defaults for ``blending='minimum'``: complements of PALETTE ("inks" on white paper).
PALETTE_MINIMUM = ["black", "magenta", "lime", "blue", "red", "cyan", "yellow"]

BLENDING_MODES = ("additive", "maximum", "screen", "minimum")
_BLENDING_ALIASES = {"add": "additive", "max": "maximum", "min": "minimum"}

# CSS colour names that are also colormap names; the colormap wins. 'gray' and
# 'grey' are intended (both are grey ramps); 'pink' surprises, so it warns.
_SURPRISING_CMAP_NAMES = {"pink"}


def _colormaps():
    import matplotlib

    return matplotlib.colormaps


def is_colormap_name(c) -> bool:
    """True if ``c`` is a string naming a registered matplotlib colormap."""
    if not isinstance(c, str):
        return False
    try:
        _colormaps()[c]
        return True
    except (KeyError, ValueError):
        return False


def _registered_name(cmap):
    """The registered name of a Colormap instance, else TypeError."""
    name = getattr(cmap, "name", None)
    registry = _colormaps()
    if isinstance(name, str):
        try:
            if registry[name] == cmap:
                return name
        except (KeyError, ValueError):
            pass
    raise TypeError(
        f"Expected a registered colormap name or a colour, but got an unregistered Colormap instance "
        f"{name!r}. Register it first (matplotlib.colormaps.register(cmap, name=...)) and pass it or its name."
    )


def channel_spec(c):
    """Canonical channel colour specification: a colormap name or a colour, as a string.

    Strings are returned unchanged (colormap names win over CSS names); a
    registered Colormap instance becomes its registered name; RGB(A) tuples
    and other matplotlib colours become hex. An unregistered Colormap instance
    raises TypeError; anything else that is not a colour raises ValueError.
    """
    from matplotlib import colors as mcolors

    if isinstance(c, mcolors.Colormap):
        return _registered_name(c)
    if isinstance(c, str):
        if is_colormap_name(c) or mcolors.is_color_like(c):
            return c
        raise ValueError(f"{c!r} is neither a colour nor a registered colormap name")
    if isinstance(c, np.ndarray):
        c = tuple(c.tolist())
    if mcolors.is_color_like(c):
        return mcolors.to_hex(c)
    raise ValueError(f"{c!r} is neither a colour nor a registered colormap name")


def channel_color_hex(c, *, strict=True):
    """Hex colour (``'#rrggbb'``) of a channel: a colour, or a colormap's end colour.

    Accepts colormap names (they win over CSS names: 'gray' -> white; 'pink'
    warns because the colormap's end colour is white), CSS names, hex strings,
    RGB(A) tuples and registered Colormap instances (resolved by name). An
    unregistered Colormap instance raises TypeError. Anything else raises
    ValueError, or gives None with ``strict=False``.
    """
    from matplotlib import colors as mcolors

    try:
        spec = channel_spec(c)
    except ValueError:
        if strict:
            raise
        return None
    if is_colormap_name(spec):
        if spec in _SURPRISING_CMAP_NAMES:
            warnings.warn(
                f"{spec!r} is both a colormap and a CSS colour name; using the colormap (end colour "
                f"{mcolors.to_hex(_colormaps()[spec](1.0)[:3])}). Pass {mcolors.to_hex(spec)!r} for the CSS colour.",
                UserWarning, stacklevel=2,
            )
        return mcolors.to_hex(_colormaps()[spec](1.0)[:3])
    return mcolors.to_hex(spec)


def channel_colormap(c, *, background="black"):
    """Colormap of a channel for compositing.

    A colormap name or registered Colormap instance gives that colormap; a
    plain colour gives a linear ramp from ``background`` ('black', or 'white'
    for minimum blending) to the colour.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb

    spec = channel_spec(c)
    if is_colormap_name(spec):
        return _colormaps()[spec]
    if background not in ("black", "white"):
        raise ValueError(f"background must be 'black' or 'white', got {background!r}")
    start = (0.0, 0.0, 0.0) if background == "black" else (1.0, 1.0, 1.0)
    return LinearSegmentedColormap.from_list(f"{background}_to_{spec}", [start, to_rgb(spec)])


def starts_at_white(c, threshold=0.88) -> bool:
    """True if ``c`` names a colormap whose low end is near white (relative luminance > ``threshold``).

    The exported I-series ('I Blue', ...) and matplotlib's light sequential
    maps ('Blues', 'Greys', 'binary', ...) start at white; plain colours never do.
    """
    try:
        spec = channel_spec(c)
    except (TypeError, ValueError):
        return False
    if not is_colormap_name(spec):
        return False
    r, g, b = _colormaps()[spec](0.0)[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > threshold


def _check_blending(blending):
    if blending is None:
        return None
    b = _BLENDING_ALIASES.get(blending, blending)
    if b not in BLENDING_MODES:
        raise ValueError(f"blending must be one of {BLENDING_MODES} (or None), got {blending!r}")
    return b


def resolve_blending(colors, blending=None, *, stacklevel=2):
    """The blending mode for channels coloured by ``colors``; the single rule for ``blending=None``.

    An explicit ``blending`` is validated and returned (aliases 'add', 'max',
    'min' accepted). For None: 'additive' when every channel is a plain
    colour; 'maximum' when any channel is a colormap name (pixel-identical to
    the historical default); 'minimum' when every colormap channel starts at
    white (e.g. the I-series), whose maximum would be white everywhere. A mix
    of white-start and dark-start colormaps keeps 'maximum' and warns.
    """
    b = _check_blending(blending)
    if b is not None:
        return b
    cmaps = []
    for c in colors:
        try:
            spec = channel_spec(c)
        except (TypeError, ValueError):
            continue
        if is_colormap_name(spec):
            cmaps.append(spec)
    if not cmaps:
        return "additive"
    white = [starts_at_white(c) for c in cmaps]
    if all(white):
        return "minimum"
    if any(white):
        warnings.warn(
            f"Colormaps {[c for c, w in zip(cmaps, white) if w]} start at white and "
            f"{[c for c, w in zip(cmaps, white) if not w]} at a dark colour; with the default "
            "'maximum' blending the white-start channels hide the others. Pass blending='minimum' "
            "(white background) or use dark-start colormaps.",
            UserWarning, stacklevel=stacklevel + 1,
        )
    return "maximum"


def default_colors(given, n, blending=None):
    """``given`` channel colours padded to ``n`` with the unused palette colours (cycled).

    ``given`` is None, one colour / colormap (a string, a colour tuple or a
    Colormap), or a sequence of them. Missing entries take the colours of
    :data:`PALETTE` (:data:`PALETTE_MINIMUM` for ``blending='minimum'``) not
    already given; when all are used, the palette cycles.
    """
    from matplotlib import colors as mcolors

    palette = PALETTE_MINIMUM if _check_blending(blending) == "minimum" else PALETTE
    if given is None:
        given = []
    elif isinstance(given, (str, mcolors.Colormap)) or (isinstance(given, tuple) and mcolors.is_color_like(given)):
        given = [given]
    else:
        given = list(given)
    fill = [c for c in palette if not any(c is g or (isinstance(g, str) and c == g) for g in given)] or palette
    return given[:n] + [fill[i % len(fill)] for i in range(max(0, n - len(given)))]


# ---------------------------------------------------------------------------
# Per-channel parameters and deprecations
# ---------------------------------------------------------------------------

def per_channel(value, n, name, default=None):
    """``value`` as a list of ``n`` per-channel values.

    A scalar (or None) applies to every channel. A list or tuple shorter than
    ``n`` is padded with ``default``, and its None entries take ``default``.
    A longer one raises ValueError naming the parameter (``'<name> has 3
    entries for 2 channels'``).
    """
    if isinstance(value, np.ndarray) and value.ndim > 0:
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [default if value is None else value] * n
    if len(value) > n:
        raise ValueError(f"{name} has {len(value)} entries for {n} channels")
    return [default if v is None else v for v in value] + [default] * (n - len(value))


def warn_deprecated(old, new=None, *, removal="0.1.0", stacklevel=3):
    """Emit the package's deprecation warning for ``old``.

    A ``FutureWarning`` (shown to users, unlike DeprecationWarning from library
    code) reading "``old`` is deprecated and will be removed in eigenp_utils
    <removal>; use ``new`` instead." ``stacklevel`` 3 points at the caller of
    the function that calls this helper.
    """
    msg = f"{old} is deprecated and will be removed in eigenp_utils {removal}"
    msg += f"; use {new} instead." if new else "."
    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
