"""Deprecated alias of :mod:`eigenp_utils.tnia_plotting_anywidgets`.

Importing it emits a ``FutureWarning``; the module will be removed in
eigenp_utils 0.1.0. Changed (D08, 2026-11): the warning was a hidden
DeprecationWarning without a removal version; it now points at the importing
line.
"""
from ._conventions import warn_deprecated

warn_deprecated(
    "The 'tnia_plotting_3d' module",
    "'eigenp_utils.tnia_plotting_anywidgets' (import the plotting functions from there)",
    stacklevel=3,  # past this shim (importlib frames are skipped): the importing line
)

from .tnia_plotting_anywidgets import (
    is_colormap,
    resolve_color,
    show_zyx_slice,
    show_zyx_max,
    show_zyx_projection,
    show_zyx,
    show_zyx_max_slabs,
    show_zyx_projection_slabs,
    create_multichannel_rgb,
    create_multichannel_rgb_cmap,
    black_to,
    blend_colors,
)

__all__ = [
    "is_colormap",
    "resolve_color",
    "show_zyx_slice",
    "show_zyx_max",
    "show_zyx_projection",
    "show_zyx",
    "show_zyx_max_slabs",
    "show_zyx_projection_slabs",
    "create_multichannel_rgb",
    "create_multichannel_rgb_cmap",
    "black_to",
    "blend_colors",
]
