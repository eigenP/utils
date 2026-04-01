import warnings

warnings.warn(
    "The 'tnia_plotting_3d' module is deprecated. Please import plotting functions "
    "directly from 'eigenp_utils.tnia_plotting_anywidgets'.",
    DeprecationWarning,
    stacklevel=2,
)

from .tnia_plotting_anywidgets import (
    is_colormap,
    resolve_color,
    _norm,
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
    "_norm",
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
