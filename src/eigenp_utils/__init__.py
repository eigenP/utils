"""Utility functions for eigenp workflows.

The top-level names are resolved lazily (PEP 562): ``import eigenp_utils`` imports
only :mod:`eigenp_utils.plotting_utils` (core dependencies; it registers the
package colormaps by name), and ``eigenp_utils.load_image`` imports
:mod:`eigenp_utils.image_io` on first access. When an optional package is missing, the access raises an
ImportError naming the extra to install (``pip install "eigenp-utils[image-analysis]"``);
any other import error propagates unchanged, so a genuine bug is not hidden.

Deprecated names (``hist_imshow``, ``numpy_to_stczyx_xarray``) stay importable
but are not in ``__all__``. ``from eigenp_utils import *`` imports every module
in ``__all__`` and therefore needs the ``image-analysis`` extra.

Changed (D06, 2026-11): lazy attributes instead of eager ``try: ... except
ImportError: pass`` blocks, which silently dropped a whole group of functions
on any import error. ``__all__`` now lists the image API.
"""

import importlib
import importlib.util

# Imported eagerly (core dependencies only) so that the package's colormaps
# ('labels_cmap', 'labels_cmap_black', the I-series) are registered with
# matplotlib on `import eigenp_utils`, as before the lazy attributes.
from . import plotting_utils as _plotting_utils  # noqa: F401

# module -> extra that provides its optional packages (None: core dependencies only)
_EXTRA_OF_MODULE = {
    "_conventions": None,
    "plotting_utils": None,
    "tnia_plotting_anywidgets": None,
    "label_annotator": None,
    "spline_utils": None,
    "clahe_equalize_adapthist": None,
    "median_filter": None,
    "image_io": "image-analysis",
    "stitching": "image-analysis",
    "image_and_labels_utils": "image-analysis",
    "maxproj_registration": "image-analysis",
    "extended_depth_of_focus": "image-analysis",
    "intensity_rescaling": "image-analysis",
    "surface_extraction": "image-analysis",
}

# extra -> the importable top-level packages it provides
_EXTRA_PACKAGES = {
    "image-analysis": ("skimage", "xarray", "tifffile"),
}

_PUBLIC = {
    "plotting_utils": (
        "color_coded_projection", "describe_image", "labels_cmap", "labels_cmap_black",
        "make_labels_cmap", "label_rgba", "LabelNorm", "PlottingStyle",
        "set_plotting_style", "savefig_svg", "raincloud_plot",
        "i_blue", "i_bordeaux", "i_forest", "i_orange", "i_purple", "i_teal",
        "i_rose", "i_gold", "i_slate", "I_SERIES_COLORMAPS",
    ),
    "tnia_plotting_anywidgets": (
        "show_zyx_slice", "show_zyx_max", "show_zyx_projection", "show_zyx",
        "show_zyx_max_slabs", "show_zyx_projection_slabs", "create_multichannel_rgb",
        "show_zyx_max_slice_interactive", "show_zyx_max_slice_interactive_point_annotator",
        "show_zyx_max_scatter_interactive", "show_iso_scatter",
    ),
    "label_annotator": ("LabelAnnotator",),
    "_conventions": ("check_zyx_input",),
    "image_io": (
        "load_image", "set_channel_info", "channels_where", "channels_by", "to_zyx",
        "array_to_stczyx_xarray", "get_stczyx_pixel_sizes", "get_tiff_voxel_size",
    ),
    "stitching": ("stitch_tiles",),
    "image_and_labels_utils": (
        "voronoi_otsu_labeling", "windowed_slice_projection",
        "optimized_entire_labels_touching_mask", "sample_intensity_around_points",
        "sample_intensity_along_surface_normals", "create_ellipsoid_struct",
        "generate_morphological_surface_mask", "estimate_inter_label_distance",
        "fit_plane_ransac", "generate_plane_basis", "sample_volume_plane",
    ),
    "spline_utils": (
        "generate_random_3d_coordinates", "fit_cubic_spline", "create_3d_image_from_spline",
        "create_nd_image_from_spline", "plot_3d_spline", "create_resampled_spline",
        "calculate_vector_difference", "calculate_tangent_vectors", "project_onto_plane",
        "normalize_vectors", "spline_length", "calculate_spline_length",
    ),
    "maxproj_registration": (
        "zero_shift_multi_dimensional", "estimate_drift", "compute_drift_trajectory",
        "apply_drift_correction", "apply_subpixel_drift_correction",
    ),
    "extended_depth_of_focus": ("apply_median_filter", "best_focus_image"),
    "intensity_rescaling": (
        "contrast_stretching", "normalize_image", "correct_z_intensity_decay",
        "contrast_stretch_per_slice", "check_z_axis_orientation",
        "fit_basic_shading", "apply_basic_shading",
    ),
    "clahe_equalize_adapthist": ("clahe_u8",),
    "median_filter": ("median_filter_fast",),
    "surface_extraction": ("extract_surface", "extract_surface_tiled"),
}

# Deprecated: importable (they warn when called), not in __all__.
_DEPRECATED = {
    "hist_imshow": "plotting_utils",
    "numpy_to_stczyx_xarray": "image_io",
}

_LAZY = {name: mod for mod, names in _PUBLIC.items() for name in names}
_LAZY.update(_DEPRECATED)

__all__ = [name for names in _PUBLIC.values() for name in names]


def _is_available(pkg):
    """True when the top-level package ``pkg`` can be found (any error counts as missing)."""
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


def _install_hint(name, module, exc):
    """ImportError naming the extra when ``exc`` comes from a missing optional package, else None."""
    extra = _EXTRA_OF_MODULE.get(module)
    if extra is None:
        return None
    missing = [p for p in _EXTRA_PACKAGES[extra] if not _is_available(p)]
    failing = (exc.name or "").split(".")[0]
    # exc.name is None for a plain ImportError raised by an import hook (e.g. a
    # sandbox blocking the package); a named error must come from the missing package.
    if not missing or (exc.name is not None and failing not in missing):
        return None
    return ImportError(
        f"eigenp_utils.{name} needs {', '.join(missing)}: "
        f'pip install "eigenp-utils[{extra}]"'
    )


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        mod = importlib.import_module(f".{module}", __name__)
    except ImportError as exc:
        hint = _install_hint(name, module, exc)
        if hint is None:
            raise
        raise hint from exc
    value = getattr(mod, name)
    if name not in _DEPRECATED:
        globals()[name] = value  # cache; deprecated names stay on the slow path
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_DEPRECATED))
