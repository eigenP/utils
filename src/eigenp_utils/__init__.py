"""Utility functions for eigenp workflows."""

from .plotting_utils import (
    color_coded_projection,
    hist_imshow,
    labels_cmap,
)

from .tnia_plotting_anywidgets import (
    show_zyx_slice,
    show_zyx_max,
    show_zyx_projection,
    show_zyx,
    show_zyx_max_slabs,
    create_multichannel_rgb,
)

__all__ = [
    "color_coded_projection",
    "hist_imshow",
    "labels_cmap",
    "show_zyx_slice",
    "show_zyx_max",
    "show_zyx_projection",
    "show_zyx",
    "show_zyx_max_slabs",
    "create_multichannel_rgb",
]

# Robustly import heavy dependencies
try:
    from .image_io import numpy_to_stczyx_xarray, get_tiff_voxel_size
    __all__.extend(["numpy_to_stczyx_xarray", "get_tiff_voxel_size"])
except ImportError:
    pass

try:
    from .image_and_labels_utils import (
        windowed_slice_projection,
        optimized_entire_labels_touching_mask,
        sample_intensity_around_points_optimized
    )
    __all__.extend([
        "windowed_slice_projection",
        "optimized_entire_labels_touching_mask",
        "sample_intensity_around_points_optimized"
    ])
except ImportError:
    pass

try:
    from .spline_utils import (
        generate_random_3d_coordinates,
        fit_cubic_spline,
        create_3d_image_from_spline,
        create_nd_image_from_spline,
        plot_3d_spline,
        create_resampled_spline,
        calculate_vector_difference,
        calculate_tangent_vectors,
        project_onto_plane,
        normalize_vectors
    )
    __all__.extend([
        "generate_random_3d_coordinates",
        "fit_cubic_spline",
        "create_3d_image_from_spline",
        "create_nd_image_from_spline",
        "plot_3d_spline",
        "create_resampled_spline",
        "calculate_vector_difference",
        "calculate_tangent_vectors",
        "project_onto_plane",
        "normalize_vectors"
    ])
except ImportError:
    pass

try:
    from .maxproj_registration import (
        zero_shift_multi_dimensional,
        estimate_drift,
        apply_drift_correction,
        apply_subpixel_drift_correction,
    )
    __all__.extend([
        "zero_shift_multi_dimensional",
        "estimate_drift",
        "apply_drift_correction",
        "apply_subpixel_drift_correction",
    ])
except ImportError:
    pass

try:
    from .extended_depth_of_focus import apply_median_filter, best_focus_image
    __all__.extend([
        "apply_median_filter",
        "best_focus_image",
    ])
except ImportError:
    pass
