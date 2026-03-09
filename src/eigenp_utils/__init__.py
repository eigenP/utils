"""Utility functions for eigenp workflows."""

from .plotting_utils import (
    color_coded_projection,
    hist_imshow,
    labels_cmap,
)
from .image_io import numpy_to_stczyx_xarray, get_tiff_voxel_size
from .image_and_labels_utils import (
    windowed_slice_projection,
    optimized_entire_labels_touching_mask,
    sample_intensity_around_points_optimized
)
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
from .tnia_plotting_3d import (
    show_xyz_slice,
    show_xyz_max,
    show_xyz_projection,
    show_xyz,
    show_xyz_max_slabs,
    create_multichannel_rgb,
)
from .maxproj_registration import (
    zero_shift_multi_dimensional,
    estimate_drift,
    apply_drift_correction,
    apply_subpixel_drift_correction,
)
from .extended_depth_of_focus import apply_median_filter, best_focus_image

__all__ = [
    "color_coded_projection",
    "hist_imshow",
    "labels_cmap",
    "numpy_to_stczyx_xarray",
    "get_tiff_voxel_size",
    "windowed_slice_projection",
    "optimized_entire_labels_touching_mask",
    "sample_intensity_around_points_optimized",
    "generate_random_3d_coordinates",
    "fit_cubic_spline",
    "create_3d_image_from_spline",
    "create_nd_image_from_spline",
    "plot_3d_spline",
    "create_resampled_spline",
    "calculate_vector_difference",
    "calculate_tangent_vectors",
    "project_onto_plane",
    "normalize_vectors",
    "show_xyz_slice",
    "show_xyz_max",
    "show_xyz_projection",
    "show_xyz",
    "show_xyz_max_slabs",
    "create_multichannel_rgb",
    "zero_shift_multi_dimensional",
    "estimate_drift",
    "apply_drift_correction",
    "apply_subpixel_drift_correction",
    "apply_median_filter",
    "best_focus_image",
]
