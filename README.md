# eigenp-utils

`eigenp-utils` is a comprehensive toolkit of helper utilities for scientific Python. It provides modules for image analysis, single-cell data processing, advanced plotting, and core Python utilities.

## Features

### Image Analysis
* **Extended Depth of Focus (EDOF)**: Reconstruct focused 2D images from 3D stacks with high accuracy using log-parabolic interpolation of focus scores and continuous surface sampling.
* **Surface Extraction**: Robust extraction of 2D surfaces from 3D volumes. Includes topological filtering (Connected Components Analysis) to handle debris, nearest-neighbor inpainting for invalid regions, and precise upscaling via `RegularGridInterpolator`. Memory optimized for large datasets, with a parallelization vignette available via `dask_extract_surface`.
* **Registration & Drift Correction**: Bidirectional 2D drift correction (`apply_drift_correction_2D`, `compute_drift_trajectory`), and iterative shift-compensated windowing (`maxproj_registration`) to eliminate systematic biases and achieve sub-pixel stability.
* **Intensity Rescaling**: Tools for contrast enhancement (including CLAHE), slice-by-slice brightness adjustment (`adjust_brightness_per_slice`), Z-axis intensity decay correction using exact analytical Ordinary Least Squares (OLS) fitting (`correct_z_intensity_decay`), and pure-NumPy/SciPy BaSiCPy shading correction (`fit_basic_shading`, `apply_basic_shading`).
* **Segmentation**: Fast 2D/3D spot labeling using `voronoi_otsu_labeling`.
* **Anisotropic Pixel Support**: Core spatial processing and morphology functions natively handle physical pixel sizes to accurately support anisotropic microscopy data without structural distortion.
* **3D Plane Sampling & Geometry**: Utilities to fit planes using RANSAC (`fit_plane_ransac`), compute orthonormal bases (`generate_plane_basis`), and dynamically extract or sample 2D oriented planes from anisotropic 3D volumes (`sample_volume_plane`).

### Plotting & Visualization
* **Interactive 3D Widgets**: Jupyter and Marimo-compatible, `anywidget`-based orthogonal slicers (`TNIASliceWidget`, `show_zyx` for dynamic multichannel viewers) natively supporting independent anisotropic pixel dimensions (independent X, Y, Z scaling) with rotatable crosshairs, interactive point cloud visualization (`show_iso_scatter`), and 3D point annotation (`TNIAAnnotatorWidget`). Provides custom labeling via the `channel_labels` parameter, and includes a one-click UI parameter copy feature for reproducibility.
* **Interactive 3D Scatter Plots**: Utilities for generating interactive 3D scatter plots with Plotly, including native support for AnnData embeddings (`plotly_scatter_3d`, `plotly_scatter_3d_from_adata_obsm`).
* **Publication-Ready Plots**: `raincloud_plot` supporting Seaborn-style arguments (grouped and colored with automatic position dodging), pre-KDE outlier filtering, and data subset highlighting. Custom Matplotlib colormap generation via `colormap_maker`, and threshold-based scatter point rasterization to minimize SVG file sizes while preserving vector shapes via `savefig_svg`. Includes the `editable_text` parameter in `set_plotting_style` and `savefig_svg` for exporting SVGs with editable text.

### Single-Cell Analysis
* **Robust Cluster Annotation**: Score cell types via the Empirical Probability of Superiority ($P(S_1 > S_2)$) to ensure robustness against outliers and non-normal distributions (`annotate_clusters_by_markers`).
* **Dataset Integration (kkNN)**: Adaptive curvature-based k-nearest neighbors mapping (`kknn_ingest`) to dynamically project metadata and embeddings across references based on local manifold geometry.
* **Data I/O**: `import_obs_to_adata_from_csv` simplifies metadata ingestion into AnnData objects, providing helpful diagnostic messages for missing observation keys.
* **Label Classification & Smoothing**: Distance-weighted majority voting or averaging (`kknn_classifier`) to smooth categorical or continuous cell metadata using the kkNN backbone.
* **Gene Archetypes**: Cluster genes by expression patterns to find dominant archetypes using hierarchical Ward clustering and SVD (`find_expression_archetypes`).
* **Multiscale Clustering**: Run multi-resolution Leiden clustering and track lineage hierarchies across scales (`multiscale_coarsening`, `plot_clustering_tree`).
* **Lineage Coupling**: Compute exact co-occurrence expectations and z-scores analytically using a vectorized hypergeometric log-gamma formulation and the inclusion-exclusion principle (`calculate_lineage_coupling`).
* **Feature Correlation**: Find highly correlated features with respect to targets, optionally utilizing graph-based diffusion to smooth over the cell-cell graph (`find_correlated_features`).
* **Spatial Autocorrelation**: Fast Moran's I implementation (`morans_i_all_fast`) that correctly handles general (non-row-standardized) spatial weights.
* **Dimensionality Reduction**: `tl_pacmap` for PaCMAP embeddings supporting versatile initialization strategies (e.g., PAGA, PCA, random).

### Statistical Utilities
* **General Statistics**: `stats.py` provides comprehensive statistical functions including `cohens_d`, `bootstrap_ci` (with bias-corrected and accelerated (BCa) bootstrap methods), `summary_stats`, robust outlier removal (`remove_outliers`) supporting Mahalanobis distance, `robust_standardize` with principled hierarchical dispersion fallback (MAD -> MeanAD -> STD) for zero-inflated or heavily tied data, and `add_stat_annotations` for annotating plots with significance markers.
* **Distribution Distances**: Exact closed-form Wasserstein distance for equal-sized empirical distributions.


### Core Utilities
* **Spline Utilities**: Calculate tangent vectors and project points onto planes for arbitrary splines and discrete curves (`spline_utils.py`).
* **Data Handling**: Standardize image dataset dimensions strictly to STCZYX via `numpy_to_stczyx_xarray`.
* **I/O Utilities**: Functions to streamline file and data reading.

### Examples
* **Notebooks**: The `notebooks/` directory contains Marimo notebooks demonstrating package functionalities, such as using statistical utilities with classic datasets.

## Usage Examples

Here is a quick example demonstrating how to configure publication-ready plots with editable text and how to use the statistical utilities:

```python
import numpy as np
import matplotlib.pyplot as plt
from eigenp_utils import plotting_utils as pu
from eigenp_utils import stats

# Set a clean plotting style for publications (SVG with editable text enabled)
pu.set_plotting_style(editable_text=True)

# Generate some data
data = np.random.normal(loc=0, scale=1, size=100)
data = np.append(data, [10.0, -10.0]) # Add some outliers

# Calculate robust standard deviation (fallback to MeanAD if MAD collapses)
std_robust = stats.robust_standardize(data, return_scale=True)[1]
print(f"Robust Standard Deviation: {std_robust:.2f}")

# Compute a bootstrapped confidence interval (Bias-Corrected and Accelerated)
ci_lower, ci_upper = stats.bootstrap_ci(data, stat_func=np.mean, method='bca')
print(f"95% CI for the mean: ({ci_lower:.2f}, {ci_upper:.2f})")

# Plotting and exporting with editable text
fig, ax = plt.subplots()
ax.hist(data, bins=20)
ax.set_title("Histogram of Data")

# Save the figure as an SVG, preserving vector shapes and editable text fonts
pu.savefig_svg(fig, "histogram", editable_text=True)
```

## Installation

By default, the package installs a minimal set of dependencies (like `numpy`, `scipy`, `pandas`, `matplotlib`, etc).
To install it, run:

```bash
pip install eigenp-utils
```

Alternatively, to install the latest development version directly from GitHub:

```bash
pip install "eigenp_utils @ git+https://github.com/eigenP/utils.git"
```

Using `uv`:

```bash
uv pip install "eigenp_utils @ git+https://github.com/eigenP/utils.git"
```

### Optional Dependencies

You can choose to install optional dependencies if you need functionality such as single-cell analysis or image analysis:

- `[image-analysis]` - installs `scikit-image`.
- `[single-cell]` - installs packages like `scanpy`, `pacmap`, `leidenalg`, etc.
- `[plotting]` - installs `plotly`.
- `[all]` - installs all of the optional dependencies above.
- `[dev]` - installs all dependencies and additional tools for testing (e.g. `pytest`).

e.g. (uv install)

```bash
uv pip install "eigenp-utils[all]"
```
*(Note: quotes are required so the shell doesn't misinterpret the brackets.)*

For the latest development versions with optional dependencies:

```bash
pip install "eigenp_utils[all] @ git+https://github.com/eigenP/utils.git"
```
or
```bash
uv pip install "eigenp_utils[all] @ git+https://github.com/eigenP/utils.git"
```

You can replace `[all]` with other groups like `[single-cell]` or `[image-analysis,single-cell]` depending on your specific needs.

## License

License CC BY-NC https://creativecommons.org/licenses/by-nc/4.0/
