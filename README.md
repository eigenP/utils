# eigenp-utils

`eigenp-utils` is a comprehensive toolkit of helper utilities for scientific Python. It provides modules for image analysis, single-cell data processing, advanced plotting, and core Python utilities.

## Features

### Image Analysis
* **Extended Depth of Focus (EDOF)**: Reconstruct focused 2D images from 3D stacks with high accuracy using log-parabolic interpolation of focus scores and continuous surface sampling.
* **Surface Extraction**: Robust extraction of 2D surfaces from 3D volumes. Includes topological filtering (Connected Components Analysis) to handle debris, nearest-neighbor inpainting for invalid regions, and precise upscaling via `RegularGridInterpolator`.
* **Registration & Drift Correction**: Bidirectional 2D drift correction (`apply_drift_correction_2D`, `compute_drift_trajectory`), and iterative shift-compensated windowing (`maxproj_registration`) to eliminate systematic biases and achieve sub-pixel stability.
* **Intensity Rescaling**: Tools for contrast enhancement, including CLAHE (`clahe_equalize_adapthist`), pure-NumPy/SciPy implementations of BaSiCPy shading correction (flatfield/darkfield estimation), and per-slice brightness/gamma invariant adjustments.

### Plotting & Visualization
* **Interactive 3D Widgets**: Jupyter and Marimo-compatible, `anywidget`-based orthogonal slicers (`TNIASliceWidget`, `show_xyz` for dynamic multichannel viewers), interactive point cloud visualization (`IsoScatterWidget`), and 3D point annotation (`TNIAAnnotatorWidget`).
* **Publication-Ready Plots**: `raincloud_plot` supporting Seaborn-style arguments (grouped and colored with automatic position dodging). Custom Matplotlib colormap generation via `colormap_maker`, and SVGs embedded with metadata via `savefig_svg`.
* **Interactive Scatter Plots**: `plotly_scatter_3d` and `plotly_scatter_3d_from_adata_obsm` for dynamic, interactive 3D embeddings directly from arrays or AnnData objects.

### Single-Cell Analysis
* **Robust Cluster Annotation**: Score cell types via the Empirical Probability of Superiority ($P(S_1 > S_2)$) to ensure robustness against outliers and non-normal distributions (`annotate_clusters_by_markers`).
* **Dataset Integration (kkNN)**: Adaptive curvature-based k-nearest neighbors mapping (`kknn_ingest`) to dynamically project metadata and embeddings across references based on local manifold geometry.
* **Label Classification & Smoothing**: Distance-weighted majority voting or averaging (`kknn_classifier`) to smooth categorical or continuous cell metadata using the kkNN backbone.
* **Gene Archetypes**: Cluster genes by expression patterns to find dominant archetypes using hierarchical Ward clustering and SVD (`find_expression_archetypes`).
* **Multiscale Clustering**: Run multi-resolution Leiden clustering and track lineage hierarchies across scales (`multiscale_coarsening`, `plot_clustering_tree`).
* **Feature Correlation**: Find highly correlated features with respect to targets, optionally utilizing graph-based diffusion to smooth over the cell-cell graph (`find_correlated_features`).
* **Spatial Autocorrelation**: Fast Moran's I implementation (`morans_i_all_fast`) that correctly handles general (non-row-standardized) spatial weights.
* **Dimensionality Reduction**: `tl_pacmap` for PaCMAP embeddings supporting versatile initialization strategies (e.g., PAGA, PCA, random).

### Statistical Utilities
* **General Statistics**: `stats.py` provides comprehensive statistical functions including `cohens_d`, `bootstrap_ci`, `summary_stats`, `remove_outliers`, and `add_stat_annotations` for annotating plots with significance markers.


### Core Utilities
* **Spline Utilities**: Calculate tangent vectors and project points onto planes for arbitrary splines and discrete curves (`spline_utils.py`).
* **Data Handling**: Standardize image dataset dimensions strictly to STCZYX via `numpy_to_stczyx_xarray`.
* **Task Calendar Scheduler**: Generate interactive, Gantt-style timeline schedules for linked events using Plotly (`task_calendar_scheduler.py`), fully compatible with Marimo notebooks.
* **I/O Utilities**: Functions to streamline file and data reading.

## Interactive Demos
The `notebooks/` directory contains numerous interactive `marimo` application demos showcasing the toolkit's capabilities. You can run them using `marimo edit notebooks/demo_name.py`.

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

### WASM / Pyodide Compatibility

The package is designed for compatibility with Pyodide and WebAssembly (WASM) environments, such as Marimo in WASM mode. It lazily loads heavy dependencies and can be installed natively inside the browser via `micropip`:

```python
import micropip
await micropip.install("eigenp-utils")
```
The CI pipeline automatically builds and publishes a pure-Python wheel suitable for these deployments.

## License

License CC BY-NC https://creativecommons.org/licenses/by-nc/4.0/
