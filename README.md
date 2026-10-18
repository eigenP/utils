# eigenp-utils

`eigenp-utils` is a comprehensive toolkit of helper utilities for scientific Python. It provides modules for image analysis, single-cell data processing, advanced plotting, and core Python utilities.

## Features

### Image Analysis
* **Extended Depth of Focus (EDOF)**: Reconstruct focused 2D images from 3D stacks with high accuracy using log-parabolic interpolation of focus scores and continuous surface sampling.
* **Surface Extraction**: Robust extraction of 2D surfaces from 3D volumes. Includes topological filtering (Connected Components Analysis) to handle debris, nearest-neighbor inpainting for invalid regions, and precise upscaling via `RegularGridInterpolator`. Memory optimized for large datasets, with a parallelization vignette available via `dask_extract_surface`.
* **Registration & Drift Correction**: Bidirectional 2D drift correction (`apply_drift_correction_2D`, `compute_drift_trajectory`), and iterative shift-compensated windowing (`maxproj_registration`) to eliminate systematic biases and achieve sub-pixel stability.
* **Intensity Rescaling**: Tools for contrast enhancement (including CLAHE), Z-axis intensity decay correction using exact analytical Ordinary Least Squares (OLS) fitting and gamma adjustment (`correct_z_intensity_decay`), and pure-NumPy/SciPy BaSiCPy shading correction (`fit_basic_shading`, `apply_basic_shading`).
* **Segmentation**: Fast 2D/3D spot labeling using `voronoi_otsu_labeling`.
* **Anisotropic Pixel Support**: Core spatial processing and morphology functions natively handle physical pixel sizes to accurately support anisotropic microscopy data without structural distortion.
* **3D Plane Sampling & Geometry**: Utilities to fit planes using RANSAC (`fit_plane_ransac`), compute orthonormal bases (`generate_plane_basis`), and dynamically extract or sample 2D oriented planes from anisotropic 3D volumes (`sample_volume_plane`).
* **Projection Utilities**: Generate windowed slice projections (`windowed_slice_projection`).

### Plotting & Visualization
* **Interactive 3D Widgets**: Jupyter and Marimo-compatible, `anywidget`-based orthogonal slicers (`TNIASliceWidget`, `show_zyx` for dynamic multichannel viewers) with rotatable crosshairs, interactive point cloud visualization (`IsoScatterWidget`, `show_iso_scatter`), and 3D point annotation (`TNIAAnnotatorWidget`). Includes thick-slab projections (`show_zyx_max_slabs`, `show_zyx_projection_slabs`) and a one-click UI parameter copy feature for reproducibility.
* **Interactive 3D Scatter Plots**: Utilities for generating interactive 3D scatter plots with Plotly, including native support for AnnData embeddings (`plotly_scatter_3d`, `plotly_scatter_3d_from_adata_obsm`).
* **Publication-Ready Plots**: Set global styles including editable SVG text (`set_plotting_style`). `raincloud_plot` supporting Seaborn-style arguments (grouped and colored with automatic position dodging), pre-KDE outlier filtering, and data subset highlighting. Custom Matplotlib colormap generation via `colormap_maker`, threshold-based scatter point rasterization to minimize SVG file sizes while preserving vector shapes via `savefig_svg`, and colormap truncation/modification (`adjust_colormap`).
* **Image Projections & Compositing**: Generate histograms over image plots (`hist_imshow`), produce depth color-coded projections (`color_coded_projection`), perform advanced multi-channel color blending (`create_multichannel_rgb`), and visualize intensity transformations (`brightness_diagnostic_plotter`).

### Single-Cell Analysis
* **Normalization**: Fast `pflogpf` normalization wrapper leveraging Rust-based PFlog / shifted-CLR approaches.
* **Feature Selection**: Integration with `triku` (`run_triku`) for OT distance based feature selection.
* **Robust Cluster Annotation**: Score cell types via the Empirical Probability of Superiority ($P(S_1 > S_2)$) to ensure robustness against outliers and non-normal distributions (`annotate_clusters_by_markers`).
* **Dataset Integration (kkNN)**: Adaptive curvature-based k-nearest neighbors mapping (`kknn_ingest`) to dynamically project metadata and embeddings across references based on local manifold geometry.
* **Label Classification & Smoothing**: Distance-weighted majority voting or averaging (`kknn_classifier`) to smooth categorical or continuous cell metadata using the kkNN backbone. Graph-based smoothing of expression data (`smooth_expression_on_graph`).
* **Gene Archetypes**: Cluster genes by expression patterns to find dominant archetypes using hierarchical Ward clustering and SVD (`find_expression_archetypes`).
* **Multiscale Clustering**: Run multi-resolution Leiden clustering and track lineage hierarchies across scales (`multiscale_coarsening`, `plot_clustering_tree`).
* **Lineage Coupling**: Compute exact co-occurrence expectations and z-scores analytically using a vectorized hypergeometric log-gamma formulation and the inclusion-exclusion principle (`calculate_lineage_coupling`).
* **Feature Correlation**: Find highly correlated features with respect to targets, optionally utilizing graph-based diffusion to smooth over the cell-cell graph (`find_correlated_features`).
* **Spatial Autocorrelation**: Fast Moran's I implementation (`morans_i_all_fast`) that correctly handles general (non-row-standardized) spatial weights.
* **Dimensionality Reduction**: `tl_pacmap` for PaCMAP embeddings supporting versatile initialization strategies (e.g., PAGA, PCA, random).
* **Palette Management**: Utilities for transferring color palettes between `AnnData` objects (`carry_palette`).

### Statistical Utilities
* **General Statistics**: `stats.py` provides comprehensive statistical functions including `cohens_d`, `bootstrap_ci` (with bias-corrected and accelerated (BCa) bootstrap methods), `summary_stats`, robust outlier removal (`remove_outliers`) supporting Mahalanobis distance, `robust_standardize` with principled hierarchical dispersion fallback (MAD -> MeanAD -> STD) for zero-inflated or heavily tied data, and `add_stat_annotations` for annotating plots with significance markers.
* **Distribution Distances**: Exact closed-form Wasserstein distance for equal-sized empirical distributions.


### Core Utilities
* **Spline Utilities**: Fit cubic splines (`fit_cubic_spline`), resample them uniformly (`create_resampled_spline`), render them into ND image volumes (`create_nd_image_from_spline`), calculate tangent vectors, project points onto planes, and calculate real-world arc lengths (`calculate_spline_length`) (`spline_utils.py`).
* **Data Handling**: Standardize image dataset dimensions strictly to STCZYX via `numpy_to_stczyx_xarray`.
* **I/O Utilities**: Functions to streamline file and data reading, check files before saving (`check_file_before_save`), ensure directory existence (`ensure_directory`), and fast file downloading (`download_file`).

### Examples
* **Notebooks**: The `notebooks/` directory contains Marimo notebooks demonstrating package functionalities, such as using statistical utilities with classic datasets.

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
- `[plotting]` - installs `plotly`, `statannotations` and `statsmodels`.
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

### Browser / WASM (marimo, Pyodide)

Everything except single-cell analysis runs in [Pyodide](https://pyodide.org), the WebAssembly runtime behind marimo's browser notebooks, including the interactive anywidget ortho viewers. CI runs the test suite inside Pyodide (`tests/wasm/`).

| Module(s) | Needs | WASM |
| --- | --- | --- |
| `plotting_utils`, `tnia_plotting_anywidgets`, `stats`, `spline_utils`, `io` (file helpers) | core | yes |
| `image_and_labels_utils`, `intensity_rescaling`, `extended_depth_of_focus`, `maxproj_registration`, `surface_extraction`, `clahe_equalize_adapthist` | `scikit-image` | yes |
| `image_io` | `xarray`, `tifffile` | yes |
| `stats.add_stat_annotations`, `plotting_utils.plotly_scatter_3d` | `[plotting]` | yes |
| `single_cell` graph helpers (`smooth_expression_on_graph`, `find_correlated_features`, ...) | `anndata`, `scikit-learn` | yes (scanpy is imported lazily) |
| `single_cell` functions that call scanpy / pacmap / leidenalg | `[single-cell]` (`numba` has no WASM build) | no |

micropip installs only wheels, and only from hosts that send CORS headers, so a GitHub archive or `git+` URL does not work in the browser. CI publishes the latest `main` build to the `wasm-wheel` branch:

```python
import micropip
await micropip.install([
    "https://raw.githubusercontent.com/eigenP/utils/wasm-wheel/eigenp_utils-0-py3-none-any.whl",
    "scikit-image", "tifffile",  # only for the image-analysis modules / reading TIFFs
])
```

(`eigenp_utils-0-...whl` is a version-independent alias accepted by micropip; pip/uv reject it, so use the versioned file next to it or `git+` natively.)

Install everything in **one** `micropip.install([...])` call: micropip never upgrades a package an earlier call installed (e.g. anndata needs a newer `typing-extensions` than Pyodide bundles).

Notebooks keep `git+` in their PEP 723 header for native use, gated with `; sys_platform != 'emscripten'` so marimo WASM skips it (and anything else without a WASM build) instead of failing the whole header install. Demo notebooks that need scanpy (`demo_sc_*`) cannot run in the browser and say so; run them locally (`uvx marimo edit --sandbox <file>`) or on [molab](https://molab.marimo.io). `blog_gene_distances_ot.py` runs in WASM with substitutions it announces (kNN graph instead of UMAP connectivities, HiGHS instead of POT).

The browser cannot block on network I/O, so use `await eigenp_utils.io.download_file_async(url, dest)` there (it uses `pyodide.http.pyfetch`; natively it runs `download_file` in a thread). The server must send CORS headers: e.g. for scikit-image's `cells3d`, use GitLab's API endpoint `https://gitlab.com/api/v4/projects/scikit-image%2Fdata/repository/files/cells3d.tif/raw?ref=master` rather than the `/-/raw/` URL.

## License

License CC BY-NC https://creativecommons.org/licenses/by-nc/4.0/
