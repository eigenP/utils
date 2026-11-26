# eigenp-utils

`eigenp-utils` is a comprehensive toolkit of helper utilities for scientific Python. It provides modules for image analysis, single-cell data processing, advanced plotting, and core Python utilities.

## Features

### Image Analysis
* **Extended Depth of Focus (EDOF)**: Reconstruct focused 2D images from 3D stacks with high accuracy using log-parabolic interpolation of focus scores and continuous surface sampling.
* **Surface Extraction**: Robust extraction of 2D surfaces from 3D volumes. Includes topological filtering (Connected Components Analysis) to handle debris, nearest-neighbor inpainting for invalid regions, and precise upscaling via `RegularGridInterpolator`. Volumes that do not fit in memory (dask, zarr, memmap, or `load_image(path, lazy=True)`) are processed tile by tile with `extract_surface_tiled`, which gives the same result as a whole-volume `extract_surface` call (`dask_extract_surface` is a deprecated stub). The default blur is `gaussian_sigma=(1, 4, 4)` binned voxels (Z, Y, X); pass `gaussian_sigma_um=` with `pixel_sizes=` for a blur in µm.
* **Registration & Drift Correction**: Drift correction of 2D+t (T, Y, X) and 3D+t (T, Z, Y, X) stacks (`apply_drift_correction`, `compute_drift_trajectory`, `apply_subpixel_drift_correction`), with iterative shift-compensated windowing (`maxproj_registration`) for sub-pixel stability. `estimate_drift(frame1, frame2, order='zyx')` returns the correction in array-axis order ((dy, dx), or (dz, dy, dx) for 3D frames), which `apply_subpixel_drift_correction(image, shift)` takes directly; the old (dx, dy) order is the default until 0.1.0, with a FutureWarning. Drift values are in pixels.
* **Intensity Rescaling**: Tools for contrast enhancement (`contrast_stretching` / `normalize_image` with `percentiles=(0.1, 99.9)` by default, and `clahe_u8`, a fast N-D CLAHE on 8 bits that needs no scikit-image), Z-axis intensity decay correction (`correct_z_intensity_decay(method='gamma'|'gain', fit_model='exponential'|'linear')`: the exponential model is a bounded non-linear least-squares fit seeded by log-space OLS, the linear model an exact OLS fit; slices that hold only background are left out of the fit in sparse stacks), a fast k x k per-slice median filter that matches `scipy.ndimage.median_filter(mode='reflect')` exactly (`median_filter_fast` in `eigenp_utils.median_filter`, numba or NumPy backend), and pure-NumPy/SciPy BaSiCPy shading correction (`fit_basic_shading`, `apply_basic_shading`).
* **Segmentation**: Fast 2D/3D spot labeling using `voronoi_otsu_labeling`.
* **Anisotropic Pixel Support**: The functions that take `pixel_sizes=` (segmentation, morphology, plane and normal sampling, `windowed_slice_projection`, `spline_length`, the viewers, and `extract_surface` with `gaussian_sigma_um=`) work in µm on anisotropic data. `best_focus_image`, the drift functions and CLAHE work in pixels.
* **3D Plane Sampling & Geometry**: Utilities to fit planes using RANSAC (`fit_plane_ransac`), compute orthonormal bases (`generate_plane_basis`), and dynamically extract or sample 2D oriented planes from anisotropic 3D volumes (`sample_volume_plane`).
* **Projection Utilities**: Generate windowed slice projections (`windowed_slice_projection`).
* **xarray input**: the (Z)YX processing functions take NumPy-like arrays and 2-D / 3-D DataArrays. A DataArray with more dims (e.g. the STCZYX output of `load_image`) raises a TypeError that points to `to_zyx(da, C=...)`, which returns the (Z, Y, X) array and its pixel sizes in µm.

### Plotting & Visualization
* **Interactive 3D Widgets**: Jupyter and Marimo-compatible, `anywidget`-based orthogonal slicers with rotatable views and thick slabs (`show_zyx_max_slice_interactive` / `TNIASliceWidget`), 3D point annotation (`show_zyx_max_slice_interactive_point_annotator` / `TNIAAnnotatorWidget`; `snap_depth=True` stores the depth the projection shows), a 3D scatter viewer (`show_zyx_max_scatter_interactive` / `TNIAScatterWidget`; `channel_mode=` picks values or categories, and its `slabs_position` / `slabs_thickness` are in physical units) and an isotropic point cloud viewer (`IsoScatterWidget`, `show_iso_scatter`). They accept DataArrays, a list or tuple of channels, `channel_axis=` and 2D images, and have a one-click UI parameter copy feature for reproducibility. The static renderers return plain Matplotlib figures (not registered with pyplot): `show_zyx` (from three planes), `show_zyx_slice`, `show_zyx_max`, `show_zyx_projection` and the thick-slab projections `show_zyx_max_slabs` (full range by default) and `show_zyx_projection_slabs`. `figsize=None` sizes the figure from the physical extents. Deprecated names, which still work until 0.1.0: `colors=` (use `colormap=`), `sxy`/`sz` (use `pixel_sizes=`), `x_s`...`z_t` (use `slabs_position=` / `slabs_thickness=`).
* **In-notebook Label Annotation**: `LabelAnnotator` (`eigenp_utils.label_annotator`) paints instance masks (e.g. Cellpose training masks; strokes are closed and filled, ids auto-increment) or semantic class scribbles (e.g. pixel-classifier labels) on 2D slices of multichannel images in Jupyter or marimo, without napari. Brush, eraser, flood fill, label picker, undo, zoom/pan, per-channel colour / contrast / gamma; z-linking ground truth for per-slice 2D masks (one id across z = one object): *keep label*, neighbour-slice (z±1) outlines, whole-track merge / split (`merge_ids`, `split_id`, undoable) and the active id's z extent; sparse training annotations with `background=True` (0 = unannotated, reserved id 65535 = annotated background painted as open scribbles, 1.. = cells; `labels_signed()` / `save(signed=True)` give -1 background), a delete-mask tool (`delete_id`, undoable) and a counts readout (`counts()`: cells, background / annotated pixels and annotated fraction per slice and volume); a list of stacks as `image` (stack selector, PageUp/PageDown; `stack_labels`, `save(stacks="all")`), z ‹ › buttons, a viewport that fits the window and refits on resize / fullscreen, collapsible channel controls; pixel sizes from DataArray coordinates or `pixel_sizes=` (a 2D image also takes a (Y, X) pair), written by `save` as TIFF resolution with axes metadata so `load_image` reads them back; `vmin` / `vmax` in data units (default: the package display range; double-click a slider to restore it); `colormap=` / `channel_labels=` as in the ortho viewers; keys act only while the viewport has focus, and a plain wheel zooms only after a click; filled or outlined label display (`display="outlined"` / key `o`: coloured boundary over a black border band; `outline_px`, `outline_border_px`); compositing and painting run in the browser and only the edited slice's labels travel back to Python. `save(dir, name)` writes `*_img.tif` / `*_masks.tif` training pairs (unscaled data). Demo: `notebooks/demo_label_annotator.py`.
* **3D Scatter Plots**: use the scatter anywidget, e.g. `show_zyx_max_scatter_interactive(points_zyx, channels=labels)` (or `TNIAScatterWidget`) from `eigenp_utils.tnia_plotting_anywidgets`. The Plotly helpers `plotly_scatter_3d` and `plotly_scatter_3d_from_adata_obsm` were removed (they raise `NotImplementedError` with this pointer), and plotly is no longer a dependency.
* **Publication-Ready Plots**: Set global styles including editable SVG text (`set_plotting_style()` returns a handle: `.restore()` it, or use `with set_plotting_style(): ...`; fonts are Inter / DejaVu Sans, including math text). `raincloud_plot` supporting Seaborn-style arguments (grouped and colored with automatic position dodging), pre-KDE outlier filtering, and data subset highlighting. Custom Matplotlib colormap generation via `colormap_maker`, threshold-based scatter point rasterization to minimize SVG file sizes while preserving vector shapes via `savefig_svg(path, fig=fig)` (white background by default; omitting `fig` is deprecated), and colormap truncation/modification (`adjust_colormap`). For label images: `labels_cmap` (transparent background), `labels_cmap_black`, `make_labels_cmap(background=...)`, and `label_rgba(labels)` / `LabelNorm()`, which colour ids exactly as `LabelAnnotator` does (ids above 255 cycle); plus `I Series` colormaps (`i_blue`, `i_bordeaux`, `i_forest`, `i_orange`, `i_purple`, `i_teal`, `i_rose`, `i_gold`, `i_slate`).
* **Image Projections & Compositing**: Show an image slice with its histogram (display range marked) and an info table below (`describe_image`: shape/dtype, or for DataArrays dims, pixel sizes and a channel table with colour swatches; also returned as HTML in `info_html`; row 0 is drawn at the top (`origin='upper'`, as in the viewers, Fiji and napari), RGB(A) arrays are drawn as colour, and integer histograms use level-aligned bins; `hist_imshow` is a deprecated alias), produce depth color-coded projections (`color_coded_projection(stack, 'plasma')`: one robust range for the whole stack by default, `normalize='frame'` for per-frame scaling, `vmin`/`vmax`, `percentiles`, and `combine='argmax'` to colour each pixel by the frame of its maximum. plasma, inferno and magma code depth as brightness, so early frames look dark; `'turbo'` or `'hsv'` keep the brightness equal), perform advanced multi-channel color blending (`create_multichannel_rgb`), and visualize intensity transformations (`brightness_diagnostic_plotter`).

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
* **Bonsai Cell Trees** (`bonsai_tree`, extra `[bonsai]`): wrapper around [Bonsai](https://doi.org/10.1038/s41587-026-03220-2) (de Groot et al., Nat. Biotechnol. 2026), which reconstructs the maximum-likelihood tree over cells from feature estimates with error bars. `tl_bonsai(adata)` takes raw counts from a standard scanpy object (`layers['counts']`, `X` or `.raw`, optionally restricted with `mask_var='highly_variable'`), turns them into log quotients with Poisson error bars (`poisson_log_quotients`), runs Bonsai in a subprocess (optionally in parallel with MPI: `n_procs=`; faster, but the tree then differs from the single-process one at equal likelihood) and stores the tree in `adata.uns`; `bonsai_neighbors` turns it into a scanpy neighbour graph for `sc.tl.leiden` / `umap` / `paga` (`neighbors_key='bonsai_neighbors'`). `run_bonsai` works on plain arrays or a Sanity output folder and returns a `BonsaiTree` with patristic distances and tree k-nearest neighbours. Bonsai is not pip-installable: clone [Bonsai-data-representation](https://github.com/dhdegroot/Bonsai-data-representation) (CC BY-NC 4.0) and set `BONSAI_DIR`.

### Statistical Utilities
* **General Statistics**: `stats.py` provides comprehensive statistical functions including `cohens_d`, `bootstrap_ci` (with bias-corrected and accelerated (BCa) bootstrap methods), `summary_stats`, robust outlier removal (`remove_outliers`) supporting Mahalanobis distance, `robust_standardize` with principled hierarchical dispersion fallback (MAD -> MeanAD -> STD) for zero-inflated or heavily tied data, and `add_stat_annotations` for annotating plots with significance markers.
* **Distribution Distances**: Exact closed-form Wasserstein distance for equal-sized empirical distributions.


### Core Utilities
* **Spline Utilities**: Fit cubic splines (`fit_cubic_spline`), resample them uniformly (`create_resampled_spline`), render them into ND image volumes (`create_nd_image_from_spline`), calculate tangent vectors, project points onto planes, and calculate real-world arc lengths (`spline_length(points, pixel_sizes=...)`, point-major; `calculate_spline_length` with `pixel_sizes=`, whose `resolution=` is deprecated) (`spline_utils.py`).
* **Data Handling**: Standardize image dataset dimensions strictly to STCZYX via `array_to_stczyx_xarray` (NumPy/dask arrays or existing `xarray.DataArray`s, e.g. from `nd2`; `numpy_to_stczyx_xarray` is a deprecated alias), with pixel sizes stored as physical coordinates and `attrs['pixel_sizes']` (read back with `get_stczyx_pixel_sizes`) and tile stage positions as `stage_y`/`stage_x` coordinates, and extract voxel sizes with `get_tiff_voxel_size`.
* **Format-agnostic loading**: `load_image(source, channel_info=..., pixel_sizes=..., lazy=False)` converts ImageJ/OME `.tif` files (RGB/planar samples read as channels; same-shape multi-position series stacked along S), arrays, or a DataArray from any vendor reader (single-letter dims; `P` = position -> `S`) into a strict STCZYX DataArray with raw values and physical coordinates in µm (known length units such as nm, mm or cm are converted); pixel-size overrides are per axis. `lazy=True` does not read the pixels: uncompressed TIFFs are memory-mapped, others are read through dask over tifffile's zarr store (needs `dask` and `zarr`; pyramidal TIFFs give the full-resolution level). `to_zyx(da, C=...)` selects one scene, time point and channel and returns the (Z, Y, X) array plus its pixel sizes for the processing functions. `channels_by(da, "role", ["membrane", "dna"])` returns channel labels in the requested order (unique by default). `describe_image(da)` shows a slice, its histogram and an info panel (dims, pixel sizes, channels with metadata; also returned as `["info"]`).
* **Channel metadata on xarray images**: keep channel names in the `C` coordinate (as provided by the file or reader) and attach per-channel metadata as coordinates along `C` with `set_channel_info(da, {"DAPI": {"color": "gray", "role": "dna"}, ...})` (strict: unknown names raise, so typos and channel-order assumptions fail loudly); select by metadata with `da.sel(C=channels_where(da, role="dna"))`. The metadata follows `sel` / reordering, unlike parallel lists or dicts. The ortho viewers (`show_zyx_max_slice_interactive`, `..._point_annotator`) and `LabelAnnotator` accept such DataArrays directly (C/Z/Y/X dims; select S/T first): channel labels from `C`, colours from the `color` coordinate, pixel sizes from the calibrated coordinates; explicit arguments still win.
* **Tile Stitching**: `stitch_tiles` is a minimal translation-only 2D stitcher for STCZYX tiles (stage-position prior, phase correlation on Z max projections, global least squares, linear blending). `register=False` places the tiles at their stage positions only; stage positions in other length units are converted; `max_area_ratio` guards against a runaway canvas. For more control use [multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher). Demo: `notebooks/pybasic_demo.py`.
* **I/O Utilities**: Functions to streamline file and data reading, check files before saving (`check_file_before_save`), ensure directory existence (`ensure_directory`), and fast file downloading (`download_file`, `download_file_async`), and unzipping (`un_gzip`). They report through the `eigenp_utils.io` logger instead of printing.

### Conventions
* **Units**: µm is the unit of every length inside the package, including processing parameters named `*_um`. Readers (`load_image`, `LabelAnnotator`) convert known length units (nm, µm, mm, cm, m, Å) to µm; `pixel`/`px` means uncalibrated. Scale bars pick nm, µm or mm by magnitude (e.g. EM data in nm, mm-scale tissue), but the numbers underneath stay in µm. When a pixel size is missing, viewers draw a `px` scale bar and processing functions assume 1.0 with a warning.
* **Display range**: the viewers default to vmin = 0 (or the data minimum when it is negative) and vmax = the 99.99th percentile of the whole volume. Processing functions (`contrast_stretching`, `normalize_image`, `clahe_u8`) rescale between the `(0.1, 99.9)` percentiles by default (`intensity_rescaling.PROCESSING_PERCENTILES`).
* **Channel colours**: one default palette for every viewer and `LabelAnnotator` (white, lime, magenta, yellow, cyan, red, blue); `blending=None` picks additive for colours, maximum for colormaps, and minimum for colormaps that start at white.
* **Deprecations**: legacy names keep working and emit a `FutureWarning` saying they will be removed in eigenp_utils 0.1.0.
* **Output**: library code never prints. Information goes to module loggers (`logging.getLogger("eigenp_utils.<module>")`, nothing is configured on the root logger) and anything you can act on is a `UserWarning`. The drift functions take `progress=` to switch off their progress bars.
* **Imports**: `import eigenp_utils` is lazy: apart from `plotting_utils` (which registers the package colormaps by name), each top-level name imports its module on first use. A name whose optional dependencies are missing raises an ImportError naming the extra to install (e.g. `pip install "eigenp-utils[image-analysis]"`); other import errors are not hidden.

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

Using [`pixi`](https://pixi.sh) (inside a pixi workspace, which needs a conda `python`; `pixi add python` if it has none):

```bash
pixi add --pypi "eigenp_utils[image-analysis,plotting,single-cell] @ git+https://github.com/eigenP/utils.git"
```

This writes the dependency to `[pypi-dependencies]` in `pixi.toml` (or `[tool.pixi.pypi-dependencies]` in `pyproject.toml`). The equivalent manifest entry, if you prefer to edit it by hand:

```toml
[pypi-dependencies]
eigenp_utils = { git = "https://github.com/eigenP/utils.git", branch = "main", extras = ["image-analysis", "plotting", "single-cell"] }
```

`pixi.lock` pins the exact commit, so new commits on `main` are not picked up until you update the lock file:

```bash
pixi update eigenp_utils
```

`pixi update` re-resolves only that package (to the latest commit of the branch) and rewrites `pixi.lock`; the environment is synced on the next `pixi run`/`pixi shell`, or immediately with `pixi install`. For the PyPI release instead of `git+`, use `pixi add --pypi "eigenp-utils[image-analysis,plotting,single-cell]"`; `pixi update eigenp-utils` then moves to the newest version allowed by the manifest constraint, and `pixi upgrade eigenp-utils` also raises that constraint.

### Optional Dependencies

You can choose to install optional dependencies if you need functionality such as single-cell analysis or image analysis:

- `[image-analysis]` - installs `scikit-image`, `xarray` and `tifffile`.
- `[single-cell]` - installs packages like `scanpy`, `pacmap`, `leidenalg`, `seaborn`, etc.
- `[plotting]` - installs `statannotations` and `statsmodels`.
- `[fast]` - installs `numba`, the optional accelerator of `clahe_u8` and `median_filter_fast` (both fall back to NumPy without it).
- `[bonsai]` - installs Bonsai's run-time requirements (`scikit-learn`, `ruamel.yaml`, `psutil`); independent of `[single-cell]`. Bonsai itself is cloned separately (see `eigenp_utils.bonsai_tree`). `[bonsai-mpi]` adds `mpi4py` and the MPICH wheels for parallel runs (`run_bonsai(n_procs=...)`) without a system MPI (Linux/macOS).
- `[all]` - installs all of the optional dependencies above (except `[bonsai-mpi]`) **plus** `[dev]` (pytest, pytest-playwright, marimo, pot). It is meant for development; to use the package, list the groups you need instead.
- `[dev]` - installs all dependencies and additional tools for testing (e.g. `pytest`).

e.g. (uv install)

```bash
uv pip install "eigenp-utils[image-analysis,plotting,single-cell]"
```
*(Note: quotes are required so the shell doesn't misinterpret the brackets.)*

For the latest development versions with optional dependencies:

```bash
pip install "eigenp_utils[image-analysis,plotting,single-cell] @ git+https://github.com/eigenP/utils.git"
```
or
```bash
uv pip install "eigenp_utils[image-analysis,plotting,single-cell] @ git+https://github.com/eigenP/utils.git"
```

Pick the groups you need, e.g. `[single-cell]` or `[image-analysis,single-cell]`.

Each module in `src/eigenp_utils/` also carries a PEP 723 header (`# /// script`) listing its third-party imports, for running a module in a sandbox with `uv run`; a test keeps the headers in step with `pyproject.toml`.

### Browser / WASM (marimo, Pyodide)

Everything except single-cell analysis runs in [Pyodide](https://pyodide.org), the WebAssembly runtime behind marimo's browser notebooks, including the interactive anywidget ortho viewers. CI runs the test suite inside Pyodide (`tests/wasm/`).

| Module(s) | Needs | WASM |
| --- | --- | --- |
| `plotting_utils`, `tnia_plotting_anywidgets`, `label_annotator`, `stats`, `spline_utils`, `io` (file helpers) | core | yes |
| `clahe_equalize_adapthist.clahe_u8`, `median_filter.median_filter_fast` | core (NumPy) + optional `numba` (`[fast]`, several times faster) | yes (NumPy backend) |
| `image_and_labels_utils`, `intensity_rescaling`, `extended_depth_of_focus`, `maxproj_registration`, `surface_extraction` | `scikit-image` | yes |
| `image_io` | `xarray`, `tifffile` | yes |
| `stitching` | `xarray`, `scikit-image` | yes |
| `stats.add_stat_annotations` | `[plotting]` | yes |
| `single_cell` graph helpers (`smooth_expression_on_graph`, `find_correlated_features`, ...) | `anndata`, `scikit-learn` | yes (scanpy is imported lazily) |
| `single_cell` functions that call scanpy / pacmap / leidenalg | `[single-cell]` (`numba` has no WASM build) | no |
| `bonsai_tree` (`BonsaiTree`, `poisson_log_quotients`) / (`run_bonsai`, `tl_bonsai`) | core / `[bonsai]` + a Bonsai clone | yes / no (subprocess) |

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
