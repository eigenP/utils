# eigenp-utils

`eigenp-utils` is a comprehensive toolkit of helper utilities for scientific Python. It provides modules for image analysis, single-cell data processing, advanced plotting, and core Python utilities.

## Features

### Image Analysis
* **Extended Depth of Focus (EDOF)**: Reconstruct focused 2D images from 3D stacks with high accuracy using log-parabolic interpolation of focus scores and continuous surface sampling.
* **Surface Extraction**: Robust extraction of 2D surfaces from 3D volumes. Includes topological filtering (Connected Components Analysis) to handle debris, nearest-neighbor inpainting for invalid regions, and precise upscaling via `RegularGridInterpolator`.
* **Registration & Drift Correction**: Bidirectional 2D drift correction (`apply_drift_correction_2D`, `compute_drift_trajectory`), and iterative shift-compensated windowing (`maxproj_registration`) to eliminate systematic biases and achieve sub-pixel stability.
* **Intensity Rescaling**: Tools for contrast enhancement, including CLAHE.

### Plotting & Visualization
* **Interactive 3D Widgets**: Jupyter-compatible, `anywidget`-based orthogonal slicers (`TNIASliceWidget`) and dynamic point cloud visualization (`IsoScatterWidget`) with interactive channel visibility and manual matrix-driven rotation.
* **Publication-Ready Plots**: `raincloud_plot` supporting Seaborn-style arguments (grouped and colored with automatic position dodging), and utility functions to generate SVGs.

### Single-Cell Analysis
* **Robust Cluster Annotation**: Score cell types via the Empirical Probability of Superiority ($P(S_1 > S_2)$) to ensure robustness against outliers and non-normal distributions (`annotate_clusters_by_markers`).
* **Spatial Autocorrelation**: Fast Moran's I implementation (`morans_i_all_fast`) that correctly handles general (non-row-standardized) spatial weights.

### Core Utilities
* **I/O Utilities**: Functions to streamline file and data reading.
* **Dimensionality Parser**: Robust dimensionality inference for arbitrary N-dimensional functions, using unique shape probing to solve ambiguity (`dimensionality_parser`).
* **Task Calendar Scheduler**: Interactive schedule plotting application powered by Marimo, Plotly, and pandas for timeline visualization of linked events and deadlines (`task_calendar_scheduler`).

## Installation

By default, the package installs a minimal set of dependencies (like `numpy`, `scipy`, `pandas`, `matplotlib`, etc).
To install it, run:

```bash
pip install "eigenp_utils @ git+https://github.com/eigenP/utils.git"
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
uv pip install "eigenp-utils[all] @ git+https://github.com/eigenP/utils.git"
```
*(Note: quotes are required so the shell doesn't misinterpret the brackets.)*



You can replace `[all]` with other groups like `[single-cell]` or `[image-analysis,single-cell]` depending on your specific needs.

## Usage Examples

Here are some brief examples of how to use `eigenp-utils`:

**Interactive Schedule Plotting:**
```python
import marimo
from eigenp_utils.task_calendar_scheduler import Calendar, Event, plot_calendar
from datetime import datetime, timedelta

events = [
    Event("Task A", datetime(2025, 5, 12, 9, 0), timedelta(hours=2), group="Project 1", resource="Blue")
]
cal = Calendar(events)
# Plots the calendar using Plotly inside marimo or Jupyter
fig = plot_calendar(cal)
```

**Robust Dimensionality Inference:**
```python
from eigenp_utils.dimensionality_parser import dimensionality_parser

@dimensionality_parser(target_dims='YX')
def my_2d_filter(image_2d):
    # Process a 2D slice
    return image_2d * 2

# Apply to a 4D image (T, Z, Y, X) and it automatically iterates over T and Z
result_4d = my_2d_filter(image_4d)
```

**Single-Cell Cluster Annotation:**
```python
from eigenp_utils.single_cell import annotate_clusters_by_markers
# Assuming `adata` is a Scanpy AnnData object
markers = {'T-cells': ['CD3D', 'CD3E'], 'B-cells': ['CD79A', 'MS4A1']}
annotate_clusters_by_markers(adata, markers, cluster_key='leiden')
```

## Contributing

We welcome contributions! Please follow these steps to contribute:
1. Fork the repository.
2. Install the package in development mode with `pip install -e ".[dev]"`.
3. Create a new branch for your feature or bugfix.
4. Ensure all tests pass by running `pytest tests/`.
5. Submit a pull request detailing your changes.

## License

License CC BY-NC https://creativecommons.org/licenses/by-nc/4.0/
