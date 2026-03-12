## 2024-05-22 - Generalized Moran's I Formula for Irregular Graphs
**Learning:** The standard simplified formula for Moran's I ($I = \frac{N}{S_0} \frac{x^T W x}{x^T x}$) assumes either row-standardized weights ($S_0=N$) OR ignores the impact of the mean ($\bar{x}$) on the cross-product term for irregular graphs. Specifically, the expansion of the numerator $\sum_{ij} w_{ij} (x_i - \bar{x})(x_j - \bar{x})$ includes terms $-\bar{x} \sum_i x_i R_i$ and $-\bar{x} \sum_j x_j C_j$ (where $R_i, C_j$ are row/col sums of $W$). If the graph has isolated nodes ($R_i=0$) or general weights ($R_i \neq 1$), these terms do not cancel out with $+S_0 \bar{x}^2$ as they do in the regular/standardized case. Neglecting them leads to bias when the mean is non-zero, making the statistic sensitive to simple shifts in data.

**Action:** When implementing spatial statistics, always use the general quadratic form expansion rather than simplified versions that rely on implicit assumptions about graph regularity or weight normalization. I updated `morans_i_all_fast` to include the full expansion: $\text{Num} = x^T W x - \bar{x}(x^T W \mathbf{1}) - \bar{x}(\mathbf{1}^T W x) + \bar{x}^2 S_0$. This ensures correctness for any weight matrix (binary, inverse-distance, etc.) and any topology (including disconnected components).

## 2024-05-23 - Cubic Interpolation for Z-Reconstruction
**Learning:** In Extended Depth of Focus (EDOF), using linear interpolation to reconstruct pixel intensities from fractional Z-depths acts as a low-pass filter, significantly attenuating contrast and peak intensity (by >10% for sharp foci). This degradation counteracts the benefits of sub-pixel depth estimation. While linear interpolation is sufficient for smooth geometric transformations, it fails to preserve the spectral characteristics of the in-focus image, which is by definition a local maximum in sharpness/intensity.

**Action:** Use Cubic (Catmull-Rom) interpolation for intensity reconstruction when sampling from a discrete stack at fractional coordinates. This preserves the high-frequency content and peak intensity of the focused features, aligning the reconstruction quality with the precision of the depth map.

## 2024-05-24 - Stationary Window Bias in Phase Correlation
**Learning:** Applying a stationary window (e.g., Tukey) to moving objects before cross-correlation introduces a systematic bias towards zero shift. The window attenuates the signal overlap asymmetrically, pulling the correlation peak towards the window center. This bias is small per step but accumulates linearly in pairwise registration (random walk drift), leading to significant errors over long sequences (e.g., >0.8 px over 5 cycles).
Wait, "random walk" usually implies variance growing as sqrt(T), but here the bias is *linear* with T.
**Action:** Implement iterative windowing or "shift-compensated windowing". Estimate the shift, align the moving signal to the reference (using high-order interpolation to preserve spectral content), and re-estimate the residual shift. This ensures the window is applied symmetrically to the aligned signals, eliminating the bias. Using cubic interpolation for the alignment step is crucial to prevent interpolation smoothing from biasing the correlation peak.

<<<<<<< matth/decouple-drift-estimation-2279110621154388654
## 2024-05-25 - Trajectory Integration for Robust Drift Correction
**Learning:**
Accumulating drift corrections by adding pairwise estimates (`cum_dx += dx`) introduces "integrator windup" or random walk drift, where small errors accumulate over time (variance $\propto T$).
Furthermore, coupling the trajectory estimation (integration) with the correction application (shifting) inside a single loop complicates logic (especially for reverse time or bidirectional modes) and leads to off-by-one indexing errors.
The standard approach of calculating drift between $t$ and $t-1$ for correction leads to ambiguity in reverse mode about whether the correction is for $t$ or $t-1$ relative to the reference.

**Action:**
Decouple the problem into two distinct phases:
1. **Trajectory Estimation:** Calculate the global absolute position $P_t$ for every frame relative to a fixed reference (Frame 0). This allows global smoothing or constraints to be applied to the trajectory $P(t)$ before any image data is modified.
2. **Correction Application:** Treat correction as a functional mapping $I'_t(x) = I_t(x - P_t)$. This is stateless and parallelizable.
For bidirectional estimation, explicitly average the forward step $\Delta P_{fwd} = P_t - P_{t-1}$ and backward step $\Delta P_{bwd} = -(P_{t-1} - P_t)$ to reduce bias, rather than relying on implicit loop ordering.
=======
## 2024-05-25 - Spatial Shift Bias in Binned Surface Extraction
**Learning:** Using `scipy.ndimage.zoom` or standard interpolation on binned data introduces a systematic spatial shift of 0.5 reduced pixels ($\approx S/2$ original pixels) because `zoom` assumes input samples are at integers $0, 1, \dots$ rather than block centers $(i+0.5)S - 0.5$. This causes misalignment between the extracted surface and the original image. Additionally, interpolating against a sentinel value (e.g., -1.0) causes catastrophic boundary artifacts ("curl down") where valid surface values are pulled towards the sentinel, creating false surfaces.

**Action:** Replace `ndimage.zoom` with `RegularGridInterpolator` using explicit block-center coordinates for the source grid. Before upscaling, inpaint invalid regions with the nearest valid neighbor to define proper boundary conditions (0th-order extension) for cubic interpolation, preventing ringing and boundary corruption. Correct the Z-scaling formula to $Z_{full} = (Z_{red} + 0.5)S - 0.5$ to account for bin centering.

## 2024-05-26 - Topological Filtering for Surface Robustness
**Learning:**
Extracting surfaces via `argmax` (finding the first threshold-crossing voxel) is inherently sensitive to disconnected outliers ("floating debris"). A single bright voxel above the true surface causes a catastrophic false positive for that column.
Simple median filtering or Gaussian smoothing is insufficient because outliers can be bright enough to persist through linear filters.
The "surface" is by definition a large, coherent structure.

**Action:**
Enforce topological coherence by identifying connected components in the thresholded volume. Filter out any component that is significantly smaller than the largest foreground component (e.g., $< 10\% S_{max}$). This robustly rejects floating debris regardless of intensity, provided it is spatially disconnected from the main body.

## 2024-05-26 - Package Import Standards for Tests
**Learning:**
Tests should not import from `src.package_name` (e.g., `from src.eigenp_utils...`). This fails in CI environments where the package is installed as `eigenp_utils`.
Using `from eigenp_utils...` is correct but requires the package to be installed in the local environment (e.g., via `pip install -e .` or `PYTHONPATH` manipulation) to run tests locally.

**Action:**
Always write tests assuming the package is installed (`from package_name import ...`). To run tests locally, ensure the package is installed in editable mode.
>>>>>>> main

## 2025-02-28 - Simplification of Moran's I Quadratic Form
**Learning:** In computing the numerator of Moran's I ($I \propto \sum_{i,j} w_{ij} (x_i - \bar{x})(x_j - \bar{x})$), expanding the quadratic form to terms of $x$ (e.g. $x^T W x - \bar{x}(x^T W \mathbf{1}) - \dots$) is computationally inefficient, requires additional memory, and suffers from catastrophic cancellation errors when variables have large means or the graph is irregular.
**Action:** Instead of algebraic expansion, directly center the sufficient statistics first ($z = x - \bar{x}$) and evaluate the cross product directly as $z^T W z$. This eliminates edge cases with irregular topologies (isolated nodes, unstandardized weights), strictly bounds numerical errors around zero, and drastically simplifies implementation complexity while improving runtime.

## 2024-05-18 - [Invalid N-Dimensional Interpolation for Spline Parameterization]
**Learning:** In `spline_utils.py`, the `calculate_vector_difference(overlap_only=True)` function attempted to construct a spatial bounding box using Python's built-in `max()` and `min()` functions on N-dimensional arrays. This is invalid in NumPy and raises a `ValueError: The truth value of an array with more than one element is ambiguous`. Furthermore, even if `np.maximum` and `np.minimum` were used correctly, the original implementation passed an N-dimensional spatial array (the `start` to `end` bounding box diagonal) directly to `np.interp` as the `x` coordinates. `np.interp` evaluates piecewise linear interpolants in 1D space, not N-dimensional space. To compute the difference between the geometric overlap of two parametric splines, one must filter the original points using the geometric bounding box intersection and re-parameterize the cropped segments via B-splines (`splprep`) into the common number of required output points, establishing a proper mapping.
**Action:** Always verify dimensional assumptions of interpolation functions. `np.interp` is strictly for 1D scalar mapping. Use geometric operations to clip curves (e.g. bounding box masking) and re-parameterize them using `splprep`/`splev` when evaluating parametric distances. Never use `max()` or `min()` on NumPy arrays.
