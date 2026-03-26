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

## 2025-03-23 - Central Differences for Discrete Curve Tangents
**Learning:** Approximating tangent vectors along a discrete parametric curve using forward differences (`np.diff`) introduces a half-step phase shift. The forward difference $t_i = p_{i+1} - p_i$ corresponds to the secant over $[i, i+1]$, which best approximates the tangent at the midpoint $i+0.5$, not at $i$. This systematic misalignment reduces accuracy to $O(h)$ and introduces significant errors in normal plane projection in regions of high curvature. Furthermore, duplicating the last vector to match array lengths artificially forces zero curvature at the boundary.
**Action:** Use central differences (`np.gradient`) to compute tangent vectors at sample points. This provides an $O(h^2)$ approximation of the derivative aligned exactly at $i$ (i.e. $(p_{i+1} - p_{i-1}) / 2$), and handles boundaries mathematically consistently via second-order one-sided differences. Additionally, mask zero-norm tangents when projecting vectors onto the normal plane to prevent catastrophic division by zero and `NaN` propagation.

## 2025-02-28 - Simplification of Moran's I Quadratic Form
**Learning:** In computing the numerator of Moran's I ($I \propto \sum_{i,j} w_{ij} (x_i - \bar{x})(x_j - \bar{x})$), expanding the quadratic form to terms of $x$ (e.g. $x^T W x - \bar{x}(x^T W \mathbf{1}) - \dots$) is computationally inefficient, requires additional memory, and suffers from catastrophic cancellation errors when variables have large means or the graph is irregular.
**Action:** Instead of algebraic expansion, directly center the sufficient statistics first ($z = x - \bar{x}$) and evaluate the cross product directly as $z^T W z$. This eliminates edge cases with irregular topologies (isolated nodes, unstandardized weights), strictly bounds numerical errors around zero, and drastically simplifies implementation complexity while improving runtime.

## 2024-05-18 - [Invalid N-Dimensional Interpolation for Spline Parameterization]
**Learning:** In `spline_utils.py`, the `calculate_vector_difference(overlap_only=True)` function attempted to construct a spatial bounding box using Python's built-in `max()` and `min()` functions on N-dimensional arrays. This is invalid in NumPy and raises a `ValueError: The truth value of an array with more than one element is ambiguous`. Furthermore, even if `np.maximum` and `np.minimum` were used correctly, the original implementation passed an N-dimensional spatial array (the `start` to `end` bounding box diagonal) directly to `np.interp` as the `x` coordinates. `np.interp` evaluates piecewise linear interpolants in 1D space, not N-dimensional space. To compute the difference between the geometric overlap of two parametric splines, one must filter the original points using the geometric bounding box intersection and re-parameterize the cropped segments via B-splines (`splprep`) into the common number of required output points, establishing a proper mapping.
**Action:** Always verify dimensional assumptions of interpolation functions. `np.interp` is strictly for 1D scalar mapping. Use geometric operations to clip curves (e.g. bounding box masking) and re-parameterize them using `splprep`/`splev` when evaluating parametric distances. Never use `max()` or `min()` on NumPy arrays.

## 2024-06-15 - Bias-Corrected Bootstrap and Robust Effect Sizes
**Learning:** Common statistical implementations often rely on estimators that assume infinite data or perfect distributions, leading to biases in small or contaminated samples:
1. **Bootstrap Bias**: The basic percentile method assumes the bootstrap distribution is median-unbiased ($\text{median}(\hat{\theta}^*) = \hat{\theta}$). If the distribution is skewed, the confidence intervals are asymmetrical and under-cover the true parameter. The Bias-Corrected (BC) bootstrap calculates the empirical bias $z_0 = \Phi^{-1}(\text{prop}(\hat{\theta}^* < \hat{\theta}))$ and adjusts the extracted percentiles to $\Phi(2z_0 + z_{\alpha})$ to correct this.
2. **Effect Size Bias**: Cohen's $d$ systematically overestimates the true population effect size for small sample sizes because the sample standard deviation underestimates the population standard deviation. Hedges' $g$ corrects this using an exact gamma-function scaling factor $J(df) = \frac{\Gamma(df/2)}{\sqrt{df/2} \Gamma((df-1)/2)}$.
3. **Outlier Masking**: The standard Z-score uses the sample mean and variance, which have a breakdown point of $1/N$. A single extreme outlier inflates the variance, pulling the Z-score of other true outliers below the detection threshold (masking). Robust Z-scores use Median and MAD, raising the breakdown point to 50%.
**Action:** Default to statistically robust estimators. I updated `bootstrap_ci` to use the BC method, added Hedges' correction as the default for `cohens_d`, and added a `robust_zscore` option to `remove_outliers`.

## 2025-05-18 - Bias-Corrected and Accelerated (BCa) Bootstrap
**Learning:** While Bias-Corrected (BC) bootstrap adjusts for median bias in the bootstrap distribution, it assumes that the standard error of the estimator is constant with respect to the true parameter. If the sampling distribution is skewed (meaning the variance scales with the mean, for instance), BC intervals will systematically under-cover on one side. The Bias-Corrected and Accelerated (BCa) method introduces an acceleration factor, $a$, calculated via jackknife resampling, which models the rate of change of the standard error. Additionally, for discrete statistics like the median, calculating the empirical bias $z_0$ solely as the proportion of bootstrap samples strictly less than the estimate underestimates the true proportion; ties must be handled by adding half the proportion of ties (i.e. `prop_less = mean(x < theta) + 0.5 * mean(x == theta)`).
**Action:** Upgraded `bootstrap_ci` in `stats.py` to support the BCa method (now the default) by computing the jackknife acceleration factor. Fixed the $z_0$ calculation to robustly handle ties, ensuring accurate percentile mapping for discrete estimators.

## 2025-05-20 - Participation Ratio for Parameter-Free Effective Dimensionality
**Learning:** In adaptive $k$-nearest neighbors, curvature mapping is frequently approximated by the eigenvalue spectrum of the local covariance matrix. Previously, the proxy summed the smallest 50% of the eigenvalues to capture "thickness". This assumes that the underlying manifold uses exactly less than half the embedded dimensions. If the intrinsic dimensionality is greater, meaningful variance is erroneously labeled as noise/curvature, destroying the metric.
**Action:** Replace arbitrary percentile-based eigenvalue sum thresholds with the **Participation Ratio**: $PR = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}$. The PR provides a continuous, scale-free, and threshold-free measure of the local covariance matrix's effective dimensionality. A flat manifold has $PR \approx d$, whereas curvature or noise will continuously inflate the PR, properly capturing local complexity without requiring hard cutoffs.

## 2025-05-20 - The "Quantile Trap" in Feature Normalization
**Learning:** Normalizing an array of scalar features (e.g. geometric curvature estimates) and subsequently splitting it into uniformly populated groups using `np.quantile` arbitrarily forces high variance. If an entire dataset consists of perfectly flat, uniform manifolds, forcing 10% of those points into the "highest curvature" bin penalizes points that differ by numerical noise. Rank-based methods discard the absolute magnitude of structure, forcing uniformly distributed output even when the true underlying distribution is tightly clustered around zero.
**Action:** Replace `np.quantile` with absolute linear binning `np.linspace` when assigning data to buckets that determine structural penalties (such as $k$-NN neighbor pruning based on curvature). Absolute binning objectively clusters data relative to the global max/min, preserving the underlying structural distribution and ensuring perfectly flat datasets aren't artificially shattered.

## 2025-05-20 - Winsorization for Outlier-Resistant Absolute Scaling
**Learning:** While absolute linear binning (e.g. `np.digitize(x, np.linspace(min, max))`) successfully resolves the "Quantile Trap" by preserving global structural ratios, it is extremely vulnerable to technical noise. A single extreme outlier (e.g. an isolated cell creating $PR=D$) expands the $ptp$ (peak-to-peak) range so drastically that all valid, meaningful variations in the dataset are compressed into the lowest bin.
**Action:** When performing absolute scaling on empirical biological data or geometric estimates prone to noise, clip (Winsorize) the distribution (e.g., between the 1st and 99th percentiles) before applying min-max normalization. This ensures the linear bins differentiate meaningful variations without being hijacked by single catastrophic artifacts.

## 2025-05-23 - Exact Non-Negative Least Squares for Locally Linear Embedding
**Learning:** When calculating Regularized Locally Linear Embedding (LLE) weights subject to non-negativity and sum-to-one constraints, finding the unconstrained weights via solving the Gram matrix system ($Gw = \mathbf{1}$) and heuristically clipping negative values ($w = \max(w, 0)$) violates the optimality conditions. The clipped weights no longer minimize the reconstruction error.
**Action:** The problem $\min_{w \ge 0, \sum w = 1} \frac{1}{2} w^T G w$ can be mapped exactly to an unconstrained Non-Negative Least Squares (NNLS) problem. By taking the Cholesky decomposition of the regularized Gram matrix $G_{reg} = L L^T$ and solving $L b = \mathbf{1}$, the original LLE problem is mathematically equivalent to $\min_{x \ge 0} ||L^T x - b||^2$. Solving this exact NNLS problem and normalizing the result $w = x / \sum x$ enforces non-negativity rigorously without sacrificing the geometric optimality of the embedding projection.
