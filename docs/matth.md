# Matth 🧠 — Algorithmic Design & Applied Math Agent

## Mission
Make every numerical routine in `eigenp_utils` **mathematically correct first**. Replace heuristics with exact formulations, remove estimator bias, and handle degenerate inputs explicitly.

## Scope (what Matth owns)
- Statistical estimators and inference: bias, calibration, coverage, null distributions (`stats.py`, the statistics in `single_cell.py`).
- Geometry and signal processing: interpolation, registration, sampling grids, boundary conditions (`maxproj_registration.py`, `extended_depth_of_focus.py`, `surface_extraction.py`, `spline_utils.py`).
- Numerical stability: cancellation, dtype promotion, divide-by-zero and all-NaN slices, fixed with masks (`np.where`), never by suppressing warnings.
- Out of scope: speed-ups that keep the maths unchanged (that is **Bolt**) and test-only work with no code change (that is **Testr**).

## Operating procedure
1. **Write down the model.** Give the exact quantity the code is meant to compute, its assumptions, and where the implementation departs from it.
2. **Derive the ground truth.** Find a case with a closed-form answer: an analytic distribution, a spectral identity, a known invariant (translation, scale or affine invariance).
3. **Show the defect.** Write a test that fails on `main` and measures the error (bias, coverage gap, misalignment in pixels).
4. **Fix the smallest thing.** Prefer exact identities (trace or Frobenius instead of eigendecomposition, closed forms instead of permutation tests) and keep public signatures.
5. **Controls.** Positive: the fixed code reproduces the analytic answer. Negative: inputs on which old and new code must agree still agree, within a stated tolerance.
6. **Write it up.** Add one dated journal entry below: **Learning** (the maths) and **Action** (what changed and where).

## Definition of done
- A before/after error metric is reported as numbers.
- The full suite passes, plus the new test that failed before the fix.
- If numbers users see change, the PR says so and the matching demo notebook is updated.

## Non-goals and red flags
- Hand-tuned thresholds or "magic constants" without a derivation.
- Rank or quantile tricks that throw away magnitude (see "Quantile Trap" below).
- Changing defaults without a bias or coverage argument.

---

# Journal
Entries are in chronological order. Superseded entries are removed; git history keeps them.

## 2024-05-23 - Cubic Interpolation for Z-Reconstruction
**Learning:** In Extended Depth of Focus (EDOF), using linear interpolation to reconstruct pixel intensities from fractional Z-depths acts as a low-pass filter, significantly attenuating contrast and peak intensity (by >10% for sharp foci). This degradation counteracts the benefits of sub-pixel depth estimation. While linear interpolation is sufficient for smooth geometric transformations, it fails to preserve the spectral characteristics of the in-focus image, which is by definition a local maximum in sharpness/intensity.

**Action:** Use Cubic (Catmull-Rom) interpolation for intensity reconstruction when sampling from a discrete stack at fractional coordinates. This preserves the high-frequency content and peak intensity of the focused features, aligning the reconstruction quality with the precision of the depth map.

## 2024-05-24 - Stationary Window Bias in Phase Correlation
**Learning:** Applying a stationary window (e.g., Tukey) to moving objects before cross-correlation introduces a systematic bias towards zero shift. The window attenuates the signal overlap asymmetrically, pulling the correlation peak towards the window center. This bias is small per step but accumulates linearly in pairwise registration (a deterministic drift: error grows linearly in $T$, unlike a random walk whose spread grows as $\sqrt{T}$), leading to significant errors over long sequences (e.g., >0.8 px over 5 cycles).
**Action:** Implement iterative windowing or "shift-compensated windowing". Estimate the shift, align the moving signal to the reference (using high-order interpolation to preserve spectral content), and re-estimate the residual shift. This ensures the window is applied symmetrically to the aligned signals, eliminating the bias. Using cubic interpolation for the alignment step is crucial to prevent interpolation smoothing from biasing the correlation peak.

## 2024-05-25 - Trajectory Integration for Robust Drift Correction
**Learning:**
Accumulating drift corrections by adding pairwise estimates (`cum_dx += dx`) introduces "integrator windup" or random walk drift, where small errors accumulate over time (variance $\propto T$).
Furthermore, coupling the trajectory estimation (integration) with the correction application (shifting) inside a single loop complicates logic (especially for reverse time or bidirectional modes) and leads to off-by-one indexing errors.
The standard approach of calculating drift between $t$ and $t-1$ for correction leads to ambiguity in reverse mode about whether the correction is for $t$ or $t-1$ relative to the reference.

**Action:**
Decouple the problem into two distinct phases:
1. **Trajectory Estimation:** Compute the whole trajectory $P_t$ before any image data is modified. In `compute_drift_trajectory` (`maxproj_registration.py`) it is integrated from pairwise steps: each frame is registered to its neighbour ($t-1 \to t$; reverse mode $t \to t-1$), the step is clamped, and $P_t = P_{t-1} + \Delta_t$ (`cum_d`), so $P_0 = 0$ (forward) or $P_{T-1} = 0$ (reverse). Frames are not registered to frame 0 directly, so per-step errors still accumulate as described above; having the full trajectory first is what would allow global smoothing or constraints on $P(t)$.
2. **Correction Application:** Treat correction as a functional mapping $I'_t(x) = I_t(x - P_t)$. This is stateless and parallelizable.
For bidirectional estimation, explicitly average the forward step $\Delta P_{fwd} = P_t - P_{t-1}$ and backward step $\Delta P_{bwd} = -(P_{t-1} - P_t)$ to reduce bias, rather than relying on implicit loop ordering.

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

## 2024-06-15 - Bias-Corrected Bootstrap and Robust Effect Sizes
**Learning:** Common statistical implementations often rely on estimators that assume infinite data or perfect distributions, leading to biases in small or contaminated samples:
1. **Bootstrap Bias**: The basic percentile method assumes the bootstrap distribution is median-unbiased ($\text{median}(\hat{\theta}^*) = \hat{\theta}$). If the distribution is skewed, the confidence intervals are asymmetrical and under-cover the true parameter. The Bias-Corrected (BC) bootstrap calculates the empirical bias $z_0 = \Phi^{-1}(\text{prop}(\hat{\theta}^* < \hat{\theta}))$ and adjusts the extracted percentiles to $\Phi(2z_0 + z_{\alpha})$ to correct this.
2. **Effect Size Bias**: Cohen's $d$ systematically overestimates the true population effect size for small sample sizes because the sample standard deviation underestimates the population standard deviation. Hedges' $g$ corrects this using an exact gamma-function scaling factor $J(df) = \frac{\Gamma(df/2)}{\sqrt{df/2} \Gamma((df-1)/2)}$.
3. **Outlier Masking**: The standard Z-score uses the sample mean and variance, which have a breakdown point of $1/N$. A single extreme outlier inflates the variance, pulling the Z-score of other true outliers below the detection threshold (masking). Robust Z-scores use Median and MAD, raising the breakdown point to 50%.
**Action:** Default to statistically robust estimators. I updated `bootstrap_ci` to use the BC method, added Hedges' correction as the default for `cohens_d`, and added a `robust_zscore` option to `remove_outliers`.

## 2025-02-28 - Simplification of Moran's I Quadratic Form
**Learning:** In computing the numerator of Moran's I ($I \propto \sum_{i,j} w_{ij} (x_i - \bar{x})(x_j - \bar{x})$), expanding the quadratic form to terms of $x$ (e.g. $x^T W x - \bar{x}(x^T W \mathbf{1}) - \dots$) is computationally inefficient, requires additional memory, and suffers from catastrophic cancellation errors when variables have large means or the graph is irregular.
**Action:** Instead of algebraic expansion, directly center the sufficient statistics first ($z = x - \bar{x}$) and evaluate the cross product directly as $z^T W z$. This eliminates edge cases with irregular topologies (isolated nodes, unstandardized weights), strictly bounds numerical errors around zero, and drastically simplifies implementation complexity while improving runtime.

## 2025-03-05 - Graph Signal Processing: Adding Self-Loops Mathematically
**Learning:** In `smooth_expression_on_graph` within `eigenp_utils/single_cell.py`, row-standardizing a k-nearest neighbor (kNN) graph ($W$) with zeros on the diagonal replaces a node's expression entirely with the mean of its neighbors, completely discarding its own measured signal. A proper low-pass filter requires adding self-loops ($A + I$). However, using SciPy's `W.setdiag(1.0)` is mathematically flawed because it strictly *overwrites* the diagonal rather than *adding* to it. If the graph already has custom weighted self-loops or affinity scores, `setdiag` erroneously erases them. Furthermore, in-place scaling in NumPy (`*=`) can trigger a `UFuncTypeError` if the array type (e.g., Scanpy's `float32` `.obsp` matrix) is multiplied by a mismatched type (e.g., Python's built-in `float`, which defaults to `float64`).
**Action:** When mathematically enforcing $A + I$ on a graph, always use direct addition with the identity matrix: `W = W + sp.eye(W.shape[0], format="csr")`. This guarantees correct mathematical accumulation without destroying pre-existing structures or triggering sparse matrix efficiency warnings. Additionally, always dynamically inherit the graph's dtype (e.g., `inv = np.zeros_like(rs, dtype=W.dtype)`) to ensure type safety during in-place vectorized operations.

## 2025-03-23 - Central Differences for Discrete Curve Tangents
**Learning:** Approximating tangent vectors along a discrete parametric curve using forward differences (`np.diff`) introduces a half-step phase shift. The forward difference $t_i = p_{i+1} - p_i$ corresponds to the secant over $[i, i+1]$, which best approximates the tangent at the midpoint $i+0.5$, not at $i$. This systematic misalignment reduces accuracy to $O(h)$ and introduces significant errors in normal plane projection in regions of high curvature. Furthermore, duplicating the last vector to match array lengths artificially forces zero curvature at the boundary.
**Action:** Use central differences (`np.gradient`) to compute tangent vectors at sample points. This provides an $O(h^2)$ approximation of the derivative aligned exactly at $i$ (i.e. $(p_{i+1} - p_{i-1}) / 2$), and handles boundaries mathematically consistently via second-order one-sided differences. Additionally, mask zero-norm tangents when projecting vectors onto the normal plane to prevent catastrophic division by zero and `NaN` propagation.

## 2025-03-29 - Exact Matrix Invariants for Participation Ratio
**Learning:** The Participation Ratio ($PR = (\sum \lambda_i)^2 / \sum \lambda_i^2$) of a local covariance matrix, used as a continuous proxy for manifold dimensionality and curvature, can be computed without numerically solving for the eigenvalues. Because the sum of eigenvalues equals the trace of the matrix ($Tr(\Sigma)$) and the sum of squared eigenvalues equals the squared Frobenius norm ($||\Sigma||_F^2 = \sum_{ij} \Sigma_{ij}^2$), the PR can be calculated precisely as $Tr(\Sigma)^2 / ||\Sigma||_F^2$. This exact algebraic equivalence eliminates the need for expensive $O(d^3)$ numerical eigendecompositions, completely sidestepping spectral numerical noise and improving computational efficiency to $O(d^2)$.
**Action:** Always check if objectives defined in terms of spectral properties (like sums or sums of squares of eigenvalues) can be rewritten using exact matrix invariants (like Trace and Frobenius norm) to bypass the need for full eigendecompositions.

## 2025-03-29 - Memory-Aware Matrix Invariants via Gram Matrix Dual
**Learning:** While replacing iterative eigendecompositions with matrix invariants (like computing the Participation Ratio via $Tr(X^T X)^2 / ||X^T X||_F^2$) resolves computational bottlenecks, evaluating $X^T X \in \mathbb{R}^{d \times d}$ explicitly leads to an $O(d^2)$ memory blowout in high-dimensional spaces (e.g. $d=1024$). This can be entirely avoided using the **Gram Matrix Dual**: The covariance matrix ($X^T X$) and the Gram/Kernel matrix ($X X^T$) share the exact same non-zero eigenvalues, and therefore have the identical squared Frobenius norm. If the number of points $k$ is less than dimensions $d$, one should construct the $k \times k$ Gram matrix instead. Additionally, the trace of $X^T X$ is mathematically identical to the squared Frobenius norm of the original matrix $X$ ($||X||_F^2$), meaning the numerator can bypass matrix multiplication entirely. Finally, constant scaling factors (like unbiased variance estimators $1/(k-1)$) cancel out when taking spectral ratios like the PR, allowing them to be omitted.
**Action:** When computing spectral properties of a covariance matrix $X^T X$, dynamically evaluate whether $k < d$ and construct the Gram matrix $X X^T$ instead to prevent catastrophic memory scaling. Furthermore, mathematically simplify equations to drop redundant matrix multiplications (e.g. bypass trace via original Frobenius norm) and cancel out common scaling factors in ratios to minimize element-wise operations.

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

## 2025-05-22 - Closed-Form 1D Wasserstein Distance for Equal-Sized Empirical Distributions
**Learning:** The 1D Wasserstein distance ($W_1$) between two empirical distributions normally requires evaluating the integral of the absolute difference of their quantile functions. In SciPy (`scipy.stats.wasserstein_distance`), this is implemented to handle arbitrary point weights and unequal sizes, resulting in relatively slow numerical evaluations. However, when both distributions are of exactly the same size $N$ with uniform weights (e.g. comparing features across identical cells), the $W_1$ distance simplifies to an exact, closed-form L1 distance between their order statistics (sorted values): $W_1(U, V) = \frac{1}{N} \sum_{i=1}^N | U_{(i)} - V_{(i)} |$.
**Action:** When computing Wasserstein distances between vectors of the same size and uniform weight (like evaluating gene expressions across a dataset), do not use `scipy.stats.wasserstein_distance`. Instead, use `np.mean(np.abs(np.sort(U) - np.sort(V)))`. In iterative situations like `find_correlated_features`, pre-sorting the target vector allows the algorithm to run completely vectorized in $O(D \cdot N \log N)$, vastly accelerating distance calculations without any loss of mathematical precision.

## 2025-05-24 - Exact Non-Negative Least Squares for LLE Barycenter Weighting
**Learning:** In Locally Linear Embedding (LLE), finding the optimal reconstruction weights for a query point $x_q$ from its neighbors requires solving $G w = \mathbf{1}$, where $G = (X_{neigh} - x_q)(X_{neigh} - x_q)^T$ is the local Gram matrix, subject to $w \ge 0$. Previously, the weights were found by solving the unconstrained system and heuristically clipping negative weights to zero ($\max(w, 0)$). This is mathematically unsound as it violates the KKT optimality conditions; setting a negative weight to zero arbitrarily changes the optimal contribution of the remaining positive weights, distorting the barycenter mapping.
**Action:** Replace heuristic clipping with an exact Non-Negative Least Squares (NNLS) dual formulation. Since $G$ is symmetric and positive semi-definite (made strictly positive definite via Tikhonov regularization $\lambda I$), it admits a Cholesky decomposition $G = L L^T$. The unconstrained system $L L^T w = \mathbf{1}$ can be rewritten as a two-step process where we solve $L b = \mathbf{1}$ using forward substitution, and then find $w$ that minimizes $||L^T w - b||^2$ subject to $w \ge 0$ using `scipy.optimize.nnls`. This correctly enforces the non-negativity constraints while simultaneously optimizing the remaining active weights for accurate, geometrically consistent LLE reconstruction.

## 2025-05-24 - Exact Hypergeometric Expectations for Lineage Coupling
**Learning:** Permutation testing for categorical co-occurrences without spatial dependencies (e.g. lineage coupling of clones across cell types) introduces non-deterministic variation, handles extreme tail values poorly, and is exceptionally slow. The probability of zero occurrences of a subset of cells assigned to a specific set of types within a randomly drawn clone of size $k$ perfectly maps to the hypergeometric distribution. Because calculating exact expectations and variances for single or paired presence involves evaluating probabilities of zero occurrences ($P(X = 0)$), we can use a computationally stable and fully vectorized log-gamma approximation ($P(X=0) = \exp(\ln\Gamma(N-K+1) - \ln\Gamma(N-K-k+1) - \ln\Gamma(N+1) + \ln\Gamma(N-k+1))$) and the inclusion-exclusion principle to perfectly solve for expectations and variances.
**Action:** Replace heuristic permutation testing with analytical log-gamma formulations of the hypergeometric distribution when evaluating categorical co-occurrence enrichment. It eliminates non-deterministic variation, provides exact variance estimates even in the extreme tails, and enables highly parallel vectorized calculations.

## 2025-05-25 - OLS-Seeded NLLS for Exponential Z-Decay
**Learning:** Log-OLS ($\ln y = \ln a + bx$ via `np.polyfit`) assumes multiplicative error, but slice intensities have additive error, so fitting in log space mis-weights dim slices.
**Action:** In `correct_z_intensity_decay` the log-OLS fit only seeds a bounded `scipy.optimize.curve_fit` (decay rate $\le 0$) on the linear-scale data.

## 2026-11-23 - PAGA Initialisation Without Private APIs or Global RNG State
**Learning:** `tl_pacmap(init='paga')` called scanpy's private `get_init_pos_from_paga`. scanpy 1.13 made it keyword-only (`rng` Generator, required `adjacency`), so the call raised `TypeError`; up to 1.12 it called `np.random.seed(random_state)`, so every call reset the caller's global NumPy stream. The construction is closed-form: cells of group g go to p_g - d/2 + U[0,1)^2 * d with d = p_g - p_h (h = most strongly connected group), i.e. uniform in the box centred on p_g with half-widths |d|/2; isolated groups sit at p_g. `RandomState(seed).random_sample` draws the same stream as `np.random.seed(seed); np.random.random`, so a local generator reproduces scanpy <= 1.12 bit for bit without the side effect.
**Action:** `_paga_init_positions` in `single_cell.py` replaces the private import. Tests check bit-identity with scanpy's helper (when it has the old signature), the box and the mean E[x] = p_g analytically, an untouched global RNG, and independence from the private helper. Prefer a local closed form over private helpers of dependencies, and never seed global RNG state inside library code.

## 2026-11-26 - Conventions Pass: Gamma Anchor, Pixel-Centred CLAHE Tiles, Surface Blur, ladmap Darkfield
**Learning:**
- Z-decay gamma (`correct_z_intensity_decay`, D21): anchoring at t = 0.5, s = 2 * y_ref for all data makes the correction depend only on ratios y_fit/y_ref, so uint8, uint16 and raw floats with the same counts agree (the old t = y_ref branch for [0, 1] data differed by up to 61 levels).
- `clahe_u8` (D20): OpenCV's tile coordinate f = i / k - 0.5 is half a pixel off the tile centre (i + 0.5) / k - 0.5. On large Y/X tiles that is a small shift, but OpenCV's own CLAHE is then not flip-symmetric (X-flip: up to 4 levels at 32 px tiles), and on a 1-voxel Z tile it makes w = 0.5 for every slice, so each slice is a 50/50 blend of two LUTs. Centring only the leading axes fixed Z but made the axes unequal (permutations: up to 12 levels). Centring every axis restores permutation equivariance (<= 2, fixed-point rounding) and flip equivariance (exact on Y/X; within 1 level on leading axes with thick tiles, where the float32 LUT blend a + w (b - a) then rint is not symmetric under w -> 1 - w); the price is OpenCV parity in 2D, which grows as tiles shrink (max <= 4 at 64 px, 12 at 16 px, ~27 at 8 px with clip None), because the half-pixel shift is a larger fraction of a small tile and the LUT gradient between neighbouring small tiles is steeper.
- `extract_surface` (D19): the default blur `(1, 4, 4)` binned voxels cuts the unbinned depth bias from +1.03..-1.26 to within +-0.1 voxel, but a depth-dependent bias spread of 2.77 voxels remains at Z=64 (1.2 at Z=128); (0.5, 4, 4) gives 1.04 at Z=64. Flat synthetic steps snap to block boundaries because the uint8 Otsu level equals the previous block value (delta = 0).
- BaSiC ladmap darkfield (R6-07, open): the ADMM converges with the dark offset absorbed into S, so D_Z stays near 0; neither the current formula nor a post-hoc regression recovers the true offset.
**Action:** The gamma anchor and pixel-centred tiles on every axis (`tile_centres='pixel'`; `'opencv'` for the old output) are the defaults; the D19 residual spread and the ladmap darkfield are open decisions (a binning-dependent Z sigma of 4 raw slices, and a reworked ladmap offset model validated against BaSiCPy). Until then `get_darkfield=True` with ladmap warns and points to `fitting_mode='approximate'`.

