## 2025-05-22 - Replacing Ad-Hoc Softmax with Probabilistic Superiority

**Learning:** The function `annotate_clusters_by_markers` previously used a softmax function with an arbitrary temperature parameter (`beta=2.0`) applied to robustly normalized scores (median/MAD) to estimate "confidence".
This approach had two flaws:
1.  **Arbitrary Scaling:** The `beta` parameter implies a specific belief about the signal-to-noise ratio of the *global* normalization that may not hold for individual clusters.
2.  **Global vs. Local Variance:** Normalization used global MAD, but confidence is a function of *local* separability within the cluster. A tight cluster should have higher confidence for the same mean separation than a loose one. Using global variance ignores this.

**Action:** Replaced the softmax heuristic with the **Probability of Superiority** (also known as Common Language Effect Size).
We compute $P(S_{top1} > S_{top2})$ for a random cell in the cluster by modeling the pairwise difference of scores $D = S_{top1} - S_{top2}$ as normally distributed: $D \sim \mathcal{N}(\mu_D, \sigma_D^2)$.
The confidence is then $\Phi(\mu_D / \sigma_D)$, where $\Phi$ is the standard normal CDF.
This is a parameter-free, statistically grounded metric that naturally adapts to the intra-cluster variance.

## 2025-05-23 - Scale Mismatch in Discrete Regularization

**Learning:** The focus stacking algorithm used a fixed-size median filter (`disk(3)`, diameter 7) on the reconstructed depth map index grid. However, the grid resolution depends on `patch_size` (derived from image size). For default patch sizes, the grid can be very coarse (e.g., 10x10). A 7x7 filter on such a grid is a global operator, not a local regularizer, effectively flattening the depth map and erasing all features smaller than ~50% of the image width.

**Action:** Regularization kernels must be scaled relative to the signal resolution (grid size), or chosen conservatively (`3x3`) to remove outliers without enforcing global smoothness. When discretizing a continuous field (depth) onto a coarse grid, feature preservation requires minimal kernel sizes.
## 2025-05-22 - Fixing Integrator Windup in Drift Correction

**Learning:** The `apply_drift_correction_2D` function was accumulating drift using integer quantization at each time step (`cum_dx = int(cum_dx + dx)`).
This is a classic control theory failure mode known as **Integrator Windup** (or quantization deadband).
If the physical drift is slow (e.g., 0.4 pixels/frame) and the estimator returns sub-pixel shifts (or is forced to integer 0 due to low precision), the `int()` cast truncates the signal to 0 at every step.
The cumulative drift remains 0 indefinitely, failing to correct significant total drift (e.g., 40 pixels over 100 frames).

**Action:**
1.  **Continuous State:** Accumulate drift in floating point (`cum_dx += dx`) to preserve fractional contributions.
2.  **Actuation Quantization:** Perform rounding/quantization *only* at the actuation step (shifting the image), i.e., `shift_amount = round(cum_dx)`.
3.  **Subpixel Estimation:** Increase estimator precision (`upsample_factor=100`) to ensure `dx` captures fractional drift, preventing the "measurement deadband" problem.
This ensures that small, consistent biases integrate up to a correction step, mathematically guaranteeing zero steady-state velocity error.

## 2025-05-23 - Sub-pixel Peak Finding and Coordinate Correctness

**Learning:**
1.  **Quantization Artifacts:** Using integer `argmax` for focal plane detection introduces step-like quantization errors ($\sigma^2 \approx 1/12$ slice) in the depth map.
2.  **Coordinate Misalignment:** Upscaling patch-based parameter maps (like focus scores) using naive `zoom` causes significant spatial distortion because it assumes corner-to-corner alignment rather than mapping patch centers to pixels.
    *   Patch center: $y_c = i \cdot \text{stride} + \text{offset}$.
    *   Naive zoom implies $y = i \cdot \text{scale}$.
    *   The offset is ignored, leading to sub-patch shifts that dominate error metrics.

**Action:**
1.  **Parabolic Refinement:** Implemented parabolic interpolation around the integer peak to recover continuous depth values. Formula: $\delta = 0.5 \frac{S_{l} - S_{r}}{S_{l} - 2S_{c} + S_{r}}$.
2.  **Explicit Projection:** Replaced `scipy.ndimage.zoom` with `map_coordinates`. We explicitly calculate the fractional index $i = (y_{pixel} - \text{offset}) / \text{stride}$ for each pixel, ensuring the high-resolution map is geometrically aligned with the input patches. This reduced RMSE on a synthetic slant test from ~0.42 to ~0.03 slices.
