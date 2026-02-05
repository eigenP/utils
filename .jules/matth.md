# Matth's Journal

## 2025-02-19 - Drift Correction: The Cost of Integer Rounding

**Learning:** Detection precision means nothing without application precision.

The `estimate_drift_2D` function uses `phase_cross_correlation` with `upsample_factor=100`, providing drift estimates with 0.01-pixel precision. However, the original `apply_drift_correction_2D` rounded these values to the nearest integer before shifting.

This creates a mismatch: we *know* the drift is 0.5 pixels, but we correct it by 0 or 1.
- If we shift by 0: Residual error is 0.5 pixels.
- If we shift by 1: Residual error is 0.5 pixels (overshoot).

For a continuous signal (like an image), the "value" at x=0.5 exists and can be reconstructed via interpolation. Ignoring this is mathematically unsound if the goal is alignment.

**Action:** Upgraded `apply_drift_correction_2D` to support subpixel correction using bicubic interpolation (`order=3`).

**Crucial Detail:** Bicubic interpolation is non-monotone; it can produce values outside the original range (overshoot/ringing) near sharp edges (Gibbs phenomenon).
- If the image is `uint8` (0-255), a value of -5 becomes 251 (wrap-around) if simply cast.
- **Robustness Fix:** Explicitly cast to `float32` before interpolation, then clip the result to the valid range of the data type before casting back. This preserves the topology of the data and prevents "salt-and-pepper" noise at edges.

This moves the algorithm from a discrete grid approximation to a continuous signal reconstruction, aligning with the precision of the detection step.

## 2025-02-19 - Surface Extraction: Moving Beyond the Grid

**Learning:** The "Surface" is a continuous manifold, not a stack of lego bricks.

The original `extract_surface` algorithm relied on `np.argmax` over a boolean mask to find the surface height. This effectively applies a `floor` or `ceil` operation to the true surface position, introducing a hard quantization error of $\sim U(-0.5, 0.5)$ pixels (RMSE $\approx 0.29$).

This quantization destroys sub-pixel information and creates "staircase" artifacts on slanted surfaces, which cannot be recovered by subsequent smoothing without blurring genuine features.

**Action:** Implemented intensity-based subpixel refinement using linear interpolation.
Instead of just finding the index $z$ where $I_z > \text{thresh}$, we calculate the fractional offset $\delta$:
$$ \delta = \frac{\text{thresh} - I_{z-1}}{I_z - I_{z-1}} $$
The refined height is $z - 1 + \delta$.

**Validation:** On a synthetic slanted plane, this reduced the RMSE from quantization levels ($\sim 0.60$ px in the test setup due to aliasing) to $\sim 0.29$ px and allowed the algorithm to correctly distinguish a 0.5 pixel shift, which was previously lost in the integer grid.

## 2025-02-19 - Subpixel EDoF: The Grid is Not The Territory

**Learning:** Discrete signal processing often implicitly assumes sample values exist at specific coordinates (e.g., pixel centers vs corners). `scipy.ndimage.zoom` introduces systematic spatial shifts because its coordinate mapping does not align with the custom patch-grid centers used in the EDoF algorithm.

When upscaling a "sparse" map (like a patch-based heightmap) to a dense grid, standard interpolation functions blindly map -1$ to -1$. If the sparse samples were generated at offsets (e.g., center of patches), this naive mapping introduces a shift $\Delta x \approx \text{patch\_size}/2$, destroying subpixel accuracy.

**Action:** Replaced `zoom` with `scipy.interpolate.RegularGridInterpolator`, explicitly mapping the *exact* centers of the patches to the target pixel grid. This reduced heightmap RMSE from $\sim 0.44$ px to $\sim 0.05$ px.

**Mathematical Insight:** Parabolic interpolation of discrete focus scores ({z-1}, S_z, S_{z+1}$) recovers the continuous peak with high precision, provided the underlying focus metric (Laplacian energy) is locally quadratic near the optimum. This allows reconstructing (z^*)$ via linear interpolation between slices {\lfloor z^* \rfloor}$ and {\lceil z^* \rceil}$, effectively treating the Z-stack as a continuous function (x,y,z)$.

## 2025-02-19 - Drift Correction: The Perils of Coupled Windowing

**Learning:** Windowing for FFT must respect the projection geometry.

The original implementation of `estimate_drift_2D` applied a 2D weighting window (tapering 33% of the image!) to the raw image *before* computing max projections.
This coupled the X and Y axes: a feature located at the top edge (Y=0) was attenuated to zero, making it invisible to the X-projection (which should only care about X-position).
This effectively reduced the Field of View for registration, causing failure when valid features were only present near the boundaries.

**Action:** Decoupled the windowing process.
1. Compute raw Max Projections (capturing full signal intensity regardless of orthogonal position).
2. Apply a 1D Tukey window ($\alpha=0.1$) to the *projections* to satisfy the periodic boundary requirement for the 1D FFT in `phase_cross_correlation`.

**Result:** The algorithm now robustly detects drift even when the only trackable object is at the extreme edge of the FOV (verified by `test_drift_edge_robustness.py`), while still preventing FFT edge ringing.

## 2025-02-20 - Moran's I: The Danger of Hidden Assumptions (Row Standardization)

**Learning:** Optimized formulas often carry implicit assumptions that break in general cases.

The optimized Moran's I implementation used the simplified formula:
$$ I \propto X^T W X - \mu (X^T C) $$
This formulation is **only valid** if the weight matrix $W$ is row-standardized (where row sums $R \approx 1 \implies X^T R \approx N \mu \implies \mu X^T R \approx \mu^2 S_0$).

For general weights (e.g., symmetric binary adjacency or distance-decay), this cancellation fails.
Specifically, if the signal $X$ is correlated with the node degree (e.g., hubs have high expression), the missing term $\mu (X^T R)$ introduces significant bias. In a star graph test case, this caused an error of 0.375 (True -1.0 vs Fast -0.625).

**Action:** Implemented the full expansion of the centered quadratic form:
$$ \text{Num} = X^T W X - \mu (X^T C) - \mu (X^T R) + \mu^2 S_0 $$
This correctly handles any weight structure (symmetric, asymmetric, binary, general) with negligible computational overhead (one extra sparse matrix-vector product).

**Validation:** Verified against the ground-truth formula on Star Graphs (hubs) and Random General Matrices. The new implementation matches exact values while maintaining the speed of the optimized block-wise calculation.
