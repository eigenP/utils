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

## 2025-02-21 - Log-Parabolic Interpolation: Respecting the Signal Model

**Learning:** A Parabola is a poor model for a Gaussian.

Most focus metrics (e.g., squared Laplacian) exhibit a Gaussian-like decay ($e^{-(z-z_0)^2}$) near the focal plane. Standard 3-point parabolic interpolation assumes the signal itself is quadratic ($S(z) \approx a z^2 + b z + c$). This mismatch biases the peak estimation, systematically underestimating the peak width or shifting the center for asymmetric sampling.

Fitting a parabola to the **logarithm** of the scores ($\ln S(z) \approx -(z-z_0)^2$) recovers the underlying Gaussian parameters exactly.

**Action:** Upgraded `_get_fractional_peak` in EDoF to perform parabolic fitting on `np.log(score + eps)`.

**Validation:** RMSE on a synthetic Gaussian-weighted slanted plane dropped from **0.0523 px** to **0.0180 px** (a ~3x improvement), confirming the theoretical superiority of the estimator.

## 2025-02-21 - CLAHE Monotonicity: Zero means Zero

**Learning:** Discontinuous parameters are user traps.

The `clip_limit` parameter in CLAHE had a sentinel value: `0` meant "Maximum Contrast" (Unlimited AHE), while `epsilon` meant "Minimum Contrast" (Identity). This discontinuity violated the principle of monotonicity ("higher limit -> more contrast") and confused the mathematical definition of a limit.

**Action:** Redefined `clip_limit` logic:
- `clip_limit=0.0`: Identity (No contrast enhancement).
- `clip_limit=1.0` (or `None`): Standard AHE (Max contrast enhancement).
- Values in between scale monotonically.

This restores the mathematical intuition that a limit of zero implies zero effect.
## 2025-02-19 - Drift Correction: The Perils of Coupled Windowing

**Learning:** Windowing for FFT must respect the projection geometry.

The original implementation of `estimate_drift_2D` applied a 2D weighting window (tapering 33% of the image!) to the raw image *before* computing max projections.
This coupled the X and Y axes: a feature located at the top edge (Y=0) was attenuated to zero, making it invisible to the X-projection (which should only care about X-position).
This effectively reduced the Field of View for registration, causing failure when valid features were only present near the boundaries.

**Action:** Decoupled the windowing process.
1. Compute raw Max Projections (capturing full signal intensity regardless of orthogonal position).
2. Apply a 1D Tukey window ($\alpha=0.1$) to the *projections* to satisfy the periodic boundary requirement for the 1D FFT in `phase_cross_correlation`.

**Result:** The algorithm now robustly detects drift even when the only trackable object is at the extreme edge of the FOV (verified by `test_drift_edge_robustness.py`), while still preventing FFT edge ringing.

## 2024-05-23 - Per-Pixel Sampling in EDOF
**Learning:** Patch-based image reconstruction (stitching weighted patches) introduces visible block artifacts when the underlying focus surface is continuous but slanted. Per-pixel sampling (x,y) = V(x,y, Z(x,y))$ using a smoothly interpolated height map eliminates these artifacts and respects the sub-pixel continuity of the focus manifold. Additionally, reflection padding creates artificial derivative discontinuities at boundaries which distort spline interpolation; filtering boundary points allows for safer linear extrapolation of the surface.
**Action:** Replace patch blending with vectorized per-pixel sampling in EDOF. Ensure interpolation grid excludes padding-dominated points.
