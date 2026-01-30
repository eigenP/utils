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

## 2025-02-19 - Drift Registration: Decoupling Orthogonal Projections

**Learning:** Windowing for FFT must be applied in the domain of the correlation (1D), not the source (2D), to preserve orthogonality.

The previous implementation of `estimate_drift_2D` applied a 2D window (tapering at edges) to the image *before* computing max projections.
$$ P_x(x) = \max_y [ I(x,y) \cdot w_x(x) \cdot w_y(y) ] $$
This coupled the axes: an object moving in Y (into the taper) would see its intensity in the X-projection modulated by $w_y(y)$. This amplitude modulation creates artificial "flicker" or shape changes in the projection signal, acting as noise or bias for the phase correlation, even if the object is stationary in X.

**Action:** Refactored the pipeline to project the raw image first, then apply the 1D window to the projection.
$$ P_x(x) = [\max_y I(x,y)] \cdot w_x(x) $$
This ensures that $P_x(x)$ is invariant to any shifts in Y (assuming the object stays within the FOV), making X-drift estimation mathematically independent of Y-motion.

**Result:** Eliminated "cross-talk" where Y-motion induced false X-drift measurements (reduced error from ~0.02 px to 0.00 px in synthetic tests). Also improved memory efficiency by removing the need for large 2D float buffers.
