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

## 2025-02-19 - Depth of Focus: Interpolating the Unseen Peak

**Learning:** Discrete maximization throws away curvature information.

The `best_focus_image` algorithm calculates a focus score $S(z)$ for each slice and picks $\text{argmax}_z S(z)$. This limits depth resolution to the integer Z-spacing, creating quantization artifacts (staircase depth maps) and suboptimal image reconstruction (always picking a slice, never blending).

However, the focus metric (Laplacian energy) typically follows a Gaussian-like profile near the focus plane. The "true" peak lies between samples.
By assuming a Gaussian profile $S(z) \approx A \exp(-(z-z_0)^2/2\sigma^2)$, the log-score $\ln S(z)$ is parabolic. We can fit a parabola to the top 3 points $(z-1, z, z+1)$ to find the sub-pixel vertex $z^*$.

**Action:**
1.  **Estimation:** Implemented parabolic interpolation on log-scores to estimate fractional depth $z^*$.
2.  **Reconstruction:** Updated the image fusion to linearly interpolate pixel values between slices $\lfloor z^* \rfloor$ and $\lceil z^* \rceil$, weighted by the fractional distance.

**Validation:** On a synthetic stack with focus at $z=4.5$, RMSE improved from $0.50$ (quantization error) to $0.00$ (perfect recovery). Image intensity reconstruction also matched theoretical expectations after fixing a boundary weighting bug.
