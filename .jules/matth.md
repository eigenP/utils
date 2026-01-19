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

## 2025-02-20 - Focus Stacking: Sub-pixel Peak Estimation and Geometric Alignment

**Learning:** Energy-based focus measures behave as Gaussian profiles, and geometric alignment requires phase correction.

1.  **Log-Domain Interpolation:** The Laplacian Energy focus metric exhibits a Gaussian-like decay away from the focal plane ($E \propto e^{-z^2}$). Standard parabolic interpolation on the raw energy scores ($E$) is biased because a Gaussian is not a parabola. Fitting a parabola to the *log-energy* ($\ln E \propto -z^2$) aligns with the theoretical model, recovering the peak $z$ with significantly higher accuracy.

2.  **Geometric Phase Shift:** When estimating properties on a sparse patch grid, simply upsampling the result (e.g., `scipy.ndimage.zoom`) implicitly assumes the first sample aligns with the first pixel index. However, patch centers are offset by `patch_size // 2`. Ignoring this offset introduces a spatial shift in the reconstructed map. Using explicit coordinate mapping (`map_coordinates`) that accounts for the sampling grid's phase is essential for geometrically correct reconstruction.

**Action:** Upgraded `best_focus_image` to use log-domain parabolic refinement for sub-pixel depth estimation and replaced `zoom` with grid-aware `map_coordinates` interpolation.
