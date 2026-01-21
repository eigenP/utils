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

## 2025-05-18 - Empirical Probability of Superiority in Single-Cell Annotation

**Learning:** Parametric estimators (like Z-scores) fail catastrophically for single-cell confidence metrics because marker gene scores are often zero-inflated, skewed, or contain massive outliers. A single outlier can drag a mean-based Z-score to 0 (indicating "no confidence") even if 99% of cells show a clear signal. Additionally, robust normalization (Median/MAD) can amplify noise in sparse vectors (where MAD is near zero) to infinity.

**Action:** Prefer the empirical "Probability of Superiority" (Common Language Effect Size), calculated as `mean(Score_A > Score_B)`, over parametric Gaussian approximations. Avoid default normalization of scores that lack variance; trust the raw sign/rank signal over magnitude when data is sparse.
