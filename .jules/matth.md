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

## 2025-02-19 - Depth of Focus: The Map is the Territory (Only if Aligned)

**Learning:** Patch-based processing creates a disconnect between the "index space" of the results and the "physical space" of the image.

The `best_focus_image` function computes optimal depth indices for discrete patches. When upsampling this patch-level map to the full image size using standard `scipy.ndimage.zoom`, the function implicitly maps the patch grid range $[0, N_{patches}-1]$ to the pixel range $[0, H-1]$.
However, the center of the first patch is at $y = P/2$, not $y=0$. This mismatch introduces a systematic spatial shift of $\approx P/2$ pixels, which manifests as a significant depth error on slanted surfaces (RMSE ~0.93 px).

**Action:** Replaced `zoom` with explicit coordinate mapping using `scipy.ndimage.map_coordinates`.
We calculate the precise index $k$ in the patch grid for every pixel $y$:
$$ k = \frac{y - P/2}{S} $$
where $S$ is the stride. This aligns the "map" with the "territory".

**Result:** Combined with parabolic peak interpolation (converting discrete Z-slices to continuous depth), the RMSE dropped from 0.93 px (dominated by alignment error) to 0.03 px. This validates that the "Depth Map" is now a faithful, sub-pixel accurate representation of the object's geometry, not just a processing artifact.
