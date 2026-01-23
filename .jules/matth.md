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

## 2025-02-19 - Extended Depth of Field: The Geometry of Patches

**Learning:** Patch-based focus measures are sampled on a sparse grid; treating them as a pixel-aligned image requires rigorous geometric projection.

The previous implementation of `best_focus_image` calculated focus scores on a grid of patch centers but used `scipy.ndimage.zoom(order=0)` to upsample this to the full image. This introduced two critical errors:
1.  **Quantization**: Peak Z was an integer.
2.  **Geometric Misalignment**: `zoom` stretches the array bounds 0..N to 0..M, ignoring the fact that patch centers are inset by `patch_size//2`. This shift caused the height map to be spatially misaligned with the image features.

**Action:**
1.  **Sub-pixel Peak Finding**: Implemented log-parabolic interpolation to refine the Z-peak, treating the focus metric as a sampled Gaussian.
2.  **Correct Coordinate Mapping**: Replaced `zoom` with `map_coordinates`, explicitly mapping output pixel coordinates to the continuous index space of the patch grid ($y_{out} \to y_{in} = (y_{out} - \text{offset}) / \text{stride}$).
3.  **Linear Interpolation**: Upgraded height map reconstruction to bilinear (`order=1`) to prevent staircase artifacts.

**Validation:**
RMSE on a synthetic slanted plane dropped from **1.94** to **0.06** pixels. This confirms that the staircase artifacts were primarily due to geometric aliasing and quantization, both of which are now resolved. The "height map" is now a continuous surface estimate rather than a blocky segmentation.
