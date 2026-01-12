## 2025-02-12 - Replacing Variance with Laplacian Energy for Focus Measurement

**Learning:**
Standard deviation of intensity (contrast) is a flawed proxy for image focus. It is sensitive to mean intensity shifts and can incorrectly identify bright, blurred regions as "in-focus" over dim, sharp regions. In contrast, the Laplacian operator acts as a second-derivative high-pass filter. The energy (sum of squares) or variance of the Laplacian response is a robust measure of high-frequency content (edges/texture) and is much less sensitive to low-frequency intensity variations.

**Action:**
Replacing `np.std(patch)` with a vectorized `uniform_filter(laplace(img)**2)` approach in `best_focus_image`.
1.  **Correctness:** Prioritizes sharpness over contrast.
2.  **Robustness:** Correctly handles cases where out-of-focus planes are brighter than in-focus planes (common in fluorescence microscopy due to background accumulation or scattering).
3.  **Efficiency:** Vectorizing the metric calculation eliminates Python loops over patches, reducing complexity from $O(N \cdot K^2)$ to $O(N)$ (where $K$ is patch size).

## 2025-02-12 - Memory Optimization in Focus Stacking

**Learning:**
Casting entire 3D image stacks to `float64` for processing can cause severe memory inflation (e.g., 8x expansion for `uint8` data), leading to OOM errors on large datasets. However, `float32` precision (7 decimal digits) is sufficient for focus metrics like Laplacian Energy. Furthermore, `scipy.ndimage` functions often support an `output` parameter that allows computing results directly into a destination buffer (optionally with type promotion), avoiding the need for intermediate full-stack copies or explicit `.astype()` calls.

**Action:**
In `best_focus_image`:
1.  Process the focus metric slice-by-slice instead of allocating the whole stack.
2.  Use `scipy.ndimage.laplace(slice, output=np.float32)` to implicitly cast and compute the gradient in one step, avoiding a separate float copy of the input slice.
3.  Use `float32` for all accumulation buffers (`score_matrix`, `final_img`, `counts`).
4.  This reduces memory overhead for the processing buffers by 50% compared to a naive `float64` approach and avoids O(Z*Y*X) temporary allocations.
