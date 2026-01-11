## 2025-02-12 - Replacing Variance with Laplacian Energy for Focus Measurement

**Learning:**
Standard deviation of intensity (contrast) is a flawed proxy for image focus. It is sensitive to mean intensity shifts and can incorrectly identify bright, blurred regions as "in-focus" over dim, sharp regions. In contrast, the Laplacian operator acts as a second-derivative high-pass filter. The energy (sum of squares) or variance of the Laplacian response is a robust measure of high-frequency content (edges/texture) and is much less sensitive to low-frequency intensity variations.

**Action:**
Replacing `np.std(patch)` with a vectorized `uniform_filter(laplace(img)**2)` approach in `best_focus_image`.
1.  **Correctness:** Prioritizes sharpness over contrast.
2.  **Robustness:** Correctly handles cases where out-of-focus planes are brighter than in-focus planes (common in fluorescence microscopy due to background accumulation or scattering).
3.  **Efficiency:** Vectorizing the metric calculation eliminates Python loops over patches, reducing complexity from $O(N \cdot K^2)$ to $O(N)$ (where $K$ is patch size).
