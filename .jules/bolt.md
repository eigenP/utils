## 2024-05-23 - Memory-Efficient Focus Stacking
**Learning:** In image processing pipelines (specifically `best_focus_image`), allocating large arrays (`np.pad`, `laplace` output) inside a loop over Z-slices creates massive memory churn.
**Action:** Use pre-allocated buffers and manual slicing logic (simulating `np.pad` 'reflect' mode) to reuse memory. This reduces peak memory usage (from O(Z*H*W) or high churn to O(H*W) constant) and improves runtime by avoiding `malloc` overhead.
**Caveat:** Manual implementation of `mode='reflect'` must carefully handle slice indices and edge cases; a fallback to `np.pad` is essential for robustness when padding exceeds dimensions.

## 2024-05-24 - SciPy Laplacian Overhead
**Learning:** `scipy.ndimage.laplace` carries significant overhead (generic filter machinery) compared to a specialized 3x3 NumPy implementation using `np.pad` and in-place accumulation. A custom `fast_laplace` yielded a ~3x speedup for the filter step and ~30% total runtime improvement.
**Action:** For fixed, small kernels (like 3x3 Laplacian) in tight loops, prefer explicit NumPy vectorization over generic SciPy filters.
**Detail:** SciPy's `mode='reflect'` corresponds to NumPy's `mode='symmetric'`.
