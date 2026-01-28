## 2024-05-23 - Memory-Efficient Focus Stacking
**Learning:** In image processing pipelines (specifically `best_focus_image`), allocating large arrays (`np.pad`, `laplace` output) inside a loop over Z-slices creates massive memory churn.
**Action:** Use pre-allocated buffers and manual slicing logic (simulating `np.pad` 'reflect' mode) to reuse memory. This reduces peak memory usage (from O(Z*H*W) or high churn to O(H*W) constant) and improves runtime by avoiding `malloc` overhead.
**Caveat:** Manual implementation of `mode='reflect'` must carefully handle slice indices and edge cases; a fallback to `np.pad` is essential for robustness when padding exceeds dimensions.

## 2024-05-23 - Scipy Shift Pre-allocation
**Learning:** Benchmarking `scipy.ndimage.shift` (order=3) reveals it is compute-bound, meaning pre-allocation strategies reduce memory churn/GC pressure significantly but provide negligible runtime speedup.
**Action:** Use `output=` parameter for memory stability in long loops, but don't expect runtime gains.
