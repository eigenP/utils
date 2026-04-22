## 2024-05-23 - Memory-Efficient Focus Stacking
**Learning:** In image processing pipelines (specifically `best_focus_image`), allocating large arrays (`np.pad`, `laplace` output) inside a loop over Z-slices creates massive memory churn.
**Action:** Use pre-allocated buffers and manual slicing logic (simulating `np.pad` 'reflect' mode) to reuse memory. This reduces peak memory usage (from O(Z*H*W) or high churn to O(H*W) constant) and improves runtime by avoiding `malloc` overhead.
**Caveat:** Manual implementation of `mode='reflect'` must carefully handle slice indices and edge cases; a fallback to `np.pad` is essential for robustness when padding exceeds dimensions.

## 2024-03-18 - [Pandas .apply() Overhead in Matrix Math]
**Learning:** Using `Pandas.DataFrame.apply()` with a lambda function for element/column-wise array mathematics (e.g., computing robust median and MAD normalization on scores) introduces massive Python-level overhead compared to pure NumPy, even when the underlying functions (`np.nanmedian`) are compiled. Benchmarks showed replacing `.apply()` with `S.to_numpy()` and `axis=0` NumPy operations sped up a 50k cell x 20 celltype normalization by over 20x.
**Action:** When performing matrix-wide standardizations, normalizations, or centering in Pandas DataFrames, always extract the NumPy array first (`.to_numpy()`), perform the vectorized math using `axis` arguments, and reconstruct the DataFrame. Avoid `.apply()` for pure numerical array operations.
