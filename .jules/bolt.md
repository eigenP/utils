## 2024-05-23 - Memory-Efficient Focus Stacking
**Learning:** In image processing pipelines (specifically `best_focus_image`), allocating large arrays (`np.pad`, `laplace` output) inside a loop over Z-slices creates massive memory churn.
**Action:** Use pre-allocated buffers and manual slicing logic (simulating `np.pad` 'reflect' mode) to reuse memory. This reduces peak memory usage (from O(Z*H*W) or high churn to O(H*W) constant) and improves runtime by avoiding `malloc` overhead.
**Caveat:** Manual implementation of `mode='reflect'` must carefully handle slice indices and edge cases; a fallback to `np.pad` is essential for robustness when padding exceeds dimensions.

## 2025-05-23 - Fast Laplace for Focus Stacking
**Learning:** `scipy.ndimage.laplace` is a generic N-dimensional convolution which has significant overhead (approx 4.5x slower) compared to a specialized 2D NumPy implementation for fixed 3x3 kernels. However, manual implementation must carefully handle integer overflow by enforcing float accumulation.
**Action:** Implemented `fast_laplace` using vectorized slicing and in-place operations, mimicking `mode='reflect'` boundary conditions. To prevent integer overflow on `uint8` inputs, accumulation is forced into the float output buffer using `np.copyto` and in-place ops.
