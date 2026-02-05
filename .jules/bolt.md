## 2024-05-23 - Memory-Efficient Focus Stacking
**Learning:** In image processing pipelines (specifically `best_focus_image`), allocating large arrays (`np.pad`, `laplace` output) inside a loop over Z-slices creates massive memory churn.
**Action:** Use pre-allocated buffers and manual slicing logic (simulating `np.pad` 'reflect' mode) to reuse memory. This reduces peak memory usage (from O(Z*H*W) or high churn to O(H*W) constant) and improves runtime by avoiding `malloc` overhead.
**Caveat:** Manual implementation of `mode='reflect'` must carefully handle slice indices and edge cases; a fallback to `np.pad` is essential for robustness when padding exceeds dimensions.

## 2025-02-12 - Scipy vs Numpy Laplace Overhead
**Learning:** `scipy.ndimage.laplace` (and generic filters) incurs significant overhead compared to manual NumPy slicing for small kernels (e.g., 3x3), due to generic dispatch and C-level wrapping. For a 3x3 Laplacian, a manual NumPy implementation is ~5x faster (0.018s vs 0.105s for 2k^2 image).
**Learning:** `scipy.ndimage`'s `mode='reflect'` (d c b a | a b c d | d c b a) corresponds to `numpy.pad`'s `mode='symmetric'`, NOT `numpy.pad`'s `mode='reflect'` (which reflects about the edge center). Mismatched padding modes can cause subtle boundary errors.
**Action:** For performance-critical tight loops involving simple stencils (like Laplacian), prefer manual NumPy slicing with correct padding over `scipy.ndimage` functions.
