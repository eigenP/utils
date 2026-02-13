## 2025-02-12 - Sparse Matrix Optimization Pitfalls
**Learning:** Optimizing sparse matrix operations by accessing `.data` directly (e.g., `W.data**2` for $\sum w_{ij}^2$) is a powerful technique but dangerous if the matrix is not in canonical format. `scipy.sparse` matrices can contain duplicate entries for the same index, which `W.data` treats as separate values, whereas standard matrix operations sum them implicitly.
**Action:** Always call `.sum_duplicates()` on a sparse matrix before performing direct operations on its `.data` attribute if the provenance of the matrix is uncertain or if it was constructed from random indices.

## 2025-02-12 - Precision in Large Sums
**Learning:** Summing millions of `float32` values (e.g., in Moran's I moments calculation) can lead to significant relative errors (~0.02%) compared to `float64`. This error magnitude is unacceptable for statistical moments used in variance calculations ($S_1$).
**Action:** Explicitly cast to `float64` (e.g., `np.sum(arr.astype(np.float64))`) when accumulating global statistics from large arrays, even if the input data is `float32` for memory efficiency.
