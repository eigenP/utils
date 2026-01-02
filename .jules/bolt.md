## 2025-02-14 - Optimized Color Coded Projection Memory
**Learning:** Pre-allocating a large 4D array `(T, Y, X, 3)` for accumulating a max-projection result is highly memory inefficient (O(T*Y*X)).
**Action:** Replaced with iterative accumulation. Initialized a `(Y, X, 3)` result array and updated it in-place inside the loop using `np.maximum`.
**Note:** Also iterated over channels `c` inside the loop to compute `colored_frame = frame_normalized * color[c]` and update the specific channel of the result. This avoids allocating even a `(Y, X, 3)` temporary buffer, reducing it to `(Y, X)`. Peak memory dropped from ~600MB to ~24MB for a 50-frame 1k x 1k input.

## 2025-02-14 - Vectorized 2D Image Weighting
**Learning:** Python loops iterating over pixel overlap regions to apply a weight window are slow (O(overlap)) and `float16` accumulation leads to precision errors.
**Action:** Replaced with vectorized 1D profile broadcasting: `image * profile_y[:, None] * profile_x[None, :]`.
**Note:** Used `float32` for profiles to improve precision. Speedup ~10x, memory usage reduced by avoiding full 2D mask allocation.
