## 2025-02-14 - Optimized Color Coded Projection Memory
**Learning:** Pre-allocating a large 4D array `(T, Y, X, 3)` for accumulating a max-projection result is highly memory inefficient (O(T*Y*X)).
**Action:** Replaced with iterative accumulation. Initialized a `(Y, X, 3)` result array and updated it in-place inside the loop using `np.maximum`.
**Note:** Also iterated over channels `c` inside the loop to compute `colored_frame = frame_normalized * color[c]` and update the specific channel of the result. This avoids allocating even a `(Y, X, 3)` temporary buffer, reducing it to `(Y, X)`. Peak memory dropped from ~600MB to ~24MB for a 50-frame 1k x 1k input.

## 2025-02-14 - Vectorized 2D Weighting for Registration
**Learning:** Applying a weighting mask iteratively (slicing and multiplying) in a Python loop for `overlap` iterations is slow (O(overlap)).
**Action:** Replaced the loop with vectorized construction of 1D weight profiles and used broadcasting (`y[:, None] * x[None, :]`) to create the 2D mask.
**Impact:** ~1.5x speedup for typical parameters (1024x1024 image, ~340 overlap) and safer handling of edge cases (overlap=1).
