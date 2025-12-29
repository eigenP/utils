## 2025-02-14 - Optimized Subpixel Drift Correction and Weighting
**Learning:** `np.meshgrid` combined with `map_coordinates` for simple translations is extremely memory inefficient ($O(3 \cdot N_{pixels})$ additional allocation). For subpixel shifts, `scipy.ndimage.shift` is functionally equivalent and vastly more memory-efficient.
**Action:** Replaced the meshgrid approach with `scipy.ndimage.shift`. Also vectorized a slow Python loop in `_2D_weighted_image` using broadcasting, speeding up weighting by ~5x.
