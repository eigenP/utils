
## 2025-02-23 - Stationary Windowing Bias in Drift Correction

**Learning:** Cross-correlation with a stationary window (e.g., Tukey window fixed in the image frame) introduces a systematic bias when estimating shifts of moving objects. The window attenuates the overlap region asymmetrically depending on the shift direction, pulling the correlation peak towards zero (underestimation). This bias accumulates linearly in pairwise registration schemes (e.g., ~0.2 px/cycle for sinusoidal motion).

**Action:** Implemented **Matched Windowing (Iterative Re-windowing)**. Instead of correlating windowed signals directly, we first estimate a coarse shift, then shift the *window* itself to align with the expected position of the object in the moving frame. This ensures that the window is centered on the same features in both the reference and moving signals, eliminating the asymmetric attenuation bias. This reduced accumulated drift error by ~22% in synthetic tests and is theoretically robust without requiring complex global optimization.
