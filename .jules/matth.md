# Matth's Journal

## 2024-05-22 - Decoupling Estimation from Application
**Learning:** In drift correction, coupling the iterative estimation loop with the frame-by-frame application loop (especially with varying directions) invites off-by-one indexing errors and makes boundary conditions fragile.
**Action:** Explicitly separate the "Trajectory Estimation" (computing global positions of all frames) from "Resampling" (applying shifts relative to a reference). This transforms a stateful, path-dependent loop into a functional mapping $T \to \text{Shift}_T$, preventing indexing bugs and enabling global trajectory analysis (e.g. smoothing) before committing to pixel operations.
