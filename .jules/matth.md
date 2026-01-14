## 2025-05-22 - Replacing Ad-Hoc Softmax with Probabilistic Superiority

**Learning:** The function `annotate_clusters_by_markers` previously used a softmax function with an arbitrary temperature parameter (`beta=2.0`) applied to robustly normalized scores (median/MAD) to estimate "confidence".
This approach had two flaws:
1.  **Arbitrary Scaling:** The `beta` parameter implies a specific belief about the signal-to-noise ratio of the *global* normalization that may not hold for individual clusters.
2.  **Global vs. Local Variance:** Normalization used global MAD, but confidence is a function of *local* separability within the cluster. A tight cluster should have higher confidence for the same mean separation than a loose one. Using global variance ignores this.

**Action:** Replaced the softmax heuristic with the **Probability of Superiority** (also known as Common Language Effect Size).
We compute $P(S_{top1} > S_{top2})$ for a random cell in the cluster by modeling the pairwise difference of scores $D = S_{top1} - S_{top2}$ as normally distributed: $D \sim \mathcal{N}(\mu_D, \sigma_D^2)$.
The confidence is then $\Phi(\mu_D / \sigma_D)$, where $\Phi$ is the standard normal CDF.
This is a parameter-free, statistically grounded metric that naturally adapts to the intra-cluster variance.

## 2025-05-23 - Scale Mismatch in Discrete Regularization

**Learning:** The focus stacking algorithm used a fixed-size median filter (`disk(3)`, diameter 7) on the reconstructed depth map index grid. However, the grid resolution depends on `patch_size` (derived from image size). For default patch sizes, the grid can be very coarse (e.g., 10x10). A 7x7 filter on such a grid is a global operator, not a local regularizer, effectively flattening the depth map and erasing all features smaller than ~50% of the image width.

**Action:** Regularization kernels must be scaled relative to the signal resolution (grid size), or chosen conservatively (`3x3`) to remove outliers without enforcing global smoothness. When discretizing a continuous field (depth) onto a coarse grid, feature preservation requires minimal kernel sizes.
