## 2025-05-22 - Replacing Ad-Hoc Softmax with Probabilistic Superiority

**Learning:** The function `annotate_clusters_by_markers` previously used a softmax function with an arbitrary temperature parameter (`beta=2.0`) applied to robustly normalized scores (median/MAD) to estimate "confidence".
This approach had two flaws:
1.  **Arbitrary Scaling:** The `beta` parameter implies a specific belief about the signal-to-noise ratio of the *global* normalization that may not hold for individual clusters.
2.  **Global vs. Local Variance:** Normalization used global MAD, but confidence is a function of *local* separability within the cluster. A tight cluster should have higher confidence for the same mean separation than a loose one. Using global variance ignores this.

**Action:** Replaced the softmax heuristic with the **Probability of Superiority** (also known as Common Language Effect Size).
We compute $P(S_{top1} > S_{top2})$ for a random cell in the cluster by modeling the pairwise difference of scores $D = S_{top1} - S_{top2}$ as normally distributed: $D \sim \mathcal{N}(\mu_D, \sigma_D^2)$.
The confidence is then $\Phi(\mu_D / \sigma_D)$, where $\Phi$ is the standard normal CDF.
This is a parameter-free, statistically grounded metric that naturally adapts to the intra-cluster variance.
