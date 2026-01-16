# Testr Journal

## 2025-02-18 - Drift Correction Integrator Windup
**Learning:** The drift correction algorithm in `maxproj_registration.py` was casting cumulative drift to integer at every time step (`cum_dx = int(cum_dx + dx)`). This introduced a severe "integrator windup" bug (or rather, failure to integrate) where fractional drifts smaller than 1.0 would be discarded repeatedly if they didn't cross an integer boundary in a single step. For slow drifts (e.g., 0.5 px/frame), this resulted in zero correction over time.
**Action:** The accumulation logic was fixed to maintain float precision (`cum_dx += dx`) and only round to integer when applying the final shift. A high-level invariant test (`tests/test_drift_integrity.py`) was added to verify that fractional drift accumulates correctly over time (Linearity/conservation of total drift).

## 2024-05-22 - Archetype Recovery and Invariants
**Learning:** Verified that find_expression_archetypes correctly identifies gene modules using PC1 and aligns them with the cluster mean. This test ensures that the algorithm is robust to scale and shift transformations (affine invariance) and correctly separates orthogonal signals.
**Action:** Future tests for clustering algorithms should always include synthetic ground-truth scenarios with known invariants (like scale/shift invariance for Pearson correlation-based methods).


## 2024-05-22 - Initial Setup
**Learning:** Testr agent initialized.
**Action:** Starting verification of `morans_i_all_fast`.

## 2024-05-22 - Moran's I Verification
**Learning:** Verified `morans_i_all_fast` against analytical properties.
- **Invariant:** The Constant Vector on a row-standardized graph yields Moran's I = 1.0 (matching eigenvalue 1).
- **Invariant:** The Checkerboard pattern on a bipartite grid yields Moran's I = -1.0 (matching eigenvalue -1).
- **Statistics:** Random noise converges to E[I] = -1/(n-1) as expected under the null hypothesis.
- **Optimization:** The memory-optimized algebraic expansion is numerically equivalent to the naive definition.
**Action:** These invariants confirm both the statistical correctness and the implementation stability of the fast Moran's I calculator. Future spatial statistic tests should leverage known spectral graph theory properties.

## 2025-02-23 - Annotation Confidence Statistics
**Learning:** Verified that `annotate_clusters_by_markers` correctly implements the "Probability of Superiority" metric ($P(X > Y)$) for cell type assignment.
- **Invariant:** Perfect separation yields $P=1.0$.
- **Invariant:** Indistinguishable distributions yield $P \approx 0.5$.
- **Calibration:** Controlled Gaussian overlap ($N(1,1)$ vs $N(0,1)$) yields $P \approx 0.76$, matching the analytical solution $\Phi(\frac{\Delta \mu}{\sqrt{\sigma_1^2 + \sigma_2^2}})$.
**Action:** This confirms that the "uncertainty" metric is a statistically valid probability derived from the distribution of scores, not an arbitrary heuristic. Future probabilistic classifiers should always be tested against analytical distributions to verify calibration.

## 2025-02-23 - Moran's I Statistical Calibration
**Learning:** Validated that `morans_i_all_fast` produces correctly calibrated Z-scores under the null hypothesis, specifically handling non-Gaussian data.
- **Gaussian Noise:** Z-scores follow $N(0, 1)$ ($Mean \approx 0.04$, $Std \approx 1.01$). P-values are Uniform (Prop < 0.05 is 0.049).
- **Kurtotic Noise:** Even with sparse, spiky data (high kurtosis), Z-scores maintain unit variance ($Std \approx 0.997$).
**Action:** This confirms that the complex analytical variance formula (Cliff & Ord) including the kurtosis correction term ($b_2$) is implemented correctly. It proves the statistic is robust to data distribution, allowing its use on raw or log-transformed expression data without false positives from non-normality.
