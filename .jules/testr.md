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

## 2025-02-23 - Focus Stacking Verification
**Learning:** Verified `best_focus_image` using a "Checkerboard Depth Field".
- **Invariant:** The focus map accurately recovers a discrete step function (checkerboard) of indices from a noisy 3D stack, proving the Laplacian energy metric correctly identifies high-frequency texture.
- **Signal Preservation:** The algorithm preserves the original texture variance (Output Std $\approx$ Input Std) and strongly rejects blurred versions (MSE Ratio > 40), confirming that the blending process does not wash out signals.
**Action:** Future tests for image fusion or reconstruction algorithms should utilize synthetic "depth fields" with known ground-truth index maps. This decouples the verification of the decision logic (selection) from the reconstruction quality (blending).
## 2025-02-23 - Moran's I Statistical Calibration
**Learning:** Validated that `morans_i_all_fast` produces correctly calibrated Z-scores under the null hypothesis, specifically handling non-Gaussian data.
- **Gaussian Noise:** Z-scores follow $N(0, 1)$ ($Mean \approx 0.04$, $Std \approx 1.01$). P-values are Uniform (Prop < 0.05 is 0.049).
- **Kurtotic Noise:** Even with sparse, spiky data (high kurtosis), Z-scores maintain unit variance ($Std \approx 0.997$).
**Action:** This confirms that the complex analytical variance formula (Cliff & Ord) including the kurtosis correction term ($b_2$) is implemented correctly. It proves the statistic is robust to data distribution, allowing its use on raw or log-transformed expression data without false positives from non-normality.

## 2025-02-23 - Surface Extraction Invariance
**Learning:** Verified `extract_surface` accuracy and stability using a synthetic sinusoidal landscape.
- **Accuracy:** Reconstructed surface matches analytical ground truth with MAE ~0.55 pixels (sub-pixel accuracy despite downsampling).
- **Invariance:** Z-axis translation of the input volume results in an equivalent translation of the output surface (Mean shift error < 0.03 px, Std < 0.2 px).
**Action:** Confirms that the `downscale` -> `smooth` -> `threshold` -> `zoom` pipeline is robust and preserves geometry. Future 3D segmentation tests should rely on translation invariance as a primary correctness check.
## 2025-02-23 - Surface Extraction Accuracy & Invariance
**Learning:** Verified `extract_surface` ("surface peeler") accuracy using a "Sine Wave Reconstruction" test.
- **Accuracy:** The algorithm (with bicubic upscaling) recovers the analytical surface with low MAE (< 2 pixels), confirming that the geometry is preserved despite downsampling.
- **Invariance:** Translation invariance holds (Mean Shift $\approx$ Actual Shift), *provided* that `gaussian_sigma` is not excessively large relative to the volume size. Large smoothing combined with global Otsu thresholding can induce bias when the foreground/background ratio changes significantly.
**Action:** For small volumes or precise surface extraction, `gaussian_sigma` should be kept low (e.g., 1.0) to minimize threshold-induced shift errors. Tests for geometric algorithms must verify invariance to rigid transformations (translation, rotation).

## 2025-02-23 - Brightness Correction Invariants
**Learning:** Verified `adjust_brightness_per_slice` as a Global Trend Corrector, not a local normalizer.
- **Global Stability:** It perfectly flattens systematic exponential decay (CV < 1.5%) when the signal follows the model.
- **Local Preservation:** It preserves local anomalies (outliers) relative to the corrected trend, rather than normalizing them away. This confirms the intent is to correct for physics (e.g., attenuation), not biology (e.g., sparse cells).
- **Statistical Idempotence:** The correction is stable/idempotent, but finite sampling noise in the 99th percentile prevents perfect identity (Delta < 0.1%).
**Action:** When testing statistical estimators on finite samples (like P99), exact equality assertions must be replaced with tolerance checks that account for sampling variance/overfitting.
