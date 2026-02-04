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

## 2025-02-24 - Drift Correction Windowing Bias
**Learning:** The windowing function (`_2D_weighted_image`) used in drift estimation can significantly bias results on small images if the object is large relative to the field of view. A Gaussian moving by 1.0 px was estimated as moving only 0.2 px on a 32x32 image because the window attenuated the object's tails, effectively shifting its centroid back towards the center.
**Action:** Tests for drift correction must use images large enough (e.g., 64x64 or 128x128) so that the object resides within the flat region of the window, or explicitly handle windowing effects.
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

## 2025-02-24 - Multiscale Coarsening Lineage Inconsistency
**Learning:** Verified that `multiscale_coarsening` correctly handles "Simpson's Paradox" scenarios where the direct lineage (Fine->Coarse) disagrees with the indirect lineage (Fine->Mid->Coarse) due to aggregation "majority vote" flips.
**Action:** Tests for hierarchical aggregation must explicitly mock or construct edge cases where the majority shifts across levels to ensure robustness against voting anomalies.

## 2025-02-24 - CLAHE Limits & Identity
**Learning:** Verified the mathematical properties of `_my_clahe_` (Adaptive Histogram Equalization):
1.  **Monotonicity:** Contrast (entropy) increases monotonically with `clip_limit`.
2.  **Identity Convergence:** As `clip_limit` $\to$ 0 (e.g. $10^{-5}$), the operation converges to a linear identity transform (correlation > 0.99), preserving relative signal shape rather than enforcing a uniform histogram.
3.  **Sentinel Discontinuity:** `clip_limit=0` is a sentinel for "Unlimited AHE" (Max Contrast), causing a sharp discontinuity from `clip_limit=0.001` (Low Contrast). This behavior is correct but counter-intuitive.
4.  **Saturation:** For smooth/sparse images, effective contrast limits saturate quickly (e.g. at 0.05) once the clip limit exceeds the maximum bin count of the local histogram.
**Action:** Tests for contrast enhancement must account for signal content (saturation) and the specific sentinel value (0) to avoid false failures. Using `clip_limit=0` should be preferred for "Maximum Effect" rather than `clip_limit=1.0` to avoid ambiguity.

## 2025-02-24 - Focus Stacking Boundary Artifacts
**Learning:** The `best_focus_image` algorithm blindly applied tapering (windowing) to all patches, including those at the image boundaries. This violated the Partition of Unity property, causing significant signal loss (fading to black) at the edges of the reconstructed image. Identity reproduction failed: a uniform input stack resulted in an image with dark borders.
**Action:** The algorithm was updated to use context-aware windowing (`_get_1d_weight_variants`), selecting flat-start/flat-end profiles for boundary patches. A new invariant test (`tests/test_focus_properties.py`) was added to enforce Identity Reproduction (Input=1s -> Output=1s) and Partition of Unity, ensuring future regressions are caught.

## 2025-02-24 - Subpixel Drift Precision
**Learning:** Verified that `apply_drift_correction_2D` with `method='subpixel'` achieves sub-pixel stability (RMSE < 0.1 px) significantly better than integer correction (RMSE ~ 0.4 px).
- **Subpixel Magic:** Bicubic interpolation effectively eliminates quantization noise for fractional drifts, providing near-perfect stability for smooth signals.
- **Windowing Trap:** Reinforced that small image sizes (e.g., 64x64) combined with drift can push objects into the tapered window region, causing massive estimation bias. Tests must ensure the object stays in the "flat" central region of the window (requiring size >= 128 for moderate drifts).
**Action:** Always prefer `method='subpixel'` for precision tasks. When testing registration, verify that the object's trajectory does not intersect the windowing taper to avoid confounding estimation errors.
## 2025-02-24 - Subpixel Drift Correction Precision
**Learning:** Verified the precision of `apply_drift_correction_2D(method='subpixel')` using synthetic moving Gaussians.
- **Precision:** Subpixel (bicubic) correction reduces centroid jitter by nearly 200x (0.0004px vs 0.08px) compared to integer correction, confirming it is not just "interpolated" but "stabilized" to near-theoretical limits.
- **Convention:** The drift table stores the **Correction Vector** (negative of object motion), which is the shift required to stabilize the image, not the motion vector itself.
- **Cycle Consistency:** The algorithm satisfies cycle consistency: correcting a known shift restores the original image with negligible residual error ($< 10^{-5}$).
**Action:** High-precision video stabilization requires testing against subpixel metrics; integer-based metrics (like "within 0.5px") are insufficient to validate bicubic/subpixel logic. The distinction between "Motion" and "Correction" vectors must be explicit in test expectations.

## 2025-02-24 - Bidirectional Drift Sign Inversion
**Learning:** The `reverse_time='both'` mode in `apply_drift_correction_2D` was incorrectly calculating the average drift as `(dx_forward - dx_backward) / 2`. Since `dx_backward` and `dx_forward` have opposite signs for the same motion (e.g., -0.5 and +0.5 for +0.5 motion), this formula resulted in a correction vector with the **same sign** as the motion (e.g., +0.5), exacerbating drift instead of correcting it (Positive Feedback Loop).
**Action:** The formula was corrected to `(dx_backward - dx_forward) / 2`, ensuring the correction vector opposes the motion. A new test `tests/test_bidirectional_drift_sign.py` was created to verify the sign of correction for a known drift direction, catching any future regression into positive feedback.

## 2025-02-24 - Dimensionality Parser Resizing Logic
**Learning:** The `dimensionality_parser` decorator relied on a strict size-matching heuristic to determine if a dimension was preserved or reduced. This caused it to fail on **Resizing** operations (e.g., downscaling) where dimensions are preserved but sizes change.
**Action:** Implemented a **Rank Preservation Check** (`rank_in == rank_out`) to override the heuristic. If the number of dimensions is preserved, the parser now assumes all dimensions are kept and correctly maps the new output sizes. A new test `tests/test_dimensionality_parser_resizing.py` validates this invariant.
