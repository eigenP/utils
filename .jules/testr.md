## 2024-05-22 - [Archetype Recovery and Invariants]
**Learning:** Verified that find_expression_archetypes correctly identifies gene modules using PC1 and aligns them with the cluster mean. This test ensures that the algorithm is robust to scale and shift transformations (affine invariance) and correctly separates orthogonal signals.
**Action:** Future tests for clustering algorithms should always include synthetic ground-truth scenarios with known invariants (like scale/shift invariance for Pearson correlation-based methods).

## 2024-05-23 - [Focus vs Contrast Invariance]
**Learning:** A high-level test revealed that `best_focus_image` was selecting images with high global variance (like gradients) instead of high-frequency sharpness (Laplacian energy). This clarified that focus metrics must be invariant to low-frequency trends.
**Action:** Image processing algorithms claiming "quality" selection must be tested against adversarial inputs where quality metric and trivial statistics (like variance or mean) diverge.
