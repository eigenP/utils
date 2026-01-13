# Testr Journal


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
