# Testr Journal

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
