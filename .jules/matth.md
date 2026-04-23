## 2025-02-25 - Lineage Coupling Combinatorics
**Learning:** Verified that `calculate_lineage_coupling` can compute exact co-occurrence expectations and z-scores analytically using the hypergeometric distribution (`scipy.stats.hypergeom`) and the inclusion-exclusion principle.
- **Mathematical Equivalence:** This formulation replaces heuristic permutation testing, eliminating non-deterministic variation and mathematically mirroring the exact null distribution assumed by permutation.
- **Performance:** Changes the complexity from $O(P \cdot C)$ to $O(N_c \cdot N_t^2)$ (where $P$ is permutations, $C$ is cells, $N_c$ is clones, $N_t$ is cell types), bypassing massive matrix operations and rendering the estimation instant.
**Action:** Always seek to replace permutation-based resampling with exact combinatorial limits (Poisson Binomial / Hypergeometric / Multinomial) when evaluating simple categorical co-occurrences or subsetting.
