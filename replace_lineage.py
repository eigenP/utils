import re

with open('src/eigenp_utils/single_cell.py', 'r') as f:
    content = f.read()

# find the calculate_lineage_coupling function
old_func = """def calculate_lineage_coupling(
    adata, label_key="cell_type", clone_key="CloneID", n_permutations=1000
):
    \"\"\"
    Calculates lineage coupling statistics via permutation testing.
    Returns matrices for Observed counts, Z-scores, and P-values.
    \"\"\"

    # 1. Prepare Data
    # Create a binary matrix: Clones (rows) x Cell Types (columns)
    # Value is 1 if the clone exists in that cell type, 0 otherwise.
    df = adata.obs[[label_key, clone_key]].dropna()

    # Efficient way to create the binary matrix
    # Group by clone and get unique labels, then get dummies
    # Pivot: Index=Clone_ID, Columns=Cell_Type, Values=1 (if present)
    binary_matrix = pd.crosstab(df[clone_key], df[label_key]).clip(upper=1)

    # 2. Calculate Observed Intersections
    # Matrix Multiplication: (Types x Clones) @ (Clones x Types) = (Types x Types)
    # The result [i, j] is the number of clones shared between Type i and Type j
    observed_counts = binary_matrix.T @ binary_matrix

    # 3. Permutation Test
    null_matrices = []

    # We shuffle the labels array to break the link between clone and cell type
    # Convert to standard numpy array to avoid categorical shuffling warnings
    # Use astype(str) or astype(object) before extracting array to avoid read-only or categorical warnings
    labels_array = np.array(df[label_key].astype(str))

    print(f"Running {n_permutations} permutations...")
    for _ in range(n_permutations):
        # Shuffle labels in place
        np.random.shuffle(labels_array)

        # Reconstruct the binary matrix with shuffled labels
        # Note: We rely on the original index (clone_key) structure
        shuffled_df = pd.DataFrame(
            {clone_key: df[clone_key].values, label_key: labels_array}
        )

        shuffled_binary = pd.crosstab(
            shuffled_df[clone_key], shuffled_df[label_key]
        ).clip(upper=1)

        # Ensure columns match observed (in case shuffling drops a rare type entirely)
        shuffled_binary = shuffled_binary.reindex(
            columns=binary_matrix.columns, fill_value=0
        )

        # Calculate null intersection
        null_matrices.append((shuffled_binary.T @ shuffled_binary).values)

    null_matrices = np.array(null_matrices)  # Shape: (n_perms, n_types, n_types)

    # 4. Calculate Statistics
    null_mean = null_matrices.mean(axis=0)
    null_std = null_matrices.std(axis=0)

    # Avoid division by zero
    null_std[null_std == 0] = 1.0

    z_scores = (observed_counts.values - null_mean) / null_std
    z_scores = pd.DataFrame(
        z_scores, index=observed_counts.index, columns=observed_counts.columns
    )

    # Calculate empirical P-values
    # (Count how many nulls were >= observed) / n_permutations
    # Note: This is a one-sided test for enrichment.
    # For two-sided, you'd check both tails. The image implies enrichment focus.
    p_values = (null_matrices >= observed_counts.values).sum(axis=0) / n_permutations
    p_values = pd.DataFrame(
        p_values, index=observed_counts.index, columns=observed_counts.columns
    )

    return observed_counts, z_scores, p_values
"""

new_func = """def calculate_lineage_coupling(
    adata, label_key="cell_type", clone_key="CloneID", n_permutations=None
):
    \"\"\"
    Calculates lineage coupling statistics analytically using the hypergeometric distribution.
    Returns matrices for Observed counts, Z-scores, and P-values.
    \"\"\"
    if n_permutations is not None:
        warnings.warn(
            "`n_permutations` is deprecated. Lineage coupling is now calculated exactly "
            "and analytically using the hypergeometric distribution.",
            DeprecationWarning,
            stacklevel=2,
        )

    # 1. Prepare Data
    # Create a binary matrix: Clones (rows) x Cell Types (columns)
    # Value is 1 if the clone exists in that cell type, 0 otherwise.
    df = adata.obs[[label_key, clone_key]].dropna()

    # Efficient way to create the binary matrix
    binary_matrix = pd.crosstab(df[clone_key], df[label_key]).clip(upper=1)

    # 2. Calculate Observed Intersections
    # The result [i, j] is the number of clones shared between Type i and Type j
    observed_counts = binary_matrix.T @ binary_matrix

    # 3. Analytical Null Distribution
    from scipy.stats import hypergeom, norm
    import numpy as np

    N_total = len(df)
    type_counts = df[label_key].value_counts()

    # We only care about the distribution of clone sizes (number of cells per clone)
    clone_sizes = df[clone_key].value_counts().values

    types = binary_matrix.columns
    n_types = len(types)

    null_mean = np.zeros((n_types, n_types))
    null_var = np.zeros((n_types, n_types))

    for i, t1 in enumerate(types):
        N1 = type_counts.get(t1, 0)
        for j, t2 in enumerate(types):
            N2 = type_counts.get(t2, 0)

            p_expected = 0.0
            var_expected = 0.0

            if N1 == 0 or (t1 != t2 and N2 == 0):
                pass
            elif t1 == t2:
                p_no1 = hypergeom.pmf(0, N_total, N1, clone_sizes)
                p = 1.0 - p_no1
                p_expected = np.sum(p)
                var_expected = np.sum(p * (1.0 - p))
            else:
                p_no1 = hypergeom.pmf(0, N_total, N1, clone_sizes)
                p_no2 = hypergeom.pmf(0, N_total, N2, clone_sizes)
                n_both = min(N1 + N2, N_total)
                p_no_both = hypergeom.pmf(0, N_total, n_both, clone_sizes)

                p = 1.0 - p_no1 - p_no2 + p_no_both
                p_expected = np.sum(p)
                var_expected = np.sum(p * (1.0 - p))

            null_mean[i, j] = p_expected
            null_var[i, j] = var_expected

    # 4. Calculate Statistics
    null_std = np.sqrt(null_var)
    # Avoid division by zero
    null_std[null_std == 0] = 1.0

    z_scores_val = (observed_counts.values - null_mean) / null_std
    z_scores = pd.DataFrame(
        z_scores_val, index=observed_counts.index, columns=observed_counts.columns
    )

    # Calculate analytical P-values (one-sided for enrichment)
    # P(Z >= z_observed)
    p_values_val = norm.sf(z_scores_val)
    p_values = pd.DataFrame(
        p_values_val, index=observed_counts.index, columns=observed_counts.columns
    )

    return observed_counts, z_scores, p_values
"""

content = content.replace(old_func, new_func)

with open('src/eigenp_utils/single_cell.py', 'w') as f:
    f.write(content)
