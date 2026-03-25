import re

with open('src/eigenp_utils/single_cell.py', 'r') as f:
    content = f.read()

# I want to add a helper function `_compute_kknn_curvatures` above `compute_kknn_neighbors`
helper_code = """
def _compute_kknn_curvatures(
    X: np.ndarray,
    indices: np.ndarray,
    max_neighbors: int
) -> np.ndarray:
    \"\"\"
    Helper function to compute local curvature (Participation Ratio)
    given a dataset and nearest neighbor indices.
    \"\"\"
    # Extract all neighborhoods at once.
    # neigh_data shape: (N, max_neighbors, m)
    neigh_data = X[indices]

    # Mean-center the data per neighborhood
    neigh_mean = neigh_data.mean(axis=1, keepdims=True)
    centered = neigh_data - neigh_mean

    # Compute covariance matrices in bulk using Einstein summation
    cov_matrices = np.einsum('nij,nil->njl', centered, centered) / max(1, (max_neighbors - 1))

    # Calculate all eigenvalues simultaneously
    eigvals = np.linalg.eigvalsh(cov_matrices)
    eigvals = np.maximum(eigvals, 0)  # Handle minor numerical noise

    # Compute Participation Ratios
    # PR = (\sum \lambda_i)^2 / \sum \lambda_i^2
    total_var = np.sum(eigvals, axis=1)
    sq_var = np.sum(eigvals ** 2, axis=1)

    curvatures = np.ones(indices.shape[0]) # Default to 1.0 (min possible PR)

    # Apply PR formula where variance exists to prevent division by zero
    mask = total_var > 0
    curvatures[mask] = (total_var[mask] ** 2) / sq_var[mask]

    return curvatures
"""

new_kknn_func = """def compute_kknn_neighbors(
    adata_query: sc.AnnData,
    adata_ref: sc.AnnData,
    use_rep: str = "X_pca",
    query_use_rep: Optional[str] = None,
    n_neighbors: Optional[int] = None,
    min_neighbors: Optional[int] = None,
    max_neighbors: Optional[int] = None,
    quantile_bins: int = 10
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    \"\"\"
    Compute adaptive k-nearest neighbors (kkNN) for a query dataset against a reference.

    The number of neighbors kept for each query cell scales with the local curvature of the
    reference manifold around the query cell. Flat regions retain more neighbors (up to max_neighbors),
    while highly curved regions retain fewer (down to min_neighbors).

    Args:
        adata_query: The query AnnData object.
        adata_ref: The reference AnnData object.
        use_rep: The representation key in .obsm to use (e.g., 'X_pca').
        n_neighbors: The base number of neighbors P (heuristic used if None).
        min_neighbors: Minimum neighbors to keep (defaults to max(3, P // 2)).
        max_neighbors: Maximum neighbors to keep (defaults to max(10, 2 * P)).
        quantile_bins: Number of bins to quantize curvature scores.

    Returns:
        A tuple of (pruned_distances, pruned_indices) where each element is a list of arrays
        (one array per query cell).
    \"\"\"
    if query_use_rep is None:
        query_use_rep = use_rep

    if query_use_rep not in adata_query.obsm:
        raise ValueError(f"Query dataset must have '{query_use_rep}' in .obsm.")
    if use_rep not in adata_ref.obsm:
        raise ValueError(f"Reference dataset must have '{use_rep}' in .obsm.")

    X_query = adata_query.obsm[query_use_rep]
    X_ref = adata_ref.obsm[use_rep]

    N_ref = X_ref.shape[0]
    N_query = X_query.shape[0]

    if n_neighbors is None:
        P = pacmap_heuristic_n_neighbors(N_ref)
    else:
        P = n_neighbors

    if min_neighbors is None:
        min_neighbors = max(3, P // 2)
    if max_neighbors is None:
        max_neighbors = max(10, 2 * P)

    # Ensure min <= max
    min_neighbors = min(min_neighbors, max_neighbors)

    print(f"kkNN: Querying up to {max_neighbors} neighbors, pruning down to {min_neighbors} (P={P})...")

    # 1. Fit Nearest Neighbors on Reference
    nn = NearestNeighbors(n_neighbors=max_neighbors, algorithm='auto', n_jobs=-1)
    nn.fit(X_ref)

    # 2. Query neighbors for all query cells
    distances, indices = nn.kneighbors(X_query)

    # 3. Compute reference curvature bounds if not cached
    bounds_key = 'kknn_curvature_bounds'
    if bounds_key in adata_ref.uns:
        lower_bound, upper_bound = adata_ref.uns[bounds_key]
    else:
        print("kkNN: Computing and caching reference curvature bounds...")
        # Get neighbors for the reference dataset itself
        _, ref_indices = nn.kneighbors(X_ref)

        # Compute curvatures for the reference dataset
        ref_curvatures = _compute_kknn_curvatures(X_ref, ref_indices, max_neighbors)

        # Calculate bounds and cache them
        lower_bound = np.percentile(ref_curvatures, 1)
        upper_bound = np.percentile(ref_curvatures, 99)
        adata_ref.uns[bounds_key] = (lower_bound, upper_bound)

    # 4. Compute local curvature for query cells
    # Note: the query cells use the reference data for neighborhood,
    # but we extracted the neighborhood data correctly because indices point to X_ref
    curvatures = _compute_kknn_curvatures(X_ref, indices, max_neighbors)

    # 5. Robust Normalization and Linear Binning using reference bounds

    # Clip extreme outliers to prevent them from stretching the K space (Winsorization)
    curvatures_clipped = np.clip(curvatures, lower_bound, upper_bound)

    # Now perform absolute scaling on the cleaned data
    ptp = upper_bound - lower_bound
    if ptp == 0:
        K = np.zeros_like(curvatures_clipped)
    else:
        K = (curvatures_clipped - lower_bound) / (ptp + 1e-9)

    # Use linear bins on the outlier-resistant K values
    # Absolute scaling ensures global structural consistency, while the percentile clipping
    # prevents single massive outliers from compressing the valid variance.
    bins = np.linspace(0.0, 1.0, quantile_bins + 1)[1:-1]
    disc_curv = np.digitize(K, bins) # 0 to quantile_bins-1

    pruned_distances = []
    pruned_indices = []

    for i in range(N_query):
        bin_idx = disc_curv[i]

        # Linear interpolation between max_neighbors and min_neighbors
        # bin_idx = 0 -> keep = max_neighbors
        # bin_idx = quantile_bins - 1 -> keep = min_neighbors
        fraction = bin_idx / max(1, (quantile_bins - 1))
        keep = int(round(max_neighbors - fraction * (max_neighbors - min_neighbors)))

        # Failsafe bounds
        keep = max(min_neighbors, min(max_neighbors, keep))

        pruned_distances.append(distances[i, :keep])
        pruned_indices.append(indices[i, :keep])

    return pruned_distances, pruned_indices"""

# Replace the old compute_kknn_neighbors with the helper and new function
import re

start = content.find('def compute_kknn_neighbors(')
end = content.find('def kknn_ingest(', start)

new_content = content[:start] + helper_code + '\n' + new_kknn_func + '\n\n\n' + content[end:]

with open('src/eigenp_utils/single_cell.py', 'w') as f:
    f.write(new_content)
