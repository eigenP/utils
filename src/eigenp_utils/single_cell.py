# /// script
# dependencies = [
#     "scanpy",
#     "esda",
#     "libpysal",
#     "triku",
#     "numpy",
#     "pandas",
#     "scipy",
#     "matplotlib",
# ]
# ///

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Literal, Any
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

# Try importing third-party libraries that might be installed via the inline dependencies
try:
    from libpysal.weights import WSP
    from esda.moran import Moran
except ImportError:
    pass  # Allow import to succeed even if dependencies aren't immediately available in dev env

try:
    import triku as tk
except ImportError:
    tk = None

# ------------------------- Constants -------------------------

CELL_CYCLE_GENES = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2", "MCM6",
    "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2", "RPA2", "NASP",
    "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7", "POLD3", "MSH2", "ATAD2",
    "RAD51", "RRM2", "CDC45", "CDC6", "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2",
    "USP1", "CLSPN", "POLA1", "CHAF1B", "BRIP1", "E2F8", "HMGB2", "CDK1", "NUSAP1",
    "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67",
    "TMPO", "CENPF", "TACC3", "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB",
    "BUB1", "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1",
    "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2",
    "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA"
]

# ------------------------- General Utilities -------------------------

def check_gene_adata(adata, gene):
    """
    Checks if a gene or a collection of genes exist in the adata object.

    Parameters:
    - adata: The AnnData object containing gene information.
    - gene: A string, list, or dictionary of genes to check.

    Returns:
    - A sanitized version of the input 'gene', containing only genes present in adata.
    """

    if isinstance(gene, str):
        if gene in adata.var_names:
            return gene
        else:
            print(f"Gene '{gene}' not found in adata.")
            return None  # Or you could raise an exception here

    elif isinstance(gene, list):
        sanitized_genes = [g for g in gene if g in adata.var_names]
        not_found_genes = [g for g in gene if g not in adata.var_names]
        for g in not_found_genes:
            print(f"Gene '{g}' not found in adata.")
        return sanitized_genes

    elif isinstance(gene, dict):
        sanitized_dict = {}
        for key, genes in gene.items():
            if isinstance(genes, str):
                genes = [genes]  # Convert single gene string to a list
            sanitized_genes = [g for g in genes if g in adata.var_names]
            not_found_genes = [g for g in genes if g not in adata.var_names]
            for g in not_found_genes:
                print(f"Gene '{g}' not found in adata (key: '{key}').")
            sanitized_dict[key] = sanitized_genes
        return sanitized_dict

    else:
        raise TypeError("Input 'gene' must be a string, list, or dictionary.")


def ensure_neighbors(
    adata,
    n_neighbors: int = 15,
    use_rep: Optional[str] = "X_pca",
    key: str = "connectivities",
    n_pcs: Optional[int] = None,
) -> None:
    """
    Make sure adata.obsp[key] exists; builds neighbors in-place if missing.
    Supports both standard Scanpy usage (via use_rep) and Triku usage (via n_pcs).
    """
    if key not in adata.obsp:
        if use_rep is None and n_pcs is not None:
            # Triku-style: use specific number of PCs without a named representation
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        else:
            # Standard style: use provided representation (default "X_pca")
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)


def _select_matrix_and_names(
    adata, source: Literal["X", "raw"] | str = "X"
) -> Tuple[sp.spmatrix | np.ndarray, np.ndarray]:
    """Return (matrix, var_names) according to `source` ('X', 'raw', or layer name)."""
    if source == "raw":
        if adata.raw is None:
            raise ValueError("source='raw' requested but adata.raw is None.")
        return adata.raw.X, np.array(adata.raw.var_names)
    if source == "X":
        return adata.X, np.array(adata.var_names)
    # otherwise assume a layer name
    if source not in adata.layers:
        raise KeyError(f"Layer '{source}' not found.")
    return adata.layers[source], np.array(adata.var_names)


def _match_gene_indices(var_names: np.ndarray, gene: str, case_insensitive: bool = True) -> np.ndarray:
    """Return all column indices matching `gene`."""
    arr = np.asarray(var_names).astype(str)
    if case_insensitive:
        try:
            up = pd.Index(arr).str.upper().to_numpy()
        except Exception:
            up = np.char.upper(arr)
        return np.where(up == gene.upper())[0]
    return np.where(arr == gene)[0]


def _extract_gene_vector(
    adata,
    gene: str,
    *,
    source: Literal["X", "raw"] | str = "X",
    duplicate_policy: Literal["mean", "sum", "first", "last"] = "mean",
) -> np.ndarray:
    """
    Dense (n_cells,) float vector for `gene` from the chosen source.
    Handles duplicates deterministically via `duplicate_policy`.
    """
    M, names = _select_matrix_and_names(adata, source)
    idx = _match_gene_indices(names, gene, case_insensitive=True)
    if idx.size == 0:
        raise KeyError(f"Gene '{gene}' not found in {source}.")
    if sp.issparse(M):
        if idx.size == 1 or duplicate_policy in ("first", "last"):
            j = idx[0] if duplicate_policy != "last" else idx[-1]
            x = M[:, j].toarray().ravel()
        elif duplicate_policy == "sum":
            x = M[:, idx].sum(axis=1).A1
        else:  # mean
            k = idx.size
            x = (M[:, idx] @ np.full((k, 1), 1.0 / k, dtype=M.dtype)).A1
    else:
        M = np.asarray(M)
        if idx.size == 1 or duplicate_policy in ("first", "last"):
            j = idx[0] if duplicate_policy != "last" else idx[-1]
            x = M[:, j].ravel()
        elif duplicate_policy == "sum":
            x = M[:, idx].sum(axis=1)
        else:
            x = M[:, idx].mean(axis=1)
    return np.asarray(x, dtype=float)


# ------------------------- Moran Tools -------------------------

def build_rowstd_connectivities(
    adata,
    *,
    key: str = "connectivities",
    recompute: bool = False,
    n_neighbors: int = 15,
    use_rep: Optional[str] = "X_pca",
) -> sp.csr_matrix:
    """
    Return row-standardized SciPy CSR (W) from Scanpy's neighbor graph.
    Suitable for fast block matmuls used in morans_i_all_fast.
    """
    if recompute or key not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    W = adata.obsp[key].tocsr(copy=True)
    W.sort_indices()
    W = W.astype(np.float32, copy=False)
    rs = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    nz = rs > 0
    inv = np.zeros_like(rs, dtype=np.float32)
    inv[nz] = 1.0 / rs[nz]
    W = sp.diags(inv) @ W
    W.sum_cache = float(W.sum())  # cache S0
    return W


def build_pysal_weights(
    adata,
    key: str = "connectivities",
    *,
    recompute: bool = False,
    n_neighbors: int = 15,
    use_rep: str = "X_pca",
    method: str = "umap",
    random_state: int = 0,
):
    """
    Return a row-standardized PySAL weights object built from adata.obsp[key].
    """
    if recompute or (key not in adata.obsp):
        sc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            method=method,
            random_state=random_state,
        )
    A = adata.obsp[key].tocsr().astype(float)
    w = WSP(A).to_W()     # <-- correct constructor
    w.transform = "R"     # row-standardize
    return w


def morans_i(
    adata,
    genes: Sequence[str],
    w=None,
    *,
    weights_key: str = "connectivities",
    source: Literal["X", "raw"] | str = "X",
    duplicate_policy: Literal["mean", "sum", "first", "last"] = "mean",
    permutations: int = 999,
    center: bool = True,
) -> pd.DataFrame:
    """
    Moran's I (with permutation p) for a set of genes. Deterministic if `w` is reused.
    """
    if w is None:
        w = build_pysal_weights(adata, key=weights_key)

    rows = []
    for g in genes:
        x = _extract_gene_vector(adata, g, source=source, duplicate_policy=duplicate_policy)
        if center:
            x = x - x.mean()
        mi = Moran(x, w, permutations=permutations)
        rows.append({"gene": g, "I": mi.I, "p_sim": mi.p_sim, "z_sim": mi.z_sim})
    return pd.DataFrame(rows).sort_values("I", ascending=False).reset_index(drop=True)


def _ensure_csr_f32(A: sp.csr_matrix) -> sp.csr_matrix:
    A = A.tocsr(copy=False)
    A.sort_indices()
    return A.astype(np.float32, copy=False)


def _build_dedup_aggregator(
    var_names: np.ndarray,
    how: Literal["mean", "sum"] = "mean",
    case_insensitive: bool = True,
) -> Tuple[np.ndarray, sp.csr_matrix]:
    """
    Return (uniq_names, G) where X_uniq = X @ G collapses duplicate columns.
    """
    names = np.asarray(var_names).astype(str)
    keys = np.char.upper(names) if case_insensitive else names
    uniq, inv = np.unique(keys, return_inverse=True)
    n, m = names.size, uniq.size
    data = np.ones(n, dtype=np.float32)
    rows = np.arange(n, dtype=np.int32)
    cols = inv.astype(np.int32)
    G = sp.csr_matrix((data, (rows, cols)), shape=(n, m))
    if how == "mean":
        col_counts = np.asarray(G.sum(axis=0)).ravel().astype(np.float32)
        G = G @ sp.diags(1.0 / np.maximum(col_counts, 1.0))
    return uniq, G


def morans_i_all_fast(
    adata,
    *,
    W_rowstd: Optional[sp.csr_matrix] = None,     # row-standardized CSR
    weights_key: str = "connectivities",
    source: Literal["X", "raw"] | str = "X",
    deduplicate: Literal["none", "mean", "sum", "first", "last"] = "mean",
    block_genes: int = 1024,
    dtype=np.float32,
    center: bool = True,
) -> pd.DataFrame:
    """
    Fast Moran's I for all genes (no permutation p-values).
    Uses the identity I = (n/S0) * (x^T W x) / (x^T x), with row-standardized W.
    Returns DataFrame ['gene','I'] sorted by I desc.
    """
    if W_rowstd is None:
        W_rowstd = build_rowstd_connectivities(adata, key=weights_key)
    W = _ensure_csr_f32(W_rowstd)

    # pick matrix
    M, names = _select_matrix_and_names(adata, source)
    X = M.tocsr(copy=False) if sp.issparse(M) else np.asarray(M, dtype=dtype)

    # optional de-duplication (collapses columns once)
    if deduplicate == "none":
        out_names = np.asarray(names).astype(str)
    elif deduplicate in ("mean", "sum"):
        uniq, G = _build_dedup_aggregator(np.asarray(names).astype(str), how=deduplicate)
        X = (X @ G) if sp.issparse(X) else (np.asarray(X, dtype=dtype) @ G.toarray())
        out_names = uniq
    else:  # first / last
        df = pd.DataFrame({"name": np.asarray(names).astype(str)})
        keep_idx = (df[::-1].drop_duplicates("name").index[::-1]
                    if deduplicate == "last" else df.drop_duplicates("name").index)
        X = X[:, keep_idx]
        out_names = np.asarray(names)[keep_idx].astype(str)

    # cast shapes/types for block ops
    if sp.issparse(X):
        X = X.tocsc(copy=False)
        if X.dtype != dtype:
            X = X.astype(dtype)
    else:
        X = np.asarray(X, dtype=dtype)

    n = X.shape[0]
    S0 = getattr(W, "sum_cache", None) or float(W.sum())
    nfac = (n / S0) if S0 > 0 else 0.0

    # precompute means if centering
    if center:
        if sp.issparse(X):
            mu = (np.asarray(X.sum(axis=0)).ravel() / float(n)).astype(np.float32)
        else:
            mu = (X.sum(axis=0, dtype=np.float64) / float(n)).astype(np.float32)
    else:
        mu = np.zeros(X.shape[1], dtype=np.float32)

    G = X.shape[1]
    I_vals = np.empty(G, dtype=np.float32)

    for j0 in range(0, G, block_genes):
        j1 = min(G, j0 + block_genes)
        Xb = X[:, j0:j1]
        WXb = W @ Xb
        Xb = Xb.toarray().astype(np.float32, copy=False) if sp.issparse(Xb) else np.asarray(Xb, dtype=np.float32)
        WXb = WXb.toarray().astype(np.float32, copy=False) if sp.issparse(WXb) else np.asarray(WXb, dtype=np.float32)

        if center:
            mub = mu[j0:j1][None, :]
            Xc, WXc = (Xb - mub), (WXb - mub)  # because row-standardized W ⇒ W·1 = 1
        else:
            Xc, WXc = Xb, WXb

        num = (Xc * WXc).sum(axis=0, dtype=np.float64)
        den = (Xc * Xc).sum(axis=0, dtype=np.float64)
        den = np.where(den > 0, den, np.nan)
        I_vals[j0:j1] = nfac * (num / den).astype(np.float32)

    out = pd.DataFrame({"gene": out_names, "I": I_vals})
    out.sort_values("I", ascending=False, inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out


# ------------------------- Archetype Tools -------------------------

def _pearson_r_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculates row-wise Pearson correlation between two 2D arrays A and B."""
    if A.shape != B.shape:
        raise ValueError("Input arrays A and B must have the same shape.")
    if A.ndim != 2:
        raise ValueError("Input arrays must be 2D.")

    A_m = A - A.mean(axis=1, keepdims=True)
    B_m = B - B.mean(axis=1, keepdims=True)

    ssA = (A_m**2).sum(axis=1)
    ssB = (B_m**2).sum(axis=1)

    # Add epsilon to denominator to avoid division by zero
    denom = np.sqrt(ssA * ssB) + 1e-9

    dot_prod = (A_m * B_m).sum(axis=1)

    corrs = dot_prod / denom
    # Clip values to be strictly within [-1, 1]
    return np.clip(corrs, -1.0, 1.0)


def find_expression_archetypes(
    adata: sc.AnnData,
    gene_list: List[str],
    num_clusters: int,
    *,
    source: Literal["X", "raw"] | str = "X",
) -> Dict[str, Any]:
    """
    Clusters genes based on expression data and finds gene archetypes.

    Re-implements the user's function to work with AnnData and a specified gene list.
    The clustering is hierarchical (Ward method) on the genes based on their
    expression profiles across all cells.

    Args:
        adata: The annotated data matrix.
        gene_list: A list of gene names to use for clustering (e.g., top Moran's I genes).
        num_clusters: The target number of gene clusters (archetypes).
        source: The data source in adata to use ('X', 'raw', or a layer name).

    Returns:
        A dictionary containing:
        - 'archetypes': (num_clusters, n_cells) array of archetype profiles.
        - 'clusters': (n_genes,) array of cluster assignments (1-based) for each gene.
        - 'gene_corrs': (n_genes,) array of Pearson correlations for each gene
                          to its assigned archetype.
        - 'gene_list': The list of gene names used, in the same order as 'clusters'
                       and 'gene_corrs'.
        - 'linkage_matrix': The (n_genes-1, 4) linkage matrix from hierarchy.ward.
    """
    print('Finding gene archetypes ... ', flush=True, end='')

    if not gene_list:
        raise ValueError("`gene_list` cannot be empty.")

    if num_clusters < 1:
        raise ValueError("`num_clusters` must be at least 1.")

    # 1. Get the full (n_cells, n_all_genes) matrix and all gene names
    M, all_names = _select_matrix_and_names(adata, source=source)
    all_names_index = pd.Index(all_names)

    # 2. Find the integer indices for the requested gene_list
    # We use get_indexer to maintain the order of gene_list
    var_idx = all_names_index.get_indexer(gene_list)

    # Check for missing genes
    missing_mask = (var_idx == -1)
    if missing_mask.any():
        missing_genes = np.array(gene_list)[missing_mask]
        warnings.warn(
            f"The following genes were not found in source '{source}' and will be skipped: "
            f"{list(missing_genes)}"
        )
        # Filter out missing genes and their indices
        var_idx = var_idx[~missing_mask]
        final_gene_list = list(np.array(gene_list)[~missing_mask])
        if not final_gene_list:
             raise ValueError("No requested genes were found in the data source.")
    else:
        final_gene_list = gene_list

    # 3. Create the (n_cells, n_genes) sub-matrix
    M_sub = M[:, var_idx]

    # 4. Transpose to (n_genes, n_cells) and ensure dense for clustering
    # This is the `sdge` matrix from the original function
    if sp.issparse(M_sub):
        sdge = M_sub.toarray().T
    else:
        sdge = np.asarray(M_sub).T

    if sdge.shape[0] <= 1:
        raise ValueError(f"Only {sdge.shape[0]} valid genes found. Cannot perform clustering.")

    # Ensure num_clusters is not more than the number of genes
    if num_clusters > sdge.shape[0]:
        warnings.warn(f"`num_clusters` ({num_clusters}) is greater than the number of "
                      f"genes ({sdge.shape[0]}). Setting `num_clusters` to {sdge.shape[0]}.")
        num_clusters = sdge.shape[0]

    # 5. Perform hierarchical clustering on genes
    # hierarchy.ward computes the linkage matrix from the (n_genes, n_cells) matrix
    linkage_matrix = hierarchy.ward(sdge)
    clusters = hierarchy.fcluster(linkage_matrix,
                                  num_clusters,
                                  criterion='maxclust')

    # 6. Calculate archetypes (average profile for each cluster)
    archetypes = np.array([
        np.mean(sdge[np.where(clusters == i)[0], :], axis=0)
        for i in range(1, num_clusters + 1)
    ])

    # 7. Calculate Pearson correlations for each gene to its archetype (vectorized)

    # Create an (n_genes, n_cells) array where each row is the archetype
    # corresponding to that gene's cluster.
    # `clusters` is 1-based, so subtract 1 for 0-based indexing.
    archetypes_per_gene = archetypes[clusters - 1]

    # Use the fast vectorized Pearson correlation
    gene_corrs = _pearson_r_vectorized(sdge, archetypes_per_gene)

    print('done')

    return {
        "archetypes": archetypes,
        "clusters": clusters,
        "gene_corrs": gene_corrs,
        "gene_list": final_gene_list,
        "linkage_matrix": linkage_matrix  # <-- ADDED THIS
    }


def plot_archetype_summary(
    adata: sc.AnnData,
    archetype_results: Dict[str, Any],
    archetype_id: int,
    *,
    k: int = 5,
    use_rep: str = "X_umap",
    **kwargs
) -> None:
    """
    Plots an archetype's score and its top k genes on an embedding.

    Args:
        adata: The annotated data matrix.
        archetype_results: The dictionary output from `find_expression_archetypes`.
        archetype_id: The cluster ID (1-based) of the archetype to plot.
        k: The number of top correlated genes to plot.
        use_rep: The embedding to use (e.g., "X_umap", "X_pca").
        **kwargs: Additional arguments passed to `sc.pl.embedding`.
    """
    # Extract results
    try:
        archetypes = archetype_results["archetypes"]
        clusters = archetype_results["clusters"]
        gene_corrs = archetype_results["gene_corrs"]
        gene_list = archetype_results["gene_list"]
    except KeyError:
        raise ValueError("`archetype_results` dictionary is missing required keys.")

    num_clusters = archetypes.shape[0]
    if not (1 <= archetype_id <= num_clusters):
        raise ValueError(f"`archetype_id` must be between 1 and {num_clusters}.")

    # 1. Get the archetype profile to plot
    arch_idx = archetype_id - 1  # Convert to 0-based index
    archetype_profile = archetypes[arch_idx, :]

    # Store in adata.obs for plotting
    profile_name = f'archetype_{archetype_id}_score'
    adata.obs[profile_name] = archetype_profile

    # 2. Find the top k genes for this archetype
    # Find indices (relative to gene_list) of genes in this cluster
    gene_indices_in_cluster = np.where(clusters == archetype_id)[0]

    if gene_indices_in_cluster.size == 0:
        warnings.warn(f"Archetype {archetype_id} has no assigned genes. Cannot plot top genes.")
        top_k_genes = []
    else:
        # Get their correlations
        corrs_in_cluster = gene_corrs[gene_indices_in_cluster]

        # Get the top k indices (relative to `gene_indices_in_cluster`)
        # Use min(k, ...) in case cluster has fewer than k genes
        num_top_genes = min(k, gene_indices_in_cluster.size)
        top_k_local_idx = np.argsort(corrs_in_cluster)[-num_top_genes:][::-1]

        # Get the original gene indices (relative to `gene_list`)
        top_k_global_idx = gene_indices_in_cluster[top_k_local_idx]

        # Get the gene names
        top_k_genes = [gene_list[i] for i in top_k_global_idx]

    # 3. Plotting
    basis = use_rep.replace('X_', '') # sc.pl.embedding wants 'umap', not 'X_umap'

    print(f"Plotting Archetype {archetype_id} Score")
    sc.pl.embedding(
        adata,
        basis=basis,
        color=profile_name,
        title=f'Archetype {archetype_id} Score',
        **kwargs
    )

    if top_k_genes:
        print(f"Plotting Top {len(top_k_genes)} Genes for Archetype {archetype_id}")
        sc.pl.embedding(
            adata,
            basis=basis,
            color=top_k_genes,
            title=f'Top {len(top_k_genes)} Genes for Archetype {archetype_id}',
            **kwargs
        )

    # Clean up obs
    del adata.obs[profile_name]


def plot_archetype_assignments(
    adata: sc.AnnData,
    archetype_results: Dict[str, Any],
    *,
    use_rep: str = "X_umap",
    **kwargs
) -> None:
    """
    Plots each archetype's score on its own subplot on an embedding.

    This helps visualize the "spatial" or "latent space" distribution
    of all archetypes simultaneously.

    Args:
        adata: The annotated data matrix.
        archetype_results: The dictionary output from `find_expression_archetypes`.
        use_rep: The embedding to use (e.g., "X_umap", "X_pca").
        **kwargs: Additional arguments passed to `sc.pl.embedding`.
    """
    archetypes = archetype_results["archetypes"] # (n_clusters, n_cells)
    num_clusters = archetypes.shape[0]

    obs_names = []
    # Add each archetype score to adata.obs
    for i in range(num_clusters):
        profile_name = f'archetype_{i+1}_score'
        adata.obs[profile_name] = archetypes[i, :]
        obs_names.append(profile_name)

    basis = use_rep.replace('X_', '')

    print(f"Plotting {num_clusters} Archetype Scores per Cell")

    # Set a default `cmap` if not provided, as it looks better for continuous values
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'viridis'

    # Plotting all archetypes as subplots
    sc.pl.embedding(
        adata,
        basis=basis,
        color=obs_names, # Pass the list of names
        title='Archetype Scores per Cell', # Title will be adapted by scanpy
        **kwargs
    )

    # Clean up obs
    for name in obs_names:
        del adata.obs[name]


def plot_archetype_hierarchy(
    archetype_results: Dict[str, Any],
    *,
    p: int = 0,
    truncate_mode: Optional[str] = None,
    **kwargs
) -> None:
    """
    Plots the dendrogram of the gene clustering hierarchy.

    Args:
        archetype_results: The dictionary output from `find_expression_archetypes`.
        p: Number of clusters to show in truncated mode.
           If 0, defaults to `num_clusters`.
        truncate_mode: 'lastp' (show p last merged clusters), 'level', etc.
                       If None, defaults to 'lastp'.
        **kwargs: Additional arguments passed to `plt.figure` or `hierarchy.dendrogram`.
    """
    try:
        Z = archetype_results["linkage_matrix"]
        gene_list = archetype_results["gene_list"]
        num_clusters = archetype_results["archetypes"].shape[0]
    except KeyError:
        raise ValueError(
            "`archetype_results` dictionary is missing required keys: "
            "'linkage_matrix', 'gene_list', 'archetypes'."
        )

    # --- Smart Defaults ---
    # If p is not set, set it to num_clusters.
    if p == 0:
        p = num_clusters
    # If truncate_mode is not set, set it to 'lastp'
    if truncate_mode is None:
        truncate_mode = 'lastp'

    plt.figure(figsize=kwargs.pop('figsize', (12, 6)))
    plt.title('Gene Clustering Hierarchy (Dendrogram)')
    plt.ylabel('Distance (Ward)')

    # --- Set color threshold ---
    # We set the threshold to color the `num_clusters`
    color_threshold = 0
    if num_clusters > 1 and num_clusters <= Z.shape[0]:
        # We find the distance of the merge that creates `num_clusters-1` clusters
        # and the merge that creates `num_clusters` clusters, and average them.
        # Z[-(k), 2] is the distance of the merge that results in k clusters.
        try:
            dist_k = Z[-(num_clusters), 2]
            dist_k_minus_1 = Z[-(num_clusters - 1), 2]
            color_threshold = (dist_k + dist_k_minus_1) / 2.0
        except IndexError:
             # This can happen if num_clusters is very close to n_genes
             warnings.warn("Could not determine optimal color threshold. Using default.")
             color_threshold = 0


    # Set gene labels
    labels = np.asarray(gene_list)

    hierarchy.dendrogram(
        Z,
        labels=labels,
        truncate_mode=truncate_mode,
        p=p,
        leaf_rotation=90.,
        leaf_font_size=8.,
        color_threshold=color_threshold,
        above_threshold_color='grey', # Color for links above the threshold
        **kwargs
    )

    if truncate_mode == 'lastp':
         plt.xlabel(f"Genes / Gene Clusters (showing last {p} merged clusters)")
    else:
         plt.xlabel("Genes")

    plt.tight_layout()
    plt.show()


# ------------------------- Triku Tools -------------------------

def run_triku(
    adata,
    layer: Optional[str] = "log1p",   # <- run Triku on log1p by default
    use_raw: bool = False,
    n_features: int = 2000,
    n_neighbors: int = 15,
    n_pcs_for_knn: int = 30,
) -> pd.DataFrame:
    """
    Run Triku in-place and return a tidy DataFrame with scores.
    """
    if tk is None:
        raise ImportError("The 'triku' package is required for this function. Please install it.")

    sc.pp.filter_genes(adata, min_cells=3)

    # If a layer is specified (recommended: "log1p"), set it as active for selection/knn
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Requested layer '{layer}' not found. Build it before Triku.")
        _X_backup = adata.X
        adata.X = adata.layers[layer].copy()
    else:
        _X_backup = None

    # KNN graph for Triku on the current X (log1p); use PCA for stability/speed
    # This avoids the earlier "X_pca doesn't exist" issue.
    sc.tl.pca(adata, n_comps=min(50, adata.n_vars - 1), random_state=0)  # small, temporary PCA
    ensure_neighbors(adata, n_neighbors=n_neighbors, use_rep=None, n_pcs=n_pcs_for_knn)

    tk.tl.triku(
        adata,
        n_features=n_features,
        use_raw=use_raw,
    )
    out = adata.var[["triku_highly_variable", "triku_distance"]].copy()
    out.index.name = "gene"
    out.sort_values("triku_distance", ascending=False, inplace=True)

    # restore X if we changed it
    if _X_backup is not None:
        adata.X = _X_backup
        # Drop the temp PCA/KNN computed for triku to avoid confusion
        for k in ("X_pca",):
            if k in adata.obsm: del adata.obsm[k]
        for k in ("neighbors", "distances", "connectivities"):
            if k in adata.uns: del adata.uns[k]
            if k in adata.obsp: del adata.obsp[k]

    return out


def triku_gene_set(adata, min_score: Optional[float] = None) -> set:
    """
    Return set of genes kept by Triku, optionally thresholded by triku_distance.
    """
    if "triku_highly_variable" not in adata.var or "triku_distance" not in adata.var:
        raise RuntimeError("Run `run_triku` first.")
    mask = adata.var["triku_highly_variable"].astype(bool)
    if min_score is not None:
        mask &= adata.var["triku_distance"] >= float(min_score)
    return set(adata.var_names[mask])


def filter_marker_dict_by_triku(
    adata,
    marker_dict: Dict[str, Iterable[str]],
    min_score: Optional[float] = None,
) -> Dict[str, List[str]]:
    """
    Keep only genes selected by Triku (case-insensitive against adata.var_names).
    """
    keep = triku_gene_set(adata, min_score=min_score)
    keep_lower = {g.lower() for g in keep}
    out = {}
    for t, genes in marker_dict.items():
        kept = []
        for g in genes:
            if (g in keep) or (g.lower() in keep_lower):
                kept.append(g)
        out[t] = kept
    return out
