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
#     "pacmap",
#     "scikit-learn",
# ]
# ///

from __future__ import annotations
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Literal, Any
import warnings
from collections import defaultdict
from uuid import uuid4

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from .plotting_utils import adjust_colormap

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

def plot_marker_genes_dict_on_embedding(
    adata,
    marker_genes: Dict[str, List[str] | str],
    basis: str = 'X_umap',
    colormaps: Optional[List[str]] = None,
    **pl_kwargs
) -> List[plt.Axes]:
    """
    Plot marker genes defined in a dictionary {tissue: [genes]} on an embedding (e.g. UMAP).

    Parameters
    ----------
    adata
        Annotated data matrix.
    marker_genes
        Dictionary where keys are tissue names (or categories) and values are lists of gene names.
    basis
        The basis to plot on (e.g., 'X_umap', 'X_pca'). Defaults to 'X_umap'.
    colormaps
        List of colormap names to cycle through for different tissues.
    **pl_kwargs
        Additional keyword arguments passed to `sc.pl.embedding`.
        Defaults set: s=50, show=False, frameon=False.

    Returns
    -------
    List[plt.Axes]
        A list of matplotlib Axes objects containing the plots.
    """

    # 1. Validate Basis
    key_to_check = f"X_{basis}" if not basis.startswith("X_") else basis
    # Scanpy usually looks for 'X_basis' if 'basis' is passed as argument 'basis'.
    # e.g., sc.pl.embedding(..., basis='umap') looks for 'X_umap'.
    # Here we check if the relevant key exists in obsm.

    # Common behavior: user passes 'umap', expects 'X_umap'.
    # Or user passes 'X_umap', expects 'X_umap'.
    possible_keys = [basis, f"X_{basis}" if not basis.startswith("X_") else basis]

    # Actually, scanpy's sc.pl.embedding `basis` argument handles the X_ prefix automatically.
    # But for our explicit check as requested:
    # We check if the likely key exists.

    basis_key_exists = False
    for k in possible_keys:
        if k in adata.obsm:
            basis_key_exists = True
            break

    if not basis_key_exists:
        raise ValueError(
            f"Basis '{basis}' (checked {possible_keys}) not found in adata.obsm. "
            f"Please compute it and add in obsm, or choose from available keys: {list(adata.obsm.keys())}"
        )

    # 2. Check Genes
    # Updates the marker_genes dict to only include found genes
    marker_genes = check_gene_adata(adata, marker_genes)

    # 3. Setup Colormaps
    if colormaps is None:
        colormaps = [
            'Blues', 'Reds', 'Purples', 'Oranges',
            'Greens', 'Greens', 'YlOrBr', 'Greens',
            'RdPu', 'Oranges', 'PuRd', 'YlOrBr',
        ]

    adjusted_colormaps = {cmap: adjust_colormap(cmap) for cmap in set(colormaps)}

    # 4. Default Kwargs
    pl_kwargs.setdefault('s', 50)
    pl_kwargs.setdefault('show', False)
    pl_kwargs.setdefault('frameon', False)

    # 5. Plotting Loop
    axes_list = []

    for idx_i, (tissue, genes) in enumerate(marker_genes.items()):
        if not genes:
            print(f"Skipping {tissue}: No valid marker genes found.")
            continue

        # Calculate module score
        score_name = f"{tissue}_score"
        # Check if use_raw is in pl_kwargs to pass to score_genes
        use_raw = pl_kwargs.get("use_raw", None)

        score_computed = False
        try:
            sc.tl.score_genes(
                adata,
                gene_list=genes,
                score_name=score_name,
                use_raw=use_raw
            )
            score_computed = True
        except Exception as e:
            print(f"Could not compute score for {tissue}: {e}")
            score_name = None

        current_cmap_name = colormaps[idx_i % len(colormaps)]
        current_cmap = adjusted_colormaps[current_cmap_name]

        # Prepare items to plot
        items_to_plot = list(genes)
        if score_computed and score_name:
            items_to_plot.append(score_name)

        # sc.pl.embedding returns a list of axes if show=False and multiple genes are plotted,
        # or a single axis if one gene. Or None if show=True.
        # Here 'color' is a list of genes.
        res = sc.pl.embedding(
            adata,
            basis=basis,
            color=items_to_plot,
            cmap=current_cmap,
            **pl_kwargs
        )

        # Remove the score column from obs
        if score_computed and score_name in adata.obs:
            del adata.obs[score_name]

        # Normalize result to a single axis or list of axes
        # Scanpy's embedding returns: Union[Axes, List[Axes], None]
        # If multiple genes, it returns List[Axes].

        # The user snippet logic:
        # ax = axes[0] if isinstance(axes, list) and axes else axes
        # This implies we want to grab the axis to modify it.
        # Warning: if multiple genes are passed, sc.pl.embedding creates multiple subplots (one per gene).
        # The user snippet seems to assume one plot or just modifies the first?
        # "Plotting UMAPs for each tissue type ... color=genes"
        # If 'genes' is a list, scanpy plots multiple panels.
        # The user's code: "ax = axes[0] ... ax.set_ylabel(tissue...)"
        # This puts the tissue label on the first panel.

        # If 'res' is a list, scanpy returned multiple subplots (one per gene).
        # If 'res' is a single Axes, scanpy returned one plot (one gene).

        ax = res[0] if isinstance(res, list) and res else res

        # We disabled axis drawing in UMAP to have plots without background and border
        # so we need to re-enable axis to plot the ylabel
        # We apply this to the first axis (where we put the ylabel), as per snippet logic.

        if ax:
            ax.axis("on")
            ax.tick_params(
                top="off",
                bottom="off",
                left="off",
                right="off",
                labelleft="on",
                labelbottom="off",
            )
            ax.set_ylabel(tissue + "\n", rotation=90, fontsize=14)
            ax.set_xlabel(" ", fontsize=1)
            ax.set(frame_on=False)

        # Collect axes.
        # Since we modified the first axis in place, 'res' (if list) or 'res' (if object) still holds it.
        # But wait: if res is a list, I appended 'ax' (res[0]) in my previous logic, then appended 'res' (the list).
        # This resulted in [ax, ax, ax2, ax3]. That's bad.

        if isinstance(res, list):
            axes_list.extend(res)
        else:
            axes_list.append(res)

    return axes_list


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
    max_genes_for_triku: int = 20000,
) -> pd.DataFrame:
    """
    Run Triku in-place and return a tidy DataFrame with scores.

    To avoid memory issues or warnings with large gene sets, this function:
      1. Creates a temporary subset of genes (filtering < 3 cells and keeping top
         `max_genes_for_triku` by total counts).
      2. Runs Triku on this subset (using a copy to protect the original adata).
      3. Maps the results back to `adata.var` (setting unselected genes to False/NaN).
    """
    if tk is None:
        raise ImportError("The 'triku' package is required for this function. Please install it.")

    # --- 1. Identify genes to process (Heuristic Filtering) ---

    # Calculate basic QC metrics if not present (to filter min_cells and sort by counts)
    # We do this manually to avoid modifying adata.var with sc.pp.calculate_qc_metrics
    if sp.issparse(adata.X):
        n_cells_per_gene = adata.X.getnnz(axis=0)
        total_counts_per_gene = np.asarray(adata.X.sum(axis=0)).ravel()
    else:
        n_cells_per_gene = np.count_nonzero(adata.X, axis=0)
        total_counts_per_gene = np.asarray(adata.X.sum(axis=0)).ravel()

    # Step A: Filter genes with < 3 cells (temporary)
    keep_mask = n_cells_per_gene >= 3

    # Step B: If still too many genes, keep top `max_genes_for_triku` by total counts
    n_remaining = np.sum(keep_mask)
    if n_remaining > max_genes_for_triku:
        print(f"Filtering genes for Triku: reducing from {n_remaining} to {max_genes_for_triku} by total counts.")
        # Get indices of genes that passed the first mask
        valid_indices = np.where(keep_mask)[0]
        # Get their counts
        valid_counts = total_counts_per_gene[valid_indices]
        # Sort indices by count descending
        sorted_local_indices = np.argsort(valid_counts)[::-1]
        # Keep top N
        top_local_indices = sorted_local_indices[:max_genes_for_triku]
        # Map back to global indices
        final_indices = valid_indices[top_local_indices]

        # Create final mask
        final_mask = np.zeros(adata.n_vars, dtype=bool)
        final_mask[final_indices] = True
        keep_mask = final_mask

    # --- 2. Create Temporary Adata for Triku ---

    # We create a copy of the subset to ensure Triku runs in isolation
    # (protecting original adata from inplace changes like PCA/Neighbors/filtering)
    adata_triku = adata[:, keep_mask].copy()

    # If a layer is specified (recommended: "log1p"), set it as active for selection/knn
    if layer is not None:
        if layer not in adata_triku.layers:
            # If layer is missing in copy, check original.
            # Note: slicing adata[:, mask] should preserve layers.
            raise KeyError(f"Requested layer '{layer}' not found. Build it before Triku.")
        adata_triku.X = adata_triku.layers[layer].copy()

    # KNN graph for Triku on the current X (log1p); use PCA for stability/speed
    # This avoids the earlier "X_pca doesn't exist" issue.
    # Note: We run this on the *subset* (adata_triku), which is faster.
    sc.tl.pca(adata_triku, n_comps=min(50, adata_triku.n_vars - 1), random_state=0)
    ensure_neighbors(adata_triku, n_neighbors=n_neighbors, use_rep=None, n_pcs=n_pcs_for_knn)

    # Run Triku on the subset
    tk.tl.triku(
        adata_triku,
        n_features=n_features,
        use_raw=use_raw,
    )

    # --- 3. Map Results Back to Original Adata ---

    # Initialize columns in original adata
    adata.var["triku_highly_variable"] = False
    adata.var["triku_distance"] = np.nan # Or 0.0, but NaN distinguishes "not calculated" from "low score"

    # Map values from the subset back to the original using index (gene names)
    # adata_triku.var contains the results. We align by index.

    # Update boolean mask
    # We can rely on pandas index alignment
    adata.var.update(adata_triku.var[["triku_highly_variable"]])
    # Note: update works in place. "triku_highly_variable" in adata_triku is likely boolean.
    # We should ensure the column in adata is boolean (though NaNs might force object/float if not careful, but update handles overlap)
    # Actually, pandas update doesn't add new rows, it updates existing.

    # For safe updates:
    # 1. Extract the series
    hv_series = adata_triku.var["triku_highly_variable"]
    dist_series = adata_triku.var["triku_distance"]

    # 2. Assign to original adata at the specific gene locations
    # Using .loc to align by gene name
    adata.var.loc[hv_series.index, "triku_highly_variable"] = hv_series
    adata.var.loc[dist_series.index, "triku_distance"] = dist_series

    # Ensure boolean type for highly_variable (NaNs might have cast it to object/float if alignment failed, but we initialized with False)
    adata.var["triku_highly_variable"] = adata.var["triku_highly_variable"].astype(bool)

    # Construct the return DataFrame
    # Returning the subset of results (sorted) is usually most useful
    out = adata.var.loc[adata_triku.var_names, ["triku_highly_variable", "triku_distance"]].copy()
    out.index.name = "gene"
    out.sort_values("triku_distance", ascending=False, inplace=True)

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


# ------------------------- PaCMAP Tools -------------------------

def tl_pacmap(
    adata: sc.AnnData,
    n_neighbors: Optional[int] = None,
    MN_ratio: float = 0.5,
    FP_ratio: float = 2.0,
    n_components: int = 2,
    use_rep: str = "X_pca",
    random_state: int = 42,
    **kwargs,
) -> None:
    """
    Compute PaCMAP embedding.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        Number of neighbors for PaCMAP. If None, PaCMAP defaults are used (usually 10-30 depending on n).
    MN_ratio
        Ratio of mid-near pairs to near pairs.
    FP_ratio
        Ratio of far pairs to near pairs.
    n_components
        The dimension of the embedding.
    use_rep
        Representation to use for computing the embedding.
        Default is 'X_pca'. If 'X_pca' is not present, it will fall back to .X.
    random_state
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to pacmap.PaCMAP.

    Returns
    -------
    None
        Updates adata.obsm['X_pacmap'] with the embedding.
    """
    try:
        import pacmap
    except ImportError:
        raise ImportError(
            "PaCMAP is not installed. Please install it using: `uv pip install pacmap`"
        )

    # 1. Print Parameters
    print(f"Computing PaCMAP with parameters:")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  MN_ratio: {MN_ratio}")
    print(f"  FP_ratio: {FP_ratio}")
    print(f"  n_components: {n_components}")
    print(f"  use_rep: {use_rep}")
    print(f"  random_state: {random_state}")
    if kwargs:
        print(f"  kwargs: {kwargs}")

    # 2. Extract Data
    if use_rep in adata.obsm:
        X_in = adata.obsm[use_rep]
        print(f"Using representation '{use_rep}' from adata.obsm.")
    elif use_rep == "X":
        X_in = adata.X
        print(f"Using adata.X directly.")
    else:
        warnings.warn(
            f"Representation '{use_rep}' not found in adata.obsm. Falling back to adata.X."
        )
        X_in = adata.X

    # Handle sparse input if necessary (PaCMAP usually expects dense or handles sparse internally?
    # PaCMAP documentation says it accepts numpy arrays. It might handle sparse, but safer to densify if small enough,
    # or let PaCMAP handle it if it supports it. Let's convert to array if sparse to be safe as PaCMAP is numba based)
    if sp.issparse(X_in):
        print("Input is sparse, converting to dense array for PaCMAP...")
        X_in = X_in.toarray()

    # 3. Initialize and Fit
    # Filter out None values from kwargs to let defaults shine
    pmap_kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Extract 'init' if present, so it's not passed to the constructor
    user_init = pmap_kwargs.pop("init", None)

    # Pass n_neighbors only if not None
    if n_neighbors is not None:
        pmap_kwargs["n_neighbors"] = n_neighbors

    embedder = pacmap.PaCMAP(
        n_components=n_components,
        MN_ratio=MN_ratio,
        FP_ratio=FP_ratio,
        random_state=random_state,
        **pmap_kwargs
    )

    # Determine initialization method
    if user_init is not None:
        init_method = user_init
    else:
        # Check feature count to determine initialization method
        n_features = X_in.shape[1]
        init_method = "pca"

        if n_features <= 100:
            warnings.warn(
                f"Input data has {n_features} features, which is <= 100. "
                "Switching initialization from 'pca' to 'random' to avoid potential issues."
            )
            init_method = "random"

    print(f"Fitting PaCMAP using init='{init_method}'...")
    X_embedded = embedder.fit_transform(X_in, init=init_method)

    # 4. Store Result
    adata.obsm["X_pacmap"] = X_embedded
    print(f"PaCMAP embedding finished. Result stored in `adata.obsm['X_pacmap']`.")
    
    
# ------------------------- Multiscale Coarsening -------------------------

def multiscale_coarsening(
    adata: sc.AnnData,
    resolutions: Optional[List[float]] = None,
    use_rep: str = "X_pca",
    store_in_uns: bool = False,
    uns_key: str = "multiscale_hierarchy",
    return_output: bool = True,
    random_state: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    Perform multiscale Leiden clustering and establish a hierarchy between resolutions.

    Clusters the data at multiple resolutions (independent Leiden runs) and links clusters
    across scales based on overlap (majority vote). Also calculates centroids and checks for
    lineage inconsistencies.

    Args:
        adata: Annotated data matrix.
        resolutions: List of resolutions for Leiden clustering.
                     Defaults to [0.5, 1.0, 10.0, 100.0].
        use_rep: Representation to use for clustering and centroid calculation (e.g., 'X_pca').
                 If 'X', uses adata.X.
        store_in_uns: Whether to store the results in .
        uns_key: Key under which to store results in .
        return_output: Whether to return the results dictionary.
        random_state: Random seed for Leiden clustering.

    Returns:
        A dictionary containing:
        - 'clustering': Dict of {resolution: pandas.Series of cluster labels}.
        - 'centroids': Dict of {resolution: pandas.DataFrame of centroids}.
        - 'hierarchy': Dict containing:
            - 'tree': Mapping of {(fine_res, coarse_res): {fine_cluster: coarse_cluster}}.
            - 'purity': Mapping of {(fine_res, coarse_res): {fine_cluster: purity_score}}.
              Purity is the fraction of cells in the fine cluster that belong to the assigned coarse cluster.
        - 'consistency': DataFrame flagging clusters with lineage inconsistencies (direct vs indirect parent mismatch).
    """

    # 1. Validation & Setup
    if resolutions is None:
        resolutions = [0.5, 1.0, 10.0, 100.0]

    # Sort resolutions: Low (coarse) -> High (fine)
    resolutions = sorted(list(set(resolutions)))

    if use_rep != "X" and use_rep not in adata.obsm:
        warnings.warn(f"Representation '{use_rep}' not found in adata.obsm. Using 'X' (data matrix).")
        use_rep = "X"

    # Ensure neighbors exist if we are going to run leiden
    if "connectivities" not in adata.obsp:
        print("Computing neighbors for Leiden clustering...")
        sc.pp.neighbors(adata, use_rep=use_rep if use_rep != "X" else None)

    # 2. Run Clustering
    clustering_results = {}
    obs_keys = []

    print(f"Running multiscale clustering on resolutions: {resolutions}")
    for res in resolutions:
        key = f"leiden_res_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key, random_state=random_state)
        clustering_results[res] = adata.obs[key].copy()
        obs_keys.append(key)

    # 3. Calculate Centroids
    centroids = {}

    # Get the data matrix for centroids
    if use_rep == "X":
        if sp.issparse(adata.X):
             data_matrix = adata.X.toarray()
        else:
             data_matrix = adata.X
    else:
        data_matrix = adata.obsm[use_rep]

    for res in resolutions:
        clusters = clustering_results[res]
        unique_clusters = clusters.unique().sort_values() # Categories are usually strings '0', '1'...

        # We need to ensure we iterate in a stable order corresponding to the cluster labels
        # Leiden labels are categorical strings.

        res_centroids = []
        for cluster_id in unique_clusters:
            mask = (clusters == cluster_id).to_numpy()
            if mask.any():
                centroid = data_matrix[mask].mean(axis=0)
                res_centroids.append(centroid)
            else:
                # Should not happen for existing clusters
                res_centroids.append(np.zeros(data_matrix.shape[1]))

        # Store as DataFrame for better labeling, or just array?
        # Array is cleaner for pure math, but DataFrame keeps indices.
        # Let's return a DataFrame indexed by cluster ID.
        centroids[res] = pd.DataFrame(
            np.array(res_centroids),
            index=unique_clusters,
            columns=[f"dim_{i}" for i in range(data_matrix.shape[1])]
        )

    # 4. Build Hierarchy & Linkage (All-to-All)
    hierarchy_tree = {}
    purity_scores = {}

    # Compare every fine resolution to every coarser resolution
    for i, res_fine in enumerate(resolutions):
        for j in range(i): # 0 to i-1 are coarser
            res_coarse = resolutions[j]

            fine_labels = clustering_results[res_fine]
            coarse_labels = clustering_results[res_coarse]

            # Confusion Matrix: Rows = Fine, Cols = Coarse
            confusion = pd.crosstab(fine_labels, coarse_labels)

            # For each fine cluster, find the coarse cluster with max overlap
            mapping = {}
            purity = {}

            for f_clust in confusion.index:
                # Row for this fine cluster
                counts = confusion.loc[f_clust]
                if counts.sum() == 0:
                    continue

                # Dominant parent
                best_parent = counts.idxmax()
                max_count = counts.max()
                total_count = counts.sum()

                mapping[f_clust] = best_parent
                purity[f_clust] = max_count / total_count

            hierarchy_tree[(res_fine, res_coarse)] = mapping
            purity_scores[(res_fine, res_coarse)] = purity

    # 5. Consistency Check (Lineage Analysis)
    # We check triplets: Fine -> Mid -> Coarse
    # Direct: Fine -> Coarse (mapped directly)
    # Indirect: Fine -> Mid (mapped) -> Coarse (mapped)

    inconsistencies = []

    for i in range(2, len(resolutions)):
        res_fine = resolutions[i]
        for j in range(1, i):
            res_mid = resolutions[j]
            for k in range(j):
                res_coarse = resolutions[k]

                # Maps
                map_fine_mid = hierarchy_tree.get((res_fine, res_mid), {})
                map_mid_coarse = hierarchy_tree.get((res_mid, res_coarse), {})
                map_fine_coarse_direct = hierarchy_tree.get((res_fine, res_coarse), {})

                for f_clust, mid_parent in map_fine_mid.items():
                    # Indirect path
                    if mid_parent in map_mid_coarse:
                        indirect_grandparent = map_mid_coarse[mid_parent]
                    else:
                        continue # Should not happen if maps are complete

                    # Direct path
                    direct_grandparent = map_fine_coarse_direct.get(f_clust)

                    if direct_grandparent != indirect_grandparent:
                        inconsistencies.append({
                            "fine_res": res_fine,
                            "mid_res": res_mid,
                            "coarse_res": res_coarse,
                            "fine_cluster": f_clust,
                            "mid_parent": mid_parent,
                            "direct_grandparent": direct_grandparent,
                            "indirect_grandparent": indirect_grandparent,
                            "purity_fine_mid": purity_scores[(res_fine, res_mid)].get(f_clust, 0),
                            "purity_mid_coarse": purity_scores[(res_mid, res_coarse)].get(mid_parent, 0),
                            "purity_fine_coarse": purity_scores[(res_fine, res_coarse)].get(f_clust, 0)
                        })

    consistency_df = pd.DataFrame(inconsistencies)

    results = {
        "clustering": clustering_results,
        "centroids": centroids,
        "hierarchy": {
            "tree": hierarchy_tree,
            "purity": purity_scores
        },
        "consistency": consistency_df,
        "resolutions": resolutions
    }

    if store_in_uns:
        adata.uns[uns_key] = results

    if return_output:
        return results
    return None

# ------------------------- Cell Type Annotation -------------------------

def _normalize_symbol(s: str) -> str:
    """Robust normalization for matching gene symbols."""
    return str(s).strip().upper().replace("_", "-")


def _dedupe_preserve_order(seq: Iterable) -> List:
    """Deduplicate sequence while preserving original order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _build_norm_map(adata: sc.AnnData, extra_cols: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Build a mapping from normalized symbol -> actual var_name in `adata`.
    Includes var_names and (if present) selected alternate symbol columns.
    """
    if extra_cols is None:
        extra_cols = [
            "gene_symbol", "gene_symbols", "gene_name", "Gene", "Symbol",
            "feature_name", "features", "name"
        ]
    norm_map = {}

    # 1) var_names / index
    for var_name in adata.var_names:
        norm_map[_normalize_symbol(var_name)] = var_name

    # 2) alternate columns (row-aligned)
    for col in extra_cols:
        if col in adata.var.columns:
            col_series = adata.var[col]
            # tolerate non-string / NaN
            for idx, val in col_series.items():
                if val is None:
                    continue
                sval = str(val).strip()
                if sval == "" or sval.lower() == "nan":
                    continue
                norm_map[_normalize_symbol(sval)] = str(idx)  # idx is var_name for that row

    return norm_map


def filter_markers_to_adata(
    marker_genes: Dict[str, List[str]],
    adata: sc.AnnData,
    extra_cols: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Filter marker genes to those present in `adata`.

    Parameters
    ----------
    marker_genes
        Dictionary of {cell_type: [gene_list]}.
    adata
        Annotated data matrix.
    extra_cols
        List of column names in `adata.var` to check for alternative gene symbols.
    verbose
        Whether to print a summary of removed/replaced markers.

    Returns
    -------
    cleaned
        Filtered marker dictionary with valid gene names.
    removed
        Dictionary of {cell_type: [removed_genes]}.
    replacements
        Dictionary of {cell_type: {original_gene: matched_var_name}}.
    """
    norm_map = _build_norm_map(adata, extra_cols=extra_cols)

    cleaned = {}
    removed = defaultdict(list)
    replacements = defaultdict(dict)  # original -> matched var_name (if case/alias fix)

    for tissue, genes in marker_genes.items():
        keep = []
        for g in genes:
            key = _normalize_symbol(g)
            hit = norm_map.get(key, None)
            if hit is not None:
                keep.append(hit)
                if hit != g:
                    replacements[tissue][g] = hit
            else:
                removed[tissue].append(g)

        cleaned[tissue] = _dedupe_preserve_order(keep)

    # prune empties from reporting dicts
    removed = {k: v for k, v in removed.items() if len(v) > 0}
    replacements = {k: v for k, v in replacements.items() if len(v) > 0}

    if verbose:
        total_removed = sum(len(v) for v in removed.values())
        print(f"[marker filter] {total_removed} markers removed across {len(removed)} categories.")
        if replacements:
            print("[marker filter] Case/alias replacements (original -> adata var_name):")
            for k, m in replacements.items():
                print(f"  - {k}: {m}")
        if removed:
            print("[marker filter] Removed (not found in adata):")
            for k, lost in removed.items():
                print(f"  - {k} ({len(lost)}): {lost}")

    return cleaned, removed, replacements


def filter_markers_by_moran(
    filtered: Dict[str, List[str]],
    df_top_dict: pd.DataFrame,
    *,
    gene_col: str = "gene",
    score_col: str = "I",
    threshold: float = 0.10,
    top_k: int = 5,
    min_k: int = 3,
    fallback_to_min_k: bool = True,
    case_insensitive: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, Any]]:
    """
    From `filtered` (category -> [genes]) keep only genes found in `df_top_dict`
    with Moran's I >= `threshold`, then return top_k by Moran's I (desc).
    If fewer than `min_k` pass the threshold and `fallback_to_min_k` is True,
    backfill with the best-scoring genes (even if below threshold) until min_k.

    Returns
    -------
    filtered_by_moran : dict
        {category: [genes]}
    report : dict
        Per-category details: missing_in_df, below_threshold, used_fallback
    """

    # Build score lookup
    # If the DF has duplicates, keep the maximum score per gene.
    df = df_top_dict[[gene_col, score_col]].copy()
    # normalize gene key
    def norm(x):
        return str(x).upper() if case_insensitive else str(x)

    df["_key"] = df[gene_col].map(norm)
    # best score per gene
    best = df.sort_values(score_col, ascending=False).drop_duplicates("_key")
    score_map = dict(zip(best["_key"], best[score_col]))
    # also map canonical name to return (use the 'gene' from DF if case differs)
    canon_map = dict(zip(best["_key"], best[gene_col]))

    filtered_by_moran = {}
    report = {}

    for cat, genes in filtered.items():
        # normalize and gather scores
        present = []
        missing_in_df = []
        for g in genes:
            key = norm(g)
            if key in score_map and pd.notna(score_map[key]):
                # keep original symbol from filtered if it matches canon ignoring case,
                # otherwise use DF's canon (helps unify case/alias differences)
                out_name = g if norm(g) == norm(canon_map[key]) else canon_map[key]
                present.append((out_name, float(score_map[key])))
            else:
                missing_in_df.append(g)

        # sort by Moran's I desc
        present.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # apply threshold
        above = [(g, s) for (g, s) in present if s >= threshold]
        below = [(g, s) for (g, s) in present if s < threshold]

        picked = above[:top_k]
        used_fallback = False
        if len(picked) < min_k and fallback_to_min_k:
            need = min_k - len(picked)
            # take best from below to reach min_k (avoid duplicates)
            for g, s in below:
                if g not in [x[0] for x in picked]:
                    picked.append((g, s))
                    need -= 1
                    if need == 0:
                        break
            used_fallback = True if len(above) < min_k else False

        # store only gene names
        filtered_by_moran[cat] = [g for g, _ in picked]

        report[cat] = {
            "n_input": len(genes),
            "n_present_in_df": len(present),
            "n_above_threshold": len(above),
            "missing_in_df": missing_in_df,
            "below_threshold": [g for g, _ in below],
            "used_fallback": used_fallback,
        }

    if verbose:
        total_missing = sum(len(r["missing_in_df"]) for r in report.values())
        total_below = sum(len(r["below_threshold"]) for r in report.values())
        print(f"[moran filter] threshold={threshold}, top_k={top_k}, min_k={min_k}")
        print(f"[moran filter] Missing in df_top_dict: {total_missing} genes across categories.")
        print(f"[moran filter] Below threshold: {total_below} genes across categories.")
        fb_cats = [k for k, r in report.items() if r["used_fallback"]]
        if fb_cats:
            print(f"[moran filter] Fallback to min_k used for: {fb_cats}")

    return filtered_by_moran, report
# ------------------------- Annotation Core -------------------------

def score_celltypes(
    adata: sc.AnnData,
    cell_type_markers_dict: Dict[str, Sequence[str]],
    layer: Optional[str] = None,
    use_raw: bool = True,
    zscore: bool = True,  # kept for API compat; ignored
    min_markers: int = 1,
) -> pd.DataFrame:
    """
    Compute per-cell scores for each cell type using sc.tl.score_genes.
    Notes:
      - `zscore` is ignored (kept only for backward compatibility).
      - If fewer than `min_markers` genes are present, that cell type's score is NaN.
    """
    warnings.warn(
        "score_celltypes now uses sc.tl.score_genes; the `zscore` parameter is ignored.",
        RuntimeWarning,
        stacklevel=2,
    )

    # Determine gene universe for presence/missing checks
    if use_raw and getattr(adata, "raw", None) is not None and adata.raw is not None:
        gene_universe = set(map(str, adata.raw.var_names))
    else:
        gene_universe = set(map(str, adata.var_names))

    scores = {}
    missing_report = {}
    tmp_cols = []

    for ct, markers in cell_type_markers_dict.items():
        markers = [str(g) for g in markers]
        present = [g for g in markers if g in gene_universe]
        missing = [g for g in markers if g not in gene_universe]
        missing_report[ct] = missing

        if len(present) < min_markers:
            scores[ct] = np.full(adata.n_obs, np.nan, dtype=float)
            continue

        tmp_name = f"__ctscore__{ct}__{uuid4().hex}"
        sc.tl.score_genes(
            adata,
            gene_list=present,
            score_name=tmp_name,
            use_raw=use_raw,
            layer=layer,
        )
        scores[ct] = adata.obs[tmp_name].to_numpy()
        tmp_cols.append(tmp_name)

    # Assemble result DF and clean up temp columns
    df = pd.DataFrame(scores, index=adata.obs_names)
    if tmp_cols:
        adata.obs.drop(columns=[c for c in tmp_cols if c in adata.obs.columns], inplace=True, errors="ignore")

    # Merge missing-gene report into .uns
    adata.uns.setdefault("marker_missing_report", {})
    adata.uns["marker_missing_report"].update(missing_report)

    return df


def annotate_clusters_by_markers(
    adata: sc.AnnData,
    cluster_key: str,
    cell_type_markers_dict: Optional[Dict[str, Sequence[str]]] = None,
    layer: Optional[str] = None,
    use_raw: bool = True,
    beta: float = 2.0,
    min_markers: int = 1,
    zscore: bool = True,  # ignored by score_celltypes
    write_to_obs: bool = True,
    obs_prefix: Optional[str] = None,
    scores: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    If `scores` is provided, it must be a DataFrame with index == adata.obs_names
    and columns == cell types. Otherwise, scores are computed once here.
    """
    # Get per-cell scores S only if not provided
    if scores is None:
        if cell_type_markers_dict is None:
            raise ValueError("Provide `cell_type_markers_dict` when `scores` is None.")
        S = score_celltypes(
            adata,
            cell_type_markers_dict,
            layer=layer,
            use_raw=use_raw,
            zscore=zscore,  # ignored downstream
            min_markers=min_markers,
        )
    else:
        S = scores
        # Align/validate
        if not S.index.equals(adata.obs_names):
            S = S.reindex(adata.obs_names)

    cts = list(S.columns)
    clabs = adata.obs[cluster_key].astype(str)

    # Compute per-cluster medians for all cell types in one shot
    med_by_cluster = S.groupby(clabs).median()  # rows=clusters, cols=cell types

    # Per-cell best type once (used for consensus_frac)
    arr = S.to_numpy()
    all_nan = np.all(np.isnan(arr), axis=1)
    # nanargmax fails on all-NaN rows, so safe compute:
    argmax = np.zeros(arr.shape[0], dtype=int)
    argmax[~all_nan] = np.nanargmax(arr[~all_nan], axis=1)
    argmax[all_nan] = -1
    cell_best = np.array([cts[j] if j >= 0 else None for j in argmax], dtype=object)

    rows = []
    per_cluster_label, per_cluster_softmaxp, per_cluster_unc = {}, {}, {}

    for g, med in med_by_cluster.iterrows():
        order = med.sort_values(ascending=False)

        if order.size == 0 or np.all(~np.isfinite(order.values)):
            top1_ct, top1, top2 = None, np.nan, np.nan
            margin = np.nan
            consensus_frac = np.nan
            softmax_p = np.nan
            uncertainty = np.nan
        else:
            top1_ct = order.index[0]
            top1 = order.iloc[0]
            top2 = order.iloc[1] if order.size > 1 else np.nan
            margin = (top1 - top2) if np.isfinite(top1) and np.isfinite(top2) else np.nan

            # consensus among cell-wise winners within the cluster
            mask = (clabs == g).values
            winners = cell_best[mask]
            consensus_frac = np.mean(winners == top1_ct) if winners.size else np.nan

            # softmax over median scores within the cluster
            v = med.values.astype(float)
            if np.all(~np.isfinite(v)):
                softmax_p = np.nan
            else:
                v = v - np.nanmax(v)
                sm = np.exp(beta * np.nan_to_num(v, nan=-np.inf))
                sm[~np.isfinite(med.values)] = 0.0
                Z = sm.sum()
                idx_top1 = list(med.index).index(top1_ct) if top1_ct in med.index else None
                softmax_p = float(sm[idx_top1] / Z) if (Z > 0 and idx_top1 is not None) else np.nan

            uncertainty = 1.0 - (softmax_p * consensus_frac) if (np.isfinite(softmax_p) and np.isfinite(consensus_frac)) else np.nan

        rows.append({
            "cluster": g,
            "assigned_cell_type": top1_ct,
            "top1_score": top1,
            "top2_score": top2,
            "margin": margin,
            "consensus_frac": consensus_frac,
            "softmax_p": softmax_p,
            "uncertainty": uncertainty,
            **{f"median_{ct}": med.get(ct, np.nan) for ct in cts},
        })
        per_cluster_label[g] = top1_ct
        per_cluster_softmaxp[g] = softmax_p
        per_cluster_unc[g] = uncertainty

    cluster_df = pd.DataFrame(rows).set_index("cluster").sort_index()

    if write_to_obs:
        prefix = obs_prefix or cluster_key
        adata.obs[f"{prefix}_cell_type"] = clabs.map(per_cluster_label).astype("category")
        adata.obs[f"{prefix}_ctype_softmaxp"] = clabs.map(per_cluster_softmaxp).astype(float)
        adata.obs[f"{prefix}_ctype_uncertainty"] = clabs.map(per_cluster_unc).astype(float)

    return cluster_df


def sweep_leiden_and_annotate(
    adata: sc.AnnData,
    cell_type_markers_dict: Dict[str, Sequence[str]],
    resolutions: Sequence[float] = (0.2, 0.5, 1.0, 2.0, 5.0),
    neighbors_already_computed: bool = True,
    random_state: int = 0,
    layer: Optional[str] = None,
    use_raw: bool = True,
    zscore: bool = True,  # ignored by score_celltypes
    min_markers: int = 1,
    beta: float = 2.0,
    leiden_key_prefix: str = "leiden",
) -> Dict[str, Any]:
    """
    For a fixed graph (recommended), run Leiden at multiple `resolutions`,
    annotate clusters at each resolution, and compute agreement between
    per-cell assigned cell types across adjacent resolutions.

    Scores are computed ONCE and reused across all resolutions.
    """
    if not neighbors_already_computed and "connectivities" not in adata.obsp:
        raise ValueError("Please compute neighbors once (sc.pp.neighbors) before sweeping or ensure they exist.")

    # --- compute once ---
    S = score_celltypes(
        adata,
        cell_type_markers_dict,
        layer=layer,
        use_raw=use_raw,
        zscore=zscore,
        min_markers=min_markers,
    )

    res_list = [float(r) for r in resolutions]
    cluster_annotations = {}
    celltype_labels = {}

    for r in res_list:
        key = f"{leiden_key_prefix}_{r:.1f}"
        # We check if leiden key exists or compute it if missing?
        # The user snippet raised ValueError. I will modify to compute if missing for convenience, or match intent.
        # "sweep_leiden" implies it might run it.
        # The user's snippet raised: "Please compute leiden before sweeping." if key missing.
        # But earlier "multiscale_coarsening" computes it.
        # Let's compute it if missing, as it is helpful.
        if key not in adata.obs:
            print(f"Computing Leiden resolution {r:.1f}...")
            sc.tl.leiden(adata, resolution=r, key_added=key, random_state=random_state)

        cdf = annotate_clusters_by_markers(
            adata,
            cluster_key=key,
            cell_type_markers_dict=None,  # not needed since we pass scores
            layer=layer,
            use_raw=use_raw,
            zscore=zscore,
            min_markers=min_markers,
            beta=beta,
            write_to_obs=True,
            obs_prefix=key,
            scores=S,  # <- reuse
        )
        cluster_annotations[r] = cdf
        celltype_labels[r] = adata.obs[f"{key}_cell_type"].astype(str).copy()

    # ARI across resolutions for per-cell TYPE labels (not cluster IDs)
    R = np.zeros((len(res_list), len(res_list)))
    for i, ri in enumerate(res_list):
        for j, rj in enumerate(res_list):
            R[i, j] = adjusted_rand_score(celltype_labels[ri], celltype_labels[rj])

    all_pair_ARI = pd.DataFrame(R, index=res_list, columns=res_list)

    # Adjacent pairs
    rows = []
    for i in range(len(res_list) - 1):
        ri, rj = res_list[i], res_list[i + 1]
        ari = adjusted_rand_score(celltype_labels[ri], celltype_labels[rj])
        rows.append({"res_i": ri, "res_j": rj, "ARI": ari})
    adjacent_ARI = pd.DataFrame(rows)

    return {
        "cluster_annotations": cluster_annotations,
        "celltype_labels": celltype_labels,
        "adjacent_ARI": adjacent_ARI,
        "all_pair_ARI": all_pair_ARI,
    }
# ------------------------- Visualization & Export -------------------------

def plot_celltype_ari(all_pair_ARI: pd.DataFrame) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ARI heatmap using standard matplotlib.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(all_pair_ARI, cmap="viridis", vmin=0, vmax=1)

    # We want to show the ticks as the resolutions
    res_labels = [str(r) for r in all_pair_ARI.index]

    ax.set_xticks(np.arange(len(res_labels)))
    ax.set_yticks(np.arange(len(res_labels)))
    ax.set_xticklabels(res_labels)
    ax.set_yticklabels(res_labels)

    ax.set_xlabel("resolution")
    ax.set_ylabel("resolution")
    ax.set_title("ARI of per-cell TYPE labels across resolutions")

    # Add colorbar
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig, ax


def barplot_cluster_uncertainty(cluster_df: pd.DataFrame, top_k: int = 25) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar plot of cluster uncertainty.
    """
    df = cluster_df.sort_values("uncertainty").head(top_k)
    fig, ax = plt.subplots(figsize=(7, 3 + 0.25 * len(df)))
    ax.barh(df.index.astype(str), df["uncertainty"], color="#4444aa")
    ax.invert_yaxis()
    ax.set_xlabel("Uncertainty (lower is better)")
    ax.set_ylabel("Cluster")
    ax.set_title("Cluster annotation uncertainty")
    plt.tight_layout()
    return fig, ax


def export_cell_type_annotations(
    adata: sc.AnnData,
    annotation_key: str,
    output_path: Optional[str] = None,
    color_key: Optional[str] = None
) -> None:
    """
    Export cell type annotations to CSV.
    """
    from pathlib import Path

    # 1. Define your column names
    if color_key is None:
        color_key = annotation_key + '_colors'

    # 2. Create a temporary dataframe with the cell type
    # We explicitly copy the index to a column to ensure the Barcode is preserved
    if annotation_key not in adata.obs:
         raise ValueError(f"Annotation key '{annotation_key}' not found in adata.obs")

    df_export = adata.obs[[annotation_key]].copy()

    # 3. Map the colors from adata.uns to the individual cells
    # (Scanpy stores colors in .uns in the same order as the categories)
    if color_key in adata.uns:
        # Create a dictionary: { 'T-cell': '#1f77b4', 'B-cell': '#ff7f0e', ... }
        categories = adata.obs[annotation_key].cat.categories
        colors = adata.uns[color_key]
        if len(categories) == len(colors):
            category_map = dict(zip(categories, colors))
            # Map this to a new column in the dataframe
            df_export[color_key] = df_export[annotation_key].map(category_map)
        else:
            print(f"Warning: Number of colors in '{color_key}' does not match categories in '{annotation_key}'. Colors not exported.")
    else:
        print(f"Warning: '{color_key}' not found in adata.uns. Colors not exported.")

    # 4. Rename the index to 'Cell_ID' for clarity
    df_export.index.name = 'Cell_ID'

    # 5. Export to CSV
    if output_path is None:
        out_file = Path('cell_type_export.csv')
    else:
        out_file = Path(output_path)
        if not out_file.suffix.lower() == '.csv':
            out_file = out_file.with_suffix('.csv')

    print(f'Saving at {out_file.resolve()}')
    df_export.to_csv(out_file)

    # Verify the output
    print(df_export.head())
