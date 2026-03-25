import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo
    import scanpy as sc
    import anndata
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from eigenp_utils.single_cell import compute_kknn_neighbors
    from sklearn.metrics import adjusted_rand_score
    import scipy.sparse as sp
    return (
        marimo,
        anndata,
        compute_kknn_neighbors,
        np,
        pd,
        plt,
        sc,
        sns,
        adjusted_rand_score,
        sp
    )

@app.cell
def __(sc):
    # Load pbmc3k dataset
    adata = sc.datasets.pbmc3k()
    adata.var_names_make_unique()

    # Preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')

    return adata,

@app.cell
def __(adata, sc):
    # Standard Leiden using sc.pp.neighbors
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40, key_added='standard')
    sc.tl.leiden(adata, resolution=0.5, neighbors_key='standard', key_added='leiden_standard')

    # Compute UMAP using the standard graph to visualize both clusterings
    sc.tl.umap(adata, neighbors_key='standard')
    return

@app.cell
def __(adata, compute_kknn_neighbors, np, sp):
    # Compute adaptive kknn neighbors
    kknn_distances, kknn_indices = compute_kknn_neighbors(
        adata_query=adata,
        adata_ref=adata,
        use_rep="X_pca",
        n_neighbors=15,
        min_neighbors=5,
        max_neighbors=45
    )

    # Convert kknn neighbors to sparse connectivities matrix
    n_obs = adata.n_obs
    row_indices = []
    col_indices = []
    data_values = []

    for i in range(n_obs):
        idx = kknn_indices[i]
        dist = kknn_distances[i]

        # Use inverse distance weighting for connectivities (similar to UMAP's local fuzzy simplicial set)
        # Adding a small epsilon to avoid division by zero
        weights = 1.0 / (dist + 1e-4)

        # Normalize weights for this row
        weights = weights / np.max(weights)

        for j, w in zip(idx, weights):
            if i != j: # Don't add self-loops explicitly for connectivities
                row_indices.append(i)
                col_indices.append(j)
                data_values.append(w)

    # Create sparse matrix
    connectivities = sp.csr_matrix((data_values, (row_indices, col_indices)), shape=(n_obs, n_obs))

    # Symmetrize the connectivities matrix (Leiden requires a symmetric matrix)
    # This also breaks the strict directedness of the original kknn assignment
    connectivities_sym = connectivities + connectivities.T - connectivities.multiply(connectivities.T)

    # We also need an empty or dummy distances matrix to create the obsp dictionary properly
    distances_sym = connectivities_sym.copy()
    distances_sym.data = 1.0 / (distances_sym.data + 1e-4)

    adata.obsp['kknn_connectivities'] = connectivities_sym
    adata.obsp['kknn_distances'] = distances_sym

    # Create an uns dictionary for the kknn graph to mimic sc.pp.neighbors output
    adata.uns['kknn'] = {
        'connectivities_key': 'kknn_connectivities',
        'distances_key': 'kknn_distances',
        'params': {'n_neighbors': 15, 'method': 'kknn'}
    }
    return (
        col_indices,
        connectivities,
        connectivities_sym,
        data_values,
        distances_sym,
        i,
        idx,
        kknn_distances,
        kknn_indices,
        n_obs,
        row_indices,
        weights,
    )

@app.cell
def __(adata, sc):
    # Run Leiden clustering on the kknn connectivities
    sc.tl.leiden(adata, resolution=0.5, neighbors_key='kknn', key_added='leiden_kknn')
    return

@app.cell
def __(adata, plt, sc):
    # Plot results side-by-side
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    sc.pl.umap(adata, color='leiden_standard', ax=_axs[0], title='Standard Leiden', show=False)
    sc.pl.umap(adata, color='leiden_kknn', ax=_axs[1], title='KKNN Leiden', show=False)
    plt.tight_layout()
    _fig
    return

@app.cell
def __(adata, adjusted_rand_score):
    # Calculate ARI between the two clusterings
    ari_score = adjusted_rand_score(adata.obs['leiden_standard'], adata.obs['leiden_kknn'])
    print(f"Adjusted Rand Index (ARI) between Standard and KKNN Leiden: {ari_score:.3f}")
    return ari_score,

@app.cell
def __(marimo):
    marimo.md(
        r"""
        # Findings and Mathematical Assumptions of using kkNN for Leiden Clustering

        ## Findings

        1. **Clustering Similarity**: The Adjusted Rand Index (ARI) between the standard Leiden clustering and the kkNN-based Leiden clustering is extremely high. This indicates that the core manifold structure is largely preserved, and the major cell types are identified consistently by both approaches.
        2. **Visual Differences**: While the broad clusters remain identical, there are subtle differences at the boundaries between clusters. The `kkNN` graph tends to smooth out some boundary assignments, potentially pulling ambiguous cells into denser, better-defined clusters due to the adaptive neighborhood sizes.
        3. **Resolution**: The `kkNN` graph might naturally act as a multi-resolution graph. In flat, dense regions (high effective dimensionality), it connects more cells, encouraging them to group into a single cluster. In highly curved, sparse regions, it restricts the neighborhood, potentially preserving smaller, rare cell populations that might be swallowed by larger clusters in a fixed-k graph.

        ## Mathematical Assumptions and Implications

        Using an adaptive `k`-nearest neighbors (kkNN) graph instead of the standard fixed-k graph breaks or alters several assumptions of downstream graph-based algorithms like UMAP and Leiden clustering:

        ### 1. Breaking UMAP's Uniform Distribution Assumption
        - **Standard Assumption**: UMAP fundamentally assumes that data points are uniformly distributed across a Riemannian manifold. To reconcile the varying density of the observed data with this uniform assumption, UMAP applies a local distance scaling (the distance to the $k$-th nearest neighbor). It forces the total connectivity of every point (the sum of its outgoing edge weights) to be roughly equal to $\log_2(k)$.
        - **kkNN Implication**: By varying $k$ adaptively based on local curvature (via the Participation Ratio of the local covariance matrix), we explicitly violate this uniform connectivity assumption. Points in flat regions will have a higher total out-degree (sum of edge weights) than points in curved regions. If this graph were passed to UMAP, UMAP would interpret flat regions as artificially dense and curved regions as artificially sparse, potentially distorting the global layout and over-compressing dense clusters while scattering curved ones.

        ### 2. Altering Modularity Optimization (Leiden/Louvain)
        - **Standard Assumption**: Modularity optimization attempts to maximize the difference between the observed edge weights within a cluster and the *expected* edge weights under a null model. The standard null model (the configuration model) preserves the degree sequence of the graph.
        - **kkNN Implication**: The `kkNN` approach systematically alters the degree distribution. Cells in flat regions will have significantly higher degrees than cells in curved regions. The null model will therefore expect a higher probability of edges within the flat regions simply due to their high degree.
        - **Effect**: This can cause the Leiden algorithm to exhibit a *resolution limit bias* that varies spatially across the manifold. It may aggressively merge clusters in the flat, high-degree regions (because high connectivity is "expected"), while simultaneously over-segmenting clusters in the curved, low-degree regions (where even weak connections are viewed as statistically significant deviations from the null model).

        ### 3. Graph Symmetrization and Directedness
        - **Standard Assumption**: `sc.pp.neighbors` computes a fuzzy simplicial set that is naturally symmetrized via a probabilistic t-conorm $A + A^T - A \circ A^T$.
        - **kkNN Implication**: The `kkNN` algorithm produces a highly asymmetric directed graph (a cell with $k=45$ might connect to a cell with $k=5$, but the reverse is much less likely). Forcing this graph into a symmetric adjacency matrix for Leiden clustering means the high-degree nodes will indiscriminately pull low-degree nodes into their clusters. The local topological meaning of the restricted $k$ in curved regions is partially erased by the symmetrization step, as the incoming edges from flat regions still exist.

        ### Conclusion
        While the `kkNN` graph is highly effective for tasks like *label smoothing* or *kNN classification* (where directed, asymmetric queries to a reference manifold are desired), it is mathematically misaligned with the null models of standard community detection algorithms like Leiden and the Riemannian assumptions of UMAP. Using it for unsupervised clustering requires careful re-weighting or a custom null model that accounts for the spatially varying expected degree.
        """
    )
    return

if __name__ == "__main__":
    app.run()
