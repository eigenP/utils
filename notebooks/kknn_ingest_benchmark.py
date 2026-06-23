import marimo

__generated_with = "0.2.13"
app = marimo.App(width="medium")

@app.cell
def __():
    import marimo as mo
    return mo,

@app.cell
def __(mo):
    return mo.md("""
# Benchmark `lle_reg_lambda` in `kknn_ingest`

This notebook investigates the effect of `lle_reg_lambda` when using the Locally Linear Embedding (`barycenter='lle'`) in the `kknn_ingest` function.

We generate a synthetic reference dataset and a noisy "query" dataset. Then we map the query to the reference using various values of `lle_reg_lambda` to visualize how it controls the "snapping" (sparse assignment) versus smoothing (uniform average).
""")

@app.cell
def __():
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import matplotlib.pyplot as plt
    from scipy.stats import entropy
    import warnings
    warnings.filterwarnings('ignore')

    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

    from eigenp_utils.single_cell import kknn_ingest, tl_pacmap, compute_kknn_neighbors
    return (
        np, pd, sc, plt, entropy, warnings, sys, os, kknn_ingest, tl_pacmap, compute_kknn_neighbors
    )

@app.cell
def __(np, pd, sc, tl_pacmap):
    # 1. Generate Synthetic Data
    np.random.seed(42)

    # We create a 2D synthetic manifold that curves, embedded in a 20D space
    t = np.linspace(0, 4 * np.pi, 1000)
    x = t * np.cos(t)
    y = t * np.sin(t)

    # Base 2D structure
    ref_base = np.column_stack([x, y])

    # Project to 20D with some random mixing and noise
    W = np.random.randn(2, 20)
    X_ref = ref_base @ W + np.random.randn(1000, 20) * 0.5

    adata_ref = sc.AnnData(X_ref)
    adata_ref.var_names = [f"Gene_{i}" for i in range(20)]
    adata_ref.obs['cell_type'] = pd.Categorical((t // (np.pi)).astype(int).astype(str))

    # Add a continuous variable to trace along the manifold
    adata_ref.obs['pseudotime'] = t

    # Compute PCA for reference
    sc.tl.pca(adata_ref, n_comps=10)

    # Compute PaCMAP on reference
    tl_pacmap(adata_ref, n_neighbors=15, random_state=42)

    # 2. Generate Query Data (Subset + Noise)
    # Take a subset of reference and add noise
    query_idx = np.random.choice(1000, 300, replace=False)
    X_query = X_ref[query_idx] + np.random.randn(300, 20) * 2.0  # High noise to make them "float" off manifold

    adata_query_orig = sc.AnnData(X_query)
    adata_query_orig.var_names = adata_ref.var_names
    adata_query_orig.obs['true_pseudotime'] = t[query_idx]

    # Provide PCA representation for the query
    # (kknn_ingest usually does this internally if recompute_ref_PCA is True, but we'll do it manually to ensure consistency)
    adata_query_orig.obsm['X_pca'] = (X_query - X_query.mean(axis=0)) @ adata_ref.varm['PCs'][:, :10]

    adata_query_orig
    return (t, x, y, ref_base, W, X_ref, adata_ref, query_idx, X_query, adata_query_orig)

@app.cell
def __(np, entropy, sc, kknn_ingest, adata_ref, adata_query_orig, plt):
    # 3. Sweep over lle_reg_lambda
    lambdas_to_test = [1e-8, 1e-4, 1e-1, 1e2, 1e8]

    results = {}

    fig, axes = plt.subplots(1, len(lambdas_to_test), figsize=(20, 4))

    for _i, _lam in enumerate(lambdas_to_test):
        # We need a fresh copy of the query for each ingest
        _adata_query = adata_query_orig.copy()

        # Run kknn_ingest
        # We will transfer the PaCMAP coordinates
        kknn_ingest(
            adata_query=_adata_query,
            adata_ref=adata_ref,
            obsm_keys=['X_pacmap'],
            use_rep='X_pca',
            barycenter='lle',
            lle_reg_lambda=_lam,
            n_neighbors=15, # fixed n_neighbors for simplicity here
            recompute_ref_PCA=False # Already aligned
        )

        results[_lam] = _adata_query.obsm['X_pacmap_kknn']

        # Plot
        _ax = axes[_i]

        # Plot reference in grey
        _ax.scatter(adata_ref.obsm['X_pacmap'][:, 0], adata_ref.obsm['X_pacmap'][:, 1], c='lightgray', s=10, label='Ref', alpha=0.5)

        # Plot mapped query in color
        _ax.scatter(_adata_query.obsm['X_pacmap_kknn'][:, 0], _adata_query.obsm['X_pacmap_kknn'][:, 1], c=_adata_query.obs['true_pseudotime'], cmap='viridis', s=20, label='Query Mapped')

        _ax.set_title(rf"$\lambda$ = {_lam:1.0e}")
        _ax.axis('off')

    plt.tight_layout()
    fig
    return lambdas_to_test, results, fig, axes

@app.cell
def __(np, entropy, sc, compute_kknn_neighbors, adata_ref, adata_query_orig, lambdas_to_test, plt):
    # 4. Analyze Weight Entropy Manually
    # Since `kknn_ingest` hides the weights internally, we will extract the neighbors and compute the LLE weights manually to observe their entropy.

    def compute_lle_weights_and_entropy(X_q_coords, X_ref_coords, indices, lam):
        entropies = []
        for i in range(len(X_q_coords)):
            idx = indices[i]
            x_q = X_q_coords[i]
            X_neigh = X_ref_coords[idx]

            Z = X_neigh - x_q
            G = Z @ Z.T
            trace = np.trace(G)
            if trace > 0:
                reg = lam * trace / len(idx)
            else:
                reg = lam
            G_reg = G + reg * np.eye(len(idx))

            try:
                w = np.linalg.solve(G_reg, np.ones(len(idx)))
                w = np.maximum(w, 0)
                w_sum = np.sum(w)
                if w_sum > 0:
                    weights = w / w_sum
                else:
                    weights = np.ones(len(idx)) / len(idx)
            except np.linalg.LinAlgError:
                weights = np.ones(len(idx)) / len(idx)

            # Compute entropy (ignoring zeros)
            w_pos = weights[weights > 0]
            ent = -np.sum(w_pos * np.log(w_pos))
            entropies.append(ent)

        return np.array(entropies)

    # Get neighbors using the exact same function kknn_ingest uses
    _, _pruned_indices = compute_kknn_neighbors(
        adata_query_orig, adata_ref, use_rep='X_pca', n_neighbors=15
    )

    _X_q_coords = adata_query_orig.obsm['X_pca']
    _X_ref_coords = adata_ref.obsm['X_pca']

    entropy_results = {}
    for _lam in lambdas_to_test:
        entropy_results[_lam] = compute_lle_weights_and_entropy(_X_q_coords, _X_ref_coords, _pruned_indices, _lam)

    # Plot Entropy Distributions
    fig_ent, ax_ent = plt.subplots(figsize=(8, 5))

    data_to_plot = [entropy_results[_lam] for _lam in lambdas_to_test]
    ax_ent.violinplot(data_to_plot, showmeans=True, showmedians=False)

    ax_ent.set_xticks(np.arange(1, len(lambdas_to_test) + 1))
    ax_ent.set_xticklabels([f"{_lam:1.0e}" for _lam in lambdas_to_test])
    ax_ent.set_xlabel("lle_reg_lambda")
    ax_ent.set_ylabel("Weight Entropy (lower = sparser/snapping)")
    ax_ent.set_title("Distribution of LLE Weight Entropy vs. Lambda")

    # Max possible entropy for uniform distribution of ~15 neighbors: log(15) ~ 2.7
    max_ent = np.mean([np.log(len(idx)) for idx in _pruned_indices])
    ax_ent.axhline(max_ent, color='red', linestyle='--', label='Uniform Avg Entropy')
    ax_ent.legend()

    fig_ent
    return compute_lle_weights_and_entropy, entropy_results, fig_ent, ax_ent, data_to_plot, max_ent

if __name__ == "__main__":
    app.run()
