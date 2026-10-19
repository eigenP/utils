# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scanpy",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "pacmap",
#     "leidenalg",
#     "eigenp-utils[single-cell] @ git+https://github.com/eigenP/utils.git@main",
# ]
# ///
#
# Not runnable in the browser (marimo WASM / Pyodide): scanpy, pacmap and their
# numba dependency have no WebAssembly builds. Run it locally with
#   uvx marimo edit --sandbox <this file>
# or on molab (https://molab.marimo.io), which runs a regular Python kernel.

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    import sys

    mo.callout(
        mo.md(
            "**This notebook cannot run in the browser (WebAssembly).** scanpy, pacmap and their "
            "numba dependency have no WebAssembly builds. Run it locally with "
            "`uvx marimo edit --sandbox` on this file, or open it on "
            "[molab](https://molab.marimo.io), which runs a regular Python kernel."
        ),
        kind="warn",
    ) if sys.platform == "emscripten" else None
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Cell Annotation & EPS (Empirical Probability of Superiority)

    Demonstrating `annotate_clusters_by_markers`, multiresolution clustering (`sweep_leiden`), and multiresolution trees.
    """)
    return


@app.cell
def _():
    import scanpy as sc
    import matplotlib.pyplot as plt
    from eigenp_utils.single_cell import (
        tl_pacmap,
        annotate_clusters_by_markers,
        sweep_leiden_and_annotate,
        plot_clustering_tree,
    )

    # Load PBMC3k
    adata = sc.datasets.pbmc3k_processed()
    sc.pp.neighbors(adata)
    tl_pacmap(adata, init='pca')
    return (
        adata,
        annotate_clusters_by_markers,
        plot_clustering_tree,
        plt,
        sc,
        sweep_leiden_and_annotate,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Cell Annotation via EPS
    """)
    return


@app.cell
def _(adata, annotate_clusters_by_markers, mo, plt, sc):
    # Marker dictionary
    markers = {
        'B cells': ['CD79A', 'MS4A1'],
        'CD4 T cells': ['CD4', 'IL7R'],
        'CD8 T cells': ['CD8A', 'CD8B'],
        'NK cells': ['GNLY', 'NKG7'],
        'Monocytes': ['CD14', 'LYZ'],
        'Dendritic cells': ['FCER1A', 'CST3'],
        'Megakaryocytes': ['PPBP']
    }

    with mo.status.spinner("Annotating..."):
        # We will use 'louvain' as the base clusters
        # writes adata.obs['eps_cell_type'] (plus softmax probability / uncertainty)
        df_annot = annotate_clusters_by_markers(
            adata,
            cluster_key='louvain',
            cell_type_markers_dict=markers,
            obs_prefix='eps',
        )

    sc.pl.embedding(adata, basis='pacmap', color=['louvain', 'eps_cell_type'], show=False)
    _fig = plt.gcf()

    mo.vstack([
        mo.ui.table(df_annot),
        _fig
    ])
    return (markers,)


@app.cell
def _(mo):
    mo.md("""
    ## Sweep Leiden & Clustering Tree
    """)
    return


@app.cell
def _(mo):
    run_sweep_btn = mo.ui.run_button(label="Sweep Leiden & Plot Tree")
    run_sweep_btn
    return (run_sweep_btn,)


@app.cell
def _(
    adata,
    markers,
    mo,
    plot_clustering_tree,
    run_sweep_btn,
    sweep_leiden_and_annotate,
):
    if run_sweep_btn.value:
        with mo.status.spinner("Sweeping Leiden..."):
            resolutions = [0.1, 0.3, 0.5, 0.8, 1.2]
            # computes adata.obs['leiden_<res>'] and annotates each resolution
            sweep_results = sweep_leiden_and_annotate(adata, markers, resolutions=resolutions)

            _tree_fig = plot_clustering_tree(adata, [f"leiden_{r:.1f}" for r in resolutions])
            _res = mo.vstack([
                _tree_fig,
                mo.ui.table(sweep_results["adjacent_ARI"]),
            ])
    else:
        _res = mo.md("Click to sweep leiden and plot.")
    _res
    return


if __name__ == "__main__":
    app.run()
