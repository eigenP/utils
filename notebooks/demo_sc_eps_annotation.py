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

import marimo

# TODO: Plan B - Fix "Markdown cell should be dedented for better readability"
# Marimo check complains about markdown indentation inside mo.md().
# Need to use `marimo edit` to generate a markdown cell, inspect the exact string literal format
# (e.g., whether it uses `r"""`, `"""`, no leading spaces, or specific spacing),
# and apply that exact structure via an automated script.

__generated_with = "0.20.4"
app = marimo.App()

@app.cell
def _():
    import marimo as mo
    return mo,

@app.cell
def _(mo):
    mo.md(
        r"""
        # Cell Annotation & EPS (Empirical Probability of Superiority)

        Demonstrating `annotate_clusters_by_markers`, multiresolution clustering (`sweep_leiden`), and multiresolution trees.
        """
    )
    return

@app.cell
def _():
    import scanpy as sc
    import matplotlib.pyplot as plt
    from eigenp_utils.single_cell import (
        tl_pacmap,
        annotate_clusters_by_markers,
        sweep_leiden,
        multiscale_coarsening,
        plot_multiresolution_tree
    )

    # Load PBMC3k
    adata = sc.datasets.pbmc3k_processed()
    sc.pp.neighbors(adata)
    tl_pacmap(adata, init='pca')
    return adata, sc, plt, tl_pacmap, annotate_clusters_by_markers, sweep_leiden, multiscale_coarsening, plot_multiresolution_tree

@app.cell
def _(mo):
    mo.md("## Cell Annotation via EPS")
    return

@app.cell
def _(adata, annotate_clusters_by_markers, sc, plt, mo):
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
        df_annot = annotate_clusters_by_markers(
            adata,
            cluster_key='louvain',
            marker_dict=markers,
            new_annotation_key='eps_annotation'
        )

    sc.pl.embedding(adata, basis='pacmap', color=['louvain', 'eps_annotation'])
    _fig = plt.gcf()

    mo.vstack([
        mo.ui.table(df_annot),
        _fig
    ])
    return df_annot, markers, _fig

@app.cell
def _(mo):
    mo.md("## Sweep Leiden & Multiscale Coarsening Tree")
    return

@app.cell
def _(mo):
    run_sweep_btn = mo.ui.run_button(label="Sweep Leiden & Plot Tree")
    run_sweep_btn
    return run_sweep_btn,

@app.cell
def _(adata, run_sweep_btn, sweep_leiden, multiscale_coarsening, plot_multiresolution_tree, plt, mo):
    if run_sweep_btn.value:
        with mo.status.spinner("Sweeping Leiden..."):
            resolutions = [0.1, 0.3, 0.5, 0.8, 1.2]
            sweep_leiden(adata, resolutions=resolutions, key_prefix='leiden_')

            ms_results = multiscale_coarsening(
                adata,
                resolutions=resolutions,
                resolution_key_prefix='leiden_',
                return_output=True
            )

            fig = plot_multiresolution_tree(
                ms_results['mappings'],
                resolutions=resolutions
            )
            _res = fig
    else:
        _res = mo.md("Click to sweep leiden and plot.")
    _res
    return

if __name__ == "__main__":
    app.run()
