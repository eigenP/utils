# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scanpy",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "pacmap",
#     "esda",
#     "libpysal",
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
        # Moran's I & Archetype Analysis

        Demonstrating `morans_i_all_fast` and `find_expression_archetypes`.
        """
    )
    return

@app.cell
def _():
    import scanpy as sc
    import matplotlib.pyplot as plt
    from eigenp_utils.single_cell import (
        tl_pacmap,
        morans_i_all_fast,
        find_expression_archetypes,
        plot_archetype_assignments
    )

    # Load PBMC3k
    adata = sc.datasets.pbmc3k_processed()
    sc.pp.neighbors(adata)
    tl_pacmap(adata, init='pca')
    return adata, sc, plt, tl_pacmap, morans_i_all_fast, find_expression_archetypes, plot_archetype_assignments

@app.cell
def _(mo):
    run_moran_btn = mo.ui.run_button(label="Compute Moran's I")
    run_moran_btn
    return run_moran_btn,

@app.cell
def _(adata, run_moran_btn, morans_i_all_fast, mo):
    if run_moran_btn.value:
        with mo.status.spinner("Computing Moran's I..."):
            mi_results = morans_i_all_fast(adata)
        _res = mi_results.head(15)
    else:
        mi_results = None
        _res = mo.md("Click to compute Moran's I.")
    _res
    return mi_results,

@app.cell
def _(mo):
    num_arch_slider = mo.ui.slider(start=3, stop=10, step=1, value=5, label="Num Archetypes")
    run_arch_btn = mo.ui.run_button(label="Find Archetypes")
    return num_arch_slider, run_arch_btn

@app.cell
def _(mo, num_arch_slider, run_arch_btn):
    mo.vstack([num_arch_slider, run_arch_btn])
    return

@app.cell
def _(adata, mi_results, num_arch_slider, run_arch_btn, find_expression_archetypes, plot_archetype_assignments, plt, mo):
    if run_arch_btn.value:
        if mi_results is None:
            _res = mo.md("Please run Moran's I first!")
        else:
            top_genes = mi_results['gene'].head(100).tolist()
            with mo.status.spinner("Finding archetypes..."):
                arch_results = find_expression_archetypes(
                    adata,
                    gene_list=top_genes,
                    num_clusters=num_arch_slider.value
                )

            plot_archetype_assignments(adata, arch_results, use_rep='X_pacmap')
            _fig = plt.gcf()
            _res = _fig
    else:
        _res = mo.md("Click Find Archetypes.")

    _res
    return

if __name__ == "__main__":
    app.run()
