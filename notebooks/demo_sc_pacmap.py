# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scanpy",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "pacmap",
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
        # PaCMAP Single Cell Embedding

        Demonstrating the `tl_pacmap` integration with Scanpy.
        """
    )
    return

@app.cell
def _():
    import scanpy as sc
    import matplotlib.pyplot as plt
    from eigenp_utils.single_cell import tl_pacmap

    # Load PBMC3k
    adata = sc.datasets.pbmc3k_processed()
    sc.pp.neighbors(adata)
    return adata, sc, plt, tl_pacmap

@app.cell
def _(mo):
    init_dropdown = mo.ui.dropdown(
        options=['pca', 'random', 'paga'],
        value='pca',
        label='Initialization Method'
    )
    run_btn = mo.ui.run_button(label="Run PaCMAP")
    return init_dropdown, run_btn

@app.cell
def _(mo, init_dropdown, run_btn):
    mo.vstack([init_dropdown, run_btn])
    return

@app.cell
def _(adata, init_dropdown, run_btn, sc, tl_pacmap, plt, mo):
    if run_btn.value:
        if init_dropdown.value == 'paga':
            sc.tl.paga(adata, groups='louvain')

        with mo.status.spinner("Running PaCMAP..."):
            tl_pacmap(adata, init=init_dropdown.value)

        sc.pl.embedding(adata, basis='pacmap', color='louvain')
        _fig = plt.gcf()
        _res = _fig
    else:
        _res = mo.md("Click Run PaCMAP")

    _res
    return

if __name__ == "__main__":
    app.run()
