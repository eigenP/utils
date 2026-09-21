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

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # PaCMAP Single Cell Embedding

    Demonstrating the `tl_pacmap` integration with Scanpy.
    """)
    return


@app.cell
def _():
    import scanpy as sc
    import matplotlib.pyplot as plt
    from eigenp_utils.single_cell import tl_pacmap

    # Load PBMC3k
    adata = sc.datasets.pbmc3k_processed()
    sc.pp.neighbors(adata)
    return adata, plt, sc, tl_pacmap


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
def _(init_dropdown, mo, run_btn):
    mo.vstack([init_dropdown, run_btn])
    return


@app.cell
def _(adata, init_dropdown, mo, plt, run_btn, sc, tl_pacmap):
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
