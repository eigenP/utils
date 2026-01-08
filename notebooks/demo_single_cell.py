# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scanpy",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
#     "esda",
#     "libpysal",
#     "triku",
#     "pacmap",
#     "leidenalg",
#     "eigenp-utils @ git+https://github.com/eigenP/utils.git@main",
# ]
# ///

import marimo

__generated_with = "0.16.4"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
async def _(mo):
    mo.md(
        """
        ## Setup
        Installing the package from GitHub...
        """
    )

    import sys

    def in_wasm():
        return sys.platform in ("emscripten", "wasi")

    OWNER, REPO, REF = "eigenP", "utils", "main"
    if in_wasm():
        GIT_URL = f"https://github.com/{OWNER}/{REPO}/archive/{REF}.zip"
    else:
        GIT_URL = f"git+https://github.com/{OWNER}/{REPO}.git@{REF}"

    def install_local(url):
        import subprocess, sys, shutil

        if shutil.which("uv"):
            try:
                subprocess.check_call([
                    "uv", "pip", "install",
                    "--python", sys.executable,
                    url,
                ])
                return
            except subprocess.CalledProcessError:
                pass  # fall back to pip

        subprocess.check_call([sys.executable, "-m", "pip", "install", url])

    def install_github():
        if in_wasm():
            import micropip
            return micropip.install(GIT_URL)
        else:
            install_local(GIT_URL)

    res = install_github()
    if res is not None:
        await res

    import eigenp_utils
    print("eigenp_utils imported from:", eigenp_utils.__file__)

    return (
        GIT_URL,
        OWNER,
        REF,
        REPO,
        eigenp_utils,
        in_wasm,
        install_github,
        install_local,
        res,
        sys,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Single Cell Utils Demo

    Demonstrating tools for single-cell analysis:
    * `morans_i_all_fast`
    * `find_expression_archetypes`
    * `multiscale_coarsening`
    * `tl_pacmap` (PaCMAP embedding)
    """
    )
    return


@app.cell
def _():
    import scanpy as sc
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from eigenp_utils.single_cell import (
        morans_i_all_fast,
        find_expression_archetypes,
        plot_archetype_assignments,
        multiscale_coarsening,
        tl_pacmap,
        ensure_neighbors
    )
    return (
        ensure_neighbors,
        find_expression_archetypes,
        morans_i_all_fast,
        multiscale_coarsening,
        np,
        pd,
        plot_archetype_assignments,
        plt,
        sc,
        tl_pacmap,
    )


@app.cell
def _(mo, sc):
    # Load dataset
    adata = sc.datasets.pbmc3k_processed()
    # Ensure some preprocessing if needed (pbmc3k_processed is already log1p/scaled usually)
    # But let's re-run neighbors just in case we need specific settings
    mo.md(f"Loaded PBMC3k: {adata.shape}")
    return adata,


@app.cell
def _(adata, ensure_neighbors, mo):
    ensure_neighbors(adata)
    mo.md("Neighbors computed/verified.")
    return


@app.cell
def _(mo):
    mo.md("## Moran's I (Spatial Autocorrelation on Graph)")
    return


@app.cell
def _(adata, mo, morans_i_all_fast):
    # Compute Moran's I for all genes (using graph connectivities)
    mi_results = morans_i_all_fast(adata)

    top_genes_table = mo.ui.table(mi_results.head(10))
    top_genes_table
    return mi_results, top_genes_table


@app.cell
def _(mo):
    mo.md("## Gene Archetypes")
    num_clusters_slider = mo.ui.slider(start=3, stop=10, value=5, label="Num Archetypes")
    num_clusters_slider
    return (num_clusters_slider,)


@app.cell
def _(
    adata,
    find_expression_archetypes,
    mi_results,
    num_clusters_slider,
    plot_archetype_assignments,
    plt,
):
    # Use top 100 genes from Moran's I for archetypes
    top_genes = mi_results['gene'].head(100).tolist()

    arch_results = find_expression_archetypes(
        adata,
        gene_list=top_genes,
        num_clusters=num_clusters_slider.value
    )

    # Plot assignments
    # Using existing UMAP in adata
    plot_archetype_assignments(adata, arch_results, use_rep='X_umap')
    fig = plt.gcf()
    fig
    return arch_results, fig, top_genes


@app.cell
def _(mo):
    mo.md("## Multiscale Coarsening (Leiden Hierarchy)")
    return


@app.cell
def _(adata, mo, multiscale_coarsening):
    ms_results = multiscale_coarsening(
        adata,
        resolutions=[0.5, 1.0], # Keep it small for demo speed
        return_output=True
    )

    # Show consistency table
    consistency_df = ms_results['consistency']
    if consistency_df.empty:
        msg = "No lineage inconsistencies found."
    else:
        msg = "Inconsistencies found:"

    mo.vstack([mo.md(msg), mo.ui.table(consistency_df)])
    return consistency_df, msg, ms_results


@app.cell
def _(mo):
    mo.md("## PaCMAP Embedding")
    run_pacmap_btn = mo.ui.run_button(label="Run PaCMAP")
    run_pacmap_btn
    return (run_pacmap_btn,)


@app.cell
def _(adata, plt, run_pacmap_btn, sc, tl_pacmap):
    if run_pacmap_btn.value:
        tl_pacmap(adata)
        sc.pl.embedding(adata, basis='pacmap', color='louvain')
        fig_pacmap = plt.gcf()
        fig_pacmap
    else:
        fig_pacmap = None
    return (fig_pacmap,)


if __name__ == "__main__":
    app.run()
