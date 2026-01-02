# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scikit-image",
#     "numpy",
#     "matplotlib",
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
def _():
    import os, sys, subprocess, shutil
    from pathlib import Path

    OWNER, REPO, REF = "eigenP", "utils", "main"  # or a tag/branch/commit
    work = Path.cwd() / "_ext" / f"{REPO}-{REF}"
    src = work / "src"

    # clean old checkout
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)

    # shallow clone at the desired ref
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REF, f"https://github.com/{OWNER}/{REPO}.git", str(work)],
        check=True
    )

    # ensure src is importable
    p = str(src.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

    import eigenp_utils  # noqa: E402
    print("eigenp_utils imported from:", eigenp_utils.__file__)

    return


@app.cell
def _(mo):
    mo.md(
        """
    # Plotting Utils Demo

    Demonstrates `hist_imshow` and `color_coded_projection`.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import data
    from eigenp_utils.plotting_utils import (
        hist_imshow,
        color_coded_projection,
        set_plotting_style
    )

    # Try to set style (might fail if files missing in checkout, but good to call)
    set_plotting_style()

    return (
        color_coded_projection,
        data,
        hist_imshow,
        np,
        plt,
        set_plotting_style,
    )


@app.cell
def _(data):
    cells = data.cells3d()
    # Normalize for color projection (it expects floats often or handles it)
    # The function docs say "Normalize ... if frame_max > frame_min" internally.
    stack = cells[:, 1, :, :].astype(float)
    stack = (stack - stack.min()) / (stack.max() - stack.min())
    return cells, stack


@app.cell
def _(mo):
    cmap_dropdown = mo.ui.dropdown(
        options=['plasma', 'viridis', 'inferno', 'magma', 'cividis'],
        value='plasma',
        label="Colormap for Projection"
    )
    cmap_dropdown
    return (cmap_dropdown,)


@app.cell
def _(color_coded_projection, cmap_dropdown, stack):
    proj_img = color_coded_projection(stack, color_map=cmap_dropdown.value)
    return (proj_img,)


@app.cell
def _(plt, proj_img):
    fig_p, ax_p = plt.subplots(figsize=(5, 5))
    ax_p.imshow(proj_img)
    ax_p.set_title("Color Coded Projection (Time/Z)")
    ax_p.axis('off')
    fig_p
    return ax_p, fig_p


@app.cell
def _(mo):
    mo.md("## Histogram + Image (`hist_imshow`)")
    return


@app.cell
def _(hist_imshow, stack):
    # Just pass the 3D stack, it should slice middle automatically
    res = hist_imshow(stack, bins=100)
    res['fig']
    return (res,)


if __name__ == "__main__":
    app.run()
