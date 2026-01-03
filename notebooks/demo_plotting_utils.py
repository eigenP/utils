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
    # Plotting Utils Demo

    Demonstrates `hist_imshow` and `color_coded_projection`.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from eigenp_utils.io import download_file
    from eigenp_utils.plotting_utils import (
        hist_imshow,
        color_coded_projection,
        set_plotting_style
    )

    # Try to set style (might fail if files missing in checkout, but good to call)
    set_plotting_style()

    return (
        color_coded_projection,
        download_file,
        hist_imshow,
        imread,
        np,
        plt,
        set_plotting_style,
    )


@app.cell
def _(download_file):
    url_to_fetch = "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif"
    download_file(url_to_fetch, "./cells3d.tif")
    return


@app.cell
def _(imread):
    cells = imread("./cells3d.tif")
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
