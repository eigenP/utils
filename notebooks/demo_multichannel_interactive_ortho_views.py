# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scikit-image",
#     "anywidget",
#     "traitlets",
#     "eigenp-utils @ git+https://github.com/eigenP/utils.git@main",
# ]
# ///

import marimo

# TODO: Plan B - Fix "Markdown cell should be dedented for better readability"
# Marimo check complains about markdown indentation inside mo.md().
# Need to use `marimo edit` to generate a markdown cell, inspect the exact string literal format
# (e.g., whether it uses `r"""`, `"""`, no leading spaces, or specific spacing),
# and apply that exact structure via an automated script.

__generated_with = "0.20.4"
app = marimo.App(auto_download=["html"])

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell(hide_code=True)
async def _(mo):
    mo.md(
        r"""
        ## Setup
        Installing the package from GitHub...
        """
    )

    import sys

    def in_wasm():
        return sys.platform in ("emscripten", "wasi")

    OWNER, REPO, REF = "eigenP", "utils", "main"
    if in_wasm():
        GIT_URL = f"eigenp-utils @ https://github.com/{OWNER}/{REPO}/archive/{REF}.zip"
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
        r"""
        # Multichannel Interactive Orthogonal Views

        Demonstrating the interactive slice and point annotator widgets on multichannel 3D datasets.
        """
    )
    return

@app.cell
def _():
    import numpy as np
    from skimage.data import cells3d
    from eigenp_utils.tnia_plotting_anywidgets import (
        show_zyx_max_slice_interactive,
        show_zyx_max_slice_interactive_point_annotator
    )

    # Load the 3D cell data (Z, C, Y, X)
    cells = cells3d()
    print(f"Loaded cells3d with shape: {cells.shape}")

    # Extract channels and cast to float to prevent clipping/overflow issues in blending
    nuclei = cells[:, 1, :, :].astype(float)
    membrane = cells[:, 0, :, :].astype(float)
    return cells, membrane, nuclei, np, show_zyx_max_slice_interactive, show_zyx_max_slice_interactive_point_annotator

@app.cell
def _(mo):
    mo.md(
        r"""
        ## Interactive 3D Maximum Intensity Projection & Slicing

        You can use the sliders below to navigate the 3D volume along any axis. The `Thickness` sliders control the maximum intensity projection slab size, while `Position` navigates the slab center.
        """
    )
    return

@app.cell
def _(membrane, nuclei, show_zyx_max_slice_interactive):
    # Standard Viewer
    viewer = show_zyx_max_slice_interactive(
        [nuclei, membrane],
        colors=['viridis', 'magma'],
        x_t=3,
        y_t=3,
        z_t=nuclei.shape[0] // 2 - 1,
        show_crosshair=True
    )
    return (viewer,)

@app.cell
def _(viewer):
    viewer
    return

@app.cell
def _(mo):
    mo.md(
        r"""
        ## Point Annotator Widget

        Toggle the "ANNOTATION" checkbox to start adding or deleting points. Points added are persistent and sync directly back to the `widget.points` list in Python.
        """
    )
    return

@app.cell
def _(membrane, nuclei, show_zyx_max_slice_interactive_point_annotator):
    # Annotator Viewer
    annotator = show_zyx_max_slice_interactive_point_annotator(
        [nuclei, membrane],
        colors=['green', 'magenta'],
        x_t=5,
        y_t=5,
        z_t=nuclei.shape[0] // 2 - 1,
        show_crosshair=True,
        point_size_scale=0.015  # Adjust marker size
    )
    return (annotator,)

@app.cell
def _(annotator):
    annotator
    return

if __name__ == "__main__":
    app.run()
