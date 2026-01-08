# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.6",
#     "numpy==2.2.6",
#     "scikit-image==0.25.2",
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
    # Color-coded projection and CLAHE demo

    This notebook demonstrates how to use the
    `color_coded_projection` and `_my_clahe_` utilities provided in this
    repository. We load the sample `cells3d` dataset from scikit-image and
    showcase both functions:

    * **color_coded_projection** for creating a time/volume color projection
    * **_my_clahe_** for applying Contrast Limited Adaptive Histogram Equalization (CLAHE)

    Use the controls below to explore different color mappings for the
    projection and adjust the CLAHE clip limit to see its effect on the
    enhanced slice.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread

    from eigenp_utils.clahe_equalize_adapthist import _my_clahe_
    from eigenp_utils.color_coded_projection import color_coded_projection
    from eigenp_utils.io import download_file
    return (
        _my_clahe_,
        color_coded_projection,
        download_file,
        imread,
        np,
        plt,
    )


@app.cell
def _(mo):

    colormap_dropdown = mo.ui.dropdown(
        label="Projection colormap",
        options=[
            ("plasma"),
            ("viridis"),
            ("inferno"),
            ("magma"),
            ("cividis"),
        ],
        value="plasma",
    )

    colormap_dropdown
    return (colormap_dropdown,)


@app.cell
def _(mo):

    clahe_clip_slider = mo.ui.slider(
        label="CLAHE clip limit",
        start=0.01,
        stop=0.1,
        step=0.005,
        value=0.03,
    )

    clahe_clip_slider
    return (clahe_clip_slider,)


@app.cell
def _(download_file):
    url_to_fetch = "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif"
    download_file(url_to_fetch, "./cells3d.tif")
    return


@app.cell
def _(imread):
    cells = imread("./cells3d.tif")
    # Select the membrane channel (index 1)
    membrane_stack = cells[:, 1, :, :]
    return cells, membrane_stack


@app.cell
def _(membrane_stack):
    # Normalize the stack to the range [0, 1]
    stack_min = membrane_stack.min()
    stack_max = membrane_stack.max()
    normalized_stack = (membrane_stack - stack_min) / (stack_max - stack_min)
    return (normalized_stack,)


@app.cell
def _(color_coded_projection, colormap_dropdown, normalized_stack, np):
    projection = color_coded_projection(
        normalized_stack.astype(np.float32),
        color_map=colormap_dropdown.value,
    )
    return (projection,)


@app.cell
def _(colormap_dropdown, plt, projection):
    fig_proj, ax_proj = plt.subplots(figsize=(5, 5))
    ax_proj.imshow(projection)
    ax_proj.set_title(
        f"Color-coded projection of membrane channel (cmap: {colormap_dropdown.value})"
    )
    ax_proj.axis("off")
    fig_proj.tight_layout()
    fig_proj
    return


@app.cell
def _(membrane_stack):
    slice_index = 30
    original_slice = membrane_stack[slice_index]
    return original_slice, slice_index


@app.cell
def _(clahe_clip_slider, original_slice):
    clahe_slice = _my_clahe_(
        original_slice,
        clip_limit=float(clahe_clip_slider.value),
        nbins=256,
    )
    return (clahe_slice,)


@app.cell
def _(clahe_clip_slider, clahe_slice, original_slice, plt, slice_index):
    fig_clahe, axes_clahe = plt.subplots(1, 2, figsize=(10, 4))
    axes_clahe[0].imshow(original_slice, cmap="gray")
    axes_clahe[0].set_title(f"Original slice {slice_index}")
    axes_clahe[0].axis("off")

    axes_clahe[1].imshow(clahe_slice, cmap="gray")
    axes_clahe[1].set_title(
        f"CLAHE enhanced slice (clip_limit={float(clahe_clip_slider.value):.3f})"
    )
    axes_clahe[1].axis("off")

    fig_clahe.tight_layout()
    fig_clahe
    return


if __name__ == "__main__":
    app.run()
