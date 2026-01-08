# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scikit-image",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "anywidget",
#     "traitlets",
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
    # Extended Depth of Focus (EDOF) Demo

    Demonstrates `best_focus_image` which fuses a Z-stack into a single in-focus image using a patch-based sharpness metric.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from eigenp_utils.io import download_file
    from eigenp_utils.extended_depth_of_focus import best_focus_image
    from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive
    return (
        best_focus_image,
        download_file,
        imread,
        np,
        plt,
        show_xyz_max_slice_interactive,
    )


@app.cell
def _(download_file):
    url_to_fetch = "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif"
    download_file(url_to_fetch, "./cells3d.tif")
    return


@app.cell
def _(imread):
    # Use cells3d, membrane channel (channel 0)
    # cells3d shape: (60, 2, 256, 256) -> (Z, C, Y, X)
    cells = imread("./cells3d.tif")
    membrane_stack = cells[:, 0, :, :]
    return cells, membrane_stack


@app.cell
def _(cells, show_xyz_max_slice_interactive):
    nuclei = cells[:, 1, :, :]
    membrane = cells[:, 0, :, :]
    show_xyz_max_slice_interactive([nuclei, membrane], colors=['lime', 'magenta'])
    return membrane, nuclei


@app.cell
def _(mo):
    patch_size_slider = mo.ui.slider(start=16, stop=128, step=16, value=64, label="Patch Size")
    patch_size_slider
    return (patch_size_slider,)


@app.cell
def _(best_focus_image, membrane_stack, patch_size_slider):
    edof_img, height_map = best_focus_image(
        membrane_stack,
        patch_size=patch_size_slider.value,
        return_heightmap=True
    )

    # Standard max projection for comparison
    max_proj = membrane_stack.max(axis=0)
    return edof_img, height_map, max_proj


@app.cell
def _(edof_img, height_map, max_proj, plt):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(max_proj, cmap='gray')
    axes[0].set_title("Standard Max Projection")
    axes[0].axis('off')

    axes[1].imshow(edof_img, cmap='gray')
    axes[1].set_title("EDOF (best_focus_image)")
    axes[1].axis('off')

    im2 = axes[2].imshow(height_map, cmap='viridis')
    axes[2].set_title("Height Map (Z-index)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], orientation='vertical')

    fig.tight_layout()
    fig
    return axes, fig, im2


if __name__ == "__main__":
    app.run()
