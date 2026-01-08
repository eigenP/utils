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
    # CLAHE Demo

    Demonstrating `_my_clahe_` from `eigenp_utils.clahe_equalize_adapthist`.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from eigenp_utils.io import download_file
    from eigenp_utils.clahe_equalize_adapthist import _my_clahe_
    from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive
    return (
        _my_clahe_,
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
    # Use cells3d nuclei channel (channel 1 is nuclei, 0 is membrane usually... let's check doc or just pick one)
    # cells3d shape: (Z, C, Y, X). C=0 membrane, C=1 nuclei.
    cells = imread("./cells3d.tif")
    # Pick a middle slice of the membrane channel
    membrane_slice = cells[30, 1, :, :]
    return cells, membrane_slice


@app.cell
def _(cells, show_xyz_max_slice_interactive):
    nuclei = cells[:, 1, :, :]
    membrane = cells[:, 0, :, :]
    show_xyz_max_slice_interactive([nuclei, membrane], colors=['lime', 'magenta'])
    return membrane, nuclei


@app.cell
def _(mo):
    clip_slider = mo.ui.slider(start=0.0, stop=0.1, step=0.005, value=0.01, label="Clip Limit")
    nbins_slider = mo.ui.slider(start=64, stop=512, step=64, value=256, label="Bins")
    kernel_size_slider = mo.ui.slider(start=8, stop=128, step=8, value=32, label="Kernel Size")

    mo.vstack([clip_slider, nbins_slider, kernel_size_slider])
    return clip_slider, kernel_size_slider, nbins_slider


@app.cell
def _(_my_clahe_, clip_slider, kernel_size_slider, membrane_slice, nbins_slider):
    clahe_img = _my_clahe_(
        membrane_slice,
        clip_limit=clip_slider.value,
        nbins=nbins_slider.value,
        kernel_size=kernel_size_slider.value
    )
    return (clahe_img,)


@app.cell
def _(clahe_img, membrane_slice, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(membrane_slice, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(clahe_img, cmap='gray')
    axes[1].set_title("CLAHE")
    axes[1].axis('off')
    fig.tight_layout()
    fig
    return axes, fig


if __name__ == "__main__":
    app.run()
