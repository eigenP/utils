# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scikit-image",
#     "numpy",
#     "matplotlib",
#     "eigenp-utils @ git+https://github.com/eigenP/utils.git@main ; sys_platform != 'emscripten'",
# ]
# ///

import marimo

__generated_with = "0.24.2"
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
        # micropip installs wheels only, from CORS-enabled hosts: CI publishes
        # the latest main build to the wasm-wheel branch (build-wasm-wheel.yml)
        GIT_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/wasm-wheel/eigenp_utils-0-py3-none-any.whl"
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

    def local_checkout():
        # A source checkout or editable install (code outside site-packages) is a
        # developer's working copy: never replace it with GitHub main. Setting
        # EIGENP_UTILS_NO_INSTALL also skips the install.
        import importlib.util, os
        if os.environ.get("EIGENP_UTILS_NO_INSTALL"):
            return True
        spec = importlib.util.find_spec("eigenp_utils")
        origin = (spec.origin or "") if spec is not None else ""
        return bool(origin) and not any(d in origin for d in ("site-packages", "dist-packages"))

    def install_github():
        if in_wasm():
            import micropip
            # also install imports of dependencies, which marimo cannot detect
            return micropip.install([GIT_URL, "scikit-image", "tifffile"])
        elif local_checkout():
            print("Using the local eigenp_utils checkout; not installing from GitHub.")
        else:
            install_local(GIT_URL)

    res = install_github()
    if res is not None:
        await res

    import eigenp_utils
    print("eigenp_utils imported from:", eigenp_utils.__file__)
    return


@app.cell
def _(mo):
    mo.md("""
    # Plotting Utils Demo

    Demonstrates `hist_imshow` and `color_coded_projection`.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from eigenp_utils.io import download_file_async
    from eigenp_utils.plotting_utils import (
        hist_imshow,
        color_coded_projection,
        set_plotting_style
    )

    # Try to set style (might fail if files missing in checkout, but good to call)
    set_plotting_style()
    return color_coded_projection, download_file_async, hist_imshow, imread, plt


@app.cell
async def _(download_file_async):
    # GitLab's API endpoint sends CORS headers (its /-/raw/ URL does not), so this
    # also works in the browser; same file as skimage.data.cells3d()
    url_to_fetch = "https://gitlab.com/api/v4/projects/scikit-image%2Fdata/repository/files/cells3d.tif/raw?ref=master"
    cells_path = await download_file_async(url_to_fetch, "./cells3d.tif")
    return (cells_path,)


@app.cell
def _(cells_path, imread):
    cells = imread(cells_path)
    # Normalize for color projection (it expects floats often or handles it)
    # The function docs say "Normalize ... if frame_max > frame_min" internally.
    stack = cells[:, 1, :, :].astype(float)
    stack = (stack - stack.min()) / (stack.max() - stack.min())
    return (stack,)


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
def _(cmap_dropdown, color_coded_projection, stack):
    proj_img = color_coded_projection(stack, color_map=cmap_dropdown.value)
    return (proj_img,)


@app.cell
def _(plt, proj_img):
    fig_p, ax_p = plt.subplots(figsize=(5, 5))
    ax_p.imshow(proj_img)
    ax_p.set_title("Color Coded Projection (Time/Z)")
    ax_p.axis('off')
    fig_p
    return


@app.cell
def _(mo):
    mo.md("""
    ## Histogram + Image (`hist_imshow`)
    """)
    return


@app.cell
def _(hist_imshow, stack):
    # Just pass the 3D stack, it should slice middle automatically
    _res = hist_imshow(stack, bins=100)
    _res['fig']
    return


if __name__ == "__main__":
    app.run()
