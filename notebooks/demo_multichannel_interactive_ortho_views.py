# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scikit-image",
#     "anywidget",
#     "traitlets",
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
    mo.md(r"""
    # Multichannel Interactive Orthogonal Views

    Demonstrating the interactive slice and point annotator widgets on multichannel 3D datasets.
    """)
    return


@app.cell
async def _():
    import numpy as np
    from skimage.data import cells3d
    from eigenp_utils.tnia_plotting_anywidgets import (
        show_zyx_max_slice_interactive,
        show_zyx_max_slice_interactive_point_annotator
    )

    # Load the 3D cell data (Z, C, Y, X)
    try:
        cells = cells3d()
    except Exception:
        # e.g. no pooch, or in the browser (WASM). GitLab's API endpoint sends
        # CORS headers (its /-/raw/ URL does not); same file as cells3d().
        from eigenp_utils.io import download_file_async
        from skimage.io import imread
        url_to_fetch = "https://gitlab.com/api/v4/projects/scikit-image%2Fdata/repository/files/cells3d.tif/raw?ref=master"
        cells = imread(await download_file_async(url_to_fetch, "./cells3d.tif"))

    print(f"Loaded cells3d with shape: {cells.shape}")

    # Extract channels and cast to float to prevent clipping/overflow issues in blending
    nuclei = cells[:, 1, :, :].astype(float)
    membrane = cells[:, 0, :, :].astype(float)
    return (
        membrane,
        nuclei,
        show_zyx_max_slice_interactive,
        show_zyx_max_slice_interactive_point_annotator,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Interactive 3D Maximum Intensity Projection & Slicing

    You can use the sliders below to navigate the 3D volume along any axis. The `Thickness` sliders control the maximum intensity projection slab size, while `Position` navigates the slab center.
    """)
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
    mo.md(r"""
    ## Point Annotator Widget

    Toggle the "ANNOTATION" checkbox to start adding or deleting points. Points added are persistent and sync directly back to the `widget.points` list in Python.
    """)
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
