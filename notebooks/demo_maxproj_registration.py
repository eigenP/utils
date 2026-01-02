# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scikit-image",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "pandas",
#     "tqdm",
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
            subprocess.check_call(["uv", "pip", "install", "--system", url])
        else:
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
    # Registration / Drift Correction Demo

    Demonstrates `estimate_drift_2D` and `apply_drift_correction_2D` on a synthetically drifted stack.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from scipy.ndimage import shift
    from eigenp_utils.io import download_file
    from eigenp_utils.maxproj_registration import (
        estimate_drift_2D,
        apply_drift_correction_2D
    )
    return (
        apply_drift_correction_2D,
        download_file,
        estimate_drift_2D,
        imread,
        np,
        plt,
        shift,
    )


@app.cell
def _(download_file):
    url_to_fetch = "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif"
    download_file(url_to_fetch, "./cells3d.tif")
    return


@app.cell
def _(imread, np, shift):
    # Use cells3d
    cells = imread("./cells3d.tif")
    # Take a smaller subset of frames/slices to speed up
    # Treat Z as Time for this demo
    original_stack = cells[:20, 1, :, :] # Nuclei channel

    # Introduce synthetic drift
    drifted_stack = np.zeros_like(original_stack)
    true_drifts = []

    current_dx, current_dy = 0, 0
    for t in range(original_stack.shape[0]):
        # Random walk drift
        if t > 0:
            current_dx += np.random.randint(-2, 3)
            current_dy += np.random.randint(-2, 3)

        drifted_stack[t] = shift(original_stack[t], shift=(current_dy, current_dx), mode='constant')
        true_drifts.append((current_dy, current_dx))

    return (
        cells,
        current_dx,
        current_dy,
        drifted_stack,
        original_stack,
        t,
        true_drifts,
    )


@app.cell
def _(apply_drift_correction_2D, drifted_stack):
    corrected_stack, drift_table = apply_drift_correction_2D(drifted_stack)
    return corrected_stack, drift_table


@app.cell
def _(drift_table, mo):
    mo.ui.table(drift_table)
    return


@app.cell
def _(corrected_stack, drifted_stack, plt):
    # Visualize Max Projections
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(drifted_stack.max(axis=0), cmap='inferno')
    axes[0].set_title("Drifted Stack (Max Proj)")
    axes[0].axis('off')

    axes[1].imshow(corrected_stack.max(axis=0), cmap='inferno')
    axes[1].set_title("Corrected Stack (Max Proj)")
    axes[1].axis('off')

    fig.tight_layout()
    fig
    return axes, fig


if __name__ == "__main__":
    app.run()
