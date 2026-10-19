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
    # Registration / Drift Correction Demo

    Demonstrates `estimate_drift` and `apply_drift_correction` on a synthetically drifted stack.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from scipy.ndimage import shift
    from eigenp_utils.io import download_file_async
    from eigenp_utils.maxproj_registration import (
        estimate_drift,
        apply_drift_correction
    )
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

    return (
        apply_drift_correction,
        download_file_async,
        imread,
        np,
        plt,
        shift,
        show_zyx_max_slice_interactive,
    )


@app.cell
async def _(download_file_async):
    # GitLab's API endpoint sends CORS headers (its /-/raw/ URL does not), so this
    # also works in the browser; same file as skimage.data.cells3d()
    url_to_fetch = "https://gitlab.com/api/v4/projects/scikit-image%2Fdata/repository/files/cells3d.tif/raw?ref=master"
    cells_path = await download_file_async(url_to_fetch, "./cells3d.tif")
    return (cells_path,)


@app.cell
def _(cells_path, imread, np, shift):
    # Use cells3d
    cells = imread(cells_path)
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
    return cells, drifted_stack


@app.cell
def _(cells, show_zyx_max_slice_interactive):
    nuclei = cells[:, 1, :, :]
    membrane = cells[:, 0, :, :]
    show_zyx_max_slice_interactive([nuclei, membrane], colors=['lime', 'magenta'])
    return


@app.cell
def _(drifted_stack, show_zyx_max_slice_interactive):
    show_zyx_max_slice_interactive(drifted_stack)
    return


@app.cell
def _(apply_drift_correction, drifted_stack):
    corrected_stack, drift_table = apply_drift_correction(drifted_stack)
    return corrected_stack, drift_table


@app.cell
def _(drift_table, mo):
    mo.ui.table(drift_table)
    return


@app.cell
def _(corrected_stack, show_zyx_max_slice_interactive):
    show_zyx_max_slice_interactive(corrected_stack)
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
    return


if __name__ == "__main__":
    app.run()
