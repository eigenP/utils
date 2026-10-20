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
    # Intensity Rescaling Demo

    Showcasing:
    * `contrast_stretching`
    * `adjust_brightness_per_slice` (correcting Z-decay)
    * `normalize_image`
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from eigenp_utils.io import download_file_async
    from eigenp_utils.intensity_rescaling import (
        contrast_stretching,
        adjust_brightness_per_slice,
        normalize_image
    )
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

    return (
        adjust_brightness_per_slice,
        contrast_stretching,
        download_file_async,
        imread,
        np,
        plt,
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
def _(cells_path, imread, np):
    # Use cells3d membrane
    cells = imread(cells_path)
    stack = cells[:, 0, :, :]

    # Simulate Z-decay (bleaching)
    z_decay = np.linspace(1.0, 0.3, stack.shape[0])
    decayed_stack = stack * z_decay[:, None, None]
    return cells, decayed_stack


@app.cell
def _(cells, show_zyx_max_slice_interactive):
    nuclei = cells[:, 1, :, :]
    membrane = cells[:, 0, :, :]
    show_zyx_max_slice_interactive([nuclei, membrane], colors=['lime', 'magenta'])
    return


@app.cell
def _(mo):
    mo.md("## Contrast Stretching")
    p_min = mo.ui.slider(0, 50, value=2, label="P Min")
    p_max = mo.ui.slider(50, 100, value=98, label="P Max")
    mo.vstack([p_min, p_max])
    return p_max, p_min


@app.cell
def _(contrast_stretching, decayed_stack, p_max, p_min):
    # Just show middle slice
    mid_slice = decayed_stack[30]
    stretched = contrast_stretching(mid_slice, p_min=p_min.value, p_max=p_max.value)
    return mid_slice, stretched


@app.cell
def _(decayed_stack, show_zyx_max_slice_interactive):
    show_zyx_max_slice_interactive(decayed_stack)
    return


@app.cell
def _(mid_slice, plt, stretched):
    fig_cs, ax_cs = plt.subplots(1, 2, figsize=(10, 4))
    ax_cs[0].imshow(mid_slice, cmap='gray')
    ax_cs[0].set_title("Original (Decayed)")
    ax_cs[1].imshow(stretched, cmap='gray')
    ax_cs[1].set_title("Stretched")
    fig_cs.tight_layout()
    fig_cs
    return


@app.cell
def _(mo):
    mo.md("## Z-Decay Correction (`adjust_brightness_per_slice`)")
    fit_dropdown = mo.ui.dropdown(
        options={'Linear Ramp': None, 'Exponential Fit': 'exponential', 'Linear Fit': 'linear'},
        value='Exponential Fit',
        label="Fit Method"
    )

    correction_method = mo.ui.radio(
        options=['gamma', 'gain'],
        value='gamma',
        label="Correction Mode"
    )

    mo.vstack([fit_dropdown, correction_method])
    return correction_method, fit_dropdown


@app.cell
def _(
    adjust_brightness_per_slice,
    correction_method,
    decayed_stack,
    fit_dropdown,
):
    corrected_stack = adjust_brightness_per_slice(
        decayed_stack,
        gamma_fit_func=fit_dropdown.value,
        method=correction_method.value,
        final_gamma=0.5 # Used if None (manual ramp)
    )
    return (corrected_stack,)


@app.cell
def _(corrected_stack, show_zyx_max_slice_interactive):
    show_zyx_max_slice_interactive(corrected_stack)
    return


@app.cell
def _(corrected_stack, decayed_stack, np, plt):
    # Plot mean intensity per slice
    means_original = np.mean(decayed_stack, axis=(1, 2))
    means_corrected = np.mean(corrected_stack, axis=(1, 2))

    fig_z, ax_z = plt.subplots(figsize=(6, 4))
    ax_z.plot(means_original, label="Original (Decayed)")
    ax_z.plot(means_corrected, label="Corrected")
    ax_z.set_xlabel("Z Slice")
    ax_z.set_ylabel("Mean Intensity")
    ax_z.legend()
    ax_z.set_title("Intensity Profile Along Z")
    fig_z
    return


if __name__ == "__main__":
    app.run()
