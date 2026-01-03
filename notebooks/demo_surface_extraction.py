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
    # Surface Extraction Demo

    Demonstrating `extract_surface` to find the "top" surface of a volumetric object (e.g., cell membrane).
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from eigenp_utils.io import download_file
    from eigenp_utils.surface_extraction import extract_surface
    from eigenp_utils.tnia_plotting_anywidgets import show_xyz_max_slice_interactive
    return (
        extract_surface,
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
    # Use cells3d membrane channel
    cells = imread("./cells3d.tif")
    membrane = cells[:, 0, :, :] # ZYX
    # Crop to a smaller ROI for speed/visualization
    roi = membrane[:, 100:200, 100:200]
    return cells, membrane, roi


@app.cell
def _(cells, membrane, show_xyz_max_slice_interactive):
    nuclei = cells[:, 1, :, :]
    show_xyz_max_slice_interactive([nuclei, membrane], colors=['lime', 'magenta'])
    return (nuclei,)


@app.cell
def _(mo):
    sigma_slider = mo.ui.slider(1.0, 10.0, step=0.5, value=4.0, label="Gaussian Sigma")
    downscale_slider = mo.ui.slider(1, 8, step=1, value=2, label="Downscale Factor")
    mo.vstack([sigma_slider, downscale_slider])
    return downscale_slider, sigma_slider


@app.cell
def _(downscale_slider, extract_surface, roi, sigma_slider):
    # Extract surface mask
    surface_mask = extract_surface(
        roi,
        downscale_factor=downscale_slider.value,
        gaussian_sigma=sigma_slider.value
    )
    return (surface_mask,)


@app.cell
def _(roi, show_xyz_max_slice_interactive, surface_mask):
    show_xyz_max_slice_interactive([roi, surface_mask], colors=['gray', 'yellow'])
    return


@app.cell
def _(plt, roi, surface_mask):
    # Visualize: Show a side view (Z-X projection or similar)
    # Or just overlay on a slice where the surface is present.

    # Let's project max intensity of ROI and Surface Mask along Y
    roi_proj = roi.max(axis=1)
    mask_proj = surface_mask.max(axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(roi_proj, cmap='gray')
    ax[0].set_title("ROI Max Proj (Side View)")

    # Overlay
    ax[1].imshow(roi_proj, cmap='gray')
    # Create red overlay
    overlay = np.zeros(roi_proj.shape + (4,))
    overlay[mask_proj > 0] = [1, 0, 0, 0.5] # Red, semitransparent
    ax[1].imshow(overlay)
    ax[1].set_title("Surface Overlay")

    fig.tight_layout()
    fig
    return ax, fig, mask_proj, overlay, roi_proj


if __name__ == "__main__":
    app.run()
