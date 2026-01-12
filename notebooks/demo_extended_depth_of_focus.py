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
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))

    _axes[0].imshow(max_proj, cmap='gray')
    _axes[0].set_title("Standard Max Projection")
    _axes[0].axis('off')

    _axes[1].imshow(edof_img, cmap='gray')
    _axes[1].set_title("EDOF (best_focus_image)")
    _axes[1].axis('off')

    _im2 = _axes[2].imshow(height_map, cmap='viridis')
    _axes[2].set_title("Height Map (Z-index)")
    _axes[2].axis('off')
    plt.colorbar(_im2, ax=_axes[2], orientation='vertical')

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(membrane, np, patch_size_slider):
    ### Show difference between laplace energy and std


    from scipy.ndimage import generic_filter, zoom, laplace, uniform_filter


    _PATCH_SIZE = patch_size_slider.value

    slice_img = membrane[30, ...]


    # 1. Compute Laplacian
    lap = laplace(slice_img)
    lap = lap / lap.max()

    # 2. Compute Energy (Squared)
    energy = lap ** 2

    # 3. Local Average Energy (proxy for sum over patch)
    # We use uniform_filter with the patch size
    mean_energy = uniform_filter(energy, size=_PATCH_SIZE, mode='reflect')


    # Get image dimensions
    H, W = slice_img.shape

    # Initialize a grid to store results (optional, depends on your goal)
    # Using // ensures we only count full patches
    std_grid = np.zeros((H // _PATCH_SIZE, W // _PATCH_SIZE))

    for i in range(H // _PATCH_SIZE):
        for j in range(W // _PATCH_SIZE):
            # Calculate start and end indices for the patch
            y_start = i * _PATCH_SIZE
            y_end = y_start + _PATCH_SIZE
        
            x_start = j * _PATCH_SIZE
            x_end = x_start + _PATCH_SIZE
        
            # Extract the patch
            patch = slice_img[y_start:y_end, x_start:x_end]

            # Compute focus metric (STD of the raw patch or the Laplacian)
            # Note: If slice_img is 2D, axis=(1, 2) will throw an error. 
            # For a 2D patch, use np.std(patch).
            std_values = np.std(patch)
        
            # Store result
            std_grid[i, j] = std_values


    return lap, mean_energy, slice_img, std_grid


@app.cell
def _(mean_energy):
    mean_energy.shape
    return


@app.cell
def _(lap, mean_energy, plt, slice_img, std_grid):
    _fig, _axes = plt.subplots(1, 4, figsize=(15, 5))

    _axes[0].imshow(slice_img, cmap='gray')
    _axes[0].set_title("Original Image")
    _axes[0].axis('off')

    _axes[1].imshow(lap, cmap='gray')
    _axes[1].set_title("Lap")
    _axes[1].axis('off')

    _axes[2].imshow(mean_energy, cmap='gray')
    _axes[2].set_title("Energy")
    _axes[2].axis('off')

    _axes[3].imshow(std_grid, cmap='gray')
    _axes[3].set_title("Energy")
    _axes[3].axis('off')


    # plt.colorbar(im2, ax=_axes[2], orientation='vertical')

    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

