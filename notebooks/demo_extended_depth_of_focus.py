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
    # Extended Depth of Focus (EDOF) Demo

    Demonstrates `best_focus_image` which fuses a Z-stack into a single in-focus image using a patch-based sharpness metric.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from eigenp_utils.io import download_file_async
    from eigenp_utils.extended_depth_of_focus import best_focus_image
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

    return (
        best_focus_image,
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
def _(cells_path, imread):
    # Use cells3d, membrane channel (channel 0)
    # cells3d shape: (60, 2, 256, 256) -> (Z, C, Y, X)
    cells = imread(cells_path)
    membrane_stack = cells[:, 0, :, :]
    return cells, membrane_stack


@app.cell
def _(cells, show_zyx_max_slice_interactive):
    nuclei = cells[:, 1, :, :]
    membrane = cells[:, 0, :, :]
    show_zyx_max_slice_interactive([nuclei, membrane], colors=['lime', 'magenta'])
    return (membrane,)


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


    # 1. Compute Laplacian (in float: on uint16 data negative values would wrap around)
    lap = laplace(slice_img.astype(np.float32))
    lap = lap / np.abs(lap).max()

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
    _axes[3].set_title("Patch STD")
    _axes[3].axis('off')


    # plt.colorbar(im2, ax=_axes[2], orientation='vertical')

    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
