# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "scikit-image",
#     "tifffile",
#     "eigenp-utils @ git+https://github.com/eigenP/utils.git@main ; sys_platform != 'emscripten'",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # BaSiC shading correction demo

    This notebook demonstrates the basic usage of `fit_basic_shading` and `apply_basic_shading`.
    We artificially degrade the `skimage.data.cells3d` dataset and then recover the true intensities.
    """)
    return


@app.cell(hide_code=True)
async def _():
    import sys

    # In the browser (marimo WASM): install the package wheel and the packages
    # marimo cannot detect from this notebook's own imports
    IN_WASM = sys.platform == "emscripten"
    if IN_WASM:
        import micropip

        await micropip.install([
            # micropip needs a wheel from a CORS-enabled host (build-wasm-wheel.yml)
            "https://raw.githubusercontent.com/eigenP/utils/wasm-wheel/eigenp_utils-0-py3-none-any.whl",
            "scikit-image",
            "tifffile",
        ])
    wasm_ready = True
    return IN_WASM, wasm_ready


@app.cell
def _(wasm_ready):
    import numpy as np
    import skimage.data
    from eigenp_utils.intensity_rescaling import fit_basic_shading, apply_basic_shading
    import matplotlib.pyplot as plt

    return apply_basic_shading, fit_basic_shading, np, plt, skimage


@app.cell
async def _(apply_basic_shading, fit_basic_shading, mo, np, plt, skimage):
    try:
        cells3d = skimage.data.cells3d()[:, 1, :, :] # DAPI channel
    except Exception:
        # e.g. no pooch, or in the browser (WASM). GitLab's API endpoint sends
        # CORS headers (its /-/raw/ URL does not); same file as cells3d().
        import tifffile
        from eigenp_utils.io import download_file_async
        url = "https://gitlab.com/api/v4/projects/scikit-image%2Fdata/repository/files/cells3d.tif/raw?ref=master"
        file_path = await download_file_async(url, "./cells3d.tif")
        cells3d = tifffile.imread(file_path)[:, 1, :, :]

    sizes = cells3d.shape[1:]
    grid = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes], indexing='ij'))
    gradient = np.sum(grid**2, axis=0)
    gradient = 0.5 * (np.max(gradient) - gradient) / np.max(gradient) + 0.5 # 0.5 to 1.0
    darkfield_ground_truth = np.ones(sizes) * 5.0

    # apply degradation
    degraded = cells3d * gradient[np.newaxis, ...] + darkfield_ground_truth[np.newaxis, ...]

    # run basic
    res = fit_basic_shading(degraded, is_3d=False, get_darkfield=True, fitting_mode='approximate')
    corrected = apply_basic_shading(degraded, res['flatfield'], res['darkfield'], res['baseline'])

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(cells3d[30], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(degraded[30], cmap='gray')
    axes[1].set_title('Degraded')
    axes[1].axis('off')

    axes[2].imshow(res['flatfield'], cmap='gray')
    axes[2].set_title('Estimated Flatfield')
    axes[2].axis('off')

    axes[3].imshow(res['darkfield'], cmap='gray')
    axes[3].set_title('Estimated Darkfield')
    axes[3].axis('off')

    axes[4].imshow(corrected[30], cmap='gray')
    axes[4].set_title('Corrected')
    axes[4].axis('off')

    plot = mo.as_html(fig)
    return cells3d, plot


@app.cell
def _(plot):
    plot
    return


@app.cell
def _(IN_WASM, cells3d, mo):
    # The 3D fits take minutes even natively; in the browser use a downsampled
    # volume and only run them on demand
    if IN_WASM:
        cells3d_vol = cells3d[::2, ::4, ::4]
        run_3d = mo.ui.run_button(label="Run the 3D sections")
        _view = mo.callout(
            mo.vstack([
                mo.md(
                    "**Browser (WebAssembly) version:** the 3D fits below are slow here, so they use a "
                    f"downsampled volume {cells3d_vol.shape} (every 2nd z, every 4th y/x) and run only "
                    "when you click (expect a few minutes)."
                ),
                run_3d,
            ]),
            kind="info",
        )
    else:
        cells3d_vol, run_3d, _view = cells3d, None, None
    _view
    return cells3d_vol, run_3d


@app.cell
def _(mo):
    mo.md("""
    ## 3D Shading Correction

    For volumes with smoothly varying intensity biases across three dimensions, we can fit a 3D shading correction by passing `is_3d=True`. Here we demonstrate recovering a simulated 3D degradation.

    A darkfield cannot be separated from the flatfield using a single volume, so this section fits the flatfield only (`get_darkfield=False`). Estimating a darkfield needs several volumes, as in the tile section below.
    """)
    return


@app.cell
def _(apply_basic_shading, cells3d_vol, fit_basic_shading, mo, np, plt, run_3d):
    mo.stop(run_3d is not None and not run_3d.value)
    sizes_3d = cells3d_vol.shape
    grid_3d = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in sizes_3d], indexing='ij'))
    gradient_3d = np.sum(grid_3d**2, axis=0)
    gradient_3d = 0.5 * (np.max(gradient_3d) - gradient_3d) / np.max(gradient_3d) + 0.5 # 0.5 to 1.0
    darkfield_ground_truth_3d = np.ones(sizes_3d) * 5.0

    degraded_3d = cells3d_vol * gradient_3d + darkfield_ground_truth_3d

    # A single volume cannot identify a darkfield: fit the flatfield only
    res_3d = fit_basic_shading(degraded_3d, is_3d=True, get_darkfield=False, fitting_mode='approximate')
    corrected_3d = apply_basic_shading(degraded_3d, res_3d['flatfield'], res_3d['darkfield'], res_3d['baseline'])

    mid_z = cells3d_vol.shape[0] // 2

    fig_3d, axes_3d = plt.subplots(1, 4, figsize=(20, 5))
    axes_3d[0].imshow(cells3d_vol[mid_z], cmap='gray')
    axes_3d[0].set_title('Original 3D (mid z-slice)')
    axes_3d[0].axis('off')

    axes_3d[1].imshow(degraded_3d[mid_z], cmap='gray')
    axes_3d[1].set_title('Degraded 3D')
    axes_3d[1].axis('off')

    axes_3d[2].imshow(res_3d['flatfield'][mid_z], cmap='gray')
    axes_3d[2].set_title('Estimated Flatfield 3D')
    axes_3d[2].axis('off')

    axes_3d[3].imshow(corrected_3d[mid_z], cmap='gray')
    axes_3d[3].set_title('Corrected 3D')
    axes_3d[3].axis('off')

    plot_3d = mo.as_html(fig_3d)
    return (plot_3d,)


@app.cell
def _(plot_3d):
    plot_3d
    return


@app.cell
def _(mo):
    mo.md("""
    ## Shading Correction on Simulated Tiles

    Here we simulate a tiled acquisition by splitting the 3D stack into 4 spatial quadrants. We apply a common degradation with slight realistic noise to each tile. We then pass the list of 4 tiles to `fit_basic_shading` with `is_3d=True` to compute a single, shared 3D flatfield and darkfield across the tiles.
    """)
    return


@app.cell
def _(apply_basic_shading, cells3d_vol, fit_basic_shading, mo, np, plt, run_3d):
    mo.stop(run_3d is not None and not run_3d.value)
    z, y, x = cells3d_vol.shape
    y2, x2 = y // 2, x // 2

    # 4 spatial quadrants
    tiles = [
        cells3d_vol[:, :y2, :x2],
        cells3d_vol[:, :y2, x2:],
        cells3d_vol[:, y2:, :x2],
        cells3d_vol[:, y2:, x2:]
    ]

    tile_shape = tiles[0].shape
    grid_tile = np.array(np.meshgrid(*[np.linspace(-s // 2 + 1, s // 2, s) for s in tile_shape], indexing='ij'))
    gradient_tile = np.sum(grid_tile**2, axis=0)
    gradient_tile = 0.5 * (np.max(gradient_tile) - gradient_tile) / np.max(gradient_tile) + 0.5 # 0.5 to 1.0
    darkfield_ground_truth_tile = np.ones(tile_shape) * 5.0

    degraded_tiles = []
    for tile in tiles:
        # Apply degradation + slight gaussian noise
        noise = np.random.normal(0, 1.0, size=tile.shape)
        _degraded = tile * gradient_tile + darkfield_ground_truth_tile + noise
        degraded_tiles.append(_degraded)

    # Fit basic on the list of tiles, computing a shared 3D shading
    res_tiles = fit_basic_shading(degraded_tiles, is_3d=True, get_darkfield=True, fitting_mode='approximate')

    # Apply to the first tile; the fit returns one baseline per tile, so pass tile 0's
    corrected_tile0 = apply_basic_shading(degraded_tiles[0], res_tiles['flatfield'], res_tiles['darkfield'], res_tiles['baseline'][:1])

    mid_z_tile = tile_shape[0] // 2

    fig_tiles, axes_tiles = plt.subplots(1, 5, figsize=(25, 5))
    axes_tiles[0].imshow(tiles[0][mid_z_tile], cmap='gray')
    axes_tiles[0].set_title('Original Tile 0 (mid z)')
    axes_tiles[0].axis('off')

    axes_tiles[1].imshow(degraded_tiles[0][mid_z_tile], cmap='gray')
    axes_tiles[1].set_title('Degraded Tile 0')
    axes_tiles[1].axis('off')

    axes_tiles[2].imshow(res_tiles['flatfield'][mid_z_tile], cmap='gray')
    axes_tiles[2].set_title('Estimated Shared Flatfield 3D')
    axes_tiles[2].axis('off')

    axes_tiles[3].imshow(res_tiles['darkfield'][mid_z_tile], cmap='gray')
    axes_tiles[3].set_title('Estimated Shared Darkfield 3D')
    axes_tiles[3].axis('off')

    axes_tiles[4].imshow(corrected_tile0[mid_z_tile], cmap='gray')
    axes_tiles[4].set_title('Corrected Tile 0')
    axes_tiles[4].axis('off')

    plot_tiles = mo.as_html(fig_tiles)
    return (plot_tiles,)


@app.cell
def _(plot_tiles):
    plot_tiles
    return


if __name__ == "__main__":
    app.run()
