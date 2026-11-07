# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "matplotlib",
#     "scikit-image",
#     "tifffile",
#     "xarray",
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
    The last section runs a full tiled-acquisition pipeline: standardising tiles to an STCZYX
    `xarray` (`array_to_stczyx_xarray`), shading correction, and stitching (`stitch_tiles`).
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
            "xarray",
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
                    "when you click (expect a few minutes). The tiling and stitching section also "
                    "waits for this button."
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


@app.cell
def _(mo):
    mo.md("""
    ## Tiled acquisition: standardise, correct shading, stitch

    A tiled acquisition is simulated from the nuclei volume (every 3rd z-plane): a 3×3 grid of
    overlapping tiles, each with the same vignetting, a darkfield offset and noise. As on a real
    microscope, the stage *reports* the nominal grid positions while the tiles were actually taken a
    few pixels off (stage error).

    1. `array_to_stczyx_xarray` turns the `(S, Z, Y, X)` stack into the standard 6D `STCZYX` layout
       (singleton T and C), with the channel name, the pixel sizes (physical coordinates and
       `attrs['pixel_sizes']`) and the stage positions (`stage_y` / `stage_x` coordinates along S).
       The same call standardises an existing `xarray.DataArray`, e.g. `nd2.ND2File(...).to_xarray()`.
    2. `fit_basic_shading` estimates one flatfield and darkfield from all planes of all tiles.
       BaSiC needs many images: fitted on the 9 max projections alone, it mistakes the sparse
       nuclei for shading and the correction adds seams.
    3. `stitch_tiles` registers overlapping neighbours on the max projection over Z, solves the
       tile positions globally and fuses every plane with linear blending. It is deliberately minimal (translation only, 2D
       registration on a max projection); for more control use
       [multiview-stitcher](https://github.com/multiview-stitcher/multiview-stitcher).
    """)
    return


@app.cell
def _(apply_basic_shading, cells3d, fit_basic_shading, mo, np, run_3d):
    mo.stop(run_3d is not None and not run_3d.value)
    from eigenp_utils.image_io import array_to_stczyx_xarray
    from eigenp_utils.stitching import stitch_tiles

    _rng = np.random.default_rng(0)
    tiled_volume = cells3d[::3].astype(np.float32)  # (20, 256, 256) nuclei
    _tile = 110
    _nominal = np.array([(_y, _x) for _y in (4, 72, 140) for _x in (4, 72, 140)])  # 42 px overlap
    true_px = _nominal + _rng.integers(-4, 5, size=_nominal.shape)                # stage error

    _g = np.linspace(-1, 1, _tile)
    _vignetting = 1.0 - 0.5 * (_g[:, None] ** 2 + _g[None, :] ** 2) / 2  # 1.0 centre -> 0.5 corners
    _clean = np.stack([tiled_volume[:, _y:_y + _tile, _x:_x + _tile] for _y, _x in true_px])
    _raw = (_clean * _vignetting + 100.0 + _rng.normal(0, 20.0, _clean.shape)).astype(np.float32)

    # 1. Standardise: (S, Z, Y, X) -> STCZYX with channel name, pixel sizes and stage positions
    _px = 0.26  # cells3d XY pixel size (µm); z-step 0.29 µm, every 3rd plane kept
    tile_stack = array_to_stczyx_xarray(
        _raw, 'SZYX',
        channel_names='DAPI',
        pixel_sizes={'Z': 3 * 0.29, 'Y': _px, 'X': _px, 'unit': 'µm'},
        stage_positions={'Y': _nominal[:, 0] * _px, 'X': _nominal[:, 1] * _px},
    )

    # 2. One shading model for all planes of all tiles
    _planes = tile_stack.values.reshape(-1, _tile, _tile)
    _shading = fit_basic_shading(_planes, is_3d=False, get_darkfield=True)
    _corrected = apply_basic_shading(tile_stack.values, _shading['flatfield'], _shading['darkfield'])
    _tiles_corrected = tile_stack.copy(data=_corrected)

    # 3. Stitch: stage-only placement (registration disabled) vs registered
    mosaic_stage = stitch_tiles(_tiles_corrected, min_ncc=1.1)
    mosaic = stitch_tiles(_tiles_corrected)
    mosaic_raw = stitch_tiles(tile_stack)  # without shading correction, for comparison
    return mosaic, mosaic_raw, mosaic_stage, tile_stack, true_px, tiled_volume


@app.cell
def _(tile_stack):
    tile_stack
    return


@app.cell
def _(mo, mosaic, mosaic_stage, np, true_px):
    _truth = true_px - true_px.min(axis=0)
    _err_stage = np.abs(np.array(mosaic_stage.attrs['stitching']['positions_px']) - _truth).max()
    _err_reg = np.abs(np.array(mosaic.attrs['stitching']['positions_px']) - _truth).max()
    mo.md(f"""
    Largest tile-position error: **{_err_stage} px** from the stage alone,
    **{_err_reg} px** after registration ({mosaic.attrs['stitching']['n_pairs']} registered pairs used).
    """)
    return


@app.cell
def _(mo, mosaic, mosaic_raw, mosaic_stage, plt, true_px, tiled_volume):
    _lo = true_px.min(axis=0)  # the mosaic starts at the top-left-most tile
    _truth = tiled_volume.max(axis=0)[_lo[0]:_lo[0] + mosaic.sizes['Y'], _lo[1]:_lo[1] + mosaic.sizes['X']]

    def _mip(da):
        return da.isel(S=0, T=0, C=0).max(dim='Z').values

    _panels = [
        (_truth, 'Ground truth (max over Z)'),
        (_mip(mosaic_raw), 'Registered, no shading correction'),
        (_mip(mosaic_stage), 'Stage positions only (corrected tiles)'),
        (_mip(mosaic), 'Registered (corrected tiles)'),
    ]
    fig_stitch, axes_stitch = plt.subplots(1, 4, figsize=(24, 6))
    for _ax, (_img, _title) in zip(axes_stitch, _panels):
        _ax.imshow(_img, cmap='gray')
        _ax.set_title(_title)
        _ax.axis('off')
    plot_stitch = mo.as_html(fig_stitch)
    return (plot_stitch,)


@app.cell
def _(plot_stitch):
    plot_stitch
    return


if __name__ == "__main__":
    app.run()
