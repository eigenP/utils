import marimo

__generated_with = "0.23.2"
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


@app.cell
def _():
    import numpy as np
    import skimage.data
    from eigenp_utils.intensity_rescaling import fit_basic_shading, apply_basic_shading
    import matplotlib.pyplot as plt

    return apply_basic_shading, fit_basic_shading, np, plt, skimage


@app.cell
def _(apply_basic_shading, fit_basic_shading, mo, np, plt, skimage):
    try:
        cells3d = skimage.data.cells3d()[:, 1, :, :] # DAPI channel
    except Exception:
        import pooch
        import tifffile
        url = "https://gitlab.com/scikit-image/data/-/raw/master/cells3d.tif"
        file_path = pooch.retrieve(url, known_hash=None)
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cells3d[30], cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(degraded[30], cmap='gray')
    axes[1].set_title('Degraded (with darkfield & flatfield)')
    axes[1].axis('off')

    axes[2].imshow(corrected[30], cmap='gray')
    axes[2].set_title('Corrected')
    axes[2].axis('off')

    plot = mo.as_html(fig)
    return (plot,)


@app.cell
def _(plot):
    plot
    return


if __name__ == "__main__":
    app.run()
