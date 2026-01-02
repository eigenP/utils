# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scikit-image",
#     "numpy",
#     "matplotlib",
#     "scipy",
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
def _():
    import os, sys, subprocess, shutil
    from pathlib import Path

    OWNER, REPO, REF = "eigenP", "utils", "main"  # or a tag/branch/commit
    work = Path.cwd() / "_ext" / f"{REPO}-{REF}"
    src = work / "src"

    # clean old checkout
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)

    # shallow clone at the desired ref
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REF, f"https://github.com/{OWNER}/{REPO}.git", str(work)],
        check=True
    )

    # ensure src is importable
    p = str(src.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

    import eigenp_utils  # noqa: E402
    print("eigenp_utils imported from:", eigenp_utils.__file__)

    return


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
    from skimage import data
    from eigenp_utils.surface_extraction import extract_surface
    return extract_surface, data, np, plt


@app.cell
def _(data):
    # Use cells3d membrane channel
    cells = data.cells3d()
    membrane = cells[:, 0, :, :] # ZYX
    # Crop to a smaller ROI for speed/visualization
    roi = membrane[:, 100:200, 100:200]
    return cells, membrane, roi


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
