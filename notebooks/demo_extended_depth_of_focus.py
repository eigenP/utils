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
    # Extended Depth of Focus (EDOF) Demo

    Demonstrates `best_focus_image` which fuses a Z-stack into a single in-focus image using a patch-based sharpness metric.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import data
    from eigenp_utils.extended_depth_of_focus import best_focus_image
    return best_focus_image, data, np, plt


@app.cell
def _(data):
    # Use cells3d, membrane channel (channel 0)
    # cells3d shape: (60, 2, 256, 256) -> (Z, C, Y, X)
    cells = data.cells3d()
    membrane_stack = cells[:, 0, :, :]
    return cells, membrane_stack


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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(max_proj, cmap='gray')
    axes[0].set_title("Standard Max Projection")
    axes[0].axis('off')

    axes[1].imshow(edof_img, cmap='gray')
    axes[1].set_title("EDOF (best_focus_image)")
    axes[1].axis('off')

    im2 = axes[2].imshow(height_map, cmap='viridis')
    axes[2].set_title("Height Map (Z-index)")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], orientation='vertical')

    fig.tight_layout()
    fig
    return axes, fig, im2


if __name__ == "__main__":
    app.run()
