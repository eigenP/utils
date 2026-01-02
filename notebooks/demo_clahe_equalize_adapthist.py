# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scikit-image",
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
    # CLAHE Demo

    Demonstrating `_my_clahe_` from `eigenp_utils.clahe_equalize_adapthist`.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import data
    from eigenp_utils.clahe_equalize_adapthist import _my_clahe_
    return _my_clahe_, data, np, plt


@app.cell
def _(data):
    # Use cells3d nuclei channel (channel 1 is nuclei, 0 is membrane usually... let's check doc or just pick one)
    # cells3d shape: (Z, C, Y, X). C=0 membrane, C=1 nuclei.
    cells = data.cells3d()
    # Pick a middle slice of the membrane channel
    membrane_slice = cells[30, 1, :, :]
    return cells, membrane_slice


@app.cell
def _(mo):
    clip_slider = mo.ui.slider(start=0.0, stop=0.1, step=0.005, value=0.01, label="Clip Limit")
    nbins_slider = mo.ui.slider(start=64, stop=512, step=64, value=256, label="Bins")
    kernel_size_slider = mo.ui.slider(start=8, stop=128, step=8, value=32, label="Kernel Size")

    mo.vstack([clip_slider, nbins_slider, kernel_size_slider])
    return clip_slider, kernel_size_slider, nbins_slider


@app.cell
def _(_my_clahe_, clip_slider, kernel_size_slider, membrane_slice, nbins_slider):
    clahe_img = _my_clahe_(
        membrane_slice,
        clip_limit=clip_slider.value,
        nbins=nbins_slider.value,
        kernel_size=kernel_size_slider.value
    )
    return (clahe_img,)


@app.cell
def _(clahe_img, membrane_slice, plt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(membrane_slice, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(clahe_img, cmap='gray')
    axes[1].set_title("CLAHE")
    axes[1].axis('off')
    fig.tight_layout()
    fig
    return axes, fig


if __name__ == "__main__":
    app.run()
