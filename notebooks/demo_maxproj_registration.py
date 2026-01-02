# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "scikit-image",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "pandas",
#     "tqdm",
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
    # Registration / Drift Correction Demo

    Demonstrates `estimate_drift_2D` and `apply_drift_correction_2D` on a synthetically drifted stack.
    """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import data
    from scipy.ndimage import shift
    from eigenp_utils.maxproj_registration import (
        estimate_drift_2D,
        apply_drift_correction_2D
    )
    return (
        apply_drift_correction_2D,
        data,
        estimate_drift_2D,
        np,
        plt,
        shift,
    )


@app.cell
def _(data, np, shift):
    # Use cells3d
    cells = data.cells3d()
    # Take a smaller subset of frames/slices to speed up
    # Treat Z as Time for this demo
    original_stack = cells[:20, 1, :, :] # Nuclei channel

    # Introduce synthetic drift
    drifted_stack = np.zeros_like(original_stack)
    true_drifts = []

    current_dx, current_dy = 0, 0
    for t in range(original_stack.shape[0]):
        # Random walk drift
        if t > 0:
            current_dx += np.random.randint(-2, 3)
            current_dy += np.random.randint(-2, 3)

        drifted_stack[t] = shift(original_stack[t], shift=(current_dy, current_dx), mode='constant')
        true_drifts.append((current_dy, current_dx))

    return (
        cells,
        current_dx,
        current_dy,
        drifted_stack,
        original_stack,
        t,
        true_drifts,
    )


@app.cell
def _(apply_drift_correction_2D, drifted_stack):
    corrected_stack, drift_table = apply_drift_correction_2D(drifted_stack)
    return corrected_stack, drift_table


@app.cell
def _(drift_table, mo):
    mo.ui.table(drift_table)
    return


@app.cell
def _(corrected_stack, drifted_stack, plt):
    # Visualize Max Projections
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(drifted_stack.max(axis=0), cmap='inferno')
    axes[0].set_title("Drifted Stack (Max Proj)")
    axes[0].axis('off')

    axes[1].imshow(corrected_stack.max(axis=0), cmap='inferno')
    axes[1].set_title("Corrected Stack (Max Proj)")
    axes[1].axis('off')

    fig.tight_layout()
    fig
    return axes, fig


if __name__ == "__main__":
    app.run()
