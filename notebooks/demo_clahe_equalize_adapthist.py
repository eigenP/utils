# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scikit-image",
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
    # CLAHE Demo

    Demonstrating `clahe_u8` from `eigenp_utils.clahe_equalize_adapthist`:
    fast N-D 8-bit CLAHE for segmentation preprocessing (uint8 output), on a
    2D slice and on the whole 3D stack. It replaces the deprecated
    `_my_clahe_` (same `clip_limit` scale; float output in [0, 1]), which is
    shown only for the timing comparison below.
    """)
    return


@app.cell
def _():
    import importlib.util
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.io import imread
    from eigenp_utils.io import download_file_async
    import warnings
    # marimo keeps names starting with "_" private to their cell: alias it
    from eigenp_utils.clahe_equalize_adapthist import _my_clahe_ as my_clahe
    from eigenp_utils.clahe_equalize_adapthist import clahe_u8
    from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

    return (
        clahe_u8,
        download_file_async,
        imread,
        importlib,
        my_clahe,
        np,
        plt,
        show_zyx_max_slice_interactive,
        time,
        warnings,
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
    # cells3d shape: (Z, C, Y, X). C=0 membrane, C=1 nuclei.
    cells = imread(cells_path)
    # Pick a middle slice of the membrane channel
    membrane_slice = cells[30, 0, :, :]
    return cells, membrane_slice


@app.cell
def _(cells, show_zyx_max_slice_interactive):
    nuclei = cells[:, 1, :, :]
    membrane = cells[:, 0, :, :]
    show_zyx_max_slice_interactive([nuclei, membrane], colormap=['lime', 'magenta'])
    return


@app.cell
def _(mo):
    clip_slider = mo.ui.slider(start=0.0, stop=0.1, step=0.005, value=0.01, label="Clip Limit")
    kernel_size_slider = mo.ui.slider(start=8, stop=128, step=8, value=32, label="Kernel Size")

    mo.vstack([clip_slider, kernel_size_slider])
    return clip_slider, kernel_size_slider


@app.cell
def _(clahe_u8, clip_slider, kernel_size_slider, membrane_slice):
    # uint16 slice -> uint8 between the 0.1 / 99.9 percentiles, then CLAHE
    clahe_img = clahe_u8(
        membrane_slice,
        clip_limit=clip_slider.value,
        kernel_size=kernel_size_slider.value,
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Fast 3D CLAHE with `clahe_u8`

    `clahe_u8` returns uint8 and is meant for segmentation preprocessing,
    not quantitative intensities. Non-uint8 input (here the raw uint16 stack)
    is mapped to 8 bits between the 0.1 and 99.9 percentiles, so a few hot or
    dead pixels do not set the range (this is why it looks brighter than
    `_my_clahe_`, which maps the stack's min/max). Contextual regions are 3D, so contrast
    varies smoothly along Z too, without slice-to-slice banding (XZ views).
    The sliders above set the clip limit and the Y/X tile size; the Z tile is
    1/8 of the stack.

    With `numba` installed it runs fused, multithreaded loops (the first call
    in a fresh environment compiles for a few seconds); otherwise, e.g. in
    the browser, it falls back to an identical NumPy implementation.
    """)
    return


@app.cell
def _(
    cells,
    clahe_u8,
    clip_slider,
    importlib,
    kernel_size_slider,
    my_clahe,
    time,
    warnings,
):
    membrane_stack = cells[:, 0, :, :]
    kernel_3d = (max(membrane_stack.shape[0] // 8, 1),) + (kernel_size_slider.value,) * 2
    clahe_backend = "numba" if importlib.util.find_spec("numba") else "numpy"
    # warm-up: excludes numba's one-off compilation / cache load from the timing
    clahe_u8(membrane_stack[:2, :16, :16])

    # _my_clahe_ is deprecated: run here only as the timing reference
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "_my_clahe_ is deprecated", DeprecationWarning)
        _t0 = time.perf_counter()
        volume_ref = my_clahe(membrane_stack, clip_limit=clip_slider.value, kernel_size=kernel_3d)
        t_ref = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    volume_fast = clahe_u8(membrane_stack, clip_limit=clip_slider.value, kernel_size=kernel_3d)
    t_fast = time.perf_counter() - _t0
    return clahe_backend, kernel_3d, membrane_stack, t_fast, t_ref, volume_fast, volume_ref


@app.cell
def _(clahe_backend, kernel_3d, membrane_stack, mo, t_fast, t_ref):
    mo.md(f"""
    3D CLAHE of the {membrane_stack.shape} membrane stack, tiles {kernel_3d}:
    `_my_clahe_` (deprecated) **{t_ref * 1e3:.0f} ms**, `clahe_u8` ({clahe_backend} backend)
    **{t_fast * 1e3:.0f} ms** (**{t_ref / t_fast:.0f}x** faster).
    """)
    return


@app.cell
def _(clip_slider, membrane_stack, np, plt, volume_fast, volume_ref):
    z_show = membrane_stack.shape[0] // 2
    y_show = membrane_stack.shape[1] // 2
    panels = [
        ("Original", membrane_stack),
        ("_my_clahe_ (deprecated)", volume_ref),
        ("clahe_u8", volume_fast),
    ]
    fig_3d, axes_3d = plt.subplots(
        2, 3, figsize=(12, 6),
        height_ratios=[membrane_stack.shape[1], 2 * membrane_stack.shape[0]],
    )
    for col, (name, vol) in enumerate(panels):
        axes_3d[0, col].imshow(vol[z_show], cmap="gray")
        axes_3d[0, col].set_title(f"{name}: XY, z={z_show}")
        # aspect=2: cells3d's Z step is about twice its XY pixel size
        axes_3d[1, col].imshow(
            np.asarray(vol[:, y_show, :], dtype=np.float32), cmap="gray", aspect=2
        )
        axes_3d[1, col].set_title(f"XZ, y={y_show}")
    for ax in axes_3d.ravel():
        ax.axis("off")
    fig_3d.suptitle(f"3D CLAHE, clip_limit={clip_slider.value:.3f}")
    fig_3d.tight_layout()
    fig_3d
    return


if __name__ == "__main__":
    app.run()
