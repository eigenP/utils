# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "anywidget",
#     "traitlets",
#     "tifffile",
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
            return micropip.install([GIT_URL, "tifffile"])
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
    # LabelAnnotator: in-notebook label painting

    `eigenp_utils.label_annotator.LabelAnnotator` paints labels on 2D slices of a
    multichannel image directly in the notebook (no napari):

    * **instance** mode: every closed stroke becomes a new object id (e.g. Cellpose
      training masks); strokes are closed and filled.
    * **semantic** mode: paint class ids (e.g. pixel-classifier scribbles); 0 = unlabelled.

    Tools: brush (`b`), eraser (`e`), flood fill (`f`), label picker (`p`), brush size
    `[` / `]`, undo `Ctrl+Z`, new id `n`, class `1`-`9`, slice `↑`/`↓`, zoom with the
    wheel, pan with right-drag or space+drag, `0` to fit. Click the image first so it
    receives key presses.
    """)
    return


@app.cell
def _():
    import numpy as np
    from eigenp_utils.label_annotator import LabelAnnotator

    # synthetic 2-channel z-stack: membranes (ring) + a tubular "organelle"
    rng = np.random.default_rng(0)
    nz, ny, nx = 5, 128, 160
    zz, yy, xx = np.mgrid[:nz, :ny, :nx]
    r = np.hypot(yy - 64, xx - 80)
    membrane = np.exp(-((r - 45 + 2 * zz) ** 2) / 10.0)
    organelle = np.exp(-((yy - 64 - 12 * np.sin(xx / 10.0)) ** 2) / 4.0) * (r < 38)
    image = np.stack([membrane, organelle]) + 0.15 * rng.random((2, nz, ny, nx))
    image = (1000 * image).astype(np.uint16)  # CZYX
    return LabelAnnotator, image, np


@app.cell
def _(LabelAnnotator, image, mo):
    instance_widget = LabelAnnotator(
        image, axes="CZYX", mode="instance",
        channel_names=["membrane", "organelle"], channel_colors=["#ff00ff", "#00ff00"],
    )
    instance_ui = mo.ui.anywidget(instance_widget)
    instance_ui
    return (instance_widget,)


@app.cell
def _(instance_widget, mo):
    mo.md(f"""
    Instance labels: shape `{instance_widget.labels.shape}`, ids present:
    `{sorted(set(instance_widget.labels.ravel().tolist()) - {0})[:20]}`;
    annotated slices: `{instance_widget.annotated_slices()}`.
    Rerun this cell after painting (or save with `instance_widget.save(dir, name)`).
    """)
    return


@app.cell
def _(LabelAnnotator, image, mo):
    semantic_widget = LabelAnnotator(
        image, axes="CZYX", mode="semantic", classes=["organelle", "cytosol", "background"],
        channel_names=["membrane", "organelle"], brush_radius=3,
    )
    semantic_ui = mo.ui.anywidget(semantic_widget)
    semantic_ui
    return (semantic_widget,)


@app.cell
def _(np, semantic_widget):
    counts = {name: int((semantic_widget.labels == k + 1).sum()) for k, name in enumerate(semantic_widget.class_names)}
    counts["unlabelled"] = int((semantic_widget.labels == 0).sum())
    counts
    return


if __name__ == "__main__":
    app.run()
