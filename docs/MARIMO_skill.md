# marimo notebooks in this repo

Rules for the notebooks in `notebooks/` (marimo `.py` apps). Edit only the bodies of `@app.cell` functions; marimo derives each cell's arguments and returns from the names it uses and defines.

## Names
- A name may be defined in only one cell. Never redeclare a global name in another cell (rename it, or make it private).
- Names starting with `_` are private to their cell (`_fig`, `_axes`, `_im` are reused freely across cells). To use a private helper from another cell, import it under a public alias and return it, e.g. `from eigenp_utils.tnia_plotting_anywidgets import _rotate_expand as rotate_expand`.
- The repo's pinned marimo cannot resolve a recursive cell-private function (`def _walk(...)` calling `_walk`); write tree walks as explicit stack loops (see `docs/MEMORIES.md`).

## Cell layout used by the demo notebooks
1. A PEP 723 header (`# /// script`, `requires-python = ">=3.11"`); entries without a WASM build (`git+` URLs, numba, scanpy, ...) carry `; sys_platform != 'emscripten'`.
2. Cell 1: `import marimo as mo`, `return (mo,)`.
3. Cell 2 (`hide_code=True`, `async def`): the setup cell. It defines `in_wasm()` (`sys.platform in ("emscripten", "wasi")`); in WASM it `await`s `micropip.install([...])` with the `wasm-wheel` branch wheel plus pure-Python extras in one call; natively it installs from GitHub with `uv pip` (falling back to pip) unless `local_checkout()` is true (a source checkout or editable install, or `EIGENP_UTILS_NO_INSTALL` set). It then imports `eigenp_utils` and prints where it came from. `tests/test_notebook_setup.py` checks `local_checkout`.
4. Later cells import what they need and return the names other cells use. Use the current API only: `tests/test_notebook_setup.py` fails on deprecated names (`_my_clahe_`, `hist_imshow`, `adjust_*_per_slice`, ...) and deprecated viewer kwargs (`colors=`, `sxy`/`sz`, `x_s…z_t`). Once a viewer gets `pixel_sizes=`, `slabs_position` / `slabs_thickness` are in µm, so convert voxel counts (the `cells3d` demos use `CELLS3D_PIXEL_SIZES = {'Z': 0.29, 'Y': 0.26, 'X': 0.26, 'unit': 'µm'}`). Downloads use `await eigenp_utils.io.download_file_async(...)` and return the path, so reads run after them.

## Display
- Matplotlib: use the object-oriented interface. Create figures with `fig, ax = plt.subplots(...)`, draw on `ax`, and make the Figure the cell's last expression (or return it). Save with `savefig_svg(path, fig=fig)`. Never end a cell with `plt.gca()` or `plt.show()`, and never rely on pyplot's current figure.
- anywidgets (`LabelAnnotator`, the ortho viewers): wrap in `mo.ui.anywidget(widget)` when other cells must react to its traits, and display the wrapper as the cell's last expression.
- Text: `mo.md(...)`; derive numbers quoted in prose from a cell output instead of hard-coding them.

## Verification
- Run every changed notebook headlessly: `python notebooks/<name>.py` runs `app.run()` and exits non-zero on a cell error. `marimo check` alone is not enough.
- From a source checkout set `PYTHONPATH=src` (and optionally `EIGENP_UTILS_NO_INSTALL=1`) so the notebook uses the local code. If the environment has no marimo (e.g. the shared sandbox `.venv`), install it into a scratch directory (`pip install --target <dir> marimo`) and add that directory to `PYTHONPATH` rather than changing the venv.
- Test under both marimo versions: the locked 0.20.4 (`uv.lock`) and 0.25.1 (`uv run --with marimo==0.25.1 python notebooks/<name>.py`).
- Notebooks that cannot run in WASM (`demo_sc_*`) say so in a header comment and point to molab or `uvx marimo edit --sandbox`.
