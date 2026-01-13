# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
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
        GIT_URL = f"eigenp-utils @ https://github.com/{OWNER}/{REPO}/archive/{REF}.zip"
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

    def install_github():
        if in_wasm():
            import micropip
            return micropip.install(GIT_URL)
        else:
            install_local(GIT_URL)

    res = install_github()
    if res is not None:
        await res

    import eigenp_utils
    print("eigenp_utils imported from:", eigenp_utils.__file__)

    return (
        GIT_URL,
        OWNER,
        REF,
        REPO,
        eigenp_utils,
        in_wasm,
        install_github,
        install_local,
        res,
        sys,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Dimensionality Parser Demo

    Demonstrates the `dimensionality_parser` decorator which allows functions written for lower dimensions (e.g., 2D or 3D) to automatically iterate over higher dimensions (e.g., Time, Channel, Z).
    """
    )
    return


@app.cell
def _():
    import numpy as np
    from eigenp_utils.dimensionality_parser import dimensionality_parser
    return dimensionality_parser, np


@app.cell
def _(mo):
    target_dims_input = mo.ui.text(value="YX", label="Target Dimensions (e.g., YX, ZYX)")
    input_shape_text = mo.ui.text(value="10, 5, 20, 20", label="Input Shape (S, C, T, Z, Y, X) subset")
    target_dims_input
    input_shape_text
    return input_shape_text, target_dims_input


@app.cell
def _(dimensionality_parser, input_shape_text, mo, np, target_dims_input):
    # Parse input shape
    try:
        shape = tuple(map(int, input_shape_text.value.split(',')))
        dummy_data = np.random.rand(*shape)

        # Define a simple function that processes the target dimensions
        # For demo, let's say it just adds 1 to the slice

        target_dims = target_dims_input.value

        @dimensionality_parser(target_dims=target_dims)
        def process_slice(arr):
            # Verify arr has the expected dimensionality
            # Note: This runs inside the loop
            return arr + 1.0

        result = process_slice(dummy_data)

        output_msg = f"""
        **Input Shape**: {dummy_data.shape}
        **Target Dimensions**: {target_dims}
        **Result Shape**: {result.shape}
        **Successfully iterated!**
        """
    except Exception as e:
        output_msg = f"**Error**: {str(e)}"

    mo.md(output_msg)
    return dummy_data, output_msg, process_slice, result, shape, target_dims


if __name__ == "__main__":
    app.run()
