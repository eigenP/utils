# Executed inside Pyodide by run_pyodide_tests.mjs; WHEEL is injected as a global.
import sys

import micropip

sys.dont_write_bytecode = True  # /repo is the host checkout mounted via NODEFS

# 1. Core install only: this is what a marimo WASM notebook gets by default.
await micropip.install(WHEEL)
import eigenp_utils
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

assert "skimage" not in sys.modules, "core import must not pull in scikit-image"
print(f"core import OK: {len(eigenp_utils.__all__)} names exported")

# 2. Optional extras that have Pyodide builds, then the regular test suite.
await micropip.install([
    "scikit-image", "xarray", "tifffile",
    "plotly", "statannotations", "statsmodels",
    "pytest",
])

import pytest

rc = pytest.main([
    "/repo/tests", "-q", "-p", "no:cacheprovider", "-rfE",
    # single_cell needs scanpy/numba (no WASM build); frontend tests drive
    # Chromium via Playwright, which cannot run inside Pyodide.
    "--ignore=/repo/tests/test_single_cell.py",
    "--ignore=/repo/tests/test_frontend_isolated.py",
    "--ignore=/repo/tests/test_frontend_rangeslider.py",
    "--ignore=/repo/tests/wasm",
])
int(rc)
