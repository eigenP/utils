# Executed inside Pyodide by run_pyodide_tests.mjs, which injects the globals
# WHEEL, BLOB_URL and BLOB_SHA256.
import hashlib
import sys

import micropip

sys.dont_write_bytecode = True  # /repo is the host checkout mounted via NODEFS

# 1. Core install only: this is what a marimo WASM notebook gets by default.
await micropip.install(WHEEL)
import eigenp_utils
from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

assert "skimage" not in sys.modules, "core import must not pull in scikit-image"
print(f"core import OK: {len(eigenp_utils.__all__)} names exported")

# Browser download path: pyfetch, binary-safe, and caching an existing file
from pathlib import Path
from eigenp_utils.io import download_file_async

blob = await download_file_async(BLOB_URL, "/tmp/dl/blob.bin")
assert hashlib.sha256(Path(blob).read_bytes()).hexdigest() == BLOB_SHA256
Path(blob).write_bytes(b"cached")
await download_file_async(BLOB_URL, blob)
assert Path(blob).read_bytes() == b"cached"
print("download_file_async (pyfetch) OK")

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
