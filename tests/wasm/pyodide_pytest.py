# Executed inside Pyodide by run_pyodide_tests.mjs, which injects the globals
# WHEEL, BLOB_URL and BLOB_SHA256.
import hashlib
import sys

import micropip

sys.dont_write_bytecode = True  # /repo is the host checkout mounted via NODEFS

# 1. Core install: this is what a marimo WASM notebook gets by default. anndata
#    (pure Python, for step 3) must be resolved in the same call: it needs a newer
#    typing-extensions than Pyodide bundles, and micropip never upgrades a package
#    that an earlier call already installed.
await micropip.install([WHEEL, "anndata"])
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
    "scikit-learn",  # imported by single_cell
    "pytest",
])

# 3. single_cell imports scanpy lazily, so its numpy/scipy graph helpers work
#    in the browser even though scanpy (numba) has no WebAssembly build.
import anndata
import numpy as np
import scipy.sparse as sp
from eigenp_utils.single_cell import find_correlated_features, smooth_expression_on_graph

rng = np.random.default_rng(0)
adata = anndata.AnnData(rng.random((6, 3)))
adata.var_names = ["a", "b", "c"]
ring = np.arange(6)
adata.obsp["connectivities"] = sp.csr_matrix(
    (np.ones(12), (np.r_[ring, ring], np.r_[(ring + 1) % 6, (ring - 1) % 6])), shape=(6, 6)
)
A = adata.obsp["connectivities"].toarray() + np.eye(6)
assert np.allclose(smooth_expression_on_graph(adata, adata.X), (A / A.sum(1, keepdims=True)) @ adata.X)
res = find_correlated_features(adata, "a", metrics=("pearson", "wasserstein"), use_graph=True)
assert np.isclose(res.loc["a", "pearson"], 1.0)
assert "scanpy" not in sys.modules
print("single_cell graph helpers without scanpy OK")

import pytest

rc = pytest.main([
    "/repo/tests", "-q", "-p", "no:cacheprovider", "-rfE",
    # test_single_cell needs scanpy/numba (no WASM build); frontend tests drive
    # Chromium via Playwright, which cannot run inside Pyodide.
    "--ignore=/repo/tests/test_single_cell.py",
    "--ignore=/repo/tests/test_frontend_isolated.py",
    "--ignore=/repo/tests/test_frontend_rangeslider.py",
    "--ignore=/repo/tests/wasm",
])
int(rc)
