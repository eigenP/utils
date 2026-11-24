"""clahe_u8 must work without scikit-image (minimal segmentation environments)."""

import importlib
import sys

import numpy as np

import eigenp_utils  # noqa: F401  (parent package must be loaded)


def test_clahe_u8_imports_and_runs_without_skimage(monkeypatch):
    """clahe_u8 imports and runs while every scikit-image import fails.

    Strategy: map skimage and its already-imported submodules to None in
    sys.modules (any `import skimage...` then raises ImportError), drop the
    module so it is imported afresh, and run the NumPy backend. In-process
    rather than in a subprocess, because Pyodide cannot start processes;
    monkeypatch restores sys.modules afterwards.
    Expected: uint8 output of the input shape.
    """
    for name in list(sys.modules):
        if name.split(".")[0] == "skimage":
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "skimage", None)
    mod = "eigenp_utils.clahe_equalize_adapthist"
    monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.delattr(sys.modules["eigenp_utils"], "clahe_equalize_adapthist", raising=False)
    clahe_u8 = importlib.import_module(mod).clahe_u8

    img = (np.arange(64 * 64) % 251).reshape(64, 64).astype(np.uint8)
    a = clahe_u8(img, kernel_size=(16, 16), clip_limit=0.01, backend="numpy")
    assert a.dtype == np.uint8 and a.shape == img.shape
