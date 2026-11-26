"""Repo setup and demo notebook installation-guard tests."""
import ast
import importlib.util
import re
import tomllib
import types
from pathlib import Path

import pytest

NOTEBOOKS = sorted(
    p for p in (Path(__file__).parents[1] / "notebooks").glob("*.py")
    if "install_local(GIT_URL)" in p.read_text(encoding="utf-8")
)


def _local_checkout_fn(path):
    """Compile the `local_checkout` helper defined inside the notebook's setup cell."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "local_checkout":
            namespace = {}
            exec(compile(ast.Module([node], []), str(path), "exec"), namespace)
            return namespace["local_checkout"]
    raise AssertionError(f"{path.name}: setup cell has no local_checkout guard")


def test_notebooks_with_a_setup_cell_were_found():
    """Verify the repository contains the expected marimo demo notebooks with setup cells.

    Strategy: scan ``notebooks/*.py`` for setup cells invoking ``install_local``.
    Expected outcome: at least 9 demo notebooks are discovered (or skipped when
    running against an installed wheel without ``notebooks/``).
    """
    if not (Path(__file__).parents[1] / "notebooks").exists():
        pytest.skip("notebooks/ not available")
    assert len(NOTEBOOKS) >= 9


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_setup_cell_skips_install_for_a_local_checkout(path, monkeypatch):
    """Ensure every demo notebook's setup cell avoids overwriting a local checkout.

    Strategy: compile each notebook's ``local_checkout`` helper from its AST and
    probe editable/source paths, site-packages installs, missing specs, and the
    ``EIGENP_UTILS_NO_INSTALL`` opt-out env var, plus branch ordering after WASM.
    Expected outcome: returns True for source checkouts and explicit opt-outs,
    False for site-packages or uninstalled packages, and guards only local installs.
    """
    local_checkout = _local_checkout_fn(path)
    monkeypatch.delenv("EIGENP_UTILS_NO_INSTALL", raising=False)

    def spec_at(origin):
        return lambda name: types.SimpleNamespace(origin=origin)

    # editable install / source checkout: keep it
    monkeypatch.setattr(importlib.util, "find_spec", spec_at("/home/me/utils/src/eigenp_utils/__init__.py"))
    assert local_checkout() is True
    # regular install (e.g. molab): update from GitHub as before
    monkeypatch.setattr(importlib.util, "find_spec", spec_at("/env/lib/python3.12/site-packages/eigenp_utils/__init__.py"))
    assert local_checkout() is False
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert local_checkout() is False
    # explicit opt-out
    monkeypatch.setenv("EIGENP_UTILS_NO_INSTALL", "1")
    assert local_checkout() is True

    # the guard only applies to the local install, after the WASM (micropip) branch
    src = path.read_text(encoding="utf-8")
    assert "elif local_checkout():" in src
    assert src.index("micropip.install") < src.index("elif local_checkout():")


def test_dev_extra_includes_pytest_playwright():
    """Verify the ``dev`` optional dependency extra declares ``pytest-playwright``.

    Strategy: parse ``pyproject.toml`` and normalize each PEP 508 requirement name
    in ``[project.optional-dependencies].dev``; also verify no legacy
    ``requirements-dev.txt`` file exists.
    Expected outcome: ``pytest-playwright`` is declared in ``dev`` and
    ``requirements-dev.txt`` is absent.
    """
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not available (installed-package test run)")
    dev = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["optional-dependencies"]["dev"]
    pkg_names = {re.split(r"[<>=!~;\[\s]", d, maxsplit=1)[0].strip().lower() for d in dev}
    assert "pytest-playwright" in pkg_names
    assert not (pyproject.parent / "requirements-dev.txt").exists()


ALL_NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.py"))

# deprecated names (D08) that the demos must not use any more
DEPRECATED_NAMES = {
    "_my_clahe_", "adjust_brightness_per_slice", "adjust_gamma_per_slice", "hist_imshow",
    "plotly_scatter_3d", "plotly_scatter_3d_from_adata_obsm", "numpy_to_stczyx_xarray",
    "test_z_axis_orientation", "dask_extract_surface", "tnia_plotting_3d",
}
DEPRECATED_VIEWER_KWARGS = {"colors", "sxy", "sz", "x_s", "y_s", "z_s", "x_t", "y_t", "z_t"}


def _call_name(call):
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


@pytest.mark.parametrize("path", ALL_NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_use_no_deprecated_names(path):
    """R8-22 / D08: the demo notebooks show the current API only.

    Strategy: AST-scan the notebook code (markdown strings are not code, so a
    sentence mentioning ``_my_clahe_`` is fine).
    (+) no deprecated function / module name is imported, referenced or called, and
        no viewer (``show_*`` / ``TNIA*``) call passes a deprecated keyword.
    (-) a viewer called with ``pixel_sizes=`` gets ``slabs_thickness`` /
        ``slabs_position`` in µm, not a literal tuple of voxel counts (they are
        physical once pixel sizes are given).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in DEPRECATED_NAMES:
            bad.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in DEPRECATED_NAMES:
            bad.append(node.attr)
        elif isinstance(node, ast.alias) and node.name.split(".")[-1] in DEPRECATED_NAMES:
            bad.append(node.name)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] in DEPRECATED_NAMES:
            bad.append(node.module)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if not (name.startswith("show_") or name.startswith("TNIA")):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            bad += [f"{name}({k}=)" for k in DEPRECATED_VIEWER_KWARGS & set(kw)]
            if "pixel_sizes" in kw:
                bad += [f"{name}({k}=<voxel tuple> with pixel_sizes)"
                        for k in ("slabs_thickness", "slabs_position")
                        if isinstance(kw.get(k), (ast.Tuple, ast.List))]
    assert not bad, bad
