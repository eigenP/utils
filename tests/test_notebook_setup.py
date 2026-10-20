"""The demo notebooks' setup cell must not overwrite a developer's local checkout."""
import ast
import importlib.util
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
    if not (Path(__file__).parents[1] / "notebooks").exists():
        pytest.skip("notebooks/ not available")
    assert len(NOTEBOOKS) >= 9


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_setup_cell_skips_install_for_a_local_checkout(path, monkeypatch):
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
