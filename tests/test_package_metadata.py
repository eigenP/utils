"""Tests for the package surface: ``eigenp_utils.__init__`` (D06), the deprecations (D08)
and the PEP 723 headers of ``src/eigenp_utils/*.py`` (D09 = A)."""

import ast
import importlib
import importlib.abc
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

import eigenp_utils


# ---------------------------------------------------------------------------
# D06: lazy top-level API
# ---------------------------------------------------------------------------

DEPRECATED_TOP_LEVEL = ("hist_imshow", "numpy_to_stczyx_xarray")

# Names that were missing from __all__ (R8-12) and the Stage-2 additions.
NEW_PUBLIC = (
    "load_image", "set_channel_info", "to_zyx", "check_zyx_input",
    "savefig_svg", "set_plotting_style", "label_rgba", "LabelNorm",
    "labels_cmap_black", "make_labels_cmap", "PlottingStyle",
    "spline_length", "compute_drift_trajectory", "check_z_axis_orientation",
    "median_filter_fast", "clahe_u8", "correct_z_intensity_decay",
    "extract_surface", "extract_surface_tiled", "voronoi_otsu_labeling",
    "LabelAnnotator", "show_zyx_max_slice_interactive", "show_iso_scatter",
)


def test_all_lists_the_image_api_and_every_name_resolves():
    """D06: ``__all__`` lists the image API, and every listed name is an attribute.

    (+) the names R8-12 found missing and the Stage-2 additions are in ``__all__``;
        ``getattr`` resolves each of them (all extras are installed here).
    (-) deprecated names are NOT in ``__all__`` but stay importable;
        ``test_z_axis_orientation`` (pytest would collect it) is not exported at all.
    """
    missing = [n for n in NEW_PUBLIC if n not in eigenp_utils.__all__]
    assert not missing
    assert len(set(eigenp_utils.__all__)) == len(eigenp_utils.__all__)
    for name in eigenp_utils.__all__:
        assert getattr(eigenp_utils, name) is not None, name
    for name in DEPRECATED_TOP_LEVEL:
        assert name not in eigenp_utils.__all__
        assert callable(getattr(eigenp_utils, name))
    assert "test_z_axis_orientation" not in eigenp_utils.__all__
    assert not hasattr(eigenp_utils, "test_z_axis_orientation")
    assert set(DEPRECATED_TOP_LEVEL) <= set(dir(eigenp_utils))
    with pytest.raises(AttributeError, match="no attribute 'not_a_name'"):
        eigenp_utils.not_a_name


def test_top_level_names_are_the_module_objects():
    """D06: a lazily resolved name is the same object as in its module (no wrapper)."""
    from eigenp_utils import image_io, plotting_utils, maxproj_registration

    assert eigenp_utils.load_image is image_io.load_image
    assert eigenp_utils.label_rgba is plotting_utils.label_rgba
    assert eigenp_utils.estimate_drift is maxproj_registration.estimate_drift
    # the guard lives in the WASM-safe _conventions and is re-exported from image_io
    assert eigenp_utils.check_zyx_input is image_io.check_zyx_input


class _Raise(importlib.abc.MetaPathFinder):
    """Make importing one module raise a given ImportError."""

    def __init__(self, fullname, exc):
        self.fullname, self.exc = fullname, exc

    def find_spec(self, name, path, target=None):
        if name == self.fullname:
            raise self.exc
        return None


def _force_reimport(monkeypatch, module, attr):
    """Drop the cached submodule and the cached top-level name (restored on teardown)."""
    monkeypatch.delitem(sys.modules, f"eigenp_utils.{module}", raising=False)
    monkeypatch.delitem(vars(eigenp_utils), attr, raising=False)
    monkeypatch.delitem(vars(eigenp_utils), module, raising=False)


def test_missing_optional_dependency_gives_an_install_hint(monkeypatch):
    """D06: a missing optional package becomes an ImportError naming the extra.

    Strategy: pretend scikit-image is not installed (``_is_available`` False and
    the module import raising ModuleNotFoundError for 'skimage').
    (+) ``eigenp_utils.estimate_drift`` raises ImportError mentioning
        ``eigenp-utils[image-analysis]``, chained to the original error.
    (-) core names (``show_zyx``) still resolve.
    """
    real = eigenp_utils._is_available
    monkeypatch.setattr(eigenp_utils, "_is_available", lambda pkg: pkg != "skimage" and real(pkg))
    _force_reimport(monkeypatch, "maxproj_registration", "estimate_drift")
    exc = ModuleNotFoundError("No module named 'skimage'", name="skimage")
    monkeypatch.setattr(sys, "meta_path", [_Raise("eigenp_utils.maxproj_registration", exc)] + sys.meta_path)

    with pytest.raises(ImportError, match=r"eigenp-utils\[image-analysis\]") as info:
        eigenp_utils.estimate_drift
    assert info.value.__cause__ is exc
    assert "estimate_drift" in str(info.value)
    # the hint names the distribution to install, not the import name
    assert "needs scikit-image" in str(info.value)
    assert callable(eigenp_utils.show_zyx)

    # a submodule accessed as an attribute gets the same hint (_force_reimport dropped it)
    with pytest.raises(ImportError, match=r"eigenp-utils\[image-analysis\]"):
        eigenp_utils.maxproj_registration


def test_submodules_are_attributes_after_a_bare_import(monkeypatch):
    """D06 compat: ``eigenp_utils.image_io`` etc. resolve as attributes, as with the
    old eager imports, even when nothing has imported the submodule by name yet.

    Strategy: remove the submodule attribute the import system bound on the package
    (and, for one module, the sys.modules entry), then access it.
    (+) ``eigenp_utils.image_io`` / ``tnia_plotting_anywidgets`` / ``label_annotator``
        are the submodule objects; a dropped module is imported afresh.
    (-) a name that is neither a public name nor a submodule raises AttributeError;
        dunder lookups do not try to import anything.
    """
    import eigenp_utils.image_io as image_io_mod

    for name in ("image_io", "tnia_plotting_anywidgets", "label_annotator", "maxproj_registration"):
        monkeypatch.delattr(eigenp_utils, name, raising=False)
        mod = getattr(eigenp_utils, name)
        assert mod is sys.modules[f"eigenp_utils.{name}"]
    assert eigenp_utils.image_io is image_io_mod

    _force_reimport(monkeypatch, "spline_utils", "spline_length")
    monkeypatch.delattr(eigenp_utils, "spline_utils", raising=False)
    assert eigenp_utils.spline_utils.spline_length is sys.modules["eigenp_utils.spline_utils"].spline_length

    with pytest.raises(AttributeError):
        eigenp_utils.no_such_module_or_name
    with pytest.raises(AttributeError):
        eigenp_utils.__wrapped__


def test_real_import_errors_propagate_unchanged(monkeypatch):
    """D06 / R8-12: a genuine import bug is not hidden or turned into an install hint.

    Strategy: scikit-image IS installed, but importing the module fails the way a
    moved private skimage helper would (``cannot import name '_upsampled_dft'``).
    (+) the original ImportError propagates (same object, no 'pip install' hint).
    (-) a ModuleNotFoundError for a NON-optional package propagates as well.
    """
    _force_reimport(monkeypatch, "maxproj_registration", "estimate_drift")
    bug = ImportError("cannot import name '_upsampled_dft' from 'skimage.registration'",
                      name="skimage.registration")
    monkeypatch.setattr(sys, "meta_path", [_Raise("eigenp_utils.maxproj_registration", bug)] + sys.meta_path)
    with pytest.raises(ImportError) as info:
        eigenp_utils.estimate_drift
    assert info.value is bug

    _force_reimport(monkeypatch, "spline_utils", "spline_length")
    other = ModuleNotFoundError("No module named 'scipy'", name="scipy")
    monkeypatch.setattr(sys, "meta_path", [_Raise("eigenp_utils.spline_utils", other)] + sys.meta_path)
    with pytest.raises(ModuleNotFoundError) as info:
        eigenp_utils.spline_length
    assert info.value is other


@pytest.mark.skipif(sys.platform == "emscripten", reason="no subprocesses in Pyodide")
def test_import_is_lazy_and_works_without_scikit_image():
    """D06: ``import eigenp_utils`` imports no optional package; without scikit-image the
    image API raises an install hint while the core viewers work.

    (+) after the import, skimage / xarray / tifffile / scipy.ndimage are not loaded;
        with skimage blocked, ``eigenp_utils.estimate_drift`` raises ImportError
        naming ``eigenp-utils[image-analysis]``.
    (-) ``show_zyx`` and ``clahe_u8`` (no scikit-image needed) still resolve.
    """
    code = textwrap.dedent("""
        import sys, importlib.abc
        import eigenp_utils
        loaded = [m for m in ("skimage", "xarray", "tifffile", "eigenp_utils.image_io")
                  if m in sys.modules]
        assert not loaded, loaded
        # as before the lazy __init__: the package's colormaps are usable by name
        import matplotlib
        assert {"labels_cmap", "labels_cmap_black", "i_blue"} <= set(matplotlib.colormaps)

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "skimage" or name.startswith("skimage."):
                    raise ImportError("skimage blocked for test")
        sys.meta_path.insert(0, Block())
        assert callable(eigenp_utils.show_zyx)
        assert callable(eigenp_utils.clahe_u8)
        try:
            eigenp_utils.estimate_drift
        except ImportError as e:
            assert "eigenp-utils[image-analysis]" in str(e), str(e)
        else:
            raise AssertionError("no ImportError")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# ---------------------------------------------------------------------------
# D08: every legacy name emits FutureWarning naming 0.1.0
# ---------------------------------------------------------------------------

def _vol():
    import numpy as np

    rng = np.random.default_rng(0)
    return (rng.random((6, 24, 24)) * 1000).astype(np.uint16)


def _hist_imshow():
    import matplotlib.pyplot as plt
    from eigenp_utils.plotting_utils import hist_imshow

    import numpy as np
    try:
        hist_imshow(np.arange(16.0).reshape(4, 4))
    finally:
        plt.close("all")


def _savefig_without_fig(tmp_path):
    import matplotlib.pyplot as plt
    from eigenp_utils.plotting_utils import savefig_svg

    fig, ax = plt.subplots()
    try:
        savefig_svg(str(tmp_path / "x.svg"))
    finally:
        plt.close(fig)


def _tnia_plotting_3d():
    sys.modules.pop("eigenp_utils.tnia_plotting_3d", None)
    importlib.import_module("eigenp_utils.tnia_plotting_3d")


def _annotate_beta():
    pytest.importorskip("scanpy")
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from eigenp_utils.single_cell import annotate_clusters_by_markers

    X = np.array([[10.0, 0, 1], [0, 10.0, 1], [9.0, 0, 2], [8.0, 0, 1]])
    adata = AnnData(X=X, obs=pd.DataFrame({"leiden": ["0", "1", "0", "0"]}, index=list("abcd")),
                    var=pd.DataFrame(index=["CD3E", "CD19", "GAPDH"]))
    adata.raw = adata
    annotate_clusters_by_markers(adata, cluster_key="leiden", beta=2.0, score_method="binned",
                                 cell_type_markers_dict={"T": ["CD3E"], "B": ["CD19"]})


def _call(module, func, *args, **kwargs):
    def run(tmp_path):
        f = getattr(importlib.import_module(f"eigenp_utils.{module}"), func)
        a = [x() if callable(x) else x for x in args]
        return f(*a, **kwargs)
    return run


def _drift():
    import numpy as np
    from eigenp_utils.maxproj_registration import estimate_drift

    a = _vol()[0].astype(float)
    estimate_drift(a, np.roll(a, 1, axis=0), return_ccm=True, order="xy")


def _spline_resolution():
    import numpy as np
    from eigenp_utils.spline_utils import calculate_spline_length

    calculate_spline_length(np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                            resolution=(1.0, 1.0, 1.0))


def _bicubic():
    import numpy as np
    from eigenp_utils.image_and_labels_utils import sample_volume_plane

    sample_volume_plane(_vol().astype(float), pixel_sizes=(1, 1, 1),
                        p0_phys=np.array([3.0, 12.0, 12.0]), normal_phys=np.array([1.0, 0, 0]),
                        u_range_um=(-2, 2), v_range_um=(-2, 2), u_res=4, v_res=4, method="bicubic")


def _widget(**kw):
    def run(tmp_path):
        import matplotlib.pyplot as plt
        from eigenp_utils.tnia_plotting_anywidgets import show_zyx_max_slice_interactive

        try:
            show_zyx_max_slice_interactive(_vol(), **kw)
        finally:
            plt.close("all")
    return run


def _iso_scatter(tmp_path):
    import numpy as np
    from eigenp_utils.tnia_plotting_anywidgets import show_iso_scatter

    x = np.arange(5.0)
    show_iso_scatter(x, x, x, sxy=1.0, sz=2.0)


def _drift_default_order():
    import numpy as np
    from eigenp_utils.maxproj_registration import estimate_drift

    a = _vol()[0].astype(float)
    estimate_drift(a, np.roll(a, 1, axis=0))


def _slice_widget_colors(tmp_path):
    from eigenp_utils.tnia_plotting_anywidgets import TNIASliceWidget

    TNIASliceWidget(_vol(), colors=["lime"])


def _scatter(**kw):
    def run(tmp_path):
        import numpy as np
        from eigenp_utils.tnia_plotting_anywidgets import TNIAScatterWidget

        x = np.linspace(0.0, 10.0, 20)
        TNIAScatterWidget(x, x, x, **kw)
    return run


LEGACY = {
    # Viewers
    "colors=": _widget(colors=["lime"]),
    "sxy/sz": _widget(sxy=0.5, sz=1.0),
    "x_s..z_s": _widget(x_s=3),
    "x_t..z_t": _widget(x_t=3),
    "IsoScatter sxy/sz": _iso_scatter,
    "TNIASliceWidget(colors=)": _slice_widget_colors,
    "TNIAScatterWidget x_s..z_s": _scatter(x_s=3.0),
    "TNIAScatterWidget x_t..z_t": _scatter(x_t=2.0),
    "tnia_plotting_3d": lambda tmp_path: _tnia_plotting_3d(),
    # Plotting / I/O
    "hist_imshow": lambda tmp_path: _hist_imshow(),
    "numpy_to_stczyx_xarray": _call("image_io", "numpy_to_stczyx_xarray", _vol, "ZYX"),
    "savefig_svg without fig": _savefig_without_fig,
    # Intensity
    "_my_clahe_": _call("clahe_equalize_adapthist", "_my_clahe_", lambda: _vol()[0]),
    "adjust_gamma_per_slice": _call("intensity_rescaling", "adjust_gamma_per_slice", _vol),
    "adjust_brightness_per_slice": _call("intensity_rescaling", "adjust_brightness_per_slice", _vol),
    "test_z_axis_orientation": _call("intensity_rescaling", "test_z_axis_orientation", _vol),
    "contrast_stretching(p_max=)": _call("intensity_rescaling", "contrast_stretching", _vol, p_max=99.0),
    "normalize_image(upper_percentile=)": _call("intensity_rescaling", "normalize_image", _vol,
                                                upper_percentile=99.0),
    "contrast_stretching(p_min=)": _call("intensity_rescaling", "contrast_stretching", _vol, p_min=1.0),
    "normalize_image(lower_percentile=)": _call("intensity_rescaling", "normalize_image", _vol,
                                                lower_percentile=1.0),
    "normalize_image(dtype=)": _call("intensity_rescaling", "normalize_image", _vol, dtype="float32"),
    "contrast_stretch_per_slice(p_min_array=)": _call(
        "intensity_rescaling", "contrast_stretch_per_slice", _vol,
        p_min_array=[0] * 6, p_max_array=[1000] * 6),
    "contrast_stretch_per_slice(FLIP_Z_AXIS=)": _call(
        "intensity_rescaling", "contrast_stretch_per_slice", _vol,
        in_min=[0] * 6, in_max=[1000] * 6, FLIP_Z_AXIS=True),
    # Geometry
    "dask_extract_surface": _call("surface_extraction", "dask_extract_surface"),
    "best_focus_image(test=)": _call("extended_depth_of_focus", "best_focus_image", _vol, test=True),
    "estimate_drift(return_ccm=)": lambda tmp_path: _drift(),
    "estimate_drift default order ('xy')": lambda tmp_path: _drift_default_order(),
    "apply_subpixel_drift_correction(drift=)": _call(
        "maxproj_registration", "apply_subpixel_drift_correction", _vol, drift=(0.0, 0.0, 0.0)),
    "spacing=": _call("image_and_labels_utils", "voronoi_otsu_labeling", _vol, spacing=(1, 1, 1)),
    "calculate_spline_length(resolution=)": lambda tmp_path: _spline_resolution(),
    "interpolation='bicubic'": lambda tmp_path: _bicubic(),
    # Single cell
    "annotate_clusters_by_markers(beta=)": lambda tmp_path: _annotate_beta(),
}


# deprecated on import: the warning points at the importing line (here importlib,
# because the test imports through importlib.import_module)
IMPORT_LEGACY = {"tnia_plotting_3d"}


@pytest.mark.parametrize("name", list(LEGACY))
def test_every_legacy_name_emits_future_warning_naming_0_1_0(name, tmp_path):
    """D08: each legacy name in the plan's table emits a FutureWarning through
    ``warn_deprecated``: "... will be removed in eigenp_utils 0.1.0 ...".

    (+) the call emits such a FutureWarning (not only a DeprecationWarning, which
        Python hides when attributed to library code).
    (-) the warning is attributed to this test file or the importing line, not
        to the library frame that emitted it (stacklevel is right) -- checked
        for the plain function calls, where the caller is this file.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        LEGACY[name](tmp_path)
    ours = [w for w in rec if issubclass(w.category, FutureWarning)
            and "eigenp_utils 0.1.0" in str(w.message)]
    assert ours, [f"{w.category.__name__}: {w.message}" for w in rec]
    assert not any(issubclass(w.category, DeprecationWarning) and "eigenp" in w.filename
                   for w in rec if "deprecated" in str(w.message))
    if name not in IMPORT_LEGACY:
        # stacklevel: attributed to the caller (this file), not to eigenp_utils
        assert all(Path(w.filename).name == Path(__file__).name for w in ours), \
            [(w.filename, w.lineno, str(w.message)) for w in ours]


# ---------------------------------------------------------------------------
# D07: output policy
# ---------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "eigenp_utils"
# a marimo notebook kept in src/ (user decision: leave it alone)
NOT_LIBRARY = {"task_calendar_scheduler.py"}
ROOT_LOGGER_CALLS = {"basicConfig", "debug", "info", "warning", "error", "exception", "critical", "log"}


def _library_modules():
    return [p for p in sorted(SRC.glob("*.py")) if p.name not in NOT_LIBRARY]


def test_library_code_neither_prints_nor_uses_the_root_logger():
    """D07 / R8-13 / R7-09: no ``print`` and no root-logger call in library code.

    (+) every ``print(...)`` call and every ``logging.<level>()`` /
        ``logging.basicConfig()`` / ``logging.getLogger()`` (no name) call is reported.
    (-) module loggers (``logging.getLogger(__name__)``) and ``logger.info`` pass;
        ``print`` inside docstrings (examples) is not a call and passes.
    """
    offenders = []
    for path in _library_modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Name) and f.id == "print":
                offenders.append(f"{path.name}:{node.lineno} print")
            elif (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                  and f.value.id == "logging"):
                if f.attr in ROOT_LOGGER_CALLS or (f.attr == "getLogger" and not node.args):
                    offenders.append(f"{path.name}:{node.lineno} logging.{f.attr}")
    assert not offenders, offenders


# functions whose printed output moved to the module logger or to UserWarnings (D07)
D07_CHANGED = {
    "single_cell.py": (
        "pflogpf", "import_obs_to_adata_from_csv", "plot_marker_genes_dict_on_embedding",
        "check_gene_adata", "find_expression_archetypes", "run_triku", "save_pacmap_model",
        "load_pacmap_model", "tl_pacmap", "calculate_lineage_coupling", "multiscale_coarsening",
        "filter_markers_to_adata", "filter_markers_by_moran", "sweep_leiden_and_annotate",
        "export_obs_from_adata_to_csv", "plot_volcano_adata", "compute_kknn_neighbors",
        "_project_onto_reference_pca", "kknn_ingest",
    ),
    "bonsai_tree.py": ("run_bonsai", "tl_bonsai", "bonsai_neighbors"),
}


def test_functions_that_stopped_printing_document_it():
    """D07 / COMMON 'Changed:' rule: each function whose output stopped being printed
    says so in its docstring (verbose=True output now needs logging configured).

    Strategy: AST-read the module (no scanpy import needed).
    (+) every listed function has a 'Changed (D07' note in its docstring.
    (-) the listed functions still exist (a rename would make the check vacuous).
    """
    missing = []
    for fname, funcs in D07_CHANGED.items():
        tree = ast.parse((SRC / fname).read_text(encoding="utf-8"))
        docs = {n.name: ast.get_docstring(n) or "" for n in tree.body
                if isinstance(n, ast.FunctionDef)}
        assert set(funcs) <= set(docs), set(funcs) - set(docs)
        missing += [f"{fname}:{f}" for f in funcs if "Changed (D07" not in docs[f]]
    assert not missing, missing


# ---------------------------------------------------------------------------
# D09 = A: PEP 723 headers in src/ are complete and agree with pyproject
# ---------------------------------------------------------------------------

PYPROJECT = REPO / "pyproject.toml"
NOTEBOOKS_DIR = REPO / "notebooks"
HEADER_RE = re.compile(r"^# /// script\s*$(.*?)^# ///\s*$", re.M | re.S)

# import name -> pyproject distribution name, where they differ
IMPORT_TO_DIST = {
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "umap": "umap-learn",
    "ruamel": "ruamel.yaml",
    "yaml": "pyyaml",
    "mpl_toolkits": "matplotlib",
    "PIL": "pillow",
}
# provided by the platform, never pip-installed
PLATFORM_MODULES = {"pyodide", "js"}
# PEP 723 headers whose dependency list is not checked against pyproject
# (a marimo notebook kept in src/: user decision, it keeps plotly)
HEADER_SUBSET_EXEMPT = {"task_calendar_scheduler.py"}


def _norm(dist):
    """PEP 503 normalised distribution name, without extras, versions or markers."""
    name = re.split(r"[\s;\[<>=!~@]", dist.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _pyproject():
    if not PYPROJECT.exists():
        pytest.skip("pyproject.toml not available (installed package)")
    import tomllib

    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _declared_dists():
    proj = _pyproject()
    names = {_norm(d) for d in proj["dependencies"]}
    for extra in proj["optional-dependencies"].values():
        names |= {_norm(d) for d in extra if not d.startswith("eigenp_utils[")}
    return names


def _header(path):
    """(requires_python, [dependencies]) of a PEP 723 header, or None."""
    import tomllib

    m = HEADER_RE.search(path.read_text(encoding="utf-8"))
    if m is None:
        return None
    body = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                     for line in m.group(1).splitlines())
    data = tomllib.loads(body)
    return data.get("requires-python"), data.get("dependencies", [])


def _guards_import_error(handler):
    names = []
    t = handler.type
    for n in (t.elts if isinstance(t, ast.Tuple) else [t]):
        if isinstance(n, ast.Name):
            names.append(n.id)
    return t is None or bool({"ImportError", "ModuleNotFoundError", "Exception"} & set(names))


def _required_imports(path):
    """Top-level names of the third-party packages a module imports unconditionally.

    Scans every import (module level and inside functions). An import inside a
    ``try`` whose handler catches ImportError is optional and not required.
    ``_LazyModule("name")`` (single_cell's lazy scanpy/anndata) counts as an import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    optional = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(_guards_import_error(h) for h in node.handlers):
            for stmt in node.body:
                optional |= {id(n) for n in ast.walk(stmt)}
    found = set()
    for node in ast.walk(tree):
        if id(node) in optional:
            continue
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            mods = [node.module]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "_LazyModule" and node.args
              and isinstance(node.args[0], ast.Constant)):
            mods = [node.args[0].value]
        else:
            continue
        for m in mods:
            top = m.split(".")[0]
            if top not in sys.stdlib_module_names and top not in PLATFORM_MODULES \
                    and top != "eigenp_utils" and top != "__future__":
                found.add(top)
    return found


def _src_modules():
    if not SRC.exists() or not (SRC / "__init__.py").exists():
        pytest.skip("src/ not available")
    return sorted(SRC.glob("*.py"))


def test_header_parser_and_import_scanner_controls():
    """Controls for the D09 checker itself.

    (+) a header missing a dependency is detected; a function-level import counts;
        ``_LazyModule("scanpy")`` counts.
    (-) an import guarded by ``try/except ImportError`` is optional; stdlib,
        relative and platform (pyodide) imports are ignored.
    """
    import tempfile

    src = textwrap.dedent('''\
        # /// script
        # requires-python = ">=3.11"
        # dependencies = [
        #     "numpy",  # comment
        # ]
        # ///
        import os
        import numpy as np
        from . import _conventions
        sc = _LazyModule("scanpy")
        def f():
            from skimage import filters
            from pyodide.http import pyfetch
            try:
                import numba
            except ImportError:
                numba = None
    ''')
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.py"
        p.write_text(src, encoding="utf-8")
        assert _header(p) == (">=3.11", ["numpy"])
        assert _required_imports(p) == {"numpy", "scanpy", "skimage"}
    assert _norm("eigenp-utils @ git+https://x ; sys_platform != 'emscripten'") == "eigenp-utils"
    assert _norm("ruamel.yaml>=0.17") == "ruamel-yaml"


def test_src_pep723_headers_list_every_third_party_import():
    """D09 = A (R8-21): every src module with third-party imports has a PEP 723 header that

    (+) lists each unconditional third-party import under its pyproject distribution
        name (AST scan, including function-level imports; optional imports guarded
        by ``try/except ImportError`` may be omitted);
    (+) declares ``requires-python`` equal to pyproject's;
    (-) lists only distributions that pyproject declares (core or an extra), so a
        header cannot drift from the package metadata (task_calendar_scheduler, a
        marimo notebook kept in src/, is exempt from this subset check only).
    """
    requires = _pyproject()["requires-python"]
    declared = _declared_dists()
    problems = []
    for path in _src_modules():
        needed = {_norm(IMPORT_TO_DIST.get(m, m)) for m in _required_imports(path)}
        header = _header(path)
        if header is None:
            if needed:
                problems.append(f"{path.name}: no PEP 723 header (imports {sorted(needed)})")
            continue
        rp, deps = header
        listed = {_norm(d) for d in deps}
        if rp != requires:
            problems.append(f"{path.name}: requires-python {rp!r} != pyproject {requires!r}")
        if needed - listed:
            problems.append(f"{path.name}: header misses {sorted(needed - listed)}")
        if path.name not in HEADER_SUBSET_EXEMPT and listed - declared:
            problems.append(f"{path.name}: header lists {sorted(listed - declared)}, not in pyproject")
    assert not problems, "\n".join(problems)


def test_notebook_headers_require_the_package_python_without_pins():
    """D09 / R8-21: notebook headers use pyproject's requires-python and no hard pins.

    (+) every ``notebooks/*.py`` with a header says requires-python == pyproject's.
    (-) no dependency is pinned with ``==`` (color_projection_clahe_demo had
        matplotlib==3.10.6, numpy==2.2.6, scikit-image==0.25.2 and >=3.12).
    """
    if not NOTEBOOKS_DIR.exists():
        pytest.skip("notebooks/ not available")
    requires = _pyproject()["requires-python"]
    problems = []
    for path in sorted(NOTEBOOKS_DIR.glob("*.py")):
        header = _header(path)
        if header is None:
            continue
        rp, deps = header
        if rp != requires:
            problems.append(f"{path.name}: requires-python {rp!r}")
        problems += [f"{path.name}: pinned {d!r}" for d in deps if "==" in d]
    assert not problems, "\n".join(problems)
