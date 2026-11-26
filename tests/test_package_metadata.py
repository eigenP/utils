"""Tests for the package surface: ``eigenp_utils.__init__`` (D06), the deprecations (D08)
and the PEP 723 headers of ``src/eigenp_utils/*.py`` (D09 = A)."""

import ast
import importlib
import importlib.abc
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
    assert callable(eigenp_utils.show_zyx)


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


LEGACY = {
    # Viewers
    "colors=": _widget(colors=["lime"]),
    "sxy/sz": _widget(sxy=0.5, sz=1.0),
    "x_s..z_s": _widget(x_s=3),
    "x_t..z_t": _widget(x_t=3),
    "IsoScatter sxy/sz": _iso_scatter,
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
    "normalize_image(dtype=)": _call("intensity_rescaling", "normalize_image", _vol, dtype="float32"),
    "contrast_stretch_per_slice(p_min_array=)": _call(
        "intensity_rescaling", "contrast_stretch_per_slice", _vol,
        p_min_array=[0] * 6, p_max_array=[1000] * 6),
    # Geometry
    "dask_extract_surface": _call("surface_extraction", "dask_extract_surface"),
    "best_focus_image(test=)": _call("extended_depth_of_focus", "best_focus_image", _vol, test=True),
    "estimate_drift(return_ccm=)": lambda tmp_path: _drift(),
    "apply_subpixel_drift_correction(drift=)": _call(
        "maxproj_registration", "apply_subpixel_drift_correction", _vol, drift=(0.0, 0.0, 0.0)),
    "spacing=": _call("image_and_labels_utils", "voronoi_otsu_labeling", _vol, spacing=(1, 1, 1)),
    "calculate_spline_length(resolution=)": lambda tmp_path: _spline_resolution(),
    "interpolation='bicubic'": lambda tmp_path: _bicubic(),
    # Single cell
    "annotate_clusters_by_markers(beta=)": lambda tmp_path: _annotate_beta(),
}


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
            and "will be removed in eigenp_utils 0.1.0" in str(w.message)]
    assert ours, [f"{w.category.__name__}: {w.message}" for w in rec]
    assert not any(issubclass(w.category, DeprecationWarning) and "eigenp" in w.filename
                   for w in rec if "deprecated" in str(w.message))


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
