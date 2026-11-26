"""Tests for the package surface: ``eigenp_utils.__init__`` (D06), the deprecations (D08)
and the PEP 723 headers of ``src/eigenp_utils/*.py`` (D09 = A)."""

import importlib
import importlib.abc
import subprocess
import sys
import textwrap
import warnings

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
