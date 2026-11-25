"""Tests for ``eigenp_utils._conventions``: units, pixel sizes, display ranges, colours, compositing rule."""

import sys
import warnings

import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import pytest

from eigenp_utils import _conventions as conv
from eigenp_utils._conventions import (
    PALETTE,
    PALETTE_MINIMUM,
    channel_color_hex,
    channel_colormap,
    channel_spec,
    default_colors,
    default_display_range,
    format_length,
    per_channel,
    resolve_blending,
    resolve_pixel_sizes,
    robust_range,
    sample_values,
    starts_at_white,
    to_um,
    warn_deprecated,
)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_to_um_converts_every_known_unit_and_rejects_unknown_ones():
    """Verify ``to_um`` converts nm/µm spellings/mm/cm/m/Å, warns for inch, and raises for unknown units.

    Strategy: convert 2.0 in each unit (including ImageJ's escaped micro sign
    and upper case), a list with a None entry, and an array.
    Expected outcome: exact µm values; inch converts with a UserWarning; 'frobs'
    and 'pixel' raise ValueError; unit None leaves the value unchanged.
    """
    cases = {"nm": 2e-3, "µm": 2.0, "um": 2.0, "micron": 2.0, "\\u00B5m": 2.0, "MM": 2e3,
             "cm": 2e4, "m": 2e6, "angstrom": 2e-4}
    for unit, expected in cases.items():
        assert to_um(2.0, unit) == pytest.approx(expected), unit
    assert to_um(2.0, None) == 2.0
    assert to_um([1.0, None], "nm") == [1e-3, None]
    np.testing.assert_allclose(to_um(np.array([1.0, 2.0]), "mm"), [1e3, 2e3])
    with pytest.warns(UserWarning, match="inch"):
        assert to_um(1.0, "inch") == pytest.approx(25400.0)
    for bad in ("frobs", "pixel"):
        with pytest.raises(ValueError, match="unknown length unit"):
            to_um(1.0, bad)


@pytest.mark.parametrize("value, expected", [
    (0.5, (500.0, "nm")), (0.002, (2.0, "nm")), (1.0, (1.0, "µm")), (20.0, (20.0, "µm")),
    (999.0, (999.0, "µm")), (1000.0, (1.0, "mm")), (2000.0, (2.0, "mm")), (0.0, (0.0, "µm")),
])
def test_format_length_picks_nm_um_mm_by_magnitude(value, expected):
    """Verify ``format_length`` switches to nm below 1 µm and to mm from 1000 µm.

    Strategy: format lengths around both thresholds.
    Expected outcome: (value, unit) pairs as listed; 0 stays in µm.
    """
    v, u = format_length(value)
    assert u == expected[1] and v == pytest.approx(expected[0])


# ---------------------------------------------------------------------------
# Pixel sizes
# ---------------------------------------------------------------------------

def test_resolve_pixel_sizes_accepts_dict_sequence_scalar_and_none():
    """Verify one parser for dict (any key case), sequence, scalar, 0-d arrays and None.

    Strategy: parse equivalent inputs in every accepted form, a ZYX triple for
    2D, and None.
    Expected outcome: identical (values, calibrated, unit); None gives all
    unknown; a scalar is isotropic.
    """
    ref = ((2.0, 0.5, 0.25), (True, True, True), "µm")
    assert resolve_pixel_sizes({"Z": 2.0, "Y": 0.5, "X": 0.25}) == ref
    assert resolve_pixel_sizes({"z": 2.0, "y": 0.5, "x": 0.25, "unit": "um"}) == ref
    assert resolve_pixel_sizes([2.0, 0.5, 0.25]) == ref
    assert resolve_pixel_sizes(np.array([2.0, 0.5, 0.25])) == ref
    assert resolve_pixel_sizes((np.array(2.0), 0.5, np.float32(0.25))) == ref
    assert resolve_pixel_sizes(0.5) == ((0.5, 0.5, 0.5), (True, True, True), "µm")
    assert resolve_pixel_sizes(None) == ((None, None, None), (False, False, False), "µm")
    assert resolve_pixel_sizes((2.0, 0.5, 0.25), 2) == ((0.5, 0.25), (True, True), "µm")
    assert resolve_pixel_sizes({"Y": 0.5, "X": 0.25}, 2) == ((0.5, 0.25), (True, True), "µm")
    assert all(isinstance(v, float) for v in resolve_pixel_sizes((np.array(2.0), 1, 1))[0])


def test_resolve_pixel_sizes_converts_units_to_um():
    """Verify known length units are converted to µm and unknown units are passed through.

    Strategy: parse nm, cm and escaped-micro dicts, and a dict with an unknown unit.
    Expected outcome: values in µm with unit 'µm'; the unknown unit keeps the
    raw values and is returned as given.
    """
    vals, cal, unit = resolve_pixel_sizes({"Z": 300.0, "Y": 100.0, "X": 100.0, "unit": "nm"})
    assert vals == pytest.approx((0.3, 0.1, 0.1)) and unit == "µm"
    vals, _, unit = resolve_pixel_sizes({"Z": None, "Y": 1e-4, "X": 1e-4, "unit": "cm"}, warn=False)
    assert vals == pytest.approx((1.0, 1.0, 1.0)) and unit == "µm"
    assert resolve_pixel_sizes({"Z": 1, "Y": 1, "X": 1, "unit": "pixel"}) == ((1.0, 1.0, 1.0), (True,) * 3, "pixel")
    with pytest.warns(UserWarning, match="inch"):
        resolve_pixel_sizes({"Y": 0.001, "X": 0.001, "unit": "inch"}, fill_z=False)


def test_resolve_pixel_sizes_fills_missing_z_from_finest_xy_only():
    """Verify a missing Z with calibrated X and Y takes the finest of them, with one warning.

    Strategy: Z None with Y/X known (+), Z None with only Y known (-), all
    None (-), and ``fill_z=False`` / ``warn=False``.
    Expected outcome: (+) Z = min(Y, X), calibrated False for Z, a warning
    naming axis Z; (-) Z stays None without that warning.
    """
    with pytest.warns(UserWarning, match="No value for axis Z.*0.25 µm"):
        vals, cal, _ = resolve_pixel_sizes({"Z": None, "Y": 0.5, "X": 0.25})
    assert vals == (0.25, 0.5, 0.25) and cal == (False, True, True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_pixel_sizes({"Z": None, "Y": 0.5, "X": None})[0] == (None, 0.5, None)
        assert resolve_pixel_sizes({"Z": None, "Y": None, "X": None})[0] == (None, None, None)
        assert resolve_pixel_sizes({"Y": 0.5, "X": 0.5}, fill_z=False)[0] == (None, 0.5, 0.5)
        assert resolve_pixel_sizes({"Y": 0.5, "X": 0.5}, warn=False)[0] == (0.5, 0.5, 0.5)


@pytest.mark.parametrize("bad", [
    {"Z": 0.0, "Y": 1.0, "X": 1.0}, {"Z": -1.0, "Y": 1, "X": 1}, (1.0, np.nan, 1.0), (1.0, 1.0, np.inf),
    (1.0, 1.0), (1.0, 1.0, 1.0, 1.0), "1.0", ("bad", 1.0, 1.0), 0.0,
])
def test_resolve_pixel_sizes_rejects_invalid_values(bad):
    """Verify zero, negative, non-finite, non-numeric and wrong-length pixel sizes raise ValueError.

    Strategy: parse each invalid specification.
    Expected outcome: ValueError.
    """
    with pytest.raises(ValueError):
        resolve_pixel_sizes(bad)


# ---------------------------------------------------------------------------
# Sampling and display ranges
# ---------------------------------------------------------------------------

def test_sample_values_is_deterministic_finite_and_exact_for_small_arrays():
    """Verify the fixed-seed sampler: whole array when small, random (not strided) indices when large.

    Strategy: (1) a small float array with NaN/inf; (2) a large array whose
    every other column is 1000 (a strided sample with an even step sees one
    column only, R1-20); (3) booleans.
    Expected outcome: (1) exactly the finite values; (2) same sample on two
    calls, both column populations present in about equal shares; (3) uint8.
    """
    a = np.array([[1.0, np.nan], [np.inf, 4.0]])
    np.testing.assert_array_equal(np.sort(sample_values(a)), [1.0, 4.0])

    big = np.zeros((1000, 2000), np.uint16)
    big[:, ::2] = 1000
    s1, s2 = sample_values(big, n=10_000), sample_values(big, n=10_000)
    np.testing.assert_array_equal(s1, s2)
    assert s1.size == 10_000 and 0.4 < np.mean(s1 == 1000) < 0.6
    assert sample_values(np.ones((3, 3), bool)).dtype == np.uint8


class _FakeDask:
    """Minimal dask-like array: ``blocks``/``numblocks``/``compute`` (counts full computes)."""

    def __init__(self, arr, chunk):
        self._arr, self._chunk = arr, chunk
        self.shape, self.dtype, self.size, self.ndim = arr.shape, arr.dtype, arr.size, arr.ndim
        self.numblocks = tuple(-(-s // c) for s, c in zip(arr.shape, chunk))
        self.full_computes = 0
        outer = self

        class _Blocks:
            def __getitem__(self, idx):
                sl = tuple(slice(i * c, (i + 1) * c) for i, c in zip(idx, chunk))
                return _Block(outer._arr[sl])

        self.blocks = _Blocks()

    def compute(self):
        self.full_computes += 1
        return self._arr


class _Block:
    def __init__(self, a):
        self._a = a

    def compute(self):
        return self._a


def test_sample_values_samples_lazy_arrays_chunkwise_without_full_compute():
    """Verify dask-like input is sampled from a fixed-seed choice of chunks, never computed whole.

    Strategy: wrap a (40, 100, 100) array in a dask-like object with (4, 100,
    100) chunks and sample 50_000 values; also via ``default_display_range``.
    Expected outcome: no full compute; deterministic sample of the requested
    size drawn from the data; a valid display range.
    """
    arr = np.random.default_rng(0).integers(0, 1000, (40, 100, 100)).astype(np.uint16)
    lazy = _FakeDask(arr, (4, 100, 100))
    s = sample_values(lazy, n=50_000)
    assert s.size == 50_000 and lazy.full_computes == 0
    np.testing.assert_array_equal(s, sample_values(lazy, n=50_000))
    assert set(np.unique(s)) <= set(np.unique(arr))
    lo, hi = default_display_range(lazy)
    assert lo == 0.0 and 900 < hi <= 1000 and lazy.full_computes == 0


def test_default_display_range_rules():
    """Verify vmin = 0 or the negative minimum, vmax = robust p99.99 with max fallback, and edge cases.

    Strategy: (+) a hot pixel at a 1e-5 fraction on a 12-bit volume with a
    blob; (+) signed data; (-) outlier-free data; sparse signal (< 0.01%);
    constant, all-NaN, empty and boolean input.
    Expected outcome: the hot pixel does not set vmax (vmax near the blob
    level); signed data give the data minimum (not clipped at 0); outlier-free
    data give vmax within 1 grey level of the max; sparse signal falls back
    to the max; constant zeros give (0, 1); all-NaN gives (0, 1); bool (0, 1).
    """
    rng = np.random.default_rng(0)
    vol = rng.poisson(100, (20, 100, 100)).astype(np.uint16)
    vol[8:12, 40:60, 40:60] += 300
    vol[0, 0, 0] = 4000  # hot pixel: 1 of 200_000 voxels
    lo, hi = default_display_range(vol)
    assert lo == 0.0 and 400 < hi < 500 and hi == int(hi)

    signed = rng.normal(0, 1, (10, 50, 50)).astype(np.float32)
    lo, hi = default_display_range(signed)
    assert lo == pytest.approx(float(signed.min())) and lo < -3

    ramp = np.arange(4096, dtype=np.uint16).reshape(16, 16, 16)  # no outliers
    lo, hi = default_display_range(ramp)
    assert lo == 0.0 and abs(hi - 4095) <= 1

    sparse = np.zeros((20, 100, 100), np.uint16)
    sparse[5, 5, 5] = 3000  # 0.0005% of the voxels
    assert default_display_range(sparse) == (0.0, 3000.0)

    assert default_display_range(np.zeros((4, 4))) == (0.0, 1.0)
    assert default_display_range(np.full((4, 4), 7.0)) == (0.0, 7.0)
    assert default_display_range(np.full((4, 4), np.nan)) == (0.0, 1.0)
    assert default_display_range(np.zeros((0,))) == (0.0, 1.0)
    assert default_display_range(np.ones((4, 4), bool)) == (0.0, 1.0)


def test_robust_range_falls_back_to_max_for_sparse_signal_and_warns():
    """Verify ``robust_range`` keeps percentiles, falls back to the max when degenerate (R6-02), and warns.

    Strategy: (+) 15 bright spots on a zero 200x200 image (0.04% foreground);
    (-) a ramp (non-degenerate percentiles, no warning); constant data;
    all-NaN data.
    Expected outcome: spots give (0, 1000) with a UserWarning; the ramp gives
    its exact (0.1, 99.9) percentiles silently; constant data give (v, v)
    silently; all-NaN gives (0, 1).
    """
    img = np.zeros((200, 200), np.float32)
    img[::50, ::15][:3, :5] = 1000.0
    with pytest.warns(UserWarning, match="percentiles"):
        assert robust_range(img) == (0.0, 1000.0)
    ramp = np.linspace(0, 1, 10_001)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert robust_range(ramp) == pytest.approx(tuple(np.percentile(ramp, [0.1, 99.9])))
        assert robust_range(np.full(5, 3.0)) == (3.0, 3.0)
        assert robust_range(np.full(5, np.nan)) == (0.0, 1.0)
    mostly_max = np.full(10_000, 5.0)
    mostly_max[0] = 1.0
    with pytest.warns(UserWarning):
        assert robust_range(mostly_max) == (1.0, 5.0)


# ---------------------------------------------------------------------------
# Colours and compositing
# ---------------------------------------------------------------------------

def test_channel_color_hex_resolves_names_tuples_and_registered_instances():
    """Verify the single colour resolver: colormap names win, tuples and registered instances work.

    Strategy: resolve CSS names, hex, RGB/RGBA tuples, colormap names
    ('viridis', 'gray', 'pink' which warns), the package's registered I-series
    instance, an unregistered instance and nonsense.
    Expected outcome: hex strings; 'gray' -> white end colour; 'pink' warns;
    registered instance == its name; unregistered instance -> TypeError;
    nonsense -> ValueError (None with strict=False).
    """
    from eigenp_utils.plotting_utils import i_blue

    assert channel_color_hex("red") == "#ff0000"
    assert channel_color_hex("#00FF00") == "#00ff00"
    assert channel_color_hex((0.0, 0.0, 1.0)) == "#0000ff"
    assert channel_color_hex((1.0, 0.0, 0.0, 0.5)) == "#ff0000"
    assert channel_color_hex("viridis") == "#fde725"
    assert channel_color_hex("gray") == "#ffffff"
    with pytest.warns(UserWarning, match="'pink' is both a colormap"):
        assert channel_color_hex("pink") == "#ffffff"
    assert channel_color_hex(i_blue) == channel_color_hex("I Blue")
    assert channel_spec(i_blue) == "I Blue"
    unregistered = mcolors.LinearSegmentedColormap.from_list("not_registered_xyz", ["black", "red"])
    with pytest.raises(TypeError, match="unregistered Colormap"):
        channel_color_hex(unregistered)
    with pytest.raises(ValueError):
        channel_color_hex("notacolor")
    assert channel_color_hex("notacolor", strict=False) is None
    assert channel_spec(np.array([1.0, 0.0, 0.0])) == "#ff0000"


def test_channel_colormap_ramps_plain_colours_and_returns_named_colormaps():
    """Verify ``channel_colormap``: colormap names as given, colours as black/white -> colour ramps.

    Strategy: evaluate the colormaps at 0 and 1.
    Expected outcome: 'viridis' is matplotlib's; 'red' runs black -> red, or
    white -> red with ``background='white'``.
    """
    assert channel_colormap("viridis")(0.5) == matplotlib.colormaps["viridis"](0.5)
    np.testing.assert_allclose(channel_colormap("red")(0.0)[:3], (0, 0, 0))
    np.testing.assert_allclose(channel_colormap("red")(1.0)[:3], (1, 0, 0))
    np.testing.assert_allclose(channel_colormap("red", background="white")(0.0)[:3], (1, 1, 1))
    with pytest.raises(ValueError):
        channel_colormap("red", background="grey")


def test_resolve_blending_rule():
    """Verify the ``blending=None`` rule (D04): additive for colours, maximum with a colormap, minimum when all start white.

    Strategy: colour-only, colour + dark colormap, all I-series, I-series +
    plain colour, I-series + viridis (mix), and an explicit alias.
    Expected outcome: 'additive', 'maximum', 'minimum', 'minimum', 'maximum'
    with a warning, and the canonical name of the alias.
    """
    import eigenp_utils.plotting_utils  # noqa: F401  (registers the I-series)

    assert starts_at_white("I Blue") and starts_at_white("Blues") and not starts_at_white("viridis")
    assert not starts_at_white("white")
    assert resolve_blending(["red", "lime"]) == "additive"
    assert resolve_blending(["viridis", "lime"]) == "maximum"
    assert resolve_blending(["I Blue", "I Bordeaux"]) == "minimum"
    assert resolve_blending(["I Blue", "red"]) == "minimum"
    with pytest.warns(UserWarning, match="start at white"):
        assert resolve_blending(["I Blue", "viridis"]) == "maximum"
    assert resolve_blending(["I Blue"], "max") == "maximum"
    with pytest.raises(ValueError, match="blending"):
        resolve_blending(["red"], "multiply")


def test_default_colors_pads_with_unused_palette_and_cycles():
    """Verify palette padding (unused colours first), cycling past 7 channels and the minimum palette.

    Strategy: pad None, one string, a list with a palette colour, 9 channels.
    Expected outcome: as listed; the palette constants are the D03 order.
    """
    assert PALETTE == ["white", "lime", "magenta", "yellow", "cyan", "red", "blue"]
    assert default_colors(None, 2) == ["white", "lime"]
    assert default_colors("magma", 3) == ["magma", "white", "lime"]
    assert default_colors(["lime"], 3) == ["lime", "white", "magenta"]
    assert default_colors(None, 9) == PALETTE + PALETTE[:2]
    assert default_colors(None, 2, "minimum") == PALETTE_MINIMUM[:2]


def test_per_channel_broadcasts_pads_and_rejects_long_lists():
    """Verify ``per_channel`` broadcast, padding with the default, and the error for long lists.

    Strategy: scalar, None, short list with None, ndarray, long list.
    Expected outcome: n values; ValueError naming the parameter for 3 entries
    for 2 channels.
    """
    assert per_channel(2.0, 3, "gamma") == [2.0, 2.0, 2.0]
    assert per_channel(None, 2, "vmin", default=0.0) == [0.0, 0.0]
    assert per_channel([1.0, None], 3, "vmax", default=5.0) == [1.0, 5.0, 5.0]
    assert per_channel(np.array([1.0, 2.0]), 2, "gamma") == [1.0, 2.0]
    with pytest.raises(ValueError, match="gamma has 3 entries for 2 channels"):
        per_channel([1, 1, 1], 2, "gamma")


def test_warn_deprecated_emits_future_warning_naming_the_removal_version():
    """Verify ``warn_deprecated`` emits a FutureWarning with the 0.1.0 removal text, pointing at the caller.

    Strategy: call it from a wrapper function (stacklevel 3 -> this test).
    Expected outcome: FutureWarning containing "will be removed in eigenp_utils
    0.1.0" and the replacement; its filename is this test file.
    """
    def legacy():
        warn_deprecated("legacy()", "modern()")

    with pytest.warns(FutureWarning, match="will be removed in eigenp_utils 0.1.0; use modern") as rec:
        legacy()
    assert rec[0].filename == __file__


def test_module_is_wasm_safe(monkeypatch):
    """Verify ``_conventions`` imports with scikit-image, scipy, xarray and tifffile blocked.

    Strategy: block the packages in ``sys.modules`` (any import then raises
    ImportError), drop ``_conventions`` and re-import it.
    Expected outcome: the import succeeds and the module works.
    """
    for name in ("skimage", "scipy", "xarray", "tifffile", "dask"):
        monkeypatch.setitem(sys.modules, name, None)
    # restored on teardown, so later tests keep the original module object
    monkeypatch.setattr(sys.modules["eigenp_utils"], "_conventions", conv)
    monkeypatch.delitem(sys.modules, "eigenp_utils._conventions")
    import importlib

    mod = importlib.import_module("eigenp_utils._conventions")
    assert mod.default_display_range(np.arange(10)) == (0.0, 9.0)
    assert mod.channel_color_hex("lime") == "#00ff00"
