import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from eigenp_utils.stats import (
    add_stat_annotations,
    bootstrap_ci,
    cohens_d,
    remove_outliers,
    robust_standardize,
    summary_stats,
)


def _bracket_lines(ax, start_idx):
    """Return 4-vertex bracket Line2D objects added to ``ax`` after ``start_idx``."""
    return [line for line in ax.lines[start_idx:] if len(line.get_xdata()) == 4]


# =========================================
# add_stat_annotations
# =========================================


def test_add_stat_annotations():
    """Verify annotation text, bracket placement, return values, orientation, hue, and order.

    Strategy: build deterministic, cleanly separated and identical groups so Welch's t-test
    yields known star labels ('****' vs 'ns'), then inspect both the rendered text and the
    4-vertex bracket coordinates across vertical, horizontal, hue/hue_order, and explicit
    order configurations.
    Expected: brackets span the exact plotted group coordinates, significance labels match
    the underlying distributions, and ``return_annotator=True`` returns an ``Annotator``.
    """
    pytest.importorskip("statannotations")
    sns = pytest.importorskip("seaborn")
    from statannotations.Annotator import Annotator

    base = np.array([0.0, 0.2, -0.2, 0.4, -0.4, 0.1, -0.1, 0.3, -0.3, 0.0])
    df = pd.DataFrame(
        {
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
            "value": np.concatenate([base, base + 5.0, base]),
        }
    )

    # 1. Vertical orientation: significant pair ('****'), null pair ('ns'), return_annotator
    fig, ax = plt.subplots()
    try:
        sns.boxplot(data=df, x="group", y="value", ax=ax)
        n_lines = len(ax.lines)
        ax_ret, annotator = add_stat_annotations(
            ax,
            df,
            [("A", "B"), ("A", "C")],
            x="group",
            y="value",
            test="t-test_welch",
            text_format="star",
            return_annotator=True,
            verbose=False,
        )
        assert ax_ret is ax
        assert isinstance(annotator, Annotator)
        texts = {t.get_text() for t in ax.texts}
        assert "****" in texts
        assert "ns" in texts
        brackets = _bracket_lines(ax, n_lines)
        assert len(brackets) == 2
        spanned_sets = [sorted({int(round(v)) for v in b.get_xdata()}) for b in brackets]
        assert [0, 1] in spanned_sets
        assert [0, 2] in spanned_sets
    finally:
        plt.close(fig)

    # 2. Horizontal orientation: categorical variable on y, value on x
    fig, ax = plt.subplots()
    try:
        sns.boxplot(data=df, x="value", y="group", ax=ax)
        y_labels = [t.get_text() for t in ax.get_yticklabels()]
        n_lines = len(ax.lines)
        ax_ret = add_stat_annotations(
            ax,
            df,
            [("A", "B")],
            x="value",
            y="group",
            test="t-test_welch",
            text_format="star",
            verbose=False,
        )
        assert ax_ret is ax
        assert any(t.get_text() == "****" for t in ax.texts)
        brackets = _bracket_lines(ax, n_lines)
        assert len(brackets) == 1
        spanned = sorted({y_labels[int(round(v))] for v in brackets[0].get_ydata()})
        assert spanned == ["A", "B"]
    finally:
        plt.close(fig)

    # 3. Explicit order controls bracket placement along the categorical axis
    fig, ax = plt.subplots()
    try:
        custom_order = ["C", "A", "B"]
        sns.boxplot(data=df, x="group", y="value", order=custom_order, ax=ax)
        n_lines = len(ax.lines)
        add_stat_annotations(
            ax,
            df,
            [("C", "B")],
            x="group",
            y="value",
            order=custom_order,
            test="t-test_welch",
            text_format="star",
            verbose=False,
        )
        assert any(t.get_text() == "****" for t in ax.texts)
        brackets = _bracket_lines(ax, n_lines)
        assert len(brackets) == 1
        assert sorted({int(round(v)) for v in brackets[0].get_xdata()}) == [0, 2]
    finally:
        plt.close(fig)

    # 4. hue and hue_order shift dodged bracket endpoints
    df_hue = pd.DataFrame(
        {
            "group": ["G1"] * 20 + ["G2"] * 20,
            "cond": (["H1"] * 10 + ["H2"] * 10) * 2,
            "value": np.concatenate([base, base + 5.0, base, base + 5.0]),
        }
    )
    bracket_centers = {}
    for hue_order in (["H1", "H2"], ["H2", "H1"]):
        fig, ax = plt.subplots()
        try:
            sns.boxplot(data=df_hue, x="group", y="value", hue="cond", hue_order=hue_order, ax=ax)
            n_lines = len(ax.lines)
            add_stat_annotations(
                ax,
                df_hue,
                [(("G1", "H1"), ("G2", "H1"))],
                x="group",
                y="value",
                hue="cond",
                hue_order=hue_order,
                test="t-test_welch",
                text_format="star",
                verbose=False,
            )
            assert any(t.get_text() == "ns" for t in ax.texts)
            brackets = _bracket_lines(ax, n_lines)
            assert len(brackets) == 1
            xs = sorted({float(v) for v in brackets[0].get_xdata()})
            assert len(xs) == 2
            bracket_centers[tuple(hue_order)] = xs
        finally:
            plt.close(fig)

    # With H1 first, H1 sits to the left of each tick (< 0 and < 1); reversed, to the right (> 0 and > 1)
    assert bracket_centers[("H1", "H2")][0] < 0.0 < bracket_centers[("H2", "H1")][0]
    assert bracket_centers[("H1", "H2")][1] < 1.0 < bracket_centers[("H2", "H1")][1]


def test_add_stat_annotations_follows_plot_order():
    """Brackets land on the named groups even when the plot reorders categories.

    raincloud_plot sorts groups alphabetically; statannotations assumed first-
    appearance order, so ('ctrl', 'drug') was drawn over alpha and ctrl. Expected:
    the bracket spans the x positions of 'ctrl' and 'drug' on the axis.
    """
    pytest.importorskip("statannotations")
    from eigenp_utils.plotting_utils import raincloud_plot

    rng = np.random.default_rng(0)
    d = pd.DataFrame(
        {
            "g": np.repeat(["ctrl", "drug", "alpha"], 20),
            "v": np.r_[rng.normal(0, 1, 20), rng.normal(3, 1, 20), rng.normal(0, 1, 20)],
        }
    )
    fig, ax = plt.subplots()
    try:
        raincloud_plot(d, x="g", y="v", ax=ax)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        n_lines = len(ax.lines)
        add_stat_annotations(ax, d, [("ctrl", "drug")], x="g", y="v", verbose=False)
        bracket = _bracket_lines(ax, n_lines)[-1]
        spanned = sorted({labels[int(round(v))] for v in bracket.get_xdata()})
        assert spanned == ["ctrl", "drug"]
    finally:
        plt.close(fig)


# =========================================
# cohens_d
# =========================================


def test_cohens_d():
    """Verify Cohen's d and Hedges' g against hand-calculated values and boundary cases.

    Strategy: evaluate two shifted 5-element samples with known pooled SD (sqrt(2.5))
    both with and without Hedges' correction, identical groups, and an undersized group.
    Expected: exact agreement to 4 decimals, 0.0 for identical groups, and NaN for n < 2.
    """
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([2, 3, 4, 5, 6])

    # mean1 = 3, mean2 = 4, var1 = var2 = 2.5 => d = -1 / sqrt(2.5) = -0.63245...
    d = cohens_d(group1, group2, correction=False)
    np.testing.assert_almost_equal(d, -0.63245, decimal=4)

    # Hedges' g with df = 8: J(8) ~ 0.9027 => g ~ -0.5709
    d_corrected = cohens_d(group1, group2)
    np.testing.assert_almost_equal(d_corrected, -0.5709, decimal=4)

    # Identical groups yield zero effect size
    np.testing.assert_almost_equal(cohens_d(group1, group1), 0.0)

    # Fewer than 2 observations in either group returns NaN
    assert np.isnan(cohens_d([1], [2, 3]))
    assert np.isnan(cohens_d([1, 2], [3]))


def test_cohens_d_constant_groups():
    """Zero-variance groups: 0 only if the means are equal, otherwise an unbounded effect."""
    assert cohens_d([1, 1, 1], [2, 2, 2]) == -np.inf
    assert cohens_d([2, 2, 2], [1, 1, 1]) == np.inf
    assert cohens_d([2, 2, 2], [2, 2, 2]) == 0.0


@pytest.mark.parametrize("correction", [False, True])
def test_cohens_d_unequal_sizes_and_invariants(correction):
    """Verify Cohen's d / Hedges' g for unequal sample sizes and algebraic invariants.

    Strategy: compare unequal-sized samples (n1=4, n2=6) against the closed-form pooled
    variance and exact Hedges' J(df) factor, then check antisymmetry under group swap,
    translation invariance, positive scale invariance, and sign flip under negation.
    Expected: analytic match to floating-point precision and all invariants hold.
    """
    from scipy.special import gammaln

    g1 = np.array([2.0, 4.0, 6.0, 8.0])
    g2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    n1, n2 = len(g1), len(g2)
    df = n1 + n2 - 2
    s_pooled = np.sqrt(((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1)) / df)
    expected = (np.mean(g1) - np.mean(g2)) / s_pooled
    if correction:
        expected *= np.exp(gammaln(df / 2.0) - 0.5 * np.log(df / 2.0) - gammaln((df - 1) / 2.0))

    val = cohens_d(g1, g2, correction=correction)
    np.testing.assert_allclose(val, expected, rtol=1e-12)

    # Antisymmetry under group swap
    np.testing.assert_allclose(cohens_d(g2, g1, correction=correction), -val, rtol=1e-12)
    # Translation invariance
    np.testing.assert_allclose(cohens_d(g1 + 17.5, g2 + 17.5, correction=correction), val, rtol=1e-12)
    # Positive scale invariance and negation sign flip
    np.testing.assert_allclose(cohens_d(3.5 * g1, 3.5 * g2, correction=correction), val, rtol=1e-12)
    np.testing.assert_allclose(cohens_d(-g1, -g2, correction=correction), -val, rtol=1e-12)


# =========================================
# summary_stats
# =========================================


def test_summary_stats():
    """Verify grouped summary statistics including std, sem, multi-column grouping, and edge cases.

    Strategy: compute summary_stats on exact numeric groups, multi-column keys, groups with
    NaNs, a single-observation group, and an all-NaN group.
    Expected: all columns (count, mean, median, std, sem, min, max) match analytic values,
    with NaN std/sem for single-observation groups and count=0 with NaN stats for all-NaN groups.
    """
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )

    summary = summary_stats(df, group_by="group", value_col="value")
    assert len(summary) == 2
    assert list(summary.columns) == ["group", "count", "mean", "median", "std", "sem", "min", "max"]

    a_stats = summary[summary["group"] == "A"].iloc[0]
    assert a_stats["count"] == 3
    assert a_stats["mean"] == pytest.approx(2.0)
    assert a_stats["median"] == pytest.approx(2.0)
    assert a_stats["std"] == pytest.approx(1.0)
    assert a_stats["sem"] == pytest.approx(1.0 / np.sqrt(3))
    assert a_stats["min"] == pytest.approx(1.0)
    assert a_stats["max"] == pytest.approx(3.0)

    b_stats = summary[summary["group"] == "B"].iloc[0]
    assert b_stats["count"] == 3
    assert b_stats["mean"] == pytest.approx(20.0)
    assert b_stats["median"] == pytest.approx(20.0)
    assert b_stats["std"] == pytest.approx(10.0)
    assert b_stats["sem"] == pytest.approx(10.0 / np.sqrt(3))
    assert b_stats["min"] == pytest.approx(10.0)
    assert b_stats["max"] == pytest.approx(30.0)

    # Multi-column group_by + NaN-containing, single-observation, and all-NaN groups
    df_multi = pd.DataFrame(
        {
            "g1": ["X", "X", "X", "X", "Y", "Z", "Z"],
            "g2": ["p", "p", "p", "p", "q", "r", "r"],
            "val": [2.0, 4.0, 6.0, np.nan, 7.0, np.nan, np.nan],
        }
    )
    multi = summary_stats(df_multi, group_by=["g1", "g2"], value_col="val")
    assert list(multi.columns[:2]) == ["g1", "g2"]
    assert len(multi) == 3

    xp = multi[(multi["g1"] == "X") & (multi["g2"] == "p")].iloc[0]
    assert xp["count"] == 3  # NaN excluded from count and sem denominator
    assert xp["mean"] == pytest.approx(4.0)
    assert xp["std"] == pytest.approx(2.0)
    assert xp["sem"] == pytest.approx(2.0 / np.sqrt(3))

    yq = multi[(multi["g1"] == "Y") & (multi["g2"] == "q")].iloc[0]
    assert yq["count"] == 1
    assert yq["mean"] == pytest.approx(7.0)
    assert yq["median"] == pytest.approx(7.0)
    assert yq["min"] == pytest.approx(7.0)
    assert yq["max"] == pytest.approx(7.0)
    assert np.isnan(yq["std"])
    assert np.isnan(yq["sem"])

    zr = multi[(multi["g1"] == "Z") & (multi["g2"] == "r")].iloc[0]
    assert zr["count"] == 0
    for stat_col in ("mean", "median", "std", "sem", "min", "max"):
        assert np.isnan(zr[stat_col])


# =========================================
# robust_standardize
# =========================================


def test_robust_standardize():
    """Verify robust_standardize across MAD, MeanAD fallback, constant arrays, and NaNs.

    Strategy: exercise each level of the dispersion hierarchy (MAD, MeanAD fallback when
    MAD collapses to zero, constant input -> 0.0) and verify NaNs are preserved in place.
    Expected: large outliers receive large scores, MeanAD fallback matches the exact
    sqrt(pi/2) formula, constant arrays become 0.0, and NaNs remain NaN.
    """
    # Normal case without collapse
    normal_data = np.array([1, 2, 3, 4, 5, 100])
    std_data = robust_standardize(normal_data)
    assert not np.isnan(std_data).any()
    assert std_data[-1] > 10.0

    # MAD collapse (needs MeanAD fallback)
    mad_zero_data = np.array([0, 0, 0, 0, 100])
    std_data = robust_standardize(mad_zero_data)
    assert not np.isnan(std_data).any()
    assert std_data[-1] > 1.5
    # (0 - 20) / (32 * sqrt(pi / 2)) = -0.49867785...
    assert np.isclose(std_data[0], -0.49868, atol=1e-4)

    # Constant array has zero dispersion -> standardized to 0.0
    constant_data = np.array([5, 5, 5, 5, 5])
    std_data = robust_standardize(constant_data)
    assert np.all(std_data == 0.0)

    # Array with NaNs preserves NaN positions
    nan_data = np.array([1, 2, np.nan, 4, 5, 100])
    std_data = robust_standardize(nan_data)
    assert np.isnan(std_data[2])
    assert std_data[-1] > 10.0


def test_robust_standardize_axis_all_nan_and_invariants():
    """Verify axis=0/1 slicing, warning-free all-NaN handling, and affine/sign invariants.

    Strategy: apply robust_standardize along axis=0 and axis=1 of a 2D array mixing MAD,
    MeanAD-fallback, constant, and all-NaN slices under RuntimeWarning='error'; verify
    location invariance, positive scale invariance, and sign equivariance on both dispersion
    branches.
    Expected: each slice matches 1D robust_standardize, all-NaN inputs emit no RuntimeWarning
    and stay NaN, and all three algebraic invariants hold to floating-point precision.
    """
    # 4 rows x 4 cols: MAD col, MeanAD-fallback col, constant col, all-NaN col
    mat = np.array(
        [
            [1.0, 0.0, 7.0, np.nan],
            [2.0, 0.0, 7.0, np.nan],
            [4.0, 0.0, 7.0, np.nan],
            [10.0, 100.0, 7.0, np.nan],
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        col_std = robust_standardize(mat, axis=0)
        row_std = robust_standardize(mat.T, axis=1)
        all_nan_1d = robust_standardize(np.array([np.nan, np.nan, np.nan]))

    assert np.isnan(all_nan_1d).all() and all_nan_1d.shape == (3,)
    assert np.isnan(col_std[:, 3]).all()
    np.testing.assert_allclose(row_std, col_std.T, rtol=1e-12, equal_nan=True)
    for j in range(3):
        np.testing.assert_allclose(col_std[:, j], robust_standardize(mat[:, j]), rtol=1e-12)

    # Location invariance, positive scale invariance, and negation equivariance
    for sample in (
        np.array([1.0, 2.0, 3.0, 5.0, 9.0, 20.0]),  # MAD path
        np.array([0.0, 0.0, 0.0, 0.0, 10.0]),       # MeanAD fallback path
    ):
        base = robust_standardize(sample)
        np.testing.assert_allclose(robust_standardize(sample + 12.5), base, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(robust_standardize(3.5 * sample), base, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(robust_standardize(-sample), -base, rtol=1e-12, atol=1e-12)


# =========================================
# remove_outliers
# =========================================


@pytest.mark.parametrize("method", ["iqr", "zscore", "robust_zscore"])
@pytest.mark.parametrize("container", ["ndarray", "df_column", "df_all"])
def test_remove_outliers_univariate_containers(method, container):
    """Verify univariate outlier filtering across ndarray, single-column DF, and all-column DF.

    Strategy: construct a clean baseline sample with known extreme outliers (and a secondary
    column outlier for multi-column DataFrame filtering) and apply each univariate method.
    Expected: extreme outliers are removed while central inliers are retained; single-column
    filtering only filters on the requested column while all-column filtering drops rows with
    an outlier in any numeric column.
    """
    col_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 2.0])
    col_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 2.0, -100.0])

    if container == "ndarray":
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0])
        cleaned = remove_outliers(arr, method=method, threshold=1.5)
        assert 100.0 not in cleaned
        assert -100.0 not in cleaned
        assert 3.0 in cleaned
    elif container == "df_column":
        df = pd.DataFrame({"A": col_a, "B": col_b})
        cleaned = remove_outliers(df, method=method, threshold=1.5, column="A")
        assert 100.0 not in cleaned["A"].values
        assert -100.0 in cleaned["B"].values  # untouched because only column 'A' is filtered
        assert len(cleaned) == 6
    else:
        df = pd.DataFrame({"A": col_a, "B": col_b})
        cleaned = remove_outliers(df, method=method, threshold=1.5)
        assert 100.0 not in cleaned["A"].values
        assert -100.0 not in cleaned["B"].values
        assert len(cleaned) == 5


def test_remove_outliers_robust_zscore_overcomes_masking():
    """Verify robust_zscore detects masked extreme outliers that standard zscore misses.

    Strategy: place two extreme outliers (100, 105) alongside 10 compact inliers in [0, 2].
    The outliers inflate the sample standard deviation so their classical |z| values are
    ~2.17 and ~2.30 (below threshold=2.5), whereas their MAD-based robust |z| exceeds 65.
    Expected: standard zscore misses both outliers (keeping all 12 values), while
    robust_zscore removes both outliers and keeps all 10 inliers.
    """
    data = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 100.0, 105.0])

    cleaned_z = remove_outliers(data, method="zscore", threshold=2.5)
    assert 100.0 in cleaned_z and 105.0 in cleaned_z
    assert len(cleaned_z) == len(data)

    cleaned_rz = remove_outliers(data, method="robust_zscore", threshold=2.5)
    assert 100.0 not in cleaned_rz and 105.0 not in cleaned_rz
    np.testing.assert_array_equal(cleaned_rz, data[:-2])


@pytest.mark.parametrize("container", ["ndarray", "dataframe"])
def test_remove_outliers_mahalanobis_containers_and_nan_preservation(container):
    """Verify Mahalanobis outlier filtering and NaN-row preservation for arrays and DataFrames.

    Strategy: draw a correlated 2D Gaussian cloud, append off-diagonal extreme outliers, and
    also verify that a row containing NaN is bypassed (retained) rather than dropped.
    Expected: both extreme outliers are removed, almost all inliers are retained, and the
    NaN-containing row stays in the output for both ndarray and DataFrame inputs.
    """
    rng = np.random.default_rng(42)
    data = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.8], [0.8, 1.0]], size=100)
    outlier1 = np.array([10.0, -10.0])
    outlier2 = np.array([-10.0, 10.0])
    combined = np.vstack((data, outlier1, outlier2))

    if container == "ndarray":
        cleaned = remove_outliers(combined, method="mahalanobis", threshold=0.99)
        assert 95 <= len(cleaned) <= 100
        for row in cleaned:
            assert not np.allclose(row, outlier1)
            assert not np.allclose(row, outlier2)

        with_nan = combined.copy()
        with_nan[0, 0] = np.nan
        cleaned_nan = remove_outliers(with_nan, method="mahalanobis", threshold=0.99)
        assert np.isnan(cleaned_nan[0, 0])
    else:
        df = pd.DataFrame(combined, columns=["x", "y"])
        cleaned = remove_outliers(df, method="mahalanobis", threshold=0.99)
        assert 95 <= len(cleaned) <= 100
        assert not ((cleaned["x"] == 10.0) & (cleaned["y"] == -10.0)).any()
        assert not ((cleaned["x"] == -10.0) & (cleaned["y"] == 10.0)).any()

        df_nan = df.copy()
        df_nan.loc[0, "x"] = np.nan
        cleaned_nan = remove_outliers(df_nan, method="mahalanobis", threshold=0.99)
        assert np.isnan(cleaned_nan.loc[0, "x"])


@pytest.mark.parametrize("container", ["dataframe", "ndarray"])
def test_mahalanobis_finite_sample_bound(container):
    """Exact finite-sample Beta thresholding catches outliers in small datasets (DF and ndarray).

    In a dataset of N=10, d=2, sample D^2 is bounded by (N-1)^2/N = 8.1, which is strictly
    less than the asymptotic chi2(df=2) 0.99 quantile (9.21). The scaled Beta threshold
    (~7.25) detects the extreme outlier on both DataFrame and ndarray inputs.
    """
    rng = np.random.default_rng(42)
    N, d = 10, 2
    X = rng.standard_normal((N, d))
    X[0] = [1_000_000.0, 1_000_000.0]

    if container == "dataframe":
        df = pd.DataFrame(X, columns=["x", "y"])
        filtered = remove_outliers(df, method="mahalanobis", threshold=0.99)
        assert len(filtered) == N - 1
        assert 1_000_000.0 not in filtered["x"].values
    else:
        filtered = remove_outliers(X, method="mahalanobis", threshold=0.99)
        assert filtered.shape == (N - 1, d)
        assert 1_000_000.0 not in filtered[:, 0]


def test_remove_outliers_mahalanobis_catches_correlated_off_diagonal_outlier():
    """Verify Mahalanobis catches a bivariate off-diagonal outlier missed by 1D methods.

    Strategy: construct 60 points tightly clustered around the diagonal y = x over [-3, 3]
    plus one off-diagonal point (1.5, -1.5) whose marginals sit well inside every univariate
    threshold (|z| < 1, |rz| < 1, well inside IQR) yet yields D^2 > 50.
    Expected: iqr, zscore, and robust_zscore keep all 61 rows, whereas mahalanobis drops
    exclusively the off-diagonal outlier.
    """
    xs = np.linspace(-3.0, 3.0, 60)
    ys = xs + np.where(np.arange(60) % 2 == 0, 0.08, -0.08)
    X = np.vstack([np.column_stack([xs, ys]), [1.5, -1.5]])
    df = pd.DataFrame(X, columns=["x", "y"])

    for method, thresh in (("iqr", 1.5), ("zscore", 2.5), ("robust_zscore", 2.5)):
        assert len(remove_outliers(df, method=method, threshold=thresh)) == len(df)

    cleaned = remove_outliers(df, method="mahalanobis", threshold=0.99)
    assert len(cleaned) == 60
    np.testing.assert_allclose(cleaned.to_numpy(), X[:-1])


def test_remove_outliers_mahalanobis_singular_covariance_pinv():
    """Verify Mahalanobis handles rank-deficient (collinear) covariance via pinv.

    Strategy: create strictly collinear 2D data (y = 2x, rank-1 singular covariance) plus an
    outlier [100, 200] still on the line so covariance remains singular.
    Expected: no LinAlgError, the outlier is removed, and all 20 inliers are retained.
    """
    xs = np.concatenate([np.linspace(-2.0, 2.0, 20), [100.0]])
    X = np.column_stack([xs, 2.0 * xs])

    cleaned = remove_outliers(X, method="mahalanobis", threshold=0.99)
    assert cleaned.shape == (20, 2)
    np.testing.assert_allclose(cleaned, X[:-1])


def test_remove_outliers_ignores_non_numeric_dataframe_columns():
    """Verify non-numeric DataFrame columns are ignored during filtering and preserved in output.

    Strategy: add a string label column alongside two numeric columns with an outlier and
    filter using both univariate ('iqr') and multivariate ('mahalanobis') methods.
    Expected: filtering succeeds without type errors, removes the outlier row, and preserves
    the non-numeric column values for all retained rows.
    """
    rng = np.random.default_rng(7)
    vals = rng.normal(0.0, 1.0, size=(20, 2))
    vals[0] = [50.0, -50.0]
    df = pd.DataFrame(
        {
            "label": [f"s{i}" for i in range(20)],
            "x": vals[:, 0],
            "y": vals[:, 1],
        }
    )

    cleaned_iqr = remove_outliers(df, method="iqr", threshold=1.5)
    assert "label" in cleaned_iqr.columns
    assert "s0" not in cleaned_iqr["label"].values
    assert len(cleaned_iqr) <= 19

    cleaned_mah = remove_outliers(df, method="mahalanobis", threshold=0.99)
    assert "label" in cleaned_mah.columns
    assert "s0" not in cleaned_mah["label"].values
    assert len(cleaned_mah) <= 19


def test_remove_outliers_validation_errors():
    """Verify clear ValueErrors for invalid methods, dimensions, columns, and thresholds.

    Strategy: test invalid Mahalanobis configurations (single column, <2D input, threshold
    outside (0, 1)), unknown method names across containers, and 3D array input.
    Expected: every invalid call raises ValueError with a descriptive message.
    """
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [4.0, 5.0, 6.0, 7.0]})
    arr_2d = df.to_numpy()

    with pytest.raises(ValueError, match="cannot be applied to a single column"):
        remove_outliers(df, method="mahalanobis", threshold=0.95, column="x")

    with pytest.raises(ValueError, match="requires at least 2 dimensions"):
        remove_outliers(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), method="mahalanobis", threshold=0.95)

    with pytest.raises(ValueError, match="requires at least 2 dimensions"):
        remove_outliers(np.array([1.0, 2.0, 3.0]), method="mahalanobis", threshold=0.95)

    for bad_thresh in (1.5, 1.0, 0.0, -0.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            remove_outliers(df, method="mahalanobis", threshold=bad_thresh)
        with pytest.raises(ValueError, match="between 0 and 1"):
            remove_outliers(arr_2d, method="mahalanobis", threshold=bad_thresh)

    for data_obj, col in (
        (np.array([1.0, 2.0, 3.0]), None),
        (df, "x"),
        (df, None),
    ):
        with pytest.raises(ValueError, match="Unknown method"):
            remove_outliers(data_obj, method="bogus_method", column=col)

    with pytest.raises(ValueError, match="1D or 2D"):
        remove_outliers(np.zeros((2, 3, 4)), method="iqr")


def test_remove_outliers_zscore_ignores_constant_columns():
    """A constant column has no z-score outliers and must not empty the DataFrame.

    scipy's zscore is NaN for zero variance and NaN <= threshold is False, so every
    row was dropped. Expected: all 10 rows kept, no RuntimeWarning, and a real outlier
    in another column is still removed (positive control).
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=10), "batch": np.ones(10)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert len(remove_outliers(df, method="zscore", threshold=3)) == 10
    df.loc[0, "a"] = 50
    assert len(remove_outliers(df, method="zscore", threshold=2.5)) == 9
    assert len(remove_outliers(np.ones(5), method="zscore", threshold=3)) == 5


def test_remove_outliers_2d_array_filters_rows():
    """2D arrays are filtered row-wise (observations) instead of pooled and flattened."""
    A = np.column_stack([np.linspace(0, 1, 10), np.linspace(5, 6, 10)])
    A[0, 0] = 50  # the only outlier
    out = remove_outliers(A, method="iqr")
    assert out.shape == (9, 2)
    np.testing.assert_array_equal(out, A[1:])


def test_remove_outliers_mahalanobis_single_complete_row():
    """With fewer than 2 complete rows the covariance is undefined: keep everything (no crash)."""
    X = np.array([[1, 2, 3], [np.nan, 1, 1], [np.nan, 2, 2]], dtype=float)
    assert remove_outliers(X, method="mahalanobis", threshold=0.99).shape == (3, 3)


# =========================================
# bootstrap_ci
# =========================================


def test_bootstrap_bca_on_2d_data_uses_row_jackknife():
    """BCa works for statistics of 2D data (rows = observations).

    The jackknife used np.delete without an axis, flattening the data, so a
    correlation statistic crashed. Expected: a finite interval containing the sample
    correlation; and clear errors for n < 2 or NaN input.
    """
    rng = np.random.default_rng(0)
    pairs = rng.normal(size=(30, 2))
    pairs[:, 1] += pairs[:, 0]
    corr = lambda d: np.corrcoef(d[:, 0], d[:, 1])[0, 1]
    lo, hi = bootstrap_ci(pairs, corr, random_state=0)
    assert lo < corr(pairs) < hi
    with pytest.raises(ValueError, match="at least 2"):
        bootstrap_ci([3.0])
    with pytest.raises(ValueError, match="NaN"):
        bootstrap_ci([1.0, 2.0, np.nan, 4.0])


def _bootstrap_sample(seed=3, n=25):
    """Right-skewed (lognormal) sample used by the bootstrap invariant tests."""
    return np.random.default_rng(seed).lognormal(0.0, 1.0, n)


@pytest.mark.parametrize("method", ["percentile", "bc", "bca"])
@pytest.mark.parametrize("stat_func", [np.mean, np.median])
def test_bootstrap_ci_affine_and_negation_equivariance(method, stat_func):
    """Bootstrap intervals transform exactly with the data (same seed, same resamples).

    Strategy: with a fixed ``random_state`` the resample indices are identical, so
    for a location/scale-equivariant statistic theta the interval must satisfy
      CI(x + c) = CI(x) + c,   CI(k x) = k CI(x) (k > 0),   CI(-x) = (-hi, -lo).
    Negation flips the bias-correction z0 -> -z0 (ties counted half on each side)
    and the jackknife acceleration a -> -a; the BCa percentiles then map
    p -> 1 - p, so a sign error in z0, a, or the tie handling breaks the last
    identity. The median exercises the tie path (many resampled medians equal
    the sample median).
    Expected: all three identities hold to floating-point precision.
    """
    x = _bootstrap_sample()
    kw = dict(stat_func=stat_func, n_bootstraps=400, method=method, random_state=11)
    lo, hi = bootstrap_ci(x, **kw)
    assert lo < hi

    lo_s, hi_s = bootstrap_ci(x + 5.0, **kw)
    np.testing.assert_allclose([lo_s, hi_s], [lo + 5.0, hi + 5.0], rtol=0, atol=1e-9)

    lo_k, hi_k = bootstrap_ci(4.0 * x, **kw)
    np.testing.assert_allclose([lo_k, hi_k], [4.0 * lo, 4.0 * hi], rtol=1e-12)

    lo_n, hi_n = bootstrap_ci(-x, **kw)
    np.testing.assert_allclose([lo_n, hi_n], [-hi, -lo], rtol=1e-12)


def test_bootstrap_bca_acceleration_sign_and_symmetric_null():
    """BCa skews toward the long tail, and reduces to BC when the data are symmetric.

    Positive control: for the mean of right-skewed (lognormal) data the jackknife
    acceleration a is positive, so both BCa endpoints move right of the BC
    endpoints (same resamples, same z0; only a differs).
    Negative control: for an exactly mirror-symmetric sample {x, -x} the jackknife
    cubic moment sum(d_i^3) cancels, so a = 0 and BCa must equal BC.
    """
    x = _bootstrap_sample(n=40)
    kw = dict(stat_func=np.mean, n_bootstraps=2000, random_state=5)
    bc = bootstrap_ci(x, method="bc", **kw)
    bca = bootstrap_ci(x, method="bca", **kw)
    assert bca[0] > bc[0] and bca[1] > bc[1]

    sym = np.concatenate([x, -x])
    np.testing.assert_allclose(
        bootstrap_ci(sym, method="bca", **kw),
        bootstrap_ci(sym, method="bc", **kw),
        rtol=0,
        atol=1e-12,
    )


def test_bootstrap_bca_matches_scipy_reference():
    """BCa endpoints agree with scipy.stats.bootstrap(method='BCa').

    Strategy: both implement Efron's BCa (z0 from the bootstrap distribution, a from
    the jackknife) but draw different resamples, so they agree only up to Monte
    Carlo error. With 20k resamples that error is ~1% of the interval width.
    Expected: each endpoint within 5% of the interval width of scipy's, for the
    mean and the standard deviation of skewed data.
    """
    x = _bootstrap_sample(seed=8, n=40)
    for fn in (np.mean, np.std):
        lo, hi = bootstrap_ci(x, fn, n_bootstraps=20000, method="bca", random_state=0)
        ref = sps.bootstrap(
            (x,),
            fn,
            n_resamples=20000,
            method="BCa",
            vectorized=False,
            rng=np.random.default_rng(0),  # `rng` exists from SciPy 1.15 (locked >= 1.15.3)
        ).confidence_interval
        width = ref.high - ref.low
        assert abs(lo - ref.low) < 0.05 * width
        assert abs(hi - ref.high) < 0.05 * width


@pytest.mark.parametrize("method", ["percentile", "bca"])
def test_bootstrap_ci_coverage_is_calibrated(method):
    """Nominal 95% intervals cover the true mean about 95% of the time.

    Strategy: 200 independent N(1, 1) samples of size 30, one interval each. For
    the mean of Gaussian data bootstrap intervals are close to the t-interval;
    their coverage is ~0.93-0.94 at n = 30 (slight undercoverage of the
    bootstrap). Binomial SD of the estimate is ~0.017.
    Expected: coverage in [0.88, 0.99]. This is a coarse calibration guard (it
    flags grossly mis-scaled or mis-centred intervals); the exact identities and
    the SciPy comparison above are the sensitive checks.
    """
    rng = np.random.default_rng(2024)
    hits = 0
    n_rep = 200
    for i in range(n_rep):
        sample = rng.normal(1.0, 1.0, 30)
        lo, hi = bootstrap_ci(sample, np.mean, n_bootstraps=500, method=method, random_state=i)
        hits += lo <= 1.0 <= hi
    coverage = hits / n_rep
    assert 0.88 <= coverage <= 0.99, coverage


def test_bootstrap_ci_constant_sample_is_degenerate():
    """A constant sample gives a zero-width interval at that constant for every method.

    Negative control: no resample can differ from the data, so z0 = 0 (all ties),
    the jackknife spread is 0 (a stays 0), and the interval must be (c, c) with
    no NaN from 0/0.
    """
    for method in ("percentile", "bc", "bca"):
        assert bootstrap_ci(np.full(10, 2.5), method=method, random_state=0) == (2.5, 2.5)


def test_bootstrap_ci_invalid_method_generator_and_monotonicity():
    """Verify invalid method rejection, Generator reproducibility, and CI width monotonicity.

    Strategy: pass an unknown method name, compare integer seed vs np.random.Generator with
    the same seed, and evaluate nested confidence levels (0.50, 0.80, 0.95, 0.99) with a
    fixed seed across all three bootstrap methods.
    Expected: invalid method raises ValueError, Generator matches integer seed bit-for-bit,
    and higher confidence levels yield monotonically nested wider intervals.
    """
    x = _bootstrap_sample(seed=17, n=30)

    with pytest.raises(ValueError, match="Unknown bootstrap method"):
        bootstrap_ci(x, method="invalid", random_state=0)

    ci_from_int = bootstrap_ci(x, n_bootstraps=300, method="bca", random_state=123)
    ci_from_gen = bootstrap_ci(x, n_bootstraps=300, method="bca", random_state=np.random.default_rng(123))
    assert ci_from_int == ci_from_gen

    for method in ("percentile", "bc", "bca"):
        intervals = [
            bootstrap_ci(x, n_bootstraps=1000, ci=level, method=method, random_state=7)
            for level in (0.50, 0.80, 0.95, 0.99)
        ]
        for (lo_narrow, hi_narrow), (lo_wide, hi_wide) in zip(intervals, intervals[1:]):
            assert lo_wide <= lo_narrow
            assert hi_wide >= hi_narrow
            assert (hi_wide - lo_wide) > (hi_narrow - lo_narrow)
