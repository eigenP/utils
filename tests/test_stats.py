import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eigenp_utils.stats import add_stat_annotations, cohens_d, bootstrap_ci, summary_stats, remove_outliers, robust_standardize
from statannotations.Annotator import Annotator

def test_add_stat_annotations():
    # Setup simple data
    np.random.seed(42)
    data = pd.DataFrame({
        'group': ['A']*10 + ['B']*10,
        'value': np.concatenate([np.random.normal(0, 1, 10), np.random.normal(2, 1, 10)])
    })

    fig, ax = plt.subplots()

    import seaborn as sns
    sns.boxplot(data=data, x='group', y='value', ax=ax)

    pairs = [("A", "B")]

    # Test just returning ax
    ax_ret = add_stat_annotations(
        ax, data, pairs, x='group', y='value', test='t-test_welch', text_format='star'
    )

    assert ax_ret is ax
    assert len(ax.texts) > 0  # Should have added annotation text

    # Test returning both
    ax_ret, annotator = add_stat_annotations(
        ax, data, pairs, x='group', y='value', return_annotator=True
    )

    assert ax_ret is ax
    assert isinstance(annotator, Annotator)

def test_cohens_d():
    group1 = np.array([1, 2, 3, 4, 5])
    group2 = np.array([2, 3, 4, 5, 6])

    # known value calculation without correction
    # mean1 = 3, var1 = 2.5
    # mean2 = 4, var2 = 2.5
    # spooled = sqrt(2.5)
    # d = (3 - 4) / sqrt(2.5) = -1 / 1.5811 = -0.63245

    d = cohens_d(group1, group2, correction=False)
    np.testing.assert_almost_equal(d, -0.63245, decimal=4)

    # Test with Hedges' g correction (default)
    # df = 5 + 5 - 2 = 8
    # J(8) ~ 0.9027
    # g = -0.63245 * 0.9027 = -0.5709
    d_corrected = cohens_d(group1, group2)
    np.testing.assert_almost_equal(d_corrected, -0.5709, decimal=4)

    # Test with exact same groups
    d_same = cohens_d(group1, group1)
    np.testing.assert_almost_equal(d_same, 0.0)

    # Test with too small groups
    assert np.isnan(cohens_d([1], [2, 3]))

def test_bootstrap_ci():
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)

    # For a normal distribution, mean CI should contain true mean
    lower, upper = bootstrap_ci(data, np.mean, n_bootstraps=500, random_state=42)

    assert lower < np.mean(data) < upper
    assert lower > 9.0  # reasonable bounds
    assert upper < 11.0

def test_summary_stats():
    df = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B', 'B'],
        'value': [1, 2, 3, 10, 20, 30]
    })

    summary = summary_stats(df, group_by='group', value_col='value')

    assert len(summary) == 2
    assert list(summary.columns) == ['group', 'count', 'mean', 'median', 'std', 'sem', 'min', 'max']

    # Check A
    a_stats = summary[summary['group'] == 'A'].iloc[0]
    assert a_stats['count'] == 3
    assert a_stats['mean'] == 2.0
    assert a_stats['median'] == 2.0
    assert a_stats['min'] == 1.0
    assert a_stats['max'] == 3.0

    # Check B
    b_stats = summary[summary['group'] == 'B'].iloc[0]
    assert b_stats['mean'] == 20.0
    assert b_stats['median'] == 20.0

def test_remove_outliers_array():
    data = np.array([1, 2, 3, 4, 5, 100, -100])

    # Test IQR
    cleaned_iqr = remove_outliers(data, method='iqr', threshold=1.5)
    assert 100 not in cleaned_iqr
    assert -100 not in cleaned_iqr
    assert 3 in cleaned_iqr

    # Test Z-score
    cleaned_z = remove_outliers(data, method='zscore', threshold=1.5)
    assert 100 not in cleaned_z
    assert -100 not in cleaned_z

def test_remove_outliers_dataframe():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100, 2],
        'B': [1, 2, 3, 4, 5, 2, -100]
    })

    # Test IQR on specific column
    cleaned_col_A = remove_outliers(df, method='iqr', threshold=1.5, column='A')
    assert len(cleaned_col_A) == 6
    assert 100 not in cleaned_col_A['A'].values

    # Test Z-score on all columns
    cleaned_all = remove_outliers(df, method='zscore', threshold=1.5)
    assert len(cleaned_all) == 5
    assert 100 not in cleaned_all['A'].values
    assert -100 not in cleaned_all['B'].values

def test_remove_outliers_mahalanobis_array():
    np.random.seed(42)
    # Generate large multivariate normal dataset
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]
    data = np.random.multivariate_normal(mean, cov, 100)

    # Insert obvious outliers
    outlier1 = [10, -10]
    outlier2 = [-10, 10]
    data_with_outlier = np.vstack((data, outlier1, outlier2))

    # 0.99 Chi-Square probability threshold
    cleaned = remove_outliers(data_with_outlier, method='mahalanobis', threshold=0.99)

    assert len(cleaned) >= 95
    assert len(cleaned) <= 101 # Might filter out some natural tail values

    # Check that outliers are filtered
    for row in cleaned:
        assert not np.allclose(row, outlier1)
        assert not np.allclose(row, outlier2)

def test_remove_outliers_mahalanobis_dataframe():
    np.random.seed(42)
    mean = [0, 0]
    cov = [[1, 0.8], [0.8, 1]]
    data = np.random.multivariate_normal(mean, cov, 100)
    data_with_outlier = np.vstack((data, [10, -10]))

    df = pd.DataFrame(data_with_outlier, columns=['x', 'y'])

    # Test normal cleaning
    cleaned = remove_outliers(df, method='mahalanobis', threshold=0.99)
    assert len(cleaned) >= 95
    assert 10 not in cleaned['x'].values

    # Test with NaN propagation
    df.loc[0, 'x'] = np.nan
    cleaned_nan = remove_outliers(df, method='mahalanobis', threshold=0.99)
    # The NaN row should be kept
    assert np.isnan(cleaned_nan.loc[0, 'x'])

def test_remove_outliers_mahalanobis_errors():
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

    with pytest.raises(ValueError, match="cannot be applied to a single column"):
        remove_outliers(df, method='mahalanobis', column='x')

    df_1d = pd.DataFrame({'x': [1, 2, 3]})
    with pytest.raises(ValueError, match="requires at least 2 dimensions"):
        remove_outliers(df_1d, method='mahalanobis')

    arr_1d = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="requires at least 2 dimensions"):
        remove_outliers(arr_1d, method='mahalanobis')

def test_robust_standardize_normal():
    # Verify asymptotic normality scaling on a typical dataset without zeroes
    np.random.seed(42)
    # MAD valid path
    data = np.random.normal(5, 2, 1000)
    z = robust_standardize(data)
    # Center should be near 0, standard deviation near 1 due to 1.4826 scale
    assert np.abs(np.mean(z)) < 0.1
    assert np.abs(np.std(z) - 1.0) < 0.1

def test_robust_standardize_zero_inflated_mad_collapse():
    # Verify MeanAD fallback when MAD collapses to 0
    # Create dataset with >50% ties so median=0 and MAD=0
    data = np.array([0, 0, 0, 0, 0, 0, 10, -10, 20, -20])

    # Check that MAD is indeed 0
    assert np.median(np.abs(data - np.median(data))) == 0

    # robust_standardize should fallback to MeanAD (centered on Mean)
    z = robust_standardize(data)

    # Expected behavior with MeanAD
    # Mean = 0, MeanAD = np.mean(np.abs(data - 0)) = 60 / 10 = 6
    # Scale = 6 * 1.2533 = 7.5198
    # z for 10 should be 10 / 7.5198 = 1.3298
    np.testing.assert_allclose(z[6], 10 / (6 * 1.2533))

def test_robust_standardize_meanad_collapse():
    # Verify standard deviation fallback when both MAD and MeanAD collapse to 0
    # E.g. all values are identical, except for perhaps a tiny floating point difference?
    # Wait, if all values are identical, then STD is also 0.
    # The hierarchy: scale = std, but std=0 means scale=1.0 via fallback.
    data = np.array([5, 5, 5, 5, 5])
    z = robust_standardize(data)

    # Center is 5, scale falls back to 1.0 since all dispersions are 0
    np.testing.assert_allclose(z, np.zeros_like(z))

def test_robust_standardize_multi_dimensional():
    # Verify axis parameter handles multi-dimensional arrays correctly
    # One dimension has valid MAD, the other has collapsed MAD
    # arr shape: (2, 5) -> axis=1 means calc along rows
    arr = np.array([
        [1, 2, 3, 4, 5],       # Valid MAD (median=3, MAD=1)
        [0, 0, 0, 0, 100]      # Collapsed MAD, valid MeanAD (mean=20, MeanAD=32)
    ])

    z = robust_standardize(arr, axis=1)

    assert z.shape == (2, 5)

    # Row 0: Center = 3, Scale = 1 * 1.4826 = 1.4826
    # Element 4 (val=5) -> (5 - 3) / 1.4826 = 2 / 1.4826
    np.testing.assert_allclose(z[0, 4], 2 / 1.4826)

    # Row 1: Center = 20, Scale = 32 * 1.2533 = 40.1056
    # Element 4 (val=100) -> (100 - 20) / 40.1056 = 80 / 40.1056
    np.testing.assert_allclose(z[1, 4], 80 / (32 * 1.2533))
