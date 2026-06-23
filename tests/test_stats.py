import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eigenp_utils.stats import add_stat_annotations, cohens_d, bootstrap_ci, summary_stats, remove_outliers
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
    from eigenp_utils.stats import robust_standardize
    x = np.array([1, 2, 3, 4, 5])
    # median = 3, MAD = 1
    # scale = 1 * 1.4826 = 1.4826
    # z = (x - 3) / 1.4826
    expected = (x - 3) / 1.4826
    z = robust_standardize(x)
    np.testing.assert_allclose(z, expected)

def test_robust_standardize_zero_inflated():
    from eigenp_utils.stats import robust_standardize
    x = np.array([0, 0, 0, 0, 10])
    # median = 0, MAD = 0 -> fallback to Mean & MeanAD
    # mean = 2, MeanAD = 3.2
    # scale = 3.2 * 1.2533 = 4.01056
    # z = (x - 2) / 4.01056
    expected = (x - 2) / 4.01056
    z = robust_standardize(x)
    np.testing.assert_allclose(z, expected)

def test_robust_standardize_all_zeros():
    from eigenp_utils.stats import robust_standardize
    x = np.array([0, 0, 0, 0, 0])
    # Mean and MAD and MeanAD and STD are all 0 -> returns all zeros
    z = robust_standardize(x)
    np.testing.assert_allclose(z, np.zeros_like(x))

def test_robust_standardize_axis():
    from eigenp_utils.stats import robust_standardize
    x = np.array([
        [1, 2, 3, 4, 5],
        [0, 0, 0, 0, 10]
    ])
    z = robust_standardize(x, axis=1)

    expected_row0 = (x[0] - 3) / 1.4826
    expected_row1 = (x[1] - 2) / 4.01056

    np.testing.assert_allclose(z[0], expected_row0)
    np.testing.assert_allclose(z[1], expected_row1)

def test_robust_standardize_nans():
    from eigenp_utils.stats import robust_standardize
    x = np.array([1, 2, np.nan, 4, 5])
    # nanmedian = 3, nanMAD = 1.5
    # scale = 1.5 * 1.4826 = 2.2239
    z = robust_standardize(x)

    assert np.isnan(z[2])

    expected = (np.array([1, 2, 4, 5]) - 3) / 2.2239
    np.testing.assert_allclose(z[~np.isnan(z)], expected)

def test_robust_standardize_scalar():
    from eigenp_utils.stats import robust_standardize
    z = robust_standardize(5)
    assert isinstance(z, float)
    assert z == 0.0
