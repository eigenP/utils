import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eigenp_utils.stats import add_stat_annotations, cohens_d, bootstrap_ci, summary_stats, remove_outliers
from statannotations.Annotator import Annotator

def test_add_stat_annotations():
    """Test that add stat annotations works as expected."""
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
    """Test that cohens d works as expected."""
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
    """Test that bootstrap ci works as expected."""
    np.random.seed(42)
    data = np.random.normal(10, 2, 100)

    # For a normal distribution, mean CI should contain true mean
    lower, upper = bootstrap_ci(data, np.mean, n_bootstraps=500, random_state=42)

    assert lower < np.mean(data) < upper
    assert lower > 9.0  # reasonable bounds
    assert upper < 11.0

def test_summary_stats():
    """Test that summary stats works as expected."""
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
    """Test that remove outliers array works as expected."""
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
    """Test that remove outliers dataframe works as expected."""
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
    """Test that remove outliers mahalanobis array works as expected."""
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
    """Test that remove outliers mahalanobis dataframe works as expected."""
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

def test_mahalanobis_finite_sample_bound():
    """Test that exact Beta distribution thresholding effectively catches outliers in small datasets."""
    np.random.seed(42)
    N = 10
    d = 2

    # In a very small dataset with an extreme outlier, the exact D^2 is physically bounded by (N-1)^2/N = 8.1
    # Using the asymptotic chi-square threshold (e.g. at 0.99) might exceed this bound and fail to filter the outlier.
    # The Beta threshold correctly adjusts for the sample size.
    X = np.random.randn(N, d)
    X[0] = [1000000, 1000000]

    df = pd.DataFrame(X, columns=['x', 'y'])

    # 0.99 threshold. For N=10, d=2, the Chi-Square 0.99 threshold is 9.21, which is > 8.1 (impossible to exceed).
    # The Beta threshold will be around 7.25, properly identifying the outlier.
    filtered_df = remove_outliers(df, method='mahalanobis', threshold=0.99)

    assert len(filtered_df) == N - 1
    assert 1000000 not in filtered_df['x'].values

def test_remove_outliers_mahalanobis_errors():
    """Test that remove outliers mahalanobis errors works as expected."""
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

    with pytest.raises(ValueError, match="cannot be applied to a single column"):
        remove_outliers(df, method='mahalanobis', column='x')

    df_1d = pd.DataFrame({'x': [1, 2, 3]})
    with pytest.raises(ValueError, match="requires at least 2 dimensions"):
        remove_outliers(df_1d, method='mahalanobis')

    arr_1d = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="requires at least 2 dimensions"):
        remove_outliers(arr_1d, method='mahalanobis')

def test_robust_standardize():
    """Test that robust_standardize handles normal data, zero-inflated data, and constant arrays correctly."""
    from eigenp_utils.stats import robust_standardize

    # Normal case without collapse
    normal_data = np.array([1, 2, 3, 4, 5, 100])
    std_data = robust_standardize(normal_data)
    assert not np.isnan(std_data).any()
    # 100 should be heavily penalized (large z-score)
    assert std_data[-1] > 10.0

    # MAD collapse (needs MeanAD fallback)
    mad_zero_data = np.array([0, 0, 0, 0, 100])
    std_data = robust_standardize(mad_zero_data)
    assert not np.isnan(std_data).any()
    assert std_data[-1] > 1.5
    assert np.isclose(std_data[0], -0.49868, atol=1e-4) # (0 - 20) / (32 * 1.2533...)

    # MeanAD collapse? If MeanAD is 0, the array is constant
    constant_data = np.array([5, 5, 5, 5, 5])
    std_data = robust_standardize(constant_data)
    assert np.all(std_data == 0.0)

    # Array with NaNs
    nan_data = np.array([1, 2, np.nan, 4, 5, 100])
    std_data = robust_standardize(nan_data)
    assert np.isnan(std_data[2])
    assert std_data[-1] > 10.0

def test_remove_outliers_robust_zscore_array():
    """Test that remove outliers robust zscore array works as expected."""
    data = np.array([0, 0, 0, 0, 100, -100])

    # With standard zscore, variance gets heavily inflated by the 100 and -100
    cleaned_z = remove_outliers(data, method='zscore', threshold=1.5)

    # With robust zscore, the scale is much smaller, so 100/-100 are easily identified
    cleaned_rz = remove_outliers(data, method='robust_zscore', threshold=1.5)
    assert 100 not in cleaned_rz
    assert -100 not in cleaned_rz
    assert 0 in cleaned_rz
