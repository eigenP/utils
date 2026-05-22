import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from eigenp_utils.stats import add_stat_annotations, cohens_d, bootstrap_ci, summary_stats, remove_outliers, robust_standardize
from statannotations.Annotator import Annotator

def test_robust_standardize():
    # Test MAD standardization
    d1 = np.array([0, 0, 0, 0, 10])
    res1 = robust_standardize(d1, axis=None)
    # The first 4 are 0. The median is 0.
    # The MAD is 0 (since majority is 0).
    # Since MAD is 0, it falls back to MeanAD.
    # Mean is 2, MeanAD is 3.2. Out is (x - 2) / (3.2 * 1.2533)
    np.testing.assert_array_almost_equal(res1, [-0.49867785, -0.49867785, -0.49867785, -0.49867785, 1.9947114])

    # Test 2D Array
    d2 = np.array([[0, 1], [0, 2], [0, 3], [10, 4]])
    res2 = robust_standardize(d2, axis=0)
    # Col 0: falls back to MeanAD.
    # Col 1: MAD is non-zero, uses MAD.
    expected_col0 = [-0.53192304, -0.53192304, -0.53192304, 1.59576912]
    expected_col1 = [-1.01173463, -0.33724488, 0.33724488, 1.01173463]
    np.testing.assert_array_almost_equal(res2[:, 0], expected_col0)
    np.testing.assert_array_almost_equal(res2[:, 1], expected_col1)

    # Test NaNs
    d3 = np.array([1, 2, np.nan, 4, 5])
    res3 = robust_standardize(d3)
    assert np.isnan(res3[2])
    assert not np.isnan(res3[0])

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

def test_remove_outliers_mahalanobis():
    np.random.seed(42)
    # Large dataset N > 50 to avoid sample covariance inflation
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(0, 1, 100)
    })

    # Inject a distinct multivariate outlier
    df.loc[99, 'A'] = 10
    df.loc[99, 'B'] = 10

    cleaned = remove_outliers(df, method='mahalanobis', threshold=0.95)
    assert 99 not in cleaned.index
    assert len(cleaned) < 100

    # Array test
    arr = df.values
    cleaned_arr = remove_outliers(arr, method='mahalanobis', threshold=0.95)
    assert len(cleaned_arr) == len(cleaned)

    # Test missing value behavior (NaNs bypassed)
    df_nan = df.copy()
    df_nan.loc[0, 'A'] = np.nan
    cleaned_nan = remove_outliers(df_nan, method='mahalanobis', threshold=0.95)
    assert 0 in cleaned_nan.index # NaN row preserved

    arr_nan = arr.copy()
    arr_nan[0, 0] = np.nan
    cleaned_arr_nan = remove_outliers(arr_nan, method='mahalanobis', threshold=0.95)
    assert np.isnan(cleaned_arr_nan[0, 0]) # NaN row preserved

    # Test error cases
    with pytest.raises(ValueError, match="requires all numeric columns"):
        remove_outliers(df, method='mahalanobis', threshold=0.95, column='A')

    with pytest.raises(ValueError, match="requires at least two"):
        remove_outliers(df[['A']], method='mahalanobis', threshold=0.95)

    with pytest.raises(ValueError, match="Threshold for Mahalanobis"):
        remove_outliers(df, method='mahalanobis', threshold=1.5) # Invalid prob
