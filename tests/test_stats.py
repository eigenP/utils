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

    # Test Mahalanobis distance on all columns
    # We use a lower threshold equivalent to ~0.9 p-value for this very small test dataset
    import scipy.stats as stats
    chi2_thresh = np.sqrt(stats.chi2.ppf(0.9, df=2))

    # Actually, we can't easily pass the threshold for mahalanobis from the current API
    # so we will use a different test case that is overwhelmingly far out
    np.random.seed(42)
    df2 = pd.DataFrame({
        'A': np.concatenate([np.random.normal(0, 1, 1000), [10000]]),
        'B': np.concatenate([np.random.normal(0, 1, 1000), [-10000]])
    })
    cleaned_all = remove_outliers(df2)
    # the 0.999 chi2 threshold for 2 df is ~ 3.7. Some points from the 1000 normal randoms
    # may also be trimmed. We just assert the main outlier is removed.
    assert len(cleaned_all) < 1001
    assert len(cleaned_all) > 950
    assert 10000 not in cleaned_all['A'].values

def test_robust_standardize():
    from eigenp_utils.stats import robust_standardize

    # Normal distribution
    np.random.seed(42)
    x1 = np.random.normal(0, 1, 1000)
    z1 = robust_standardize(x1)

    # Check that std of z1 is close to 1
    assert np.isclose(np.std(z1), 1.0, atol=0.1)
    # Check that median is roughly 0
    assert np.isclose(np.median(z1), 0.0, atol=0.1)

    # Zero inflated distribution (MAD = 0, but variance exists)
    x2 = np.concatenate([np.zeros(600), np.random.normal(5, 1, 400)])
    z2 = robust_standardize(x2)

    # Should not produce NaNs or Infs
    assert not np.isnan(z2).any()
    assert not np.isinf(z2).any()
    # It should have successfully standardized
    assert np.std(z2) > 0.5 and np.std(z2) < 2.0

    # Completely identical distribution (MAD=0, MeanAD=0, STD=0)
    x3 = np.ones(100)
    z3 = robust_standardize(x3)

    # Should safely return all zeros
    assert np.all(z3 == 0)
