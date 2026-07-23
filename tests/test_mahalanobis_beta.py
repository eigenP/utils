import numpy as np
import pandas as pd
from scipy import stats
from eigenp_utils.stats import remove_outliers

def test_mahalanobis_beta_threshold_array():
    """
    Test exact Beta distribution thresholding in Mahalanobis outlier detection for small N.
    An outlier in a small dataset (e.g. N=10, d=2) can mask itself because the maximum
    theoretical squared Mahalanobis distance is bounded by ((N-1)^2)/N = 8.1.
    Asymptotic Chi-Square threshold for 95% might be ~5.99, but Beta correctly adjusts for sample size.
    """
    np.random.seed(42)
    # Generate tightly clustered data
    X = np.random.randn(10, 2)

    # Insert extreme outlier
    X[0] = [10, 10]

    # Calculate expected Beta threshold directly
    n_samples, n_features = X.shape
    bound = ((n_samples - 1) ** 2) / n_samples
    beta_thresh = stats.beta.ppf(0.95, n_features / 2.0, (n_samples - n_features - 1) / 2.0)
    thresh = bound * beta_thresh

    # Calculate D2 for outlier
    mean = np.mean(X, axis=0)
    X_centered = X - mean
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    left = np.dot(X_centered, inv_cov)
    D_squared = np.sum(left * X_centered, axis=1)

    # Sanity check assumptions for the test: D2[0] should be > thresh
    assert D_squared[0] > thresh, f"Outlier D2 ({D_squared[0]}) should be greater than Beta thresh ({thresh})"

    # Test `remove_outliers` array
    res = remove_outliers(X, method='mahalanobis', threshold=0.95)
    assert len(res) == 9, f"Expected 9 valid points, got {len(res)}. Outlier was not filtered."

def test_mahalanobis_beta_threshold_dataframe():
    """
    Test exact Beta distribution thresholding for DataFrame inputs.
    """
    np.random.seed(42)
    X = np.random.randn(10, 2)
    X[0] = [10, 10]

    df = pd.DataFrame(X, columns=['A', 'B'])
    res_df = remove_outliers(df, method='mahalanobis', threshold=0.95)

    assert len(res_df) == 9, f"Expected 9 rows in DataFrame, got {len(res_df)}"
    # Index 0 should be dropped
    assert 0 not in res_df.index

def test_mahalanobis_fallback_chi2():
    """
    Test fallback to Chi-Square when N <= d + 1.
    Beta distribution parameters would be invalid (non-positive) in this case.
    """
    np.random.seed(42)
    # N=3, d=2. Condition N <= d + 1 is true.
    X = np.random.randn(3, 2)

    # Add outlier that won't exceed max bound anyway, just checking it doesn't crash
    X[0] = [5, 5]

    # Should safely fallback to Chi-Square and return a result without throwing Beta parameter errors.
    res = remove_outliers(X, method='mahalanobis', threshold=0.95)
    # At N=3, max D2 is bounded severely ( (3-1)^2/3 = 1.33 ), while Chi2(0.95, df=2) = 5.99
    # So nothing gets filtered
    assert len(res) == 3
