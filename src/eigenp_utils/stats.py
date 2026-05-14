import numpy as np
import pandas as pd
from statannotations.Annotator import Annotator
from scipy import stats

def add_stat_annotations(
    ax,
    data,
    pairs,
    x=None,
    y=None,
    hue=None,
    test='t-test_welch',
    text_format='star',
    loc='inside',
    comparisons_correction='Holm-Bonferroni',
    return_annotator=False,
    **kwargs
):
    """
    Convenience wrapper around statannotations.Annotator.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to annotate.
    data : pd.DataFrame or list
        The data used for the plot.
    pairs : list of tuple
        List of pairs of groups to compare.
    x : str, optional
        The name of the x-axis variable.
    y : str, optional
        The name of the y-axis variable.
    hue : str, optional
        The name of the hue variable.
    test : str, default 't-test_welch'
        The statistical test to run.
    text_format : str, default 'star'
        The format of the text annotation ('star' or 'simple').
    loc : str, default 'inside'
        Location of the annotations ('inside' or 'outside').
    comparisons_correction : str, default 'Holm-Bonferroni'
        The multiple testing correction method.
    return_annotator : bool, default False
        Whether to return the Annotator object in addition to the Axes.
    **kwargs : dict
        Additional kwargs passed to annotator.configure()

    Returns
    -------
    matplotlib.axes.Axes
        The modified axes.
    (matplotlib.axes.Axes, statannotations.Annotator)
        If return_annotator is True.
    """
    annotator = Annotator(ax, pairs, data=data, x=x, y=y, hue=hue)
    annotator.configure(
        test=test,
        text_format=text_format,
        loc=loc,
        comparisons_correction=comparisons_correction,
        **kwargs
    )
    annotator.apply_and_annotate()

    if return_annotator:
        return ax, annotator
    return ax

import scipy.special

def cohens_d(group1, group2, correction=True):
    """
    Calculate Cohen's d effect size for two independent samples.
    By default, applies Hedges' exact correction factor to yield an unbiased estimator (Hedges' g).

    Parameters
    ----------
    group1 : array-like
        The first group of observations.
    group2 : array-like
        The second group of observations.
    correction : bool, default True
        Whether to apply Hedges' correction for small sample sizes.
        If True, returns Hedges' g. If False, returns Cohen's d (biased sample estimate).

    Returns
    -------
    float
        The calculated effect size.
    """
    x1 = np.asarray(group1)
    x2 = np.asarray(group2)

    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return np.nan

    var1 = np.var(x1, ddof=1)
    var2 = np.var(x2, ddof=1)

    # Pooled standard deviation
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if s_pooled == 0:
        return 0.0

    d = (np.mean(x1) - np.mean(x2)) / s_pooled

    if correction:
        # Apply Hedges' exact correction factor J(df)
        df = n1 + n2 - 2
        # J(df) = Gamma(df/2) / (sqrt(df/2) * Gamma((df-1)/2))
        # Calculated via log-gamma for numerical stability:
        # exp(gammaln(df/2) - 0.5 * log(df/2) - gammaln((df-1)/2))
        j_factor = np.exp(scipy.special.gammaln(df / 2.0) - 0.5 * np.log(df / 2.0) - scipy.special.gammaln((df - 1) / 2.0))
        d = d * j_factor

    return float(d)

def bootstrap_ci(data, stat_func=np.mean, n_bootstraps=1000, ci=0.95, method='bca', random_state=None):
    """
    Compute bootstrapped confidence intervals for an array.

    Parameters
    ----------
    data : array-like
        The data to bootstrap.
    stat_func : callable, default np.mean
        The statistic to calculate (e.g., np.mean, np.median).
    n_bootstraps : int, default 1000
        The number of bootstrap samples.
    ci : float, default 0.95
        The confidence interval width (between 0 and 1).
    method : str, default 'bc'
        The bootstrap method to use. Options are:
        - 'bca': Bias-Corrected and Accelerated (BCa) bootstrap. Corrects for median bias and skewness.
        - 'bc': Bias-Corrected (BC) bootstrap. Corrects for median bias in the bootstrap distribution.
        - 'percentile': Standard percentile bootstrap.
    random_state : int or np.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    tuple
        A tuple of (lower_bound, upper_bound)
    """
    data = np.asarray(data)
    rng = np.random.default_rng(random_state)

    # Generate bootstrap indices
    indices = rng.integers(0, len(data), size=(n_bootstraps, len(data)))

    # Calculate statistic for each bootstrap sample
    # Note: apply_along_axis expects a 1D array as input to the lambda, so data[x] gives us the resampled data for that row
    bootstrapped_stats = np.apply_along_axis(lambda x: stat_func(data[x]), 1, indices)

    alpha = 1.0 - ci

    if method == 'percentile':
        lower_percentile = (alpha / 2.0) * 100
        upper_percentile = (1.0 - alpha / 2.0) * 100

        lower_bound = np.percentile(bootstrapped_stats, lower_percentile)
        upper_bound = np.percentile(bootstrapped_stats, upper_percentile)

    elif method in ('bc', 'bca'):
        # Bias-Corrected (BC) Bootstrap
        theta_hat = stat_func(data)

        # Calculate empirical bias correction factor z0
        # z0 = Phi^-1 ( proportion of bootstrapped_stats < theta_hat )
        # matth: Robustly handle ties for discrete statistics (like median)
        prop_less = np.mean(bootstrapped_stats < theta_hat) + 0.5 * np.mean(bootstrapped_stats == theta_hat)

        # Handle edge cases where all stats are >= or <= theta_hat
        if prop_less == 0:
            prop_less = 1 / (n_bootstraps + 1)
        elif prop_less == 1:
            prop_less = n_bootstraps / (n_bootstraps + 1)

        z0 = stats.norm.ppf(prop_less)

        # Calculate adjusted percentiles
        z_alpha_2 = stats.norm.ppf(alpha / 2.0)
        z_1_alpha_2 = stats.norm.ppf(1.0 - alpha / 2.0)

        a = 0.0
        if method == 'bca':
            # Calculate acceleration factor (a) using jackknife estimates
            jackknife_stats = np.empty(len(data))
            for i in range(len(data)):
                jackknife_data = np.delete(data, i)
                jackknife_stats[i] = stat_func(jackknife_data)

            mean_jackknife = np.mean(jackknife_stats)
            diffs = mean_jackknife - jackknife_stats

            denom = 6.0 * (np.sum(diffs**2))**1.5
            if denom != 0:
                a = np.sum(diffs**3) / denom

        # BCa adjustment formula (reduces to BC when a = 0)
        adj_alpha_1 = stats.norm.cdf(z0 + (z0 + z_alpha_2) / (1.0 - a * (z0 + z_alpha_2)))
        adj_alpha_2 = stats.norm.cdf(z0 + (z0 + z_1_alpha_2) / (1.0 - a * (z0 + z_1_alpha_2)))


        lower_percentile = adj_alpha_1 * 100
        upper_percentile = adj_alpha_2 * 100

        lower_bound = np.percentile(bootstrapped_stats, lower_percentile)
        upper_bound = np.percentile(bootstrapped_stats, upper_percentile)

    else:
        raise ValueError(f"Unknown bootstrap method '{method}'. Use 'bc', 'bca', or 'percentile'.")

    return float(lower_bound), float(upper_bound)

def summary_stats(df, group_by, value_col):
    """
    Generate a clean summary for grouped data in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    group_by : str or list of str
        The column(s) to group by.
    value_col : str
        The column to calculate statistics for.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the summary statistics.
    """
    if isinstance(group_by, str):
        group_by = [group_by]

    summary = df.groupby(group_by)[value_col].agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        sem=lambda x: x.std() / np.sqrt(x.count()) if x.count() > 0 else np.nan,
        min='min',
        max='max'
    ).reset_index()

    return summary

def remove_outliers(data, method='iqr', threshold=1.5, column=None):
    """
    Filter out outliers in a DataFrame or array.

    Parameters
    ----------
    data : pd.DataFrame or array-like
        The data to filter.
    method : str, default 'iqr'
        The method to use ('iqr', 'zscore', 'robust_zscore', or 'mahalanobis').
        'robust_zscore' uses Median Absolute Deviation (MAD) which is less susceptible
        to extreme outliers inflating variance compared to 'zscore'.
        'mahalanobis' detects multivariate outliers using the Mahalanobis distance based
        on the pseudo-inverse covariance matrix, avoiding the destruction of underlying
        covariance structures caused by cascading univariate constraints. The threshold
        for 'mahalanobis' is interpreted as a probability (default 0.99) for the
        Chi-Square distribution bounds.
    threshold : float, default 1.5
        The threshold for filtering (IQR multiplier, Z-score threshold, or probability for Mahalanobis).
        For 'mahalanobis', a typical value might be 0.95 or 0.99.
    column : str, optional
        If data is a DataFrame, the column to filter on. If None and data
        is a DataFrame, filters rows where any column has an outlier. For 'mahalanobis',
        if column is None, it uses all numeric columns.

    Returns
    -------
    pd.DataFrame or array-like
        The filtered data.
    """
    def _robust_zscore(x):
        median = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - median))
        if mad == 0:
            return np.zeros_like(x)
        # 0.6745 is the 75th percentile of the standard normal distribution
        # making the robust z-score comparable in magnitude to the standard z-score
        return 0.6745 * (x - median) / mad

    if isinstance(data, pd.DataFrame):
        df_out = data.copy()

        if column is not None:
            # Filtering based on a single column
            if method == 'iqr':
                q1 = df_out[column].quantile(0.25)
                q3 = df_out[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                mask = (df_out[column] >= lower_bound) & (df_out[column] <= upper_bound)
            elif method == 'zscore':
                # Use scipy.stats.zscore with nan_policy='omit'
                valid_mask = df_out[column].notna()
                z_scores = pd.Series(index=df_out.index, dtype=float)
                z_scores[valid_mask] = np.abs(stats.zscore(df_out.loc[valid_mask, column]))
                mask = valid_mask & (z_scores <= threshold)
            elif method == 'robust_zscore':
                valid_mask = df_out[column].notna()
                z_scores = pd.Series(index=df_out.index, dtype=float)
                z_scores[valid_mask] = np.abs(_robust_zscore(df_out.loc[valid_mask, column].values))
                mask = valid_mask & (z_scores <= threshold)
            elif method == 'mahalanobis':
                valid_mask = df_out[column].notna()
                valid_data = df_out.loc[valid_mask, column].values
                if len(valid_data) <= 1:
                    mask = valid_mask
                else:
                    mu = np.mean(valid_data)
                    var = np.var(valid_data, ddof=1)
                    d_sq_series = pd.Series(index=df_out.index, dtype=float)
                    if var == 0:
                        d_sq_series[valid_mask] = 0.0
                    else:
                        d_sq_series[valid_mask] = ((valid_data - mu) ** 2) / var

                    prob_threshold = threshold if threshold < 1.0 else 0.99
                    chi2_threshold = stats.chi2.ppf(prob_threshold, df=1)
                    mask = valid_mask & (d_sq_series <= chi2_threshold)
            else:
                raise ValueError(f"Unknown method '{method}'")

            # Keep NaNs if present
            mask = mask | df_out[column].isna()
            return df_out[mask].copy()

        else:
            # Filtering based on all numeric columns
            numeric_cols = df_out.select_dtypes(include=[np.number]).columns

            if method == 'mahalanobis':
                valid_mask = df_out[numeric_cols].notna().all(axis=1)
                valid_data = df_out.loc[valid_mask, numeric_cols].values

                if len(valid_data) <= 1:
                    return df_out.copy()

                k = valid_data.shape[1]
                mu = np.mean(valid_data, axis=0)
                cov_mat = np.atleast_2d(np.cov(valid_data, rowvar=False))

                inv_cov = np.linalg.pinv(cov_mat)
                diff = valid_data - mu
                d_sq = np.sum((diff @ inv_cov) * diff, axis=1)

                prob_threshold = threshold if threshold < 1.0 else 0.99
                chi2_threshold = stats.chi2.ppf(prob_threshold, df=k)

                keep_valid = d_sq <= chi2_threshold

                mask = pd.Series(True, index=df_out.index)
                mask.loc[valid_mask] = keep_valid

                # Keep NaNs if present (consistent with other methods)
                # valid_mask is True where all numeric columns are notna
                # so ~valid_mask are the rows with NaNs. We want to keep them.
                mask = mask | ~valid_mask

                return df_out[mask].copy()

            mask = pd.Series(True, index=df_out.index)

            for col in numeric_cols:
                col_data = df_out[col]
                valid_mask = col_data.notna()
                if not valid_mask.any():
                    continue

                if method == 'iqr':
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - threshold * iqr
                    upper_bound = q3 + threshold * iqr
                    col_mask = (col_data >= lower_bound) & (col_data <= upper_bound)
                elif method == 'zscore':
                    z_scores = pd.Series(index=df_out.index, dtype=float)
                    z_scores[valid_mask] = np.abs(stats.zscore(df_out.loc[valid_mask, col]))
                    col_mask = valid_mask & (z_scores <= threshold)
                elif method == 'robust_zscore':
                    z_scores = pd.Series(index=df_out.index, dtype=float)
                    z_scores[valid_mask] = np.abs(_robust_zscore(df_out.loc[valid_mask, col].values))
                    col_mask = valid_mask & (z_scores <= threshold)
                else:
                    raise ValueError(f"Unknown method '{method}'")

                # For NaNs, we keep them so we don't drop rows just for NaN unless intended
                col_mask = col_mask | df_out[col].isna()
                mask = mask & col_mask

            return df_out[mask].copy()

    else:
        # Array-like
        values = np.asarray(data)
        # Filter out NaNs for calculation
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]

        if len(valid_values) == 0:
            return values

        if method == 'iqr':
            q1 = np.percentile(valid_values, 25)
            q3 = np.percentile(valid_values, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            keep_valid = (valid_values >= lower_bound) & (valid_values <= upper_bound)
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(valid_values))
            keep_valid = z_scores <= threshold
        elif method == 'robust_zscore':
            z_scores = np.abs(_robust_zscore(valid_values))
            keep_valid = z_scores <= threshold
        elif method == 'mahalanobis':
            if len(valid_values) <= 1:
                keep_valid = np.ones_like(valid_values, dtype=bool)
            else:
                mu = np.mean(valid_values)
                var = np.var(valid_values, ddof=1)
                if var == 0:
                    keep_valid = np.ones_like(valid_values, dtype=bool)
                else:
                    d_sq = ((valid_values - mu) ** 2) / var
                    prob_threshold = threshold if threshold < 1.0 else 0.99
                    chi2_threshold = stats.chi2.ppf(prob_threshold, df=1)
                    keep_valid = d_sq <= chi2_threshold
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Reconstruct mask for the full array
        keep_mask = np.zeros_like(values, dtype=bool)
        keep_mask[valid_mask] = keep_valid

        # We generally drop the outliers for a flat array, and we drop NaNs
        return values[keep_mask]
