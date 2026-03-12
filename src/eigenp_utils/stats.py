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

def cohens_d(group1, group2):
    """
    Calculate Cohen's d effect size for two independent samples.

    Parameters
    ----------
    group1 : array-like
        The first group of observations.
    group2 : array-like
        The second group of observations.

    Returns
    -------
    float
        The calculated Cohen's d.
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
    return float(d)

def bootstrap_ci(data, stat_func=np.mean, n_bootstraps=1000, ci=0.95, random_state=None):
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

    # Calculate confidence interval
    alpha = 1.0 - ci
    lower_percentile = (alpha / 2.0) * 100
    upper_percentile = (1.0 - alpha / 2.0) * 100

    lower_bound = np.percentile(bootstrapped_stats, lower_percentile)
    upper_bound = np.percentile(bootstrapped_stats, upper_percentile)

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
        The method to use ('iqr' or 'zscore').
    threshold : float, default 1.5
        The threshold for filtering (IQR multiplier or Z-score threshold).
    column : str, optional
        If data is a DataFrame, the column to filter on. If None and data
        is a DataFrame, filters rows where any column has an outlier.

    Returns
    -------
    pd.DataFrame or array-like
        The filtered data.
    """
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
            else:
                raise ValueError(f"Unknown method '{method}'")

            # Keep NaNs if present
            mask = mask | df_out[column].isna()
            return df_out[mask].copy()

        else:
            # Filtering based on all numeric columns
            numeric_cols = df_out.select_dtypes(include=[np.number]).columns
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
        else:
            raise ValueError(f"Unknown method '{method}'")

        # Reconstruct mask for the full array
        keep_mask = np.zeros_like(values, dtype=bool)
        keep_mask[valid_mask] = keep_valid

        # We generally drop the outliers for a flat array, and we drop NaNs
        return values[keep_mask]
