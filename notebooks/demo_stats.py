# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "seaborn",
#     "matplotlib",
#     "scikit-learn",
#     "eigenp-utils @ git+https://github.com/eigenP/utils.git@main",
# ]
# ///

import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")

@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell(hide_code=True)
async def _(mo):
    mo.md(
        """
        ## Setup
        Installing the package from GitHub...
        """
    )

    import sys

    def in_wasm():
        return sys.platform in ("emscripten", "wasi")

    OWNER, REPO, REF = "eigenP", "utils", "main"
    if in_wasm():
        GIT_URL = f"eigenp-utils @ https://github.com/{OWNER}/{REPO}/archive/{REF}.zip"
    else:
        GIT_URL = f"git+https://github.com/{OWNER}/{REPO}.git@{REF}"

    def install_local(url):
        import subprocess, sys, shutil

        if shutil.which("uv"):
            try:
                subprocess.check_call([
                    "uv", "pip", "install",
                    "--python", sys.executable,
                    url,
                ])
                return
            except subprocess.CalledProcessError:
                pass  # fall back to pip

        subprocess.check_call([sys.executable, "-m", "pip", "install", url])

    def install_github():
        if in_wasm():
            import micropip
            return micropip.install(GIT_URL)
        else:
            install_local(GIT_URL)

    res = install_github()
    if res is not None:
        await res

    import eigenp_utils
    print("eigenp_utils imported from:", eigenp_utils.__file__)

    return (
        GIT_URL,
        OWNER,
        REF,
        REPO,
        eigenp_utils,
        in_wasm,
        install_github,
        install_local,
        res,
        sys,
    )

@app.cell
def _(mo):
    mo.md(
        """
        # Stats Utilities Demo

        This notebook demonstrates the robust statistical utilities from `eigenp_utils.stats` using the classic **Iris dataset**.
        """
    )
    return

@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_iris
    from eigenp_utils.stats import (
        remove_outliers,
        summary_stats,
        cohens_d,
        bootstrap_ci,
        add_stat_annotations
    )

    # Load Iris dataset
    iris = load_iris(as_frame=True)
    df_iris = iris.frame
    df_iris['species'] = df_iris['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    df_iris = df_iris.drop(columns=['target'])

    # Inject an artificial extreme outlier to demonstrate robustness
    df_outlier = df_iris.copy()
    df_outlier.loc[0, 'sepal length (cm)'] = 100.0 # Extreme outlier

    return (
        add_stat_annotations,
        bootstrap_ci,
        cohens_d,
        df_iris,
        df_outlier,
        load_iris,
        np,
        pd,
        plt,
        remove_outliers,
        sns,
        summary_stats,
    )

@app.cell
def _(mo, df_outlier):
    mo.md(
        f"""
        ### 1. Robust Outlier Removal

        We've taken the iris dataset and explicitly modified the first row to have an artificial extreme outlier (`sepal length (cm) = 100.0`).

        Using `remove_outliers(method='robust_zscore')`, which uses Median Absolute Deviation (MAD), the function correctly filters out the extreme value without being skewed by it (which standard variance/Z-scores would be).
        """
    )
    return

@app.cell
def _(df_outlier, remove_outliers):
    df_clean = remove_outliers(df_outlier, method='robust_zscore', threshold=5.0, column='sepal length (cm)')

    num_outliers_removed = len(df_outlier) - len(df_clean)
    max_val_clean = df_clean['sepal length (cm)'].max()

    print(f"Removed {num_outliers_removed} outlier(s).")
    print(f"Max 'sepal length (cm)' after cleaning: {max_val_clean}")

    return df_clean, num_outliers_removed, max_val_clean

@app.cell
def _(mo):
    mo.md(
        """
        ### 2. Summary Statistics

        Using `summary_stats`, we can quickly generate a comprehensive table of aggregations (mean, median, standard error of the mean, etc.) for any numeric column grouped by a categorical variable.
        """
    )
    return

@app.cell
def _(df_clean, summary_stats):
    stats_df = summary_stats(df_clean, group_by='species', value_col='sepal length (cm)')
    stats_df
    return stats_df,

@app.cell
def _(mo):
    mo.md(
        """
        ### 3. Cohen's d (Effect Size)

        We calculate Cohen's d to quantify the effect size between 'setosa' and 'versicolor' sepal lengths.
        By default, `cohens_d(correction=True)` uses Hedges' exact correction factor via log-gamma approximation, yielding an unbiased estimate (Hedges' g).
        """
    )
    return

@app.cell
def _(cohens_d, df_clean):
    setosa_vals = df_clean[df_clean['species'] == 'setosa']['sepal length (cm)']
    versicolor_vals = df_clean[df_clean['species'] == 'versicolor']['sepal length (cm)']

    d_value = cohens_d(setosa_vals, versicolor_vals, correction=True)
    print(f"Cohen's d (Hedges' g) between Setosa and Versicolor sepal length: {d_value:.3f}")

    return d_value, setosa_vals, versicolor_vals

@app.cell
def _(mo):
    mo.md(
        """
        ### 4. Bootstrap Confidence Intervals

        `bootstrap_ci` provides robust confidence intervals. Here we compute the 95% Bias-Corrected and Accelerated (BCa) bootstrap interval for the mean `sepal width (cm)` of the virginica species.
        The BCa method correctly handles median bias and skewness in the underlying distribution.
        """
    )
    return

@app.cell
def _(bootstrap_ci, df_clean, np):
    virginica_width = df_clean[df_clean['species'] == 'virginica']['sepal width (cm)']

    ci_lower, ci_upper = bootstrap_ci(
        virginica_width,
        stat_func=np.mean,
        n_bootstraps=1000,
        ci=0.95,
        method='bca',
        random_state=42
    )
    mean_val = np.mean(virginica_width)

    print(f"Virginica sepal width mean: {mean_val:.3f}")
    print(f"95% BCa Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")

    return ci_lower, ci_upper, mean_val, virginica_width

@app.cell
def _(mo):
    mo.md(
        """
        ### 5. Statistical Annotations on Plots

        `add_stat_annotations` makes it easy to compute and render significance tests (e.g., Welch's t-test with Holm-Bonferroni correction) directly onto seaborn plots.
        """
    )
    return

@app.cell
def _(add_stat_annotations, df_clean, plt, sns):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(data=df_clean, x='species', y='petal length (cm)', ax=ax, hue='species', palette='Set2', legend=False)

    pairs = [
        ('setosa', 'versicolor'),
        ('versicolor', 'virginica'),
        ('setosa', 'virginica')
    ]

    add_stat_annotations(
        ax,
        data=df_clean,
        x='species',
        y='petal length (cm)',
        pairs=pairs,
        test='t-test_welch',
        text_format='star',
        loc='inside',
        comparisons_correction='Holm-Bonferroni'
    )

    ax.set_title("Petal Length by Species with Significance Annotations")
    fig.tight_layout()
    fig

    return ax, fig, pairs

if __name__ == "__main__":
    app.run()
