
import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from eigenp_utils.single_cell import annotate_clusters_by_markers

def test_outlier_robustness():
    """
    Testr 🔎: Verify that annotation confidence is robust to massive outliers.

    Scenario:
    - 99 cells prefer Type B (score 11 vs 10).
    - 1 cell strongly prefers Type A (score 1000 vs 0).

    Parametric method (Mean/Std) is skewed by the outlier and gives low confidence to Type B (or flips to A).
    Empirical method (Proportion) should correctly identify Type B with high confidence (~0.99).
    """
    n_cells = 100
    n_outliers = 1
    n_majority = n_cells - n_outliers

    # Type A scores: 10 normally, 1000 outlier
    sA = np.array([10.0] * n_majority + [1000.0] * n_outliers)
    # Type B scores: 11 normally, 0 outlier
    sB = np.array([11.0] * n_majority + [0.0] * n_outliers)

    # Intuitively: Type B wins in 99% of cells.

    obs = pd.DataFrame({"leiden": ["0"]*n_cells}, index=[f"c{i}" for i in range(n_cells)])
    scores = pd.DataFrame({"TypeA": sA, "TypeB": sB}, index=obs.index)
    adata = AnnData(X=np.zeros((n_cells, 2)), obs=obs)

    res_df = annotate_clusters_by_markers(
        adata,
        cluster_key="leiden",
        scores=scores,
        normalize_scores=False
    )

    # Check assignment
    assigned = res_df.loc["0", "assigned_cell_type"]
    p_val = res_df.loc["0", "softmax_p"]

    # Assertions
    assert assigned == "TypeB", "Should assign to the majority winner (Median based)."

    # The robust metric should be close to 0.99 (99/100 wins)
    # The non-robust metric gives ~0.46
    assert p_val > 0.90, f"Confidence should be high (>0.90) for 99% majority, got {p_val:.4f}"
