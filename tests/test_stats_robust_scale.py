import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from eigenp_utils.single_cell import score_celltypes
from eigenp_utils.stats import remove_outliers

def test_score_celltypes_mad_collapse():
    """
    Tests that if >50% of cells have identical scores (e.g. 0),
    the MAD collapses to 0, but the robust standardizer correctly
    falls back to Mean Absolute Deviation to avoid blowing up the scale.
    """
    np.random.seed(42)
    # 95 cells with 0 expression, 5 cells with high expression
    X = np.zeros((100, 2))
    X[0:5, 0] = 100.0

    adata = anndata.AnnData(X=X)
    adata.var_names = ["G1", "G2"]
    adata.obs_names = [f"cell_{i}" for i in range(100)]

    markers = {"RareType": ["G1"]}

    # Scanpy score method uses robust_scale
    scores = score_celltypes(adata, markers, score_method="scanpy", use_raw=False)

    # If it was dividing by 1e-8, the max score would be astronomically high (e.g. >1e6)
    # With the fallback, the max score should be properly Z-scaled (e.g. ~4-16)
    max_score = scores["RareType"].max()
    assert max_score > 0
    assert max_score < 100.0, f"Expected scaled z-score, but got massive inflation: {max_score}"

    # Median should be properly centered at ~0
    assert np.isclose(np.median(scores["RareType"]), 0, atol=1e-5)

def test_remove_outliers_mad_collapse():
    """
    Tests that outlier removal does not silently fail (by returning 0 z-scores)
    when MAD collapses to 0 on a zero-inflated array containing an extreme outlier.
    """
    # 99 zeros and 1 extreme outlier
    data = np.zeros(100)
    data[0] = 1000.0

    # robust_zscore should correctly identify the outlier using hierarchical scale
    res = remove_outliers(data, method='robust_zscore', threshold=3)

    # The outlier should be dropped
    assert len(res) == 99
    assert 1000.0 not in res
