
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import pytest
from eigenp_utils.single_cell import annotate_clusters_by_markers

def test_robust_confidence_score():
    """
    Verify that annotate_clusters_by_markers uses a robust empirical probability
    instead of a sensitive parametric Z-score.
    """
    # Create 100 cells
    n_cells = 100
    obs_names = [f"cell_{i}" for i in range(n_cells)]

    # Create outliers: 95 normal, 5 outliers
    # Normal: Type A > Type B
    # Outlier: Type B >>> Type A (skewing mean difference)

    # Scores
    # Type A: 95 cells with 1.0, 5 cells with 0.0
    scores_A = np.concatenate([np.ones(95), np.zeros(5)])

    # Type B: 95 cells with 0.0, 5 cells with 100.0
    scores_B = np.concatenate([np.zeros(95), np.ones(5) * 100.0])

    scores_df = pd.DataFrame({
        "TypeA": scores_A,
        "TypeB": scores_B
    }, index=obs_names)

    # Create AnnData
    adata = anndata.AnnData(X=np.zeros((n_cells, 2)))
    adata.obs_names = obs_names
    adata.obs["cluster"] = "C1"
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")

    # Run annotation
    # We pass scores directly to avoid marker computation overhead
    annotate_clusters_by_markers(
        adata,
        cluster_key="cluster",
        scores=scores_df,
        normalize_scores=False, # Use raw scores to control the distribution exactly
        write_to_obs=True
    )

    # Check results
    # The cluster should be assigned to Type A (Median A=1.0, Median B=0.0)
    assigned_type = adata.obs["cluster_cell_type"].iloc[0]
    confidence = adata.obs["cluster_ctype_softmaxp"].iloc[0]

    print(f"Assigned Type: {assigned_type}")
    print(f"Confidence (Softmax P): {confidence}")

    assert assigned_type == "TypeA", "Should assign to Type A based on median"

    # The confidence should be 0.95 (95/100 cells support Type A)
    # If it were parametric, it would be much lower (around 0.42)
    assert confidence == pytest.approx(0.95, abs=0.01), \
        f"Confidence {confidence} does not match empirical probability 0.95"

if __name__ == "__main__":
    test_robust_confidence_score()
