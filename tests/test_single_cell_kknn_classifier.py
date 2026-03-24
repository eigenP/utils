import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from eigenp_utils.single_cell import kknn_classifier, compute_kknn_neighbors

def test_kknn_classifier_categorical():
    # Create simple dataset
    adata = sc.AnnData(np.random.randn(100, 10))
    adata.obsm["X_pacmap"] = np.random.randn(100, 2)

    # Introduce some clear categorical clusters for X_pacmap
    adata.obsm["X_pacmap"][:50, 0] += 100 # Shift cluster 1 very far
    adata.obsm["X_pacmap"][50:, 0] -= 100 # Shift cluster 2 very far

    # Set ground truth labels with a few errors
    labels = np.array(["A"] * 50 + ["B"] * 50)
    labels[0] = "B" # Outlier in cluster A
    labels[99] = "A" # Outlier in cluster B

    adata.obs["celltype"] = pd.Categorical(labels)

    # Run classifier
    kknn_classifier(adata, obs_key="celltype", use_rep="X_pacmap", n_neighbors=5, max_neighbors=10)

    assert "celltype_kknn" in adata.obs
    smoothed = adata.obs["celltype_kknn"].values

    # Check that outliers were corrected
    assert smoothed[0] == "A"
    assert smoothed[99] == "B"
    # Check that most others stayed the same
    assert np.sum(smoothed[:50] == "A") >= 45
    assert np.sum(smoothed[50:] == "B") >= 45

def test_kknn_classifier_continuous():
    adata = sc.AnnData(np.random.randn(100, 10))
    adata.obsm["X_pacmap"] = np.zeros((100, 2)) # All points exact same place

    # Continuous values with a single huge outlier
    vals = np.zeros(100)
    vals[0] = 1000.0
    adata.obs["score"] = vals

    kknn_classifier(adata, obs_key="score", use_rep="X_pacmap", n_neighbors=5, max_neighbors=10)

    assert "score_kknn" in adata.obs
    smoothed = adata.obs["score_kknn"].values

    # Outlier should be drastically smoothed towards 0 by its neighbors
    assert smoothed[0] < 500.0

def test_kknn_classifier_not_inplace():
    adata = sc.AnnData(np.random.randn(10, 10))
    adata.obsm["X_pacmap"] = np.random.randn(10, 2)
    adata.obs["celltype"] = pd.Categorical(["A"] * 5 + ["B"] * 5)

    # Needs max_neighbors to be <= n_samples
    res = kknn_classifier(adata, obs_key="celltype", use_rep="X_pacmap", inplace=False, n_neighbors=2, max_neighbors=4)

    assert "celltype_kknn" not in adata.obs
    assert len(res) == 10
