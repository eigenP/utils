import numpy as np
import pandas as pd
import scanpy as sc
import anndata
from eigenp_utils.single_cell import score_celltypes, annotate_clusters_by_markers

def test_score_celltypes_binned():
    np.random.seed(42)
    X = np.zeros((10, 5))
    # Gene 0: Highly expressed in cells 0-4
    X[0:5, 0] = np.random.uniform(5, 10, 5)
    X[5:10, 0] = np.random.uniform(0, 1, 5)
    X[5, 0] = 0 # add a zero
    # Gene 1: Expressed mostly in 5-9
    X[0:5, 1] = 0
    X[5:10, 1] = np.random.uniform(2, 5, 5)

    adata = anndata.AnnData(X=X)
    adata.var_names = ["G0", "G1", "G2", "G3", "G4"]
    adata.obs_names = [f"cell_{i}" for i in range(10)]
    adata.obs["leiden_1.0"] = pd.Categorical(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
    sc.pp.neighbors(adata, n_neighbors=3, use_rep="X")

    markers = {
        "Type1": ["G0"],
        "Type2": ["G1", "G2"], # G2 is completely zero
    }

    neg_markers = {
        "Type1": ["G1"]
    }

    res_binned = score_celltypes(adata, markers, neg_markers, score_method="binned", use_raw=False)

    # Assert values are between -1 and 1
    assert res_binned["Type1"].max() <= 1.0
    assert res_binned["Type1"].min() >= -1.0
    assert res_binned["Type2"].max() <= 1.0
    assert res_binned["Type2"].min() >= 0.0 # No negative markers

    # Assert cells 0-4 are higher for Type1 than cells 6-9
    assert res_binned["Type1"]["cell_0"] > res_binned["Type1"]["cell_9"]

def test_score_celltypes_binned_weighted():
    np.random.seed(42)
    X = np.zeros((10, 5))
    X[0:5, 0] = np.random.uniform(5, 10, 5)
    X[5:10, 0] = np.random.uniform(0, 1, 5)
    X[0:5, 1] = 0
    X[5:10, 1] = np.random.uniform(2, 5, 5)

    adata = anndata.AnnData(X=X)
    adata.var_names = ["G0", "G1", "G2", "G3", "G4"]
    adata.obs_names = [f"cell_{i}" for i in range(10)]

    markers = {
        "Type2": ["G1", "G2"], # G2 is completely zero
    }

    res_binned = score_celltypes(adata, markers, score_method="binned", use_raw=False)
    res_weighted = score_celltypes(adata, markers, score_method="binned_weighted", use_raw=False)

    # For Type2, since G2 is completely zero, only 1/2 markers are detected.
    # Therefore the weighted score should be exactly half of the binned score.
    np.testing.assert_allclose(res_weighted["Type2"].values, res_binned["Type2"].values * 0.5)
