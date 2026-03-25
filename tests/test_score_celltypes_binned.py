def test_score_celltypes_binned():
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import anndata
    from eigenp_utils.single_cell import score_celltypes, annotate_clusters_by_markers, sweep_leiden_and_annotate

    # Create synthetic anndata
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
    sc.pp.neighbors(adata, n_neighbors=3, use_rep="X") # Need neighbors for sweeping

    markers = {
        "Type1": ["G0"],
        "Type2": ["G1", "G2"], # G2 is completely zero
    }

    neg_markers = {
        "Type1": ["G1"]
    }

    # Test scanpy
    res_scanpy = score_celltypes(adata, markers, neg_markers, score_method="scanpy", use_raw=False)
    print("Scanpy")
    print(res_scanpy.head(2))

    # Test binned
    res_binned = score_celltypes(adata, markers, neg_markers, score_method="binned", use_raw=False)
    print("Binned")
    print(res_binned)

    # Test binned_weighted
    res_weighted = score_celltypes(adata, markers, neg_markers, score_method="binned_weighted", use_raw=False)
    print("Weighted")
    print(res_weighted)

    # Test annotate
    ann_res = annotate_clusters_by_markers(adata, "leiden_1.0", markers, neg_markers, score_method="binned", use_raw=False)
    print("Annotate")
    print(ann_res)

    # Test sweep
    sweep_res = sweep_leiden_and_annotate(adata, markers, neg_markers, score_method="binned", use_raw=False, neighbors_already_computed=True, resolutions=[0.5, 1.0])
    print("Sweep")
    print(sweep_res["cluster_annotations"][1.0])
