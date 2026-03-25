import scanpy as sc
import numpy as np
import pandas as pd
from src.eigenp_utils.single_cell import kknn_classifier, compute_kknn_neighbors

adata = sc.AnnData(np.random.randn(100, 10))
adata.obsm["X_pacmap"] = np.random.randn(100, 2)
adata.obsm["X_pacmap"][:50, 0] += 100
adata.obsm["X_pacmap"][50:, 0] -= 100

labels = np.array(["A"] * 50 + ["B"] * 50)
labels[0] = "B"
labels[99] = "A"

adata.obs["celltype"] = pd.Categorical(labels)

mask = np.zeros(100, dtype=bool)
mask[0] = True

kknn_classifier(adata, obs_key="celltype", use_rep="X_pacmap", n_neighbors=5, max_neighbors=10, mask=mask)

print("Original Label[0]:", adata.obs["celltype"].values[0])
print("Smoothed Label[0]:", adata.obs["celltype_kknn"].values[0])
