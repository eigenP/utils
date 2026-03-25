import scanpy as sc
import numpy as np
import pandas as pd
from src.eigenp_utils.single_cell import kknn_classifier

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

import sys
with open('src/eigenp_utils/single_cell.py', 'r') as f:
    content = f.read()
import re
new_content = content.replace("smoothed.append(winner)", "if i == 0: print(f'dist={dist}, weights={weights}, labels={neighbors_labels}, tally={vote_tally}'); smoothed.append(winner)")
with open('src/eigenp_utils/single_cell_debug.py', 'w') as f:
    f.write(new_content)

sys.path.insert(0, 'src')
from eigenp_utils.single_cell_debug import kknn_classifier

kknn_classifier(adata, obs_key="celltype", use_rep="X_pacmap", n_neighbors=5, max_neighbors=10, mask=mask)
