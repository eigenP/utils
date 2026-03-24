import scanpy as sc
import numpy as np
from eigenp_utils.single_cell import kknn_ingest

adata_ref = sc.AnnData(np.random.normal(size=(100, 10)))
adata_ref.obsm['X_pca'] = np.random.normal(size=(100, 5))
adata_ref.obs['cell_type'] = np.random.choice(['A', 'B'], size=100)

adata_query = sc.AnnData(np.random.normal(size=(20, 10)))
adata_query.obsm['X_pca'] = np.random.normal(size=(20, 5))

kknn_ingest(adata_query, adata_ref, obs_keys=['cell_type'])
