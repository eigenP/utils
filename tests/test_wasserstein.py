import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import wasserstein_distance
from eigenp_utils.single_cell import find_correlated_features

def test_wasserstein_exact_match():
    # Set up some synthetic data
    np.random.seed(42)
    n_cells = 100
    n_genes = 10
    X = np.random.randn(n_cells, n_genes)
    var_names = [f"gene_{i}" for i in range(n_genes)]
    obs_names = [f"cell_{i}" for i in range(n_cells)]

    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))

    # Target will be the first gene
    target_gene = "gene_0"

    # Run the new optimized version
    res_df = find_correlated_features(adata, target=target_gene, metrics=["wasserstein"])

    # Compute the expected answers using scipy.stats.wasserstein_distance directly
    target_vec = X[:, 0]
    target_z = (target_vec - np.mean(target_vec)) / np.std(target_vec)

    expected_dists = {}
    for i, g in enumerate(var_names):
        col = X[:, i]
        col_z = (col - np.mean(col)) / np.std(col)
        expected_dists[g] = wasserstein_distance(col_z, target_z)

    # Check that they match
    for g in var_names:
        actual = res_df.loc[g, "wasserstein"]
        expected = expected_dists[g]
        assert np.isclose(actual, expected), f"Mismatch for {g}: Expected {expected}, got {actual}"

    print("Success: The exact closed-form Wasserstein distance matches scipy.stats.wasserstein_distance perfectly!")

if __name__ == "__main__":
    test_wasserstein_exact_match()
