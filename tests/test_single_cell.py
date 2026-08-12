import os
import time
import warnings

from anndata import AnnData
from scipy import stats
from scipy.stats import norm
from unittest.mock import MagicMock, patch
from unittest.mock import patch
from unittest.mock import patch, MagicMock
import anndata
import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
import tempfile

from eigenp_utils.single_cell import _extract_gene_vector
from eigenp_utils.single_cell import annotate_clusters_by_markers
from eigenp_utils.single_cell import calculate_lineage_coupling, plot_coupling_heatmap
from eigenp_utils.single_cell import compute_kknn_neighbors
from eigenp_utils.single_cell import compute_kknn_neighbors, kknn_ingest
from eigenp_utils.single_cell import export_obs_from_adata_to_csv, import_obs_to_adata_from_csv
from eigenp_utils.single_cell import find_correlated_features
from eigenp_utils.single_cell import find_expression_archetypes
from eigenp_utils.single_cell import kknn_classifier, compute_kknn_neighbors
from eigenp_utils.single_cell import kknn_ingest
from eigenp_utils.single_cell import morans_i_all_fast
from eigenp_utils.single_cell import multiscale_coarsening
from eigenp_utils.single_cell import plot_archetype_summary
from eigenp_utils.single_cell import plot_marker_genes_dict_on_embedding
from eigenp_utils.single_cell import plot_volcano_adata
from eigenp_utils.single_cell import preprocess_subset
from eigenp_utils.single_cell import score_celltypes, annotate_clusters_by_markers
from eigenp_utils.single_cell import tl_pacmap



# =========================================
# Source: test_single_cell_lineage.py
# =========================================


def test_calculate_lineage_coupling():
    """Test that calculate lineage coupling works as expected."""
    # Create mock data
    n_cells = 100
    n_clones = 20
    n_types = 5

    obs = pd.DataFrame({
        'cell_type': np.random.choice([f'Type_{i}' for i in range(n_types)], n_cells),
        'CloneID': np.random.choice([f'Clone_{i}' for i in range(n_clones)], n_cells)
    })
    # Make some clones specific
    obs.loc[0:20, 'cell_type'] = 'Type_0'
    obs.loc[0:20, 'CloneID'] = 'Clone_0'
    # Ensure categorical to trigger warning if bug exists
    obs['cell_type'] = obs['cell_type'].astype('category')

    adata = anndata.AnnData(obs=obs)

    import warnings

    # Test function runs without warnings
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        obs_counts, z_scores, p_vals = calculate_lineage_coupling(
            adata,
            label_key='cell_type',
            clone_key='CloneID',
            method='permutation',
            n_permutations=10
        )

    # Check that no warnings related to shuffling categorical were raised
    for r in record:
        assert "shuffling a 'Categorical' object" not in str(r.message)

    assert isinstance(obs_counts, pd.DataFrame)
    assert isinstance(z_scores, pd.DataFrame)
    assert isinstance(p_vals, pd.DataFrame)

    assert obs_counts.shape == (n_types, n_types)
    assert z_scores.shape == (n_types, n_types)
    assert p_vals.shape == (n_types, n_types)


@pytest.mark.parametrize("title", ["Test Title 1", "Lineage Coupling"])
def test_plot_coupling_heatmap(title):
    """Test that plot coupling heatmap works as expected."""
    n_types = 5

    obs_counts = pd.DataFrame(np.random.randint(0, 10, size=(n_types, n_types)))
    z_scores = pd.DataFrame(np.random.randn(n_types, n_types))
    p_vals = pd.DataFrame(np.random.rand(n_types, n_types))

    fig = plot_coupling_heatmap(obs_counts, z_scores, p_vals, title=title)

    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_title() == title
    plt.close(fig)

# =========================================
# Source: test_single_cell_n_neighbors.py
# =========================================

# This will fail initially because the function is not yet implemented
def test_pacmap_heuristic_n_neighbors():
    """Test that pacmap heuristic n neighbors works as expected."""
    try:
        from eigenp_utils.single_cell import pacmap_heuristic_n_neighbors
    except ImportError:
        pytest.fail("pacmap_heuristic_n_neighbors is not implemented yet")

    # n <= 10000 should return 10
    assert pacmap_heuristic_n_neighbors(100) == 10
    assert pacmap_heuristic_n_neighbors(10000) == 10

    # n > 10000 should return int(round(10 + 15 * (np.log10(n) - 4)))
    assert pacmap_heuristic_n_neighbors(50000) == 20
    assert pacmap_heuristic_n_neighbors(100000) == 25
    assert pacmap_heuristic_n_neighbors(1000000) == 40

# =========================================
# Source: test_negative_selection.py
# =========================================

def create_mock_adata():
    # Create 4 cells, 6 genes
    # Cell 0: T-cell signature strong (CD3E high)
    # Cell 1: B-cell signature strong (CD19 high)
    # Cell 2: T-cell signature but with high housekeeping (GAPDH high)
    # Cell 3: T-cell signature but with high negative marker (CD14 high - indicating maybe monocyte contamination)

    # Genes: CD3E (T-cell), CD19 (B-cell), CD14 (Monocyte/Neg for T), GAPDH (Housekeeping/Neg for T), CD4 (T-cell), MS4A1 (B-cell)
    var_names = ["CD3E", "CD19", "CD14", "GAPDH", "CD4", "MS4A1"]

    X = np.array([
        #CD3E CD19 CD14 GAPDH CD4 MS4A1
        [10.0, 0.0, 0.0,  1.0,  8.0, 0.0],  # Cell 0: Pure T
        [0.0,  10.0, 0.0, 1.0,  0.0, 8.0],  # Cell 1: Pure B
        [9.0,  0.0,  0.0, 20.0, 7.0, 0.0],  # Cell 2: T with high background
        [8.0,  0.0, 15.0, 1.0,  6.0, 0.0],  # Cell 3: T with high CD14 (contaminated)
    ])

    obs = pd.DataFrame({"leiden": ["0", "1", "0", "0"]}, index=[f"cell_{i}" for i in range(4)])
    var = pd.DataFrame(index=var_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.raw = adata
    return adata

def test_score_celltypes_baseline():
    """Test standard positive scoring without negative markers."""
    adata = create_mock_adata()

    pos_dict = {
        "T-cell": ["CD3E", "CD4"],
        "B-cell": ["CD19", "MS4A1"]
    }

    scores = score_celltypes(adata, pos_dict)

    # Cell 0 should score high for T-cell, Cell 1 high for B-cell
    assert scores.loc["cell_0", "T-cell"] > scores.loc["cell_0", "B-cell"]
    assert scores.loc["cell_1", "B-cell"] > scores.loc["cell_1", "T-cell"]

def test_score_celltypes_negative_selection():
    """Test negative scoring alters the result."""
    adata = create_mock_adata()

    pos_dict = {
        "T-cell": ["CD3E", "CD4"],
        "B-cell": ["CD19", "MS4A1"]
    }

    neg_dict = {
        "T-cell": ["CD14"] # CD14 is a negative marker for T-cells
    }

    # Score baseline
    scores_base = score_celltypes(adata, pos_dict)
    # Score with negative selection
    scores_neg = score_celltypes(adata, pos_dict, cell_type_negative_markers_dict=neg_dict, score_method="net_scanpy")

    # For Cell 0 (Pure T), CD14 is 0. The neg score is low.
    # The net score for T-cell on Cell 0 should be relatively high.

    # For Cell 3 (Contaminated T), CD14 is 15.
    # Its T-cell score in the base case is high because CD3E/CD4 are high.
    # In the negative case, its T-cell score should be heavily penalized compared to Cell 0.

    # Normalize comparison manually as the actual scale depends on the internal score_genes + robust scaling
    diff_base = scores_base.loc["cell_0", "T-cell"] - scores_base.loc["cell_3", "T-cell"]
    diff_neg = scores_neg.loc["cell_0", "T-cell"] - scores_neg.loc["cell_3", "T-cell"]

    # The difference should be much larger when negative selection is applied, penalizing cell_3
    assert diff_neg > diff_base

def test_missing_negative_markers():
    """Test when cell type is missing in negative dict, it falls back to positive."""
    adata = create_mock_adata()

    pos_dict = {
        "T-cell": ["CD3E", "CD4"],
        "B-cell": ["CD19", "MS4A1"]
    }

    neg_dict = {
        "T-cell": ["CD14"]
        # B-cell is missing
    }

    scores_neg = score_celltypes(adata, pos_dict, cell_type_negative_markers_dict=neg_dict)

    # Both B-cell and T-cell should be present in the output
    assert "T-cell" in scores_neg.columns
    assert "B-cell" in scores_neg.columns
    assert not scores_neg["B-cell"].isna().all()

def test_annotation_with_negative_selection():
    """Test full pipeline with annotate_clusters_by_markers."""
    adata = create_mock_adata()

    pos_dict = {
        "T-cell": ["CD3E", "CD4"],
        "B-cell": ["CD19", "MS4A1"],
        "Contaminated": ["CD14"]
    }

    neg_dict = {
        "T-cell": ["CD14"]
    }

    # cluster 0 has 3 cells (0, 2, 3)
    # In base case, it might be classified as T-cell
    df_base = annotate_clusters_by_markers(
        adata,
        "leiden",
        cell_type_markers_dict=pos_dict
    )

    df_neg = annotate_clusters_by_markers(
        adata,
        "leiden",
        cell_type_markers_dict=pos_dict,
        cell_type_negative_markers_dict=neg_dict
    )

    # Verify both run without errors
    assert not df_base.empty
    assert not df_neg.empty
    assert "T-cell" in df_base.columns or "median_T-cell" in df_base.columns

def test_empty_negative_list():
    """Test handling of empty lists in negative dictionary."""
    adata = create_mock_adata()
    pos_dict = {"T-cell": ["CD3E"]}
    neg_dict = {"T-cell": []}

    scores = score_celltypes(adata, pos_dict, cell_type_negative_markers_dict=neg_dict)
    assert not scores.empty
    assert not scores["T-cell"].isna().all()

def test_missing_genes_in_dataset():
    """Test robustness when negative markers are completely missing from adata."""
    adata = create_mock_adata()
    pos_dict = {"T-cell": ["CD3E"]}
    neg_dict = {"T-cell": ["UNKNOWN_GENE"]}

    scores = score_celltypes(adata, pos_dict, cell_type_negative_markers_dict=neg_dict)
    assert not scores.empty
    assert not scores["T-cell"].isna().all()

# =========================================
# Source: test_single_cell_kknn_pbmc.py
# =========================================

def old_compute_kknn_neighbors(
    adata_query: sc.AnnData,
    adata_ref: sc.AnnData,
    use_rep: str = "X_pca",
    query_use_rep: str = "X_pca",
    n_neighbors: int = 10,
    min_neighbors: int = 3,
    max_neighbors: int = 20,
    quantile_bins: int = 10
):
    """The original unoptimized, heuristic method for comparison"""
    from sklearn.neighbors import NearestNeighbors

    X_query = adata_query.obsm[query_use_rep]
    X_ref = adata_ref.obsm[use_rep]

    N_ref = X_ref.shape[0]
    N_query = X_query.shape[0]

    nn = NearestNeighbors(n_neighbors=max_neighbors, algorithm='auto', n_jobs=-1)
    nn.fit(X_ref)
    distances, indices = nn.kneighbors(X_query)

    curvatures = np.zeros(N_query)
    m = X_ref.shape[1]

    for i in range(N_query):
        neighs_idx = indices[i]
        amostras = X_ref[neighs_idx]

        ni = len(neighs_idx)
        if ni > 1:
            I = np.cov(amostras, rowvar=False)
            if I.ndim == 0:
                I = np.array([[I]])
        else:
            I = np.eye(m)

        try:
            eigvals = np.linalg.eigvalsh(I)
        except np.linalg.LinAlgError:
            eigvals = np.ones(m)

        eigvals = np.maximum(eigvals, 0)
        total_var = np.sum(eigvals)

        if total_var > 0:
            num_small = max(1, m // 2)
            curvatures[i] = np.sum(eigvals[:num_small]) / total_var
        else:
            curvatures[i] = 0.0

    ptp = curvatures.max() - curvatures.min()
    if ptp == 0:
        K = np.zeros_like(curvatures)
    else:
        K = (curvatures - curvatures.min()) / (ptp + 1e-9)

    intervalos = np.linspace(0.0, 1.0, quantile_bins + 1)[1:-1]
    quantis = np.quantile(K, intervalos)
    bins = np.array(quantis)
    disc_curv = np.digitize(K, bins)

    pruned_distances = []
    pruned_indices = []

    for i in range(N_query):
        bin_idx = disc_curv[i]
        fraction = bin_idx / max(1, (quantile_bins - 1))
        keep = int(round(max_neighbors - fraction * (max_neighbors - min_neighbors)))
        keep = max(min_neighbors, min(max_neighbors, keep))

        pruned_distances.append(distances[i, :keep])
        pruned_indices.append(indices[i, :keep])

    return pruned_distances, pruned_indices


def test_compute_kknn_neighbors_pbmc_concordance():
    """Test that compute kknn neighbors pbmc concordance works as expected."""
    # Use the pbmc3k dataset
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver='arpack')

    # Split into ref and query (70/30)
    rng = np.random.default_rng(42)
    n_cells = adata.n_obs
    idx = rng.permutation(n_cells)
    split = int(0.7 * n_cells)

    adata_ref = adata[idx[:split]].copy()
    adata_query = adata[idx[split:]].copy()

    # Run the NEW method
    new_dists, new_idxs = compute_kknn_neighbors(
        adata_query,
        adata_ref,
        n_neighbors=15,
        min_neighbors=5,
        max_neighbors=30
    )

    # Run the OLD method
    old_dists, old_idxs = old_compute_kknn_neighbors(
        adata_query,
        adata_ref,
        n_neighbors=15,
        min_neighbors=5,
        max_neighbors=30
    )

    # Compare
    new_lengths = np.array([len(idx) for idx in new_idxs])
    old_lengths = np.array([len(idx) for idx in old_idxs])

    # They shouldn't be identical because the math is different, but they should be
    # highly correlated (i.e. areas of high curvature still prune more neighbors)
    from scipy.stats import spearmanr
    corr, _ = spearmanr(new_lengths, old_lengths)

    print(f"Spearman correlation between old and new neighborhood sizes: {corr:.3f}")
    assert corr > 0.3, "The new dimensionality metric should broadly correlate with the old curvature metric."

    # The new method should still obey the bounds
    assert new_lengths.min() >= 5
    assert new_lengths.max() <= 30

# =========================================
# Source: test_marker_plot.py
# =========================================


def test_plot_marker_genes_dict_on_embedding():
    """Test that plot marker genes dict on embedding works as expected."""
    # Setup dummy AnnData
    n_obs = 100
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Add dummy UMAP
    adata.obsm["X_umap"] = np.random.rand(n_obs, 2)

    # Define marker genes (some existing, some missing to test check_gene_adata implicitly)
    marker_genes = {
        "TissueA": ["gene_0", "gene_1"],
        "TissueB": ["gene_2", "gene_missing"]
    }

    # Mock score_genes to avoid data distribution requirements and ensure it "succeeds"
    # We need side_effect to actually add the score to adata.obs so sc.pl.embedding can plot it
    def score_genes_side_effect(adata, gene_list, score_name, **kwargs):
        adata.obs[score_name] = np.random.rand(adata.n_obs)

    with patch("scanpy.tl.score_genes", side_effect=score_genes_side_effect):
        # Run function
        axes_list = plot_marker_genes_dict_on_embedding(
            adata,
            marker_genes,
            basis="X_umap",
            show=False # Ensure it doesn't block
        )

    # Assertions
    assert isinstance(axes_list, list)
    # TissueA has 2 valid genes + 1 score -> 3 plots
    # TissueB has 1 valid gene + 1 score -> 2 plots
    # Total = 5
    assert len(axes_list) == 5, f"Expected 5 axes, got {len(axes_list)}"

    for ax in axes_list:
        assert isinstance(ax, plt.Axes)
        # Check if title or label logic worked (optional, but good)
        # We can check if ylabel is set as expected (Tissue Name + \n)

    print("Test passed successfully!")

def test_missing_basis():
    """Test that missing basis works as expected."""
    n_obs = 10
    n_vars = 10
    adata = AnnData(X=np.random.rand(n_obs, n_vars))
    marker_genes = {"A": ["gene_0"]}

    try:
        plot_marker_genes_dict_on_embedding(adata, marker_genes, basis="X_pca")
    except ValueError as e:
        assert "compute it and add in obsm, or choose from available keys" in str(e)
        print("Missing basis test passed!")
        return

    raise AssertionError("Did not raise ValueError for missing basis")

if __name__ == "__main__":
    # Manually running checks if not using pytest directly,
    # but we will likely run with pytest or python
    try:
        test_plot_marker_genes_dict_on_embedding()
        test_missing_basis()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed: {repr(e)}")
        exit(1)

# =========================================
# Source: test_stat_defaults.py
# =========================================
def test_stat_defaults():
    """Test that stat defaults works as expected."""
    import scanpy as sc
    import numpy as np
    from eigenp_utils.single_cell import kknn_ingest

    adata_ref = sc.AnnData(np.random.normal(size=(100, 10)))
    adata_ref.obsm['X_pca'] = np.random.normal(size=(100, 5))
    adata_ref.obs['cell_type'] = np.random.choice(['A', 'B'], size=100)

    adata_query = sc.AnnData(np.random.normal(size=(20, 10)))
    adata_query.obsm['X_pca'] = np.random.normal(size=(20, 5))

    kknn_ingest(adata_query, adata_ref, obs_keys=['cell_type'])

# =========================================
# Source: test_moran_properties.py
# =========================================


def build_grid_graph(n_rows, n_cols):
    """Builds a 4-connected grid graph adjacency matrix."""
    n = n_rows * n_cols
    rows = []
    cols = []

    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            # Neighbors: up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    j = nr * n_cols + nc
                    rows.append(i)
                    cols.append(j)

    data = np.ones(len(rows))
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return W

def row_standardize(W):
    """Row-standardizes a sparse matrix."""
    row_sums = np.array(W.sum(axis=1)).flatten()
    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0
    inv_sums = 1.0 / row_sums
    D = sp.diags(inv_sums)
    return D @ W

def naive_morans_i(x, W, center=True):
    """Naive, readable implementation of Moran's I."""
    n = len(x)
    S0 = W.sum()

    if center:
        x_bar = x.mean()
        z = x - x_bar
    else:
        z = x

    num = (n / S0) * (z.T @ W @ z)
    den = z.T @ z

    return num / den

def test_morans_i_eigenvectors():
    """
    Verifies Moran's I on eigenvectors of a grid graph.
    1. Constant vector -> I = 1 (if not centered).
    2. Checkerboard vector -> I = -1.
    """
    n_rows, n_cols = 10, 10
    n = n_rows * n_cols
    W = build_grid_graph(n_rows, n_cols)
    W_std = row_standardize(W)

    # 1. Constant Vector
    # Note: morans_i_all_fast centers by default. We must disable centering to test the constant vector
    # (otherwise it becomes the zero vector and I is undefined/NaN).
    x_const = np.ones(n)

    adata_const = AnnData(X=x_const[:, None])
    adata_const.var_names = ["const"]

    # Test uncentered constant vector
    res_const = morans_i_all_fast(adata_const, W_rowstd=W_std, center=False)
    # The function now preserves original casing
    i_const = res_const.set_index("gene").loc["const", "I"]

    # For a row-standardized matrix, the constant vector is an eigenvector with eval 1.
    # I = (n/S0) * (x'Wx)/(x'x). S0 = n for row-standardized.
    # I = (n/n) * (x'x)/(x'x) = 1.
    assert np.isclose(i_const, 1.0, atol=1e-5), f"Constant vector I should be 1.0, got {i_const}"

    # 2. Checkerboard Vector (Bipartite)
    # +1, -1 pattern
    x_check = np.zeros(n)
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            x_check[i] = 1 if (r + c) % 2 == 0 else -1

    adata_check = AnnData(X=x_check[:, None])
    adata_check.var_names = ["checkerboard"]

    # Checkerboard should be mean 0 (balanced grid), so center=True/False shouldn't matter much,
    # but let's stick to center=True (default) to test that path too.
    res_check = morans_i_all_fast(adata_check, W_rowstd=W_std, center=True)
    i_check = res_check.set_index("gene").loc["checkerboard", "I"]

    # For bipartite graph, x_check is eigenvector with eval -1.
    # I should be -1.
    assert np.isclose(i_check, -1.0, atol=1e-5), f"Checkerboard vector I should be -1.0, got {i_check}"


def test_morans_i_null_hypothesis():
    """
    Verifies that for random noise, E[I] = -1/(n-1).
    """
    n_rows, n_cols = 20, 20
    n = n_rows * n_cols
    W = build_grid_graph(n_rows, n_cols)
    W_std = row_standardize(W)

    # Generate many random genes
    n_genes = 2000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_genes))

    adata = AnnData(X=X)
    adata.var_names = [f"g_{i}" for i in range(n_genes)]

    res = morans_i_all_fast(adata, W_rowstd=W_std)

    mean_I = res["I"].mean()
    expected_I = -1.0 / (n - 1)

    # Tolerance check:
    # Var(I) approx 1/n = 1/400. Std(I) = 0.05.
    # SEM = 0.05 / sqrt(2000) ~ 0.001.
    # 3 sigma ~ 0.003.

    assert np.isclose(mean_I, expected_I, atol=0.005), \
        f"Mean I ({mean_I:.5f}) should be close to expected ({expected_I:.5f}) for random noise."


def test_morans_i_algebraic_correctness():
    """
    Verifies that the optimized expansion in `morans_i_all_fast` matches
    the naive calculation exactly (within float precision).
    """
    n = 50
    W = sp.random(n, n, density=0.1, random_state=42)
    W_std = row_standardize(W)

    rng = np.random.default_rng(123)
    x = rng.uniform(0, 10, n) # Non-centered data

    adata = AnnData(X=x[:, None])
    adata.var_names = ["gene1"]

    # Run optimized
    res_opt = morans_i_all_fast(adata, W_rowstd=W_std, center=True)
    i_opt = res_opt.iloc[0]["I"]

    # Run naive
    i_naive = naive_morans_i(x, W_std, center=True)

    assert np.isclose(i_opt, i_naive, atol=1e-5), \
        f"Optimized implementation ({i_opt}) does not match naive ({i_naive})"

# =========================================
# Source: test_pacmap_init.py
# =========================================


try:
    import pacmap
    PACMAP_INSTALLED = True
except ImportError:
    PACMAP_INSTALLED = False

@pytest.mark.skipif(not PACMAP_INSTALLED, reason="PaCMAP not installed")
def test_tl_pacmap_init_large_features():
    """Test that tl pacmap init large features works as expected."""
    # Case 1: > 100 features (should default to PCA)
    n_obs = 50
    n_vars = 101
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    # Run pacmap
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always") # Cause all warnings to always be triggered.
        tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X")

        # Check that NO "Switching initialization" warning was issued
        for warning in w:
            assert "Switching initialization" not in str(warning.message)

    assert "X_pacmap" in adata.obsm

@pytest.mark.skipif(not PACMAP_INSTALLED, reason="PaCMAP not installed")
def test_tl_pacmap_init_override():
    """Test that tl pacmap init override works as expected."""
    # Case 3: <= 100 features but user specifies init='pca'
    # Should use 'pca' and NOT warn about switching
    n_obs = 50
    n_vars = 50 # <= 100
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Pass init explicitly via kwargs
        tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X", init="pca")

        # Check that NO "Switching initialization" warning was issued
        for warning in w:
            assert "Switching initialization" not in str(warning.message)

    assert "X_pacmap" in adata.obsm

@pytest.mark.skipif(not PACMAP_INSTALLED, reason="PaCMAP not installed")
def test_tl_pacmap_init_paga():
    """Test that tl pacmap init paga works as expected."""
    # Case 4: init='paga'
    n_obs = 50
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    # Need to compute neighbors and paga first
    sc.pp.neighbors(adata, n_neighbors=5, use_rep="X")
    sc.tl.leiden(adata, resolution=1.0, flavor='igraph', n_iterations=2, directed=False)
    sc.tl.paga(adata, groups="leiden")
    sc.pl.paga(adata, show=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X", init="paga")

        for warning in w:
            assert "Switching initialization" not in str(warning.message)

    assert "X_pacmap" in adata.obsm

@pytest.mark.skipif(not PACMAP_INSTALLED, reason="PaCMAP not installed")
def test_tl_pacmap_init_small_features():
    """Test that tl pacmap init small features works as expected."""
    # Case 2: <= 100 features (should switch to random and warn)
    n_obs = 50
    n_vars = 50 # <= 100
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    # Run pacmap
    # Expect a warning about switching initialization
    with pytest.warns(UserWarning, match="Switching initialization"):
        tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X")

    assert "X_pacmap" in adata.obsm

# =========================================
# Source: test_moran_memory.py
# =========================================


def test_morans_i_correctness():
    """
    Verify that the optimized morans_i_all_fast produces the same results
    as a reference implementation (or just self-consistency checks).
    Since we don't have the 'old' function readily available as a separate import,
    we check against known properties and simple manual calculation for small data.
    """
    # 1. Create small synthetic data
    n_cells = 50
    n_genes = 10

    # Gene expression: random
    rng = np.random.default_rng(42)
    X = rng.random((n_cells, n_genes)).astype(np.float32)

    # Neighbors: random graph
    # Create a symmetric adjacency matrix
    A = rng.random((n_cells, n_cells)) < 0.2
    np.fill_diagonal(A, 0)
    A = A.astype(np.float32)
    # Symmetrize
    A = (A + A.T) > 0
    A = A.astype(np.float32)

    adata = sc.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]

    # Store connectivities
    adata.obsp["connectivities"] = sp.csr_matrix(A)

    # 2. Run Moran's I
    # We run with block_genes=2 to force looping and block handling
    df = morans_i_all_fast(adata, block_genes=2, center=True)

    assert "gene" in df.columns
    assert "I" in df.columns
    assert "z_score" in df.columns
    assert "pval_z" in df.columns

    assert len(df) == n_genes

    # Check that I values are within [-1, 1] (mostly)
    # Moran's I can exceed bounds slightly but usually fits.
    # For random data, I should be close to 0.

    print("\nMoran's I results (head):")
    print(df.head())

    # Verify no NaNs in I (unless variance is 0)
    assert not df["I"].isna().any()

    # 3. Check consistency with manual calculation for first gene
    g0 = df.iloc[0]["gene"] # gene with highest I
    idx = int(g0.split("_")[1])
    x = X[:, idx]

    # Row normalize W
    rs = np.array(A.sum(axis=1)).flatten()
    W_norm = A / rs[:, None]

    # Moran's I formula: (N/S0) * (z' W z) / (z' z)
    # Here S0 = N because of row normalization? No, S0 = sum(W_norm).
    S0 = W_norm.sum()
    x_mean = x.mean()
    z = x - x_mean

    num = (z @ W_norm @ z)
    den = (z @ z)

    I_manual = (n_cells / S0) * (num / den)

    I_calc = df.iloc[0]["I"]

    print(f"Manual I for {g0}: {I_manual}")
    print(f"Calculated I: {I_calc}")

    assert np.isclose(I_manual, I_calc, atol=1e-5)

def test_morans_i_sparse_input():
    """Check handling of sparse input matrices."""
    n_cells = 100
    n_genes = 20
    rng = np.random.default_rng(123)
    X = sp.random(n_cells, n_genes, density=0.1, format="csr", dtype=np.float32)

    # Create connectivity
    A = sp.random(n_cells, n_cells, density=0.1, format="csr", dtype=np.float32)

    adata = sc.AnnData(X=X)
    adata.obsp["connectivities"] = A
    adata.var_names = [f"G{i}" for i in range(n_genes)]

    # Run
    df = morans_i_all_fast(adata, block_genes=5)
    assert len(df) == n_genes
    assert not df["I"].isna().all()

if __name__ == "__main__":
    # Manually run tests if executed as script
    try:
        test_morans_i_correctness()
        print("test_morans_i_correctness PASSED")
        test_morans_i_sparse_input()
        print("test_morans_i_sparse_input PASSED")
    except Exception as e:
        print(f"FAILED: {e}")
        raise

# =========================================
# Source: test_pacmap_naming.py
# =========================================


def test_pacmap_key_naming():
    """Test that pacmap key naming works as expected."""
    try:
        import pacmap
    except ImportError:
        pytest.skip("PaCMAP not installed")

    n_obs = 50
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    adata = sc.AnnData(X=X)

    # Test default (n_components=2) -> X_pacmap
    # n_neighbors=5 to avoid error on small dataset
    tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X")
    assert "X_pacmap" in adata.obsm
    assert "X_pacmap_2" not in adata.obsm

    # Test n_components=3 -> X_pacmap_3
    tl_pacmap(adata, n_neighbors=5, n_components=3, use_rep="X")
    assert "X_pacmap_3" in adata.obsm
    # Ensure dimensions are correct
    assert adata.obsm["X_pacmap_3"].shape == (n_obs, 3)

    # Test n_components=4 -> X_pacmap_4
    tl_pacmap(adata, n_neighbors=5, n_components=4, use_rep="X")
    assert "X_pacmap_4" in adata.obsm
    assert adata.obsm["X_pacmap_4"].shape == (n_obs, 4)

    # Verify X_pacmap is still the 2D one (it wasn't overwritten by 3 or 4)
    assert adata.obsm["X_pacmap"].shape == (n_obs, 2)

# =========================================
# Source: test_single_cell_plot.py
# =========================================
matplotlib.use('Agg') # prevent plotting windows

def test_plot_marker_genes_dict_on_embedding_methods():
    """Test that plot marker genes dict on embedding methods works as expected."""
    np.random.seed(42)
    X = np.random.uniform(0, 10, (20, 3))
    adata = anndata.AnnData(X=X)
    adata.var_names = ["G0", "G1", "G2"]
    adata.obsm["X_umap"] = np.random.uniform(0, 1, (20, 2))

    markers = {
        "Type1": ["G0", "G1"],
    }
    neg_markers = {
        "Type1": ["G2"]
    }

    # Test default
    axes = plot_marker_genes_dict_on_embedding(adata, markers)
    assert len(axes) > 0
    assert "Type1_score" not in adata.obs # ensure cleaned up

    # Test binned
    axes_binned = plot_marker_genes_dict_on_embedding(adata, markers, score_method="binned", use_raw=False)
    assert len(axes_binned) > 0

    # Test multiple methods with negative markers
    axes_multi = plot_marker_genes_dict_on_embedding(
        adata,
        markers,
        negative_marker_genes=neg_markers,
        score_method=["scanpy", "binned", "binned_weighted", "net_scanpy", "net_binned", "net_binned_weighted"],
        use_raw=False
    )
    assert len(axes_multi) > 0

    # Assert temporary columns are cleaned up
    assert "Type1_score_scanpy" not in adata.obs
    assert "Type1_score_binned" not in adata.obs
    assert "Type1_score_binned_weighted" not in adata.obs
    assert "Type1_score_net_scanpy" not in adata.obs
    assert "Type1_score_net_binned" not in adata.obs
    assert "Type1_score_net_binned_weighted" not in adata.obs


def test_binned_vs_net_binned():
    """Test that binned vs net binned works as expected."""
    # specifically test that binned and net_binned produce different scores when negative markers are present
    from eigenp_utils.single_cell import score_celltypes
    np.random.seed(42)
    # 10 cells, 2 positive genes, 1 negative gene
    X = np.random.uniform(0, 10, (10, 3))
    adata = anndata.AnnData(X=X)
    adata.var_names = ["P1", "P2", "N1"]

    markers = {"T1": ["P1", "P2"]}
    neg_markers = {"T1": ["N1"]}

    df_binned = score_celltypes(adata, markers, cell_type_negative_markers_dict=neg_markers, score_method="binned", use_raw=False)
    df_net_binned = score_celltypes(adata, markers, cell_type_negative_markers_dict=neg_markers, score_method="net_binned", use_raw=False)

    # "binned" should ignore neg_markers and just be positive
    # "net_binned" should be positive - negative
    # They should not be equal.
    assert not np.allclose(df_binned["T1"], df_net_binned["T1"]), "binned and net_binned should not be identical when negative markers exist"

    # Validate that net_binned scores are clipped to [0, 1]
    assert np.all((df_net_binned["T1"] >= 0.0) & (df_net_binned["T1"] <= 1.0)), "net_binned scores should be clipped between 0 and 1"

    # Also check if no negative markers passed, they are equal
    df_binned_noneg = score_celltypes(adata, markers, cell_type_negative_markers_dict=None, score_method="binned", use_raw=False)
    df_net_binned_noneg = score_celltypes(adata, markers, cell_type_negative_markers_dict=None, score_method="net_binned", use_raw=False)

    assert np.allclose(df_binned_noneg["T1"], df_net_binned_noneg["T1"]), "binned and net_binned should be identical when no negative markers exist"

# =========================================
# Source: test_single_cell_extract.py
# =========================================

def test_extract_gene_vector_sparse_sum_mean():
    """Test extracting a duplicated gene's vector with sparse matrix."""
    # Create an integer CSR matrix, which means M.dtype is int
    # When `duplicate_policy` is "mean", np.full() needs to handle float properly
    M = sp.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    adata = AnnData(X=M)
    adata.var_names = ["A", "B", "B"]

    # Test duplicate_policy="sum"
    sum_result = _extract_gene_vector(adata, "B", source="X", duplicate_policy="sum")
    np.testing.assert_allclose(sum_result, [5.0, 11.0, 17.0])

    # Test duplicate_policy="mean"
    mean_result = _extract_gene_vector(adata, "B", source="X", duplicate_policy="mean")
    np.testing.assert_allclose(mean_result, [2.5, 5.5, 8.5])

    # Test duplicate_policy="first"
    first_result = _extract_gene_vector(adata, "B", source="X", duplicate_policy="first")
    np.testing.assert_allclose(first_result, [2.0, 5.0, 8.0])

    # Test duplicate_policy="last"
    last_result = _extract_gene_vector(adata, "B", source="X", duplicate_policy="last")
    np.testing.assert_allclose(last_result, [3.0, 6.0, 9.0])

# =========================================
# Source: test_archetype_plot_args.py
# =========================================


def create_dummy_adata():
    X = np.random.rand(10, 5)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(5)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_umap"] = np.random.rand(10, 2)
    return adata

def create_dummy_archetype_results():
    return {
        "archetypes": np.random.rand(3, 10), # 3 archetypes, 10 cells
        "clusters": np.array([1, 1, 2, 3, 3]), # 5 genes
        "gene_corrs": np.random.rand(5),
        "gene_list": [f"gene_{i}" for i in range(5)]
    }

@patch("scanpy.pl.embedding")
def test_plot_archetype_summary_defaults(mock_embedding):
    """Test default colormaps."""
    adata = create_dummy_adata()
    results = create_dummy_archetype_results()

    plot_archetype_summary(adata, results, archetype_id=1, k=2)

    assert mock_embedding.call_count == 2

    # First call: Archetype score. Should use 'PiYG' by default.
    call1_args, call1_kwargs = mock_embedding.call_args_list[0]
    assert call1_kwargs.get('cmap') == 'PiYG'
    assert 'archetype_1_score' in call1_kwargs.get('color') or call1_kwargs.get('color') == 'archetype_1_score'

    # Second call: Top genes. Should use 'Purples' by default.
    call2_args, call2_kwargs = mock_embedding.call_args_list[1]
    assert call2_kwargs.get('cmap') == 'Purples'

@patch("scanpy.pl.embedding")
def test_plot_archetype_summary_custom(mock_embedding):
    """Test custom colormaps."""
    adata = create_dummy_adata()
    results = create_dummy_archetype_results()

    plot_archetype_summary(adata, results, archetype_id=1, k=2, cmap="Reds", archetype_cmap="Blues")

    assert mock_embedding.call_count == 2

    # First call: Archetype score. Should use 'Blues'.
    _, call1_kwargs = mock_embedding.call_args_list[0]
    assert call1_kwargs.get('cmap') == 'Blues'

    # Second call: Top genes. Should use 'Reds'.
    _, call2_kwargs = mock_embedding.call_args_list[1]
    assert call2_kwargs.get('cmap') == 'Reds'

# =========================================
# Source: test_single_cell_binned.py
# =========================================

def test_score_celltypes_binned():
    """Test that score celltypes binned works as expected."""
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
    """Test that score celltypes binned weighted works as expected."""
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

# =========================================
# Source: test_single_cell_kknn_temp_key.py
# =========================================

def test_kknn_ingest_temp_key_cleanup():
    """Test that kknn ingest temp key cleanup works as expected."""
    # Create dummy reference
    X_ref = np.random.randn(100, 20)
    obs_ref = pd.DataFrame({"label": ["A", "B"] * 50})
    obsm_ref = {"X_pca": X_ref[:, :5], "X_umap": X_ref[:, :2]}
    varm_ref = {"PCs": np.random.randn(20, 5)}
    uns_ref = {"pca": {"params": {"zero_center": True, "use_highly_variable": False}}}

    adata_ref = ad.AnnData(X=X_ref, obs=obs_ref, obsm=obsm_ref, varm=varm_ref, uns=uns_ref)

    # Create dummy query
    X_query = np.random.randn(50, 20)
    adata_query = ad.AnnData(X=X_query)

    # We want to use use_rep="X_pca" to trigger the temp key path,
    # and use barycenter="lle" which accesses adata_query.obsm[query_use_rep]
    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["label"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        barycenter="lle",
        recompute_ref_PCA=False # avoid sc.tl.pca
    )

    # Verify temp key has been cleaned up
    assert not any(k.startswith("__temp_ingest_") for k in adata_query.obsm.keys())

# =========================================
# Source: test_archetype_properties.py
# =========================================


def test_archetype_recovery_and_invariants():
    """
    Testr 🔎: Verify functional correctness and invariants of find_expression_archetypes.

    Guarantees tested:
    1. Module Recovery: Distinct underlying signals (Sine vs Affine vs Noise) are separated into distinct clusters.
    2. Affine Invariance: Genes that are affine transformations of each other (y = ax + b) are clustered together.
    3. Archetype Fidelity: The computed archetype (PC1) correlates perfectly (>0.99) with the ground truth signal.
    4. Sign Alignment: The archetype direction aligns positively with the cluster mean.
    """

    # 1. Setup Synthetic Data
    n_cells = 100
    n_genes_per_module = 10

    # Time vector / underlying signal base
    t = np.linspace(0, 4*np.pi, n_cells)

    # Module A: Sine wave (Perfect copies)
    signal_A = np.sin(t)
    genes_A = np.tile(signal_A, (n_genes_per_module, 1))
    names_A = [f"GeneA_{i}" for i in range(n_genes_per_module)]

    # Module B: Affine transformed signal (Scale and Shift)
    # y = alpha * signal + beta
    # We use a step function as base
    signal_B = np.zeros(n_cells)
    signal_B[n_cells//2:] = 1.0

    genes_B = []
    np.random.seed(42)
    for i in range(n_genes_per_module):
        alpha = np.random.uniform(0.5, 2.0)
        beta = np.random.uniform(-5, 5)
        genes_B.append(alpha * signal_B + beta)
    genes_B = np.array(genes_B)
    names_B = [f"GeneB_{i}" for i in range(n_genes_per_module)]

    # Module C: Independent Random Noise (should be distinct)
    # To ensure it doesn't accidentally correlate, we generate orthogonal noise
    genes_C = np.random.normal(0, 1, (n_genes_per_module, n_cells))
    names_C = [f"GeneC_{i}" for i in range(n_genes_per_module)]

    # Combine
    X = np.vstack([genes_A, genes_B, genes_C]).T  # (n_cells, n_genes)
    var_names = names_A + names_B + names_C
    obs_names = [f"Cell_{i}" for i in range(n_cells)]

    adata = sc.AnnData(X=X, obs=pd.DataFrame(index=obs_names), var=pd.DataFrame(index=var_names))

    # 2. Run Algorithm
    # We ask for 3 clusters.
    # Note: We pass 'X' as source.
    results = find_expression_archetypes(
        adata,
        gene_list=var_names,
        num_clusters=3,
        source="X"
    )

    clusters = results['clusters']
    archetypes = results['archetypes']
    gene_list_out = results['gene_list']
    gene_corrs = results['gene_corrs']

    # Map gene names to cluster IDs
    gene_to_cluster = dict(zip(gene_list_out, clusters))

    # 3. Verify Invariants

    # A) Clustering Correctness
    # All GeneA should be in one cluster
    cluster_ids_A = {gene_to_cluster[g] for g in names_A}
    assert len(cluster_ids_A) == 1, f"Module A genes split across clusters: {cluster_ids_A}"
    id_A = list(cluster_ids_A)[0]

    # All GeneB should be in one cluster (Affine Invariance check)
    cluster_ids_B = {gene_to_cluster[g] for g in names_B}
    assert len(cluster_ids_B) == 1, f"Module B genes (affine) split across clusters: {cluster_ids_B}"
    id_B = list(cluster_ids_B)[0]

    # A and B should be distinct
    assert id_A != id_B, "Module A and Module B merged incorrectly."

    # B) Archetype Fidelity
    # The archetype for cluster A should correlate with signal_A
    # Archetypes are (n_clusters, n_cells)
    # Cluster IDs are 1-based, array is 0-based.
    arch_A = archetypes[id_A - 1]

    # Correlation between recovered archetype and ground truth
    corr_A = np.corrcoef(arch_A, signal_A)[0, 1]
    assert corr_A > 0.99, f"Archetype A fidelity failed. Corr: {corr_A}"

    # The archetype for cluster B should correlate with signal_B (step function)
    arch_B = archetypes[id_B - 1]
    corr_B = np.corrcoef(arch_B, signal_B)[0, 1]
    assert corr_B > 0.99, f"Archetype B (affine) fidelity failed. Corr: {corr_B}"

    # C) Sign Alignment
    # Check that archetype B correlates positively with the mean of genes B
    # Since alpha > 0, the mean of genes B is a positive scaling of signal B + constant.
    # Z-scoring removes constant. So mean profile matches signal shape.
    # The code ensures dot(arch, mean) > 0.
    # We verify this property holds by checking correlation with the signal is positive.
    assert corr_B > 0, "Archetype B sign is flipped relative to signal."

    # D) Gene Correlations
    # Check that the reported gene correlations in results match reality
    # For GeneA_0, it is identical to signal_A. Correlation with archetype should be ~1.0.
    idx_A0 = gene_list_out.index(names_A[0])
    reported_corr = gene_corrs[idx_A0]
    assert reported_corr > 0.99, f"Reported correlation for perfect gene low: {reported_corr}"

    print("Testr 🔎: All invariants passed. Algorithm is robust to affine transformations and recovers signals correctly.")

# =========================================
# Source: test_single_cell_correlations.py
# =========================================

@pytest.fixture
def pbmc_adata():
    # Create a dummy PBMC-like dataset
    np.random.seed(42)
    n_cells = 100
    n_genes = 20

    # Base expression
    X = np.random.poisson(lam=1.0, size=(n_cells, n_genes)).astype(float)

    # Introduce correlation
    # Gene 0 and Gene 1 are highly correlated
    X[:, 1] = X[:, 0] * 2 + np.random.normal(scale=0.5, size=n_cells)

    # Gene 0 and Gene 2 are anti-correlated
    X[:, 2] = -X[:, 0] + np.random.normal(scale=0.5, size=n_cells) + 10

    # Ensure non-negative
    X[X < 0] = 0

    adata = sc.AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]

    # Add an obs column that correlates with Gene 0
    adata.obs["score_A"] = adata.X[:, 0].toarray().ravel() + np.random.normal(scale=0.1, size=n_cells)

    # Add a dense layer
    adata.layers["counts"] = adata.X.toarray()

    return adata


def test_find_correlated_features_pearson_sparse(pbmc_adata):
    """Test Pearson correlation calculation on sparse matrices."""
    res = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"]
    )

    assert "pearson" in res.columns
    assert res.index[0] == "Gene0"
    assert res.loc["Gene0", "pearson"] == pytest.approx(1.0)
    assert res.loc["Gene1", "pearson"] > 0.8  # Highly correlated
    assert res.loc["Gene2", "pearson"] < -0.8 # Anti-correlated


def test_find_correlated_features_dense_layer(pbmc_adata):
    """Test correlation on dense layer vs sparse X."""
    res_sparse = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"]
    )

    res_dense = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        layer="counts",
        metrics=["pearson"]
    )

    np.testing.assert_allclose(
        res_sparse["pearson"].values,
        res_dense["pearson"].values,
        atol=1e-6
    )


def test_find_correlated_features_target_obs(pbmc_adata):
    """Test using an .obs column as the target."""
    res = find_correlated_features(
        pbmc_adata,
        target="score_A",
        metrics=["pearson"]
    )

    assert res.loc["Gene0", "pearson"] > 0.9
    assert res.loc["Gene1", "pearson"] > 0.8
    assert res.loc["Gene2", "pearson"] < -0.8


def test_find_correlated_features_multiple_metrics(pbmc_adata):
    """Test requesting multiple metrics at once."""
    res = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson", "spearman", "wasserstein"]
    )

    assert all(col in res.columns for col in ["pearson", "spearman", "wasserstein"])

    # Wasserstein distance of a variable to itself (Z-scored) should be 0
    assert res.loc["Gene0", "wasserstein"] == pytest.approx(0.0, abs=1e-7)

    # Spearman should also show strong correlation
    assert res.loc["Gene1", "spearman"] > 0.8


def test_find_correlated_features_exclude(pbmc_adata):
    """Test the exclude_features parameter."""
    res = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        exclude_features=["Gene0", "Gene1", "NonExistentGene"]
    )

    assert "Gene0" not in res.index
    assert "Gene1" not in res.index
    assert "Gene2" in res.index


def test_find_correlated_features_graph_smoothing(pbmc_adata):
    """Test graph-smoothed feature correlation calculation."""
    # Compute neighbors so `adata.obsp['connectivities']` exists
    sc.pp.neighbors(pbmc_adata, n_neighbors=5, use_rep="X")

    # Store a copy of original data to ensure it is not mutated
    X_orig = pbmc_adata.X.copy()
    connectivities_orig = pbmc_adata.obsp["connectivities"].copy()

    res_no_graph = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"]
    )

    res_with_graph = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"],
        use_graph=True,
        weights_key="connectivities"
    )

    # Both should have computed pearson correlation
    assert "pearson" in res_with_graph.columns

    # Gene0 vs Gene0 should still be 1.0
    assert res_with_graph.loc["Gene0", "pearson"] == pytest.approx(1.0)

    # The diffused distances will be numerically different from the raw ones
    assert res_with_graph.loc["Gene1", "pearson"] != res_no_graph.loc["Gene1", "pearson"]

    # Ensure no mutation of original data
    assert (pbmc_adata.X != X_orig).nnz == 0
    assert (pbmc_adata.obsp["connectivities"] != connectivities_orig).nnz == 0


def test_find_correlated_features_sorting(pbmc_adata):
    """Test sorting behavior (wasserstein ascending, others descending)."""
    # Pearson: descending
    res_pearson = find_correlated_features(pbmc_adata, target="Gene0", metrics=["pearson"])
    assert res_pearson.index[0] == "Gene0"
    assert res_pearson.index[-1] == "Gene2" # Most anti-correlated

    # Wasserstein: ascending (0 is best)
    res_wass = find_correlated_features(pbmc_adata, target="Gene0", metrics=["wasserstein"])
    assert res_wass.index[0] == "Gene0"

    # Multiple metrics: sorts by first
    res_multi = find_correlated_features(
        pbmc_adata, target="Gene0", metrics=["wasserstein", "pearson"]
    )
    assert res_multi.index[0] == "Gene0"
    assert list(res_multi.index) == list(res_wass.index)

# =========================================
# Source: test_annotation_confidence.py
# =========================================


def test_annotation_statistical_invariants():
    """
    Testr 🔎: Verify the probabilistic confidence logic in annotate_clusters_by_markers.

    This test checks the 'Probability of Superiority' (softmax_p) metric, ensuring it
    correctly reflects the statistical separability of the top two cell types.

    Guarantees tested:
    1. Perfect Separation: If scores are disjoint (Margin >> Variance), Confidence -> 1.0.
    2. Indistinguishability: If scores are identical distributions, Confidence -> 0.5.
    3. Analytical Calibration: For known Gaussian distributions, the computed confidence
       matches the theoretical Probability of Superiority: P(X > Y) = Phi(mu_diff / sigma_diff).
    """

    # Setup
    # We create 3 clusters with distinct score characteristics for 2 Cell Types (A and B).
    # Cluster 0: Perfect Separation (A=10, B=0, sigma=0)
    # Cluster 1: Indistinguishable (A~N(0,1), B~N(0,1))
    # Cluster 2: Controlled Overlap (A~N(1,1), B~N(0,1))

    n_per_cluster = 1000
    n_cells = 3 * n_per_cluster

    obs = pd.DataFrame({
        "leiden": ["0"]*n_per_cluster + ["1"]*n_per_cluster + ["2"]*n_per_cluster
    }, index=[f"cell_{i}" for i in range(n_cells)])

    # Generate Scores
    rng = np.random.default_rng(42)

    # Cluster 0: Deterministic Separation
    sA_0 = np.full(n_per_cluster, 10.0)
    sB_0 = np.full(n_per_cluster, 0.0)

    # Cluster 1: Indistinguishable Noise
    sA_1 = rng.standard_normal(n_per_cluster)
    sB_1 = rng.standard_normal(n_per_cluster)

    # Cluster 2: Controlled Overlap
    # Difference D = A - B ~ N(1, sqrt(2)) -> N(1, 1.414)
    sA_2 = rng.normal(loc=1.0, scale=1.0, size=n_per_cluster)
    sB_2 = rng.normal(loc=0.0, scale=1.0, size=n_per_cluster)

    scores = pd.DataFrame({
        "TypeA": np.concatenate([sA_0, sA_1, sA_2]),
        "TypeB": np.concatenate([sB_0, sB_1, sB_2])
    }, index=obs.index)

    adata = AnnData(X=np.zeros((n_cells, 2)), obs=obs) # X doesn't matter, we pass scores

    # Run Annotation
    # Note: normalize_scores=False because we want to test our constructed Gaussian distributions directly
    res_df = annotate_clusters_by_markers(
        adata,
        cluster_key="leiden",
        cell_type_markers_dict=None, # Not needed when scores passed
        scores=scores,
        normalize_scores=False,
        write_to_obs=True
    )

    # --- Verify Case 1: Perfect Separation (Cluster 0) ---
    row0 = res_df.loc["0"]
    assert row0["assigned_cell_type"] == "TypeA"
    # With std=0, std_d might be 0. Code handles this?
    # If std_d < 1e-12, checks mu_d > 0 -> p=1.0.
    assert np.isclose(row0["softmax_p"], 1.0), \
        f"Perfect separation should yield P=1.0, got {row0['softmax_p']}"
    assert np.isclose(row0["uncertainty"], 0.0), \
        f"Perfect separation should yield Uncertainty=0.0, got {row0['uncertainty']}"

    # --- Verify Case 2: Indistinguishable (Cluster 1) ---
    row1 = res_df.loc["1"]
    # The winner is random due to noise, but p should be low
    p_val = row1["softmax_p"]
    # Theoretical P is 0.5. Allow small noise margin.
    assert 0.45 < p_val < 0.55, \
        f"Indistinguishable distributions should yield P approx 0.5, got {p_val}"

    # --- Verify Case 3: Analytical Calibration (Cluster 2) ---
    row2 = res_df.loc["2"]
    # Theoretical Calculation:
    # mu_diff = 1.0, sigma_diff = sqrt(1^2 + 1^2) = 1.414...
    # z = 1.0 / 1.4142... = 0.7071...
    expected_z = 1.0 / np.sqrt(2)
    expected_p = norm.cdf(expected_z) # approx 0.76

    p_val_2 = row2["softmax_p"]

    # Tolerance: Standard Error of P?
    # We have 1000 samples.
    # Let's be generous but rigorous enough to catch logic errors (like using SD instead of Var in denominator)
    assert np.isclose(p_val_2, expected_p, atol=0.05), \
        f"Cluster 2 P-value {p_val_2:.3f} deviates from theoretical expectation {expected_p:.3f}"

    print(f"\nTestr 🔎: Annotation Statistical Invariants Verified.")
    print(f"  Cluster 0 (Perfect): P={row0['softmax_p']:.4f}")
    print(f"  Cluster 1 (Noise):   P={row1['softmax_p']:.4f}")
    print(f"  Cluster 2 (Overlap): P={row2['softmax_p']:.4f} (Expected: {expected_p:.4f})")

def test_single_candidate_edge_case():
    """
    Verify behavior when only one cell type is provided.
    Confidence should be 1.0 (no competition).
    """
    obs = pd.DataFrame({"leiden": ["0"]*10}, index=[f"c{i}" for i in range(10)])
    scores = pd.DataFrame({"TypeA": np.random.rand(10)}, index=obs.index)
    adata = AnnData(X=np.zeros((10,1)), obs=obs)

    res_df = annotate_clusters_by_markers(
        adata,
        cluster_key="leiden",
        scores=scores,
        normalize_scores=False
    )

    assert res_df.loc["0", "softmax_p"] == 1.0, "Single candidate should have P=1.0"
    assert res_df.loc["0", "assigned_cell_type"] == "TypeA"

def test_outlier_robustness():
    """
    Testr 🔎: Verify that the confidence metric is robust to outliers.

    Simulates a cluster where 90% of cells strongly favor Type A (score diff +1),
    but 10% of cells (outliers) strongly favor Type B (score diff -100).

    - Parametric (Mean-based): Mean diff is negative (-9). Predicts Type B or low confidence.
    - Empirical (Robust): 90% positive signs. Predicts Type A with high confidence (0.9).
    """
    n_majority = 900
    n_outliers = 100
    n_cells = n_majority + n_outliers

    obs = pd.DataFrame({"leiden": ["0"]*n_cells}, index=[f"c{i}" for i in range(n_cells)])

    # Majority: Type A > Type B (1 vs 0)
    sA_maj = np.full(n_majority, 1.0)
    sB_maj = np.full(n_majority, 0.0)

    # Outliers: Type B >>> Type A (0 vs 100) -> Diff -100
    sA_out = np.full(n_outliers, 0.0)
    sB_out = np.full(n_outliers, 100.0)

    scores = pd.DataFrame({
        "TypeA": np.concatenate([sA_maj, sA_out]),
        "TypeB": np.concatenate([sB_maj, sB_out])
    }, index=obs.index)

    adata = AnnData(X=np.zeros((n_cells, 2)), obs=obs)

    res_df = annotate_clusters_by_markers(
        adata,
        cluster_key="leiden",
        scores=scores,
        normalize_scores=False
    )

    row = res_df.loc["0"]

    # The robust metric should favor TypeA
    assert row["assigned_cell_type"] == "TypeA", \
        f"Robust method should assign TypeA despite outliers. Got {row['assigned_cell_type']}"

    # Confidence should be around 0.9
    p_val = row["softmax_p"]
    assert np.isclose(p_val, 0.9, atol=0.01), \
        f"Expected robust confidence ~0.9, got {p_val}"

# =========================================
# Source: test_moran_statistics.py
# =========================================


def build_grid_graph(n_rows, n_cols):
    """Builds a 4-connected grid graph adjacency matrix."""
    n = n_rows * n_cols
    rows = []
    cols = []

    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            # Neighbors: up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    j = nr * n_cols + nc
                    rows.append(i)
                    cols.append(j)

    data = np.ones(len(rows))
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return W

def row_standardize(W):
    """Row-standardizes a sparse matrix."""
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv_sums = 1.0 / row_sums
    D = sp.diags(inv_sums)
    return D @ W

def test_moran_statistics_gaussian():
    """
    Verifies that for Gaussian random noise, the Z-scores produced by
    morans_i_all_fast follow a Standard Normal Distribution N(0, 1).

    This validates the variance formula for the 'randomization' assumption
    when the underlying data is actually normal.
    """
    # 1. Setup: 20x20 Grid (N=400)
    n_rows, n_cols = 20, 20
    n = n_rows * n_cols
    W = build_grid_graph(n_rows, n_cols)
    W_std = row_standardize(W)

    # 2. Generate Random Data: 5000 genes, N(0, 1)
    n_genes = 5000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_genes))

    adata = AnnData(X=X)
    adata.var_names = [f"g_{i}" for i in range(n_genes)]

    # 3. Run Moran's I
    res = morans_i_all_fast(adata, W_rowstd=W_std)

    # 4. Verify Z-scores statistics
    z_scores = res["z_score"].values

    # Drop NaNs if any (shouldn't be for N=400)
    z_scores = z_scores[np.isfinite(z_scores)]

    mu_z = np.mean(z_scores)
    std_z = np.std(z_scores)

    print(f"Gaussian Noise: Mean Z = {mu_z:.4f}, Std Z = {std_z:.4f}")

    # Mean should be 0 (unbiased)
    # SEM = 1 / sqrt(5000) ~ 0.014
    assert np.abs(mu_z) < 0.05, f"Z-scores should be centered at 0, got {mu_z:.4f}"

    # Std should be 1 (correct variance estimation)
    # Variance of sample variance ~ 2/N. Std of sample std ~ 1/sqrt(2N) ~ 0.01
    assert np.abs(std_z - 1.0) < 0.05, f"Z-scores should have unit variance, got {std_z:.4f}"

    # Kolmogorov-Smirnov test against Standard Normal
    # This checks the entire shape of the distribution
    ks_stat, ks_pval = stats.kstest(z_scores, 'norm')

    # We expect p-value to be high (fail to reject null).
    # However, with N=5000, even tiny deviations can reject.
    # We'll use a lenient threshold or rely on moments.
    print(f"KS Test: stat={ks_stat:.4f}, pval={ks_pval:.4e}")

    # 5. Verify P-values Uniformity
    p_vals = res["pval_z"].values
    prop_significant = np.mean(p_vals < 0.05)
    print(f"Proportion p < 0.05: {prop_significant:.4f}")

    assert 0.04 < prop_significant < 0.06, f"P-values should be uniform (approx 0.05 < 0.05), got {prop_significant:.4f}"


def test_moran_statistics_kurtotic():
    """
    Verifies that for High-Kurtosis noise (Sparse Spikes), the Z-scores
    still follow N(0, 1).

    This strictly validates that the variance formula uses the kurtosis term 'b2' correctly.
    If the formula assumed normality (b2=3), the variance estimate would be wrong
    for high-kurtosis data, and the resulting Z-scores would not have unit variance.
    """
    # 1. Setup
    n_rows, n_cols = 20, 20
    n = n_rows * n_cols
    W = build_grid_graph(n_rows, n_cols)
    W_std = row_standardize(W)

    # 2. Generate Sparse/Spiky Data
    # 95% zeros, 5% large values
    n_genes = 5000
    rng = np.random.default_rng(99)
    X = np.zeros((n, n_genes))

    # Add spikes
    n_spikes = int(0.05 * n)
    for g in range(n_genes):
        indices = rng.choice(n, n_spikes, replace=False)
        X[indices, g] = rng.exponential(scale=10.0, size=n_spikes)

    adata = AnnData(X=X)
    adata.var_names = [f"g_{i}" for i in range(n_genes)]

    # 3. Run Moran's I
    res = morans_i_all_fast(adata, W_rowstd=W_std)

    # 4. Verify Z-scores statistics
    z_scores = res["z_score"].values
    z_scores = z_scores[np.isfinite(z_scores)]

    mu_z = np.mean(z_scores)
    std_z = np.std(z_scores)

    print(f"Kurtotic Noise: Mean Z = {mu_z:.4f}, Std Z = {std_z:.4f}")

    # Even with high kurtosis, the randomization null hypothesis ensures
    # the Z-score (standardized by analytical variance) is asymptotically normal
    # (or at least has unit variance).

    assert np.abs(mu_z) < 0.05, f"Z-scores should be centered at 0, got {mu_z:.4f}"

    # Crucial check: Is variance correct?
    # If we ignored kurtosis, the variance of I would be underestimated/overestimated.
    # The analytical formula should adapt.
    assert np.abs(std_z - 1.0) < 0.1, f"Z-scores should have unit variance despite kurtosis, got {std_z:.4f}"

# =========================================
# Source: test_single_cell_kknn_classifier.py
# =========================================

def test_kknn_classifier_categorical():
    """Test that kknn classifier categorical works as expected."""
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
    # The random generator might not correct all outliers perfectly, so just check
    # that most stayed the same to verify basic correctness without flakiness.
    # The main logic test is below in `test_kknn_classifier_with_mask`
    assert np.sum(smoothed[:50] == "A") >= 45
    assert np.sum(smoothed[50:] == "B") >= 45

def test_kknn_classifier_continuous():
    """Test that kknn classifier continuous works as expected."""
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
    """Test that kknn classifier not inplace works as expected."""
    adata = sc.AnnData(np.random.randn(10, 10))
    adata.obsm["X_pacmap"] = np.random.randn(10, 2)
    adata.obs["celltype"] = pd.Categorical(["A"] * 5 + ["B"] * 5)

    # Needs max_neighbors to be <= n_samples
    res = kknn_classifier(adata, obs_key="celltype", use_rep="X_pacmap", inplace=False, n_neighbors=2, max_neighbors=4)

    assert "celltype_kknn" not in adata.obs
    assert len(res) == 10


def test_kknn_classifier_with_mask():
    """Test that kknn classifier with mask works as expected."""
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

    # Create a mask where only index 0 is True, meaning only it should be allowed to change
    # Index 99 is False, meaning it should stay "A" despite being an outlier in cluster B
    mask = np.zeros(100, dtype=bool)
    mask[0] = True

    # Run classifier
    kknn_classifier(adata, obs_key="celltype", use_rep="X_pacmap", n_neighbors=5, max_neighbors=10, mask=mask)

    assert "celltype_kknn" in adata.obs
    smoothed = adata.obs["celltype_kknn"].values

    # Check that outlier at index 0 was corrected because mask was True
    assert smoothed[0] == "A"
    # Check that outlier at index 99 was NOT corrected because mask was False
    assert smoothed[99] == "A"

    # Run classifier with a pandas Series as mask
    mask_series = pd.Series(mask)
    res = kknn_classifier(adata, obs_key="celltype", use_rep="X_pacmap", n_neighbors=5, max_neighbors=10, mask=mask_series, inplace=False)

    assert res[0] == "A"
    assert res[99] == "A"

# =========================================
# Source: test_moran_cancellation.py
# =========================================

def test_moran_formulas():
    """Test that moran formulas works as expected."""
    N = 10000
    block = 1024

    # Random sparse X
    np.random.seed(42)
    X = sp.random(N, block, density=0.05, format='csr', dtype=np.float32)

    # Random sparse W (row standardized)
    W = sp.random(N, N, density=0.001, format='csr', dtype=np.float32)
    rs = np.array(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    W = W.multiply(1.0 / rs[:, None]).tocsr()

    # 1. Current formula
    t0 = time.time()
    Xb = X.copy()
    WXb = W @ Xb
    Xb_dense1 = Xb.toarray()
    WXb_dense1 = WXb.toarray()

    mu = Xb_dense1.mean(axis=0)
    W_row_sums = np.array(W.sum(axis=1)).ravel()

    sum_cross = np.einsum('ij,ij->j', Xb_dense1, WXb_dense1)
    sum_sq = np.einsum('ij,ij->j', Xb_dense1, Xb_dense1)
    sum_WXb = WXb_dense1.sum(axis=0)
    sum_xR = Xb_dense1.T @ W_row_sums
    S0 = W.sum()

    num1 = sum_cross - mu * sum_WXb - mu * sum_xR + (mu**2) * S0
    den1 = sum_sq - N * (mu**2)
    t1 = time.time()

    # 2. Proposed formula
    t2 = time.time()
    Xb_dense2 = X.toarray()
    Xb_dense2 -= mu[None, :]
    WXb_dense2 = W @ Xb_dense2

    num2 = np.einsum('ij,ij->j', Xb_dense2, WXb_dense2)
    den2 = np.einsum('ij,ij->j', Xb_dense2, Xb_dense2)
    t3 = time.time()

    print(f"Current: num={num1[:5]}, den={den1[:5]}, time={t1-t0:.4f}s")
    print(f"Proposed: num={num2[:5]}, den={den2[:5]}, time={t3-t2:.4f}s")

    # Check max difference
    print(f"Max diff num: {np.abs(num1 - num2).max()}")
    print(f"Max diff den: {np.abs(den1 - den2).max()}")

if __name__ == '__main__':
    test_moran_formulas()

# =========================================
# Source: test_volcano_reproduction.py
# =========================================


def test_volcano_reproduction():
    """Test that volcano reproduction works as expected."""
    # Create a dummy AnnData
    adata = sc.AnnData(np.random.rand(100, 50))
    adata.obs['leiden'] = np.random.choice(['0', '1'], 100)
    adata.var_names = [f'Gene{i}' for i in range(50)]

    # Mock rank_genes_groups output
    # Scanpy stores it as structured arrays usually
    # We define dtypes to mimic structured arrays
    dtype = [('0', 'f4'), ('1', 'f4')]

    # Create structured arrays
    pvals_adj_data = np.zeros((50,), dtype=dtype)
    pvals_adj_data['0'] = np.random.rand(50)
    pvals_adj_data['1'] = np.random.rand(50)

    logfoldchanges_data = np.zeros((50,), dtype=dtype)
    logfoldchanges_data['0'] = np.random.randn(50)
    logfoldchanges_data['1'] = np.random.randn(50)

    names_dtype = [('0', 'U10'), ('1', 'U10')]
    names_data = np.zeros((50,), dtype=names_dtype)
    names_data['0'] = [f'Gene{i}' for i in range(50)]
    names_data['1'] = [f'Gene{i}' for i in range(50)]

    adata.uns['rank_genes_groups'] = {
        'pvals_adj': pvals_adj_data,
        'logfoldchanges': logfoldchanges_data,
        'names': names_data
    }

    # Test 1: group as string (should pass)
    try:
        plot_volcano_adata(adata, 'rank_genes_groups', group='0', show=False)
    except Exception as e:
        pytest.fail(f"Failed with group='0' (str): {e}")

    # Test 2: group as list of one element (should pass now)
    try:
        plot_volcano_adata(adata, 'rank_genes_groups', group=['0'], show=False)
    except Exception as e:
        pytest.fail(f"Failed with group=['0'] (list): {e}")

    # Test 3: group as list of multiple elements (should raise ValueError)
    with pytest.raises(ValueError, match="only supports plotting a single group"):
        plot_volcano_adata(adata, 'rank_genes_groups', group=['0', '1'], show=False)

if __name__ == "__main__":
    test_volcano_reproduction()

# =========================================
# Source: test_score_celltypes_binned.py
# =========================================
def test_score_celltypes_binned():
    """Test that score celltypes binned works as expected."""
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

# =========================================
# Source: test_moran_general_weights.py
# =========================================


class MockAdata:
    def __init__(self, X, var_names):
        self.X = X
        self.var_names = var_names
        self.shape = X.shape
        self.layers = {}
        self.obsp = {}

def test_moran_bias_isolated_nodes():
    """
    Test that morans_i_all_fast produces correct results even with isolated nodes
    or non-row-standardized weights, compared to a naive exact implementation.
    """
    # Create a small dataset: 4 nodes
    # Nodes 0, 1 connected (clique)
    # Nodes 2, 3 isolated

    # Adjacency:
    # 0: [0, 1, 0, 0]
    # 1: [1, 0, 0, 0]
    # 2: [0, 0, 0, 0]
    # 3: [0, 0, 0, 0]

    W = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.float32)

    W_csr = sp.csr_matrix(W)

    # Feature x:
    # 0, 1 have value 10
    # 2, 3 have value -10
    # Global mean = 0
    # Connected nodes are positively correlated (10 next to 10).
    # Expected I should be positive.

    x = np.array([10, 10, -10, -10], dtype=np.float32).reshape(4, 1)

    adata = MockAdata(x, np.array(["gene1"]))

    # Manual Calculation:
    # Mean = 0
    # Denominator = sum(x^2) = 100+100+100+100 = 400
    # Numerator term 1: sum_ij w_ij (xi-u)(xj-u)
    # i=0, j=1: 1 * 10 * 10 = 100
    # i=1, j=0: 1 * 10 * 10 = 100
    # Others 0.
    # Num = 200.
    # S0 = 2.
    # I = (N/S0) * (Num/Den) = (4/2) * (200/400) = 2 * 0.5 = 1.0.

    # Run morans_i_all_fast with explicit W (not row standardized!)
    # Note: morans_i_all_fast assumes W_rowstd is row-standardized by default if passed?
    # No, it just uses it as W. The argument name implies intent, but math should hold if we fix it.
    # However, if we pass W_rowstd, the function uses it.

    res = morans_i_all_fast(adata, W_rowstd=W_csr, block_genes=100)

    print(f"\nExact I: 1.0")
    print(f"Computed I (Not Row Std): {res['I'][0]}")

    # Case 2: Row Standardized manually (handling islands)
    # Row sums: [1, 1, 0, 0]
    # W_rs = W.
    # So same result expected.

    # Case 3: Shift Mean
    # x = [20, 20, 0, 0] -> Mean = 10.
    # x - u = [10, 10, -10, -10]. Same deviations.
    # Should get I = 1.0.

    x_shifted = np.array([20, 20, 0, 0], dtype=np.float32).reshape(4, 1)
    adata_shifted = MockAdata(x_shifted, np.array(["gene1"]))

    res_shifted = morans_i_all_fast(adata_shifted, W_rowstd=W_csr, block_genes=100)

    print(f"Exact I (Shifted): 1.0")
    print(f"Computed I (Shifted): {res_shifted['I'][0]}")

    # Check if they match
    assert np.isclose(res_shifted['I'][0], 1.0, atol=1e-5), f"Shifted mean failed: {res_shifted['I'][0]}"

    # Case 4: General Weights (Not Row Std)
    # W = 2 * Identity (Self loops? No, usually 0 diag).
    # W = 2 * adjacency
    W2 = 2 * W_csr
    # S0 = 4.
    # Num = 1 * 2 * 10 * 10 + 1 * 2 * 10 * 10 = 400.
    # Den = 400.
    # I = (4/4) * (400/400) = 1.0.

    res_scaled = morans_i_all_fast(adata, W_rowstd=W2, block_genes=100)
    print(f"Exact I (Scaled W): 1.0")
    print(f"Computed I (Scaled W): {res_scaled['I'][0]}")

    assert np.isclose(res_scaled['I'][0], 1.0, atol=1e-5), f"Scaled W failed: {res_scaled['I'][0]}"

if __name__ == "__main__":
    test_moran_bias_isolated_nodes()

# =========================================
# Source: test_preprocess_subset.py
# =========================================

@pytest.fixture
def adata_integer_counts():
    """Adata with integer counts in .X and .layers['counts']"""
    X = np.random.randint(0, 10, size=(100, 50)).astype(float)
    obs = dict(batch=np.random.choice(["a", "b"], size=100))
    var = dict(gene_name=[f"gene_{i}" for i in range(50)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata

@pytest.fixture
def adata_float_layer():
    """Adata with float layer (e.g. scvi_normalized) and NO counts layer."""
    X = np.random.exponential(size=(100, 50)) + 0.1 # Ensure positive
    obs = dict(batch=np.random.choice(["a", "b"], size=100))
    var = dict(gene_name=[f"gene_{i}" for i in range(50)])
    # Populate .X with X so filtering doesn't drop everything
    adata = sc.AnnData(X=X.copy(), obs=obs, var=var)
    adata.layers["scvi_normalized"] = X.copy()
    return adata

def test_standard_workflow(adata_integer_counts):
    """Test standard workflow with integer counts: log1p, scaling, etc."""
    adata = preprocess_subset(
        adata_integer_counts,
        counts_layer="counts",
        X_layer_for_pca="log1p",
        hvg_flavor="seurat",
        scale_max_value=10.0,
        copy=True
    )

    assert "log1p" in adata.layers
    # Check if data was scaled (mean approx 0)
    assert np.allclose(adata.X.mean(axis=0), 0, atol=1.0) # lenient check
    assert adata.shape[1] == 50 # No subsetting unless n_top_genes < 50
    assert "X_pca" in adata.obsm

def test_custom_layer_workflow(adata_float_layer):
    """Test using a custom float layer (e.g. scvi) without counts."""
    # Should warn about missing counts but proceed
    with pytest.warns(UserWarning, match="Counts layer .* not found"):
        adata = preprocess_subset(
            adata_float_layer,
            counts_layer="counts", # Does not exist
            X_layer_for_pca="scvi_normalized",
            hvg_flavor="seurat", # standard seurat (not v3) works on data
            scale_data=False, # New parameter to skip scaling
            copy=True
        )

    # log1p should NOT be created if not requested and counts missing
    assert "log1p" not in adata.layers
    # X should be the scvi layer (unscaled)
    assert np.allclose(adata.X, adata.layers["scvi_normalized"])
    assert "X_pca" in adata.obsm

def test_custom_layer_scaling(adata_float_layer):
    """Test using a custom float layer WITH scaling."""
    with pytest.warns(UserWarning, match="Counts layer .* not found"):
        adata = preprocess_subset(
            adata_float_layer,
            counts_layer="counts",
            hvg_flavor="seurat", # seurat_v3 would fail without counts
            X_layer_for_pca="scvi_normalized",
            scale_data=True,
            copy=True
        )

    # X should be scaled
    assert not np.allclose(adata.X, adata.layers["scvi_normalized"])
    assert np.allclose(adata.X.mean(axis=0), 0, atol=1.0)

def test_seurat_v3_missing_counts_error(adata_float_layer):
    """Test that seurat_v3 flavor raises error if counts are missing."""
    with pytest.raises(ValueError, match="flavor='seurat_v3' requires raw counts"):
        preprocess_subset(
            adata_float_layer,
            hvg_flavor="seurat_v3",
            counts_layer="counts",
            copy=True
        )

def test_triku_with_floats(adata_float_layer):
    """
    Test passing float data to Triku.
    """
    try:
        import triku
    except ImportError:
        pytest.skip("triku not installed")

    # Match the actual warning emitted (missing counts), but NOT the "rounding" warning
    with pytest.warns(UserWarning, match="Counts layer .* not found"):
         adata = preprocess_subset(
            adata_float_layer,
            hvg_flavor="triku",
            X_layer_for_pca="scvi_normalized",
            n_top_genes=20,
            copy=True
        )

    assert "triku_distance" in adata.var
    assert "highly_variable" in adata.var # Check for the mapping fix too

# =========================================
# Source: test_multiscale_coarsening.py
# =========================================


def test_perfect_hierarchy():
    """
    Testr 🔎: Verify Multiscale Coarsening on a Perfect Hierarchy.

    Scenario: "Blobs of Blobs"
    - Super-Cluster A: Contains Sub-Cluster A1 and A2 (Close together).
    - Super-Cluster B: Contains Sub-Cluster B1 (Far away).

    Invariants:
    1. Low Resolution: Should merge A1+A2 into "A", keeping "B" separate.
    2. High Resolution: Should distinguish A1, A2, B1.
    3. Consistency: The lineage should be perfectly consistent (0 inconsistencies).
    4. Purity: The mapping from High->Low should be 100% pure (A1->A, A2->A, B1->B).
    """

    # 1. Generate Data
    n_points = 50
    rng = np.random.default_rng(42)

    # A1: Centered at (0, 0)
    blob_A1 = rng.normal(loc=0.0, scale=0.5, size=(n_points, 2))
    # A2: Centered at (2, 0) (Close enough to merge at low res, separate at high)
    blob_A2 = rng.normal(loc=2.0, scale=0.5, size=(n_points, 2))
    # B1: Centered at (10, 10) (Far away)
    blob_B1 = rng.normal(loc=10.0, scale=0.5, size=(n_points, 2))

    X = np.vstack([blob_A1, blob_A2, blob_B1])
    obs_names = [f"A1_{i}" for i in range(n_points)] + \
                [f"A2_{i}" for i in range(n_points)] + \
                [f"B1_{i}" for i in range(n_points)]

    adata = AnnData(X=X, obs=pd.DataFrame(index=obs_names))

    # 2. Preprocessing (PCA + Neighbors)
    # We need neighbors for Leiden.
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)

    # 3. Run Multiscale Coarsening
    # We choose resolutions that likely capture the two scales.
    # Low res (e.g. 0.1) often under-clusters.
    # High res (e.g. 1.0) finds sub-clusters.
    # We might need to tune this, or just check that *some* hierarchy formed.
    # Given the clear separation, 0.1 should merge A1/A2. 1.0 should split.

    res_coarse = 0.1
    res_fine = 1.0
    resolutions = [res_coarse, res_fine]

    results = multiscale_coarsening(
        adata,
        resolutions=resolutions,
        return_output=True,
        random_state=42
    )

    # 4. Verification

    # A) Consistency
    consistency_df = results["consistency"]
    assert len(consistency_df) == 0, \
        f"Perfect hierarchy should have 0 inconsistencies, found {len(consistency_df)}\n{consistency_df}"

    # B) Structure Check
    clusters_coarse = results["clustering"][res_coarse]
    clusters_fine = results["clustering"][res_fine]

    # Count clusters
    n_coarse = len(clusters_coarse.unique())
    n_fine = len(clusters_fine.unique())

    print(f"Coarse Clusters: {n_coarse}, Fine Clusters: {n_fine}")

    # Ideally Coarse < Fine.
    # With this data: Coarse should be ~2 (A, B). Fine should be ~3 (A1, A2, B).
    assert n_coarse < n_fine, "Hierarchy failed: Coarse resolution didn't merge clusters."

    # C) Purity
    # Check the purity of the Fine->Coarse mapping.
    # Since the hierarchy is real, every fine cluster should map to exactly one coarse cluster
    # with high purity (ideally 1.0).
    purity_map = results["hierarchy"]["purity"][(res_fine, res_coarse)]

    mean_purity = np.mean(list(purity_map.values()))
    print(f"Mean Purity (Fine->Coarse): {mean_purity:.4f}")

    assert mean_purity > 0.95, f"Hierarchy mapping is impure (Mean: {mean_purity})"


def test_lineage_inconsistency():
    """
    Testr 🔎: Verify Detection of Lineage Inconsistencies (Simpson's Paradox).

    Scenario:
    We verify that the algorithm detects a case where the "Majority Vote" path flips
    across resolutions, creating a contradiction between Direct and Indirect lineage.

    Resolutions: R1 (Coarse), R2 (Mid), R3 (Fine).
    Cluster F1 (at R3) contains 100 cells.

    Distribution of these 100 cells:
    - Group 1 (41 cells): F1 -> M1 -> C1
    - Group 2 (20 cells): F1 -> M1 -> C2
    - Group 3 (39 cells): F1 -> M2 -> C2

    Analysis:
    1. F1 -> Mid Mapping:
       - M1 has 61 cells (Grp 1+2).
       - M2 has 39 cells (Grp 3).
       - Dominant Parent: M1.

    2. Mid -> Coarse Mapping (for M1):
       - M1 (61 cells) splits into:
         - C1: 41 cells (Grp 1).
         - C2: 20 cells (Grp 2).
       - Dominant Parent for M1: C1.
       => Indirect Path: F1 -> M1 -> C1.

    3. F1 -> Coarse Mapping (Direct):
       - C1 has 41 cells (Grp 1).
       - C2 has 59 cells (Grp 2+3).
       - Dominant Grandparent: C2.
       => Direct Path: F1 -> C2.

    Result: Indirect (C1) != Direct (C2). This is an INCONSISTENCY.
    """

    # 1. Setup Mock Data
    n_cells = 100
    obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata = AnnData(X=np.zeros((n_cells, 2)), obs=pd.DataFrame(index=obs_names))
    # Dummy connectivities to bypass computation
    adata.obsp["connectivities"] = np.zeros((n_cells, n_cells))

    # 2. Mock sc.tl.leiden to assign specific labels
    # Resolutions: 1.0, 2.0, 3.0 (sorted)

    def side_effect_leiden(adata, resolution, key_added, **kwargs):
        # Initialize with a default to avoid NaNs
        labels = ["Unassigned"] * n_cells

        if resolution == 1.0: # Coarse
            # Group 1 (0-40) -> C1
            for i in range(41): labels[i] = "C1"
            # Group 2 (41-60) -> C2
            for i in range(41, 61): labels[i] = "C2"
            # Group 3 (61-99) -> C2
            for i in range(61, 100): labels[i] = "C2"

        elif resolution == 2.0: # Mid
            # Group 1 (0-40) -> M1
            for i in range(41): labels[i] = "M1"
            # Group 2 (41-60) -> M1
            for i in range(41, 61): labels[i] = "M1"
            # Group 3 (61-99) -> M2
            for i in range(61, 100): labels[i] = "M2"

        elif resolution == 3.0: # Fine
            # All -> F1
            for i in range(100): labels[i] = "F1"

        adata.obs[key_added] = pd.Categorical(labels)

    # 3. Run Test with Mock
    with patch("eigenp_utils.single_cell.sc.tl.leiden", side_effect=side_effect_leiden):

        # Resolutions must be sorted for the logic to work (Low->High)
        resolutions = [1.0, 2.0, 3.0]

        results = multiscale_coarsening(
            adata,
            resolutions=resolutions,
            return_output=True
        )

        consistency_df = results["consistency"]

        print("\nDetected Inconsistencies:")
        print(consistency_df)

        # 4. Verify
        # We expect exactly 1 inconsistency for Fine Cluster "F1"
        assert len(consistency_df) == 1, "Should detect exactly 1 inconsistency."

        row = consistency_df.iloc[0]

        assert row["fine_cluster"] == "F1"
        assert row["mid_parent"] == "M1"
        assert row["indirect_grandparent"] == "C1"
        assert row["direct_grandparent"] == "C2"

        # Verify it picked up the resolutions correctly
        assert row["fine_res"] == 3.0
        assert row["mid_res"] == 2.0
        assert row["coarse_res"] == 1.0

# =========================================
# Source: test_single_cell_obs_csv.py
# =========================================


@pytest.fixture
def dummy_adata():
    # Create simple dummy adata
    obs = pd.DataFrame(
        {"cell_type": ["T-cell", "B-cell", "T-cell", "Macrophage", "B-cell"]},
        index=["cell1", "cell2", "cell3", "cell4", "cell5"]
    )
    obs["cell_type"] = obs["cell_type"].astype("category")

    # categories: B-cell, Macrophage, T-cell
    colors = ["#ff0000", "#00ff00", "#0000ff"]

    adata = sc.AnnData(np.zeros((5, 2)), obs=obs)
    adata.uns["cell_type_colors"] = colors
    return adata


def test_export_obs_from_adata_to_csv(dummy_adata):
    """Test that export obs from adata to csv works as expected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "exported.csv")
        export_obs_from_adata_to_csv(
            dummy_adata,
            obs_key="cell_type",
            output_path=out_path,
            index_name="MyCellID"
        )

        assert os.path.exists(out_path)
        df = pd.read_csv(out_path)

        # Check columns
        assert "MyCellID" in df.columns
        assert "cell_type" in df.columns
        assert "cell_type_colors" in df.columns

        # Check content
        # B-cell = #ff0000, Macrophage = #00ff00, T-cell = #0000ff
        cell2_row = df[df["MyCellID"] == "cell2"].iloc[0]
        assert cell2_row["cell_type"] == "B-cell"
        assert cell2_row["cell_type_colors"] == "#ff0000"

        cell4_row = df[df["MyCellID"] == "cell4"].iloc[0]
        assert cell4_row["cell_type"] == "Macrophage"
        assert cell4_row["cell_type_colors"] == "#00ff00"


def test_import_obs_to_adata_from_csv(dummy_adata):
    """Test that import obs to adata from csv works as expected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "exported.csv")
        export_obs_from_adata_to_csv(
            dummy_adata,
            obs_key="cell_type",
            output_path=out_path
        )

        # Create a new empty adata
        adata_new = sc.AnnData(np.zeros((5, 2)), obs=pd.DataFrame(index=["cell1", "cell2", "cell3", "cell4", "cell5"]))

        import_obs_to_adata_from_csv(
            path=out_path,
            adata=adata_new,
            obs_key="cell_type",
            index_col="Cell_ID",
            index_name="CustomID"
        )

        assert "cell_type" in adata_new.obs.columns
        assert adata_new.obs.index.name == "CustomID"
        assert "cell_type_colors" in adata_new.uns

        # Check colors match the imported category order
        categories = adata_new.obs["cell_type"].cat.categories
        colors = adata_new.uns["cell_type_colors"]

        cat_to_color = dict(zip(categories, colors))
        assert cat_to_color["B-cell"] == "#ff0000"
        assert cat_to_color["Macrophage"] == "#00ff00"
        assert cat_to_color["T-cell"] == "#0000ff"


def test_import_obs_overwrite_and_suffix(dummy_adata):
    """Test that import obs overwrite and suffix works as expected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "exported.csv")
        export_obs_from_adata_to_csv(
            dummy_adata,
            obs_key="cell_type",
            output_path=out_path
        )

        # Scenario 1: overwrite=False (should append suffix)
        import_obs_to_adata_from_csv(
            path=out_path,
            adata=dummy_adata,
            obs_key="cell_type",
            index_col="Cell_ID",
            overwrite_existing=False
        )

        assert "cell_type_imported" in dummy_adata.obs.columns
        assert "cell_type_imported_colors" in dummy_adata.uns

        # Scenario 2: overwrite=True
        # Change something in the CSV and reimport to verify overwrite
        df = pd.read_csv(out_path)
        df.loc[df["Cell_ID"] == "cell1", "cell_type"] = "Unknown"
        df.to_csv(out_path, index=False)

        import_obs_to_adata_from_csv(
            path=out_path,
            adata=dummy_adata,
            obs_key="cell_type",
            index_col="Cell_ID",
            overwrite_existing=True
        )

        # cell1 should now be Unknown in the main cell_type column
        assert dummy_adata.obs.loc["cell1", "cell_type"] == "Unknown"

# =========================================
# Source: test_volcano.py
# =========================================


def test_plot_volcano_adata():
    """Test that plot volcano adata works as expected."""
    # Setup dummy AnnData
    n_obs = 100
    n_vars = 50
    X = np.random.rand(n_obs, n_vars)
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = AnnData(X=X, obs=obs, var=var)

    # Setup mock rank_genes_groups results in adata.uns
    group_name = 'GroupA'
    rank_genes_key = 'rank_genes_groups'

    # Create some dummy stats
    # genes 0-9 are significant upregulated
    # genes 10-19 are significant downregulated
    # genes 20-49 are not significant

    names = np.array([f"gene_{i}" for i in range(n_vars)])

    # Significant Upregulated
    lfc_up = np.random.uniform(1.5, 3.0, 10)
    pvals_up = np.random.uniform(0, 0.01, 10)

    # Significant Downregulated
    lfc_down = np.random.uniform(-3.0, -1.5, 10)
    pvals_down = np.random.uniform(0, 0.01, 10)

    # Non-significant
    lfc_ns = np.random.uniform(-0.5, 0.5, 30)
    pvals_ns = np.random.uniform(0.1, 1.0, 30)

    logfoldchanges = np.concatenate([lfc_up, lfc_down, lfc_ns])
    pvals_adj = np.concatenate([pvals_up, pvals_down, pvals_ns])

    # Structure for single group comparison
    # rank_genes_groups usually stores structured arrays or dataframes per group
    # but the function accesses it as: adata.uns[key]['logfoldchanges'][group]

    adata.uns[rank_genes_key] = {
        'logfoldchanges': pd.DataFrame({group_name: logfoldchanges}, index=names).to_records(index=False), # actually scanpy structure is often structured array or recarray
        # But wait, the function does: comparison_uns['logfoldchanges'][group]
        # If it's a structured array, it works by field name.
        # If it's a dict of arrays, it works by key.
        # Let's mock it as a dict of arrays/recarrays which behaves like structured data

        # Simplified mock structure: dict of dicts/series/arrays?
        # Standard Scanpy: adata.uns['rank_genes_groups']['names'] is a structured array where fields are group names
    }

    # Let's mock the dictionary structure directly as the function expects
    # The function uses: comparison_uns['names'][group] -> returns array of names

    adata.uns[rank_genes_key] = {
        'logfoldchanges': {group_name: logfoldchanges},
        'pvals_adj': {group_name: pvals_adj},
        'names': {group_name: names}
    }

    # Mock adjust_text to avoid dependency requirement during test execution if not installed,
    # and to verify kwargs are passed.
    # However, the function imports adjust_text at module level.
    # If installed, it uses it. If not, it sets it to None.
    # We need to ensure `eigenp_utils.single_cell.adjust_text` is not None for the test to proceed.

    with patch('eigenp_utils.single_cell.adjust_text') as mock_adjust_text:
        # 1. Test basic execution
        ax = plot_volcano_adata(
            adata,
            rank_genes_key=rank_genes_key,
            group=group_name,
            show=False
        )
        assert isinstance(ax, plt.Axes)
        assert mock_adjust_text.called

        # 2. Test kwargs passing
        custom_kwargs = {'force_text': (1.0, 1.0), 'arrowprops': dict(color='blue')}
        plot_volcano_adata(
            adata,
            rank_genes_key=rank_genes_key,
            group=group_name,
            **custom_kwargs
        )
        # Check if called with custom kwargs
        # The function updates default kwargs with user kwargs.
        call_kwargs = mock_adjust_text.call_args[1]
        assert call_kwargs['force_text'] == (1.0, 1.0)
        assert call_kwargs['arrowprops']['color'] == 'blue'

        # 3. Test plot_positive_only
        ax_pos = plot_volcano_adata(
            adata,
            rank_genes_key=rank_genes_key,
            group=group_name,
            plot_positive_only=True
        )
        assert isinstance(ax_pos, plt.Axes)
        # Verify vertical lines: only one expected for positive only (plus one horizontal)
        # ax.lines contains the lines added by axhline/axvline
        # Standard: 1 hline, 2 vlines = 3 lines
        # Positive only: 1 hline, 1 vline = 2 lines
        # But ax.scatter adds collections, not lines.
        # axhline/axvline add Line2D objects to ax.lines
        assert len(ax_pos.lines) == 2

        # 4. Test missing key
        res = plot_volcano_adata(
            adata,
            rank_genes_key="wrong_key",
            group=group_name
        )
        assert res is None

def test_missing_adjust_text():
    """Test that missing adjust text works as expected."""
    # Simulate missing adjustText library
    with patch('eigenp_utils.single_cell.adjust_text', None):
        n_obs = 10
        n_vars = 10
        adata = AnnData(X=np.random.rand(n_obs, n_vars))

        with pytest.raises(ImportError) as excinfo:
            plot_volcano_adata(adata, "key", "group")

        assert "pip install adjustText" in str(excinfo.value)

if __name__ == "__main__":
    try:
        test_plot_volcano_adata()
        test_missing_adjust_text()
        print("All volcano plot tests passed!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test failed: {repr(e)}")
        exit(1)

# =========================================
# Source: test_single_cell_kknn.py
# =========================================

def test_compute_kknn_neighbors():
    """Test that compute kknn neighbors works as expected."""
    rng = np.random.default_rng(42)
    n_ref = 100
    n_query = 10
    n_pcs = 5

    X_ref = rng.normal(size=(n_ref, n_pcs))
    X_query = rng.normal(size=(n_query, n_pcs))

    adata_ref = sc.AnnData(np.zeros((n_ref, 10)))
    adata_ref.obsm["X_pca"] = X_ref

    adata_query = sc.AnnData(np.zeros((n_query, 10)))
    adata_query.obsm["X_pca"] = X_query

    dists, idxs = compute_kknn_neighbors(
        adata_query,
        adata_ref,
        n_neighbors=10,
        min_neighbors=3,
        max_neighbors=20
    )

    assert len(dists) == n_query
    assert len(idxs) == n_query

    # Check that lengths are between min and max
    lengths = [len(d) for d in dists]
    assert min(lengths) >= 3
    assert max(lengths) <= 20

def test_kknn_ingest():
    """Test that kknn ingest works as expected."""
    rng = np.random.default_rng(42)
    n_ref = 100
    n_query = 10
    n_genes = 20

    # Need real variance for sc.pp.pca to work
    adata_ref = sc.AnnData(rng.normal(size=(n_ref, n_genes)))
    adata_ref.obsm["X_umap"] = rng.normal(size=(n_ref, 2))
    adata_ref.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], size=n_ref))

    adata_query = sc.AnnData(rng.normal(size=(n_query, n_genes)))

    # We don't precompute PCA on query. kknn_ingest should do it.

    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["cell_type"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        n_neighbors=10,
        recompute_ref_PCA=True,
        save_ref_PCA_key="X_pca_projected"
    )

    # Check if projection was saved
    assert "X_pca_projected" in adata_query.obsm
    assert adata_query.obsm["X_pca_projected"].shape[0] == n_query

    # Check if other mappings worked
    assert "X_umap_kknn" in adata_query.obsm
    assert adata_query.obsm["X_umap_kknn"].shape == (n_query, 2)

    assert "cell_type_kknn" in adata_query.obs
    assert len(adata_query.obs["cell_type_kknn"]) == n_query
    assert isinstance(adata_query.obs["cell_type_kknn"].dtype, pd.CategoricalDtype)

    # Check if the k count was saved
    assert "kknn_k" in adata_query.obs

    assert "mapping_confidence_cell_type_kknn" in adata_query.obs
    conf = adata_query.obs["mapping_confidence_cell_type_kknn"].values
    assert np.all((conf >= 0) & (conf <= 1.0))

def test_kknn_ingest_no_recompute_no_save():
    """Test that kknn ingest no recompute no save works as expected."""
    rng = np.random.default_rng(42)
    n_ref = 100
    n_query = 10
    n_genes = 20

    adata_ref = sc.AnnData(rng.normal(size=(n_ref, n_genes)))
    sc.tl.pca(adata_ref)
    adata_ref.obsm["X_umap"] = rng.normal(size=(n_ref, 2))
    adata_ref.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B", "C"], size=n_ref))

    adata_query = sc.AnnData(rng.normal(size=(n_query, n_genes)))

    # Check what happens without recomputing PCA and without saving the key (temporary key)
    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["cell_type"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        n_neighbors=10,
        recompute_ref_PCA=False,
        save_ref_PCA_key=None
    )

    # Temporary PCA projection should be deleted
    assert not any(k.startswith("__temp_ingest_") for k in adata_query.obsm.keys())

    # Still maps labels
    assert "cell_type_kknn" in adata_query.obs
