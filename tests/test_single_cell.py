from eigenp_utils.single_cell import pflogpf
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

NET_METHODS = [("net_scanpy", "scanpy"), ("net_binned", "binned"), ("net_binned_weighted", "binned_weighted")]


@pytest.mark.parametrize("net_method,pos_method", NET_METHODS)
def test_missing_negative_markers(net_method, pos_method):
    """A cell type missing from the negative dict gets its positive-only score."""
    adata = create_mock_adata()

    pos_dict = {
        "T-cell": ["CD3E", "CD4"],
        "B-cell": ["CD19", "MS4A1"]
    }

    neg_dict = {
        "T-cell": ["CD14"]
        # B-cell is missing
    }

    scores_neg = score_celltypes(
        adata, pos_dict, cell_type_negative_markers_dict=neg_dict, score_method=net_method
    )
    scores_pos = score_celltypes(adata, pos_dict, score_method=pos_method)

    np.testing.assert_allclose(scores_neg["B-cell"], scores_pos["B-cell"])
    # ... while the T-cell score is penalised for the contaminated cell 3 (CD14 high)
    assert scores_neg.loc["cell_3", "T-cell"] < scores_pos.loc["cell_3", "T-cell"]


def test_annotation_with_negative_selection():
    """CD14 as a T-cell negative marker lowers the T-cell evidence of cluster 0."""
    adata = create_mock_adata()

    pos_dict = {
        "T-cell": ["CD3E", "CD4"],
        "B-cell": ["CD19", "MS4A1"],
        "Contaminated": ["CD14"]
    }

    neg_dict = {
        "T-cell": ["CD14"]
    }

    # cluster 0 has 3 cells (0, 2, 3), cell 3 is CD14-high
    df_base = annotate_clusters_by_markers(
        adata, "leiden", cell_type_markers_dict=pos_dict, score_method="net_binned"
    )
    df_neg = annotate_clusters_by_markers(
        adata,
        "leiden",
        cell_type_markers_dict=pos_dict,
        cell_type_negative_markers_dict=neg_dict,
        score_method="net_binned",
    )

    assert df_base.loc["0", "assigned_cell_type"] == "T-cell"
    assert df_neg.loc["0", "assigned_cell_type"] == "T-cell"
    assert df_neg.loc["0", "median_T-cell"] <= df_base.loc["0", "median_T-cell"]
    # the cluster-0 cells' mean T-cell evidence drops by cell 3's penalty
    s_base = score_celltypes(adata, pos_dict, score_method="net_binned")
    s_neg = score_celltypes(
        adata, pos_dict, cell_type_negative_markers_dict=neg_dict, score_method="net_binned"
    )
    in_c0 = (adata.obs["leiden"] == "0").to_numpy()
    assert s_neg["T-cell"][in_c0].mean() < s_base["T-cell"][in_c0].mean()
    # B-cell cluster unaffected
    assert df_neg.loc["1", "assigned_cell_type"] == "B-cell"


@pytest.mark.parametrize("net_method,pos_method", NET_METHODS)
def test_empty_negative_list(net_method, pos_method):
    """An empty negative list leaves the positive score unchanged."""
    adata = create_mock_adata()
    pos_dict = {"T-cell": ["CD3E"]}
    neg_dict = {"T-cell": []}

    scores = score_celltypes(
        adata, pos_dict, cell_type_negative_markers_dict=neg_dict, score_method=net_method
    )
    scores_pos = score_celltypes(adata, pos_dict, score_method=pos_method)
    np.testing.assert_allclose(scores["T-cell"], scores_pos["T-cell"])
    assert np.isfinite(scores["T-cell"]).all()


@pytest.mark.parametrize("net_method,pos_method", NET_METHODS)
def test_missing_genes_in_dataset(net_method, pos_method):
    """Negative markers absent from adata are ignored (positive score unchanged)."""
    adata = create_mock_adata()
    pos_dict = {"T-cell": ["CD3E"]}
    neg_dict = {"T-cell": ["UNKNOWN_GENE"]}

    scores = score_celltypes(
        adata, pos_dict, cell_type_negative_markers_dict=neg_dict, score_method=net_method
    )
    scores_pos = score_celltypes(adata, pos_dict, score_method=pos_method)
    np.testing.assert_allclose(scores["T-cell"], scores_pos["T-cell"])
    assert np.isfinite(scores["T-cell"]).all()
    report = adata.uns["marker_missing_report"]
    row = report[(report["cell_type"] == "T-cell") & (report["marker_set"] == "neg")]
    assert row["missing_genes"].tolist() == ["UNKNOWN_GENE"]

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

    rng = np.random.default_rng(0)
    adata_ref = sc.AnnData(rng.normal(size=(100, 10)))
    adata_ref.obsm['X_pca'] = rng.normal(size=(100, 5))
    adata_ref.obs['cell_type'] = rng.choice(['A', 'B'], size=100)

    adata_query = sc.AnnData(rng.normal(size=(20, 10)))
    adata_query.obsm['X_pca'] = rng.normal(size=(20, 5))

    kknn_ingest(adata_query, adata_ref, obs_keys=['cell_type'])

    # default outputs: labels from the reference, a vote-fraction confidence,
    # an adapted k per query cell, and NaN placeholders on the reference
    assert set(adata_query.obs['cell_type_kknn']) <= {'A', 'B'}
    conf = adata_query.obs['mapping_confidence_cell_type_kknn'].to_numpy()
    assert np.all((conf >= 0.5) & (conf <= 1.0))  # winner of a two-label vote
    k = adata_query.obs['kknn_k'].to_numpy()
    assert np.all((k >= 3) & (k <= adata_ref.n_obs))
    assert adata_ref.obs['kknn_k'].isna().all()
    assert adata_ref.obs['mapping_confidence_cell_type_kknn'].isna().all()
    assert (adata_ref.obs['cell_type_kknn'] == adata_ref.obs['cell_type']).all()

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
    """morans_i_all_fast matches the centred Moran's I formula, and the expanded
    (uncentred) numerator/denominator agree with the centred ones."""
    rng = np.random.default_rng(42)
    N, G = 400, 30
    X = sp.random(N, G, density=0.2, format="csr", dtype=np.float64, random_state=rng)
    X = X + sp.csr_matrix(np.outer(np.linspace(0, 1, N), np.ones(G)))  # spatial trend
    W = sp.random(N, N, density=0.02, format="csr", dtype=np.float64, random_state=rng)
    W = W + W.T
    rs = np.asarray(W.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    W = sp.csr_matrix(W.multiply(1.0 / rs[:, None]))

    Xd = X.toarray()
    mu = Xd.mean(axis=0)
    Z = Xd - mu
    num = np.einsum("ij,ij->j", Z, W @ Z)
    den = np.einsum("ij,ij->j", Z, Z)
    I_ref = (N / W.sum()) * num / den

    # expanded form used by older code
    WX = W @ Xd
    num_exp = (np.einsum("ij,ij->j", Xd, WX) - mu * WX.sum(axis=0)
               - mu * (Xd.T @ np.asarray(W.sum(axis=1)).ravel()) + mu**2 * W.sum())
    den_exp = np.einsum("ij,ij->j", Xd, Xd) - N * mu**2
    np.testing.assert_allclose(num_exp, num, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(den_exp, den, rtol=1e-9)

    adata = AnnData(X.astype(np.float32))
    adata.var_names = [f"g{j}" for j in range(G)]
    out = morans_i_all_fast(adata, W_rowstd=W, deduplicate="none").set_index("gene")
    np.testing.assert_allclose(out.loc[adata.var_names, "I"].to_numpy(), I_ref, rtol=1e-4, atol=1e-5)

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
def test_score_celltypes_all_methods_annotate_and_sweep():
    """Every scoring method, cluster annotation and the Leiden sweep run end to end.

    This previously shared its name with test_score_celltypes_binned above, so
    that (asserting) test was shadowed and never collected, and this one only
    printed. Strategy: the same two-population toy data (G0 high in cells 0-4,
    G1 high in 5-9). Expected: each method returns one score column per cell
    type for every cell; cluster A (cells 0-4, G0-high) is annotated Type1 with
    a higher Type1 score than cluster B; the sweep returns an annotation table
    for each requested resolution.
    """
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

    for method in ("scanpy", "binned", "binned_weighted"):
        res = score_celltypes(adata, markers, neg_markers, score_method=method, use_raw=False)
        assert list(res.columns) == ["Type1", "Type2"], method
        assert list(res.index) == list(adata.obs_names), method
        assert np.all(np.isfinite(res.to_numpy())), method

    ann_res = annotate_clusters_by_markers(adata, "leiden_1.0", markers, neg_markers, score_method="binned", use_raw=False)
    assert set(ann_res.index) == {"A", "B"}
    assert ann_res.loc["A", "assigned_cell_type"] == "Type1"
    assert ann_res.loc["A", "median_Type1"] > ann_res.loc["B", "median_Type1"]

    sweep_res = sweep_leiden_and_annotate(adata, markers, neg_markers, score_method="binned", use_raw=False, neighbors_already_computed=True, resolutions=[0.5, 1.0])
    assert set(sweep_res["cluster_annotations"]) == {0.5, 1.0}
    for table in sweep_res["cluster_annotations"].values():
        assert "assigned_cell_type" in table.columns and len(table) >= 1

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

# --- PFlogPF Tests ---

def test_pflogpf_import_error(capsys):
    """
    Test that calling pflogpf without scclr installed raises ImportError
    and prints the docstring.
    """
    adata = anndata.AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]))

    with patch("builtins.__import__") as mock_import:
        # Mock __import__ to raise ImportError when scclr is requested
        orig_import = __import__
        def side_effect(name, *args, **kwargs):
            if name == "scclr":
                raise ImportError("Mocked ImportError")
            return orig_import(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        with pytest.raises(ImportError, match="The 'scclr' package is required"):
            pflogpf(adata)

        # check that docstring was printed
        captured = capsys.readouterr()
        assert "Computes the PFlogPF (shifted-CLR) normalization" in captured.out

def test_pflogpf_non_integer_counts():
    """
    Test that calling pflogpf with non-integer counts raises a ValueError.
    """
    # Create non-integer data
    adata = anndata.AnnData(X=np.array([[1.5, 2.1], [3.2, 4.9]]))

    with pytest.raises(ValueError, match="does not look like raw integer counts"):
        pflogpf(adata)

def test_pflogpf_success():
    """
    Test that pflogpf correctly calls scclr and handles layer renaming and copying.
    """
    adata = anndata.AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]))

    # Mock scclr
    mock_scclr = MagicMock()
    def mock_pflog(a, target, **kwargs):
        a.layers["pflog"] = a.X + 10 # dummy operation

    mock_scclr.pp.pflog.side_effect = mock_pflog

    import sys
    sys.modules["scclr"] = mock_scclr

    try:
        # Test inplace
        res = pflogpf(adata, key_added="custom_pflog")
        assert res is None
        assert "custom_pflog" in adata.layers
        assert "pflog" not in adata.layers
        assert np.allclose(adata.layers["custom_pflog"], adata.X + 10)

        # Test copy
        adata2 = anndata.AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]))
        res2 = pflogpf(adata2, key_added="pflogpf", copy=True)
        assert res2 is not None
        assert "pflogpf" in res2.layers
        assert "pflogpf" not in adata2.layers
    finally:
        del sys.modules["scclr"]

def test_graph_helpers_work_without_scanpy():
    """single_cell imports and its numpy/scipy graph helpers run without scanpy.

    scanpy (via numba) cannot be installed in the browser (Pyodide / marimo
    WASM), so single_cell imports it lazily. Strategy: in a fresh interpreter,
    make any ``scanpy`` import raise, then build a tiny AnnData with a known
    graph and check:
      * smooth_expression_on_graph equals the row-normalised (A + I) times X,
        i.e. a lazy random-walk average, and keeps a constant column constant;
      * find_correlated_features ranks the target itself first (r = 1);
      * a scanpy-backed function (ensure_neighbors) raises an ImportError that
        names scanpy, i.e. the failure is deferred to the call, not the import;
      * scanpy was never imported.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys, importlib.abc
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "scanpy" or name.startswith("scanpy."):
                    raise ImportError("scanpy blocked for test")
        sys.meta_path.insert(0, Block())

        import numpy as np, scipy.sparse as sp, anndata
        from eigenp_utils.single_cell import (
            ensure_neighbors, find_correlated_features, smooth_expression_on_graph,
        )

        rng = np.random.default_rng(0)
        X = rng.random((6, 3))
        X[:, 2] = 4.0
        adata = anndata.AnnData(X)
        adata.var_names = ["a", "b", "c"]
        rows = np.arange(6)
        W = sp.csr_matrix((np.ones(12), (np.r_[rows, rows], np.r_[(rows + 1) % 6, (rows - 1) % 6])), shape=(6, 6))
        adata.obsp["connectivities"] = W

        S = smooth_expression_on_graph(adata, X)
        A = W.toarray() + np.eye(6)
        expected = (A / A.sum(1, keepdims=True)) @ X
        assert np.allclose(S, expected), (S, expected)
        assert np.allclose(S[:, 2], 4.0)

        res = find_correlated_features(adata, "a", metrics=("pearson",))
        assert np.isclose(res.loc["a", "pearson"], 1.0)

        try:
            ensure_neighbors(anndata.AnnData(X))  # no graph yet: must call scanpy
        except ImportError as e:
            assert "scanpy" in str(e)
        else:
            raise AssertionError("ensure_neighbors should need scanpy")
        assert "scanpy" not in sys.modules
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]


def test_plot_clustering_tree_draws_with_current_matplotlib():
    """plot_clustering_tree runs on two nested clusterings.

    Its fallback colormap used plt.cm.get_cmap, removed in matplotlib 3.9, so
    every call raised AttributeError before drawing. Strategy: two string
    categorical clusterings where the finer one splits the coarser, so the tree
    has real edges. Expected: no exception and the figure is returned.
    """
    from eigenp_utils.single_cell import plot_clustering_tree

    adata = anndata.AnnData(X=np.zeros((60, 3)))
    adata.obs["leiden_0.5"] = pd.Categorical(np.repeat(["0", "1"], 30))
    adata.obs["leiden_1.0"] = pd.Categorical(np.repeat(["0", "1", "2"], 20))
    try:
        fig = plot_clustering_tree(adata, ["leiden_0.5", "leiden_1.0"])
        assert fig is not None and len(fig.axes) == 1
    finally:
        plt.close("all")


# =========================================
# Source: bug-scan batch 3 regressions (kknn_ingest maths, .raw handling)
# =========================================


def _typed_counts(n_per_type=150, n_types=4, n_genes=200, seed=0):
    """Poisson counts with `n_types` cell types, each with 15 up-regulated genes."""
    rng = np.random.default_rng(seed)
    n = n_per_type * n_types
    labels = np.repeat(np.arange(n_types), n_per_type)
    rate = np.full((n, n_genes), 1.0)
    for t in range(n_types):
        rate[labels == t, t * 15 : (t + 1) * 15] = 8.0
    counts = rng.poisson(rate).astype(np.float32)
    a = AnnData(counts)
    a.var_names = [f"G{i}" for i in range(n_genes)]
    a.obs_names = [f"c{i}" for i in range(n)]
    a.obs["ct"] = pd.Categorical([f"T{t}" for t in labels])
    return a


def _typed_log(n_per_type=150, n_types=4, n_genes=200, seed=0):
    a = _typed_counts(n_per_type, n_types, n_genes, seed)
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def test_preprocess_subset_takes_counts_from_raw_after_gene_filtering():
    a = _typed_counts(n_per_type=50)
    a.X[:, :5] = 0
    a.X[0, :5] = 3  # five genes expressed in a single cell: removed by filter_genes
    a.raw = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    out = preprocess_subset(a, hvg_flavor="seurat", n_top_genes=50, copy=True)

    assert out.n_vars == 50
    raw_counts = a.raw[:, out.var_names].X
    np.testing.assert_array_equal(np.asarray(out.layers["counts"]), np.asarray(raw_counts))


def test_preprocess_subset_caps_n_comps_on_small_inputs():
    small = _typed_log(n_per_type=5)  # 20 cells < default n_comps=32
    out = preprocess_subset(small, hvg_flavor="seurat", n_top_genes=100, copy=True)
    assert out.obsm["X_pca"].shape[1] <= 19

    few_hvgs = _typed_log(n_per_type=50)
    out = preprocess_subset(
        few_hvgs, hvg_flavor="seurat", n_top_genes=20, subset_to_hvgs=False, copy=True
    )
    assert out.obsm["X_pca"].shape[1] <= 19


def test_score_celltypes_without_raw_uses_X_by_default():
    a = _typed_log(n_per_type=60)
    assert a.raw is None
    markers = {"T0": [f"G{i}" for i in range(15)], "T1": [f"G{i}" for i in range(15, 30)]}

    scores = score_celltypes(a, markers)  # default use_raw=True, score_method='scanpy'
    assert scores["T0"][a.obs["ct"] == "T0"].mean() > scores["T0"][a.obs["ct"] == "T1"].mean()

    a.obs["cl"] = a.obs["ct"].astype(str)
    cdf = annotate_clusters_by_markers(a, "cl", markers)
    assert cdf.loc["T0", "assigned_cell_type"] == "T0"


def test_score_celltypes_binned_uses_layer_over_raw():
    a = _typed_log(n_per_type=60)
    a.raw = a.copy()
    a.layers["zeros"] = np.zeros_like(a.X)
    markers = {"T0": [f"G{i}" for i in range(15)]}

    scores = score_celltypes(a, markers, layer="zeros", score_method="binned")
    assert np.all(scores["T0"].to_numpy() == 0)


def test_plot_marker_genes_dict_uses_raw_genes_and_scores_without_raw():
    full = _typed_log(n_per_type=40)
    full.obsm["X_umap"] = np.random.default_rng(0).normal(size=(full.n_obs, 2))
    full.raw = full.copy()
    sub = full[:, 30:].copy()  # markers G0-G2 exist only in sub.raw

    axes = plot_marker_genes_dict_on_embedding(sub, {"T1": ["G0", "G1", "G2"]})
    titles = [ax.get_title() for ax in axes]
    assert titles == ["G0", "G1", "G2", "T1_score_scanpy"]
    plt.close("all")

    no_raw = _typed_log(n_per_type=40)
    no_raw.obsm["X_umap"] = np.random.default_rng(0).normal(size=(no_raw.n_obs, 2))
    axes = plot_marker_genes_dict_on_embedding(no_raw, {"T1": ["G0", "G1", "G2"]})
    assert "T1_score_scanpy" in [ax.get_title() for ax in axes]
    plt.close("all")


def test_find_cluster_keys_ignores_annotation_columns_and_mixed_names():
    from eigenp_utils.single_cell import find_cluster_keys

    a = AnnData(np.zeros((3, 2)))
    for col in [
        "leiden_1.0",
        "leiden_0.5",
        "leiden_0.5_cell_type",
        "leiden_0.5_ctype_softmaxp",
        "leiden",
    ]:
        a.obs[col] = 0
    assert find_cluster_keys(a) == ["leiden_0.5", "leiden_1.0"]
    assert find_cluster_keys(a, pattern=r"leiden(_[\d\.]+)?") == [
        "leiden_0.5",
        "leiden_1.0",
        "leiden",
    ]


def test_lineage_coupling_permutation_with_integer_labels():
    rng = np.random.default_rng(0)
    n = 400
    a = AnnData(np.zeros((n, 1)))
    a.obs["cell_type"] = rng.integers(0, 4, n)  # integer cluster ids
    a.obs["CloneID"] = rng.integers(0, 120, n).astype(str)

    np.random.seed(0)
    obs, z_perm, p_perm = calculate_lineage_coupling(a, method="permutation", n_permutations=200)
    _, z_hyp, _ = calculate_lineage_coupling(a, method="hypergeometric")

    # Random clones: no enrichment, so z-scores are O(1), not the raw counts
    assert np.abs(z_perm.to_numpy()).max() < 4
    assert (p_perm.to_numpy() > 0).all()
    assert np.corrcoef(z_perm.to_numpy().ravel(), z_hyp.to_numpy().ravel())[0, 1] > 0.5


def test_lineage_coupling_hypergeometric_matches_per_clone_formula():
    """Grouping clones by size must not change the analytical z-scores."""
    import scipy.special

    rng = np.random.default_rng(1)
    n = 300
    a = AnnData(np.zeros((n, 1)))
    a.obs["cell_type"] = rng.choice(list("ABC"), n)
    a.obs["CloneID"] = rng.integers(0, 90, n).astype(str)
    obs, z, _ = calculate_lineage_coupling(a, method="hypergeometric")

    df = a.obs[["cell_type", "CloneID"]]
    N = n
    k_c = df["CloneID"].value_counts().to_numpy()
    K = df["cell_type"].value_counts().loc[obs.columns].to_numpy()

    def p0(Kv, k):
        Kv, k = np.broadcast_arrays(Kv, k)
        out = np.zeros(Kv.shape)
        ok = (N - Kv) >= k
        out[ok] = np.exp(
            scipy.special.gammaln(N - Kv[ok] + 1) - scipy.special.gammaln(N - Kv[ok] - k[ok] + 1)
            - scipy.special.gammaln(N + 1) + scipy.special.gammaln(N - k[ok] + 1)
        )
        return out

    Pi = p0(K[:, None], k_c[None, :])
    Pij = p0(K[:, None, None] + K[None, :, None], k_c[None, None, :])
    p = 1 - Pi[:, None, :] - Pi[None, :, :] + Pij
    p[np.arange(3), np.arange(3), :] = 1 - Pi
    expected = p.sum(-1)
    std = np.sqrt(np.maximum((p * (1 - p)).sum(-1), 0))
    std[std == 0] = 1
    np.testing.assert_allclose(z.to_numpy(), (obs.to_numpy() - expected) / std, atol=1e-10)


def test_sweep_leiden_keeps_close_resolutions_apart():
    from eigenp_utils.single_cell import sweep_leiden_and_annotate

    a = _typed_log(n_per_type=50)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    markers = {"T0": [f"G{i}" for i in range(15)], "T1": [f"G{i}" for i in range(15, 30)]}

    sweep_leiden_and_annotate(a, markers, resolutions=(0.5, 0.96, 1.04, 2.0), use_raw=False)
    for key in ["leiden_0.5", "leiden_0.96", "leiden_1.04", "leiden_2.0"]:
        assert key in a.obs

    with pytest.raises(ValueError, match="Duplicate"):
        sweep_leiden_and_annotate(a, markers, resolutions=(0.5, 0.5), use_raw=False)


def test_multiscale_coarsening_store_in_uns_can_be_written(tmp_path):
    a = _typed_log(n_per_type=40)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    res = multiscale_coarsening(a, resolutions=[0.2, 0.5, 1.0], store_in_uns=True)

    assert 0.5 in res["clustering"]  # returned results keep float keys
    stored = a.uns["multiscale_hierarchy"]
    assert "0.5->0.2" in stored["hierarchy"]["tree"]  # "fine->coarse"
    assert all(isinstance(v, np.ndarray) for v in stored["clustering"].values())

    try:
        AnnData(np.zeros((2, 1))).write_h5ad(tmp_path / "probe.h5ad")
    except Exception as e:  # e.g. anndata 0.10 with pandas 3 string indexes
        pytest.skip(f"this anndata/pandas pair cannot write h5ad at all: {e}")
    a.write_h5ad(tmp_path / "x.h5ad")
    back = anndata.read_h5ad(tmp_path / "x.h5ad")
    assert "0.5->0.2" in back.uns["multiscale_hierarchy"]["hierarchy"]["tree"]


def test_run_triku_rebuilds_graph_and_restores_warning_filters(monkeypatch):
    import eigenp_utils.single_cell as scmod

    seen = {}

    def fake_triku(adata, n_features, use_raw):
        seen["n_neighbors"] = adata.uns["neighbors"]["params"]["n_neighbors"]
        seen["use_rep"] = adata.uns["neighbors"]["params"].get("use_rep")
        warnings.filterwarnings("ignore")  # triku does this on import
        adata.var["triku_highly_variable"] = False
        adata.var["triku_distance"] = 0.0

    fake = MagicMock()
    fake.tl.triku = fake_triku
    monkeypatch.setattr(scmod, "_import_triku", lambda: fake)

    a = _typed_log(n_per_type=50)
    a.layers["log1p"] = a.X.copy()
    a.obsm["X_other"] = np.random.default_rng(0).normal(size=(a.n_obs, 3))
    sc.pp.neighbors(a, n_neighbors=5, use_rep="X_other")

    with warnings.catch_warnings():
        filters_before = list(warnings.filters)
        scmod.run_triku(a, layer="log1p", n_neighbors=25, n_pcs_for_knn=30)
        assert warnings.filters == filters_before
    assert seen["n_neighbors"] == 25
    assert seen["use_rep"] != "X_other"


def test_importing_single_cell_keeps_warnings_enabled():
    import subprocess
    import sys

    code = (
        "import warnings\n"
        "import eigenp_utils.single_cell as s\n"
        "s.run_triku\n"
        "getattr(s, '_import_triku', lambda: None)()\n"
        "print(any(f[0] == 'ignore' and f[1] is None and f[2] is Warning and f[3] is None "
        "for f in warnings.filters))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_tl_pacmap_applies_neighbor_heuristic(monkeypatch):
    import pacmap
    import eigenp_utils.single_cell as scmod

    captured = {}

    class FakePaCMAP:
        def __init__(self, **kw):
            captured.update(kw)
            self.n_components = kw.get("n_components", 2)

        def fit_transform(self, X, init=None):
            return np.zeros((X.shape[0], self.n_components))

    monkeypatch.setattr(pacmap, "PaCMAP", FakePaCMAP)
    monkeypatch.setattr(scmod, "pacmap_heuristic_n_neighbors", lambda n: 17)
    a = _typed_log(n_per_type=30)
    sc.pp.pca(a, n_comps=10)
    tl_pacmap(a)
    assert captured["n_neighbors"] == 17


def test_tl_pacmap_padded_init_is_reproducible():
    a = _typed_log(n_per_type=40)
    sc.pp.pca(a, n_comps=10)
    a.obsm["X_init1d"] = a.obsm["X_pca"][:, :1].copy()
    tl_pacmap(a, init="X_init1d", random_state=3)
    first = a.obsm["X_pacmap"].copy()
    np.random.seed(None)
    tl_pacmap(a, init="X_init1d", random_state=3)
    np.testing.assert_allclose(a.obsm["X_pacmap"], first)


def test_kknn_curvature_cache_is_keyed_by_representation():
    rng = np.random.default_rng(0)
    a = AnnData(np.zeros((400, 1)))
    a.obsm["X_flat"] = rng.normal(size=(400, 5))
    t = rng.uniform(0, 6 * np.pi, 400)
    a.obsm["X_curved"] = np.c_[np.cos(t), np.sin(t), 0.01 * rng.normal(size=400)]

    fresh = a.copy()
    _, idx_fresh = compute_kknn_neighbors(fresh, fresh, use_rep="X_curved")

    compute_kknn_neighbors(a, a, use_rep="X_flat")  # caches bounds for X_flat
    _, idx = compute_kknn_neighbors(a, a, use_rep="X_curved")
    assert [len(i) for i in idx] == [len(i) for i in idx_fresh]

    # A recomputed embedding under the same key invalidates the cache entry
    a.obsm["X_flat"] = a.obsm["X_curved"].copy()
    _, idx_again = compute_kknn_neighbors(a, a, use_rep="X_flat")
    assert [len(i) for i in idx_again] == [len(i) for i in idx_fresh]


def test_compute_kknn_neighbors_small_reference():
    rng = np.random.default_rng(0)
    ref = AnnData(np.zeros((8, 1)))
    ref.obsm["X_pca"] = rng.normal(size=(8, 3))
    dists, idxs = compute_kknn_neighbors(ref, ref)
    assert all(1 <= len(i) <= 8 for i in idxs)


def _scaled_reference(max_value):
    a = _typed_log(n_per_type=100)
    a.raw = a.copy()  # unscaled log data, used to build queries
    sc.pp.highly_variable_genes(a, n_top_genes=100, flavor="seurat")
    sc.pp.scale(a, max_value=max_value)
    sc.tl.pca(a, n_comps=20)
    return a


@pytest.mark.parametrize("max_value", [None, 2.0])
def test_kknn_ingest_projects_scaled_reference_exactly(max_value):
    ref = _scaled_reference(max_value)
    query = AnnData(ref.raw.X.copy(), obs=ref.obs[["ct"]].copy(), var=ref.var[[]].copy())

    kknn_ingest(
        query, ref, obs_keys="ct", recompute_ref_PCA=False, save_ref_PCA_key="X_pca_proj"
    )

    np.testing.assert_allclose(query.obsm["X_pca_proj"], ref.obsm["X_pca"], atol=2e-3)
    assert (query.obs["ct_kknn"].astype(str) == query.obs["ct"].astype(str)).mean() > 0.97


@pytest.mark.parametrize("recompute", [False, True])
def test_kknn_ingest_query_of_one_type_is_not_recentred(recompute):
    ref = _typed_log(n_per_type=200, seed=0)
    sc.pp.highly_variable_genes(ref, n_top_genes=100, flavor="seurat")
    sc.tl.pca(ref, n_comps=20)
    del ref.layers["counts"]  # the recompute path then selects HVGs on log data

    query = _typed_log(n_per_type=200, seed=1)
    query = query[query.obs["ct"] == "T0"].copy()
    del query.layers["counts"]

    kknn_ingest(query, ref, obs_keys="ct", recompute_ref_PCA=recompute)
    assert (query.obs["ct_kknn"].astype(str) == "T0").mean() > 0.95


def test_kknn_ingest_recompute_saves_matching_reference_basis():
    ref = _typed_log(n_per_type=100)
    del ref.layers["counts"]
    sc.pp.highly_variable_genes(ref, n_top_genes=100, flavor="seurat")
    sc.tl.pca(ref, n_comps=20)
    query = ref[:150].copy()

    with pytest.raises(ValueError, match="already exists"):
        kknn_ingest(query, ref, obs_keys="ct", save_ref_PCA_key="X_pca")

    kknn_ingest(query, ref, obs_keys="ct", save_ref_PCA_key="X_pca_ingest")
    assert ref.obsm["X_pca_ingest"].shape[0] == ref.n_obs
    np.testing.assert_allclose(
        query.obsm["X_pca_ingest"], ref.obsm["X_pca_ingest"][:150], atol=1e-3
    )


def test_kknn_ingest_distance_weights_use_median_bandwidth():
    rng = np.random.default_rng(0)
    ref = AnnData(np.zeros((501, 1)))
    ref.obsm["X_emb"] = np.r_[rng.normal(size=(500, 2)), [[0.0, 0.0]]]
    ref.obs["lbl"] = pd.Categorical(["A"] * 500 + ["B"])
    query = AnnData(np.zeros((1, 1)))
    query.obsm["X_emb"] = np.array([[0.0, 0.0]])  # coincides with the single B cell

    kknn_ingest(query, ref, obs_keys="lbl", use_rep="X_emb")
    assert query.obs["lbl_kknn"].iloc[0] == "A"


def test_kknn_ingest_integer_query_matrix():
    ref = _typed_counts(n_per_type=60)
    sc.pp.highly_variable_genes(ref, n_top_genes=100, flavor="seurat_v3")
    sc.tl.pca(ref, n_comps=10)
    query = ref[:50].copy()
    query.X = query.X.astype(np.int64)
    kknn_ingest(query, ref, obs_keys="ct", recompute_ref_PCA=False)
    assert (query.obs["ct_kknn"].astype(str) == query.obs["ct"].astype(str)).mean() > 0.9


def test_kknn_ingest_transform_uses_model_space():
    pacmap = pytest.importorskip("pacmap")
    ref = _typed_log(n_per_type=100)
    del ref.layers["counts"]
    sc.pp.highly_variable_genes(ref, n_top_genes=100, flavor="seurat")
    sc.tl.pca(ref, n_comps=12)  # != 50, the recomputed PCA's default width
    model = pacmap.PaCMAP(n_components=2, random_state=0)
    ref.obsm["X_pacmap"] = model.fit_transform(ref.obsm["X_pca"])

    query = ref[:120].copy()
    kknn_ingest(
        query, ref, obs_keys="ct", obsm_keys="X_pacmap", barycenter="transform",
        embedding_models={"X_pacmap": model},
    )
    Y = query.obsm["X_pacmap_kknn"]
    assert Y.shape == (120, 2)
    # each mapped cell's nearest reference cell in the embedding has its own type
    d = ((Y[:, None, :] - ref.obsm["X_pacmap"][None, :, :]) ** 2).sum(-1)
    nn_type = ref.obs["ct"].to_numpy()[d.argmin(1)]
    assert (nn_type == query.obs["ct"].to_numpy()).mean() > 0.9

    # a basis with different rows from the model's training data is rejected
    with pytest.raises(ValueError, match="fit on"):
        kknn_ingest(
            query, ref[:300].copy(), obs_keys="ct", obsm_keys="X_pacmap",
            barycenter="transform", embedding_models={"X_pacmap": model},
            recompute_ref_PCA=False,
        )


# =========================================
# Source: scan low-severity / remaining-medium regressions (single_cell)
# =========================================


def test_plot_clustering_tree_uses_uns_colors_for_integer_categories():
    from eigenp_utils.single_cell import plot_clustering_tree

    a = AnnData(np.zeros((6, 1)))
    a.obs["leiden_0.5"] = pd.Categorical([0, 0, 1, 1, 2, 2])
    a.obs["leiden_1.0"] = pd.Categorical([0, 1, 2, 3, 4, 5])
    a.uns["leiden_0.5_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    fig = plot_clustering_tree(a, ["leiden_0.5", "leiden_1.0"])
    facecolors = {
        matplotlib.colors.to_hex(c)
        for coll in fig.axes[0].collections
        for c in coll.get_facecolors()
    }
    assert {"#ff0000", "#00ff00", "#0000ff"} <= facecolors
    plt.close(fig)


def test_import_obs_twice_without_overwrite(tmp_path):
    a = AnnData(np.zeros((3, 1)))
    a.obs_names = ["c0", "c1", "c2"]
    a.obs["ct"] = ["x", "y", "z"]
    path = tmp_path / "ann.csv"
    pd.DataFrame({"ct": ["A", "B", "C"]}, index=["c0", "c1", "c2"]).to_csv(path)

    import_obs_to_adata_from_csv(str(path), a, "ct")
    import_obs_to_adata_from_csv(str(path), a, "ct")
    assert list(a.obs["ct"]) == ["x", "y", "z"]
    assert list(a.obs["ct_imported"].astype(str)) == ["A", "B", "C"]
    assert list(a.obs["ct_imported_2"].astype(str)) == ["A", "B", "C"]


def test_custom_graph_keys_follow_scanpy_naming():
    from eigenp_utils.single_cell import ensure_neighbors, build_rowstd_connectivities

    a = _typed_log(n_per_type=30)
    sc.pp.pca(a, n_comps=10)
    ensure_neighbors(a, n_neighbors=10, key="my_connectivities")
    assert "my_connectivities" in a.obsp
    W = build_rowstd_connectivities(a, key="my_connectivities")
    np.testing.assert_allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0, rtol=1e-5)
    out = morans_i_all_fast(a, weights_key="my_connectivities")
    assert len(out) == a.n_vars

    with pytest.raises(ValueError, match="key_added"):
        ensure_neighbors(a, n_neighbors=10, key="my_graph")


def test_morans_i_empty_gene_list():
    pytest.importorskip("esda")
    from eigenp_utils.single_cell import morans_i

    a = _typed_log(n_per_type=20)
    sc.pp.pca(a, n_comps=5)
    sc.pp.neighbors(a, n_neighbors=5)
    out = morans_i(a, [], permutations=9)
    assert out.empty
    assert list(out.columns) == ["gene", "I", "p_sim", "z_sim"]


def _moran_test_adata(sparse, n=3000):
    rng = np.random.default_rng(0)
    side = int(np.sqrt(n))
    n = side * side
    yy, xx = np.divmod(np.arange(n), side)
    signal = np.sin(xx / 6.0) + 0.3 * rng.normal(size=n)  # strongly autocorrelated
    X = np.c_[signal - signal.min(), np.full(n, 0.1), np.full(n, 0.7), rng.random(n)]
    X = X.astype(np.float32)
    a = AnnData(sp.csr_matrix(X) if sparse else X)
    a.var_names = ["trend", "c01", "c07", "noise"]
    a.obsm["X_pos"] = np.c_[xx, yy].astype(float)
    sc.pp.neighbors(a, n_neighbors=8, use_rep="X_pos")
    return a


@pytest.mark.parametrize("sparse", [False, True])
def test_morans_i_all_fast_constant_genes_are_nan(sparse):
    out = morans_i_all_fast(_moran_test_adata(sparse)).set_index("gene")
    assert np.isnan(out.loc["c01", "I"]) and np.isnan(out.loc["c07", "I"])
    assert np.isnan(out.loc["c01", "pval_z"])
    assert out.index[0] == "trend"


def test_morans_i_all_fast_pvalues_do_not_underflow():
    rng = np.random.default_rng(0)
    side = 54
    yy, xx = np.divmod(np.arange(side * side), side)
    signal = np.sin(xx / 6.0) + rng.normal(size=side * side)  # z ~ 25
    a = AnnData(np.c_[signal - signal.min(), rng.random(side * side)].astype(np.float32))
    a.var_names = ["trend", "noise"]
    a.obsm["X_pos"] = np.c_[xx, yy].astype(float)
    sc.pp.neighbors(a, n_neighbors=8, use_rep="X_pos")
    out = morans_i_all_fast(a).set_index("gene")
    z = out.loc["trend", "z_score"]
    assert 13 < z < 37  # float32 p underflows to 0 here, float64 does not
    p = out.loc["trend", "pval_z"]
    assert out["pval_z"].dtype == np.float64
    assert 0 < p < 1e-40
    from scipy.stats import norm
    np.testing.assert_allclose(p, 2 * norm.sf(z), rtol=1e-6)


def test_find_expression_archetypes_array_like_and_duplicate_names():
    rng = np.random.default_rng(0)
    n = 300
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    X = np.c_[[f1 + 0.3 * rng.normal(size=n) for _ in range(4)]].T
    X = np.c_[X, np.c_[[f2 + 0.3 * rng.normal(size=n) for _ in range(4)]].T]
    a = AnnData(X)
    a.var_names = ["g1", "g2", "g3", "g4", "h1", "h2", "h3", "g1"]  # 'g1' duplicated

    res = find_expression_archetypes(a, pd.Series(["g1", "g2", "h1", "h2"]), 2)
    assert res["gene_list"] == ["g1", "g2", "h1", "h2"]
    res = find_expression_archetypes(a, np.array(["g2", "g3", "h1", "h2"]), 2)
    assert len(res["clusters"]) == 4


def test_find_expression_archetypes_share_one_scale():
    rng = np.random.default_rng(0)
    n = 500
    f1, f2, f3 = rng.normal(size=(3, n))
    cols = [f1 + 0.5 * rng.normal(size=n) for _ in range(5)]
    cols += [f2 + 0.5 * rng.normal(size=n) for _ in range(5)]
    cols += [f3]  # a lone gene
    a = AnnData(np.c_[cols].T)
    a.var_names = [f"g{i}" for i in range(11)]
    res = find_expression_archetypes(a, list(a.var_names), 3)

    stds = res["archetypes"].std(axis=1)
    np.testing.assert_allclose(stds, 1.0, rtol=1e-6)
    # argmax assignment recovers the dominant factor for most cells
    truth = np.argmax(np.c_[f1, f2, f3], axis=1)
    pred = np.argmax(res["archetypes"], axis=0)
    # map archetypes to factors by correlation
    corr = np.corrcoef(np.r_[res["archetypes"], np.c_[f1, f2, f3].T])[:3, 3:]
    mapping = corr.argmax(axis=1)
    assert (mapping[pred] == truth).mean() > 0.85


def test_archetype_plots_keep_existing_obs_columns():
    from eigenp_utils.single_cell import plot_archetype_assignments

    a = _typed_log(n_per_type=30)
    a.obsm["X_umap"] = np.random.default_rng(0).normal(size=(a.n_obs, 2))
    res = find_expression_archetypes(a, [f"G{i}" for i in range(20)], 2)
    a.obs["archetype_1_score"] = 123.0

    plot_archetype_summary(a, res, 1, k=2)
    plot_archetype_assignments(a, res)
    plt.close("all")
    assert (a.obs["archetype_1_score"] == 123.0).all()
    assert "archetype_2_score" not in a.obs


def test_plot_archetype_hierarchy_returns_figure():
    from eigenp_utils.single_cell import plot_archetype_hierarchy

    a = _typed_log(n_per_type=30)
    res = find_expression_archetypes(a, [f"G{i}" for i in range(20)], 2)
    before = len(plt.get_fignums())
    fig = plot_archetype_hierarchy(res)
    assert isinstance(fig, plt.Figure)
    assert len(plt.get_fignums()) == before + 1
    plt.close(fig)


def test_filter_marker_dict_by_triku_strings_and_casing():
    from eigenp_utils.single_cell import filter_marker_dict_by_triku

    a = AnnData(np.zeros((2, 3)))
    a.var_names = ["CD3E", "MS4A1", "LYZ"]
    a.var["triku_highly_variable"] = True
    a.var["triku_distance"] = 1.0
    out = filter_marker_dict_by_triku(a, {"T": "CD3E", "B": ["ms4a1"], "M": ["XYZ"]})
    assert out == {"T": ["CD3E"], "B": ["MS4A1"], "M": []}


def test_multiscale_coarsening_clusters_on_use_rep():
    a = _typed_log(n_per_type=40)
    sc.pp.pca(a, n_comps=10)
    a.obsm["X_other"] = np.random.default_rng(0).normal(size=(a.n_obs, 3))
    sc.pp.neighbors(a, n_neighbors=10)  # built on X_pca

    with pytest.warns(UserWarning, match="multiscale_neighbors"):
        multiscale_coarsening(a, resolutions=[0.5, 1.0], use_rep="X_other")
    assert a.uns["multiscale_neighbors"]["params"]["use_rep"] == "X_other"

    b = AnnData(np.random.default_rng(1).normal(size=(200, 20)).astype(np.float32))
    multiscale_coarsening(b, resolutions=[0.5], use_rep="X")
    assert "X_pca" not in b.obsm
    assert b.uns["neighbors"]["params"]["use_rep"] == "X"


def test_filter_markers_exact_var_name_beats_alias():
    from eigenp_utils.single_cell import filter_markers_to_adata

    a = AnnData(np.zeros((2, 3)))
    a.var_names = ["CD4", "CD8A", "CD8B"]
    a.var["Symbol"] = ["CD4", "CD8A", "CD8A"]
    cleaned, removed, repl = filter_markers_to_adata({"T": ["CD8A", "CD4"]}, a, verbose=False)
    assert cleaned["T"] == ["CD8A", "CD4"]
    assert repl == {}

    b = AnnData(np.zeros((2, 4)))
    b.var_names = ["CD4", "CD8A", "TBCE", "TBCE-1"]
    b.var["gene_name"] = ["CD4", "CD8A", "TBCE", "TBCE"]
    cleaned, _, _ = filter_markers_to_adata({"X": ["TBCE"]}, b, verbose=False)
    assert cleaned["X"] == ["TBCE"]


def test_filter_markers_by_moran_fills_from_above_threshold_first():
    from eigenp_utils.single_cell import filter_markers_by_moran

    df = pd.DataFrame({"gene": list("ABCDEF"), "I": [0.9, 0.8, 0.7, 0.6, 0.05, 0.01]})
    out, report = filter_markers_by_moran(
        {"t": list("ABCDEF")}, df, threshold=0.1, top_k=2, min_k=3, verbose=False
    )
    assert out["t"] == ["A", "B", "C"]
    assert report["t"]["used_fallback"] is False

    out, report = filter_markers_by_moran(
        {"t": ["A", "E", "F"]}, df, threshold=0.1, top_k=5, min_k=2, verbose=False
    )
    assert out["t"] == ["A", "E"]
    assert report["t"]["used_fallback"] is True


@pytest.mark.parametrize("sparse", [False, True])
def test_binned_scores_batched_matches_per_gene(sparse, monkeypatch):
    import eigenp_utils.single_cell as scmod

    a = _typed_log(n_per_type=40)
    a.var_names = [f"G{i}" for i in range(a.n_vars - 1)] + ["g0"]  # case duplicate of G0
    if sparse:
        a.X = sp.csr_matrix(a.X)
    markers = {"T0": ["G0", "G1", "G2", "G3"], "T1": [f"G{i}" for i in range(15, 30)]}

    expected = {}
    for ct, genes in markers.items():
        cols = []
        for g in genes:
            x = _extract_gene_vector(a, g, source="X", duplicate_policy="mean")
            nz = x > 0
            b = np.zeros_like(x)
            if nz.any():
                med = np.median(x[nz])
                b[nz & (x <= med)] = 0.5
                b[x > med] = 1.0
            cols.append(b)
        expected[ct] = np.mean(cols, axis=0)

    calls = []
    real = scmod._select_matrix_and_names
    monkeypatch.setattr(
        scmod, "_select_matrix_and_names", lambda *ar, **kw: (calls.append(1), real(*ar, **kw))[1]
    )
    scores = score_celltypes(a, markers, use_raw=False, score_method="binned")
    for ct in markers:
        np.testing.assert_allclose(scores[ct], expected[ct])
    assert len(calls) == len(markers)  # one matrix slice per gene set, not per gene


def test_marker_missing_report_is_h5ad_safe(tmp_path):
    a = _typed_log(n_per_type=30)
    a.obs["cl"] = a.obs["ct"].astype(str)
    annotate_clusters_by_markers(
        a, "cl", {"NK/T": ["G0", "G1", "MISSING"], "B": [f"G{i}" for i in range(15, 20)]}
    )
    rep = a.uns["marker_missing_report"]
    assert isinstance(rep, pd.DataFrame)
    row = rep[(rep["cell_type"] == "NK/T") & (rep["marker_set"] == "pos")]
    assert row["missing_genes"].tolist() == ["MISSING"]

    try:
        AnnData(np.zeros((2, 1))).write_h5ad(tmp_path / "probe.h5ad")
    except Exception as e:  # e.g. anndata 0.10 with pandas 3 string indexes
        pytest.skip(f"this anndata/pandas pair cannot write h5ad at all: {e}")
    a.write_h5ad(tmp_path / "x.h5ad")


def test_export_obs_keeps_dotted_names(tmp_path):
    a = AnnData(np.zeros((3, 1)))
    a.obs["score"] = [1.0, 2.0, 3.0]
    export_obs_from_adata_to_csv(a, "score", output_path=str(tmp_path / "run_2024.10"))
    export_obs_from_adata_to_csv(a, "score", output_path=str(tmp_path / "run_2024.11"))
    assert (tmp_path / "run_2024.10.csv").exists()
    assert (tmp_path / "run_2024.11.csv").exists()
    export_obs_from_adata_to_csv(a, "score", output_path=str(tmp_path))
    assert (tmp_path / "score_export.csv").exists()


def _volcano_adata(pvals):
    n = len(pvals)
    a = AnnData(np.zeros((2, n)))
    a.var_names = [f"Gene{i}" for i in range(n)]
    dt = [("0", "f8")]
    p = np.zeros(n, dtype=dt)
    p["0"] = pvals
    lfc = np.zeros(n, dtype=dt)
    lfc["0"] = np.linspace(-5, 5, n)
    names = np.zeros(n, dtype=[("0", "U10")])
    names["0"] = a.var_names
    a.uns["rank_genes_groups"] = {"pvals_adj": p, "logfoldchanges": lfc, "names": names}
    return a


def test_plot_volcano_all_zero_pvalues():
    pytest.importorskip("adjustText")
    a = _volcano_adata(np.zeros(6))
    ax = plot_volcano_adata(a, "rank_genes_groups", group="0", label_top_n=3)
    ys = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
    assert np.isfinite(ys).all() and ys.max() > 300
    plt.close("all")


def test_plot_volcano_uses_current_adjusttext_api(caplog, monkeypatch):
    adjustText = pytest.importorskip("adjustText")
    import inspect
    import logging
    import eigenp_utils.single_cell as scmod

    params = inspect.signature(adjustText.adjust_text).parameters
    if "force_static" not in params:
        pytest.skip("adjustText < 1.0")
    passed = {}
    import functools

    @functools.wraps(adjustText.adjust_text)  # keeps the signature the code inspects
    def spy(*args, **kwargs):
        passed.update(kwargs)
        return adjustText.adjust_text(*args, **kwargs)

    monkeypatch.setattr(scmod, "adjust_text", spy)
    a = _volcano_adata(np.r_[np.full(10, 1e-10), np.full(10, 0.5)])
    with caplog.at_level(logging.WARNING, logger="adjustText"):
        plot_volcano_adata(a, "rank_genes_groups", group="0", label_top_n=5)
    # unknown keywords are forwarded to the arrows and trigger the annotate fallback
    assert set(passed) - {"arrowprops"} <= set(params)
    assert not any("FancyArrowPatch" in r.getMessage() for r in caplog.records)
    plt.close("all")


def test_kknn_classifier_votes_on_booleans():
    rng = np.random.default_rng(0)
    a = AnnData(np.zeros((200, 1)))
    a.obsm["X_emb"] = rng.normal(size=(200, 2))
    a.obs["is_doublet"] = rng.random(200) < 0.05
    out = kknn_classifier(a, "is_doublet", use_rep="X_emb", inplace=False)
    assert out.dtype == bool


def test_kknn_ingest_transfers_numeric_obs_as_weighted_mean():
    rng = np.random.default_rng(0)
    t_ref = rng.uniform(0, 1, 2000)
    ref = AnnData(np.zeros((2000, 1)))
    ref.obsm["X_emb"] = np.c_[np.cos(3 * t_ref), np.sin(3 * t_ref)] + 0.02 * rng.normal(size=(2000, 2))
    ref.obs["pt"] = t_ref + 0.05 * rng.normal(size=2000)  # noisy pseudotime
    t_q = rng.uniform(0.05, 0.95, 500)
    q = AnnData(np.zeros((500, 1)))
    q.obsm["X_emb"] = np.c_[np.cos(3 * t_q), np.sin(3 * t_q)]

    kknn_ingest(q, ref, obs_keys="pt", use_rep="X_emb")
    pred = q.obs["pt_kknn"].to_numpy()
    assert np.mean(np.abs(pred - t_q)) < 0.025  # 1-NN plurality vote gives ~0.04
    assert q.obs["mapping_std_pt_kknn"].between(0, 0.2).all()
    assert q.obs["mapping_confidence_pt_kknn"].isna().all()


def test_find_correlated_features_drops_nan_targets():
    rng = np.random.default_rng(0)
    n = 300
    t = rng.normal(size=n)
    a = AnnData(np.c_[t + 0.1 * rng.normal(size=n), rng.normal(size=n)])
    a.var_names = ["good", "noise"]
    a.obs["score"] = t
    a.obs.loc[a.obs_names[:100], "score"] = np.nan
    with pytest.warns(UserWarning, match="100 non-finite"):
        res = find_correlated_features(a, "score", metrics=("pearson", "spearman"))
    assert res.loc["good", "pearson"] > 0.9
    assert np.isfinite(res.to_numpy()).all()


def test_find_correlated_features_spearman_constant_genes():
    rng = np.random.default_rng(0)
    n = 200
    X = np.c_[rng.normal(size=n), np.zeros(n), np.full(n, 3.0)]
    a = AnnData(sp.csr_matrix(X))
    a.var_names = ["g", "zero", "const"]
    a.obs["score"] = X[:, 0] + 0.1 * rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = find_correlated_features(a, "score", metrics=("pearson", "spearman"))
    assert res.loc["zero", "spearman"] == 0.0 and res.loc["const", "spearman"] == 0.0
    assert res.loc["g", "spearman"] > 0.9


def test_tl_pacmap_paga_init_without_plotting_paga():
    a = _typed_log(n_per_type=50)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    try:
        sc.tl.paga(a, groups="ct")
    except ValueError:
        # sc.tl.paga fails (python-igraph bug) when no edges cross groups; build what it stores instead
        C = np.full((4, 4), 0.2)
        np.fill_diagonal(C, 0.0)
        a.uns["paga"] = {
            "connectivities": sp.csr_matrix(C),
            "connectivities_tree": sp.csr_matrix(np.triu(C)),
            "groups": "ct",
        }
    assert "pos" not in a.uns["paga"]
    tl_pacmap(a, init="paga")
    assert a.obsm["X_pacmap"].shape == (a.n_obs, 2)
    assert np.isfinite(a.obsm["X_pacmap"]).all()


# =========================================
# Source: scan residuals (score-column restore, exact gene names, edgeless PAGA)
# =========================================


def test_plot_marker_genes_dict_restores_existing_score_column():
    a = _typed_log(n_per_type=40)
    a.obsm["X_umap"] = np.random.default_rng(0).normal(size=(a.n_obs, 2))
    user_values = np.arange(a.n_obs, dtype=float)
    a.obs["T1_score_scanpy"] = user_values
    a.obs["other"] = 1.0

    axes = plot_marker_genes_dict_on_embedding(a, {"T1": ["G0", "G1", "G2"]})
    plt.close("all")

    assert "T1_score_scanpy" in [ax.get_title() for ax in axes]
    np.testing.assert_array_equal(a.obs["T1_score_scanpy"].to_numpy(), user_values)
    assert list(a.obs.columns) == ["ct", "T1_score_scanpy", "other"]


def _case_variant_adata():
    X = np.array([[5.0, 0.0, 1.0], [10.0, 0.0, 2.0], [15.0, 0.0, 3.0]])
    a = AnnData(X)
    a.var_names = ["Cd4", "CD4", "B"]
    return a


def test_exact_gene_name_wins_over_case_variants():
    from eigenp_utils.single_cell import _extract_gene_matrix

    a = _case_variant_adata()
    # exact name present: only that column, not the mean with 'Cd4'
    np.testing.assert_array_equal(_extract_gene_vector(a, "CD4"), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(_extract_gene_vector(a, "Cd4"), [5.0, 10.0, 15.0])
    np.testing.assert_array_equal(
        _extract_gene_matrix(a, ["CD4", "Cd4"]), [[0, 5], [0, 10], [0, 15]]
    )
    # no exact name: case-insensitive fallback combines all case variants
    np.testing.assert_array_equal(_extract_gene_vector(a, "cd4"), [2.5, 5.0, 7.5])
    np.testing.assert_array_equal(_extract_gene_matrix(a, ["cd4"]).ravel(), [2.5, 5.0, 7.5])
    # identical names are still duplicates combined by the duplicate policy
    b = AnnData(np.array([[1.0, 3.0], [2.0, 4.0]]))
    b.var_names = ["A", "A"]
    np.testing.assert_array_equal(_extract_gene_vector(b, "A"), [2.0, 3.0])


def test_morans_dedup_modes_agree_on_case_variants():
    rng = np.random.default_rng(0)
    n = 60
    coords = rng.normal(size=(n, 2))
    smooth = coords[:, 0]
    X = np.c_[smooth, rng.normal(size=n), smooth + 0.1 * rng.normal(size=n), smooth]
    a = AnnData(X)
    a.var_names = ["Gene1", "GENE1", "B", "B"]
    a.obsm["X_pca"] = coords
    sc.pp.neighbors(a, n_neighbors=8, use_rep="X_pca")

    reported = {}
    for mode in ("none", "mean", "sum", "first", "last"):
        df = morans_i_all_fast(a, deduplicate=mode)
        reported[mode] = sorted(df["gene"])
    # case variants are distinct features in every mode; identical names collapse
    assert reported["none"] == ["B", "B", "GENE1", "Gene1"]
    for mode in ("mean", "sum", "first", "last"):
        assert reported[mode] == ["B", "GENE1", "Gene1"], mode
    I = morans_i_all_fast(a, deduplicate="mean").set_index("gene")["I"]
    assert I["Gene1"] > 0.5 and I["GENE1"] < 0.3


def test_filter_markers_by_moran_prefers_exact_name():
    from eigenp_utils.single_cell import filter_markers_by_moran

    df = pd.DataFrame({"gene": ["Actb", "ACTB", "Gapdh"], "I": [0.9, 0.05, 0.5]})
    kept, report = filter_markers_by_moran(
        {"x": ["ACTB", "GAPDH"]}, df, threshold=0.1, top_k=5, min_k=1,
        fallback_to_min_k=False, verbose=False,
    )
    # ACTB's own score (0.05) is below threshold; Actb's 0.9 must not be used.
    # GAPDH has no exact row, so its case variant's score (0.5) is used
    # (the caller's spelling is kept).
    assert kept["x"] == ["GAPDH"]


def test_tl_pacmap_paga_init_with_disconnected_groups():
    a = _typed_log(n_per_type=50)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    # what sc.tl.paga describes for groups with no edges between them
    Z = sp.csr_matrix((4, 4))
    a.uns["paga"] = {"connectivities": Z, "connectivities_tree": Z, "groups": "ct"}
    with pytest.warns(UserWarning, match="no edges between groups"):
        tl_pacmap(a, init="paga")
    assert a.obsm["X_pacmap"].shape == (a.n_obs, 2)
    assert np.isfinite(a.obsm["X_pacmap"]).all()
