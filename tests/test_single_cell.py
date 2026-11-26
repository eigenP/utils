"""Unit, property, and regression tests for ``eigenp_utils.single_cell``."""

import functools
import inspect
import logging
import subprocess
import sys
import textwrap
import warnings
import zlib
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import anndata
from anndata import AnnData
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from scipy import stats
from scipy.stats import norm

import eigenp_utils.single_cell as scmod
from eigenp_utils.single_cell import (
    _extract_gene_matrix,
    _extract_gene_vector,
    _kknn_distance_weights,
    _kknn_smooth_values,
    _paga_init_positions,
    annotate_clusters_by_markers,
    barplot_cluster_uncertainty,
    build_pysal_weights,
    build_rowstd_connectivities,
    calculate_lineage_coupling,
    carry_palette,
    check_gene_adata,
    compute_kknn_neighbors,
    ensure_neighbors,
    export_obs_from_adata_to_csv,
    filter_marker_dict_by_triku,
    filter_markers_by_moran,
    filter_markers_to_adata,
    find_cluster_keys,
    find_correlated_features,
    find_expression_archetypes,
    import_obs_to_adata_from_csv,
    kknn_classifier,
    kknn_ingest,
    load_pacmap_model,
    morans_i,
    morans_i_all_fast,
    multiscale_coarsening,
    pacmap_heuristic_n_neighbors,
    pflogpf,
    plot_archetype_assignments,
    plot_archetype_hierarchy,
    plot_archetype_summary,
    plot_celltype_ari,
    plot_clustering_tree,
    plot_coupling_heatmap,
    plot_marker_genes_dict_on_embedding,
    plot_volcano_adata,
    preprocess_subset,
    save_pacmap_model,
    score_celltypes,
    smooth_expression_on_graph,
    sweep_leiden_and_annotate,
    tl_pacmap,
    triku_gene_set,
)

try:
    import pacmap

    PACMAP_INSTALLED = True
except ImportError:
    PACMAP_INSTALLED = False

requires_pacmap = pytest.mark.skipif(not PACMAP_INSTALLED, reason="PaCMAP not installed")

NET_METHODS = [
    ("net_scanpy", "scanpy"),
    ("net_binned", "binned"),
    ("net_binned_weighted", "binned_weighted"),
]


# =============================================================================
# Shared Helpers & Fixtures
# =============================================================================


def _build_grid_graph(n_rows: int, n_cols: int) -> sp.csr_matrix:
    """Build a 4-connected grid graph adjacency matrix."""
    n = n_rows * n_cols
    rows, cols = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    rows.append(i)
                    cols.append(nr * n_cols + nc)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _row_standardize(W: sp.spmatrix) -> sp.csr_matrix:
    """Row-standardize a sparse weight matrix (zero rows stay zero)."""
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    inv_sums = np.where(row_sums > 0, 1.0 / np.where(row_sums > 0, row_sums, 1.0), 0.0)
    return (sp.diags(inv_sums) @ W).tocsr()


def _naive_morans_i(x: np.ndarray, W: sp.spmatrix, center: bool = True) -> float:
    """Reference textbook implementation of Moran's I."""
    n = len(x)
    S0 = float(W.sum())
    z = (x - x.mean()) if center else x
    num = (n / S0) * float(z.T @ (W @ z))
    den = float(z.T @ z)
    return num / den


def _typed_counts(n_per_type=150, n_types=4, n_genes=200, seed=0):
    """Poisson counts with ``n_types`` cell types, each with 15 up-regulated genes."""
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
    """Log-normalized typed dataset with raw integer counts saved in ``layers['counts']``."""
    a = _typed_counts(n_per_type, n_types, n_genes, seed)
    a.layers["counts"] = a.X.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def _mock_marker_adata():
    """Small 4-cell, 6-gene AnnData with pure T, pure B, noisy T, and CD14-contaminated T."""
    var_names = ["CD3E", "CD19", "CD14", "GAPDH", "CD4", "MS4A1"]
    X = np.array(
        [
            [10.0, 0.0, 0.0, 1.0, 8.0, 0.0],  # cell_0: Pure T
            [0.0, 10.0, 0.0, 1.0, 0.0, 8.0],  # cell_1: Pure B
            [9.0, 0.0, 0.0, 20.0, 7.0, 0.0],  # cell_2: T with high background
            [8.0, 0.0, 15.0, 1.0, 6.0, 0.0],  # cell_3: T contaminated with CD14
        ]
    )
    obs = pd.DataFrame(
        {"leiden": ["0", "1", "0", "0"]}, index=[f"cell_{i}" for i in range(4)]
    )
    var = pd.DataFrame(index=var_names)
    adata = AnnData(X=X, obs=obs, var=var)
    adata.raw = adata
    return adata


def _pacmap_adata(n_vars, n_obs=60, seed=0):
    """Small random float32 AnnData for PaCMAP tests."""
    X = np.random.default_rng(seed).random((n_obs, n_vars)).astype(np.float32)
    return AnnData(X=X)


def _paga_adata_with_layout():
    """Leiden + PAGA + PAGA positions on clustered data (shared by PAGA-init tests)."""
    adata = _pacmap_adata(50, n_obs=120)
    sc.pp.neighbors(adata, n_neighbors=8, use_rep="X")
    sc.tl.leiden(
        adata, resolution=1.0, flavor="igraph", n_iterations=2, directed=False, random_state=0
    )
    sc.tl.paga(adata, groups="leiden")
    sc.pl.paga(adata, show=False, random_state=0)
    plt.close("all")
    return adata


def _moran_test_adata(sparse, n=3000):
    """Spatial grid AnnData with trend, two constant genes, and noise."""
    rng = np.random.default_rng(0)
    side = int(np.sqrt(n))
    n = side * side
    yy, xx = np.divmod(np.arange(n), side)
    signal = np.sin(xx / 6.0) + 0.3 * rng.normal(size=n)
    X = np.c_[signal - signal.min(), np.full(n, 0.1), np.full(n, 0.7), rng.random(n)]
    X = X.astype(np.float32)
    a = AnnData(sp.csr_matrix(X) if sparse else X)
    a.var_names = ["trend", "c01", "c07", "noise"]
    a.obsm["X_pos"] = np.c_[xx, yy].astype(float)
    sc.pp.neighbors(a, n_neighbors=8, use_rep="X_pos")
    return a


def _scaled_reference(max_value):
    """Scaled reference AnnData with HVG selection and PCA computed."""
    a = _typed_log(n_per_type=100)
    a.raw = a.copy()
    sc.pp.highly_variable_genes(a, n_top_genes=100, flavor="seurat")
    sc.pp.scale(a, max_value=max_value)
    sc.tl.pca(a, n_comps=20)
    return a


def _volcano_adata(pvals):
    """Minimal AnnData with scanpy-style structured arrays in ``uns['rank_genes_groups']``."""
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


@pytest.fixture
def adata_integer_counts():
    """AnnData with deterministic integer counts in ``.X`` and ``.layers['counts']``."""
    rng = np.random.default_rng(42)
    X = rng.integers(0, 10, size=(100, 50)).astype(float)
    obs = pd.DataFrame({"batch": rng.choice(["a", "b"], size=100)})
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(50)]},
        index=[f"gene_{i}" for i in range(50)],
    )
    adata = AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = X.copy()
    return adata


@pytest.fixture
def adata_float_layer():
    """AnnData with a positive float layer (``scvi_normalized``) and no counts layer."""
    rng = np.random.default_rng(42)
    X = rng.exponential(size=(100, 50)) + 0.1
    obs = pd.DataFrame({"batch": rng.choice(["a", "b"], size=100)})
    var = pd.DataFrame(
        {"gene_name": [f"gene_{i}" for i in range(50)]},
        index=[f"gene_{i}" for i in range(50)],
    )
    adata = AnnData(X=X.copy(), obs=obs, var=var)
    adata.layers["scvi_normalized"] = X.copy()
    return adata


@pytest.fixture
def pbmc_adata():
    """Synthetic sparse/dense AnnData with planted positive and negative gene correlations."""
    rng = np.random.default_rng(42)
    n_cells, n_genes = 100, 20
    X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(float)
    X[:, 1] = X[:, 0] * 2 + rng.normal(scale=0.5, size=n_cells)
    X[:, 2] = -X[:, 0] + rng.normal(scale=0.5, size=n_cells) + 10
    X[X < 0] = 0
    adata = AnnData(X=sp.csr_matrix(X))
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]
    adata.obs["score_A"] = adata.X[:, 0].toarray().ravel() + rng.normal(
        scale=0.1, size=n_cells
    )
    adata.layers["counts"] = adata.X.toarray()
    return adata


@pytest.fixture
def obs_csv_adata():
    """Small AnnData with categorical ``cell_type`` and matching ``.uns`` color palette."""
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(["T-cell", "B-cell", "T-cell", "Macrophage", "B-cell"])},
        index=["cell1", "cell2", "cell3", "cell4", "cell5"],
    )
    adata = AnnData(np.zeros((5, 2)), obs=obs)
    # Categories in sorted order: B-cell, Macrophage, T-cell
    adata.uns["cell_type_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    return adata


# =============================================================================
# Preprocessing, Normalization & General Utilities
# =============================================================================


def test_carry_palette():
    """carry_palette transfers matching colors, drops unused categories, warns on unseen ones, and validates keys."""
    src = AnnData(
        np.zeros((3, 1)),
        obs=pd.DataFrame({"ct": pd.Categorical(["B", "M", "T"])}),
    )
    src.uns["ct_colors"] = ["#111111", "#222222", "#333333"]

    dst = AnnData(
        np.zeros((2, 1)),
        obs=pd.DataFrame({"ct": pd.Categorical(["T", "B"], categories=["B", "M", "T"])}),
    )
    ret = carry_palette(src, dst, "ct")
    assert ret is dst
    assert list(dst.obs["ct"].cat.categories) == ["B", "T"]
    assert dst.uns["ct_colors"] == ["#111111", "#333333"]

    dst_unseen = AnnData(
        np.zeros((2, 1)),
        obs=pd.DataFrame({"ct": pd.Categorical(["T", "NK"])}),
    )
    with pytest.warns(UserWarning, match="No color found in source"):
        carry_palette(src, dst_unseen, "ct", fallback_color="#ABCDEF")
    assert dict(zip(dst_unseen.obs["ct"].cat.categories, dst_unseen.uns["ct_colors"])) == {
        "NK": "#ABCDEF",
        "T": "#333333",
    }

    with pytest.raises(KeyError, match="must be in .obs"):
        carry_palette(src, dst, "missing_key")

    src_no_pal = src.copy()
    del src_no_pal.uns["ct_colors"]
    with pytest.raises(KeyError, match="Source palette"):
        carry_palette(src_no_pal, dst, "ct")


def test_check_gene_adata():
    """check_gene_adata filters str/list/dict inputs against X or raw and rejects unsupported types."""
    full = AnnData(np.zeros((3, 4)), var=pd.DataFrame(index=["G0", "G1", "G2", "RAW_ONLY"]))
    full.raw = full.copy()
    adata = full[:, ["G0", "G1", "G2"]].copy()

    assert check_gene_adata(adata, "G1") == "G1"
    assert check_gene_adata(adata, "MISSING") is None
    assert check_gene_adata(adata, ["G2", "MISSING", "G0"]) == ["G2", "G0"]
    assert check_gene_adata(
        adata, {"T": "G0", "B": ["G1", "MISSING"], "Empty": ["NOPE"]}
    ) == {"T": ["G0"], "B": ["G1"], "Empty": []}

    assert check_gene_adata(adata, "RAW_ONLY", use_raw=False) is None
    assert check_gene_adata(adata, "RAW_ONLY", use_raw=True) == "RAW_ONLY"
    assert check_gene_adata(adata, ["RAW_ONLY", "G0"], use_raw=True) == ["RAW_ONLY", "G0"]

    with pytest.raises(TypeError, match="string, list, or dictionary"):
        check_gene_adata(adata, 123)


def test_pflogpf_import_error(capsys, monkeypatch):
    """pflogpf without scclr installed raises ImportError with the install steps, printing nothing.

    Changed (D07, 2026-11): the docstring is no longer printed; the error message
    carries the install instructions instead.
    """
    adata = AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]))
    monkeypatch.setitem(sys.modules, "scclr", None)

    with pytest.raises(ImportError, match="The 'scclr' package is required.*maturin develop"):
        pflogpf(adata)

    assert capsys.readouterr().out == ""


def test_pflogpf_non_integer_counts():
    """pflogpf rejects non-integer matrices with a clear ValueError."""
    adata = AnnData(X=np.array([[1.5, 2.1], [3.2, 4.9]]))
    with pytest.raises(ValueError, match="does not look like raw integer counts"):
        pflogpf(adata)


def test_pflogpf_success(monkeypatch):
    """pflogpf invokes scclr, renames the output layer, and respects copy=True."""
    adata = AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]))

    mock_scclr = MagicMock()

    def _mock_pflog(a, target, **kwargs):
        a.layers["pflog"] = a.X + 10

    mock_scclr.pp.pflog.side_effect = _mock_pflog
    monkeypatch.setitem(sys.modules, "scclr", mock_scclr)

    res = pflogpf(adata, key_added="custom_pflog")
    assert res is None
    assert "custom_pflog" in adata.layers
    assert "pflog" not in adata.layers
    np.testing.assert_allclose(adata.layers["custom_pflog"], adata.X + 10)

    adata2 = AnnData(X=np.array([[1.0, 2.0], [3.0, 4.0]]))
    res2 = pflogpf(adata2, key_added="pflogpf", copy=True)
    assert res2 is not None
    assert "pflogpf" in res2.layers
    assert "pflogpf" not in adata2.layers


def test_preprocess_subset_standard_workflow(adata_integer_counts):
    """Standard integer-count workflow builds log1p, scales, and runs PCA."""
    adata = preprocess_subset(
        adata_integer_counts,
        counts_layer="counts",
        X_layer_for_pca="log1p",
        hvg_flavor="seurat",
        scale_max_value=10.0,
        copy=True,
    )
    assert "log1p" in adata.layers
    assert np.allclose(adata.X.mean(axis=0), 0, atol=1.0)
    assert adata.shape[1] == 50
    assert "X_pca" in adata.obsm


@pytest.mark.parametrize("scale_data", [False, True])
def test_preprocess_subset_custom_layer(adata_float_layer, scale_data):
    """Custom float layer without counts warns, skips log1p, and honors scale_data."""
    with pytest.warns(UserWarning, match="Counts layer .* not found"):
        adata = preprocess_subset(
            adata_float_layer,
            counts_layer="counts",
            X_layer_for_pca="scvi_normalized",
            hvg_flavor="seurat",
            scale_data=scale_data,
            copy=True,
        )

    assert "log1p" not in adata.layers
    assert "X_pca" in adata.obsm
    if scale_data:
        assert not np.allclose(adata.X, adata.layers["scvi_normalized"])
        assert np.allclose(adata.X.mean(axis=0), 0, atol=1.0)
    else:
        assert np.allclose(adata.X, adata.layers["scvi_normalized"])


def test_preprocess_subset_seurat_v3_happy_path():
    """preprocess_subset with hvg_flavor='seurat_v3' selects HVGs from integer counts and runs PCA/Leiden."""
    pytest.importorskip("skmisc")
    a = _typed_counts(n_per_type=30, n_types=3, n_genes=60, seed=0)
    a.layers["counts"] = a.X.copy()

    out = preprocess_subset(
        a,
        counts_layer="counts",
        hvg_flavor="seurat_v3",
        n_top_genes=25,
        X_layer_for_pca="log1p",
        copy=True,
    )
    assert out.n_vars == 25
    assert "log1p" in out.layers
    assert "highly_variable" in out.var
    assert "X_pca" in out.obsm
    assert "leiden" in out.obs


def test_preprocess_subset_seurat_v3_counts_validation(adata_float_layer, adata_integer_counts):
    """seurat_v3 raises ValueError when counts are missing or non-integer."""
    with pytest.raises(ValueError, match="flavor='seurat_v3' requires raw counts"):
        preprocess_subset(
            adata_float_layer,
            hvg_flavor="seurat_v3",
            counts_layer="counts",
            copy=True,
        )

    non_int = adata_integer_counts.copy()
    non_int.layers["counts"] = non_int.layers["counts"] + 0.37
    with pytest.raises(ValueError, match="does not look like raw integer counts"):
        preprocess_subset(
            non_int,
            hvg_flavor="seurat_v3",
            counts_layer="counts",
            copy=True,
        )


def test_preprocess_subset_triku_with_floats(adata_float_layer):
    """preprocess_subset with hvg_flavor='triku' runs on float layers and sets highly_variable."""
    pytest.importorskip("triku")
    with pytest.warns(UserWarning, match="Counts layer .* not found"):
        adata = preprocess_subset(
            adata_float_layer,
            hvg_flavor="triku",
            X_layer_for_pca="scvi_normalized",
            n_top_genes=20,
            copy=True,
        )
    assert "triku_distance" in adata.var
    assert "highly_variable" in adata.var


def test_preprocess_subset_takes_counts_from_raw_after_gene_filtering():
    """Aligned counts are extracted from adata.raw after min_cells gene filtering."""
    a = _typed_counts(n_per_type=50)
    a.X[:, :5] = 0
    a.X[0, :5] = 3  # expressed in 1 cell: dropped by filter_genes(min_cells=3)
    a.raw = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)

    out = preprocess_subset(a, hvg_flavor="seurat", n_top_genes=50, copy=True)
    assert out.n_vars == 50
    raw_counts = a.raw[:, out.var_names].X
    np.testing.assert_array_equal(np.asarray(out.layers["counts"]), np.asarray(raw_counts))


def test_preprocess_subset_caps_n_comps_on_small_inputs():
    """n_comps is capped below min(n_obs, n_hvgs) so ARPACK PCA never fails."""
    small = _typed_log(n_per_type=5)  # 20 cells < default n_comps=32
    out = preprocess_subset(small, hvg_flavor="seurat", n_top_genes=100, copy=True)
    assert out.obsm["X_pca"].shape[1] <= 19

    few_hvgs = _typed_log(n_per_type=50)
    out = preprocess_subset(
        few_hvgs, hvg_flavor="seurat", n_top_genes=20, subset_to_hvgs=False, copy=True
    )
    assert out.obsm["X_pca"].shape[1] <= 19


def test_find_cluster_keys_ignores_annotation_columns_and_mixed_names():
    """find_cluster_keys full-matches pattern and naturally sorts numeric resolutions."""
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


def test_plot_clustering_tree_draws_with_current_matplotlib():
    """plot_clustering_tree renders nested clusterings without deprecated matplotlib colormap APIs."""
    adata = AnnData(X=np.zeros((60, 3)))
    adata.obs["leiden_0.5"] = pd.Categorical(np.repeat(["0", "1"], 30))
    adata.obs["leiden_1.0"] = pd.Categorical(np.repeat(["0", "1", "2"], 20))
    try:
        fig = plot_clustering_tree(adata, ["leiden_0.5", "leiden_1.0"])
        assert isinstance(fig, plt.Figure) and len(fig.axes) == 1
    finally:
        plt.close("all")


def test_plot_clustering_tree_uses_uns_colors_for_integer_categories():
    """plot_clustering_tree applies stored .uns palette colors for integer categorical cluster IDs."""
    a = AnnData(np.zeros((6, 1)))
    a.obs["leiden_0.5"] = pd.Categorical([0, 0, 1, 1, 2, 2])
    a.obs["leiden_1.0"] = pd.Categorical([0, 1, 2, 3, 4, 5])
    a.uns["leiden_0.5_colors"] = ["#ff0000", "#00ff00", "#0000ff"]
    try:
        fig = plot_clustering_tree(a, ["leiden_0.5", "leiden_1.0"])
        facecolors = {
            mcolors.to_hex(c)
            for coll in fig.axes[0].collections
            for c in coll.get_facecolors()
        }
        assert {"#ff0000", "#00ff00", "#0000ff"} <= facecolors
    finally:
        plt.close("all")


def test_export_and_import_obs_csv_roundtrip(obs_csv_adata, tmp_path):
    """Exporting and re-importing an obs column preserves labels, custom index name, and palette colors."""
    out_path = tmp_path / "exported.csv"
    export_obs_from_adata_to_csv(
        obs_csv_adata,
        obs_key="cell_type",
        output_path=str(out_path),
        index_name="MyCellID",
    )
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert {"MyCellID", "cell_type", "cell_type_colors"} <= set(df.columns)
    assert df.loc[df["MyCellID"] == "cell2", "cell_type_colors"].iloc[0] == "#ff0000"
    assert df.loc[df["MyCellID"] == "cell4", "cell_type_colors"].iloc[0] == "#00ff00"

    adata_new = AnnData(
        np.zeros((5, 2)),
        obs=pd.DataFrame(index=["cell1", "cell2", "cell3", "cell4", "cell5"]),
    )
    import_obs_to_adata_from_csv(
        path=str(out_path),
        adata=adata_new,
        obs_key="cell_type",
        index_col="MyCellID",
        index_name="CustomID",
    )
    assert adata_new.obs.index.name == "CustomID"
    cat_to_color = dict(
        zip(adata_new.obs["cell_type"].cat.categories, adata_new.uns["cell_type_colors"])
    )
    assert cat_to_color == {
        "B-cell": "#ff0000",
        "Macrophage": "#00ff00",
        "T-cell": "#0000ff",
    }


def test_import_obs_overwrite_and_suffix(obs_csv_adata, tmp_path):
    """Repeated non-overwriting imports allocate _imported, _imported_2 suffixes; overwrite=True replaces in place."""
    out_path = tmp_path / "exported.csv"
    export_obs_from_adata_to_csv(
        obs_csv_adata, obs_key="cell_type", output_path=str(out_path)
    )
    orig = list(obs_csv_adata.obs["cell_type"])

    import_obs_to_adata_from_csv(
        path=str(out_path),
        adata=obs_csv_adata,
        obs_key="cell_type",
        index_col="Cell_ID",
        overwrite_existing=False,
    )
    import_obs_to_adata_from_csv(
        path=str(out_path),
        adata=obs_csv_adata,
        obs_key="cell_type",
        index_col="Cell_ID",
        overwrite_existing=False,
    )
    assert list(obs_csv_adata.obs["cell_type"]) == orig
    assert "cell_type_imported" in obs_csv_adata.obs.columns
    assert "cell_type_imported_colors" in obs_csv_adata.uns
    assert "cell_type_imported_2" in obs_csv_adata.obs.columns

    df = pd.read_csv(out_path)
    df.loc[df["Cell_ID"] == "cell1", "cell_type"] = "Unknown"
    df.to_csv(out_path, index=False)

    import_obs_to_adata_from_csv(
        path=str(out_path),
        adata=obs_csv_adata,
        obs_key="cell_type",
        index_col="Cell_ID",
        overwrite_existing=True,
    )
    assert obs_csv_adata.obs.loc["cell1", "cell_type"] == "Unknown"


def test_import_export_obs_edge_cases(tmp_path):
    """CSV/TSV import handles duplicate barcodes, partial overlap, TSV auto-detection, and missing keys."""
    tsv_path = tmp_path / "labels.tsv"
    tsv_path.write_text("Cell_ID\tct\nc0\tT\nc0\tDUPE_DROP\nc1\tB\nc_extra\tNK\n")

    a = AnnData(np.zeros((3, 1)), obs=pd.DataFrame(index=["c0", "c1", "c2"]))
    with pytest.warns(UserWarning, match="duplicate barcodes"):
        import_obs_to_adata_from_csv(str(tsv_path), a, obs_key="ct", index_col="Cell_ID")

    assert a.obs.loc["c0", "ct"] == "T"  # first duplicate kept
    assert a.obs.loc["c1", "ct"] == "B"
    assert pd.isna(a.obs.loc["c2", "ct"])  # unmatched cell gets NaN

    with pytest.raises(ValueError, match="not found in CSV"):
        import_obs_to_adata_from_csv(str(tsv_path), a, obs_key="missing_col", index_col="Cell_ID")

    with pytest.raises(ValueError, match="not found in adata.obs"):
        export_obs_from_adata_to_csv(a, obs_key="missing_obs", output_path=str(tmp_path / "x.csv"))


def test_export_obs_keeps_dotted_names(tmp_path):
    """export_obs_from_adata_to_csv appends .csv to dotted stems instead of stripping after the dot."""
    a = AnnData(np.zeros((3, 1)))
    a.obs["score"] = [1.0, 2.0, 3.0]
    export_obs_from_adata_to_csv(a, "score", output_path=str(tmp_path / "run_2024.10"))
    export_obs_from_adata_to_csv(a, "score", output_path=str(tmp_path / "run_2024.11"))
    assert (tmp_path / "run_2024.10.csv").exists()
    assert (tmp_path / "run_2024.11.csv").exists()
    export_obs_from_adata_to_csv(a, "score", output_path=str(tmp_path))
    assert (tmp_path / "score_export.csv").exists()


# =============================================================================
# Graph Helpers, Gene Extraction & Feature Correlations
# =============================================================================


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_smooth_expression_on_graph(sparse, dtype):
    """smooth_expression_on_graph matches row-normalized (A + I) @ X for sparse/dense float32/float64 without mutation."""
    rng = np.random.default_rng(7)
    n, g = 12, 4
    X_dense = rng.normal(size=(n, g)).astype(dtype)
    X_dense[:, -1] = 3.5  # constant column invariant under lazy random walk
    X_in = sp.csr_matrix(X_dense) if sparse else X_dense.copy()

    W = _build_grid_graph(3, 4).astype(dtype)
    W_orig = W.copy()
    adata = AnnData(X=np.zeros((n, g), dtype=dtype))
    adata.obsp["connectivities"] = W

    out = smooth_expression_on_graph(adata, X_in)
    assert isinstance(out, np.ndarray)
    A = W_orig.toarray().astype(np.float64) + np.eye(n)
    expected = (A / A.sum(axis=1, keepdims=True)) @ X_dense.astype(np.float64)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out[:, -1], 3.5, rtol=1e-5)

    # Original graph and input matrix must not be mutated
    assert (adata.obsp["connectivities"] != W_orig).nnz == 0
    if sparse:
        assert (X_in != sp.csr_matrix(X_dense)).nnz == 0
    else:
        np.testing.assert_array_equal(X_in, X_dense)

    with pytest.raises(ValueError, match="not found in adata.obsp"):
        smooth_expression_on_graph(adata, X_in, weights_key="missing_graph")


def test_graph_helpers_work_without_scanpy():
    """single_cell imports and its numpy/scipy graph helpers run without scanpy (Pyodide / WASM safe)."""
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
        W = sp.csr_matrix(
            (np.ones(12), (np.r_[rows, rows], np.r_[(rows + 1) % 6, (rows - 1) % 6])),
            shape=(6, 6),
        )
        adata.obsp["connectivities"] = W

        S = smooth_expression_on_graph(adata, X)
        A = W.toarray() + np.eye(6)
        expected = (A / A.sum(1, keepdims=True)) @ X
        assert np.allclose(S, expected), (S, expected)
        assert np.allclose(S[:, 2], 4.0)

        res = find_correlated_features(adata, "a", metrics=("pearson",))
        assert np.isclose(res.loc["a", "pearson"], 1.0)

        try:
            ensure_neighbors(anndata.AnnData(X))
        except ImportError as e:
            assert "scanpy" in str(e)
        else:
            raise AssertionError("ensure_neighbors should need scanpy")
        assert "scanpy" not in sys.modules
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-3000:]


def test_custom_graph_keys_follow_scanpy_naming():
    """ensure_neighbors and build_rowstd_connectivities honor scanpy '<key_added>_connectivities' naming."""
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


def test_extract_gene_vector_sparse_sum_mean():
    """_extract_gene_vector supports sum, mean, first, and last duplicate policies on integer CSR matrices."""
    M = sp.csr_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    adata = AnnData(X=M)
    adata.var_names = ["A", "B", "B"]

    np.testing.assert_allclose(
        _extract_gene_vector(adata, "B", source="X", duplicate_policy="sum"),
        [5.0, 11.0, 17.0],
    )
    np.testing.assert_allclose(
        _extract_gene_vector(adata, "B", source="X", duplicate_policy="mean"),
        [2.5, 5.5, 8.5],
    )
    np.testing.assert_allclose(
        _extract_gene_vector(adata, "B", source="X", duplicate_policy="first"),
        [2.0, 5.0, 8.0],
    )
    np.testing.assert_allclose(
        _extract_gene_vector(adata, "B", source="X", duplicate_policy="last"),
        [3.0, 6.0, 9.0],
    )


def test_exact_gene_name_wins_over_case_variants():
    """Exact var_name matches win over case variants; case-insensitive fallback combines variants."""
    X = np.array([[5.0, 0.0, 1.0], [10.0, 0.0, 2.0], [15.0, 0.0, 3.0]])
    a = AnnData(X)
    a.var_names = ["Cd4", "CD4", "B"]

    np.testing.assert_array_equal(_extract_gene_vector(a, "CD4"), [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(_extract_gene_vector(a, "Cd4"), [5.0, 10.0, 15.0])
    np.testing.assert_array_equal(
        _extract_gene_matrix(a, ["CD4", "Cd4"]), [[0, 5], [0, 10], [0, 15]]
    )
    np.testing.assert_array_equal(_extract_gene_vector(a, "cd4"), [2.5, 5.0, 7.5])
    np.testing.assert_array_equal(_extract_gene_matrix(a, ["cd4"]).ravel(), [2.5, 5.0, 7.5])

    b = AnnData(np.array([[1.0, 3.0], [2.0, 4.0]]))
    b.var_names = ["A", "A"]
    np.testing.assert_array_equal(_extract_gene_vector(b, "A"), [2.0, 3.0])


def test_find_correlated_features_pearson_sparse(pbmc_adata):
    """Pearson correlation on sparse X ranks self=1 and recovers planted positive/negative partners."""
    res = find_correlated_features(pbmc_adata, target="Gene0", metrics=["pearson"])
    assert "pearson" in res.columns
    assert res.index[0] == "Gene0"
    assert res.loc["Gene0", "pearson"] == pytest.approx(1.0)
    assert res.loc["Gene1", "pearson"] > 0.8
    assert res.loc["Gene2", "pearson"] < -0.8


def test_find_correlated_features_dense_layer(pbmc_adata):
    """Dense layer and sparse X produce identical Pearson correlations."""
    res_sparse = find_correlated_features(pbmc_adata, target="Gene0", metrics=["pearson"])
    res_dense = find_correlated_features(
        pbmc_adata, target="Gene0", layer="counts", metrics=["pearson"]
    )
    np.testing.assert_allclose(
        res_sparse["pearson"].values, res_dense["pearson"].values, atol=1e-6
    )


def test_find_correlated_features_target_obs(pbmc_adata):
    """An .obs column can be used directly as the correlation target."""
    res = find_correlated_features(pbmc_adata, target="score_A", metrics=["pearson"])
    assert res.loc["Gene0", "pearson"] > 0.9
    assert res.loc["Gene1", "pearson"] > 0.8
    assert res.loc["Gene2", "pearson"] < -0.8


def test_find_correlated_features_multiple_metrics_match_scipy(pbmc_adata):
    """Pearson, Spearman, and Wasserstein outputs match scipy.stats oracles for every feature."""
    res = find_correlated_features(
        pbmc_adata, target="Gene0", metrics=["pearson", "spearman", "wasserstein"]
    )
    assert list(res.columns) == ["pearson", "spearman", "wasserstein"]
    assert res.loc["Gene0", "wasserstein"] == pytest.approx(0.0, abs=1e-7)

    X = pbmc_adata.X.toarray()
    t = X[:, 0]
    t_z = (t - t.mean()) / t.std()
    for j, gene in enumerate(pbmc_adata.var_names):
        col = X[:, j]
        col_z = (col - col.mean()) / col.std()
        assert res.loc[gene, "pearson"] == pytest.approx(stats.pearsonr(col, t)[0], abs=1e-5)
        assert res.loc[gene, "spearman"] == pytest.approx(stats.spearmanr(col, t)[0], abs=1e-6)
        assert res.loc[gene, "wasserstein"] == pytest.approx(
            stats.wasserstein_distance(col_z, t_z), abs=1e-5
        )


def test_find_correlated_features_exclude(pbmc_adata):
    """exclude_features omits listed genes while ignoring nonexistent names."""
    res = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        exclude_features=["Gene0", "Gene1", "NonExistentGene"],
    )
    assert "Gene0" not in res.index
    assert "Gene1" not in res.index
    assert "Gene2" in res.index


def test_find_correlated_features_graph_smoothing(pbmc_adata):
    """use_graph=True diffuses features without mutating .X or .obsp['connectivities']."""
    sc.pp.neighbors(pbmc_adata, n_neighbors=5, use_rep="X")
    X_orig = pbmc_adata.X.copy()
    conn_orig = pbmc_adata.obsp["connectivities"].copy()

    res_raw = find_correlated_features(pbmc_adata, target="Gene0", metrics=["pearson"])
    res_graph = find_correlated_features(
        pbmc_adata,
        target="Gene0",
        metrics=["pearson"],
        use_graph=True,
        weights_key="connectivities",
    )
    assert res_graph.loc["Gene0", "pearson"] == pytest.approx(1.0)
    assert res_graph.loc["Gene1", "pearson"] != res_raw.loc["Gene1", "pearson"]
    assert (pbmc_adata.X != X_orig).nnz == 0
    assert (pbmc_adata.obsp["connectivities"] != conn_orig).nnz == 0


def test_find_correlated_features_sorting(pbmc_adata):
    """Results sort ascending for wasserstein, descending for correlation metrics, by the first requested metric."""
    res_pearson = find_correlated_features(pbmc_adata, target="Gene0", metrics=["pearson"])
    assert res_pearson.index[0] == "Gene0"
    assert res_pearson.index[-1] == "Gene2"

    res_wass = find_correlated_features(pbmc_adata, target="Gene0", metrics=["wasserstein"])
    assert res_wass.index[0] == "Gene0"

    res_multi = find_correlated_features(
        pbmc_adata, target="Gene0", metrics=["wasserstein", "pearson"]
    )
    assert list(res_multi.index) == list(res_wass.index)


def test_find_correlated_features_drops_nan_targets():
    """Non-finite target entries are warned and dropped (and rejected when use_graph=True)."""
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

    with pytest.raises(ValueError, match="use_graph=True needs a value for every cell"):
        find_correlated_features(a, "score", metrics=("pearson",), use_graph=True)


def test_find_correlated_features_spearman_constant_genes():
    """Constant and all-zero genes yield Spearman r=0 without RuntimeWarning."""
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


# =============================================================================
# Moran's I Spatial Autocorrelation
# =============================================================================


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("block_genes", [3, 100])
@pytest.mark.parametrize("center", [True, False])
def test_morans_i_all_fast_matches_naive(sparse, block_genes, center):
    """morans_i_all_fast matches textbook Moran's I across sparse/dense inputs, block sizes, and centering."""
    rng = np.random.default_rng(42)
    n_rows, n_cols = 8, 8
    n = n_rows * n_cols
    n_genes = 12
    A = _build_grid_graph(n_rows, n_cols)
    W_std = _row_standardize(A)

    X_dense = rng.uniform(0.0, 5.0, size=(n, n_genes))
    X_dense += np.outer(np.linspace(0.0, 2.0, n), np.ones(n_genes))
    if sparse:
        X_dense[rng.random((n, n_genes)) < 0.5] = 0.0
        X_in = sp.csr_matrix(X_dense.astype(np.float32))
    else:
        X_in = X_dense.astype(np.float32)

    adata = AnnData(X=X_in)
    adata.var_names = [f"Gene_{j}" for j in range(n_genes)]
    adata.obsp["connectivities"] = A

    df = morans_i_all_fast(
        adata, block_genes=block_genes, center=center, deduplicate="none"
    )
    assert {"gene", "I", "pval_z", "z_score"} <= set(df.columns)
    assert len(df) == n_genes
    assert np.all(np.diff(df["I"].to_numpy()) <= 0)

    by_gene = df.set_index("gene")
    for j, gene in enumerate(adata.var_names):
        expected = _naive_morans_i(X_dense[:, j], W_std, center=center)
        assert by_gene.loc[gene, "I"] == pytest.approx(expected, rel=1e-4, abs=1e-4)


def test_morans_i_eigenvectors():
    """Constant uncentered vector gives I=1 and bipartite checkerboard eigenvector gives I=-1."""
    n_rows, n_cols = 10, 10
    n = n_rows * n_cols
    W_std = _row_standardize(_build_grid_graph(n_rows, n_cols))

    adata_const = AnnData(X=np.ones((n, 1)))
    adata_const.var_names = ["const"]
    res_const = morans_i_all_fast(adata_const, W_rowstd=W_std, center=False)
    assert np.isclose(res_const.set_index("gene").loc["const", "I"], 1.0, atol=1e-5)

    yy, xx = np.divmod(np.arange(n), n_cols)
    x_check = np.where((yy + xx) % 2 == 0, 1.0, -1.0)
    adata_check = AnnData(X=x_check[:, None])
    adata_check.var_names = ["checkerboard"]
    res_check = morans_i_all_fast(adata_check, W_rowstd=W_std, center=True)
    assert np.isclose(
        res_check.set_index("gene").loc["checkerboard", "I"], -1.0, atol=1e-5
    )


def test_moran_bias_isolated_nodes():
    """morans_i_all_fast is exact (I=1.0) with isolated nodes, shifted means, and unstandardized weights."""
    W = sp.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    x = np.array([[10.0], [10.0], [-10.0], [-10.0]], dtype=np.float32)
    adata = AnnData(X=x, var=pd.DataFrame(index=["gene1"]))

    # Case 1: unstandardized adjacency with isolated nodes (mean 0)
    res = morans_i_all_fast(adata, W_rowstd=W, block_genes=100)
    assert np.isclose(res["I"].iloc[0], 1.0, atol=1e-5)

    # Case 2: shifted mean (+10) must be invariant under centering
    adata_shifted = AnnData(X=x + 10.0, var=pd.DataFrame(index=["gene1"]))
    res_shifted = morans_i_all_fast(adata_shifted, W_rowstd=W, block_genes=100)
    assert np.isclose(res_shifted["I"].iloc[0], 1.0, atol=1e-5)

    # Case 3: scaled non-row-standardized weights (2 * W) must cancel via n/S0
    res_scaled = morans_i_all_fast(adata, W_rowstd=2.0 * W, block_genes=100)
    assert np.isclose(res_scaled["I"].iloc[0], 1.0, atol=1e-5)


def test_moran_statistics_gaussian():
    """Under Gaussian null noise, E[I] = -1/(n-1), Z-scores follow N(0, 1), and p-values are uniform."""
    n_rows, n_cols = 20, 20
    n = n_rows * n_cols
    W_std = _row_standardize(_build_grid_graph(n_rows, n_cols))

    n_genes = 5000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, n_genes))
    adata = AnnData(X=X)
    adata.var_names = [f"g_{i}" for i in range(n_genes)]

    res = morans_i_all_fast(adata, W_rowstd=W_std)

    # Analytical null expectation E[I] = -1 / (n - 1)
    assert np.isclose(res["I"].mean(), -1.0 / (n - 1), atol=0.005)

    z_scores = res["z_score"].to_numpy()
    z_scores = z_scores[np.isfinite(z_scores)]
    assert abs(np.mean(z_scores)) < 0.05
    assert abs(np.std(z_scores) - 1.0) < 0.05

    prop_significant = np.mean(res["pval_z"].to_numpy() < 0.05)
    assert 0.04 < prop_significant < 0.06


def test_moran_statistics_kurtotic():
    """Under high-kurtosis sparse spikes, randomization Z-scores remain zero-mean and unit-variance."""
    n_rows, n_cols = 20, 20
    n = n_rows * n_cols
    W_std = _row_standardize(_build_grid_graph(n_rows, n_cols))

    n_genes = 5000
    n_spikes = int(0.05 * n)
    rng = np.random.default_rng(99)
    X = np.zeros((n, n_genes))
    spike_rows = np.argpartition(rng.random((n, n_genes)), n_spikes, axis=0)[:n_spikes]
    X[spike_rows, np.arange(n_genes)[None, :]] = rng.exponential(
        scale=10.0, size=(n_spikes, n_genes)
    )

    adata = AnnData(X=X)
    adata.var_names = [f"g_{i}" for i in range(n_genes)]

    res = morans_i_all_fast(adata, W_rowstd=W_std)
    z_scores = res["z_score"].to_numpy()
    z_scores = z_scores[np.isfinite(z_scores)]
    assert abs(np.mean(z_scores)) < 0.05
    assert abs(np.std(z_scores) - 1.0) < 0.1


@pytest.mark.parametrize("sparse", [False, True])
def test_morans_i_all_fast_constant_genes_are_nan(sparse):
    """Constant genes produce NaN Moran's I and p-value instead of spurious I=1 from float32 drift."""
    out = morans_i_all_fast(_moran_test_adata(sparse)).set_index("gene")
    assert np.isnan(out.loc["c01", "I"]) and np.isnan(out.loc["c07", "I"])
    assert np.isnan(out.loc["c01", "pval_z"])
    assert out.index[0] == "trend"


def test_morans_i_all_fast_pvalues_do_not_underflow():
    """High Z-score p-values are computed in float64 and do not underflow to 0."""
    rng = np.random.default_rng(0)
    side = 54
    yy, xx = np.divmod(np.arange(side * side), side)
    signal = np.sin(xx / 6.0) + rng.normal(size=side * side)
    a = AnnData(np.c_[signal - signal.min(), rng.random(side * side)].astype(np.float32))
    a.var_names = ["trend", "noise"]
    a.obsm["X_pos"] = np.c_[xx, yy].astype(float)
    sc.pp.neighbors(a, n_neighbors=8, use_rep="X_pos")

    out = morans_i_all_fast(a).set_index("gene")
    z = out.loc["trend", "z_score"]
    assert 13 < z < 37
    p = out.loc["trend", "pval_z"]
    assert out["pval_z"].dtype == np.float64
    assert 0 < p < 1e-40
    np.testing.assert_allclose(p, 2 * norm.sf(z), rtol=1e-6)


def test_morans_dedup_modes_agree_on_case_variants():
    """All deduplicate modes preserve case-distinct genes while collapsing identical duplicates."""
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
    assert reported["none"] == ["B", "B", "GENE1", "Gene1"]
    for mode in ("mean", "sum", "first", "last"):
        assert reported[mode] == ["B", "GENE1", "Gene1"], mode
    I = morans_i_all_fast(a, deduplicate="mean").set_index("gene")["I"]
    assert I["Gene1"] > 0.5 and I["GENE1"] < 0.3


def test_morans_i_and_build_pysal_weights():
    """build_pysal_weights row-standardizes connectivities and morans_i matches morans_i_all_fast."""
    pytest.importorskip("esda")
    pytest.importorskip("libpysal")

    rng = np.random.default_rng(42)
    n_rows, n_cols = 8, 8
    n = n_rows * n_cols
    yy, xx = np.divmod(np.arange(n), n_cols)
    X = np.c_[
        xx.astype(float),
        np.where((yy + xx) % 2 == 0, 1.0, -1.0),
        rng.standard_normal(n),
    ]
    a = AnnData(X=X)
    a.var_names = ["grad", "check", "noise"]
    a.obsp["connectivities"] = _build_grid_graph(n_rows, n_cols)

    w = build_pysal_weights(a)
    assert w.transform.upper() == "R"
    np.testing.assert_allclose(np.asarray(w.sparse.sum(axis=1)).ravel(), 1.0, atol=1e-6)

    genes = ["grad", "check", "noise"]
    df_pysal = morans_i(a, genes, w=w, permutations=19)
    df_auto = morans_i(a, genes, w=None, permutations=19)
    df_fast = morans_i_all_fast(a)

    assert list(df_pysal.columns) == ["gene", "I", "p_sim", "z_sim"]
    assert np.all(np.diff(df_pysal["I"].to_numpy()) <= 0)
    assert ((df_pysal["p_sim"] > 0) & (df_pysal["p_sim"] <= 1)).all()
    np.testing.assert_allclose(
        df_pysal.set_index("gene").loc[genes, "I"],
        df_fast.set_index("gene").loc[genes, "I"],
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        df_auto.set_index("gene").loc[genes, "I"],
        df_pysal.set_index("gene").loc[genes, "I"],
        rtol=1e-6,
    )

    empty = morans_i(a, [], w=w, permutations=9)
    assert empty.empty
    assert list(empty.columns) == ["gene", "I", "p_sim", "z_sim"]


# =============================================================================
# Cell-Type Scoring, Marker Filtering & Cluster Annotation
# =============================================================================


def test_score_celltypes_binned_analytic():
    """Exact closed-form verification of binned, binned_weighted, net_binned, and net_binned_weighted."""
    # P1: nz=[10,4,2], med=4 -> binned=[1.0, 0.5, 0.0, 0.5]
    # P2: nz=[2,8],    med=5 -> binned=[0.5, 1.0, 0.0, 0.0]
    # P3: all zero           -> binned=[0.0, 0.0, 0.0, 0.0]
    # N1: nz=[2,6,2],  med=2 -> binned=[0.0, 0.5, 1.0, 0.5]
    X = np.array(
        [
            [10.0, 2.0, 0.0, 0.0],
            [4.0, 8.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 6.0],
            [2.0, 0.0, 0.0, 2.0],
        ]
    )
    adata = AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(4)]),
        var=pd.DataFrame(index=["P1", "P2", "P3", "N1"]),
    )
    pos = {"T": ["P1", "P2"], "T_zero": ["P2", "P3"]}
    neg = {"T": ["N1", "P3"]}

    # Non-net methods must ignore neg_markers completely
    s_bin = score_celltypes(adata, pos, neg, score_method="binned", use_raw=False)
    s_bin_noneg = score_celltypes(adata, pos, None, score_method="binned", use_raw=False)
    np.testing.assert_allclose(s_bin["T"], [0.75, 0.75, 0.0, 0.25])
    np.testing.assert_allclose(s_bin["T_zero"], [0.25, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(s_bin["T"], s_bin_noneg["T"])

    s_w = score_celltypes(adata, pos, neg, score_method="binned_weighted", use_raw=False)
    s_w_noneg = score_celltypes(adata, pos, None, score_method="binned_weighted", use_raw=False)
    np.testing.assert_allclose(s_w["T"], [0.75, 0.75, 0.0, 0.125])
    np.testing.assert_allclose(s_w["T_zero"], s_bin["T_zero"].to_numpy() * 0.5)
    np.testing.assert_allclose(s_w["T"], s_w_noneg["T"])

    # Net methods subtract negative scores and clip to [0, 1]
    s_net = score_celltypes(adata, pos, neg, score_method="net_binned", use_raw=False)
    np.testing.assert_allclose(s_net["T"], [0.75, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(s_net["T_zero"], s_bin["T_zero"])

    s_net_w = score_celltypes(
        adata, pos, neg, score_method="net_binned_weighted", use_raw=False
    )
    np.testing.assert_allclose(s_net_w["T"], [0.75, 0.625, 0.0, 0.0])
    np.testing.assert_allclose(s_net_w["T_zero"], s_w["T_zero"])


@pytest.mark.parametrize("net_method,pos_method", NET_METHODS)
def test_score_celltypes_net_negative_marker_edge_cases(net_method, pos_method):
    """Net methods penalize negative markers while preserving positive scores for omitted/empty/absent neg sets."""
    adata = _mock_marker_adata()
    pos_dict = {"T-cell": ["CD3E", "CD4"], "B-cell": ["CD19", "MS4A1"]}
    scores_pos = score_celltypes(adata, pos_dict, score_method=pos_method)

    # 1. Cell type omitted from neg_dict matches pos_method; contaminated cell_3 (CD14=15) is penalized
    scores_neg = score_celltypes(
        adata,
        pos_dict,
        cell_type_negative_markers_dict={"T-cell": ["CD14"]},
        score_method=net_method,
    )
    np.testing.assert_allclose(scores_neg["B-cell"], scores_pos["B-cell"])
    assert scores_neg.loc["cell_3", "T-cell"] < scores_pos.loc["cell_3", "T-cell"]
    diff_pos = scores_pos.loc["cell_0", "T-cell"] - scores_pos.loc["cell_3", "T-cell"]
    diff_neg = scores_neg.loc["cell_0", "T-cell"] - scores_neg.loc["cell_3", "T-cell"]
    assert diff_neg > diff_pos

    # 2. Empty negative marker list leaves score unchanged and finite
    scores_empty = score_celltypes(
        adata,
        {"T-cell": ["CD3E"]},
        cell_type_negative_markers_dict={"T-cell": []},
        score_method=net_method,
    )
    scores_single_pos = score_celltypes(adata, {"T-cell": ["CD3E"]}, score_method=pos_method)
    np.testing.assert_allclose(scores_empty["T-cell"], scores_single_pos["T-cell"])
    assert np.isfinite(scores_empty["T-cell"]).all()

    # 3. Absent negative gene leaves score unchanged and logs into marker_missing_report
    scores_absent = score_celltypes(
        adata,
        {"T-cell": ["CD3E"]},
        cell_type_negative_markers_dict={"T-cell": ["UNKNOWN_GENE"]},
        score_method=net_method,
    )
    np.testing.assert_allclose(scores_absent["T-cell"], scores_single_pos["T-cell"])
    report = adata.uns["marker_missing_report"]
    row = report[(report["cell_type"] == "T-cell") & (report["marker_set"] == "neg")]
    assert row["missing_genes"].tolist() == ["UNKNOWN_GENE"]


def test_score_celltypes_min_markers():
    """Cell types with fewer present positive markers than min_markers receive all-NaN scores."""
    adata = _mock_marker_adata()
    scores = score_celltypes(
        adata,
        {"Enough": ["CD3E", "CD4"], "TooFew": ["CD3E", "NOT_A_GENE"]},
        min_markers=2,
        score_method="binned",
    )
    assert np.isfinite(scores["Enough"]).all()
    assert scores["TooFew"].isna().all()


def test_score_celltypes_without_raw_uses_X_by_default():
    """Default use_raw=True falls back cleanly to .X when adata.raw is None."""
    a = _typed_log(n_per_type=60)
    assert a.raw is None
    markers = {
        "T0": [f"G{i}" for i in range(15)],
        "T1": [f"G{i}" for i in range(15, 30)],
    }

    scores = score_celltypes(a, markers)
    assert (
        scores["T0"][a.obs["ct"] == "T0"].mean()
        > scores["T0"][a.obs["ct"] == "T1"].mean()
    )

    a.obs["cl"] = a.obs["ct"].astype(str)
    cdf = annotate_clusters_by_markers(a, "cl", markers)
    assert cdf.loc["T0", "assigned_cell_type"] == "T0"


def test_score_celltypes_binned_uses_layer_over_raw():
    """An explicit layer takes precedence over adata.raw in binned scoring."""
    a = _typed_log(n_per_type=60)
    a.raw = a.copy()
    a.layers["zeros"] = np.zeros_like(a.X)
    markers = {"T0": [f"G{i}" for i in range(15)]}

    scores = score_celltypes(a, markers, layer="zeros", score_method="binned")
    assert np.all(scores["T0"].to_numpy() == 0)


@pytest.mark.parametrize("sparse", [False, True])
def test_binned_scores_batched_matches_per_gene(sparse):
    """Batched binned scoring matches per-gene extraction on both dense and sparse matrices with case duplicates."""
    a = _typed_log(n_per_type=40)
    a.var_names = [f"G{i}" for i in range(a.n_vars - 1)] + ["g0"]
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

    scores = score_celltypes(a, markers, use_raw=False, score_method="binned")
    for ct in markers:
        np.testing.assert_allclose(scores[ct], expected[ct])


def test_annotation_with_negative_selection():
    """CD14 as a T-cell negative marker lowers cluster 0's T-cell evidence without affecting B-cells."""
    adata = _mock_marker_adata()
    pos_dict = {
        "T-cell": ["CD3E", "CD4"],
        "B-cell": ["CD19", "MS4A1"],
        "Contaminated": ["CD14"],
    }
    neg_dict = {"T-cell": ["CD14"]}

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
    s_base = score_celltypes(adata, pos_dict, score_method="net_binned")
    s_neg = score_celltypes(
        adata,
        pos_dict,
        cell_type_negative_markers_dict=neg_dict,
        score_method="net_binned",
    )
    in_c0 = (adata.obs["leiden"] == "0").to_numpy()
    assert s_neg["T-cell"][in_c0].mean() < s_base["T-cell"][in_c0].mean()
    assert df_neg.loc["1", "assigned_cell_type"] == "B-cell"


def test_annotation_statistical_invariants():
    """Probability of Superiority matches 1.0 for disjoint, 0.5 for equal, and Phi(mu/sigma) for Gaussians."""
    n_per_cluster = 1000
    n_cells = 3 * n_per_cluster
    obs = pd.DataFrame(
        {"leiden": ["0"] * n_per_cluster + ["1"] * n_per_cluster + ["2"] * n_per_cluster},
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    rng = np.random.default_rng(42)

    sA = np.concatenate(
        [
            np.full(n_per_cluster, 10.0),
            rng.standard_normal(n_per_cluster),
            rng.normal(loc=1.0, scale=1.0, size=n_per_cluster),
        ]
    )
    sB = np.concatenate(
        [
            np.full(n_per_cluster, 0.0),
            rng.standard_normal(n_per_cluster),
            rng.normal(loc=0.0, scale=1.0, size=n_per_cluster),
        ]
    )
    scores = pd.DataFrame({"TypeA": sA, "TypeB": sB}, index=obs.index)
    adata = AnnData(X=np.zeros((n_cells, 2)), obs=obs)

    res_df = annotate_clusters_by_markers(
        adata,
        cluster_key="leiden",
        cell_type_markers_dict=None,
        scores=scores,
        normalize_scores=False,
        write_to_obs=True,
    )

    row0 = res_df.loc["0"]
    assert row0["assigned_cell_type"] == "TypeA"
    assert np.isclose(row0["softmax_p"], 1.0)
    assert np.isclose(row0["uncertainty"], 0.0)

    row1 = res_df.loc["1"]
    assert 0.45 < row1["softmax_p"] < 0.55

    expected_p = norm.cdf(1.0 / np.sqrt(2.0))
    assert np.isclose(res_df.loc["2", "softmax_p"], expected_p, atol=0.05)


def test_single_candidate_edge_case():
    """A single candidate cell type yields confidence softmax_p == 1.0."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame({"leiden": ["0"] * 10}, index=[f"c{i}" for i in range(10)])
    scores = pd.DataFrame({"TypeA": rng.random(10)}, index=obs.index)
    adata = AnnData(X=np.zeros((10, 1)), obs=obs)

    res_df = annotate_clusters_by_markers(
        adata, cluster_key="leiden", scores=scores, normalize_scores=False
    )
    assert res_df.loc["0", "softmax_p"] == 1.0
    assert res_df.loc["0", "assigned_cell_type"] == "TypeA"


def test_outlier_robustness():
    """Empirical Probability of Superiority assigns TypeA (~0.9) even with 10% extreme -100 outliers."""
    n_majority, n_outliers = 900, 100
    n_cells = n_majority + n_outliers
    obs = pd.DataFrame({"leiden": ["0"] * n_cells}, index=[f"c{i}" for i in range(n_cells)])
    scores = pd.DataFrame(
        {
            "TypeA": np.concatenate([np.full(n_majority, 1.0), np.full(n_outliers, 0.0)]),
            "TypeB": np.concatenate([np.full(n_majority, 0.0), np.full(n_outliers, 100.0)]),
        },
        index=obs.index,
    )
    adata = AnnData(X=np.zeros((n_cells, 2)), obs=obs)

    res_df = annotate_clusters_by_markers(
        adata, cluster_key="leiden", scores=scores, normalize_scores=False
    )
    row = res_df.loc["0"]
    assert row["assigned_cell_type"] == "TypeA"
    assert np.isclose(row["softmax_p"], 0.9, atol=0.01)


def test_annotate_clusters_validation_and_deprecation():
    """annotate_clusters_by_markers warns on deprecated beta and raises ValueError when both scores and markers are None."""
    adata = _mock_marker_adata()
    with pytest.raises(ValueError, match="Provide `cell_type_markers_dict`"):
        annotate_clusters_by_markers(adata, cluster_key="leiden", scores=None)

    # Changed (D08): FutureWarning naming the removal version (was DeprecationWarning).
    with pytest.warns(FutureWarning, match="beta.*deprecated.*0.1.0"):
        annotate_clusters_by_markers(
            adata,
            cluster_key="leiden",
            cell_type_markers_dict={"T": ["CD3E"], "B": ["CD19"]},
            beta=2.0,
            score_method="binned",
        )


def test_marker_missing_report_is_h5ad_safe(tmp_path):
    """marker_missing_report stores a DataFrame safely writable to h5ad even with '/' in cell type names."""
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
    except Exception as e:
        pytest.skip(f"this anndata/pandas pair cannot write h5ad at all: {e}")
    a.write_h5ad(tmp_path / "x.h5ad")


def test_sweep_leiden_keeps_close_resolutions_apart():
    """sweep_leiden_and_annotate keeps close resolutions distinct, returns ARI/annotations, and rejects duplicates."""
    a = _typed_log(n_per_type=50)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    markers = {
        "T0": [f"G{i}" for i in range(15)],
        "T1": [f"G{i}" for i in range(15, 30)],
    }

    resolutions = (0.5, 0.96, 1.04, 2.0)
    out = sweep_leiden_and_annotate(
        a, markers, resolutions=resolutions, score_method="binned", use_raw=False
    )
    for key in ["leiden_0.5", "leiden_0.96", "leiden_1.04", "leiden_2.0"]:
        assert key in a.obs
    assert set(out["cluster_annotations"]) == set(resolutions)
    for table in out["cluster_annotations"].values():
        assert "assigned_cell_type" in table.columns and len(table) >= 1
    assert out["all_pair_ARI"].shape == (4, 4)
    np.testing.assert_allclose(np.diag(out["all_pair_ARI"].to_numpy()), 1.0)
    assert len(out["adjacent_ARI"]) == 3

    with pytest.raises(ValueError, match="Duplicate"):
        sweep_leiden_and_annotate(a, markers, resolutions=(0.5, 0.5), use_raw=False)


def test_plot_celltype_ari_and_cluster_uncertainty():
    """plot_celltype_ari and barplot_cluster_uncertainty return valid matplotlib Figure/Axes with expected content."""
    ari_df = pd.DataFrame(
        [[1.0, 0.85, 0.7], [0.85, 1.0, 0.9], [0.7, 0.9, 1.0]],
        index=[0.5, 1.0, 2.0],
        columns=[0.5, 1.0, 2.0],
    )
    cluster_df = pd.DataFrame(
        {"uncertainty": [0.5, 0.1, 0.8, 0.2]}, index=["c0", "c1", "c2", "c3"]
    )
    try:
        fig_ari, ax_ari = plot_celltype_ari(ari_df)
        assert isinstance(fig_ari, plt.Figure) and isinstance(ax_ari, plt.Axes)
        assert len(ax_ari.images) == 1
        assert [t.get_text() for t in ax_ari.get_xticklabels()] == ["0.5", "1.0", "2.0"]

        fig_unc, ax_unc = barplot_cluster_uncertainty(cluster_df, top_k=2)
        assert isinstance(fig_unc, plt.Figure) and isinstance(ax_unc, plt.Axes)
        assert len(ax_unc.patches) == 2
        widths = [p.get_width() for p in ax_unc.patches]
        np.testing.assert_allclose(widths, [0.1, 0.2])
    finally:
        plt.close("all")


def test_filter_markers_exact_var_name_beats_alias():
    """filter_markers_to_adata prefers exact var_names over ambiguous symbol aliases."""
    a = AnnData(np.zeros((2, 3)))
    a.var_names = ["CD4", "CD8A", "CD8B"]
    a.var["Symbol"] = ["CD4", "CD8A", "CD8A"]
    cleaned, _, repl = filter_markers_to_adata({"T": ["CD8A", "CD4"]}, a, verbose=False)
    assert cleaned["T"] == ["CD8A", "CD4"]
    assert repl == {}

    b = AnnData(np.zeros((2, 4)))
    b.var_names = ["CD4", "CD8A", "TBCE", "TBCE-1"]
    b.var["gene_name"] = ["CD4", "CD8A", "TBCE", "TBCE"]
    cleaned, _, _ = filter_markers_to_adata({"X": ["TBCE"]}, b, verbose=False)
    assert cleaned["X"] == ["TBCE"]


def test_filter_markers_by_moran_fills_from_above_threshold_first():
    """filter_markers_by_moran satisfies min_k using above-threshold genes before triggering fallback."""
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


def test_filter_markers_by_moran_prefers_exact_name():
    """filter_markers_by_moran matches exact gene casing before falling back to case variants."""
    df = pd.DataFrame({"gene": ["Actb", "ACTB", "Gapdh"], "I": [0.9, 0.05, 0.5]})
    kept, _ = filter_markers_by_moran(
        {"x": ["ACTB", "GAPDH"]},
        df,
        threshold=0.1,
        top_k=5,
        min_k=1,
        fallback_to_min_k=False,
        verbose=False,
    )
    assert kept["x"] == ["GAPDH"]


def test_plot_marker_genes_dict_on_embedding_methods():
    """plot_marker_genes_dict_on_embedding renders single/multiple score methods and cleans up temporary obs columns."""
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.uniform(0, 10, (20, 3)))
    adata.var_names = ["G0", "G1", "G2"]
    adata.obsm["X_umap"] = rng.uniform(0, 1, (20, 2))
    markers = {"Type1": ["G0", "G1"]}
    neg_markers = {"Type1": ["G2"]}

    try:
        axes = plot_marker_genes_dict_on_embedding(adata, markers)
        assert len(axes) == 3
        assert "Type1_score_scanpy" not in adata.obs

        axes_binned = plot_marker_genes_dict_on_embedding(
            adata, markers, score_method="binned", use_raw=False
        )
        assert len(axes_binned) == 3

        methods = [
            "scanpy",
            "binned",
            "binned_weighted",
            "net_scanpy",
            "net_binned",
            "net_binned_weighted",
        ]
        axes_multi = plot_marker_genes_dict_on_embedding(
            adata,
            markers,
            negative_marker_genes=neg_markers,
            score_method=methods,
            use_raw=False,
        )
        assert len(axes_multi) == 2 + len(methods)
        for m in methods:
            assert f"Type1_score_{m}" not in adata.obs
    finally:
        plt.close("all")


def test_plot_marker_genes_dict_missing_basis():
    """plot_marker_genes_dict_on_embedding raises ValueError when the requested basis is absent from obsm."""
    rng = np.random.default_rng(0)
    adata = AnnData(X=rng.random((10, 10)))
    with pytest.raises(
        ValueError, match="compute it and add in obsm, or choose from available keys"
    ):
        plot_marker_genes_dict_on_embedding(adata, {"A": ["gene_0"]}, basis="X_pca")


def test_plot_marker_genes_dict_uses_raw_genes_and_scores_without_raw():
    """plot_marker_genes_dict_on_embedding plots raw-only genes when raw exists and falls back to X otherwise."""
    rng = np.random.default_rng(0)
    full = _typed_log(n_per_type=40)
    full.obsm["X_umap"] = rng.normal(size=(full.n_obs, 2))
    full.raw = full.copy()
    sub = full[:, 30:].copy()

    try:
        axes = plot_marker_genes_dict_on_embedding(sub, {"T1": ["G0", "G1", "G2"]})
        assert [ax.get_title() for ax in axes] == ["G0", "G1", "G2", "T1_score_scanpy"]

        no_raw = _typed_log(n_per_type=40)
        no_raw.obsm["X_umap"] = rng.normal(size=(no_raw.n_obs, 2))
        axes = plot_marker_genes_dict_on_embedding(no_raw, {"T1": ["G0", "G1", "G2"]})
        assert "T1_score_scanpy" in [ax.get_title() for ax in axes]
    finally:
        plt.close("all")


def test_plot_marker_genes_dict_restores_existing_score_column():
    """Pre-existing adata.obs columns sharing a score name are restored verbatim after plotting."""
    a = _typed_log(n_per_type=40)
    a.obsm["X_umap"] = np.random.default_rng(0).normal(size=(a.n_obs, 2))
    user_values = np.arange(a.n_obs, dtype=float)
    a.obs["T1_score_scanpy"] = user_values
    a.obs["other"] = 1.0

    try:
        axes = plot_marker_genes_dict_on_embedding(a, {"T1": ["G0", "G1", "G2"]})
        assert "T1_score_scanpy" in [ax.get_title() for ax in axes]
    finally:
        plt.close("all")

    np.testing.assert_array_equal(a.obs["T1_score_scanpy"].to_numpy(), user_values)
    assert list(a.obs.columns) == ["ct", "T1_score_scanpy", "other"]


# =============================================================================
# Triku Feature Selection
# =============================================================================


def test_triku_gene_set():
    """triku_gene_set requires run_triku columns and filters by triku_highly_variable and min_score."""
    a = AnnData(np.zeros((3, 4)), var=pd.DataFrame(index=["G0", "G1", "G2", "G3"]))
    with pytest.raises(RuntimeError, match="Run `run_triku` first"):
        triku_gene_set(a)

    a.var["triku_highly_variable"] = [True, True, False, True]
    a.var["triku_distance"] = [0.2, 0.5, 0.9, 1.2]
    assert triku_gene_set(a) == {"G0", "G1", "G3"}
    assert triku_gene_set(a, min_score=0.5) == {"G1", "G3"}
    assert triku_gene_set(a, min_score=1.0) == {"G3"}


def test_filter_marker_dict_by_triku_strings_and_casing():
    """filter_marker_dict_by_triku accepts string values and returns canonical var_names spelling."""
    a = AnnData(np.zeros((2, 3)))
    a.var_names = ["CD3E", "MS4A1", "LYZ"]
    a.var["triku_highly_variable"] = True
    a.var["triku_distance"] = 1.0
    out = filter_marker_dict_by_triku(a, {"T": "CD3E", "B": ["ms4a1"], "M": ["XYZ"]})
    assert out == {"T": ["CD3E"], "B": ["MS4A1"], "M": []}


def test_run_triku_rebuilds_graph_and_restores_warning_filters(monkeypatch):
    """run_triku builds a fresh PCA neighbor graph and restores global warning filters."""
    seen = {}

    def _fake_triku(adata, n_features, use_raw):
        seen["n_neighbors"] = adata.uns["neighbors"]["params"]["n_neighbors"]
        seen["use_rep"] = adata.uns["neighbors"]["params"].get("use_rep")
        warnings.filterwarnings("ignore")
        adata.var["triku_highly_variable"] = False
        adata.var["triku_distance"] = 0.0

    fake = MagicMock()
    fake.tl.triku = _fake_triku
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
    """Importing single_cell and lazily importing triku never leaves a global 'ignore' warning filter."""
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


# =============================================================================
# Expression Archetypes
# =============================================================================


def test_archetype_recovery_and_invariants():
    """find_expression_archetypes separates distinct modules, is affine-invariant, and aligns archetype signs."""
    n_cells = 100
    n_genes_per_module = 10
    rng = np.random.default_rng(42)

    t = np.linspace(0, 4 * np.pi, n_cells)
    signal_A = np.sin(t)
    genes_A = np.tile(signal_A, (n_genes_per_module, 1))
    names_A = [f"GeneA_{i}" for i in range(n_genes_per_module)]

    signal_B = np.zeros(n_cells)
    signal_B[n_cells // 2 :] = 1.0
    alphas = rng.uniform(0.5, 2.0, size=(n_genes_per_module, 1))
    betas = rng.uniform(-5.0, 5.0, size=(n_genes_per_module, 1))
    genes_B = alphas * signal_B[None, :] + betas
    names_B = [f"GeneB_{i}" for i in range(n_genes_per_module)]

    genes_C = rng.normal(0.0, 1.0, size=(n_genes_per_module, n_cells))
    names_C = [f"GeneC_{i}" for i in range(n_genes_per_module)]

    X = np.vstack([genes_A, genes_B, genes_C]).T
    var_names = names_A + names_B + names_C
    adata = AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"Cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=var_names),
    )

    results = find_expression_archetypes(adata, gene_list=var_names, num_clusters=3, source="X")
    gene_to_cluster = dict(zip(results["gene_list"], results["clusters"]))

    cluster_ids_A = {gene_to_cluster[g] for g in names_A}
    cluster_ids_B = {gene_to_cluster[g] for g in names_B}
    assert len(cluster_ids_A) == 1
    assert len(cluster_ids_B) == 1
    id_A, id_B = next(iter(cluster_ids_A)), next(iter(cluster_ids_B))
    assert id_A != id_B

    corr_A = np.corrcoef(results["archetypes"][id_A - 1], signal_A)[0, 1]
    corr_B = np.corrcoef(results["archetypes"][id_B - 1], signal_B)[0, 1]
    assert corr_A > 0.99
    assert corr_B > 0.99

    idx_A0 = results["gene_list"].index(names_A[0])
    assert results["gene_corrs"][idx_A0] > 0.99


def test_find_expression_archetypes_array_like_and_duplicate_names():
    """find_expression_archetypes accepts pandas Series / NumPy arrays and handles duplicate var_names."""
    rng = np.random.default_rng(0)
    n = 300
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    X = np.c_[[f1 + 0.3 * rng.normal(size=n) for _ in range(4)]].T
    X = np.c_[X, np.c_[[f2 + 0.3 * rng.normal(size=n) for _ in range(4)]].T]
    a = AnnData(X)
    a.var_names = ["g1", "g2", "g3", "g4", "h1", "h2", "h3", "g1"]

    res = find_expression_archetypes(a, pd.Series(["g1", "g2", "h1", "h2"]), 2)
    assert res["gene_list"] == ["g1", "g2", "h1", "h2"]
    res = find_expression_archetypes(a, np.array(["g2", "g3", "h1", "h2"]), 2)
    assert len(res["clusters"]) == 4


def test_find_expression_archetypes_share_one_scale():
    """Every archetype has unit standard deviation so argmax assignment recovers the dominant factor."""
    rng = np.random.default_rng(0)
    n = 500
    f1, f2, f3 = rng.normal(size=(3, n))
    cols = [f1 + 0.5 * rng.normal(size=n) for _ in range(5)]
    cols += [f2 + 0.5 * rng.normal(size=n) for _ in range(5)]
    cols += [f3]
    a = AnnData(np.c_[cols].T)
    a.var_names = [f"g{i}" for i in range(11)]
    res = find_expression_archetypes(a, list(a.var_names), 3)

    np.testing.assert_allclose(res["archetypes"].std(axis=1), 1.0, rtol=1e-6)
    truth = np.argmax(np.c_[f1, f2, f3], axis=1)
    pred = np.argmax(res["archetypes"], axis=0)
    corr = np.corrcoef(np.r_[res["archetypes"], np.c_[f1, f2, f3].T])[:3, 3:]
    mapping = corr.argmax(axis=1)
    assert (mapping[pred] == truth).mean() > 0.85


def test_find_expression_archetypes_validation_and_sparse():
    """find_expression_archetypes validates arguments, caps num_clusters, and matches dense results on sparse X."""
    rng = np.random.default_rng(11)
    n = 80
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    X = np.c_[
        f1 + 0.1 * rng.normal(size=n),
        f1 + 0.1 * rng.normal(size=n),
        f2 + 0.1 * rng.normal(size=n),
        f2 + 0.1 * rng.normal(size=n),
    ]
    genes = ["A1", "A2", "B1", "B2"]
    a_dense = AnnData(X=X, var=pd.DataFrame(index=genes))
    a_sparse = AnnData(X=sp.csr_matrix(X), var=pd.DataFrame(index=genes))

    res_dense = find_expression_archetypes(a_dense, genes, num_clusters=2)
    res_sparse = find_expression_archetypes(a_sparse, genes, num_clusters=2)
    np.testing.assert_array_equal(res_sparse["clusters"], res_dense["clusters"])
    np.testing.assert_allclose(res_sparse["archetypes"], res_dense["archetypes"], atol=1e-6)

    with pytest.warns(UserWarning, match="Setting `num_clusters`"):
        capped = find_expression_archetypes(a_dense, genes, num_clusters=10)
    assert capped["archetypes"].shape == (4, n)

    with pytest.raises(ValueError, match="cannot be empty"):
        find_expression_archetypes(a_dense, [], num_clusters=2)
    with pytest.raises(ValueError, match="at least 1"):
        find_expression_archetypes(a_dense, genes, num_clusters=0)
    with pytest.warns(UserWarning, match="not found"):
        with pytest.raises(ValueError, match="No requested genes were found"):
            find_expression_archetypes(a_dense, ["NOPE1", "NOPE2"], num_clusters=2)
    with pytest.raises(ValueError, match="Cannot perform clustering"):
        find_expression_archetypes(a_dense, ["A1"], num_clusters=1)


def test_plot_archetype_summary_colormaps_and_validation():
    """plot_archetype_summary renders real figures with default/custom colormaps and validates inputs."""
    a = _typed_log(n_per_type=30)
    a.obsm["X_umap"] = np.random.default_rng(0).normal(size=(a.n_obs, 2))
    res = find_expression_archetypes(a, [f"G{i}" for i in range(20)], 2)

    try:
        before = set(plt.get_fignums())
        plot_archetype_summary(a, res, archetype_id=1, k=2, show=False)
        new_figs = [plt.figure(n) for n in plt.get_fignums() if n not in before]
        assert len(new_figs) == 2

        score_axes = [ax for ax in new_figs[0].axes if ax.get_title() == "Archetype 1 Score"]
        assert len(score_axes) == 1
        assert score_axes[0].collections[0].cmap.name == "PiYG"

        gene_set = set(res["gene_list"])
        gene_axes = [ax for ax in new_figs[1].axes if ax.get_title() in gene_set]
        assert len(gene_axes) == 2
        assert all(ax.collections[0].cmap.name == "Purples" for ax in gene_axes)

        before2 = set(plt.get_fignums())
        plot_archetype_summary(
            a, res, archetype_id=1, k=2, cmap="Reds", archetype_cmap="Blues", show=False
        )
        custom_figs = [plt.figure(n) for n in plt.get_fignums() if n not in before2]
        assert len(custom_figs) == 2
        score_axes2 = [ax for ax in custom_figs[0].axes if ax.get_title() == "Archetype 1 Score"]
        assert score_axes2[0].collections[0].cmap.name == "Blues"
        gene_axes2 = [ax for ax in custom_figs[1].axes if ax.get_title() in gene_set]
        assert all(ax.collections[0].cmap.name == "Reds" for ax in gene_axes2)
    finally:
        plt.close("all")

    with pytest.raises(ValueError, match="missing required keys"):
        plot_archetype_summary(a, {}, archetype_id=1, show=False)
    with pytest.raises(ValueError, match="must be between 1 and"):
        plot_archetype_summary(a, res, archetype_id=0, show=False)


def test_archetype_plots_keep_existing_obs_columns():
    """plot_archetype_summary and plot_archetype_assignments restore pre-existing obs columns and drop temps."""
    a = _typed_log(n_per_type=30)
    a.obsm["X_umap"] = np.random.default_rng(0).normal(size=(a.n_obs, 2))
    res = find_expression_archetypes(a, [f"G{i}" for i in range(20)], 2)
    a.obs["archetype_1_score"] = 123.0

    try:
        plot_archetype_summary(a, res, 1, k=2, show=False)
        plot_archetype_assignments(a, res, show=False)
    finally:
        plt.close("all")
    assert (a.obs["archetype_1_score"] == 123.0).all()
    assert "archetype_2_score" not in a.obs


def test_plot_archetype_hierarchy_returns_figure():
    """plot_archetype_hierarchy returns a matplotlib Figure containing the dendrogram."""
    a = _typed_log(n_per_type=30)
    res = find_expression_archetypes(a, [f"G{i}" for i in range(20)], 2)
    before = len(plt.get_fignums())
    try:
        fig = plot_archetype_hierarchy(res)
        assert isinstance(fig, plt.Figure)
        assert len(plt.get_fignums()) == before + 1
    finally:
        plt.close("all")


# =============================================================================
# PaCMAP Embeddings & Model Persistence
# =============================================================================


def test_pacmap_heuristic_n_neighbors():
    """pacmap_heuristic_n_neighbors returns 10 up to 10k cells and scales logarithmically above."""
    assert pacmap_heuristic_n_neighbors(100) == 10
    assert pacmap_heuristic_n_neighbors(10000) == 10
    assert pacmap_heuristic_n_neighbors(50000) == 20
    assert pacmap_heuristic_n_neighbors(100000) == 25
    assert pacmap_heuristic_n_neighbors(1000000) == 40


def test_save_and_load_pacmap_model(tmp_path, monkeypatch):
    """save_pacmap_model and load_pacmap_model round-trip objects via joblib and raise ImportError if missing."""
    pytest.importorskip("joblib")
    payload = {"embedding_": np.array([[1.0, 2.0], [3.0, 4.0]]), "n_components": 2}
    model_path = tmp_path / "pacmap_model.joblib"

    save_pacmap_model(payload, str(model_path))
    assert model_path.exists()
    loaded = load_pacmap_model(str(model_path))
    assert loaded["n_components"] == 2
    np.testing.assert_array_equal(loaded["embedding_"], payload["embedding_"])

    monkeypatch.setitem(sys.modules, "joblib", None)
    with pytest.raises(ImportError, match="joblib is required"):
        save_pacmap_model(payload, str(model_path))
    with pytest.raises(ImportError, match="joblib is required"):
        load_pacmap_model(str(model_path))


def test_paga_init_positions_match_scanpy_reference():
    """_paga_init_positions matches scanpy's pre-1.13 helper bit-for-bit across seeds."""
    from scanpy.tools._utils import get_init_pos_from_paga

    if "random_state" not in inspect.signature(get_init_pos_from_paga).parameters:
        pytest.skip("scanpy >= 1.13: reference helper has a different RNG contract")
    adata = _paga_adata_with_layout()
    for seed in (0, 1, 1234):
        ref = get_init_pos_from_paga(adata, random_state=seed)
        np.testing.assert_array_equal(_paga_init_positions(adata, random_state=seed), ref)


def test_paga_init_positions_analytic_box_and_mean():
    """_paga_init_positions places linked groups uniformly in p_g +- |d|/2 and isolated groups at p_g."""
    n = 4000
    adata = AnnData(np.zeros((3 * n, 1)))
    adata.obs["g"] = pd.Categorical(np.repeat(["a", "b", "c"], n))
    pos = np.array([[0.0, 0.0], [4.0, -2.0], [10.0, 10.0]])
    C = sp.csr_matrix(np.array([[0.0, 0.7, 0.0], [0.7, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    adata.uns["paga"] = {"pos": pos, "connectivities": C, "groups": "g"}

    init = _paga_init_positions(adata, random_state=7)
    for g, h in ((0, 1), (1, 0)):
        x = init[g * n : (g + 1) * n]
        d = pos[g] - pos[h]
        lo = pos[g] - 0.5 * np.abs(d)
        hi = pos[g] + 0.5 * np.abs(d)
        assert np.all((x >= lo - 1e-12) & (x <= hi + 1e-12))
        np.testing.assert_allclose(x.mean(axis=0), pos[g], atol=0.03 * np.abs(d).max())
    np.testing.assert_array_equal(init[2 * n :], np.tile(pos[2], (n, 1)))
    np.testing.assert_array_equal(init, _paga_init_positions(adata, random_state=7))


@requires_pacmap
@pytest.mark.parametrize("n_vars", [50, 101])
def test_tl_pacmap_default_init_is_pca(n_vars):
    """With init=None, tl_pacmap warns and matches explicit init='pca' on both sides of 100 features."""
    a_default = _pacmap_adata(n_vars)
    with pytest.warns(UserWarning, match=r"init not given, using init='pca'") as rec:
        tl_pacmap(a_default, n_neighbors=5, use_rep="X")
    msg = " ".join(str(w.message) for w in rec)
    assert "init='paga'" in msg and "init='random'" in msg
    assert not any("Switching initialization" in str(w.message) for w in rec)

    a_pca = _pacmap_adata(n_vars)
    tl_pacmap(a_pca, n_neighbors=5, use_rep="X", init="pca")
    np.testing.assert_array_equal(a_default.obsm["X_pacmap"], a_pca.obsm["X_pacmap"])


@requires_pacmap
def test_tl_pacmap_explicit_init_does_not_warn():
    """Explicit init='pca' and init='random' emit no default-init warning."""
    for init in ("pca", "random"):
        adata = _pacmap_adata(50)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tl_pacmap(adata, n_neighbors=5, use_rep="X", init=init)
        assert not any("init not given" in str(x.message) for x in w), init
        assert adata.obsm["X_pacmap"].shape == (60, 2)


@requires_pacmap
def test_tl_pacmap_init_same_as_use_rep_uses_pca():
    """Passing init == use_rep warns and routes to PaCMAP's 'pca' initialization."""
    base = _pacmap_adata(200)
    sc.tl.pca(base, n_comps=20)

    a_same = base.copy()
    with pytest.warns(UserWarning, match="same representation as use_rep"):
        tl_pacmap(a_same, n_neighbors=5, use_rep="X_pca", init="X_pca")

    a_pca = base.copy()
    tl_pacmap(a_pca, n_neighbors=5, use_rep="X_pca", init="pca")
    np.testing.assert_array_equal(a_same.obsm["X_pacmap"], a_pca.obsm["X_pacmap"])


@requires_pacmap
def test_tl_pacmap_paga_init_leaves_global_rng_untouched():
    """init='paga' runs without default-init warning and leaves the caller's global NumPy RNG untouched."""
    adata = _paga_adata_with_layout()
    saved_state = np.random.get_state()
    try:
        np.random.seed(2024)
        np.random.random(5)
        before = np.random.get_state()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X", init="paga")
        after = np.random.get_state()
    finally:
        np.random.set_state(saved_state)

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert not any("init not given" in str(x.message) for x in w)
    assert adata.obsm["X_pacmap"].shape == (adata.n_obs, 2)


@requires_pacmap
def test_tl_pacmap_paga_init_independent_of_scanpy_private_helper(monkeypatch):
    """init='paga' works even when scanpy's private get_init_pos_from_paga changes signature."""
    import scanpy.tools._utils as sc_tool_utils

    def _scanpy_113_signature(adata, *, rng, adjacency):
        raise AssertionError("tl_pacmap must not depend on this private helper")

    monkeypatch.setattr(sc_tool_utils, "get_init_pos_from_paga", _scanpy_113_signature)
    adata = _paga_adata_with_layout()
    tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X", init="paga")
    assert adata.obsm["X_pacmap"].shape == (adata.n_obs, 2)
    assert np.isfinite(adata.obsm["X_pacmap"]).all()


@requires_pacmap
def test_tl_pacmap_paga_init_without_plotting_paga():
    """init='paga' computes PAGA positions automatically when sc.pl.paga has not been called yet."""
    a = _typed_log(n_per_type=50)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    try:
        sc.tl.paga(a, groups="ct")
    except ValueError:
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


@requires_pacmap
def test_tl_pacmap_paga_init_with_disconnected_groups():
    """init='paga' falls back to 'pca' with a warning when the PAGA graph has no inter-group edges."""
    a = _typed_log(n_per_type=50)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    Z = sp.csr_matrix((4, 4))
    a.uns["paga"] = {"connectivities": Z, "connectivities_tree": Z, "groups": "ct"}
    with pytest.warns(UserWarning, match="no edges between groups") as rec:
        tl_pacmap(a, init="paga")
    fallback = [str(w.message) for w in rec if "init not given" in str(w.message)]
    assert fallback and "init='pca'" in fallback[0] and "init='paga'" not in fallback[0]
    assert a.obsm["X_pacmap"].shape == (a.n_obs, 2)
    assert np.isfinite(a.obsm["X_pacmap"]).all()


@requires_pacmap
def test_pacmap_key_naming():
    """tl_pacmap stores 2D embeddings in X_pacmap and >=3D embeddings in X_pacmap_<n_components>."""
    adata = _pacmap_adata(50, n_obs=50, seed=1)

    tl_pacmap(adata, n_neighbors=5, n_components=2, use_rep="X", init="pca")
    assert "X_pacmap" in adata.obsm
    assert "X_pacmap_2" not in adata.obsm

    tl_pacmap(adata, n_neighbors=5, n_components=3, use_rep="X", init="pca")
    assert adata.obsm["X_pacmap_3"].shape == (50, 3)

    tl_pacmap(adata, n_neighbors=5, n_components=4, use_rep="X", init="pca")
    assert adata.obsm["X_pacmap_4"].shape == (50, 4)
    assert adata.obsm["X_pacmap"].shape == (50, 2)


@requires_pacmap
def test_tl_pacmap_applies_neighbor_heuristic(monkeypatch):
    """When n_neighbors is None, tl_pacmap passes pacmap_heuristic_n_neighbors(n_obs) to PaCMAP."""
    captured = {}

    class _FakePaCMAP:
        def __init__(self, **kw):
            captured.update(kw)
            self.n_components = kw.get("n_components", 2)

        def fit_transform(self, X, init=None):
            return np.zeros((X.shape[0], self.n_components))

    monkeypatch.setattr(pacmap, "PaCMAP", _FakePaCMAP)
    monkeypatch.setattr(scmod, "pacmap_heuristic_n_neighbors", lambda n: 17)
    a = _typed_log(n_per_type=30)
    sc.pp.pca(a, n_comps=10)
    tl_pacmap(a, init="pca")
    assert captured["n_neighbors"] == 17


@requires_pacmap
def test_tl_pacmap_padded_init_is_reproducible():
    """Padding a lower-dimensional obsm init array uses random_state rather than global NumPy RNG."""
    a = _typed_log(n_per_type=40)
    sc.pp.pca(a, n_comps=10)
    a.obsm["X_init1d"] = a.obsm["X_pca"][:, :1].copy()

    saved_state = np.random.get_state()
    try:
        tl_pacmap(a, init="X_init1d", random_state=3)
        first = a.obsm["X_pacmap"].copy()
        np.random.seed(999)
        tl_pacmap(a, init="X_init1d", random_state=3)
    finally:
        np.random.set_state(saved_state)
    np.testing.assert_allclose(a.obsm["X_pacmap"], first)


@requires_pacmap
def test_tl_pacmap_precomputed_and_validation(monkeypatch):
    """tl_pacmap routes distance='precomputed' to init='random' and rejects unknown init or missing PAGA."""
    a = _pacmap_adata(20, n_obs=30, seed=5)
    with pytest.raises(ValueError, match="Unknown initialization method"):
        tl_pacmap(a, n_neighbors=5, use_rep="X", init="nonexistent_init")
    with pytest.raises(ValueError, match="sc.tl.paga"):
        tl_pacmap(a, n_neighbors=5, use_rep="X", init="paga")

    seen = {}

    class _FakePaCMAP:
        def __init__(self, **kw):
            seen["kw"] = kw
            self.n_components = kw.get("n_components", 2)

        def fit_transform(self, X, init=None):
            seen["init"] = init
            return np.zeros((X.shape[0], self.n_components))

    monkeypatch.setattr(pacmap, "PaCMAP", _FakePaCMAP)
    with pytest.warns(UserWarning, match="using init='random'") as rec:
        tl_pacmap(a, n_neighbors=5, use_rep="X", distance="precomputed")
    assert seen["init"] == "random"
    assert seen["kw"]["distance"] == "precomputed"
    msg = " ".join(str(w.message) for w in rec)
    assert "init='random' to check" not in msg


# =============================================================================
# Adaptive kkNN Neighbors, Classifier & Ingest
# =============================================================================


def test_compute_kknn_neighbors():
    """compute_kknn_neighbors adapts k within [min_neighbors, max_neighbors] and validates obsm keys."""
    rng = np.random.default_rng(42)
    n_ref, n_query, n_pcs = 150, 30, 5
    adata_ref = AnnData(np.zeros((n_ref, 10)))
    adata_ref.obsm["X_pca"] = rng.normal(size=(n_ref, n_pcs))
    adata_query = AnnData(np.zeros((n_query, 10)))
    adata_query.obsm["X_pca"] = rng.normal(size=(n_query, n_pcs))

    dists, idxs = compute_kknn_neighbors(
        adata_query, adata_ref, n_neighbors=10, min_neighbors=3, max_neighbors=20
    )
    assert len(dists) == n_query and len(idxs) == n_query
    lengths = [len(d) for d in dists]
    assert min(lengths) >= 3 and max(lengths) <= 20
    assert len(set(lengths)) > 1  # neighbor count actually adapts to curvature
    assert all(np.all(np.diff(d) >= 0) for d in dists)

    with pytest.raises(ValueError, match="Query dataset must have"):
        compute_kknn_neighbors(adata_query, adata_ref, use_rep="X_pca", query_use_rep="X_missing")
    with pytest.raises(ValueError, match="Reference dataset must have"):
        compute_kknn_neighbors(adata_query, adata_ref, use_rep="X_missing", query_use_rep="X_pca")


def test_compute_kknn_neighbors_small_reference():
    """compute_kknn_neighbors clamps max_neighbors when the reference has fewer cells than defaults."""
    rng = np.random.default_rng(0)
    ref = AnnData(np.zeros((8, 1)))
    ref.obsm["X_pca"] = rng.normal(size=(8, 3))
    _, idxs = compute_kknn_neighbors(ref, ref)
    assert all(1 <= len(i) <= 8 for i in idxs)


def test_kknn_curvature_cache_is_keyed_by_representation():
    """Curvature bounds cache separates representations and invalidates when an embedding is overwritten."""
    rng = np.random.default_rng(0)
    a = AnnData(np.zeros((400, 1)))
    a.obsm["X_flat"] = rng.normal(size=(400, 5))
    t = rng.uniform(0, 6 * np.pi, 400)
    a.obsm["X_curved"] = np.c_[np.cos(t), np.sin(t), 0.01 * rng.normal(size=400)]

    fresh = a.copy()
    _, idx_fresh = compute_kknn_neighbors(fresh, fresh, use_rep="X_curved")

    compute_kknn_neighbors(a, a, use_rep="X_flat")
    _, idx = compute_kknn_neighbors(a, a, use_rep="X_curved")
    assert [len(i) for i in idx] == [len(i) for i in idx_fresh]

    a.obsm["X_flat"] = a.obsm["X_curved"].copy()
    _, idx_again = compute_kknn_neighbors(a, a, use_rep="X_flat")
    assert [len(i) for i in idx_again] == [len(i) for i in idx_fresh]


def test_kknn_classifier_api_and_mask():
    """kknn_classifier honors inplace=False, numpy/Series masks, and validates obs_key and mask length."""
    rng = np.random.default_rng(123)
    emb = rng.normal(size=(100, 2))
    emb[:50, 0] += 100.0
    emb[50:, 0] -= 100.0
    labels = np.array(["A"] * 50 + ["B"] * 50)
    labels[0] = "B"  # outlier in cluster A
    labels[99] = "A"  # outlier in cluster B

    adata = AnnData(np.zeros((100, 2)))
    adata.obsm["X_pacmap"] = emb
    adata.obs["celltype"] = pd.Categorical(labels)

    mask = np.zeros(100, dtype=bool)
    mask[0] = True

    res = kknn_classifier(
        adata,
        obs_key="celltype",
        use_rep="X_pacmap",
        n_neighbors=5,
        max_neighbors=10,
        mask=pd.Series(mask),
        inplace=False,
    )
    assert "celltype_kknn" not in adata.obs
    assert len(res) == 100
    assert res[0] == "A"  # masked-in outlier corrected
    assert res[99] == "A"  # masked-out outlier preserved

    kknn_classifier(
        adata,
        obs_key="celltype",
        use_rep="X_pacmap",
        n_neighbors=5,
        max_neighbors=10,
        mask=mask,
        inplace=True,
    )
    assert "celltype_kknn" in adata.obs
    assert adata.obs["celltype_kknn"].iloc[0] == "A"
    assert adata.obs["celltype_kknn"].iloc[99] == "A"

    with pytest.raises(KeyError, match="not found in adata.obs"):
        kknn_classifier(adata, obs_key="missing", use_rep="X_pacmap")
    with pytest.raises(ValueError, match="Mask length"):
        kknn_classifier(
            adata,
            obs_key="celltype",
            use_rep="X_pacmap",
            n_neighbors=5,
            max_neighbors=10,
            mask=np.ones(10, dtype=bool),
        )


def _kknn_vote_reference(labels, pruned_distances, pruned_indices, is_categorical, mask):
    """Per-cell loop reference oracle for kknn_classifier before vectorization."""
    smoothed = []
    for i in range(len(pruned_indices)):
        if mask is not None and not mask[i]:
            smoothed.append(labels[i])
            continue
        idx = pruned_indices[i]
        dist = pruned_distances[i]
        neighbors_labels = labels[idx]
        weights = _kknn_distance_weights(dist)
        if is_categorical:
            vote_tally = {}
            for lbl, w in zip(neighbors_labels, weights):
                if pd.isna(lbl):
                    continue
                vote_tally[lbl] = vote_tally.get(lbl, 0.0) + w
            if vote_tally:
                smoothed.append(max(vote_tally.items(), key=lambda x: x[1])[0])
            else:
                smoothed.append(np.nan)
        else:
            valid_mask = ~pd.isna(neighbors_labels)
            if np.any(valid_mask):
                valid_labels = neighbors_labels[valid_mask]
                valid_weights = weights[valid_mask]
                sum_weights = np.sum(valid_weights)
                if sum_weights > 0:
                    valid_weights /= sum_weights
                    smoothed.append(np.sum(valid_labels * valid_weights))
                else:
                    smoothed.append(np.nan)
            else:
                smoothed.append(np.nan)
    return smoothed


def _random_kknn_neighbourhoods(rng, n, n_ref, sizes, dtype=np.float64, ties=False):
    """Generate variable-size kkNN neighborhoods with coincident cells and optional exact weight ties."""
    dists, idxs = [], []
    for _ in range(n):
        k = int(rng.choice(sizes))
        if ties:
            d = rng.integers(0, 3, k).astype(dtype)
        else:
            d = rng.exponential(1.0, k).astype(dtype)
            d[rng.random(k) < 0.05] = 0.0
        idxs.append(rng.integers(0, n_ref, k))
        dists.append(np.sort(d))
    return dists, idxs


def _assert_same_smoothing(new, ref, is_categorical):
    """Assert exact element-wise equality between vectorized and reference kkNN smoothing."""
    if not is_categorical:
        ref = [np.nan if v is pd.NA else v for v in ref]
    new = np.asarray(new, dtype=object if is_categorical else float)
    ref = np.asarray(ref, dtype=object if is_categorical else float)
    assert new.shape == ref.shape
    if is_categorical:
        nan_new = pd.isna(new)
        np.testing.assert_array_equal(nan_new, pd.isna(ref))
        assert all(a == b for a, b in zip(new[~nan_new], ref[~nan_new]))
    else:
        np.testing.assert_array_equal(new, ref)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize(
    "kind", ["categorical", "object", "bool", "float", "float_nan", "float32", "int", "Int64"]
)
def test_kknn_smooth_values_matches_per_cell_loop(kind, ties, dtype):
    """Vectorized _kknn_smooth_values is bit-identical to the per-cell loop across dtypes, NaNs, and vote ties."""
    seed = zlib.crc32(f"{kind}:{ties}:{np.dtype(dtype).name}".encode())
    rng = np.random.default_rng(seed)
    n_ref = 300
    cats = np.array(["T", "B", "NK", "Mono"], dtype=object)
    raw = rng.integers(0, len(cats), n_ref)
    if kind == "categorical":
        labels = pd.Categorical(cats[raw], categories=["B", "Mono", "NK", "T"])
        labels[rng.random(n_ref) < 0.1] = np.nan
    elif kind == "object":
        labels = cats[raw].copy()
        labels[rng.random(n_ref) < 0.1] = None
    elif kind == "bool":
        labels = raw % 2 == 0
    elif kind in ("float", "float_nan", "float32"):
        labels = rng.normal(size=n_ref)
        if kind == "float_nan":
            labels[rng.random(n_ref) < 0.1] = np.nan
        if kind == "float32":
            labels = labels.astype(np.float32)
    elif kind == "int":
        labels = rng.integers(-50, 50, n_ref)
    else:
        labels = pd.array(rng.integers(0, 9, n_ref), dtype="Int64")
        labels[rng.random(n_ref) < 0.1] = pd.NA
    is_categorical = kind in ("categorical", "object", "bool")

    dists, idxs = _random_kknn_neighbourhoods(
        rng, n_ref, n_ref, sizes=[1, 2, 4, 7, 12], dtype=dtype, ties=ties
    )
    for mask in (None, rng.random(n_ref) < 0.5):
        new = _kknn_smooth_values(labels, dists, idxs, is_categorical, mask)
        ref = _kknn_vote_reference(labels, dists, idxs, is_categorical, mask)
        _assert_same_smoothing(new, ref, is_categorical)


def test_kknn_classifier_vectorised_end_to_end_matches_loop():
    """kknn_classifier matches the per-cell loop end-to-end on real kkNN neighborhoods and improves accuracy."""
    rng = np.random.default_rng(0)
    n = 2000
    centre = rng.integers(0, 8, n)
    emb = rng.normal(size=(n, 2)) + np.c_[centre * 6.0, np.zeros(n)]
    lab = np.array(list("ABCDEFGH"), dtype=object)[centre]
    flip = rng.random(n) < 0.1
    lab[flip] = rng.choice(list("ABCDEFGH"), flip.sum())
    adata = AnnData(np.zeros((n, 1)))
    adata.obsm["X_pacmap"] = emb
    adata.obs["ct"] = pd.Categorical(lab)
    adata.obs["score"] = centre + rng.normal(scale=0.3, size=n)
    mask = rng.random(n) < 0.7

    kw = dict(use_rep="X_pacmap", n_neighbors=10, max_neighbors=20, min_neighbors=5)
    dists, idxs = compute_kknn_neighbors(adata, adata, query_use_rep="X_pacmap", **kw)

    for key, is_cat in (("ct", True), ("score", False)):
        new = kknn_classifier(adata, key, inplace=False, mask=mask, **kw)
        ref = _kknn_vote_reference(adata.obs[key].values, dists, idxs, is_cat, mask)
        _assert_same_smoothing(new, ref, is_cat)
        kept = np.asarray(new, dtype=object)[~mask]
        assert all(a == b for a, b in zip(kept, adata.obs[key].values[~mask]))

    smoothed = np.asarray(kknn_classifier(adata, "ct", inplace=False, **kw), dtype=object)
    truth = np.array(list("ABCDEFGH"), dtype=object)[centre]
    assert np.mean(smoothed == truth) > np.mean(lab == truth)


def test_kknn_classifier_votes_on_booleans():
    """Boolean columns are smoothed via majority vote (returning bool) rather than averaged into floats."""
    rng = np.random.default_rng(0)
    a = AnnData(np.zeros((200, 1)))
    a.obsm["X_emb"] = rng.normal(size=(200, 2))
    a.obs["is_doublet"] = rng.random(200) < 0.05
    out = kknn_classifier(a, "is_doublet", use_rep="X_emb", inplace=False)
    assert out.dtype == bool


@pytest.mark.parametrize("recompute_ref_PCA", [False, True])
@pytest.mark.parametrize("save_ref_PCA_key", [None, "X_pca_projected"])
def test_kknn_ingest(recompute_ref_PCA, save_ref_PCA_key):
    """kknn_ingest maps labels/embeddings, aligns reference metadata, and manages PCA projection keys."""
    rng = np.random.default_rng(42)
    n_ref, n_query, n_genes = 100, 20, 20

    adata_ref = AnnData(rng.exponential(1.0, size=(n_ref, n_genes)).astype(np.float32))
    sc.tl.pca(adata_ref)
    adata_ref.obsm["X_umap"] = rng.normal(size=(n_ref, 2))
    adata_ref.obs["cell_type"] = pd.Categorical(rng.choice(["A", "B"], size=n_ref))

    adata_query = AnnData(rng.exponential(1.0, size=(n_query, n_genes)).astype(np.float32))

    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["cell_type"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        n_neighbors=10,
        recompute_ref_PCA=recompute_ref_PCA,
        save_ref_PCA_key=save_ref_PCA_key,
    )

    assert not any(k.startswith("__temp_") for k in adata_query.obsm)
    assert not any(k.startswith("__temp_") for k in adata_ref.obsm)

    if save_ref_PCA_key is not None:
        assert save_ref_PCA_key in adata_query.obsm
        assert adata_query.obsm[save_ref_PCA_key].shape[0] == n_query
        if recompute_ref_PCA:
            assert save_ref_PCA_key in adata_ref.obsm
            assert adata_ref.obsm[save_ref_PCA_key].shape == (
                n_ref,
                adata_query.obsm[save_ref_PCA_key].shape[1],
            )
    else:
        assert "X_pca_projected" not in adata_query.obsm

    assert adata_query.obsm["X_umap_kknn"].shape == (n_query, 2)
    assert "X_umap_kknn" in adata_ref.obsm
    assert isinstance(adata_query.obs["cell_type_kknn"].dtype, pd.CategoricalDtype)
    assert set(adata_query.obs["cell_type_kknn"]) <= {"A", "B"}
    conf = adata_query.obs["mapping_confidence_cell_type_kknn"].to_numpy()
    assert np.all((conf >= 0.5) & (conf <= 1.0))
    k_vals = adata_query.obs["kknn_k"].to_numpy()
    assert np.all((k_vals >= 3) & (k_vals <= n_ref))
    assert adata_ref.obs["kknn_k"].isna().all()
    assert adata_ref.obs["mapping_confidence_cell_type_kknn"].isna().all()
    assert (adata_ref.obs["cell_type_kknn"] == adata_ref.obs["cell_type"]).all()


def test_kknn_ingest_validation():
    """kknn_ingest rejects mismatched var_names, missing reference keys, and missing stored PCA."""
    rng = np.random.default_rng(0)
    ref = AnnData(rng.normal(size=(30, 10)), var=pd.DataFrame(index=[f"G{i}" for i in range(10)]))
    ref.obs["ct"] = pd.Categorical(["A", "B"] * 15)
    query = AnnData(rng.normal(size=(5, 10)), var=pd.DataFrame(index=[f"H{i}" for i in range(10)]))

    with pytest.raises(KeyError, match="not found in adata_ref.obs"):
        kknn_ingest(query, ref, obs_keys=["missing_obs"])
    with pytest.raises(KeyError, match="not found in adata_ref.obsm"):
        kknn_ingest(query, ref, obsm_keys=["missing_emb"])
    with pytest.raises(ValueError, match="var_names are not identical"):
        kknn_ingest(query, ref, obs_keys=["ct"], use_rep="X_pca")

    query_aligned = AnnData(rng.normal(size=(5, 10)), var=ref.var.copy())
    with pytest.raises(ValueError, match="must have PCA computed"):
        kknn_ingest(
            query_aligned, ref, obs_keys=["ct"], use_rep="X_pca", recompute_ref_PCA=False
        )


def test_kknn_ingest_lle_manifold_and_temp_key_cleanup():
    """barycenter='lle' accurately reconstructs an affine 2D manifold in PCA space and removes temp keys."""
    rng = np.random.default_rng(7)
    n_query, n_genes = 30, 10

    # Embed a regular 15x15 grid on [-1, 1]^2 into 10D gene space via an orthonormal basis.
    # A regular grid guarantees every interior query point lies inside the convex hull of its
    # 4 nearest neighbors (required by NNLS LLE's non-negative weights even after kkNN pruning).
    Q, _ = np.linalg.qr(rng.normal(size=(n_genes, 2)))
    B = Q.T  # (2, 10)
    gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, 15), np.linspace(-1.0, 1.0, 15))
    Z_ref = np.column_stack([gx.ravel(), gy.ravel()])
    n_ref = len(Z_ref)
    X_ref = Z_ref @ B + 5.0 + 1e-4 * rng.normal(size=(n_ref, n_genes))

    A_map = np.array([[2.0, -0.5], [0.3, 1.5]])
    shift = np.array([3.0, -1.0])
    adata_ref = AnnData(X=X_ref)
    adata_ref.obs["label"] = pd.Categorical(np.where(Z_ref[:, 0] >= 0, "Pos", "Neg"))
    adata_ref.obsm["X_umap"] = Z_ref @ A_map + shift
    sc.tl.pca(adata_ref, n_comps=2, svd_solver="full")

    # Interior query points lie strictly inside the convex hull of their nearest grid neighbors
    Z_query = rng.uniform(-0.6, 0.6, size=(n_query, 2))
    X_query = Z_query @ B + 5.0
    expected_umap = Z_query @ A_map + shift
    adata_query = AnnData(X=X_query)

    kknn_ingest(
        adata_query,
        adata_ref,
        obs_keys=["label"],
        obsm_keys=["X_umap"],
        use_rep="X_pca",
        barycenter="lle",
        n_neighbors=10,
        min_neighbors=5,
        max_neighbors=12,
        recompute_ref_PCA=False,
        save_ref_PCA_key=None,
    )

    assert not any(k.startswith("__temp_ingest_") for k in adata_query.obsm)
    np.testing.assert_allclose(adata_query.obsm["X_umap_kknn"], expected_umap, atol=0.01)


@pytest.mark.parametrize("max_value", [None, 2.0])
def test_kknn_ingest_projects_scaled_reference_exactly(max_value):
    """Query projection reproduces reference PCA coordinates for both unclipped and clipped scaling."""
    ref = _scaled_reference(max_value)
    query = AnnData(ref.raw.X.copy(), obs=ref.obs[["ct"]].copy(), var=ref.var[[]].copy())

    kknn_ingest(
        query, ref, obs_keys="ct", recompute_ref_PCA=False, save_ref_PCA_key="X_pca_proj"
    )
    np.testing.assert_allclose(query.obsm["X_pca_proj"], ref.obsm["X_pca"], atol=2e-3)
    assert (query.obs["ct_kknn"].astype(str) == query.obs["ct"].astype(str)).mean() > 0.97


@pytest.mark.parametrize("recompute", [False, True])
def test_kknn_ingest_query_of_one_type_is_not_recentred(recompute):
    """A homogeneous single-type query is centered with reference means, not its own mean."""
    ref = _typed_log(n_per_type=200, seed=0)
    sc.pp.highly_variable_genes(ref, n_top_genes=100, flavor="seurat")
    sc.tl.pca(ref, n_comps=20)
    del ref.layers["counts"]

    query = _typed_log(n_per_type=200, seed=1)
    query = query[query.obs["ct"] == "T0"].copy()
    del query.layers["counts"]

    kknn_ingest(query, ref, obs_keys="ct", recompute_ref_PCA=recompute)
    assert (query.obs["ct_kknn"].astype(str) == "T0").mean() > 0.95


def test_kknn_ingest_recompute_saves_matching_reference_basis():
    """save_ref_PCA_key rejects existing reference keys on recompute and otherwise aligns ref/query bases."""
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
    """A single coincident cell (dist=0) does not overwhelm the neighborhood vote thanks to median bandwidth."""
    rng = np.random.default_rng(0)
    ref = AnnData(np.zeros((501, 1)))
    ref.obsm["X_emb"] = np.r_[rng.normal(size=(500, 2)), [[0.0, 0.0]]]
    ref.obs["lbl"] = pd.Categorical(["A"] * 500 + ["B"])
    query = AnnData(np.zeros((1, 1)))
    query.obsm["X_emb"] = np.array([[0.0, 0.0]])

    kknn_ingest(query, ref, obs_keys="lbl", use_rep="X_emb")
    assert query.obs["lbl_kknn"].iloc[0] == "A"


def test_kknn_ingest_integer_query_matrix():
    """kknn_ingest casts integer query matrices cleanly during PCA projection."""
    ref = _typed_counts(n_per_type=60)
    sc.pp.highly_variable_genes(ref, n_top_genes=100, flavor="seurat_v3")
    sc.tl.pca(ref, n_comps=10)
    query = ref[:50].copy()
    query.X = query.X.astype(np.int64)
    kknn_ingest(query, ref, obs_keys="ct", recompute_ref_PCA=False)
    assert (query.obs["ct_kknn"].astype(str) == query.obs["ct"].astype(str)).mean() > 0.9


def test_kknn_ingest_transfers_numeric_obs_as_weighted_mean():
    """Numeric obs columns are transferred as weighted means with mapping_std and NaN confidence."""
    rng = np.random.default_rng(0)
    t_ref = rng.uniform(0, 1, 2000)
    ref = AnnData(np.zeros((2000, 1)))
    ref.obsm["X_emb"] = np.c_[np.cos(3 * t_ref), np.sin(3 * t_ref)] + 0.02 * rng.normal(
        size=(2000, 2)
    )
    ref.obs["pt"] = t_ref + 0.05 * rng.normal(size=2000)
    t_q = rng.uniform(0.05, 0.95, 500)
    q = AnnData(np.zeros((500, 1)))
    q.obsm["X_emb"] = np.c_[np.cos(3 * t_q), np.sin(3 * t_q)]

    kknn_ingest(q, ref, obs_keys="pt", use_rep="X_emb")
    pred = q.obs["pt_kknn"].to_numpy()
    assert np.mean(np.abs(pred - t_q)) < 0.025
    assert q.obs["mapping_std_pt_kknn"].between(0, 0.2).all()
    assert q.obs["mapping_confidence_pt_kknn"].isna().all()


@requires_pacmap
def test_kknn_ingest_transform_uses_model_space():
    """barycenter='transform' projects queries in the model's training space and rejects mismatched references."""
    ref = _typed_log(n_per_type=100)
    del ref.layers["counts"]
    sc.pp.highly_variable_genes(ref, n_top_genes=100, flavor="seurat")
    sc.tl.pca(ref, n_comps=12)
    model = pacmap.PaCMAP(n_components=2, random_state=0)
    ref.obsm["X_pacmap"] = model.fit_transform(ref.obsm["X_pca"])

    query = ref[:120].copy()
    kknn_ingest(
        query,
        ref,
        obs_keys="ct",
        obsm_keys="X_pacmap",
        barycenter="transform",
        embedding_models={"X_pacmap": model},
    )
    Y = query.obsm["X_pacmap_kknn"]
    assert Y.shape == (120, 2)
    d = ((Y[:, None, :] - ref.obsm["X_pacmap"][None, :, :]) ** 2).sum(-1)
    nn_type = ref.obs["ct"].to_numpy()[d.argmin(1)]
    assert (nn_type == query.obs["ct"].to_numpy()).mean() > 0.9

    with pytest.raises(ValueError, match="fit on"):
        kknn_ingest(
            query,
            ref[:300].copy(),
            obs_keys="ct",
            obsm_keys="X_pacmap",
            barycenter="transform",
            embedding_models={"X_pacmap": model},
            recompute_ref_PCA=False,
        )


# =============================================================================
# Lineage Coupling & Multiscale Coarsening
# =============================================================================


@pytest.mark.parametrize("method", ["hypergeometric", "permutation"])
def test_calculate_lineage_coupling(method):
    """calculate_lineage_coupling detects planted coupling, returns symmetric matrices, drops NaNs, and rejects bad methods."""
    rows = []
    # 30 size-2 clones coupling Type_0 and Type_1 (expected under null: ~3.0, z ~ +15.7)
    for c in range(30):
        rows.extend([("Type_0", f"C01_{c}"), ("Type_1", f"C01_{c}")])
    # 60 pure Type_2 clones and 60 pure Type_3 clones of size 2
    for c in range(60):
        rows.extend([("Type_2", f"C2_{c}")] * 2)
        rows.extend([("Type_3", f"C3_{c}")] * 2)
    # NaN entries that must be dropped silently
    rows.extend([(np.nan, "C_nan1"), ("Type_0", np.nan)])

    obs = pd.DataFrame(rows, columns=["cell_type", "CloneID"])
    obs["cell_type"] = obs["cell_type"].astype("category")
    adata = AnnData(obs=obs)

    saved_state = np.random.get_state()
    try:
        np.random.seed(42)
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            obs_counts, z_scores, p_vals = calculate_lineage_coupling(
                adata,
                label_key="cell_type",
                clone_key="CloneID",
                method=method,
                n_permutations=100,
            )
    finally:
        np.random.set_state(saved_state)

    for r in record:
        assert "shuffling a 'Categorical' object" not in str(r.message)

    assert obs_counts.shape == (4, 4)
    assert obs_counts.loc["Type_0", "Type_1"] == 30
    assert obs_counts.loc["Type_0", "Type_2"] == 0
    assert obs_counts.loc["Type_2", "Type_3"] == 0

    for mat in (obs_counts, z_scores, p_vals):
        np.testing.assert_allclose(mat.to_numpy(), mat.to_numpy().T)

    assert z_scores.loc["Type_0", "Type_1"] > 5.0
    assert p_vals.loc["Type_0", "Type_1"] < 0.02
    assert z_scores.loc["Type_0", "Type_2"] < -1.0
    assert p_vals.loc["Type_0", "Type_2"] > 0.5

    if method == "hypergeometric":
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_lineage_coupling(adata, method="invalid_method")


def test_lineage_coupling_permutation_with_integer_labels():
    """Permutation lineage coupling works on integer cluster IDs and correlates with hypergeometric z-scores."""
    rng = np.random.default_rng(0)
    n = 400
    a = AnnData(np.zeros((n, 1)))
    a.obs["cell_type"] = rng.integers(0, 4, n)
    a.obs["CloneID"] = rng.integers(0, 120, n).astype(str)

    saved_state = np.random.get_state()
    try:
        np.random.seed(0)
        _, z_perm, p_perm = calculate_lineage_coupling(
            a, method="permutation", n_permutations=200
        )
    finally:
        np.random.set_state(saved_state)
    _, z_hyp, _ = calculate_lineage_coupling(a, method="hypergeometric")

    assert np.abs(z_perm.to_numpy()).max() < 4
    assert (p_perm.to_numpy() > 0).all()
    assert np.corrcoef(z_perm.to_numpy().ravel(), z_hyp.to_numpy().ravel())[0, 1] > 0.5


def test_lineage_coupling_hypergeometric_matches_per_clone_formula():
    """Grouping clones by size matches the exact per-clone hypergeometric inclusion-exclusion formula."""
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

    def _p0(Kv, k):
        Kv, k = np.broadcast_arrays(Kv, k)
        out = np.zeros(Kv.shape)
        ok = (N - Kv) >= k
        out[ok] = np.exp(
            scipy.special.gammaln(N - Kv[ok] + 1)
            - scipy.special.gammaln(N - Kv[ok] - k[ok] + 1)
            - scipy.special.gammaln(N + 1)
            + scipy.special.gammaln(N - k[ok] + 1)
        )
        return out

    Pi = _p0(K[:, None], k_c[None, :])
    Pij = _p0(K[:, None, None] + K[None, :, None], k_c[None, None, :])
    p = 1 - Pi[:, None, :] - Pi[None, :, :] + Pij
    p[np.arange(3), np.arange(3), :] = 1 - Pi
    expected = p.sum(-1)
    std = np.sqrt(np.maximum((p * (1 - p)).sum(-1), 0))
    std[std == 0] = 1
    np.testing.assert_allclose(z.to_numpy(), (obs.to_numpy() - expected) / std, atol=1e-10)


@pytest.mark.parametrize("title", ["Test Title 1", "Lineage Coupling"])
def test_plot_coupling_heatmap(title):
    """plot_coupling_heatmap masks the upper triangle, overlays lower-triangle counts, and colors significance borders."""
    labels = ["A", "B", "C"]
    obs_counts = pd.DataFrame(
        [[10, 5, 1], [5, 12, 2], [1, 2, 8]], index=labels, columns=labels
    )
    z_scores = pd.DataFrame(
        [[6.0, 3.0, -1.0], [3.0, 2.0, 0.5], [-1.0, 0.5, 1.0]], index=labels, columns=labels
    )
    p_vals = pd.DataFrame(
        [[0.0005, 0.005, 0.5], [0.005, 0.02, 0.3], [0.5, 0.3, 0.0001]],
        index=labels,
        columns=labels,
    )

    try:
        fig = plot_coupling_heatmap(obs_counts, z_scores, p_vals, title=title)
        ax = fig.axes[0]
        assert isinstance(fig, plt.Figure)
        assert ax.get_title() == title

        # Upper triangle masked, lower triangle (incl. diagonal) unmasked
        mask_2d = np.ma.getmaskarray(ax.collections[0].get_array()).reshape(3, 3)
        assert np.all(mask_2d[np.triu_indices(3, k=1)])
        assert not np.any(mask_2d[np.tril_indices(3)])

        # Lower-triangle count overlays (6 cells); |z| > 5 rendered in white
        assert len(ax.texts) == 6
        assert {t.get_text() for t in ax.texts} == {"10", "5", "12", "1", "2", "8"}
        text_at_00 = [t for t in ax.texts if t.get_text() == "10"][0]
        assert mcolors.to_hex(text_at_00.get_color()) == "#ffffff"

        # Significance borders: 2 black (p <= 0.001) and 1 orange (0.001 < p <= 0.01)
        rects = [p for p in ax.patches if isinstance(p, mpatches.Rectangle)]
        edge_colors = sorted(mcolors.to_hex(r.get_edgecolor()) for r in rects)
        expected_colors = sorted(
            [mcolors.to_hex("black"), mcolors.to_hex("black"), mcolors.to_hex("orange")]
        )
        assert edge_colors == expected_colors
    finally:
        plt.close("all")


def test_perfect_hierarchy():
    """multiscale_coarsening recovers a clean two-scale blob hierarchy with zero inconsistencies and >95% purity."""
    n_points = 50
    rng = np.random.default_rng(42)
    X = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.5, size=(n_points, 2)),
            rng.normal(loc=2.0, scale=0.5, size=(n_points, 2)),
            rng.normal(loc=10.0, scale=0.5, size=(n_points, 2)),
        ]
    )
    obs_names = (
        [f"A1_{i}" for i in range(n_points)]
        + [f"A2_{i}" for i in range(n_points)]
        + [f"B1_{i}" for i in range(n_points)]
    )
    adata = AnnData(X=X, obs=pd.DataFrame(index=obs_names))
    sc.pp.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=15)

    res_coarse, res_fine = 0.1, 1.0
    results = multiscale_coarsening(
        adata, resolutions=[res_coarse, res_fine], return_output=True, random_state=42
    )
    assert len(results["consistency"]) == 0
    assert len(results["clustering"][res_coarse].unique()) < len(
        results["clustering"][res_fine].unique()
    )
    purity_map = results["hierarchy"]["purity"][(res_fine, res_coarse)]
    assert np.mean(list(purity_map.values())) > 0.95


def test_lineage_inconsistency():
    """multiscale_coarsening detects a Simpson's-paradox direct vs. indirect parent contradiction."""
    n_cells = 100
    adata = AnnData(
        X=np.zeros((n_cells, 2)),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
    )
    adata.obsp["connectivities"] = np.zeros((n_cells, n_cells))

    def _side_effect_leiden(adata, resolution, key_added, **kwargs):
        if resolution == 1.0:
            labels = ["C1"] * 41 + ["C2"] * 59
        elif resolution == 2.0:
            labels = ["M1"] * 61 + ["M2"] * 39
        else:
            labels = ["F1"] * n_cells
        adata.obs[key_added] = pd.Categorical(labels)

    with patch("eigenp_utils.single_cell.sc.tl.leiden", side_effect=_side_effect_leiden):
        results = multiscale_coarsening(
            adata, resolutions=[1.0, 2.0, 3.0], return_output=True
        )

    consistency_df = results["consistency"]
    assert len(consistency_df) == 1
    row = consistency_df.iloc[0]
    assert row["fine_cluster"] == "F1"
    assert row["mid_parent"] == "M1"
    assert row["indirect_grandparent"] == "C1"
    assert row["direct_grandparent"] == "C2"
    assert (row["coarse_res"], row["mid_res"], row["fine_res"]) == (1.0, 2.0, 3.0)


def test_multiscale_coarsening_store_in_uns_can_be_written(tmp_path):
    """Stored multiscale_hierarchy in .uns uses string keys and round-trips through h5ad."""
    a = _typed_log(n_per_type=40)
    sc.pp.pca(a, n_comps=10)
    sc.pp.neighbors(a, n_neighbors=10)
    res = multiscale_coarsening(a, resolutions=[0.2, 0.5, 1.0], store_in_uns=True)

    assert 0.5 in res["clustering"]
    stored = a.uns["multiscale_hierarchy"]
    assert "0.5->0.2" in stored["hierarchy"]["tree"]
    assert all(isinstance(v, np.ndarray) for v in stored["clustering"].values())

    try:
        AnnData(np.zeros((2, 1))).write_h5ad(tmp_path / "probe.h5ad")
    except Exception as e:
        pytest.skip(f"this anndata/pandas pair cannot write h5ad at all: {e}")
    a.write_h5ad(tmp_path / "x.h5ad")
    back = anndata.read_h5ad(tmp_path / "x.h5ad")
    assert "0.5->0.2" in back.uns["multiscale_hierarchy"]["hierarchy"]["tree"]


def test_multiscale_coarsening_clusters_on_use_rep():
    """multiscale_coarsening builds a dedicated neighbor graph when use_rep differs from existing neighbors."""
    a = _typed_log(n_per_type=40)
    sc.pp.pca(a, n_comps=10)
    a.obsm["X_other"] = np.random.default_rng(0).normal(size=(a.n_obs, 3))
    sc.pp.neighbors(a, n_neighbors=10)

    with pytest.warns(UserWarning, match="multiscale_neighbors"):
        multiscale_coarsening(a, resolutions=[0.5, 1.0], use_rep="X_other")
    assert a.uns["multiscale_neighbors"]["params"]["use_rep"] == "X_other"

    b = AnnData(np.random.default_rng(1).normal(size=(200, 20)).astype(np.float32))
    multiscale_coarsening(b, resolutions=[0.5], use_rep="X")
    assert "X_pca" not in b.obsm
    assert b.uns["neighbors"]["params"]["use_rep"] == "X"


# =============================================================================
# Volcano Plots
# =============================================================================


def test_plot_volcano_adata():
    """plot_volcano_adata handles group formats, kwargs forwarding, positive-only, asymmetric cutoffs, custom ax, and missing keys."""
    rng = np.random.default_rng(42)
    n_vars = 50
    adata = AnnData(
        X=rng.random((20, n_vars)),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)]),
    )
    lfc = np.concatenate(
        [
            rng.uniform(2.5, 4.0, 10),  # strongly up
            rng.uniform(-4.0, -2.5, 10),  # strongly down
            rng.uniform(-1.0, -0.7, 10),  # moderately down (hits asymmetric cutoff)
            rng.uniform(-0.3, 0.3, 20),  # non-significant
        ]
    )
    pvals = np.concatenate([rng.uniform(1e-5, 0.01, 30), rng.uniform(0.2, 0.9, 20)])
    names = np.array([f"gene_{i}" for i in range(n_vars)])

    dt_f = [("0", "f8"), ("1", "f8")]
    dt_s = [("0", "U16"), ("1", "U16")]
    p_arr, lfc_arr, n_arr = np.zeros(n_vars, dtype=dt_f), np.zeros(n_vars, dtype=dt_f), np.zeros(n_vars, dtype=dt_s)
    for g in ("0", "1"):
        p_arr[g] = pvals
        lfc_arr[g] = lfc
        n_arr[g] = names
    adata.uns["rank_genes_groups"] = {
        "pvals_adj": p_arr,
        "logfoldchanges": lfc_arr,
        "names": n_arr,
    }

    try:
        with patch("eigenp_utils.single_cell.adjust_text") as mock_adjust_text:
            # group as string & single-element list
            ax_str = plot_volcano_adata(adata, "rank_genes_groups", group="0")
            assert isinstance(ax_str, plt.Axes)
            assert len(ax_str.lines) == 3  # 1 hline + 2 vlines
            assert mock_adjust_text.called

            ax_list = plot_volcano_adata(adata, "rank_genes_groups", group=["0"])
            assert isinstance(ax_list, plt.Axes)

            # Multi-group list rejected
            with pytest.raises(ValueError, match="only supports plotting a single group"):
                plot_volcano_adata(adata, "rank_genes_groups", group=["0", "1"])

            # Custom adjust_text kwargs forwarded
            plot_volcano_adata(
                adata,
                "rank_genes_groups",
                group="0",
                force_text=(1.0, 1.0),
                arrowprops={"color": "blue"},
            )
            call_kwargs = mock_adjust_text.call_args.kwargs
            assert call_kwargs["force_text"] == (1.0, 1.0)
            assert call_kwargs["arrowprops"]["color"] == "blue"

            # plot_positive_only draws only 1 vertical threshold line and highlights positive LFCs
            ax_pos = plot_volcano_adata(
                adata, "rank_genes_groups", group="0", plot_positive_only=True
            )
            assert len(ax_pos.lines) == 2
            sig_x = ax_pos.collections[1].get_offsets()[:, 0]
            assert len(sig_x) == 10 and np.all(sig_x > 1.0)

            # Asymmetric logfc_cutoff=(min, max) places threshold lines at (-0.5, 2.0)
            ax_asym = plot_volcano_adata(
                adata, "rank_genes_groups", group="0", logfc_cutoff=(-0.5, 2.0)
            )
            vline_xs = sorted(float(line.get_xdata()[0]) for line in ax_asym.lines[1:])
            np.testing.assert_allclose(vline_xs, [-0.5, 2.0])
            asym_sig_x = ax_asym.collections[1].get_offsets()[:, 0]
            assert len(asym_sig_x) == 30  # 10 up (>2.0) + 20 down (<-0.5)

            # Invalid logfc_cutoff tuple length
            with pytest.raises(ValueError, match="tuple of length 2"):
                plot_volcano_adata(
                    adata, "rank_genes_groups", group="0", logfc_cutoff=(1, 2, 3)
                )

            # Missing key returns None
            assert plot_volcano_adata(adata, "wrong_key", group="0") is None

            # Custom ax is reused and returned
            _, custom_ax = plt.subplots()
            ret_ax = plot_volcano_adata(
                adata, "rank_genes_groups", group="0", ax=custom_ax
            )
            assert ret_ax is custom_ax
    finally:
        plt.close("all")


def test_missing_adjust_text():
    """plot_volcano_adata raises a helpful ImportError when adjustText is unavailable."""
    with patch("eigenp_utils.single_cell.adjust_text", None):
        adata = AnnData(X=np.random.default_rng(0).random((10, 10)))
        with pytest.raises(ImportError, match="pip install adjustText"):
            plot_volcano_adata(adata, "key", "group")


def test_plot_volcano_all_zero_pvalues():
    """All-zero p-values are capped at finite -log10(tiny) rather than producing inf/NaN."""
    pytest.importorskip("adjustText")
    a = _volcano_adata(np.zeros(6))
    try:
        ax = plot_volcano_adata(a, "rank_genes_groups", group="0", label_top_n=3)
        ys = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections])
        assert np.isfinite(ys).all() and ys.max() > 300
    finally:
        plt.close("all")


def test_plot_volcano_uses_current_adjusttext_api(caplog, monkeypatch):
    """plot_volcano_adata passes only keywords accepted by the installed adjustText signature."""
    adjustText = pytest.importorskip("adjustText")
    params = inspect.signature(adjustText.adjust_text).parameters
    if "force_static" not in params:
        pytest.skip("adjustText < 1.0")
    passed = {}

    @functools.wraps(adjustText.adjust_text)
    def _spy(*args, **kwargs):
        passed.update(kwargs)
        return adjustText.adjust_text(*args, **kwargs)

    monkeypatch.setattr(scmod, "adjust_text", _spy)
    a = _volcano_adata(np.r_[np.full(10, 1e-10), np.full(10, 0.5)])
    try:
        with caplog.at_level(logging.WARNING, logger="adjustText"):
            plot_volcano_adata(a, "rank_genes_groups", group="0", label_top_n=5)
        assert set(passed) - {"arrowprops"} <= set(params)
        assert not any("FancyArrowPatch" in r.getMessage() for r in caplog.records)
    finally:
        plt.close("all")


def test_plot_volcano_adata_lays_out_only_the_figure_it_creates():
    """plot_volcano_adata(ax=...) leaves every existing figure's layout alone.

    Strategy: draw on a caller's axes once while an unrelated figure is current and once
    while the caller's own figure is current, recording all axes positions before and
    after; then call without ``ax`` as a control. adjust_text is patched out because only
    the layout is under test.
    Expected: with ``ax`` given, no axes of the unrelated figure or of the caller's
    figure moves (no pyplot-global tight_layout); without ``ax`` the function's own
    figure is laid out (its axes leave Matplotlib's default subplot position).
    """
    a = _volcano_adata(np.r_[np.full(10, 1e-10), np.full(10, 0.5)])
    fig_user, ax_user = plt.subplots()
    fig_other, _ = plt.subplots(1, 2)
    default_pos = ax_user.get_position().bounds

    def positions():
        return [ax.get_position().bounds for f in (fig_user, fig_other) for ax in f.axes]

    try:
        with patch("eigenp_utils.single_cell.adjust_text"):
            for current in (fig_other, fig_user):
                plt.figure(current.number)
                before = positions()
                assert plot_volcano_adata(a, "rank_genes_groups", group="0", ax=ax_user) is ax_user
                assert positions() == before

            ax_new = plot_volcano_adata(a, "rank_genes_groups", group="0")
        assert ax_new.figure not in (fig_user, fig_other)
        assert not np.allclose(ax_new.get_position().bounds, default_pos)
    finally:
        plt.close("all")


_PYPLOT_GLOBALS = ("gcf", "gca", "tight_layout", "title", "xticks", "yticks", "legend", "scatter", "sca")


@pytest.fixture
def no_pyplot_globals(monkeypatch):
    """Make the pyplot state-machine calls that act on the current figure/axes fail (C-05).

    plt.subplots (and the plt.figure it calls) stays allowed: a helper may create its own figure.
    """
    def boom(name):
        def _f(*a, **k):
            raise AssertionError(f"plt.{name} used: helpers must work on their own Figure/Axes")
        return _f

    for name in _PYPLOT_GLOBALS:
        monkeypatch.setattr(scmod.plt, name, boom(name))


def test_plot_coupling_heatmap_is_object_oriented(no_pyplot_globals):
    """C-05: plot_coupling_heatmap builds its own Figure/Axes (no plt.figure/title/xticks/legend/gcf).

    Expected: the returned Figure's first axes is the heatmap, with the title, rotated x tick
    labels and the p-value legend; a figure that was current before is left untouched.
    """
    other, ax_other = plt.subplots()
    pos = ax_other.get_position().bounds
    labels = ["A", "B"]
    obs = pd.DataFrame([[3, 1], [1, 4]], index=labels, columns=labels)
    z = pd.DataFrame([[1.0, 0.5], [0.5, 2.0]], index=labels, columns=labels)
    p = pd.DataFrame([[0.0001, 0.5], [0.5, 0.005]], index=labels, columns=labels)
    fig = plot_coupling_heatmap(obs, z, p, title="T")
    ax = fig.axes[0]
    assert fig is not other and ax.get_title() == "T"
    assert ax.get_legend() is not None
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["p ≤ 0.01", "p ≤ 0.001"]
    assert all(t.get_rotation() == 45 for t in ax.get_xticklabels())
    assert ax_other.get_position().bounds == pos and ax_other.get_legend() is None


def test_plot_clustering_tree_legend_handles_add_nothing_to_axes(no_pyplot_globals):
    """C-05: the size-legend handles are Line2D proxies, not empty scatters on the current axes."""
    import matplotlib.legend

    adata = AnnData(X=np.zeros((60, 3)))
    adata.obs["leiden_0.5"] = pd.Categorical(np.repeat(["0", "1"], 30))
    adata.obs["leiden_1.0"] = pd.Categorical(np.repeat(["0", "1", "2"], 20))
    fig = plot_clustering_tree(adata, ["leiden_0.5", "leiden_1.0"])
    ax = fig.axes[0]
    assert all(len(c.get_offsets()) > 0 for c in ax.collections if hasattr(c, "get_offsets"))
    legends = [a for a in ax.get_children() if isinstance(a, matplotlib.legend.Legend)]
    titles = {lg.get_title().get_text() for lg in legends}
    assert {"Cluster Size", "In-Proportion"} <= titles


def test_ari_and_uncertainty_plots_lay_out_their_own_figure(no_pyplot_globals):
    """C-05: plot_celltype_ari / barplot_cluster_uncertainty use fig.tight_layout, not plt.tight_layout."""
    ari = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], index=[0.5, 1.0], columns=[0.5, 1.0])
    fig, ax = plot_celltype_ari(ari)
    assert ax.figure is fig and len(ax.images) == 1
    fig2, ax2 = barplot_cluster_uncertainty(pd.DataFrame({"uncertainty": [0.3, 0.1]}, index=["a", "b"]))
    assert ax2.figure is fig2 and len(ax2.patches) == 2


def test_archetype_plots_return_figures_and_do_not_print(capsys):
    """C-05: with show=False the archetype plots return their figures; progress goes to the logger.

    plot_archetype_summary -> (score figure, top-genes figure); plot_archetype_assignments -> Figure.
    With the default show (scanpy autoshow) they return None as before, so notebook output is unchanged.
    """
    a = _typed_log(n_per_type=30)
    a.obsm["X_umap"] = np.random.default_rng(0).normal(size=(a.n_obs, 2))
    res = find_expression_archetypes(a, [f"G{i}" for i in range(20)], 2)
    capsys.readouterr()
    fig_score, fig_genes = plot_archetype_summary(a, res, archetype_id=1, k=2, show=False)
    assert isinstance(fig_score, plt.Figure) and isinstance(fig_genes, plt.Figure)
    assert any(ax.get_title() == "Archetype 1 Score" for ax in fig_score.axes)
    fig = plot_archetype_assignments(a, res, show=False)
    assert isinstance(fig, plt.Figure) and len([ax for ax in fig.axes if ax.get_title()]) >= 2
    assert capsys.readouterr().out == ""
