import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from eigenp_utils.single_cell import score_celltypes, annotate_clusters_by_markers

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
