import os
import tempfile
import pandas as pd
import numpy as np
import scanpy as sc
import pytest

from eigenp_utils.single_cell import export_obs_from_adata_to_csv, import_obs_to_adata_from_csv

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
