
import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from unittest.mock import patch, MagicMock
from eigenp_utils.single_cell import multiscale_coarsening

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
