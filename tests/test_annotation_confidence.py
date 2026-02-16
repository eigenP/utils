
import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import norm
from eigenp_utils.single_cell import annotate_clusters_by_markers

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
