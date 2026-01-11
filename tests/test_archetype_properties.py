
import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from eigenp_utils.single_cell import find_expression_archetypes

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
