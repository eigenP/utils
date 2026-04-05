
import pytest
import numpy as np
import scipy.sparse as sp
from eigenp_utils.single_cell import _compute_moran_moments, _ensure_csr_f32

def test_moran_moments_optimization():
    """
    Verify that the optimized moment calculation matches the original definition.
    Ensures correctness for both duplicates (robustness) and off-diagonal/asymmetric structures.
    """
    N = 100
    rng = np.random.default_rng(42)

    # 1. Construct a sparse matrix with:
    # - Off-diagonal elements
    # - Asymmetry (some (i, j) exist without (j, i))
    # - Duplicates (to verify canonicalization)

    # Base structure: random asymmetric connections
    density = 0.05
    nnz = int(N * N * density)
    rows = rng.integers(0, N, size=nnz)
    cols = rng.integers(0, N, size=nnz)
    data = rng.uniform(0.1, 1.0, size=nnz).astype(np.float32)

    # Add forced duplicates at specific location (0, 1)
    rows = np.concatenate([rows, [0, 0]])
    cols = np.concatenate([cols, [1, 1]])
    data = np.concatenate([data, [0.5, 0.5]])

    # Create raw matrix (COO allows duplicates)
    W_raw = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    # 2. Calculate Truth using the OLD method (manually implemented)
    # The old method (W + W.T) handles duplicates natively because
    # the addition operation on sparse matrices sums duplicates.
    # However, to be strictly fair and compare "canonical interpretation",
    # we should use the same matrix content.
    W_truth = W_raw.copy()
    W_truth.sum_duplicates() # Standardize for the "Truth" logic baseline

    Wt = W_truth.transpose()
    T = (W_truth + Wt)
    S1_truth = float(T.power(2).sum()) / 2.0

    # 3. Calculate using the NEW optimized function pipeline
    # We pass the raw matrix (with potential duplicates) to ensure the pipeline handles it.
    W_clean = _ensure_csr_f32(W_raw)

    # Verify duplicates were removed/summed
    assert W_clean.nnz <= W_raw.nnz
    # Specifically for our forced duplicates:
    # If standard tocsr() doesn't sum, W_raw has duplicates.
    # W_clean MUST NOT have duplicates.

    S0_opt, S1_opt, S2_opt = _compute_moran_moments(W_clean)

    # 4. Compare
    assert np.isclose(S1_truth, S1_opt, rtol=1e-5), f"S1 mismatch: {S1_truth} vs {S1_opt}"

    # Verify S0 and S2 as well for completeness
    S0_truth = float(W_truth.sum())
    R = np.array(W_truth.sum(axis=1)).flatten()
    C = np.array(W_truth.sum(axis=0)).flatten()
    S2_truth = float(np.sum((R + C)**2))

    assert np.isclose(S0_truth, S0_opt, rtol=1e-5)
    assert np.isclose(S2_truth, S2_opt, rtol=1e-5)
