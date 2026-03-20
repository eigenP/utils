import numpy as np
import scipy.sparse as sp
from anndata import AnnData
import pytest
from eigenp_utils.single_cell import _extract_gene_vector

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
