## 2024-05-23 - Sparse Matrix Row Standardization
**Learning:** `sp.diags(inv) @ W` (sparse @ sparse) creates a new CSR matrix, allocating full indices and data arrays.
**Action:** For row scaling of CSR matrices, use in-place multiplication of `W.data` with `np.repeat(inv, np.diff(W.indptr))` to save memory and time (~2x faster), provided `W` is a safe copy.
