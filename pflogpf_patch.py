import re

with open("src/eigenp_utils/single_cell.py", "r") as f:
    content = f.read()

func = """def pflogpf(
    adata: sc.AnnData,
    target: str = "auto",
    key_added: str = "pflogpf",
    copy: bool = False,
    **kwargs,
) -> Optional[sc.AnnData]:
    \"\"\"
    Computes the PFlogPF (shifted-CLR) normalization on raw single-cell counts.

    Single-cell count normalization should stabilize technical variance, remove
    sequencing-depth effects, and preserve within-cell gene ranks. PFlog addresses
    these requirements by applying an Anscombe-calibrated shifted logarithm followed
    by within-cell CLR centering. Equivalently, it is a shifted centered log-ratio
    (CLR) transform. Across hundreds of datasets, and in an adapted analysis of the
    benchmarks of Ahlmann-Eltze & Huber (2023), PFlog removes residual depth structure
    while preserving rank information and stabilizing technical variance.

    The shift is set by the delta method. For a negative-binomial mean-variance
    model var = μ + α·μ², the Anscombe-derived count-scale pseudocount is y₀ = 1/(4α).
    In software notation, for cell c with depth s_c, this is equivalent to the
    cell-specific normalized shifted-log scale K_c = 4·α·s_c. At a representative
    depth s_*, the corresponding plotted scale is K_* = 4·α·s_*.

    Citations:
        - Theory & Benchmarks: https://github.com/pachterlab/BHGP_2022
        - Fast Rust implementation (scclr): https://github.com/cleartools/scclr

    Install (dev):
        uv venv
        uv pip install -e ".[test]" maturin
        uv run maturin develop --release

    Quickstart Example:
        import scclr
        # scverse in-place (AnnData / MuData), shaped like scanpy:
        scclr.pp.pflog(adata, target="auto")      # -> adata.layers["pflog"] + obs center
        scclr.tl.pca(adata, n_comps=50)             # -> adata.obsm["X_pca"], varm["PCs"], uns["pca"]

        # downstream sc.pp.neighbors(adata) / sc.tl.umap(adata) work unchanged
        # This swaps in for sc.pp.normalize_total + sc.pp.log1p + sc.tl.pca.

    Parameters
    ----------
    adata
        Annotated data matrix containing raw counts in `.X`.
    target
        Target depth (e.g. "auto").
    key_added
        Key in `adata.layers` to store the resulting PFlogPF matrix. Defaults to "pflogpf".
    copy
        If True, return a new AnnData object instead of modifying in-place.
    **kwargs
        Additional arguments passed to `scclr.pp.pflog`.

    Returns
    -------
    Optional[sc.AnnData]
        If `copy` is True, returns a new AnnData object with PFlogPF values in `.layers[key_added]`.
        Otherwise, modifies `adata` in-place and returns `None`.
    \"\"\"
    if not _has_integer_like_counts(adata.X):
        raise ValueError(
            "adata.X does not look like raw integer counts. "
            "The PFlogPF transform expects raw counts. "
            "Please provide true counts."
        )

    try:
        import scclr
    except ImportError:
        print(pflogpf.__doc__)
        raise ImportError(
            "The 'scclr' package is required for this function. "
            "Please install it using the instructions in the docstring above."
        )

    # Work on a copy if requested
    A = adata.copy() if copy else adata

    # call scclr.pp.pflog
    scclr.pp.pflog(A, target=target, **kwargs)

    # scclr.pp.pflog normally stores to "pflog" layer.
    # If the user provided a different key_added, we should move it.
    if key_added != "pflog" and "pflog" in A.layers:
        A.layers[key_added] = A.layers.pop("pflog")

    if copy:
        return A
    return None

"""

content = content.replace("def preprocess_subset(", func + "\ndef preprocess_subset(")

with open("src/eigenp_utils/single_cell.py", "w") as f:
    f.write(content)
