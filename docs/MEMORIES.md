# Project Memories and Guidelines

This document tracks important context, guidelines, and behavioral rules established during development.

## 1. Code Quality & Planning
- When creating execution plans, avoid vague steps like "Add tests" (Specificity Rule); instead, provide concrete actions specifying the target files and functionality to assert. Always include an explicit step to run the test suite to verify implementation before entering the pre-commit phase (Completeness Rule).
- When creating execution plans, always use tools like `read_file`, `cat`, or `grep` to inspect the actual code structure and determine current variables before proposing modifications, to avoid violating Exploration and Groundedness rules.
- When creating an execution plan, the pre-commit step description must exactly match the required text: "Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done." without any additional formatting or text.
- The project's development environment and dependencies, including testing tools, should be installed using `uv venv && source .venv/bin/activate && uv pip install -e ".[all,dev]"` to ensure all required module dependencies (like scanpy) are available. If running specific tools, `uv run` and `uvx` may also be used.
- The project uses inline script metadata (PEP 723) to define specialized or file-specific dependencies within individual source files in `src/eigenp_utils/`, allowing scripts to run in isolation via tools like `marimo` or `uvx` / `pipx`. Core dependencies remain listed in `pyproject.toml`.
- When programmatically generating Marimo notebooks, append a run guard statement (`if __name__ == "__main__": app.run()`) at the end of the file to prevent `marimo check` warnings regarding missing run guards. Marimo notebooks should be validated using `marimo check`.
- The project's test suite can be executed using `uv run pytest tests/`.
- When writing tests for plotting utilities, prefer testing the top-level API functions (e.g., `show_zyx_max_slice_interactive`) rather than directly instantiating and testing lower-level internal widget classes (e.g., `TNIASliceWidget`). For guidance on what constitutes a "good test", refer to `testr.md`.

## 2. Personalities and Agent Roles
- **Matth**: The agent can act as "Matth 🧠 — Algorithmic Design & Applied Math Agent" when requested or appropriate. In this mode, prioritize mathematical correctness, statistical rigor, and algorithmic theory over heuristics. For details, refer to `matth.md`.
- **Testr**: For guidance on writing rigorous, conceptual tests, the agent should channel "Testr" by referring to `testr.md`.
- **Bolt**: For general development velocity and architectural guidance, refer to `bolt.md`.

## 3. General NumPy / SciPy / Pandas / Matplotlib
- The codebase assumes a (Z, Y, X) dimension order for 3D volumes and coordinate points (e.g., in `src/eigenp_utils/image_and_labels_utils.py`). Functions like `windowed_slice_projection` default to `axis=0` (Z-axis).
- When using Matplotlib's `violinplot` (e.g., in `eigenp_utils/plotting_utils.py`), use the `orientation` parameter (accepting 'vertical' or 'horizontal') instead of the deprecated `vert` boolean parameter to avoid PendingDeprecationWarnings.
- In NumPy and SciPy, always explicitly match data types when performing in-place sparse array modifications (e.g., `W.data *= array`) to avoid `UFuncTypeError` crashes. For example, Scanpy matrices typically default to `float32`, requiring `dtype=W.dtype` rather than the Python `float` default (`float64`).
- In `src/eigenp_utils/stats.py`, estimators prioritize mathematical exactness and statistical robustness: `cohens_d` calculates the exact Hedges' g correction factor using `scipy.special.gammaln` instead of standard approximations, and outlier detection employs robust Z-scores based on Median and MAD to prevent outlier masking.

## 4. AnnData / Scanpy Specifics
- When concatenating `AnnData` objects (such as merging reference and query datasets after ingestion), the modern `anndata.concat()` function is preferred and recommended over the deprecated `.concatenate()` method.
- In `src/eigenp_utils/single_cell.py`, `kknn_ingest` prevents overwriting the original query data by safely appending a `_kknn` suffix to all transferred `.obs`, `.obsm`, and mapping confidence keys.
- In `src/eigenp_utils/single_cell.py`, `score_celltypes` evaluates cell type signatures using `scanpy`, `binned`, or `binned_weighted` methods. It also supports 'net' versions (`net_scanpy`, `net_binned`, `net_binned_weighted`) which explicitly subtract the scores of negative markers (if provided) from the positive scores. Net binned scores are strictly clipped to a [0, 1] range to prevent negative scores.