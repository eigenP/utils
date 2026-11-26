# Agent and Contributor Guidelines

This document serves as the primary overview for both AI agents and human contributors working on this project. The project includes specialized guidelines and "personalities" for agents to adopt depending on the task.

## Documentation Structure
The `docs/` directory contains several important resources to help you write better code and tests for this project:

- **`MEMORIES.md`**: Tracks important context, learnings, and general behavioral rules established over time. Review this to understand existing conventions, performance gotchas, and architectural decisions.
- **`MARIMO_skill.md`**: Provides specific instructions and skills for working with `marimo` notebooks and applications within the project.

**Where to record learnings:** repo-wide rules and learnings go in `docs/MEMORIES.md`; derivations and learnings specific to one persona go in that persona's journal (`docs/matth.md`, `docs/testr.md`, `docs/bolt.md`). The root `AGENTS.md` is only a pointer to these files.

### Agent Personalities
For specialized tasks, agents should refer to these files to channel specific expertise:
- **`matth.md`**: "Matth 🧠 — Algorithmic Design & Applied Math". Owns mathematical and statistical correctness: estimator bias, calibration, exact formulations, numerical stability. Changes may alter outputs, but only with an analytic ground truth.
- **`testr.md`**: "Testr 🧪 — Invariant-Based Verification". Owns the test suite: conceptual, invariant- or analytic-oracle tests, edge cases and calibration checks, with no flaky or golden-value assertions. Fixes the bugs its tests expose.
- **`bolt.md`**: "Bolt ⚡ — Performance & Memory". Owns runtime and peak memory of hot paths. Output must be unchanged, shown by an equivalence test and a before/after benchmark.

Each persona file starts with a charter (mission, scope, operating procedure, definition of done, red flags), followed by a chronological journal of dated **Learning / Action** entries. Add persona-specific learnings to the journal of the persona that produced them; repo-wide rules go in `MEMORIES.md`.

**Weekly persona upgrade:** a scheduled task rotates the personas by ISO week number (`W mod 3`: 0 = Matth, 1 = Testr, 2 = Bolt). Each run opens one draft PR with a targeted change that introduces no regressions, plus a short report. Changes to public APIs or to visible output also update the matching demo notebook.

## Testing
- Install the test dependencies once: `uv sync --all-extras` (the `dev` extra includes pytest and pytest-playwright), then `uv run playwright install chromium` for the browser tests in `tests/test_frontend_*.py`.
- Run the suite with `uv run pytest tests/`. Without pytest-playwright or a browser the frontend tests are skipped locally; in CI (`CI=true`) they error instead.
- The Pyodide (WASM) suite: `uv build --wheel`, `npm install --no-save --prefix tests/wasm pyodide@314.0.0`, then `node tests/wasm/run_pyodide_tests.mjs` (see the `pyodide` job in `.github/workflows/ci.yml`).
- Test files mirror source modules 1:1: tests for `src/eigenp_utils/<module>.py` live in `tests/test_<module>.py` (e.g. `tests/test_single_cell.py` for `single_cell.py`). Do not create new, narrowly scoped test files. Explicit exemptions: `tests/test_integration_seams.py` (seam tests where one module's output feeds another, e.g. `load_image` -> viewers, `estimate_drift` -> `apply_subpixel_drift_correction`; each test crosses at least two modules), `tests/test_package_metadata.py` (package surface, legacy names, no-print and PEP 723 header checks) and the environment tests `tests/test_clahe_no_skimage.py` / `tests/test_notebook_setup.py`.
- Every test function has a docstring stating what it tests, the strategy and the expected outcome (no trivial auto-generated summaries).
- Test public wrappers (e.g. `show_zyx_max_slice_interactive`) rather than internal widget classes (`TNIASliceWidget`) where possible.
- Ensure all tests pass locally before submission.

## Dependencies
- Install development dependencies from the lock file with `uv sync --all-extras` (see *Testing*), then run tools with `uv run`.
- **Inline Script Metadata (PEP 723):**
    - Moving forward, we use inline script metadata to define dependencies for each individual source file in `src/eigenp_utils/`.
    - Core dependencies (like `numpy`, `scipy`) are still listed in `pyproject.toml`, but specialized or file-specific dependencies should be declared in the file header.
    - This approach facilitates running scripts in isolation (e.g., via `marimo` or `pipx`) and clarifies the specific requirements of each module.
    - Format:
      ```python
      # /// script
      # requires-python = ">=3.11"
      # dependencies = [
      #     "numpy",
      #     "pandas",
      #     "specialized-lib",
      # ]
      # ///
      ```
    - Start with permissive versioning (e.g., just the package name) unless a specific version is required.
    - Each module in `src/` with unconditional third-party imports keeps a header that lists them under their `pyproject.toml` distribution names (e.g. `scikit-image`, not `skimage`) and says `requires-python = ">=3.11"`. Imports guarded by `try/except ImportError` are optional and may be left out; a header may list only distributions `pyproject.toml` declares. Modules without third-party imports need no header. `tests/test_package_metadata.py` enforces this (`task_calendar_scheduler.py`, a marimo notebook kept in `src/`, is exempt).

## Matplotlib (object-oriented interface)
- Library code never touches pyplot global state: no `plt.figure()`, `plt.gcf()`, `plt.gca()`, `plt.tight_layout()` or `plt.get_fignums()` on figures or axes that belong to the caller.
- Widgets render with `matplotlib.figure.Figure` (not `plt.figure`/`plt.subplots`), so they never register figures with pyplot.
- Functions that draw take `fig`/`ax` explicitly and return the figure (and axes) they drew on. Lay out only figures the function created itself (e.g. `layout='tight'` at creation), never one passed in by the caller.
- Save with `savefig_svg(path, fig=fig)`; omitting `fig` is deprecated (FutureWarning) until 0.1.0. The static ortho viewers default to `use_plt=False` and return a plain `Figure` that pyplot does not track.

## Output, warnings and deprecations
- Library code never calls `print` and never logs on the root logger: information goes to a module logger (`logging.getLogger(__name__)`), anything the user can act on is a `warnings.warn(..., UserWarning)`. `tests/test_package_metadata.py` enforces this for `src/` (`task_calendar_scheduler.py` is exempt). A function that stops printing gets a `Changed (D07, ...)` docstring note.
- Deprecate through `_conventions.warn_deprecated(old, new)`: a `FutureWarning` saying the name "will be removed in eigenp_utils 0.1.0", attributed to the caller. Add every new legacy name to `LEGACY` in `tests/test_package_metadata.py`.
- Every behaviour or default change gets a `Changed (<decision>, <date>): ...` note in the function's docstring.

## Python Versions
- The project requires Python 3.11 or later (3.10 reached end-of-life in October 2026). The stdlib `tomllib` can be used without a `tomli` fallback.
- GitHub Actions tests the code on Python 3.11, 3.12 and 3.13 with the newest dependencies, plus an `oldest-deps` job on 3.11 with older dependency lines (`pandas<3`, `numpy<2.3`, `scipy<1.16`, `scikit-image<0.26`).
- Keep the code compatible with these versions.

## Contributions
- Provide tests for new features or bug fixes.
- No specific linting configuration is provided; follow standard Python style.
