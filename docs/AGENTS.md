# Agent and Contributor Guidelines

This document serves as the primary overview for both AI agents and human contributors working on this project. The project includes specialized guidelines and "personalities" for agents to adopt depending on the task.

## Documentation Structure
The `docs/` directory contains several important resources to help you write better code and tests for this project:

- **`MEMORIES.md`**: Tracks important context, learnings, and general behavioral rules established over time. Review this to understand existing conventions, performance gotchas, and architectural decisions.
- **`MARIMO_skill.md`**: Provides specific instructions and skills for working with `marimo` notebooks and applications within the project.

### Agent Personalities
For specialized tasks, agents should refer to these files to channel specific expertise:
- **`matth.md`**: "Matth 🧠 — Algorithmic Design & Applied Math". Owns mathematical and statistical correctness: estimator bias, calibration, exact formulations, numerical stability. Changes may alter outputs, but only with an analytic ground truth.
- **`testr.md`**: "Testr 🧪 — Invariant-Based Verification". Owns the test suite: conceptual, invariant- or analytic-oracle tests, edge cases and calibration checks, with no flaky or golden-value assertions. Fixes the bugs its tests expose.
- **`bolt.md`**: "Bolt ⚡ — Performance & Memory". Owns runtime and peak memory of hot paths. Output must be unchanged, shown by an equivalence test and a before/after benchmark.

Each persona file starts with a charter (mission, scope, operating procedure, definition of done, red flags), followed by a chronological journal of dated **Learning / Action** entries. Add new learnings to the journal of the persona that produced them.

**Weekly persona upgrade:** a scheduled task rotates the personas by ISO week number (`W mod 3`: 0 = Matth, 1 = Testr, 2 = Bolt). Each run opens one draft PR with a targeted change that introduces no regressions, plus a short report. Changes to public APIs or to visible output also update the matching demo notebook.

## Testing
- Install the test dependencies once: `uv sync --all-extras` (the `dev` extra includes pytest and pytest-playwright), then `uv run playwright install chromium` for the browser tests in `tests/test_frontend_*.py`.
- Run the suite with `uv run pytest tests/`. Without pytest-playwright or a browser the frontend tests are skipped locally; in CI (`CI=true`) they error instead.
- The Pyodide (WASM) suite: `uv build --wheel`, `npm install --no-save --prefix tests/wasm pyodide@314.0.0`, then `node tests/wasm/run_pyodide_tests.mjs` (see the `pyodide` job in `.github/workflows/ci.yml`).
- Ensure all tests pass locally before submission.

## Dependencies
- Install development dependencies using `uv venv && source .venv/bin/activate && uv pip install -e ".[all,dev]"` (or `uv run`).
- **Inline Script Metadata (PEP 723):**
    - Moving forward, we use inline script metadata to define dependencies for each individual source file in `src/eigenp_utils/`.
    - Core dependencies (like `numpy`, `scipy`) are still listed in `pyproject.toml`, but specialized or file-specific dependencies should be declared in the file header.
    - This approach facilitates running scripts in isolation (e.g., via `marimo` or `pipx`) and clarifies the specific requirements of each module.
    - Format:
      ```python
      # /// script
      # requires-python = ">=3.10"
      # dependencies = [
      #     "numpy",
      #     "pandas",
      #     "specialized-lib",
      # ]
      # ///
      ```
    - Start with permissive versioning (e.g., just the package name) unless a specific version is required.

## Python Versions
- The project requires Python 3.10 or later (uses newer syntax like `int | float`).
- GitHub Actions tests the code on Python 3.10, 3.11, 3.12 and 3.13.
- Keep the code compatible with these versions.

## Contributions
- Provide tests for new features or bug fixes.
- No specific linting configuration is provided; follow standard Python style.
