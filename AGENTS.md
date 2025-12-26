# Repository Guidelines

## Testing
- Use `pytest` for running tests located in the `tests/` folder.
- CI runs `pytest` after installing the package with `pip install .`. Ensure tests pass locally using the same command.

## Dependencies
- Development requirements are listed in `requirements-dev.txt` which installs the package in editable mode and includes `numpy` and `pytest`.
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
