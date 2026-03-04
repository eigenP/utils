# utils
helper files


# Install

By default, the package installs a minimal set of dependencies (like `numpy`, `scipy`, `pandas`, `matplotlib`, etc).
To install it, run:

```bash
uv pip install git+https://github.com/eigenP/utils.git
```

### Optional Dependencies

You can choose to install optional dependencies if you need functionality such as single-cell analysis or image analysis:

- `[image-analysis]` - installs `scikit-image`.
- `[single-cell]` - installs packages like `scanpy`, `pacmap`, `leidenalg`, etc.
- `[plotting]` - installs `plotly`.
- `[all]` - installs all of the optional dependencies above.
- `[dev]` - installs all dependencies and additional tools for testing (e.g. `pytest`).

**Installing with `uv`:**

```bash
uv pip install "eigenp-utils[all] @ git+https://github.com/eigenP/utils.git"
```
*(Note: quotes are required so the shell doesn't misinterpret the brackets.)*

**Installing with standard `pip`:**

```bash
pip install "eigenp-utils[all] @ git+https://github.com/eigenP/utils.git"
```

You can replace `[all]` with other groups like `[single-cell]` or `[image-analysis,single-cell]` depending on your specific needs.


# License

License CC BY-NC https://creativecommons.org/licenses/by-nc/4.0/
