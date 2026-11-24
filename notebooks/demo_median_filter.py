# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "anywidget",
#     "traitlets",
#     "numba ; sys_platform != 'emscripten'",
#     "opencv-python-headless ; sys_platform != 'emscripten'",
#     "eigenp-utils @ git+https://github.com/eigenP/utils.git@main ; sys_platform != 'emscripten'",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(auto_download=["html"])


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _(mo):
    mo.md(
        """
        ## Setup
        Installing the package from GitHub...
        """
    )

    import sys

    def in_wasm():
        return sys.platform in ("emscripten", "wasi")

    OWNER, REPO, REF = "eigenP", "utils", "main"
    if in_wasm():
        # micropip installs wheels only, from CORS-enabled hosts: CI publishes
        # the latest main build to the wasm-wheel branch (build-wasm-wheel.yml)
        GIT_URL = f"https://raw.githubusercontent.com/{OWNER}/{REPO}/wasm-wheel/eigenp_utils-0-py3-none-any.whl"
    else:
        GIT_URL = f"git+https://github.com/{OWNER}/{REPO}.git@{REF}"

    def install_local(url):
        import subprocess, sys, shutil

        if shutil.which("uv"):
            try:
                subprocess.check_call([
                    "uv", "pip", "install",
                    "--python", sys.executable,
                    url,
                ])
                return
            except subprocess.CalledProcessError:
                pass  # fall back to pip

        subprocess.check_call([sys.executable, "-m", "pip", "install", url])

    def local_checkout():
        # A source checkout or editable install (code outside site-packages) is a
        # developer's working copy: never replace it with GitHub main. Setting
        # EIGENP_UTILS_NO_INSTALL also skips the install.
        import importlib.util, os
        if os.environ.get("EIGENP_UTILS_NO_INSTALL"):
            return True
        spec = importlib.util.find_spec("eigenp_utils")
        origin = (spec.origin or "") if spec is not None else ""
        return bool(origin) and not any(d in origin for d in ("site-packages", "dist-packages"))

    def install_github():
        if in_wasm():
            import micropip
            return micropip.install([GIT_URL, "scipy"])
        elif local_checkout():
            print("Using the local eigenp_utils checkout; not installing from GitHub.")
        else:
            install_local(GIT_URL)

    res = install_github()
    if res is not None:
        await res

    import eigenp_utils
    print("eigenp_utils imported from:", eigenp_utils.__file__)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fast median filter

    `median_filter_fast(image, size=3, axes=(-2, -1), backend='auto')` from
    `eigenp_utils.median_filter` is a k x k median over two axes of an N-D
    stack (every (Y, X) slice of a ZYX or CZYX stack independently). It
    returns exactly `scipy.ndimage.median_filter(image, size=(1, ..., k, k),
    mode='reflect')` and runs at `cv2.medianBlur` speed, so OpenCV is not
    needed for this step.

    * **How:** per output row, the k values of each input column are sorted
      once with a sorting network and shared by the k windows that contain
      that column; a pruned selection network then takes the median of the
      k sorted columns (12 min/max per pixel for 3 x 3, 108 for 5 x 5). Other
      odd k use per-pixel selection.
    * **Backends:** `numba` (parallel over rows; used when installed) and a
      vectorised NumPy fallback (e.g. in the browser). They are bit-identical.
    * **dtypes:** uint8, uint16, float32, float64 (and other integer types);
      the output keeps the input dtype.
    * **Borders:** scipy `'reflect'` (`d c b a | a b c d`). OpenCV uses
      `BORDER_REPLICATE` (`a a a | a b c d`): the same for 3 x 3, but the
      outer 2 rows / columns can differ for 5 x 5 (shown below).
    * **NaN:** propagates. Any NaN in the window (borders included) gives NaN.
    """)
    return


@app.cell
def _():
    import importlib.util
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import ndimage

    from eigenp_utils.median_filter import median_filter_fast

    HAS_NUMBA = importlib.util.find_spec("numba") is not None
    HAS_CV2 = importlib.util.find_spec("cv2") is not None
    IN_WASM = importlib.util.find_spec("pyodide") is not None
    return HAS_CV2, HAS_NUMBA, IN_WASM, median_filter_fast, ndimage, np, plt, time


@app.cell
def _(IN_WASM, np):
    # Synthetic fluorescence-like slice: smooth blobs, shot noise and 0.5 %
    # hot pixels (what a 3 x 3 median is typically used to remove)
    N = 1024 if IN_WASM else 2048
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    clean = np.zeros((N, N), np.float32)
    for cy, cx, r in rng.uniform([0, 0, 4], [N, N, 24], (1500, 3)):
        clean += 2000 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
    clean += 200
    noisy = rng.poisson(clean).astype(np.float32)
    hot = rng.random((N, N)) < 0.005
    noisy[hot] = 60000
    images = {"float32": noisy, "uint16": noisy.astype(np.uint16)}
    return N, clean, images, noisy


@app.cell
def _(images, median_filter_fast, ndimage, np):
    # Exactness on the full slice: scipy reference, both dtypes, 3 x 3 and 5 x 5
    exact_rows = []
    for _name, _img in images.items():
        for _k in (3, 5):
            _ref = ndimage.median_filter(_img, size=_k, mode="reflect")
            _out = median_filter_fast(_img, _k)
            exact_rows.append(
                f"| {_name} | {_k} x {_k} | {_out.dtype} | {np.array_equal(_out, _ref)} |"
            )
    return (exact_rows,)


@app.cell(hide_code=True)
def _(exact_rows, mo):
    mo.md(
        "### Exact equality with `scipy.ndimage.median_filter(mode='reflect')`\n\n"
        "| dtype | window | output dtype | identical |\n|---|---|---|---|\n"
        + "\n".join(exact_rows)
    )
    return


@app.cell
def _(HAS_CV2, HAS_NUMBA, images, median_filter_fast, ndimage, np, time):
    N_THREADS = 4
    if HAS_NUMBA:
        import numba

        numba.set_num_threads(min(N_THREADS, numba.config.NUMBA_NUM_THREADS))
        numba_threads = numba.get_num_threads()
    else:
        numba_threads = 0
    if HAS_CV2:
        import cv2

        cv2.setNumThreads(N_THREADS)

    def best_of(fn, repeats):
        """Median wall time in ms after one warm-up call (numba compile / cache load)."""
        fn()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return 1e3 * float(np.median(times))

    timings = []
    for _name, _img in images.items():
        for _k in (3, 5):
            _row = {"dtype": _name, "k": _k}
            if HAS_CV2:
                _row["cv2.medianBlur"] = best_of(lambda: cv2.medianBlur(_img, _k), 30)
            if HAS_NUMBA:
                _row["numba"] = best_of(lambda: median_filter_fast(_img, _k, backend="numba"), 30)
            _row["numpy"] = best_of(lambda: median_filter_fast(_img, _k, backend="numpy"), 5)
            _row["scipy"] = best_of(lambda: ndimage.median_filter(_img, size=_k, mode="reflect"), 3)
            timings.append(_row)
    return N_THREADS, numba_threads, timings


@app.cell(hide_code=True)
def _(N, N_THREADS, mo, numba_threads, timings):
    def _fmt(row, key):
        return f"{row[key]:.2f}" if key in row else "n/a"

    _lines = []
    for _r in timings:
        _fast = _r.get("numba", _r["numpy"])
        _vs_cv2 = f"{_fast / _r['cv2.medianBlur']:.2f}x" if "cv2.medianBlur" in _r else "n/a"
        _lines.append(
            f"| {_r['dtype']} | {_r['k']} x {_r['k']} | {_fmt(_r, 'cv2.medianBlur')} | "
            f"{_fmt(_r, 'numba')} | {_fmt(_r, 'numpy')} | {_fmt(_r, 'scipy')} | "
            f"{_vs_cv2} | {_r['scipy'] / _fast:.0f}x |"
        )
    mo.md(
        f"### Timings, one {N} x {N} slice (median ms; numba threads: {numba_threads}, "
        f"OpenCV threads: {N_THREADS})\n\n"
        "| dtype | window | cv2 | numba | numpy | scipy | fastest ours / cv2 | scipy / fastest ours |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(_lines)
        + "\n\nscipy's `median_filter` is single-threaded. Times include allocating "
        "the output (all methods). The first numba call per dtype compiles "
        "(a few seconds, cached on disk) and is excluded by the warm-up."
    )
    return


@app.cell
def _(np, plt, timings):
    _methods = [m for m in ("cv2.medianBlur", "numba", "numpy", "scipy") if m in timings[0]]
    _labels = [f"{r['dtype']}\n{r['k']}x{r['k']}" for r in timings]
    _x = np.arange(len(timings))
    _w = 0.8 / len(_methods)
    fig_t, ax_t = plt.subplots(figsize=(9, 4))
    for _i, _m in enumerate(_methods):
        ax_t.bar(_x + (_i - (len(_methods) - 1) / 2) * _w, [r[_m] for r in timings], _w, label=_m)
    ax_t.set_xticks(_x, _labels)
    ax_t.set_yscale("log")
    ax_t.set_ylabel("time per slice (ms, log scale)")
    ax_t.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    ax_t.set_title("Median filter, one 2D slice")
    fig_t.tight_layout()
    fig_t
    return


@app.cell
def _(clean, median_filter_fast, noisy, np, plt):
    # What the 3 x 3 median does: hot pixels removed, edges kept
    _sl = (slice(512, 768), slice(512, 768))
    _den = median_filter_fast(noisy, 3)
    _vmax = float(np.percentile(clean[_sl], 99.5))
    fig_v, ax_v = plt.subplots(1, 3, figsize=(12, 4.2))
    for _ax, (_title, _im) in zip(ax_v, [("clean", clean), ("noisy + hot pixels", noisy),
                                         ("median_filter_fast, 3 x 3", _den)]):
        _ax.imshow(_im[_sl], cmap="gray", vmin=0, vmax=_vmax)
        _ax.set_title(_title)
        _ax.axis("off")
    fig_v.tight_layout()
    fig_v
    return


@app.cell
def _(HAS_CV2, images, median_filter_fast, mo, np):
    # Border convention: scipy 'reflect' vs OpenCV BORDER_REPLICATE
    if HAS_CV2:
        import cv2 as _cv2

        _img = images["uint16"]
        _rows = []
        for _k in (3, 5):
            _diff = median_filter_fast(_img, _k) != _cv2.medianBlur(_img, _k)
            _p = _k // 2
            _inner = _diff[_p:-_p, _p:-_p].sum()
            _rows.append(f"| {_k} x {_k} | {_diff.sum()} | {_inner} |")
        border_md = mo.md(
            "### Borders vs `cv2.medianBlur`\n\n"
            "| window | pixels differing | of which away from the outer k // 2 rows / columns |\n"
            "|---|---|---|\n" + "\n".join(_rows)
            + "\n\nFor 5 x 5 only border pixels differ: OpenCV repeats the edge pixel "
            "(`a a | a b c`) where scipy reflects (`b a | a b c`)."
        )
    else:
        border_md = mo.md("OpenCV is not installed: border comparison skipped.")
    border_md
    return


@app.cell
def _(median_filter_fast, mo, np):
    # NaN policy: any NaN in the window -> NaN
    _a = np.arange(49, dtype=np.float32).reshape(7, 7)
    _a[3, 3] = np.nan
    _out = median_filter_fast(_a, 3)
    mo.md(
        "### NaN propagates\n\nA single NaN at (3, 3) of a 7 x 7 image gives NaN "
        f"exactly in its 3 x 3 neighbourhood: {int(np.isnan(_out).sum())} NaN pixels, "
        f"rows/cols {sorted(set(np.argwhere(np.isnan(_out))[:, 0].tolist()))}."
    )
    return


if __name__ == "__main__":
    app.run()
