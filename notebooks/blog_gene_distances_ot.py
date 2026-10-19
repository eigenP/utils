# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "scipy",
#     "pandas",
#     "matplotlib",
#     "scikit-learn",
#     "anndata",
#     "scanpy ; sys_platform != 'emscripten'",
#     "pot ; sys_platform != 'emscripten'",
#     "wigglystuff",
#     "eigenp-utils[single-cell] @ git+https://github.com/eigenP/utils.git@main ; sys_platform != 'emscripten'",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
async def _(mo):
    import sys

    # In the browser (marimo WASM / Pyodide) scanpy and POT have no WebAssembly
    # builds; the notebook then swaps in numpy/sklearn/HiGHS equivalents.
    IN_WASM = sys.platform == "emscripten"
    if IN_WASM:
        import micropip

        await micropip.install([
            # micropip needs a wheel from a CORS-enabled host (build-wasm-wheel.yml)
            "https://raw.githubusercontent.com/eigenP/utils/wasm-wheel/eigenp_utils-0-py3-none-any.whl",
            "anndata",
            "wigglystuff",
            "scikit-learn",
        ])
    mo.callout(
        mo.md(
            "**Running in the browser (WebAssembly).** scanpy and POT cannot be installed here, so this "
            "version (1) normalises and runs PCA with numpy/scikit-learn, (2) uses a plain symmetric kNN "
            "graph instead of scanpy's UMAP connectivities (so `pearson_graph` differs slightly), and "
            "(3) solves the exact W1 linear programs with SciPy's HiGHS instead of POT (same values, slower). "
            "Defaults are smaller to keep it interactive. For the full version run it locally "
            "(`uvx marimo edit --sandbox blog_gene_distances_ot.py`) or on molab."
        ),
        kind="info",
    ) if IN_WASM else None
    return (IN_WASM,)


@app.cell
def _(IN_WASM):
    import time

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    import anndata as ad

    if IN_WASM:
        sc = ot = None  # no WebAssembly builds; fallbacks are used where they are called
    else:
        import scanpy as sc
        import ot
    from scipy.optimize import linprog
    from scipy.sparse.csgraph import connected_components, laplacian, shortest_path
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import rankdata, spearmanr
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from wigglystuff import ChartPuck

    from eigenp_utils.plotting_utils import set_plotting_style
    from eigenp_utils.single_cell import (
        ensure_neighbors,
        find_correlated_features,
        smooth_expression_on_graph,
    )

    set_plotting_style()
    return (
        ChartPuck,
        KMeans,
        NearestNeighbors,
        ad,
        connected_components,
        ensure_neighbors,
        find_correlated_features,
        laplacian,
        linprog,
        np,
        ot,
        pd,
        pdist,
        plt,
        rankdata,
        sc,
        shortest_path,
        smooth_expression_on_graph,
        sp,
        spearmanr,
        squareform,
        time,
    )


@app.cell
def _(mo):
    import base64
    import io

    # Palette shared by every figure: genes A / B (the pair under comparison) and the tree arms.
    COL_A, COL_B = "#3b6ea8", "#d1495b"
    BRANCH_COLORS = {"trunk": "#9a9a9a", "L": "#1b9e77", "R": "#8a5a9e"}

    _ROW_CSS = """
    <style>
    .rr-row{display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:.25rem .75rem}
    .rr-item{flex:1 1 var(--rr-min,260px);min-width:0;max-width:var(--rr-max,460px)}
    .rr-item img{display:block;width:100%;height:auto}
    .rr-arrow{flex:0 0 auto;display:flex;flex-direction:column;align-items:center;color:#777;line-height:1.1}
    .rr-arrow .rr-glyph{font-size:1.8rem}
    .rr-arrow small{font-size:.72rem;max-width:7.5rem;text-align:center}
    @media (max-width:700px){
      .rr-row{flex-direction:column;align-items:stretch}
      .rr-item{max-width:none}
      .rr-arrow{flex-direction:row;justify-content:center;gap:.4rem}
      .rr-arrow .rr-glyph{transform:rotate(90deg)}
      .rr-arrow small{max-width:none}
    }
    </style>
    """

    def fig_to_img(fig, dpi=160):
        """Render a Matplotlib figure to a width-responsive <img> and close it."""
        import matplotlib.pyplot as _plt

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        _plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img alt="figure" src="data:image/png;base64,{b64}"/>'

    def fig_row(*figs, arrows=None, min_px=260, max_px=460):
        """
        Lay single-panel figures out in a row that wraps on narrow screens.

        `arrows` (optional) holds one caption per gap; the arrow points right on wide screens and
        turns to point down once the row stacks vertically (viewport < 700 px).
        """
        arrows = list(arrows or [])
        parts = []
        for i, f in enumerate(figs):
            parts.append(f'<div class="rr-item">{fig_to_img(f) if hasattr(f, "savefig") else mo.as_html(f).text}</div>')
            if i < len(arrows):
                parts.append(f'<div class="rr-arrow"><span class="rr-glyph">&#10230;</span><small>{arrows[i]}</small></div>')
        style = f"--rr-min:{min_px}px;--rr-max:{max_px}px"
        return mo.Html(_ROW_CSS + f'<div class="rr-row" style="{style}">' + "".join(parts) + "</div>")

    return BRANCH_COLORS, COL_A, COL_B, fig_row


@app.cell(hide_code=True)
def _(mo):
    # Page-wide CSS for small screens: wide display maths and Markdown tables scroll sideways
    # inside their own box instead of widening the page.
    mo.Html(
        """
        <style>
        .katex-display{overflow-x:auto;overflow-y:hidden;padding-bottom:2px}
        .markdown table, .prose table{display:block;overflow-x:auto;max-width:100%}
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How far apart are two genes? Pearson, Spearman, Wasserstein, graph OT and the energy distance

    "Find me genes like *GATA1*" sounds like one question. It is really three, and each family of
    metrics answers a different one:

    $$
    \begin{array}{ccc}
    \textbf{Question} & & \textbf{Metric family} \\[0.4em]
    \text{Do } x, y \text{ rise and fall in the \emph{same cells}?} & \xrightarrow{\text{uses the pairing } (x_i,y_i)} & \text{Pearson, Spearman} \\[0.6em]
    \text{Do } x, y \text{ have the same \emph{value histogram}?} & \xrightarrow{\text{ignores the pairing}} & \text{1D Wasserstein on values} \\[0.6em]
    \text{Are } x, y \text{ expressed in the same \emph{place on the manifold}?} & \xrightarrow{\text{uses a cell--cell metric}} & \text{graph OT (GeneTrajectory), energy distance}
    \end{array}
    $$

    The difference is not cosmetic. Two of the three columns in the table at the end are computed by
    `eigenp_utils.single_cell.find_correlated_features` (`metrics=("pearson", "spearman", "wasserstein")`),
    and one of those metrics cannot tell a gene apart from a randomly cell-shuffled copy of itself.
    The rest of this notebook explains why, then builds up to the geometry-aware distances.

    Everything runs on synthetic data with a known answer, so every claim is checked against ground
    truth rather than taken on faith. The constructions carry over to any setting with **features
    measured on a set of points that carry a graph** (cells × genes, spots × genes, pixels × channels,
    users × items).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 0. The data: one matrix, two columns

    Single-cell data arrive as an expression matrix $X \in \mathbb{R}^{n \times p}$: **rows are cells**
    (`adata.obs`), **columns are genes** (`adata.var`), entry $X_{ig}$ is the (normalised) expression of gene
    $g$ in cell $i$. Every comparison in this notebook takes two columns,
    $$x = X_{:,A}, \qquad y = X_{:,B},$$
    and returns one number. The only question is *what else* the number is allowed to look at.

    Row order carries no information: a sequencer emits cells in arbitrary order (left panel). Sorting the
    rows by pseudotime (middle; trunk first, then the two arms L and R) is purely for our eyes, and it makes the
    structure visible. The right panel pulls out the two columns under comparison. The data are the synthetic
    bifurcation described in Part 2; pick any pair of genes below.

    Keep one fact in mind for Part 1: if we shuffle the rows of *both* columns together, every metric
    in this notebook is unchanged. If we shuffle *one* column alone, correlation sees it and 1D
    Wasserstein does not. Try gene B = `perm_of_…`, a copy of a trajectory gene with its cells shuffled.
    """)
    return


@app.cell
def _(genes_eval, mo):
    gene_a_dd = mo.ui.dropdown(genes_eval, value=genes_eval[0], label="gene A", searchable=True)
    _perm = [g for g in genes_eval if g.startswith("perm_of_")]
    gene_b_dd = mo.ui.dropdown(
        genes_eval, value=_perm[0] if _perm else genes_eval[1], label="gene B", searchable=True
    )
    mo.hstack([gene_a_dd, gene_b_dd], wrap=True, justify="start")
    return gene_a_dd, gene_b_dd


@app.cell
def _(
    BRANCH_COLORS,
    COL_A,
    COL_B,
    adata,
    fig_row,
    gene_a_dd,
    gene_b_dd,
    genes_eval,
    mo,
    np,
    plt,
):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    _noise = [g for g in adata.var_names if g.startswith("noise")][:15]
    _cols = list(genes_eval) + _noise
    _Xs = np.asarray(adata[:, _cols].X)
    _branch = adata.obs["branch"].to_numpy()
    _bcode = np.searchsorted(["L", "R", "trunk"], _branch)  # alphabetical codes for the colour strip
    _bcmap = ListedColormap([BRANCH_COLORS["L"], BRANCH_COLORS["R"], BRANCH_COLORS["trunk"]])
    _order = np.lexsort((adata.obs["t"].to_numpy(), np.vectorize({"trunk": 0, "L": 1, "R": 2}.get)(_branch)))
    _vmax = np.percentile(_Xs, 99)
    _ia, _ib = _cols.index(gene_a_dd.value), _cols.index(gene_b_dd.value)

    def _matrix_panel(rows, title):
        fig, (ax_s, ax) = plt.subplots(
            1, 2, figsize=(3.6, 4.4), sharey=True, gridspec_kw={"width_ratios": [1, 14], "wspace": 0.04}
        )
        ax_s.imshow(_bcode[rows, None], aspect="auto", cmap=_bcmap, vmin=0, vmax=2, interpolation="nearest")
        ax_s.set_xticks([])
        ax_s.set_ylabel(f"cells (n = {len(rows)})")
        ax.imshow(_Xs[rows], aspect="auto", cmap="Greys", vmin=0, vmax=_vmax, interpolation="nearest")
        for j, c, lab in ((_ia, COL_A, "A"), (_ib, COL_B, "B")):
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(rows), fill=False, edgecolor=c, lw=2.2))
            ax.text(j, -0.03 * len(rows), lab, color=c, ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"genes (p = {len(_cols)})")
        ax.set_title(title, fontsize=12, pad=18)
        return fig

    _fig1 = _matrix_panel(np.arange(adata.n_obs), "X, cells as acquired")
    _fig2 = _matrix_panel(_order, "rows sorted by pseudotime")

    _fig3, _axs = plt.subplots(1, 2, figsize=(3.2, 4.4), sharey=True, gridspec_kw={"wspace": 0.08})
    _ypos = np.arange(adata.n_obs)
    _bounds = np.flatnonzero(np.diff(np.vectorize({"trunk": 0, "L": 1, "R": 2}.get)(_branch[_order]))) + 0.5
    for _ax, _j, _c, _lab in ((_axs[0], _ia, COL_A, "x = X[:, A]"), (_axs[1], _ib, COL_B, "y = X[:, B]")):
        _v = _Xs[_order, _j]
        _ax.scatter(_v, _ypos, s=1.5, color=_c, alpha=0.25, rasterized=True)
        _k = max(adata.n_obs // 40, 5)
        _ax.plot(np.convolve(_v, np.ones(_k) / _k, mode="same"), _ypos, color=_c, lw=2)
        for _y0 in _bounds:
            _ax.axhline(_y0, color="0.6", lw=0.8, ls=":")
        _ax.set_title(_lab, color=_c, fontsize=11)
        _ax.set_xlabel("expression")
        _ax.tick_params(labelsize=9)
    _axs[0].set_ylim(adata.n_obs - 0.5, -0.5)
    _axs[0].set_yticks([])
    _fig3.suptitle("the two columns we compare", fontsize=12, y=1.02)

    _legend = " ".join(
        f'<span style="display:inline-block;width:.8em;height:.8em;background:{c};margin:0 .25em 0 .8em"></span>{b}'
        for b, c in BRANCH_COLORS.items()
    )
    mo.vstack([
        fig_row(_fig1, _fig2, _fig3, arrows=["sort rows (cells) by pseudotime", "take columns A and B"], min_px=220, max_px=340),
        mo.Html(f'<div style="font-size:.8rem;color:#666;text-align:center">colour strip = true position on the tree: {_legend}</div>'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1. Two vectors, three numbers

    Let $x, y \in \mathbb{R}^n$ be the expression of two genes over the same $n$ cells, and
    $z_x, z_y$ their z-scores (population SD, so $\tfrac1n\sum_i z_{x,i}^2 = 1$).

    **Pearson** is the mean product of the z-scores over the *observed* pairing:
    $$r = \tfrac1n \sum_i z_{x,i}\, z_{y,i}.$$

    **Spearman** is Pearson applied to ranks, $\rho_S = r\big(\operatorname{rank}(x), \operatorname{rank}(y)\big)$.
    Ranks are invariant to any strictly increasing transform, so $\rho_S$ measures monotone (not linear)
    association and is robust to outliers.

    **1D Wasserstein** treats $x$ and $y$ as two *bags of numbers*, not as paired measurements. With $n$
    samples each, the optimal transport pairs the $k$-th smallest with the $k$-th smallest:
    $$W_p^p(x, y) = \tfrac1n \sum_k \big|x_{(k)} - y_{(k)}\big|^p, \qquad
    W_1 = \int_{\mathbb{R}} |F_x(t) - F_y(t)|\,dt.$$
    Permuting the cells of $y$ leaves $W_p$ unchanged. **Wasserstein on values does not measure
    co-expression.**

    ### One identity that ties them together

    The sorted pairing is the coupling that *maximises* correlation (rearrangement inequality). Call that
    maximum $r_{\max} = \tfrac1n\sum_k z_{x,(k)} z_{y,(k)}$. Expanding the square for z-scored data:

    $$
    \underbrace{\tfrac1n\textstyle\sum_i (z_{x,i}-z_{y,i})^2}_{=\,2 - 2r}
    \;=\;
    \underbrace{W_2^2(z_x, z_y)}_{=\,2 - 2r_{\max}\ \text{(marginal mismatch)}}
    \;+\;
    \underbrace{2\,(r_{\max} - r)}_{\text{coupling mismatch}\ \ge 0}.
    $$

    Consequences:

    * $r \le r_{\max} = 1 - W_2^2/2$. If two genes have very different marginals (one bimodal, one
      log-normal), **Pearson cannot reach 1 even under perfect monotone co-regulation.** $W_2$ on
      z-scores is the ceiling Pearson is measured against.
    * Spearman transforms both marginals to the *same* uniform distribution, so (without ties)
      $W_2 = 0$, $r_{\max} = 1$, and all that remains is the coupling term. That is the precise sense
      in which Spearman "removes the distribution shape".
    * With **ties**, e.g. 70% zeros from dropout, rank marginals are no longer identical and Spearman's
      ceiling drops below 1. Zero-inflation turns Spearman into a partly *detection*-driven statistic.

    Use the controls to break each metric.
    """)
    return


@app.cell
def _(mo):
    relation_dd = mo.ui.dropdown(
        options=["linear", "monotone (exp)", "non-monotone (quadratic)", "independent"],
        value="monotone (exp)",
        label="relationship y = f(x)",
    )
    noise_sl = mo.ui.slider(0.0, 2.0, 0.05, value=0.3, label="noise SD", show_value=True)
    dropout_sl = mo.ui.slider(0.0, 0.9, 0.05, value=0.0, label="dropout (fraction set to 0)", show_value=True)
    shuffle_sw = mo.ui.switch(value=False, label="shuffle cells of y (destroy pairing)")
    mo.hstack([relation_dd, noise_sl, dropout_sl, shuffle_sw], wrap=True)
    return dropout_sl, noise_sl, relation_dd, shuffle_sw


@app.cell
def _(np, rankdata):
    def zscore_pop(v):
        """z-score with population SD (ddof=0) so mean(z**2) == 1 exactly."""
        v = np.asarray(v, dtype=float)
        sd = v.std()
        return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)

    def pair_metrics(x, y):
        """Pearson, Spearman, 1D W1/W2 on z-scores and the coupling decomposition."""
        zx, zy = zscore_pop(x), zscore_pop(y)
        sx, sy = np.sort(zx), np.sort(zy)
        r = float(np.mean(zx * zy))
        r_max = float(np.mean(sx * sy))
        rx, ry = zscore_pop(rankdata(x)), zscore_pop(rankdata(y))
        return {
            "pearson r": r,
            "spearman ρ": float(np.mean(rx * ry)),
            "spearman ceiling (sorted ranks)": float(np.mean(np.sort(rx) * np.sort(ry))),
            "W1(z_x, z_y)": float(np.mean(np.abs(sx - sy))),
            "W2²(z_x, z_y)": float(np.mean((sx - sy) ** 2)),
            "r_max = 1 − W2²/2": r_max,
            "coupling gap r_max − r": r_max - r,
        }

    return pair_metrics, zscore_pop


@app.cell
def _(dropout_sl, noise_sl, np, relation_dd, shuffle_sw):
    def simulate_pair(relation, noise, dropout, shuffle, n=600, seed=1):
        rng = np.random.default_rng(seed)
        x = rng.gamma(2.0, 1.0, n)
        # noise enters the latent driver, so f is applied to a noisy copy of x
        latent = x + noise * rng.standard_normal(n)
        y = {
            "linear": lambda v: v,
            "monotone (exp)": lambda v: np.exp(0.9 * v),
            "non-monotone (quadratic)": lambda v: (v - np.median(x)) ** 2,
            "independent": lambda v: rng.gamma(2.0, 1.0, v.size),
        }[relation](latent)
        y = y - min(y.min(), 0.0)
        if dropout > 0:
            x = np.where(rng.random(n) < dropout, 0.0, x)
            y = np.where(rng.random(n) < dropout, 0.0, y)
        if shuffle:
            y = rng.permutation(y)
        return x, y

    pair_x, pair_y = simulate_pair(
        relation_dd.value, noise_sl.value, dropout_sl.value, shuffle_sw.value
    )
    return pair_x, pair_y


@app.cell
def _(
    COL_A,
    COL_B,
    fig_row,
    mo,
    np,
    pair_metrics,
    pair_x,
    pair_y,
    pd,
    plt,
    zscore_pop,
):
    _m = pair_metrics(pair_x, pair_y)
    _f1, _a1 = plt.subplots(figsize=(3.6, 3.3))
    _a1.scatter(pair_x, pair_y, s=6, alpha=0.5, color="0.35")
    _a1.set(xlabel="gene x", ylabel="gene y", title="observed pairing")
    _f2, _a2 = plt.subplots(figsize=(3.6, 3.3))
    _q = np.linspace(0, 1, pair_x.size)
    _a2.plot(_q, np.sort(zscore_pop(pair_x)), label="z_x", color=COL_A)
    _a2.plot(_q, np.sort(zscore_pop(pair_y)), label="z_y", color=COL_B)
    _a2.fill_between(_q, np.sort(zscore_pop(pair_x)), np.sort(zscore_pop(pair_y)), color="0.85")
    _a2.set(xlabel="quantile u", ylabel="F⁻¹(u)", title="quantiles (area = W1)")
    _a2.legend(frameon=False)
    _f3, _a3 = plt.subplots(figsize=(3.6, 3.3))
    _a3.scatter(np.argsort(np.argsort(pair_x)), np.argsort(np.argsort(pair_y)), s=4, alpha=0.4, color="#2e8b57")
    _a3.set(xlabel="rank x", ylabel="rank y", title="rank pairing (Spearman)")
    mo.vstack([
        fig_row(_f1, _f2, _f3, min_px=230, max_px=360),
        pd.DataFrame(_m, index=["value"]).T.round(3),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What to try:

    * **monotone (exp)** → Spearman ≈ 0.95, Pearson ≈ 0.34. The gap is almost entirely the marginal
      term: `coupling gap` ≈ 0, `r_max` ≈ Pearson. Pearson is penalising *shape*, not co-regulation.
    * **non-monotone, noise 0** → $y$ is a deterministic function of $x$, yet Spearman is ≈ 0.3 and
      Pearson's value depends only on where the parabola's vertex sits relative to the bulk of a skewed
      $x$. Neither is a dependence measure (use distance correlation or mutual information for that).
    * **independent** → correlations ≈ 0 while $W_1 \approx 0$: by the Wasserstein yardstick two
      unrelated genes with similar histograms are "the same gene".
    * **shuffle on** → Pearson and Spearman collapse to ≈ 0; every Wasserstein number is *bit-identical*
      to the unshuffled case. This is the negative control that 1D-W1 fails by construction.
    * **dropout 0.7** → note `spearman ceiling` dropping below 1: ties at zero cap Spearman.

    ### Drag the outlier

    The red puck is one extra cell. Drag it around and watch Pearson swing while Spearman barely moves.
    (Built with `wigglystuff.ChartPuck`; the chart re-renders on every drag.)
    """)
    return


@app.cell
def _(ChartPuck, mo, np, pair_metrics):
    _rng = np.random.default_rng(7)
    puck_x = _rng.normal(0, 1, 60)
    puck_y = 0.6 * puck_x + 0.8 * _rng.normal(0, 1, 60)

    def _draw(ax, widget):
        _xo, _yo = widget.x[0], widget.y[0]
        _xx = np.r_[puck_x, _xo]
        _yy = np.r_[puck_y, _yo]
        _m = pair_metrics(_xx, _yy)
        _base = pair_metrics(puck_x, puck_y)
        ax.scatter(puck_x, puck_y, s=10, color="#3b6ea8", alpha=0.7)
        _b = np.polyfit(_xx, _yy, 1)
        _grid = np.linspace(-6, 6, 10)
        ax.plot(_grid, np.polyval(_b, _grid), color="#d1495b", lw=1)
        ax.set_title(
            f"Pearson {_m['pearson r']:+.2f} (was {_base['pearson r']:+.2f})\n"
            f"Spearman {_m['spearman ρ']:+.2f} (was {_base['spearman ρ']:+.2f})",
            fontsize=8,
        )
        ax.set_xlabel("gene x", fontsize=8)
        ax.set_ylabel("gene y", fontsize=8)
        ax.tick_params(labelsize=7)

    outlier_puck = mo.ui.anywidget(
        ChartPuck.from_callback(
            draw_fn=_draw, x_bounds=(-6, 6), y_bounds=(-6, 6), figsize=(3.4, 3.2), x=5.0, y=-5.0,
            puck_radius=12,
        )
    )
    outlier_puck
    return (outlier_puck,)


@app.cell
def _(mo, outlier_puck):
    mo.md(
        f"Outlier at ({outlier_puck.widget.x[0]:.2f}, {outlier_puck.widget.y[0]:.2f}). "
        "Pearson's influence function is unbounded (∝ z_x·z_y of the point); Spearman's is bounded "
        "because a rank can only move by n."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Numerical check of the decomposition

    The identity $2 - 2r = W_2^2(z_x,z_y) + 2(r_{\max}-r)$ is exact for any data. A cell that asserts it
    on random inputs (cheap insurance that the formulas above are not off by a factor):
    """)
    return


@app.cell
def _(np, pair_metrics):
    def check_decomposition(n_trials=200, seed=0):
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n_trials):
            n = int(rng.integers(5, 500))
            a = rng.lognormal(0, 1, n)
            b = a ** rng.uniform(0.2, 3) + rng.normal(0, rng.uniform(0, 2), n)
            m = pair_metrics(a, b)
            lhs = 2 - 2 * m["pearson r"]
            rhs = m["W2²(z_x, z_y)"] + 2 * m["coupling gap r_max − r"]
            worst = max(worst, abs(lhs - rhs))
            assert m["pearson r"] <= m["r_max = 1 − W2²/2"] + 1e-12
        return worst

    f"max |LHS − RHS| over 200 random pairs: {check_decomposition():.2e}"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2. Genes as distributions over cells

    The 1D Wasserstein above lives on the *expression axis*. A more useful object is the gene's
    **distribution over cells**: normalise the expression vector to a probability vector
    $a_g = x_g / \sum_i x_{g,i}$ on the cell set. Two genes are close if their mass sits in the same
    region of the manifold. "Same region" needs a ground metric $d(i, j)$ between cells, and that is
    where the cell graph enters.

    This is the core of **GeneTrajectory** (Qu et al., *Nat. Biotechnol.* 2025, KlugerLab):

    $$
    \begin{array}{ccccccc}
    \text{cells} & \xrightarrow{\text{diffusion map / PCA}} & \text{kNN graph} & \xrightarrow{\text{shortest paths}} & d_{\text{geo}}(i,j) & & \\[0.3em]
    & & \downarrow \text{\scriptsize{k-means coarsening}} & & \downarrow & & \\[0.3em]
    \text{genes} & \xrightarrow{\text{normalise}} & a_g \in \Delta^{M-1} & \xrightarrow{\text{EMD}} & W_1(a_g, a_h) & \xrightarrow{\text{gene diffusion map}} & \text{gene trajectories}
    \end{array}
    $$

    $$W_1(a, b) = \min_{\pi \ge 0,\ \pi\mathbf 1 = a,\ \pi^\top \mathbf 1 = b} \ \sum_{ij} \pi_{ij}\, d_{\text{geo}}(i,j).$$

    Note the contrast with Part 1: the *coupling* is now between **cells** (where to move mass), not
    between expression values, and the cost is **graph distance**. Pairing is back in the game, via
    geometry.

    ### Synthetic ground truth

    Cells lie on a bifurcating tree (trunk $t\in[0,1]$, arms L and R for $t\in[1,2]$). Each
    trajectory gene has a Gaussian bump of expression around a peak position on the tree, so the
    **true gene–gene distance is the tree distance between peaks**. Counts are Poisson with
    log-normal library size, then `normalize_total` + `log1p`. Two built-in controls:

    * **positive control, twin genes**: same peak and width as a parent gene, 6× amplitude. A good
      "same place" metric puts them next to their parent.
    * **negative control, permuted genes**: a copy of a trajectory gene with its cells shuffled. Same
      histogram, no spatial structure. A good metric must *not* place it next to its source.

    Plus 300 unstructured noise genes so that PCA/kNN see a realistic background.
    """)
    return


@app.cell
def _(IN_WASM, mo):
    seed_num = mo.ui.number(start=0, stop=999, value=0, label="random seed")
    n_cells_sl = mo.ui.slider(500, 4000, 250, value=1000 if IN_WASM else 2000, label="cells", show_value=True)
    n_traj_sl = mo.ui.slider(20, 100, 10, value=30 if IN_WASM else 60, label="trajectory genes", show_value=True)
    mo.hstack([seed_num, n_cells_sl, n_traj_sl])
    return n_cells_sl, n_traj_sl, seed_num


@app.cell
def _(ad, np, sc):
    def tree_distance(b1, t1, b2, t2):
        """Distance on a Y-shaped tree. Branch 0 = trunk (t∈[0,1]); 1, 2 = branches (t∈[1,2])."""
        b1, b2 = np.asarray(b1), np.asarray(b2)
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        same_lineage = (b1 == b2) | (b1 == 0) | (b2 == 0)
        return np.where(same_lineage, np.abs(t1 - t2), t1 + t2 - 2.0)

    def simulate_bifurcation(n_cells=2000, n_traj=60, n_noise=300, seed=0):
        rng = np.random.default_rng(seed)
        cb = rng.choice([0, 1, 2], n_cells, p=[0.34, 0.33, 0.33])
        ct = np.where(cb == 0, rng.uniform(0, 1, n_cells), rng.uniform(1, 2, n_cells))

        gb = rng.choice([0, 1, 2], n_traj, p=[0.34, 0.33, 0.33])
        gt = np.where(gb == 0, rng.uniform(0, 1, n_traj), rng.uniform(1, 2, n_traj))
        order = np.lexsort((gt, gb))
        gb, gt = gb[order], gt[order]
        gw = rng.uniform(0.15, 0.3, n_traj)
        ga = rng.lognormal(0, 0.5, n_traj)

        twin_parent = np.arange(0, n_traj, max(n_traj // 5, 1))
        perm_parent = np.arange(n_traj // 10 + 1, n_traj, max(n_traj // 5, 1))

        pb = np.r_[gb, gb[twin_parent]]
        pt = np.r_[gt, gt[twin_parent]]
        pw = np.r_[gw, gw[twin_parent]]
        pa = np.r_[ga, 6.0 * ga[twin_parent]]
        d_cg = tree_distance(cb[:, None], ct[:, None], pb[None], pt[None])
        rate = 0.05 + pa * np.exp(-(d_cg**2) / (2 * pw**2))
        lib = rng.lognormal(np.log(3), 0.3, n_cells)[:, None]
        counts_struct = rng.poisson(lib * rate)
        counts_perm = counts_struct[:, perm_parent][rng.permutation(n_cells)]
        counts_noise = rng.poisson(lib * rng.lognormal(-1, 1, n_noise)[None])
        counts = np.c_[counts_struct, counts_perm, counts_noise].astype(np.float32)

        names = (
            [f"g{i:02d}_{'TLR'[b]}{t:.2f}" for i, (b, t) in enumerate(zip(gb, gt))]
            + [f"twin_of_g{p:02d}" for p in twin_parent]
            + [f"perm_of_g{p:02d}" for p in perm_parent]
            + [f"noise{i:03d}" for i in range(n_noise)]
        )
        kind = (["trajectory"] * n_traj + ["twin"] * len(twin_parent)
                + ["permuted"] * len(perm_parent) + ["noise"] * n_noise)
        parent = list(range(n_traj)) + list(twin_parent) + list(perm_parent) + [-1] * n_noise

        adata = ad.AnnData(counts)
        adata.var_names = names
        adata.obs["branch"] = np.array(["trunk", "L", "R"])[cb]
        adata.obs["t"] = ct
        adata.var["kind"] = kind
        adata.var["parent"] = parent
        adata.var["peak_branch"] = np.r_[gb, gb[twin_parent], gb[perm_parent], -np.ones(n_noise, int)]
        adata.var["peak_t"] = np.r_[gt, gt[twin_parent], gt[perm_parent], np.full(n_noise, np.nan)]
        adata.layers["counts"] = counts.copy()
        if sc is not None:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.pca(adata, n_comps=20, random_state=0)
        else:
            # WASM: the same three steps with numpy / scikit-learn
            from sklearn.decomposition import PCA

            lib_size = counts.sum(1, keepdims=True)
            adata.X = np.log1p(np.divide(counts * 1e4, lib_size, out=np.zeros_like(counts), where=lib_size > 0))
            adata.obsm["X_pca"] = PCA(n_components=20, random_state=0).fit_transform(adata.X)
        return adata

    return simulate_bifurcation, tree_distance


@app.cell
def _(
    IN_WASM,
    ensure_neighbors,
    n_cells_sl,
    n_traj_sl,
    seed_num,
    simulate_bifurcation,
):
    adata = simulate_bifurcation(n_cells=n_cells_sl.value, n_traj=n_traj_sl.value, seed=seed_num.value)
    if IN_WASM:
        # scanpy's UMAP connectivities need numba; use a plain symmetric kNN graph
        from sklearn.neighbors import kneighbors_graph

        _knn = kneighbors_graph(adata.obsm["X_pca"], n_neighbors=15, mode="connectivity")
        adata.obsp["connectivities"] = _knn.maximum(_knn.T).tocsr()
    else:
        ensure_neighbors(adata, n_neighbors=15, use_rep="X_pca")
    genes_eval = adata.var_names[adata.var["kind"] != "noise"].tolist()
    adata
    return adata, genes_eval


@app.cell
def _(BRANCH_COLORS, adata, fig_row, np, plt):
    _f1, _a1 = plt.subplots(figsize=(4.2, 3.6))
    _pc = adata.obsm["X_pca"]
    for _b, _c in BRANCH_COLORS.items():
        _m = adata.obs["branch"].to_numpy() == _b
        _a1.scatter(_pc[_m, 0], _pc[_m, 1], s=3, color=_c, label=_b, alpha=0.6)
    _a1.legend(frameon=False, markerscale=4)
    _a1.set(xlabel="PC1", ylabel="PC2", title="cells (PCA), true arm")
    _f2, _a2 = plt.subplots(figsize=(4.2, 3.6))
    _traj = adata[:, adata.var["kind"] == "trajectory"]
    _ord = np.lexsort((adata.obs["t"].to_numpy(), adata.obs["branch"].map({"trunk": 0, "L": 1, "R": 2}).to_numpy()))
    _a2.imshow(_traj.X[_ord].T, aspect="auto", cmap="magma", interpolation="nearest")
    _a2.set(xlabel="cells (trunk → L → R, by t)", ylabel="genes (by peak)", title="trajectory genes")
    fig_row(_f1, _f2, min_px=280, max_px=440)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Coarsening and the ground metric

    Exact EMD on $n$ cells costs roughly $O(n^3 \log n)$ per gene pair, so GeneTrajectory coarsens the
    cell graph to $M$ metacells (k-means) first. We do the same: metacell expression is the mean over
    member cells; the metacell graph is a symmetrised kNN graph on metacell centroids in PCA space with
    Euclidean edge lengths; the ground metric is the shortest-path distance on that graph.
    (GeneTrajectory builds the kNN graph on diffusion-map coordinates; PCA is used here for simplicity.)
    """)
    return


@app.cell
def _(IN_WASM, mo):
    n_meta_sl = mo.ui.slider(50, 400, 25, value=100 if IN_WASM else 200, label="metacells M", show_value=True)
    k_meta_sl = mo.ui.slider(3, 20, 1, value=8, label="metacell kNN k", show_value=True)
    mo.hstack([n_meta_sl, k_meta_sl])
    return k_meta_sl, n_meta_sl


@app.cell
def _(
    KMeans,
    NearestNeighbors,
    connected_components,
    laplacian,
    np,
    shortest_path,
    sp,
):
    def build_metacell_graph(Z, X, n_meta=200, k=8, seed=0):
        """
        Coarsen cells into metacells and build the metacell ground geometry.

        Returns a dict with
          labels  : (n,) metacell id per cell
          centroids: (M, d) metacell centroids in Z
          Xm      : (M, g) mean expression per metacell
          W       : (M, M) sparse symmetric kNN graph, entries = Euclidean edge length
          D_geo   : (M, M) shortest-path distances on W
          L_pinv  : (M, M) pseudo-inverse of the Laplacian with conductance 1/length
        """
        km = KMeans(n_meta, n_init=3, random_state=seed).fit(Z)
        labels = km.labels_
        M = labels.max() + 1
        H = sp.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(M, len(labels)))
        H = sp.diags(1.0 / np.asarray(H.sum(1)).ravel()) @ H
        Xm = np.asarray(H @ X)
        centroids = np.asarray(H @ Z)

        kk = k
        while True:
            dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(centroids).kneighbors(centroids)
            rows = np.repeat(np.arange(M), kk)
            W = sp.csr_matrix((dist[:, 1:].ravel(), (rows, idx[:, 1:].ravel())), shape=(M, M))
            W = W.maximum(W.T)
            if connected_components(W, directed=False)[0] == 1:
                break
            kk += 1
        D_geo = shortest_path(W, directed=False)
        C = W.copy()
        C.data = 1.0 / C.data
        L_pinv = np.linalg.pinv(laplacian(C).toarray())
        dL = np.diag(L_pinv)
        R = dL[:, None] + dL[None, :] - 2 * L_pinv  # effective resistance
        return dict(labels=labels, centroids=centroids, Xm=Xm, W=W, D_geo=D_geo, L_pinv=L_pinv, R=R, k_used=kk)

    return (build_metacell_graph,)


@app.cell
def _(adata, build_metacell_graph, genes_eval, k_meta_sl, n_meta_sl, np):
    meta = build_metacell_graph(
        adata.obsm["X_pca"],
        np.asarray(adata[:, genes_eval].X),
        n_meta=n_meta_sl.value,
        k=k_meta_sl.value,
    )
    # gene distributions over metacells: columns sum to 1
    gene_dist = meta["Xm"] / meta["Xm"].sum(0, keepdims=True)
    f"{meta['Xm'].shape[0]} metacells, kNN k used = {meta['k_used']} (raised automatically until connected)"
    return gene_dist, meta


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The distances we compare

    All are gene × gene matrices over the same evaluation genes (trajectory + twins + permuted).

    | name | object compared | ground metric | formula |
    |---|---|---|---|
    | `pearson` | expression vectors over cells | none | $1 - r$ (`find_correlated_features`) |
    | `spearman` | expression vectors over cells | none | $1 - \rho_S$ (`find_correlated_features`) |
    | `pearson_graph` | graph-smoothed vectors | kNN graph (one diffusion step) | $1 - r(\hat P x, \hat P y)$ (`use_graph=True`) |
    | `W1_values` | value histograms | $\lvert z - z'\rvert$ on the real line | 1D $W_1$ on z-scores (`find_correlated_features`) |
    | `W1_graph` | distributions over metacells | shortest path | exact EMD (POT `ot.emd2`) |
    | `diffusion_EMD` | distributions over metacells | implicit, multiscale diffusion | $\sum_k 2^{k\alpha}\lVert (T^{2^{k+1}} - T^{2^k})\delta\rVert_1$ (Tong et al. 2021) |
    | `energy_geo` | distributions over metacells | shortest path | $\sqrt{-\delta^\top D_{\text{geo}}\,\delta}$ |
    | `energy_euclid` | distributions over metacells | Euclidean in PCA | $\sqrt{-\delta^\top D_{\text{euc}}\,\delta}$ |
    | `energy_resist` | distributions over metacells | effective resistance | $\sqrt{2\,\delta^\top L^{+}\delta}$ |

    with $\delta = a_g - a_h$. The three `energy_*` rows are the same estimator with different ground
    metrics; the next part explains why the resistance version is special.
    """)
    return


@app.cell
def _(linprog, np, ot, pdist, sp, squareform):
    def graph_w1_matrix(P, D, W=None):
        """
        Exact pairwise W1 between columns of P (each a probability vector) under ground cost D.

        Without POT (WASM) each entry is the equivalent min-cost-flow (Beckmann) LP on the
        graph W whose shortest paths define D, solved with HiGHS: exact for a shortest-path
        metric, with |E| instead of M² variables.
        """
        g = P.shape[1]
        out = np.zeros((g, g))
        if ot is not None:
            def w1(a, b):
                return ot.emd2(a, b, D)
        else:
            Wu = sp.triu(W).tocoo()
            E = Wu.nnz
            B = sp.csr_matrix(
                (np.r_[np.ones(E), -np.ones(E)], (np.r_[Wu.row, Wu.col], np.r_[np.arange(E), np.arange(E)])),
                shape=(W.shape[0], E),
            )
            A_eq, cost = sp.hstack([B, -B]).tocsr(), np.r_[Wu.data, Wu.data]

            def w1(a, b):
                return linprog(cost, A_eq=A_eq, b_eq=a - b, bounds=(0, None), method="highs").fun
        for i in range(g):
            for j in range(i + 1, g):
                out[i, j] = out[j, i] = w1(P[:, i], P[:, j])
        return out

    def transport_lp(a, b, D):
        """POT-free exact OT: plan T minimising <T, D> s.t. T1 = a, Tᵀ1 = b, T ≥ 0 (HiGHS). Returns (T, cost)."""
        n, m = len(a), len(b)
        A_eq = sp.vstack([sp.kron(sp.eye(n), np.ones((1, m))), sp.kron(np.ones((1, n)), sp.eye(m))])
        res = linprog(D.ravel(), A_eq=A_eq, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
        return res.x.reshape(n, m), res.fun

    def diffusion_emd_matrix(P, W, n_scales=8, alpha=0.5):
        """
        Diffusion EMD (Tong et al., ICML 2021): an L1 distance between multiscale diffusion
        wavelet coefficients, equivalent (up to constants) to W1 under a diffusion-induced metric.
        Cost is O(n_scales · M² · g) plus one pdist, instead of one LP per gene pair.
        """
        sigma = np.median(W.data)
        K = W.copy()
        K.data = np.exp(-(K.data / sigma) ** 2)
        K = K.toarray() + np.eye(W.shape[0])
        T = (K / K.sum(1, keepdims=True)).T  # column-stochastic: acts on distributions
        levels = [P]
        Tk = T.copy()
        for _ in range(n_scales + 1):
            levels.append(Tk @ P)
            Tk = Tk @ Tk
        blocks = [2.0 ** (k * alpha) * (levels[k + 1] - levels[k]) for k in range(n_scales + 1)]
        blocks.append(2.0 ** ((n_scales + 1) * alpha) * levels[-1])
        emb = np.vstack(blocks)  # (features, g)
        return squareform(pdist(emb.T, metric="cityblock"))

    def energy_matrix(P, D):
        """Energy distance sqrt(-(a-b)ᵀ D (a-b)) for all column pairs; also returns the raw quadratic form."""
        G = P.T @ D @ P
        G = 0.5 * (G + G.T)
        diag = np.diag(G)
        e2 = -(diag[:, None] + diag[None, :] - 2 * G)  # −(a−b)ᵀ D (a−b)
        return np.sqrt(np.clip(e2, 0, None)), e2

    return diffusion_emd_matrix, energy_matrix, graph_w1_matrix, transport_lp


@app.cell
def _(
    adata,
    diffusion_emd_matrix,
    energy_matrix,
    find_correlated_features,
    gene_dist,
    genes_eval,
    graph_w1_matrix,
    meta,
    mo,
    np,
    pd,
    squareform,
    pdist,
    time,
):
    def compute_all_distances():
        sub = adata[:, genes_eval].copy()
        g = len(genes_eval)
        out, secs = {}, {}

        _t = time.perf_counter()
        mats = {m: np.zeros((g, g)) for m in ("pearson", "spearman", "W1_values")}
        for i, gname in enumerate(genes_eval):
            res = find_correlated_features(sub, gname, metrics=("pearson", "spearman", "wasserstein"))
            res = res.reindex(genes_eval)
            mats["pearson"][i] = 1 - res["pearson"].to_numpy()
            mats["spearman"][i] = 1 - res["spearman"].to_numpy()
            mats["W1_values"][i] = res["wasserstein"].to_numpy()
        for m in mats:
            out[m] = 0.5 * (mats[m] + mats[m].T)
        secs["pearson / spearman / W1_values (all three)"] = time.perf_counter() - _t

        _t = time.perf_counter()
        pg = np.zeros((g, g))
        for i, gname in enumerate(genes_eval):
            res = find_correlated_features(sub, gname, metrics=("pearson",), use_graph=True)
            pg[i] = 1 - res.reindex(genes_eval)["pearson"].to_numpy()
        out["pearson_graph"] = 0.5 * (pg + pg.T)
        secs["pearson_graph"] = time.perf_counter() - _t

        _t = time.perf_counter()
        out["W1_graph"] = graph_w1_matrix(gene_dist, meta["D_geo"], W=meta["W"])
        secs["W1_graph"] = time.perf_counter() - _t

        _t = time.perf_counter()
        out["diffusion_EMD"] = diffusion_emd_matrix(gene_dist, meta["W"])
        secs["diffusion_EMD"] = time.perf_counter() - _t

        _t = time.perf_counter()
        out["energy_geo"], e2_geo = energy_matrix(gene_dist, meta["D_geo"])
        out["energy_euclid"], _ = energy_matrix(gene_dist, squareform(pdist(meta["centroids"])))
        out["energy_resist"], _ = energy_matrix(gene_dist, meta["R"])
        secs["energy (all three)"] = time.perf_counter() - _t
        return out, secs, float(e2_geo.min())

    dists, dist_secs, min_e2_geo = compute_all_distances()
    pd.Series(dist_secs, name="seconds").round(2).to_frame()
    return dists, min_e2_geo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Look at one pair

    Pick two genes. Left: their distributions over metacells (PCA layout; marker size ∝ mass). Right:
    the optimal transport plan for `W1_graph`, drawn as segments between metacells, width ∝ mass moved.
    Try a gene and its `perm_of_` copy, a gene and its `twin_of_`, and two genes on different branches.
    """)
    return


@app.cell
def _(gene_a_dd, gene_b_dd, mo):
    # the same two dropdowns as in Part 0; changing either one updates both places
    mo.hstack([gene_a_dd, gene_b_dd], wrap=True, justify="start")
    return


@app.cell
def _(
    COL_A,
    COL_B,
    dists,
    fig_row,
    gene_a_dd,
    gene_b_dd,
    gene_dist,
    genes_eval,
    meta,
    mo,
    np,
    ot,
    pd,
    plt,
    transport_lp,
):
    _ia, _ib = genes_eval.index(gene_a_dd.value), genes_eval.index(gene_b_dd.value)
    _a, _b = gene_dist[:, _ia], gene_dist[:, _ib]
    _plan = ot.emd(_a, _b, meta["D_geo"]) if ot is not None else transport_lp(_a, _b, meta["D_geo"])[0]
    _xy = meta["centroids"][:, :2]
    _fig_l, _ax_l = plt.subplots(figsize=(4.4, 3.8))
    _fig_r, _ax_r = plt.subplots(figsize=(4.4, 3.8))
    _axes = (_ax_l, _ax_r)
    _r, _c = meta["W"].nonzero()
    for _ax in _axes:
        _ax.plot(np.c_[_xy[_r, 0], _xy[_c, 0]].T, np.c_[_xy[_r, 1], _xy[_c, 1]].T, color="0.88", lw=0.5, zorder=0)
    _s = 3000 / len(_a)
    _axes[0].scatter(*_xy.T, s=_a * len(_a) * _s / 3, color=COL_A, alpha=0.55, label="A")
    _axes[0].scatter(*_xy.T, s=_b * len(_b) * _s / 3, color=COL_B, alpha=0.55, label="B")
    _axes[0].legend(frameon=False)
    _axes[0].set(title="mass of gene A and gene B", xlabel="PC1", ylabel="PC2")
    _src, _dst = np.nonzero(_plan > 1e-4)
    _wts = _plan[_src, _dst]
    for _i, _j, _w in zip(_src, _dst, _wts):
        if _i != _j:
            _axes[1].plot(*np.c_[_xy[_i], _xy[_j]], color="#2e8b57", lw=0.5 + 40 * _w, alpha=0.6)
    _axes[1].scatter(*_xy.T, s=4, color="0.4")
    _axes[1].set(title="OT plan (mass moved)", xlabel="PC1")
    _row = {m: dists[m][_ia, _ib] for m in dists}
    _med = {m: np.median(dists[m][_ia][np.arange(len(genes_eval)) != _ia]) for m in dists}
    _tab = pd.DataFrame({"d(A,B)": _row, "median d(A,·)": _med})
    _tab["ratio"] = _tab["d(A,B)"] / _tab["median d(A,·)"]
    mo.vstack([fig_row(_fig_l, _fig_r, min_px=280, max_px=440), _tab.round(3)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `ratio < 1` means "closer than a typical gene". For the default pair (gene vs its cell-shuffled copy)
    `W1_values` reports a ratio near 0, i.e. *nearly identical*; every graph-aware metric reports
    roughly a typical distance, because the permuted copy is spread uniformly over the manifold.

    ## Part 3. W1 versus the energy distance, from first principles

    ### Both are flows on the graph

    On a graph with edge lengths $\ell_e$ and shortest-path ground metric, Kantorovich–Rubinstein duality
    gives the **Beckmann** form of $W_1$: route a flow $J$ along edges whose divergence is
    $\delta = a - b$, and pay $\ell_e$ per unit per edge,
    $$W_1(a,b) = \min_{J:\ BJ = \delta}\ \sum_e \ell_e\, |J_e| .$$
    Square the per-edge cost and you get **electrical flow**, where $\ell_e$ plays the role of a
    resistance. Thomson's principle gives the minimum in closed form:
    $$\min_{J:\ BJ = \delta}\ \sum_e \ell_e\, J_e^2 \;=\; \delta^\top L^{+} \delta, \qquad
    L = B\,\mathrm{diag}(1/\ell)\,B^\top .$$
    The energy distance with the effective-resistance metric $R_{ij} = (e_i - e_j)^\top L^{+}(e_i - e_j)$ is
    $$\mathcal E_R(a,b) = 2\,\mathbb E\,R(X,Y) - \mathbb E\,R(X,X') - \mathbb E\,R(Y,Y') = -\delta^\top R\,\delta = 2\,\delta^\top L^{+}\delta ,$$
    where the middle step uses $\mathbf 1^\top\delta = 0$. So:

    $$
    \begin{array}{ccc}
    W_1 = \min \lVert \ell \odot J \rVert_1 & \xleftarrow{\ \text{same constraint } BJ=\delta\ } & \tfrac12\,\mathcal E_R = \min \lVert \sqrt{\ell} \odot J \rVert_2^2 \\[0.3em]
    \downarrow \text{\scriptsize{LP, sparse flows, one path}} & & \downarrow \text{\scriptsize{linear solve, flow spreads over all parallel paths}} \\[0.3em]
    \text{metric in } \delta & & \text{squared Hilbert norm } \lVert\delta\rVert_{H^{-1}}^2 = \textstyle\sum_k (u_k^\top\delta)^2/\lambda_k
    \end{array}
    $$

    The spectral form on the bottom right says resistance-energy is a **low-pass filter**: it weights
    graph-Fourier components of $\delta$ by $1/\lambda_k$, so smooth large-scale differences dominate and
    local wiggles are discounted. That makes it cheap (one pseudo-inverse, then every pair is a
    quadratic form) and Hilbertian (you can do kernel methods, PCA, barycentres in closed form), at the
    price of being quadratic in small displacements.

    The next cell checks both identities numerically on the actual metacell graph: an LP for the
    Beckmann problem versus POT's EMD, and an explicit electrical flow versus $\delta^\top L^+\delta$.
    """)
    return


@app.cell
def _(gene_dist, linprog, meta, np, ot, sp, transport_lp):
    def verify_flow_identities(i=0, j=1):
        W = sp.triu(meta["W"]).tocoo()
        M, E = meta["W"].shape[0], W.nnz
        B = sp.csr_matrix(
            (np.r_[np.ones(E), -np.ones(E)], (np.r_[W.row, W.col], np.r_[np.arange(E), np.arange(E)])),
            shape=(M, E),
        )
        ell = W.data
        delta = gene_dist[:, i] - gene_dist[:, j]
        # Beckmann LP: J = J⁺ − J⁻, minimise Σ ℓ (J⁺ + J⁻) s.t. B(J⁺ − J⁻) = δ
        lp = linprog(np.r_[ell, ell], A_eq=sp.hstack([B, -B]), b_eq=delta, bounds=(0, None), method="highs")
        w1_emd = (
            ot.emd2(gene_dist[:, i], gene_dist[:, j], meta["D_geo"])
            if ot is not None
            else transport_lp(gene_dist[:, i], gene_dist[:, j], meta["D_geo"])[1]
        )
        # Electrical flow: potentials φ = L⁺δ, J_e = (φ_u − φ_v)/ℓ_e
        phi = meta["L_pinv"] @ delta
        J = (phi[W.row] - phi[W.col]) / ell
        flow_energy = float(np.sum(ell * J**2))
        return {
            "W1 via Beckmann LP": lp.fun,
            "W1 via EMD on shortest paths": w1_emd,
            "Σ ℓ J² of electrical flow": flow_energy,
            "δᵀ L⁺ δ": float(delta @ meta["L_pinv"] @ delta),
            "flow conservation error |BJ − δ|∞": float(np.abs(B @ J - delta).max()),
        }

    verify_flow_identities()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Saturation: why correlation is not a distance on the manifold

    On a path graph (a 1D trajectory), put two Gaussian bumps of width $w$ at separation $s$. Pearson
    between the two profiles drops to its floor once the bumps stop overlapping. It cannot tell
    "neighbouring stages" from "opposite ends of the trajectory". $W_1 = s$ exactly (translation), and keeps
    growing. The energy distance is $\mathcal E = 2\int (F_a - F_b)^2$ on the line: quadratic, $\propto s^2/w$,
    while $s \ll w$, then linear, $\to 2s$, once the bumps separate. Slide the width to see when the
    correlation horizon kicks in.
    """)
    return


@app.cell
def _(mo):
    shift_sl = mo.ui.slider(0.0, 0.8, 0.01, value=0.2, label="separation s", show_value=True)
    width_sl = mo.ui.slider(0.01, 0.15, 0.005, value=0.05, label="bump width w", show_value=True)
    mo.hstack([shift_sl, width_sl])
    return shift_sl, width_sl


@app.cell
def _(COL_A, COL_B, fig_row, mo, np, plt, shift_sl, width_sl):
    def path_metrics(s, w, n=400):
        xs = np.linspace(0, 1, n)
        dx = xs[1] - xs[0]
        a = np.exp(-((xs - (0.5 - s / 2)) ** 2) / (2 * w**2))
        b = np.exp(-((xs - (0.5 + s / 2)) ** 2) / (2 * w**2))
        a, b = a / a.sum(), b / b.sum()
        Fa, Fb = np.cumsum(a), np.cumsum(b)
        return xs, a, b, {
            "pearson": float(np.corrcoef(a, b)[0, 1]),
            "W1": float(np.sum(np.abs(Fa - Fb)) * dx),
            "energy": float(2 * np.sum((Fa - Fb) ** 2) * dx),
        }

    _grid = np.linspace(0, 0.8, 81)
    _curves = {k: np.array([path_metrics(s, width_sl.value)[3][k] for s in _grid]) for k in ("pearson", "W1", "energy")}
    _xs, _a, _b, _now = path_metrics(shift_sl.value, width_sl.value)

    # check the 1D identity  E = 2∫(F_a − F_b)² = −δᵀ|x−x'|δ  on the same grid
    _D = np.abs(_xs[:, None] - _xs[None])
    _delta = _a - _b
    _identity_err = abs(_now["energy"] - float(-_delta @ _D @ _delta))

    _fig_l, _ax_l = plt.subplots(figsize=(4.2, 3.3))
    _fig_r, _ax_r = plt.subplots(figsize=(4.2, 3.3))
    _axes = (_ax_l, _ax_r)
    _axes[0].plot(_xs, _a, color=COL_A, label="A")
    _axes[0].plot(_xs, _b, color=COL_B, label="B")
    _axes[0].set(title="two genes on a 1D trajectory", xlabel="pseudotime", yticks=[])
    _axes[1].plot(_grid, _curves["W1"], label="W1", color="#2e8b57")
    _axes[1].plot(_grid, _curves["energy"], label="energy E", color="#8a5a9e")
    _axes[1].plot(_grid, 1 - _curves["pearson"], label="1 − Pearson", color="#d1495b")
    _axes[1].axvline(shift_sl.value, color="0.5", ls=":")
    _axes[1].set(xlabel="separation s", title="distance vs separation")
    _axes[1].legend(frameon=False, loc="lower right", fontsize=10)
    mo.vstack([
        fig_row(_fig_l, _fig_r, min_px=280, max_px=440),
        mo.md(
            f"At s = {shift_sl.value:.2f}: W1 = {_now['W1']:.3f}, E = {_now['energy']:.4f}, "
            f"Pearson = {_now['pearson']:+.3f}. Discretisation check of E = −δᵀDδ: |error| = {_identity_err:.1e}."
        ),
    ])
    return


@app.cell(hide_code=True)
def _(min_e2_geo, mo):
    mo.md(rf"""
    ### When is the energy distance a distance at all?

    $\mathcal E_D(a,b) = -\delta^\top D\,\delta \ge 0$ for all mass-balanced $\delta$ **iff** $D$ is
    *conditionally negative definite* (a metric of negative type). Euclidean distance qualifies, and so do
    $|x-y|^\beta$ for $0<\beta\le2$, tree metrics, and effective resistance. Then $\sqrt{{\mathcal E}}$ is a
    metric. Two things to know:

    * **Squared Euclidean ($\beta = 2$) is degenerate.** $-\delta^\top D\,\delta = 2\lVert \mu_a - \mu_b\rVert^2$:
      only the means are compared. Two genes with the same centroid but different spread look identical.
      Keep $\beta < 2$.
    * **Shortest-path metrics on graphs with cycles are not guaranteed to be of negative type.** On this
      graph the most negative $\mathcal E_{{\text{{geo}}}}$ over all gene pairs is `{min_e2_geo:.2e}`. Zero or
      positive means no violation *here*, but nothing guarantees that on another graph. Resistance is
      of negative type on every connected graph, which is a second reason to prefer it over
      geodesics for energy-type statistics.

    **Caveat (established result).** On large random-geometric kNN graphs, *point-to-point* effective
    resistance converges to $1/d_i + 1/d_j$ and loses all global information (von Luxburg, Radl & Hein,
    JMLR 2014). Diffuse gene distributions average much of this out, but the practical defaults are a
    coarsened graph (as here) or a regularised $L + \varepsilon I$.

    The same estimator is the **E-distance** that scPerturb (Peidli et al., *Nat. Methods* 2024) uses
    between *cell populations* (uniform weights on two sets of cells, Euclidean in PCA). Here it is applied
    to *genes*, i.e. expression-weighted distributions over a shared cell set. Same formula, different
    question.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4. Scorecard against ground truth

    For each metric:

    * **ρ(GT)**: Spearman correlation between the metric and the true tree distance between gene peaks,
      over all trajectory-gene pairs. Rank-based, so invariant to monotone rescaling, e.g. $\mathcal E$ vs
      $\sqrt{\mathcal E}$.
    * **twin rank**: median rank of the twin among its parent's neighbours (1 = nearest). Lower is better.
    * **perm→source rank**: median rank of the source gene among the permuted copy's neighbours.
      **Rank 1 means the metric confuses a gene with a cell-shuffled copy of itself.** Higher is better.
    * **kNN-5 overlap**: average fraction of each trajectory gene's 5 true nearest genes recovered.
    """)
    return


@app.cell
def _(adata, dists, genes_eval, np, pd, tree_distance):
    def scorecard(dists, var, genes, k=5):
        v = var.loc[genes]
        traj = np.where(v["kind"].to_numpy() == "trajectory")[0]
        twins = np.where(v["kind"].to_numpy() == "twin")[0]
        perms = np.where(v["kind"].to_numpy() == "permuted")[0]
        pb, pt = v["peak_branch"].to_numpy(), v["peak_t"].to_numpy()
        gt = tree_distance(pb[traj, None], pt[traj, None], pb[None, traj], pt[None, traj])
        iu = np.triu_indices(len(traj), 1)
        gt_nn = np.argsort(gt + np.diag(np.full(len(traj), np.inf)), axis=1)[:, :k]
        parent = v["parent"].to_numpy()
        rows = {}
        for name, D in dists.items():
            Dt = D[np.ix_(traj, traj)]
            rho = pd.Series(Dt[iu]).corr(pd.Series(gt[iu]), method="spearman")
            nn = np.argsort(Dt + np.diag(np.full(len(traj), np.inf)), axis=1)[:, :k]
            overlap = np.mean([len(set(a) & set(b)) / k for a, b in zip(nn, gt_nn)])

            def rank_of(row_idx, target_idx):
                d = D[row_idx].copy()
                d[row_idx] = np.inf
                return int(np.sum(d < d[target_idx]) + 1)

            twin_r = np.median([rank_of(parent[t], t) for t in twins]) if len(twins) else np.nan
            perm_r = np.median([rank_of(p, parent[p]) for p in perms]) if len(perms) else np.nan
            rows[name] = {"ρ(GT)": rho, "kNN-5 overlap": overlap, "twin rank": twin_r, "perm→source rank": perm_r}
        return pd.DataFrame(rows).T

    score_df = scorecard(dists, adata.var, genes_eval)
    score_df.round(3).sort_values("ρ(GT)", ascending=False)
    return (score_df,)


@app.cell
def _(fig_row, plt, score_df):
    _s = score_df.sort_values("ρ(GT)")
    _f1, _a1 = plt.subplots(figsize=(4.4, 3.6))
    _a1.barh(_s.index, _s["ρ(GT)"], color="#3b6ea8")
    _a1.set(xlabel="Spearman ρ with true distance", xlim=(min(0, _s["ρ(GT)"].min()) - 0.05, 1))
    _f2, _a2 = plt.subplots(figsize=(4.4, 3.6))
    _a2.barh(_s.index, _s["perm→source rank"], color="#d1495b")
    _a2.set(xlabel="source rank for its permuted copy\n(1 = fooled)", xscale="log")
    fig_row(_f1, _f2, min_px=280, max_px=440)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gene embeddings

    GeneTrajectory's last step is a diffusion map *on genes*, using the gene–gene distance. With a
    good metric, genes line up along the tree: trunk, then two arms. Pick a metric to embed (adaptive
    Gaussian kernel, $k=5$, $\alpha=1$ density normalisation; first two non-trivial eigenvectors).
    """)
    return


@app.cell
def _(dists, mo):
    embed_dd = mo.ui.dropdown(list(dists), value="W1_graph", label="metric to embed")
    embed_dd
    return (embed_dd,)


@app.cell
def _(adata, dists, embed_dd, fig_row, genes_eval, np, plt):
    def diffusion_map_from_distance(D, k=5, n_comp=2):
        D = np.asarray(D, float)
        sig = np.sort(D, axis=1)[:, k]
        sig = np.where(sig > 0, sig, np.median(sig[sig > 0]))
        K = np.exp(-(D**2) / (sig[:, None] * sig[None, :]))
        q = K.sum(1)
        K = K / np.outer(q, q)  # alpha = 1
        d = K.sum(1)
        S = K / np.sqrt(np.outer(d, d))
        vals, vecs = np.linalg.eigh(S)
        order = np.argsort(vals)[::-1]
        psi = vecs[:, order] / np.sqrt(d)[:, None]
        return psi[:, 1 : n_comp + 1] * vals[order][1 : n_comp + 1]

    _v = adata.var.loc[genes_eval]
    _emb = diffusion_map_from_distance(dists[embed_dd.value])
    _fig, _ax = plt.subplots(figsize=(4.8, 4.2))
    _cmaps = {0: "Greys", 1: "BuGn", 2: "Purples"}
    _kinds = _v["kind"].to_numpy()
    for _b, _cm in _cmaps.items():
        _m = (_v["peak_branch"].to_numpy() == _b) & (_kinds == "trajectory")
        _ax.scatter(_emb[_m, 0], _emb[_m, 1], c=_v["peak_t"].to_numpy()[_m], cmap=_cm, vmin=-0.3, vmax=2.1, s=30, edgecolor="k", lw=0.3)
    _tw = _kinds == "twin"
    _pm = _kinds == "permuted"
    _ax.scatter(_emb[_tw, 0], _emb[_tw, 1], marker="^", s=50, facecolor="none", edgecolor="#2e8b57", label="twin (pos. ctrl)")
    _ax.scatter(_emb[_pm, 0], _emb[_pm, 1], marker="x", s=50, color="#e08a00", label="permuted (neg. ctrl)")
    _ax.legend(frameon=False, fontsize=8)
    _ax.set(xlabel="DC1", ylabel="DC2")
    _ax.set_title(f"gene diffusion map: {embed_dd.value}\n(grey = trunk, green = L, purple = R; darker = later)", fontsize=10)
    fig_row(_fig, min_px=280, max_px=480)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    | metric | answers | sees geometry? | fooled by cell shuffling? | cost for $g$ genes | notes |
    |---|---|---|---|---|---|
    | Pearson | linear co-variation in the same cells | no | no | $O(n g^2)$ (one GEMM) | capped by marginal mismatch, $r\le 1-W_2^2/2$; outlier-sensitive |
    | Spearman | monotone co-variation | no | no | $O(n\log n\, g + n g^2)$ | ties from dropout cap it below 1 |
    | Pearson on smoothed data | co-variation after a low-pass filter | locally (one diffusion step) | no | $O(\text{nnz}\,g + n g^2)$ | cheap, often good; saturates beyond the filter radius |
    | 1D $W_1$ on values | same value histogram | **no** | **yes, by construction** | $O(n\log n\, g + n g^2)$ | a shape-similarity score, not co-expression |
    | graph $W_1$ (GeneTrajectory) | same place on the manifold, in path length | yes, exact | no | $O(g^2)$ LPs of size $M$ | needs coarsening; non-Hilbertian; the reference metric |
    | diffusion EMD | ≈ $W_1$ under a diffusion metric | yes, multiscale | no | $O(K M^2 g)$ + one $L_1$ pdist | near-$W_1$ quality without per-pair LPs |
    | energy, resistance | same place, $H^{-1}$ / low-frequency | yes, spectral | no | one $L^+$ + $O(M^2 g)$ | Hilbertian; quadratic for small shifts; resistance degenerates on huge kNN graphs |
    | energy, geodesic / Euclidean | same place, kernel-mean embedding | yes / ambient | no | $O(M^2 g)$ | geodesic is not guaranteed negative type; never use squared Euclidean |

    Practical recommendations, stated as hypotheses to test on your own data:

    1. For **"what is co-regulated with X"**, use Spearman or Pearson on graph-smoothed data. Drop
       `wasserstein` from `find_correlated_features` for this question, or rename it to what it is:
       distribution-shape similarity.
    2. For **"order genes along a trajectory / find gene programs over space"**, use graph-OT. Exact
       EMD on ~200–1000 metacells is fine up to a few hundred genes; past that use diffusion EMD or
       resistance-energy, which both reduce to one embedding plus a vectorised pairwise distance.
    3. **Always run the two controls**: a cell-permuted copy (must be far from its source) and an
       amplitude-rescaled twin (must be near its parent). They are a few lines of code and catch
       most metric-choice errors.
    4. The normalisation $a_g = x_g / \sum x_g$ turns a constant background (ambient RNA, baseline
       expression) into mass spread over the whole manifold. Distances between low-specificity genes then
       contract toward each other. Consider background subtraction or thresholding before forming $a_g$.

    ### References

    * Qu R., *et al.* Gene trajectory inference for single-cell data by optimal transport metrics. *Nat. Biotechnol.* (2025).
    * Tong A., *et al.* Diffusion Earth Mover's Distance and Distribution Embeddings. *ICML* (2021).
    * Székely G. J. & Rizzo M. L. Energy statistics: a class of statistics based on distances. *J. Stat. Plan. Inference* (2013).
    * Peidli S., *et al.* scPerturb: harmonized single-cell perturbation data. *Nat. Methods* (2024). (E-distance.)
    * von Luxburg U., Radl A. & Hein M. Hitting and commute times in large random neighborhood graphs. *JMLR* (2014).
    * Santambrogio F. *Optimal Transport for Applied Mathematicians*, ch. 4 (Beckmann formulation). Birkhäuser (2015).
    """)
    return


if __name__ == "__main__":
    app.run()
