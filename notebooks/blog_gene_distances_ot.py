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
#     "scanpy",
#     "pot",
#     "wigglystuff",
#     "eigenp-utils[single-cell] @ git+https://github.com/eigenP/utils.git@main",
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
def _():
    # marimo-lens: agent-managed Lens cell
    # Lens is a local pairing tool (installed in the pixi dev environment). Where marimo-lens is not
    # installed, e.g. on molab, this cell does nothing.
    import importlib.util as _ilu

    import marimo as _mo

    if _ilu.find_spec("marimo_lens") is not None:
        from marimo_lens import Lens as _Lens

        _lens = _Lens()
        _mo.output.append(_lens)
    return


@app.cell
def _():
    import contextlib
    import io
    import time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import anndata as ad
    import scanpy as sc
    import ot
    import wigglystuff
    from scipy.linalg import orthogonal_procrustes
    from scipy.optimize import linprog
    from scipy.sparse.csgraph import connected_components, shortest_path
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import rankdata, wasserstein_distance
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    from wigglystuff import ScatterWidget

    from eigenp_utils.plotting_utils import set_plotting_style
    from eigenp_utils.single_cell import (
        ensure_neighbors,
        find_correlated_features,
        multiscale_coarsening,
    )

    set_plotting_style()
    # Colour roles, used consistently in every figure:
    #   blue / orange : gene A / gene B, and nothing else
    #   green         : transport (OT plans and flows), and nothing else
    #   greys + hatch : one-off contrasts (original vs shuffled, metric vs metric, structure)
    #   Purples       : expression levels (a nod to in situ stains)
    #   viridis       : any other quantity (pseudotime, graph distance)
    pal = {
        "a": "#2a78d6", "b": "#eb6834", "flow": "#1baf7a",
        "dark": "#3a3a3a", "mid": "#8c8c8c", "light": "#c8c8c8", "faint": "#e6e6e6",
    }
    return (
        KMeans,
        Line2D,
        NearestNeighbors,
        Path,
        ScatterWidget,
        ad,
        connected_components,
        contextlib,
        ensure_neighbors,
        find_correlated_features,
        io,
        linprog,
        multiscale_coarsening,
        np,
        orthogonal_procrustes,
        ot,
        pal,
        pd,
        pdist,
        plt,
        rankdata,
        sc,
        shortest_path,
        sp,
        squareform,
        time,
        wasserstein_distance,
        wigglystuff,
    )


@app.cell(hide_code=True)
def _(np):
    import matplotlib.ticker as mticker


    def log_ticks(axis):
        """
        Ticks for a log-scaled axis that make the scale obvious: labelled major ticks plus visible,
        unlabelled minor ticks that bunch up towards each decade. Minor tick size comes from
        scientific.mplstyle (2.5 x 1 pt). Short ranges (two decades or less)
        label 1, 2 and 5 of each decade; longer ranges label only the decades. Call after plotting.
        """
        lo, hi = sorted(axis.get_view_interval())
        span = np.log10(hi / max(lo, 1e-12))
        if span > 2:
            major, minor = (1.0,), tuple(float(k) for k in range(2, 10))
        else:
            major, minor = (1.0, 2.0, 5.0), (3.0, 4.0, 6.0, 7.0, 8.0, 9.0)
        axis.set_major_locator(mticker.LogLocator(base=10, subs=major))
        axis.set_minor_locator(mticker.LogLocator(base=10, subs=minor))
        axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(mticker.NullFormatter())

    return (log_ticks,)


@app.cell(hide_code=True)
def _(io, mo, np, plt):
    import base64
    from matplotlib.transforms import Bbox

    # Responsive figure layout. Every figure is shown as one or more images in a flex row: side by
    # side on wide screens, stacked below 700 px (and any arrows between them turn to point down).
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

    def fig_to_img(fig, bbox_inches="tight", dpi=160):
        """Render a Matplotlib figure (or a crop of it) to a width-responsive <img>."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches=bbox_inches)
        return f'<img alt="figure" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"/>'

    def fig_row(*items, arrows=None, min_px=260, max_px=460):
        """
        Lay figures (or HTML strings / marimo objects) out in a row that stacks on narrow screens.

        `arrows` (optional) holds one caption per gap; the arrow points right on wide screens and
        turns to point down once the row stacks vertically (viewport < 700 px).
        """
        arrows = list(arrows or [])
        parts = []
        for i, item in enumerate(items):
            if hasattr(item, "savefig"):
                html = fig_to_img(item)
                plt.close(item)
            elif isinstance(item, str):
                html = item
            else:
                html = mo.as_html(item).text
            parts.append(f'<div class="rr-item">{html}</div>')
            if i < len(arrows):
                parts.append(f'<div class="rr-arrow"><span class="rr-glyph">&#10230;</span><small>{arrows[i]}</small></div>')
        style = f"--rr-min:{min_px}px;--rr-max:{max_px}px"
        return mo.Html(_ROW_CSS + f'<div class="rr-row" style="{style}">' + "".join(parts) + "</div>")

    def fig_panels(fig, min_px=260, max_px=460, pad=0.08):
        """
        Show a multi-panel figure as one image per column of panels, laid out by `fig_row`, so the
        columns stack on a phone instead of shrinking. Colourbars stay with the panel to their left;
        figure-level text and legends (column headers, a shared legend) stay with the column they
        sit over. A figure with a single column of panels is shown whole.
        """
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        visible = [a for a in fig.axes if a.get_visible()]

        def _is_cbar(a):
            return a.get_label() == "<colorbar>"

        def _centre(a):
            q = a.get_position()
            return (q.x0 + q.x1) / 2

        cols = []
        for a in sorted((a for a in visible if not _is_cbar(a)), key=lambda a: a.get_position().x0):
            for col in cols:
                if abs(_centre(a) - _centre(col[0])) < 0.05:
                    col.append(a)
                    break
            else:
                cols.append([a])
        if len(cols) < 2:
            return fig_row(fig, min_px=min_px, max_px=max(max_px, 720))

        centres = [np.mean([_centre(a) for a in col]) for col in cols]
        boxes = [[a.get_tightbbox(renderer) for a in col] for col in cols]
        for a in visible:
            if _is_cbar(a):
                left = [i for i, col in enumerate(cols) if max(b.get_position().x1 for b in col) <= a.get_position().x0 + 1e-6]
                boxes[left[-1] if left else 0].append(a.get_tightbbox(renderer))
        figure_level = list(fig.texts) + list(fig.legends) + ([fig._suptitle] if fig._suptitle is not None else [])
        for artist in figure_level:
            ext = artist.get_window_extent(renderer)
            xc = fig.transFigure.inverted().transform(((ext.x0 + ext.x1) / 2, 0))[0]
            boxes[int(np.argmin([abs(xc - c) for c in centres]))].append(ext)
        to_inches = fig.dpi_scale_trans.inverted()
        imgs = [fig_to_img(fig, bbox_inches=Bbox.union(b).transformed(to_inches).padded(pad)) for b in boxes]
        plt.close(fig)
        return fig_row(*imgs, min_px=min_px, max_px=max_px)

    return fig_panels, fig_row


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
    # How far apart are two genes? Pearson, Spearman, Wasserstein, and graph and tree optimal transport

    "Find me genes in my dataset that are expressed like *BRCA1*" sounds like one question. But there are many ways to interpret _"like"_, and each similarity metric below answers a different question:

    $$
    \begin{array}{ccc}
    \textbf{Question} & & \textbf{Metric family} \\[0.4em]
    \text{Do A and B rise and fall in the \emph{same cells}?} & \xrightarrow{\text{uses the pairing } (x_i,y_i)} & \text{Pearson, Spearman} \\[0.6em]
    \text{Do A and B have the same \emph{value histogram}?} & \xrightarrow{\text{ignores the pairing}} & \text{1D Wasserstein on values} \\[0.6em]
    \text{Are A and B expressed in the same \emph{region of cell space}?} & \xrightarrow{\text{uses a cell--cell distance}} & \textcolor{#1baf7a}{\text{graph optimal transport } (W_1)}
    \end{array}
    $$

    "Cell space" means the space the cells live in after embedding (PCA, a kNN graph, a diffusion map):
    nearby cells in the embedding (manifold) are in similar states.

    **The short answer**, which the rest of the notebook backs up:

    | if your question is… | use | because |
    |---|---|---|
    | which genes rise and fall with X *in the same cells* (co-regulation) | Spearman | it ignores histogram shape, which Pearson counts against the pair (Part 1) |
    | which genes are expressed *in the same region of cell space* as X, and how far apart | graph $W_1$, or tree $W_1$ for many genes | it keeps measuring distance after expression stops overlapping (Parts 3–4) |
    | which genes have the same *value histogram* as X | 1D $W_1$ on values | but this is not co-expression: it cannot tell a gene from a cell-shuffled copy (Part 2) |

    **Roadmap.**

    0. **The data**: one expression matrix, two of its columns.
    1. **Pearson and Spearman** measure co-variation in the same cells, and why Pearson is pulled down by
       distribution shape.
    2. **1D Wasserstein on values** compares histograms only, so it fails the shuffle test.
    3. **Graph optimal transport** moves expression *between cells*, at a cost given by a cell–cell
       distance. This is the distance that answers the third question. (When the cells are simply
       ordered along a line, such as pseudotime, it is the 1D Wasserstein *along that ordering*: the
       right-hand column of the first figure.)
    4. **Multiscale coarsening** picks the metacell scale for (3) from the data, and **tree $W_1$**
       on the same hierarchy gives a close approximation of (3) at a tiny fraction of the cost.
    5. **Scorecard**: all five metrics against a known answer.

    /// note | How claims are checked
    Everything runs on synthetic data with a known answer, so each claim is checked against ground
    truth rather than taken on faith. The constructions carry over to any setting with features
    measured on points that carry a graph (cells × genes, spots × genes, pixels × channels).
    ///

    ### Three views of the same two genes

    A picture first. Here the cells can be *ordered* along a 1D coordinate (for example pseudotime),
    and gene A is expressed early. In the **top row** it is compared with gene B, expressed a little
    later along the same coordinate. In the **bottom row** it is compared with a copy of itself whose
    values were shuffled across cells: all biology is gone, but the histogram is untouched. Each
    column is one metric family's view, and the panel title is its verdict.
    """)
    return


@app.cell(hide_code=True)
def _(fig_panels, log_ticks, mo, np, pal, plt, wasserstein_distance):
    def _three_views():
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 1, 600))

        def gene(center, width=0.07, amp=3.0):
            bump = amp * np.exp(-((t - center) ** 2) / (2 * width**2))
            return np.clip(bump + rng.normal(0, 0.3, t.size), 0, None)

        A = gene(0.35)
        rows = [("gene B", gene(0.55)), ("shuffled A", rng.permutation(A))]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
        bins = np.linspace(0, max(A.max(), rows[0][1].max()) + 0.1, 30)
        for row, (name, Y) in enumerate(rows):
            ax = axes[row, 0]
            ax.scatter(A, Y, s=6, color=pal["dark"], alpha=0.45, lw=0)
            ax.set(xlabel="gene A expression [a.u.]", ylabel=f"{name} expression [a.u.]")
            ax.set_title(f"Pearson r = {np.corrcoef(A, Y)[0, 1]:+.2f}", fontsize=11)

            ax = axes[row, 1]
            ax.hist(A, bins=bins, histtype="step", color=pal["a"], lw=2, label="gene A")
            ax.hist(Y, bins=bins, histtype="step", color=pal["b"], lw=2, ls="--", label=name)
            ax.set(xlabel="expression [a.u.]", ylabel="number of cells (log scale)", yscale="log")
            log_ticks(ax.yaxis)
            ax.set_title(f"$W_1$(values) = {wasserstein_distance(A, Y):.2f}", fontsize=11)
            ax.legend(loc="upper right")

            ax = axes[row, 2]
            ax.scatter(t, A, s=5, color=pal["a"], alpha=0.5, lw=0, label="gene A")
            ax.scatter(t, Y, s=5, color=pal["b"], alpha=0.5, lw=0, label=name)
            ax.set(xlabel="pseudotime, cell order [0–1]", ylabel="expression [a.u.]")
            ax.set_title(f"$W_1$(pseudotime) = {wasserstein_distance(t, t, A, Y):.2f}", fontsize=11)
            ax.legend(loc="upper right", markerscale=2.5)
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        for col, question in enumerate(["same cells?", "same histogram?", "same place in cell space?"]):
            box = axes[0, col].get_position()
            fig.text((box.x0 + box.x1) / 2, 0.965, question, ha="center", fontsize=14, fontweight="bold")
        return fig

    fig_panels(_three_views())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Bottom row.** The shuffled copy has exactly the same histogram as A, so value-Wasserstein (middle)
    calls them identical. Pearson (left) correctly calls them unrelated.

    **Top row.** A and B are neighbours along the cell order, but they are rarely high in the same cell,
    so Pearson calls them unrelated. It would say the same if B were expressed at the far end.

    Only the right-hand column, which moves expression *along the cell order*, both rejects the
    shuffled copy and says how far B is from A. The rest of the notebook makes each column precise.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 0. The data: one matrix, two columns

    Single-cell data arrive as an expression matrix $X \in \mathbb{R}^{n \times p}$: **rows are cells**
    (`adata.obs`), **columns are genes** (`adata.var`), and $X_{ig}$ is the normalised expression of gene
    $g$ in cell $i$. Every comparison in this notebook takes two columns,
    $$x = X_{:,A}, \qquad y = X_{:,B},$$
    and returns one number. The only question is *what else* the number is allowed to look at.

    Row order carries no information: cells come off the sequencer in arbitrary order (left). Sorting the
    rows by a 1D coordinate (here the position $t$ along the drawn trajectory; in practice a pseudotime or
    any other cell ordering) is only for our eyes, but it makes the structure visible (middle): each
    trajectory gene lights up in its own band of cells. The right panel pulls out the two columns under
    comparison, gene A in blue and gene B in orange. Pick any two genes below; the same two selectors
    drive the transport figures in Part 3.

    The data are simulated with a known answer (Part 3 shows how, from a shape you can draw; the default
    is a horseshoe-shaped trajectory). The gene names say what each gene is:

    * `g07_t0.35`: a **trajectory gene**, expressed in a band of cells around position $t = 0.35$ along
      the trajectory. The true distance between two trajectory genes is the distance between their
      peaks along the trajectory.
    * `twin_of_g07`: the **positive control**, same peak as `g07` at 6× the amplitude. A good metric puts
      it next to its parent.
    * `perm_of_g07`: the **negative control**, a copy of `g07` with its cells shuffled. A good metric
      must *not* put it next to its source.
    * `bg…` and `noise…`: background genes that give the cells their structure, and pure noise.

    Keep one fact in mind for Part 1: shuffling the rows of *both* columns together changes no metric in
    this notebook. Shuffling *one* column alone is what correlation notices and 1D Wasserstein does not.
    Try gene B = `perm_of_…`.
    """)
    return


@app.cell
def _(gene_a_dd, gene_b_dd, mo):
    # the same two dropdowns as in Part 3; changing either one updates both places
    mo.hstack([gene_a_dd, gene_b_dd], wrap=True, justify="start")
    return


@app.cell(hide_code=True)
def _(adata, fig_row, gene_a_dd, gene_b_dd, genes_eval, mo, np, pal, plt):
    from matplotlib.patches import Rectangle

    _cols = list(genes_eval) + [g for g in adata.var_names if g.startswith("noise")][:15]
    _Xs = np.asarray(adata[:, _cols].X)
    _t = adata.obs["t"].to_numpy()
    _order = np.argsort(_t)
    _vmax = np.percentile(_Xs, 99)
    _ia, _ib = _cols.index(gene_a_dd.value), _cols.index(gene_b_dd.value)

    def _matrix_panel(rows, title):
        fig, (ax_s, ax) = plt.subplots(
            1, 2, figsize=(3.6, 4.4), sharey=True, gridspec_kw={"width_ratios": [1, 14], "wspace": 0.04}
        )
        # left strip: each cell's position t along the 1D coordinate
        ax_s.imshow(_t[rows, None], aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        ax_s.set_xticks([])
        ax_s.set_ylabel(f"cells, n = {len(rows)}")
        ax.imshow(_Xs[rows], aspect="auto", cmap="Purples", vmin=0, vmax=_vmax, interpolation="nearest")
        for j, c, lab in ((_ia, pal["a"], "A"), (_ib, pal["b"], "B")):
            ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(rows), fill=False, edgecolor=c, lw=2.2))
            ax.text(j, -0.03 * len(rows), lab, color=c, ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"genes, p = {len(_cols)}")
        ax.set_title(title, fontsize=12, pad=30)
        return fig

    _fig1 = _matrix_panel(np.arange(adata.n_obs), "X, cells as acquired")
    _fig2 = _matrix_panel(_order, "rows sorted by t")

    _fig3, _axs = plt.subplots(1, 2, figsize=(3.2, 4.4), sharey=True, gridspec_kw={"wspace": 0.08})
    _ypos = np.arange(adata.n_obs)
    _k = max(adata.n_obs // 40, 5)
    for _ax, _j, _c, _lab in ((_axs[0], _ia, pal["a"], "x = X[:, A]"), (_axs[1], _ib, pal["b"], "y = X[:, B]")):
        _v = _Xs[_order, _j]
        _ax.scatter(_v, _ypos, s=1.5, color=_c, alpha=0.25, rasterized=True, lw=0)
        _ax.plot(np.convolve(_v, np.ones(_k) / _k, mode="same"), _ypos, color=_c, lw=2)
        _ax.set_title(_lab, color=_c, fontsize=11)
        _ax.set_xlabel("expression\n[log(1 + CP10k)]")
        _ax.tick_params(labelsize=9)
    _axs[0].set_ylim(adata.n_obs - 0.5, -0.5)
    _axs[0].set_yticks([])
    _axs[0].set_ylabel("cells, sorted by t")
    _fig3.suptitle("the two columns we compare", fontsize=12, y=1.02)

    mo.vstack([
        fig_row(_fig1, _fig2, _fig3, arrows=["sort rows (cells) by their 1D coordinate t", "take columns A and B"], min_px=220, max_px=340),
        mo.Html('<div style="font-size:.8rem;color:#666;text-align:center">colour strip = position t along the 1D coordinate (dark = start, yellow = end); line = running mean</div>'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1. Pearson and Spearman: co-variation in the same cells

    Let $x, y \in \mathbb{R}^n$ be the expression of two genes over the same $n$ cells, and $z_x, z_y$
    their z-scores (population SD, so $\tfrac1n\sum_i z_{x,i}^2 = 1$).

    * **Pearson** is the mean product of the z-scores over the *observed* pairing,
      $r = \tfrac1n \sum_i z_{x,i}\, z_{y,i}$.
    * **Spearman** is Pearson applied to ranks, $\rho_S = r\big(\operatorname{rank}(x), \operatorname{rank}(y)\big)$.
      Ranks do not change under any strictly increasing transform, so $\rho_S$ measures monotone (not
      linear) association.

    The controls simulate $y = f(x) + \text{noise}$ with skewed (gamma-distributed) $x$, as for
    expression data. The two panels show the data as each metric sees it: raw values for Pearson, ranks
    for Spearman. With the default **monotone (exp)** link, the raw cloud is curved and fanned out while
    the rank cloud is a tight diagonal. That difference *is* the gap between the two numbers.
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
    dropout_sl = mo.ui.slider(0.0, 0.9, 0.05, value=0.1, label="dropout (fraction set to 0)", show_value=True)
    mo.hstack([relation_dd, noise_sl, dropout_sl], wrap=True)
    return dropout_sl, noise_sl, relation_dd


@app.cell(hide_code=True)
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
            "W1(z_x, z_y)": float(np.mean(np.abs(sx - sy))),
            "W2²(z_x, z_y)": float(np.mean((sx - sy) ** 2)),
            "r_max = 1 − W2²/2": r_max,
            "coupling gap r_max − r": r_max - r,
        }

    return pair_metrics, zscore_pop


@app.cell(hide_code=True)
def _(dropout_sl, noise_sl, np, relation_dd):
    def simulate_pair(relation, noise, dropout, shuffle=False, n=600, seed=1):
        rng = np.random.default_rng(seed)
        x = rng.gamma(2.0, 1.0, n)
        # noise enters the latent driver, so f is applied to a noisy copy of x
        latent = x + noise * rng.standard_normal(n)
        y = {
            "linear": lambda v: v,
            "monotone (exp)": lambda v: np.exp(0.5 * v),
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

    pair_x, pair_y = simulate_pair(relation_dd.value, noise_sl.value, dropout_sl.value)
    pair_y_shuffled = simulate_pair(relation_dd.value, noise_sl.value, dropout_sl.value, shuffle=True)[1]
    return pair_x, pair_y, pair_y_shuffled


@app.cell(hide_code=True)
def _(fig_panels, mo, pair_metrics, pair_x, pair_y, pal, plt, rankdata):
    _m = pair_metrics(pair_x, pair_y)
    _fig, _axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    _axes[0].scatter(pair_x, pair_y, s=6, alpha=0.5, color=pal["dark"], lw=0)
    _axes[0].set(xlabel="gene x expression [a.u.]", ylabel="gene y expression [a.u.]")
    _axes[0].set_title(f"raw values: Pearson r = {_m['pearson r']:+.2f}", fontsize=11)
    _axes[1].scatter(rankdata(pair_x), rankdata(pair_y), s=6, alpha=0.5, color=pal["dark"], lw=0)
    _axes[1].set(xlabel="rank of x among cells", ylabel="rank of y among cells")
    _axes[1].set_title(f"ranks: Spearman ρ = {_m['spearman ρ']:+.2f}", fontsize=11)
    for _ax in _axes:
        _ax.set_box_aspect(1)
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Where Pearson's shortfall comes from

    Compare Pearson with the *best possible* pairing of the same values: sort both vectors and pair the
    $k$-th smallest values. This maximises the correlation (rearrangement inequality); call the maximum
    $r_{\max}$. For z-scored data the distance from perfect agreement splits exactly in two:

    $$
    \underbrace{\tfrac1n\textstyle\sum_i (z_{x,i}-z_{y,i})^2}_{=\,2 - 2r}
    \;=\;
    \underbrace{W_2^2(z_x, z_y)}_{\text{marginal mismatch}}
    \;+\;
    \underbrace{2\,(r_{\max} - r)}_{\text{coupling mismatch}\ \ge 0}.
    $$

    * The **marginal** term depends only on the two histograms. It is a Wasserstein distance between
      the z-scores, and Part 2 is about it.
    * The **coupling** term depends on *which cells* carry which values. Only this part is about
      co-regulation.

    The bars apply the split to raw values (Pearson) and to ranks (Spearman). Ranks give both genes
    the same uniform histogram, so for Spearman the marginal term vanishes when there are no ties.
    That is the exact sense in which Spearman "removes distribution shape".
    """)
    return


@app.cell(hide_code=True)
def _(fig_panels, mo, pair_metrics, pair_x, pair_y, pal, plt, rankdata):
    def _split(x, y):
        m = pair_metrics(x, y)
        return m["W2²(z_x, z_y)"], 2 * m["coupling gap r_max − r"]

    _rows = {"Pearson (raw values)": _split(pair_x, pair_y), "Spearman (ranks)": _split(rankdata(pair_x), rankdata(pair_y))}
    _fig, _ax = plt.subplots(figsize=(10, 2.4))
    for _k, (_name, (_marg, _coup)) in enumerate(_rows.items()):
        _ax.barh(_k, _marg, color=pal["dark"], height=0.6, label="marginal mismatch $W_2^2$" if _k == 0 else None)
        _ax.barh(_k, _coup, left=_marg, color=pal["light"], edgecolor=pal["dark"], hatch="///", lw=0, height=0.6,
                 label="coupling mismatch $2(r_{max}-r)$" if _k == 0 else None)
        _ax.text(_marg + _coup + 0.02, _k, f"2 − 2r = {_marg + _coup:.2f}", va="center", fontsize=10)
    _ax.set_yticks(range(len(_rows)), list(_rows))
    _ax.invert_yaxis()
    _ax.set_xlim(0, max(sum(v) for v in _rows.values()) * 1.3 + 0.05)
    _ax.set_xlabel("2 − 2r, distance from perfect agreement [SD²] (0 = identical z-scores)")
    _ax.legend(loc=(1.02, 0.25))
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the default **monotone (exp)** link, almost all of Pearson's bar is the solid marginal part: the
    genes are perfectly co-ordered, but $e^{x}$ has a heavier tail than $x$, so their z-scores cannot
    line up. Pearson is penalising *shape*, not a lack of co-regulation. Two more settings to try:

    * **dropout 0.7**: a solid segment appears in Spearman's bar too. Ties at zero make the rank
      histograms unequal and cap Spearman below 1, so zero-inflation partly turns Spearman into a
      detection statistic.
    * **non-monotone (quadratic), noise 0**: $y$ is a deterministic function of $x$, yet both bars are
      long. Neither metric measures general dependence.

    /// tip | Part 1 in one line
    Use Spearman for co-variation: it removes histogram shape, which Pearson silently counts against
    the pair.
    ///

    ## Part 2. 1D Wasserstein: comparing value histograms

    **1D Wasserstein** treats $x$ and $y$ as two *bags of numbers*. On the real line, optimal transport
    pairs the $k$-th smallest value of one with the $k$-th smallest of the other:

    $$W_1(x, y) = \tfrac1n \sum_k \big|x_{(k)} - y_{(k)}\big| = \int_{\mathbb{R}} |F_x(t) - F_y(t)|\,dt .$$

    The figure shows this definition on z-scores. **Left:** $W_1$ is the area between the two quantile
    functions. **Middle and right:** 30 random cells drawn as lines between two number lines, $z_x$ on top
    and $z_y$ below. The middle panel joins each cell's own two values, the pairing Pearson uses. The
    right panel joins values by rank, the pairing Wasserstein uses, and no cell identity survives in it.
    For the default monotone pair the two panels nearly coincide, because the cells already come almost
    sorted. Switch to **independent**: the middle panel's lines cross in every direction, while the
    right panel, and $W_1$, barely change.
    """)
    return


@app.cell(hide_code=True)
def _(fig_panels, mo, np, pair_metrics, pair_x, pair_y, pal, plt, zscore_pop):
    _zx, _zy = zscore_pop(pair_x), zscore_pop(pair_y)
    _w1 = pair_metrics(pair_x, pair_y)["W1(z_x, z_y)"]
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    _u = (np.arange(_zx.size) + 0.5) / _zx.size
    _axes[0].fill_between(_u, np.sort(_zx), np.sort(_zy), color=pal["faint"], lw=0)
    _axes[0].plot(_u, np.sort(_zx), color=pal["a"], lw=2, label="gene x")
    _axes[0].plot(_u, np.sort(_zy), color=pal["b"], lw=2, ls="--", label="gene y")
    _axes[0].set(xlabel="quantile u, fraction of cells [0–1]", ylabel="z-score [SD] (top 1% clipped)")
    _axes[0].set_ylim(np.quantile(np.r_[_zx, _zy], 0.001) - 0.2, np.quantile(np.r_[_zx, _zy], 0.99) + 0.5)
    _axes[0].set_title(f"shaded area = $W_1$ = {_w1:.2f}", fontsize=11)
    _axes[0].legend(loc="upper left")

    _rng = np.random.default_rng(0)
    _idx = _rng.choice(_zx.size, 30, replace=False)
    _sx, _sy = _zx[_idx], _zy[_idx]
    _lo, _hi = np.quantile(np.r_[_zx, _zy], [0.005, 0.995])
    for _ax, _top, _bot, _title in [
        (_axes[1], _sx, _sy, "observed pairing (cell i ↔ cell i)"),
        (_axes[2], np.sort(_sx), np.sort(_sy), "sorted pairing (k-th ↔ k-th)"),
    ]:
        for _a, _b in zip(_top, _bot):
            _ax.plot([_a, _b], [1, 0], color=pal["mid"], lw=0.8, alpha=0.7)
        _ax.scatter(_top, np.ones_like(_top), color=pal["a"], s=22, zorder=3, edgecolor="white", lw=0.2)
        _ax.scatter(_bot, np.zeros_like(_bot), color=pal["b"], s=22, zorder=3, edgecolor="white", lw=0.2)
        _ax.set_yticks([0, 1], ["gene y", "gene x"])
        _ax.set_xlim(_lo, _hi)
        _ax.set_ylim(-0.2, 1.2)
        _ax.set_xlabel("z-score [SD]")
        _ax.set_title(_title, fontsize=11)
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The shuffle test

    If a metric ignores which cell is which, shuffling the cells of one gene cannot change it. This is
    the simplest negative control for a co-expression score. Each panel shows one metric on the
    current pair before and after shuffling the cells of $y$.
    """)
    return


@app.cell(hide_code=True)
def _(fig_panels, mo, pair_metrics, pair_x, pair_y, pair_y_shuffled, pal, plt):
    _orig, _shuf = pair_metrics(pair_x, pair_y), pair_metrics(pair_x, pair_y_shuffled)
    _keys = [("pearson r", "Pearson r"), ("spearman ρ", "Spearman ρ"), ("W1(z_x, z_y)", "$W_1$ on z-scores")]
    _fig, _axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for _ax, (_k, _label) in zip(_axes, _keys):
        _vals = [_orig[_k], _shuf[_k]]
        _ax.bar(0, _vals[0], color=pal["dark"], width=0.6)
        _ax.bar(1, _vals[1], color=pal["light"], edgecolor=pal["dark"], hatch="///", lw=0, width=0.6)
        for _x, _v in enumerate(_vals):
            _ax.text(_x, max(_v, 0) + 0.03, f"{_v:.3f}", ha="center", fontsize=10)
        _ax.axhline(0, color=pal["mid"], lw=0.8)
        _ax.set_xticks([0, 1], ["original", "y shuffled"])
        _ax.set_ylim(min(0, min(_vals)) - 0.1, max(1.0, max(_vals)) + 0.15)
        _ax.set_title(_label, fontsize=11)
        _ax.set_ylabel({"pearson r": "Pearson r", "spearman ρ": "Spearman ρ", "W1(z_x, z_y)": "$W_1$ [SD]"}[_k])
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pearson and Spearman drop to about zero, as they should. $W_1$ does not change at all: the two
    numbers are *bit-identical*, because shuffling leaves the sorted values untouched. Setting the
    relationship to **independent** shows the converse: two unrelated genes with similar histograms get
    $W_1 \approx 0$.

    /// tip | Part 2 in one line
    Value-Wasserstein is a **distribution-shape** similarity, not a co-expression score. It answers "do
    these genes have the same expression profile shape?", and is misleading as a "genes like X" ranking.
    ///

    ## Part 3. Graph optimal transport: moving expression through cell space

    Wasserstein becomes useful once the transport happens between **cells** rather than between
    expression values. Normalise each gene's expression to a distribution over cells,
    $a_g = x_g / \sum_i x_{g,i}$. Give the cells a distance $d(i, j)$ that says how far apart two cell
    states are. Then $W_1(a, b)$ is the cheapest way to move gene A's expression onto gene B's, when
    moving one unit from cell $i$ to cell $j$ costs $d(i,j)$:

    $$W_1(a, b) = \min_{\pi \ge 0,\ \pi\mathbf 1 = a,\ \pi^\top \mathbf 1 = b} \ \sum_{ij} \pi_{ij}\, d(i,j).$$

    /// note | What the normalisation keeps and discards
    Dividing by $\sum_i x_{g,i}$ keeps *where* a gene is expressed and discards *how much*: a gene and a
    6× brighter copy of it are the same distribution, so their $W_1$ is ideally 0 (the twin control
    below). The flip side: a constant background (ambient RNA, baseline expression) becomes mass spread
    over all of cell space, and distances between low-specificity genes shrink toward each other.
    Consider background subtraction or thresholding before forming $a_g$.
    ///

    This is the core of **GeneTrajectory** (Qu *et al.*, *Nat. Biotechnol.* 2025). Both inputs come
    from the same expression matrix. Its rows (cells) give the cell distance $d$; its columns (genes)
    give the distributions that are transported. Optimal transport joins the two, and the resulting
    gene–gene distances are then embedded to order genes into trajectories. To keep OT affordable, both
    arms are computed on metacells (groups of similar cells) rather than single cells.

    $$
    X = \begin{array}{cc} & \hspace{0.35em}\begin{array}{cccc}\hspace{0.85em}\mathclap{{\scriptsize\text{gene } 1}}\hspace{0.85em} & \hspace{0.85em}\mathclap{{\scriptsize\text{gene } 2}}\hspace{0.85em} & \hspace{0.85em}\mathclap{\cdots}\hspace{0.85em} & \hspace{0.85em}\mathclap{{\scriptsize\text{gene } m}}\hspace{0.85em}\end{array} \\ \begin{array}{r}{\scriptsize\text{cell } 1} \\ {\scriptsize\text{cell } 2} \\ \vdots \\ {\scriptsize\text{cell } n}\end{array} & \left[\begin{array}{cccc}\hspace{0.85em}\mathclap{x_{11}}\hspace{0.85em} & \hspace{0.85em}\mathclap{x_{12}}\hspace{0.85em} & \hspace{0.85em}\mathclap{\cdots}\hspace{0.85em} & \hspace{0.85em}\mathclap{x_{1m}}\hspace{0.85em} \\ \hspace{0.85em}\mathclap{x_{21}}\hspace{0.85em} & \hspace{0.85em}\mathclap{x_{22}}\hspace{0.85em} & \hspace{0.85em}\mathclap{\cdots}\hspace{0.85em} & \hspace{0.85em}\mathclap{x_{2m}}\hspace{0.85em} \\ \hspace{0.85em}\mathclap{\vdots}\hspace{0.85em} & \hspace{0.85em}\mathclap{\vdots}\hspace{0.85em} & \hspace{0.85em}\mathclap{\ddots}\hspace{0.85em} & \hspace{0.85em}\mathclap{\vdots}\hspace{0.85em} \\ \hspace{0.85em}\mathclap{x_{n1}}\hspace{0.85em} & \hspace{0.85em}\mathclap{x_{n2}}\hspace{0.85em} & \hspace{0.85em}\mathclap{\cdots}\hspace{0.85em} & \hspace{0.85em}\mathclap{x_{nm}}\hspace{0.85em}\end{array}\right] \end{array}
    \;\left.\begin{array}{l}
    \nearrow\; \xrightarrow[\text{cells}]{\textcolor{#6b6b6b}{\text{(rows)}}\text{: PCA, kNN}} \;\textcolor{#6b6b6b}{d_{\text{geo}}(i,j)}  \\[2.2em]
    \searrow\; \xrightarrow{\textcolor{#6a51a3}{\text{genes}}\text{: normalise}} \;\textcolor{#6a51a3}{a_g}
    \end{array}\right\}
    \xrightarrow{\textcolor{#1baf7a}{\text{OT}}} \textcolor{#1baf7a}{W_1(a_g, a_h)}
    $$

    ### A useful distance must not saturate

    Correlation compares expression cell by cell. Once two genes are expressed in disjoint sets of
    cells, there is nothing left for it to compare: every non-overlapping pair gets the same value,
    whether the genes are expressed at neighbouring stages or at opposite ends. A distance between
    distributions should keep telling those cases apart. Below, two expression bumps sit at separation
    $s$ along the cell order. $1 - r$ reaches its ceiling as soon as the bumps stop overlapping and
    stays there. $W_1$ equals $s$ exactly, at every separation.
    """)
    return


@app.cell
def _(mo):
    shift_sl = mo.ui.slider(0.0, 0.8, 0.01, value=0.2, label="separation s", show_value=True)
    shift_sl
    return (shift_sl,)


@app.cell(hide_code=True)
def _(fig_panels, mo, np, pal, plt, shift_sl):
    def _path_metrics(s, w=0.05, n=400):
        xs = np.linspace(0, 1, n)
        a = np.exp(-((xs - (0.5 - s / 2)) ** 2) / (2 * w**2))
        b = np.exp(-((xs - (0.5 + s / 2)) ** 2) / (2 * w**2))
        a, b = a / a.sum(), b / b.sum()
        w1 = float(np.sum(np.abs(np.cumsum(a) - np.cumsum(b))) * (xs[1] - xs[0]))
        return xs, a, b, float(np.corrcoef(a, b)[0, 1]), w1

    _grid = np.linspace(0, 0.8, 81)
    _curves = np.array([_path_metrics(s)[3:] for s in _grid])
    _xs, _a, _b, _r, _w1 = _path_metrics(shift_sl.value)
    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    _axes[0].fill_between(_xs, _a, color=pal["a"], alpha=0.5, lw=0, label="gene A")
    _axes[0].fill_between(_xs, _b, color=pal["b"], alpha=0.5, lw=0, label="gene B")
    _axes[0].set(xlabel="pseudotime, cell order [0–1]", ylabel="expression [a.u.]", yticks=[])
    _axes[0].set_title(f"s = {shift_sl.value:.2f}: Pearson {_r:+.2f}, $W_1$ = {_w1:.2f}", fontsize=11)
    _axes[0].legend(loc="upper left")
    _axes[1].plot(_grid, _curves[:, 1], color=pal["dark"], lw=2)
    _axes[1].plot(_grid, 1 - _curves[:, 0], color=pal["mid"], lw=2, ls="--")
    _axes[1].text(0.6, 0.72, "$W_1$", fontsize=11)
    _axes[1].text(0.42, 1.27, "1 − Pearson (saturated)", fontsize=10)
    _axes[1].axvline(shift_sl.value, color=pal["light"], ls=":")
    _axes[1].set(xlabel="true separation s [pseudotime units]", ylabel="reported distance\n($W_1$ in pseudotime units; 1 − r unitless)", ylim=(0, 1.45))
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The cell distance matters: draw two genes in PCA space

    $W_1$ is only as good as the cell distance $d(i, j)$ it is given. Treat the canvas below as a PCA
    plot: each dot is a cell. Paint with brush **a** for cells expressing gene A, **b** for gene B, and
    **c** for cells expressing neither (they only shape cell space).

    From the drawing, the notebook simulates a plausible **cells × genes count matrix**: 200 background
    genes whose expression varies smoothly across the canvas, plus genes A and B, expressed in the cells
    painted for them. The usual pipeline then runs on that matrix: normalise, log, PCA, kNN graph. Its
    PCA recovers the drawing (up to noise), so what you draw is what the metrics see.

    * **Left, the simulated matrix:** genes A and B and a dozen background genes, with cells grouped
      as painted.
    * **Middle, Euclidean OT in the recovered PCA:** the optimal coupling is straight lines, free to jump
      across empty space.
    * **Right, graph OT:** with the kNN graph as the cell distance, expression has to travel through
      cells. $W_1$ is computed as a flow along the edges (explained further down), and line width shows
      how much crosses each edge.

    The table under the figure gives every metric in this notebook for the same two genes.
    """)
    return


@app.cell
def _(
    NearestNeighbors,
    connected_components,
    linprog,
    np,
    ot,
    shortest_path,
    sp,
):
    def knn_graph(X, k=8):
        """Symmetric kNN graph with Euclidean edge lengths; k is raised until the graph is connected."""
        n = len(X)
        for kk in range(min(k, n - 1), n):
            dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(X).kneighbors(X)
            W = sp.csr_matrix((dist[:, 1:].ravel(), (np.repeat(np.arange(n), kk), idx[:, 1:].ravel())), shape=(n, n))
            W = W.maximum(W.T)
            if connected_components(W, directed=False)[0] == 1:
                break
        return W, kk

    def beckmann_w1_edges(u, v, ell, delta):
        """
        W1 as a min-cost flow on an explicit edge list (u_e, v_e, ℓ_e):
        min Σ ℓ_e |J_e| subject to B J = δ (flow out minus flow in = δ at every node).
        Returns (W1, J). Zero-length edges are fine here, unlike in a sparse matrix.
        """
        E = len(ell)
        B = sp.csr_matrix(
            (np.r_[np.ones(E), -np.ones(E)], (np.r_[u, v], np.r_[np.arange(E), np.arange(E)])),
            shape=(len(delta), E),
        )
        lp = linprog(np.r_[ell, ell], A_eq=sp.hstack([B, -B]), b_eq=delta, bounds=(0, None), method="highs")
        return lp.fun, lp.x[:E] - lp.x[E:]

    def beckmann_w1(W, delta):
        """Graph W1 on a weighted graph W. Returns (W1, J, edge tails, edge heads) for the edges of triu(W)."""
        T = sp.triu(W).tocoo()
        w1, J = beckmann_w1_edges(T.row, T.col, T.data, delta)
        return w1, J, T.row, T.col

    def metacell_graph_from_labels(Z, X, labels, k=8):
        """
        Metacells from any partition of the cells (k-means, a Leiden level, ...).

        Returns a dict with
          centroids: (M, d) metacell centroids in Z
          Xm       : (M, g) mean expression per metacell
          W        : (M, M) sparse symmetric kNN graph, entries = Euclidean edge length
          D_geo    : (M, M) shortest-path distances on W
          k_used   : kNN k after raising it until the graph is connected
        """
        M = labels.max() + 1
        H = sp.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(M, len(labels)))
        H = sp.diags(1.0 / np.asarray(H.sum(1)).ravel()) @ H
        centroids = np.asarray(H @ Z)
        W, k_used = knn_graph(centroids, min(k, M - 1))
        return dict(centroids=centroids, Xm=np.asarray(H @ X), W=W, D_geo=shortest_path(W, directed=False), k_used=k_used)

    def graph_w1_matrix(P, D):
        """Exact pairwise W1 between columns of P (each a probability vector) under ground cost D."""
        g = P.shape[1]
        cols = [np.ascontiguousarray(P[:, i]) for i in range(g)]
        out = np.zeros((g, g))
        for i in range(g):
            for j in range(i + 1, g):
                out[i, j] = out[j, i] = ot.emd2(cols[i], cols[j], D)
        return out

    return (
        beckmann_w1,
        beckmann_w1_edges,
        graph_w1_matrix,
        knn_graph,
        metacell_graph_from_labels,
    )


@app.cell
def _(Path, ScatterWidget, mo, np, pal, wigglystuff):
    # ScatterWidget with the notebook's colours: a = gene A, b = gene B, c = background cells (grey)
    _js = (Path(wigglystuff.__file__).parent / "static" / "scatter-widget.js").read_text()
    _default = '["#1f77b4","#ff7f0e","#2ca02c","#d62728"]'
    _ours = f'["{pal["a"]}","{pal["b"]}","#9a9a9a","#d62728"]'

    class CellCanvas(ScatterWidget):
        # Two patches to wigglystuff's scatter-widget.js: our class colours, and a fix for an upstream
        # bug. While building the class buttons, render() auto-clicks the first one, and that click
        # handler touches the brush cursor `B`, which is declared later in render(): a temporal-dead-zone
        # ReferenceError that aborts rendering (the canvas shows up empty, seen in app mode). Deferring
        # the first click until render() has returned fixes it. Both replacements are no-ops if a future
        # wigglystuff changes this code.
        _esm = _js.replace(_default, _ours).replace(
            "p===0&&(C=g,g.click())", "p===0&&(C=g,setTimeout(()=>g.click(),0))"
        )

    def _horseshoe(seed=3, R=130.0, cx=300.0, cy=180.0):
        rng = np.random.default_rng(seed)
        pts = []
        for label, color, lo, hi, n in [
            ("a", pal["a"], 35, 62, 22),
            ("c", "#9a9a9a", 62, 298, 120),
            ("b", pal["b"], 298, 325, 22),
        ]:
            th = np.deg2rad(rng.uniform(lo, hi, n))
            r = R + rng.normal(0, 9, n)
            for x, y in zip(cx + r * np.cos(th), cy + r * np.sin(th)):
                pts.append({"x": float(x), "y": float(y), "color": color, "label": label, "batch": 0})
        return pts

    default_drawing = _horseshoe()
    draw_widget = mo.ui.anywidget(CellCanvas(n_classes=3, width=640, height=360, brushsize=20, data=default_drawing))
    draw_widget
    return default_drawing, draw_widget


@app.cell(hide_code=True)
def _(draw_widget, get_drawing, mo, set_drawing):
    use_drawing_btn = mo.ui.button(
        label="↻ Rebuild dataset",
        kind="success",
        tooltip="Rebuild Parts 3–5 from the current drawing (about 20 s)",
        on_click=lambda _: set_drawing(list(draw_widget.widget.data)),
    )
    _n_canvas, _n_data = len(draw_widget.widget.data), len(get_drawing())
    _status = (
        "✓ dataset matches the drawing"
        if list(draw_widget.widget.data) == list(get_drawing())
        else f"canvas changed ({_n_canvas} vs {_n_data} points): rebuild to update Parts 3–5"
    )
    mo.hstack([use_drawing_btn, mo.md(_status)], justify="start", align="center", gap=1)
    return


@app.cell
def _(
    beckmann_w1,
    draw_widget,
    fig_panels,
    knn_graph,
    mo,
    np,
    orthogonal_procrustes,
    ot,
    pair_metrics,
    pal,
    pd,
    pdist,
    plt,
    sp,
    squareform,
):
    def matrix_from_drawing(points, n_background=200, seed=0):
        """
        Treat the drawing as PCA space and simulate a cells x genes count matrix consistent with it.

        Background genes have log-rates linear in the drawn coordinates (random loadings), so their
        PCA recovers the drawing; genes A and B are expressed in the cells painted 'a' and 'b'.
        Returns labels, drawn coordinates, log-normalised matrix (A, B, background...) and the
        recovered 2D PCA, rotated onto the drawing for display.
        """
        rng = np.random.default_rng(seed)
        lab = np.array([p["label"] for p in points])
        Z = np.array([[p["x"], p["y"]] for p in points], dtype=float)
        Z = (Z - Z.mean(0)) / Z.std()
        lib = rng.lognormal(np.log(5), 0.3, len(Z))
        loadings = rng.normal(size=(n_background, 2))
        background = rng.poisson(lib[:, None] * np.exp(-0.5 + 0.8 * Z @ loadings.T))
        gene_a = rng.poisson(lib * (0.02 + 3.0 * (lab == "a")))
        gene_b = rng.poisson(lib * (0.02 + 3.0 * (lab == "b")))
        counts = np.c_[gene_a, gene_b, background].astype(float)
        X = np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)
        U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
        pcs = U[:, :2] * S[:2]
        R, _ = orthogonal_procrustes(pcs / pcs.std(), Z)
        return lab, Z, X, (pcs / pcs.std()) @ R

    def draw_two_genes(points):
        lab = np.array([p["label"] for p in points])
        if (lab == "a").sum() == 0 or (lab == "b").sum() == 0:
            return mo.md("Paint at least one **a** cell and one **b** cell.")
        lab, Z, X, P = matrix_from_drawing(points)
        a, b = np.expm1(X[:, 0]), np.expm1(X[:, 1])
        a, b = a / a.sum(), b / b.sum()

        D = squareform(pdist(P))
        plan = ot.emd(a, b, D)
        w1_euc = float((plan * D).sum())
        W, _ = knn_graph(P, k=6)
        w1_graph, J, er, ec = beckmann_w1(W, a - b)
        m = pair_metrics(X[:, 0], X[:, 1])

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={"width_ratios": [1, 1.1, 1.1]})
        order = np.lexsort((P[:, 0], np.array([{"a": 0, "c": 1, "b": 2}[l] for l in lab])))
        # cells (observations) as rows, genes as columns, like X in the pipeline above
        axes[0].imshow(X[order][:, :14], aspect="auto", cmap="Purples", interpolation="nearest")
        axes[0].set_xticks(range(14), ["gene A", "gene B"] + [f"bg {i}" for i in range(1, 13)], fontsize=8, rotation=90)
        axes[0].axvline(1.5, color=pal["dark"], lw=1)
        _bounds = np.cumsum([np.sum(lab == g) for g in ["a", "c", "b"]])
        for y in _bounds[:-1]:
            axes[0].axhline(y - 0.5, color=pal["dark"], lw=1)
        axes[0].set_yticks((np.r_[0, _bounds[:-1]] + _bounds) / 2 - 0.5, ["a cells", "c cells", "b cells"], fontsize=9)
        axes[0].set_title("simulated matrix X (log expression)", fontsize=11)

        cols = {"a": pal["a"], "b": pal["b"], "c": pal["light"]}
        for ax in axes[1:]:
            ax.scatter(P[:, 0], P[:, 1], s=12, c=[cols[l] for l in lab], zorder=3, lw=0)
            ax.set_aspect("equal")
            ax.set_axis_off()
        for i, j in zip(*np.nonzero(plan > 1e-3 * plan.max())):
            if i != j:
                axes[1].plot(*P[[i, j]].T, color=pal["flow"], lw=0.3 + 20 * plan[i, j], alpha=0.6)
        axes[1].set_title(f"Euclidean in PCA: $W_1$ = {w1_euc:.2f}", fontsize=11)
        Tw = sp.triu(W).tocoo()
        for u, v in zip(Tw.row, Tw.col):
            axes[2].plot(*P[[u, v]].T, color=pal["faint"], lw=0.5, zorder=1)
        flow = np.abs(J) > 1e-9
        for u, v, f in zip(er[flow], ec[flow], np.abs(J[flow])):
            axes[2].plot(*P[[u, v]].T, color=pal["flow"], lw=0.5 + 6 * f, zorder=2)
        axes[2].set_title(f"kNN graph: $W_1$ = {w1_graph:.2f}", fontsize=11)
        fig.tight_layout()
        table = pd.DataFrame(
            {
                "Pearson r": [m["pearson r"]],
                "Spearman ρ": [m["spearman ρ"]],
                "W1 on values (z-scores)": [m["W1(z_x, z_y)"]],
                "W1, Euclidean in PCA": [w1_euc],
                "W1, kNN graph": [w1_graph],
                "graph / Euclidean": [w1_graph / w1_euc],
            },
            index=["gene A vs gene B"],
        ).round(3)
        return mo.vstack([fig_panels(fig), table])

    draw_two_genes(draw_widget.widget.data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A dataset with a known answer, generated from your drawing

    To score metrics we need data where the true gene–gene distance is known. This dataset is built from
    the canvas above (the horseshoe until you press the rebuild button under it): the drawing *is* the cell space.

    * **Cells** are sampled around the drawn points, with a little jitter.
    * **Trajectory genes** each peak at one drawn point; their expression falls off with the distance
      *along the drawing* (geodesic distance on the kNN graph of drawn points). So the **true distance
      between two genes is the geodesic distance between their peaks**. This works for any shape: an arc,
      a branch, a loop.
    * **Positive control, twin genes**: same peak as a parent gene, 6× the amplitude. A good metric puts
      each twin next to its parent, or among the genes whose peaks are within a small tolerance ε of it:
      with 60 randomly placed peaks (the default) along one drawing, those genes are equally legitimate
      neighbours (Part 5).
    * **Negative control, permuted genes**: a copy of a trajectory gene with its cells shuffled (the
      shuffle test from Part 2). A good metric must *not* put it next to its source.
    * **200 background genes** vary smoothly across the canvas, so that PCA of the data reproduces the
      drawing, and **300 noise genes** add unstructured background.

    Draw a shape, press **↻ Rebuild the dataset** under the canvas, and Parts 3–5 recompute (about 20 s).
    Drawing strokes alone only update the quick A/B demo above.

    The figure checks the simulation. **Left:** cells in PCA, coloured by their position $t$ along the
    drawing (geodesic distance from one end). With the default horseshoe, the two ends sit close in PCA
    but far apart along the shape: exactly where a straight-line distance between cells misleads.
    **Right:** read it like a stack of in situ stains. Each row is one trajectory gene (sorted by peak
    position), each column one cell (sorted by $t$). The diagonal band is the structure a good distance
    should recover.
    """)
    return


@app.cell
def _(mo):
    seed_num = mo.ui.number(start=0, stop=999, value=0, label="random seed")
    n_cells_sl = mo.ui.slider(500, 4000, 250, value=2000, label="cells", show_value=True)
    n_traj_sl = mo.ui.slider(20, 100, 10, value=60, label="trajectory genes", show_value=True)
    mo.hstack([seed_num, n_cells_sl, n_traj_sl])
    return n_cells_sl, n_traj_sl, seed_num


@app.cell(hide_code=True)
def _(default_drawing, mo):
    # The dataset is rebuilt from a snapshot of the canvas, taken when the button below is pressed,
    # so drawing strokes only update the quick A/B demo above.
    get_drawing, set_drawing = mo.state(default_drawing)
    return get_drawing, set_drawing


@app.cell
def _(
    ad,
    knn_graph,
    np,
    orthogonal_procrustes,
    pdist,
    sc,
    shortest_path,
    squareform,
):
    def simulate_from_drawing(points, n_cells=2000, n_traj=60, n_noise=300, n_background=200, seed=0):
        """
        Simulate a cells x genes dataset whose cell space is the drawing.

        Cells are drawn points plus a little jitter. Trajectory genes peak at a drawn point and decay with
        the geodesic distance along the drawing (kNN graph of the drawn points), which is also the ground
        truth between genes. Background genes vary linearly across the canvas so that PCA reproduces the
        drawing; noise genes are unstructured.
        """
        rng = np.random.default_rng(seed)
        Zd = np.array([[p["x"], p["y"]] for p in points], dtype=float)
        Zd = (Zd - Zd.mean(0)) / Zd.std()
        W, _ = knn_graph(Zd, k=6)
        D_nodes = shortest_path(W, directed=False)
        D_nodes = D_nodes / D_nodes.max()
        start = int(np.argmax(D_nodes[0]))  # one end of the drawing
        t_node = D_nodes[start] / D_nodes[start].max()

        _nn = squareform(pdist(Zd))
        np.fill_diagonal(_nn, np.inf)
        jitter = 0.5 * np.median(_nn.min(1))
        home = rng.integers(0, len(Zd), n_cells)  # drawn point each cell is sampled around
        Zc = Zd[home] + rng.normal(0, jitter, (n_cells, 2))

        peak_node = rng.integers(0, len(Zd), n_traj)
        peak_node = peak_node[np.argsort(t_node[peak_node])]
        gw = rng.uniform(0.05, 0.1, n_traj)
        ga = rng.lognormal(0, 0.5, n_traj)
        twin_parent = np.arange(0, n_traj, max(n_traj // 5, 1))
        perm_parent = np.arange(n_traj // 10 + 1, n_traj, max(n_traj // 5, 1))

        pn = np.r_[peak_node, peak_node[twin_parent]]
        pw = np.r_[gw, gw[twin_parent]]
        pa = np.r_[ga, 6.0 * ga[twin_parent]]
        d_cg = D_nodes[home][:, pn]
        lib = rng.lognormal(np.log(3), 0.3, n_cells)[:, None]
        counts_struct = rng.poisson(lib * (0.05 + pa * np.exp(-(d_cg**2) / (2 * pw**2))))
        counts_perm = counts_struct[:, perm_parent][rng.permutation(n_cells)]
        loadings = rng.normal(size=(n_background, 2))
        counts_bg = rng.poisson(lib * np.exp(-0.5 + 0.8 * Zc @ loadings.T))
        counts_noise = rng.poisson(lib * rng.lognormal(-1, 1, n_noise)[None])
        counts = np.c_[counts_struct, counts_perm, counts_bg, counts_noise].astype(np.float32)

        names = (
            [f"g{i:02d}_t{t_node[p]:.2f}" for i, p in enumerate(peak_node)]
            + [f"twin_of_g{p:02d}" for p in twin_parent]
            + [f"perm_of_g{p:02d}" for p in perm_parent]
            + [f"bg{i:03d}" for i in range(n_background)]
            + [f"noise{i:03d}" for i in range(n_noise)]
        )
        kind = (["trajectory"] * n_traj + ["twin"] * len(twin_parent) + ["permuted"] * len(perm_parent)
                + ["background"] * n_background + ["noise"] * n_noise)
        parent = list(range(n_traj)) + list(twin_parent) + list(perm_parent) + [-1] * (n_background + n_noise)
        peak = np.r_[peak_node, peak_node[twin_parent], peak_node[perm_parent], -np.ones(n_background + n_noise, int)]

        adata = ad.AnnData(counts)
        adata.var_names = names
        adata.obs["t"] = t_node[home]
        adata.obsm["X_drawn"] = Zc
        adata.var["kind"] = kind
        adata.var["parent"] = parent
        adata.var["peak_node"] = peak
        adata.var["peak_t"] = np.where(peak >= 0, t_node[np.maximum(peak, 0)], np.nan)
        adata.uns["peak_geodesic"] = D_nodes
        adata.layers["counts"] = counts.copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.pca(adata, n_comps=20, random_state=0)

        # Orient PCs 1-2 like the drawing. PCA determines the plane spanned by PC1 and PC2 but not its
        # rotation or reflection (eigenvector signs and order are arbitrary). We rotate, and if needed
        # reflect, the PC1-PC2 scores with the orthogonal Procrustes solution R that best maps them onto
        # the drawn cell coordinates: R = argmin ||X_pca[:, :2] @ R - Zc||_F over orthogonal 2x2 R (no
        # scaling). The loadings get the same R so scores and loadings stay consistent. An orthogonal map
        # within the plane preserves every pairwise distance, so the kNN graph, metacells, cluster trees
        # and all W1 values are unchanged; only the picture lines up with the canvas. After this, PCs 1-2
        # are no longer ordered by explained variance.
        R, _ = orthogonal_procrustes(adata.obsm["X_pca"][:, :2], Zc)
        adata.obsm["X_pca"][:, :2] = adata.obsm["X_pca"][:, :2] @ R
        adata.varm["PCs"][:, :2] = adata.varm["PCs"][:, :2] @ R
        return adata

    return (simulate_from_drawing,)


@app.cell
def _(
    ensure_neighbors,
    get_drawing,
    n_cells_sl,
    n_traj_sl,
    np,
    seed_num,
    simulate_from_drawing,
):
    adata = simulate_from_drawing(get_drawing(), n_cells=n_cells_sl.value, n_traj=n_traj_sl.value, seed=seed_num.value)
    ensure_neighbors(adata, n_neighbors=15, use_rep="X_pca")
    genes_eval = adata.var_names[adata.var["kind"].isin(["trajectory", "twin", "permuted"])].tolist()
    _D_peaks = adata.uns["peak_geodesic"]


    def true_distance(p, q):
        """True distance between two gene peaks: geodesic distance between their peak points along the drawing."""
        return _D_peaks[np.asarray(p, int), np.asarray(q, int)]


    adata
    return adata, genes_eval, true_distance


@app.cell(hide_code=True)
def _(adata, fig_panels, mo, np, pal, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    _pc = adata.obsm["X_pca"]
    _t = adata.obs["t"].to_numpy()
    _scat = _axes[0].scatter(_pc[:, 0], _pc[:, 1], s=3, c=_t, cmap="viridis", lw=0)
    _fig.colorbar(_scat, ax=_axes[0], label="position along the drawing, t [0–1]")
    for _sel, _name in [(_t < 0.03, "start (t = 0)"), (_t > 0.97, "end (t = 1)")]:
        _cx, _cy = np.median(_pc[_sel, :2], axis=0)
        _axes[0].annotate(_name, (_cx, _cy), fontsize=11, fontweight="bold", ha="center",
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=pal["faint"], alpha=0.85))
    _axes[0].set(xlabel="PC1", ylabel="PC2")
    _axes[0].set_title("cells in PCA, oriented like your drawing", fontsize=12)

    _traj = adata[:, adata.var["kind"] == "trajectory"]
    _im = _axes[1].imshow(_traj.X[np.argsort(_t)].T, aspect="auto", cmap="Purples", interpolation="nearest")
    _fig.colorbar(_im, ax=_axes[1], label="expression [log(1 + CP10k)]")
    _axes[1].set(xlabel="cells, ordered by t", ylabel="trajectory genes,\nordered by peak", xticks=[])
    _axes[1].set_title("expression program", fontsize=12)
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Building the cell distance

    Exact OT on $n$ cells costs roughly $O(n^3 \log n)$ per gene pair, so GeneTrajectory first groups
    the cells into $M$ **metacells** with k-means. A metacell's expression is the mean over its member
    cells. Metacells are joined by a kNN graph on their PCA centroids, with Euclidean edge lengths, and
    the cell distance is the shortest-path (*geodesic*) distance on that graph.[^dm]

    **Why not just Euclidean distance in PCA?** The figure answers this. **Left:** the metacell graph
    over the cells. **Middle:** geodesic distance from the start of the trajectory; it grows smoothly
    along the horseshoe, as a distance in cell space should. **Right:** geodesic against Euclidean
    distance for every pair of metacells. Pairs far apart along the trajectory (dark) sit far above the
    diagonal: the arch brings them close in PCA, but cells have to travel the whole arc to get from one
    to the other. These are the pairs where Euclidean OT would take a shortcut, like the straight lines
    across the drawn horseshoe.

    [^dm]: GeneTrajectory builds this graph on diffusion-map coordinates; PCA keeps the example simple.
    """)
    return


@app.cell
def _(mo):
    n_meta_sl = mo.ui.slider(50, 400, 25, value=200, label="metacells M", show_value=True)
    k_meta_sl = mo.ui.slider(3, 20, 1, value=8, label="metacell kNN k", show_value=True)
    mo.hstack([n_meta_sl, k_meta_sl])
    return k_meta_sl, n_meta_sl


@app.cell
def _(KMeans, metacell_graph_from_labels):
    def build_metacell_graph(Z, X, n_meta=200, k=8, seed=0):
        """k-means metacells (GeneTrajectory's coarsening) and their graph; adds `labels` to the result."""
        labels = KMeans(n_meta, n_init=3, random_state=seed).fit(Z).labels_
        return dict(labels=labels, **metacell_graph_from_labels(Z, X, labels, k))

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
def _(adata, fig_panels, meta, mo, np, pal, pd, pdist, plt, squareform):
    _xy = meta["centroids"][:, :2]
    _pc = adata.obsm["X_pca"]
    _t = pd.Series(adata.obs["t"].to_numpy()).groupby(meta["labels"]).mean().to_numpy()
    _root = int(np.argmin(_t))
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.9), gridspec_kw={"width_ratios": [1, 1.2]})
    _r, _c = meta["W"].nonzero()
    for _ax in _axes[:2]:
        _ax.scatter(_pc[:, 0], _pc[:, 1], s=1, color=pal["faint"], lw=0, zorder=0)
        _ax.plot(np.c_[_xy[_r, 0], _xy[_c, 0]].T, np.c_[_xy[_r, 1], _xy[_c, 1]].T, color=pal["light"], lw=0.5, zorder=1)
        _ax.set(xlabel="PC1", ylabel="PC2")
    _axes[0].scatter(*_xy.T, s=12, color=pal["dark"], zorder=2, lw=0)
    _axes[0].set_title("metacells and their kNN graph", fontsize=11)
    _sc = _axes[1].scatter(*_xy.T, s=16, c=meta["D_geo"][_root], cmap="viridis", zorder=2, lw=0)
    _axes[1].scatter(*_xy[_root], s=110, marker="*", color="black", zorder=3)
    _fig.colorbar(_sc, ax=_axes[1], label="geodesic distance\nfrom ★ [PCA units]")
    _axes[1].set_title("distance along the graph from the start", fontsize=11)

    _De = squareform(pdist(meta["centroids"]))
    _iu = np.triu_indices(len(_xy), 1)
    _far = (np.abs(_t[:, None] - _t[None]) > 0.5)[_iu]

    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Two views of the same transport: a plan and a flow

    The OT formula gives a **plan** $\pi_{ij}$: how much of gene A's expression at metacell $i$ goes to
    metacell $j$. On a graph there is an equivalent view that matters for the rest of the notebook. The
    same $W_1$ is the cheapest **flow** $J$ along the edges whose net outflow at each node is
    $\delta = a - b$ (the Beckmann problem; Santambrogio 2015, ch. 4):

    $$W_1(a,b) \;=\; \min_{\pi} \sum_{ij}\pi_{ij}\,d_{\text{geo}}(i,j) \;=\; \min_{J:\ BJ = \delta}\ \sum_e \ell_e\, |J_e| ,$$

    with $B$ the node–edge incidence matrix and $\ell_e$ the edge lengths. The plan says *who sends to
    whom*; the flow says *how much traffic crosses each edge*. Two reasons the flow view matters: it
    needs only the graph's edges, not an $M \times M$ cost matrix, and on a **tree** it has a closed form
    (Part 4).

    The default pair below peaks at the two ends of the trajectory, which the arch brings close together
    in PCA; all of its transport still has to travel the arc. The panel titles give $W_1$ from both
    solvers, and they agree. Also try a gene with its
    `perm_of_` copy (spread everywhere, so transport goes in all directions) and with its `twin_of_`
    (almost nothing moves).

    Why the plan looks busy: every gene has some low-level expression in every metacell, and OT has to
    move that too, in many small pieces. The plan is still sparse (an exact OT plan uses at most $2M - 1$
    of the $M^2$ possible pairs), so the panel draws only the largest pairings that together carry 80% of
    the moved mass. Read each line as a pairing, not a path: mass never crosses the empty middle of the
    horseshoe; it travels along the graph, as the flow panel shows.
    """)
    return


@app.cell
def _(adata, genes_eval, mo):
    _v = adata.var.loc[genes_eval]
    _traj = _v[_v["kind"] == "trajectory"]
    gene_a_dd = mo.ui.dropdown(genes_eval, value=_traj["peak_t"].idxmin(), label="gene A", searchable=True)
    gene_b_dd = mo.ui.dropdown(genes_eval, value=_traj["peak_t"].idxmax(), label="gene B", searchable=True)
    mo.hstack([gene_a_dd, gene_b_dd], wrap=True, justify="start")
    return gene_a_dd, gene_b_dd


@app.cell(hide_code=True)
def _(
    beckmann_w1,
    fig_panels,
    gene_a_dd,
    gene_b_dd,
    gene_dist,
    genes_eval,
    meta,
    mo,
    np,
    ot,
    pal,
    plt,
):
    _ia, _ib = genes_eval.index(gene_a_dd.value), genes_eval.index(gene_b_dd.value)
    _a, _b = gene_dist[:, _ia], gene_dist[:, _ib]
    _plan = ot.emd(_a, _b, meta["D_geo"])
    _w1_plan = float((_plan * meta["D_geo"]).sum())
    _w1_flow, _J, _er, _ec = beckmann_w1(meta["W"], _a - _b)
    _xy = meta["centroids"][:, :2]
    _fig, _axes = plt.subplots(1, 3, figsize=(13.5, 4), sharex=True, sharey=True)
    _r, _c = meta["W"].nonzero()
    for _ax in _axes:
        _ax.plot(np.c_[_xy[_r, 0], _xy[_c, 0]].T, np.c_[_xy[_r, 1], _xy[_c, 1]].T, color=pal["faint"], lw=0.5, zorder=0)
        _ax.set_xlabel("PC1")
    _axes[0].set_ylabel("PC2")
    _axes[0].scatter(*_xy.T, s=_a * 100 * len(_a), color=pal["a"], alpha=0.55, lw=0, label="gene A")
    _axes[0].scatter(*_xy.T, s=_b * 100 * len(_b), color=pal["b"], alpha=0.55, lw=0, label="gene B")
    _axes[0].legend(loc=(1.02, 0.7), markerscale=0.6)
    _axes[0].set_title("expression of A and B per metacell", fontsize=11)
    # Exact OT plans are sparse (at most 2M - 1 nonzero entries), but every gene also has low-level
    # expression spread over all metacells, which gets moved in many small pieces. Draw only the largest
    # off-diagonal entries that together carry 80% of the moved mass, line width proportional to mass.
    # A straight line is a pairing (who sends to whom), not a route: the mass travels along the graph's
    # shortest path between the two metacells, which is what the flow panel shows.
    _off = _plan.copy()
    np.fill_diagonal(_off, 0)
    _order = np.argsort(_off.ravel())[::-1]
    _w = _off.ravel()[_order]
    _keep = _order[: int(np.searchsorted(np.cumsum(_w) / _w.sum(), 0.8)) + 1]
    for _i, _j in zip(*np.unravel_index(_keep, _off.shape)):
        _axes[1].plot(*np.c_[_xy[_i], _xy[_j]], color=pal["flow"], lw=0.4 + 4 * _off[_i, _j] / _w[0], alpha=0.7)
    _axes[1].scatter(*_xy.T, s=3, color=pal["mid"], lw=0)
    _axes[1].set_title(f"plan: {len(_keep)} largest pairings (80% of moved mass)\n$W_1$ = {_w1_plan:.4f}", fontsize=11)
    _on = np.abs(_J) > 1e-6
    for _u, _v, _f in zip(_er[_on], _ec[_on], np.abs(_J[_on])):
        _axes[2].plot(*np.c_[_xy[_u], _xy[_v]], color=pal["flow"], lw=0.4 + 12 * _f, solid_capstyle="round")
    _axes[2].scatter(*_xy.T, s=3, color=pal["mid"], lw=0)
    _axes[2].set_title(f"flow: traffic along each graph edge\n$W_1$ = {_w1_flow:.4f}", fontsize=11)
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Part 3 in one line
    Graph $W_1$ measures how far one gene's expression must travel *through cell space* to become the
    other's. It keeps growing with distance, and it is not fooled by shuffling.
    ///

    ## Part 4. Multiscale coarsening: choosing the scale, then reading off $W_1$

    **The problem.** Graph $W_1$ needs coarsening, and Part 3 used k-means with a hand-picked number of
    metacells $M$. That choice is arbitrary: too fine and every one of the $g(g-1)/2$ OT solves is
    slow (seconds for 80 genes, hours for 5,000); too coarse and distinct cell states merge.

    **The idea: coarsen at every scale at once.** `multiscale_coarsening` clusters the cells with Leiden
    at several resolutions and links each fine cluster to its majority parent at the next coarser
    level. The result is a *hierarchy of partitions*, from many small clusters to a few large ones.
    This is a data structure, not a claim about the biology: it can be built on a trajectory, a
    branching process or a cell cycle alike. It supports two readouts:

    $$
    \begin{array}{ccccc}
    & & \textcolor{#6b6b6b}{\text{pick a level}} \;\to\; \textcolor{#6b6b6b}{\text{metacell graph}} & \xrightarrow{\textcolor{#1baf7a}{\text{exact OT}}} & \textcolor{#1baf7a}{\text{graph } W_1} \quad \text{(readout 1: any topology)} \\[0.4em]
    \textcolor{#6b6b6b}{\text{cells}} & \xrightarrow{\text{Leiden hierarchical clustering }} & \textcolor{#6b6b6b}{\text{hierarchy of partitions}} & & \\[0.4em]
    & & \textcolor{#6b6b6b}{\text{use the hierarchy itself}} \;\to\; \textcolor{#6a51a3}{\phi(a_g)} & \xrightarrow{\textcolor{#1baf7a}{\text{closed form}}} & \textcolor{#1baf7a}{\text{tree } W_1} \quad \text{(readout 2: cheapest)}
    \end{array}
    $$

    1. **Readout 1, principled coarsening.** Use any level of the hierarchy as the metacells and run
       exact graph $W_1$ on it. Geodesics on the metacell graph follow loops and branches, so nothing
       assumes a tree. The hierarchy turns "how many metacells?" into a question the data can answer:
       compare the distances across levels and keep the coarsest level that still agrees with finer
       ones.
    2. **Readout 2, tree $W_1$.** Use the hierarchy itself as the cell distance. Then $W_1$ has a closed
       form and needs no OT solve at all. Whether that shortcut is safe when the biology is *not* a
       tree is tested on a simulated cell cycle below.

    The controls set the resolutions of the hierarchy used for the tree readout.

    /// note | How the hierarchy is made nested
    Leiden levels are independent runs, so the majority-vote hierarchy is not strictly nested. Here
    each cell follows its finest cluster's chain of majority parents, which makes it nested by
    construction; *impurity* below is the fraction of cells this reassigns, averaged over levels.
    ///
    """)
    return


@app.cell
def _(mo):
    res_ms = mo.ui.multiselect(
        ["0.3", "0.5", "1", "2", "4", "8", "16", "30", "60", "100"],
        value=["0.5", "2", "8", "30"],
        label="Leiden resolutions",
    )
    n_trees_sl = mo.ui.slider(1, 5, 1, value=1, label="trees averaged (Leiden seeds)", show_value=True)
    mo.hstack([res_ms, n_trees_sl])
    return n_trees_sl, res_ms


@app.cell(hide_code=True)
def _(contextlib, io, multiscale_coarsening, np, pd, sp):
    def _membership(labels):
        """(n_nodes, n_cells) 0/1 membership matrix for integer node ids per cell."""
        n = len(labels)
        return sp.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(labels.max() + 1, n))

    def build_cluster_tree(adata, resolutions, use_rep="X_pca", seed=0):
        """
        Nested cluster tree from `multiscale_coarsening`.

        Returns (levels, impurity). `levels` is a list of (n_cells,) integer node ids, finest
        level first and the root (all zeros) last. Each cell follows its finest cluster's chain of
        majority parents, so every node has exactly one parent.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            ms = multiscale_coarsening(adata.copy(), resolutions=resolutions, use_rep=use_rep, random_state=seed)
        res = ms["resolutions"][::-1]  # fine -> coarse
        labels = ms["clustering"][res[0]].astype(str).to_numpy()
        levels, purity = [], []
        for i, r in enumerate(res):
            if i > 0:
                parent = ms["hierarchy"]["tree"][(res[i - 1], r)]
                labels = np.array([str(parent[l]) for l in labels])
            purity.append(np.mean(labels == ms["clustering"][r].astype(str).to_numpy()))
            levels.append(pd.factorize(labels, sort=True)[0])
        levels.append(np.zeros(len(labels), dtype=int))
        return levels, 1.0 - float(np.mean(purity))

    def tree_node_centroids(Z, levels):
        """Per level, (n_nodes, d) mean of Z over each node's cells, and each node's parent id."""
        H = [_membership(lv) for lv in levels]
        cent = [np.asarray((sp.diags(1.0 / np.asarray(h.sum(1)).ravel()) @ h) @ Z) for h in H]
        parents = []
        for l in range(len(levels) - 1):
            p = np.zeros(H[l].shape[0], dtype=int)
            p[levels[l]] = levels[l + 1]
            parents.append(p)
        return cent, parents

    def tree_w1_embedding(Z, A, levels):
        """
        φ(a) for every column of A, so that W1_tree(a, b) = ‖φ(a) − φ(b)‖₁.

        Z: (n_cells, d) coordinates for edge lengths; A: (n_cells, g), columns sum to 1;
        levels: output of `build_cluster_tree`. Rows of the result are tree edges: the n_cells
        leaf edges first, then each level's node→parent edges, fine to coarse.
        """
        cent, parents = tree_node_centroids(Z, levels)
        # leaf edges: each cell to its finest cluster centroid
        blocks = [np.linalg.norm(Z - cent[0][levels[0]], axis=1)[:, None] * A]
        for l, p in enumerate(parents):
            ell = np.linalg.norm(cent[l] - cent[l + 1][p], axis=1)
            blocks.append(ell[:, None] * (_membership(levels[l]) @ A))  # mass below the edge × edge length
        return np.vstack(blocks)

    return build_cluster_tree, tree_node_centroids, tree_w1_embedding


@app.cell
def _(adata, build_cluster_tree, mo, n_trees_sl, np, res_ms, time):
    _t = time.perf_counter()
    _built = [build_cluster_tree(adata, [float(r) for r in res_ms.value], seed=s) for s in range(n_trees_sl.value)]
    trees = [lv for lv, _ in _built]
    tree_build_secs = time.perf_counter() - _t
    _sizes = " → ".join(str(lv.max() + 1) for lv in trees[0])
    mo.md(
        f"{len(trees)} tree(s), nodes per level (fine → root): {adata.n_obs} cells → {_sizes}; "
        f"impurity {np.mean([imp for _, imp in _built]):.3f}; "
        f"built in {tree_build_secs:.2f} s (Leiden runs dominate)"
    )
    return tree_build_secs, trees


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Readout 1: exact graph $W_1$ at a chosen level

    Here each level of a finer hierarchy (resolutions 0.5 to 30) becomes the metacells, and exact graph
    $W_1$ is computed on its graph, exactly as in Part 3 but with Leiden clusters in place of k-means.
    Levels with more than 170 clusters are skipped to keep the notebook responsive; the k-means result
    with $M$ = 200 from Part 3 is the fine-scale reference.

    * **Left, accuracy against the known answer** (available only in simulation). It stays flat as the
      metacells get coarser, then drops once distinct states merge.
    * **Middle, agreement with the fine-scale reference.** This needs no ground truth, so it is the
      panel to use on real data: compute the finest scale you can afford once, then coarsen until the
      agreement first drops.
    * **Right, cost.** Fewer metacells make every one of the $g(g-1)/2$ solves cheaper.
    """)
    return


@app.cell
def _(
    adata,
    build_cluster_tree,
    genes_eval,
    graph_w1_matrix,
    metacell_graph_from_labels,
    np,
    time,
):
    _levels = [lv for lv in build_cluster_tree(adata, [0.5, 1, 2, 4, 8, 16, 30], seed=0)[0] if 4 <= lv.max() + 1 <= 170]
    _X = np.asarray(adata[:, genes_eval].X)
    level_w1 = []
    for _lv in _levels:
        _t = time.perf_counter()
        _m = metacell_graph_from_labels(adata.obsm["X_pca"], _X, _lv)
        _P = _m["Xm"] / _m["Xm"].sum(0, keepdims=True)
        level_w1.append({"nodes": int(_lv.max() + 1), "D": graph_w1_matrix(_P, _m["D_geo"]), "secs": time.perf_counter() - _t})
    return (level_w1,)


@app.cell(hide_code=True)
def _(
    adata,
    dist_secs,
    dists,
    fig_panels,
    genes_eval,
    level_w1,
    log_ticks,
    meta,
    mo,
    np,
    pal,
    pd,
    plt,
    true_distance,
):
    _v = adata.var.loc[genes_eval]
    _traj = np.where(_v["kind"].to_numpy() == "trajectory")[0]
    _pt = _v["peak_node"].to_numpy()
    _gt = true_distance(_pt[_traj, None], _pt[None, _traj])
    _iu = np.triu_indices(len(_traj), 1)
    _iu_all = np.triu_indices(len(genes_eval), 1)

    def _rho(a, b):
        return pd.Series(a).corr(pd.Series(b), method="spearman")

    def _rho_gt(D):
        return _rho(D[np.ix_(_traj, _traj)][_iu], _gt[_iu])

    _nodes = np.array([r["nodes"] for r in level_w1])
    _acc = [_rho_gt(r["D"]) for r in level_w1]
    _agree = [_rho(r["D"][_iu_all], dists["W1_graph"][_iu_all]) for r in level_w1]
    _M = meta["Xm"].shape[0]
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.6))
    _axes[0].plot(_nodes, _acc, "-o", color=pal["dark"], label="Leiden levels")
    _axes[0].plot([_M], [_rho_gt(dists["W1_graph"])], "o", mfc="white", mec=pal["dark"], mew=1.5, ms=8, label=f"k-means, M = {_M}")
    _axes[0].set(xscale="log", xlabel="number of metacells, M (log scale)", ylabel="Spearman ρ with true distance")
    _axes[0].set_title("accuracy (simulation only)", fontsize=11)
    _axes[1].plot(_nodes, _agree, "-o", color=pal["dark"])
    _axes[1].set(xscale="log", xlabel="number of metacells, M (log scale)", ylabel=f"Spearman ρ with distances\nfrom k-means, M = {_M}")
    _axes[1].set_title("agreement with fine scale (no truth needed)", fontsize=11)
    _axes[2].plot(_nodes, [r["secs"] for r in level_w1], "-o", color=pal["dark"])
    _axes[2].plot([_M], [dist_secs["W1_graph (exact OT per pair)"]], "o", mfc="white", mec=pal["dark"], mew=1.5, ms=8)
    _axes[2].set(xscale="log", yscale="log", xlabel="number of metacells, M (log scale)", ylabel="runtime, all gene pairs [s]")
    _axes[2].set_title("cost of exact graph $W_1$", fontsize=11)
    for _ax in _axes:
        log_ticks(_ax.xaxis)
        _ax.invert_xaxis()  # fine (left) to coarse (right)
    log_ticks(_axes[2].yaxis)
    # one shared legend above the panels (the hollow k-means marker appears in two of them)
    _fig.legend(*_axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    _fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig_panels(_fig)
    readout1 = {
        "nodes": [int(x) for x in _nodes], "acc": [float(x) for x in _acc], "agree": [float(x) for x in _agree],
        "secs": [r["secs"] for r in level_w1], "M": _M, "ref_acc": float(_rho_gt(dists["W1_graph"])),
        "ref_secs": dist_secs["W1_graph (exact OT per pair)"],
    }
    return (readout1,)


@app.cell(hide_code=True)
def _(mo, readout1):
    def _readout1_text(r):
        """Describe the coarsest level whose agreement is still at the fine-scale plateau, and the first drop."""
        n, acc, agree, secs = r["nodes"], r["acc"], r["agree"], r["secs"]
        order = sorted(range(len(n)), key=lambda i: -n[i])  # fine to coarse
        top = max(agree)
        keep = order[0]
        for i in order:
            if agree[i] < top - 0.01:
                break
            keep = i
        drop = next((i for i in order if n[i] < n[keep]), None)
        same = "as accurate as" if acc[keep] >= r["ref_acc"] - 0.01 else "close to"
        out = (
            f"On this dataset, the level with {n[keep]} clusters is {same} {r['M']} k-means metacells "
            f"(ρ = {acc[keep]:.2f} against {r['ref_acc']:.2f}) and takes {secs[keep]:.1f} s instead of "
            f"{r['ref_secs']:.1f} s. "
        )
        if drop is None:
            return out + "No level in the range leaves the plateau, so the coarsest one is the choice."
        if acc[drop] < acc[keep] - 0.01:
            out += (
                f"At {n[drop]} clusters both curves fall (accuracy {acc[keep]:.2f} → {acc[drop]:.2f}, "
                f"agreement {agree[keep]:.2f} → {agree[drop]:.2f}), so the middle panel flags the step "
                "without any ground truth. "
            )
        else:
            out += (
                f"At {n[drop]} clusters agreement falls ({agree[keep]:.2f} → {agree[drop]:.2f}) while accuracy "
                f"barely moves ({acc[keep]:.2f} → {acc[drop]:.2f}): here the middle panel errs on the safe side. "
            )
        if any(agree[i] >= top - 0.01 for i in order if n[i] < n[drop]):
            return out + "A coarser level climbs back to the plateau; read the first drop, not the rebound."
        return out + (
            "Coarser levels stay off the plateau; small rebounds in either curve there are not a reason to "
            "pick them."
        )

    mo.md(
        "The x-axes run from fine (left) to coarse (right). "
        + _readout1_text(readout1)
        + " Because the metacells come from the data's own cluster structure at a known resolution, the "
        "coarsening is a choice you can justify, rather than a number of k-means centres."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Readout 2: $W_1$ on the hierarchy itself

    The hierarchy can also *be* the cell distance. Put the cells at the leaves and a root on top; edge
    lengths are PCA distances between a node's centroid and its parent's. On a tree, the flow problem
    from Part 3 needs no solver: each edge splits the tree in two, so the flow across it must be exactly
    the net expression on the far side:

    $$
    W_1^{\text{tree}}(a, b) = \sum_{e} \ell_e \,\bigl\lvert a(\text{below } e) - b(\text{below } e) \bigr\rvert
    = \lVert \phi(a) - \phi(b) \rVert_1,
    \qquad \phi(a)_e = \ell_e\, a(\text{below } e).
    $$

    Each gene becomes one vector $\phi(a_g)$, and all pairwise distances are a single $L_1$ distance
    computation (Indyk & Thaper 2003; Le *et al.* 2019).

    **Top row, the hierarchy.** On the left it is drawn in PCA: nodes sit at their cluster centroids,
    larger, darker and with thicker lines for coarser levels, each joined to its parent. On the right
    the same tree is lifted into a dendrogram, with the root at the top and the finest clusters along the
    bottom (coloured by mean pseudotime, ordered start to end). Each coarser node sits above the
    clusters it contains, so the levels of the pyramid are the resolutions of the hierarchy.

    **Bottom row, where the tree distance for the gene pair from Part 3 comes from.** Each edge's width
    is its term $\ell_e\,|a(\text{below } e) - b(\text{below } e)|$: the edge length times the
    difference in expression below it. On the dendrogram this is the closed form made visible: for
    genes at opposite ends of the trajectory, the large terms are the coarse edges that separate the
    start from the end, the same route the graph flow took. Edges from single cells to their finest
    cluster are too many to draw; their share of the total is in the title.
    """)
    return


@app.cell(hide_code=True)
def _(
    adata,
    fig_panels,
    gene_a_dd,
    gene_b_dd,
    mo,
    np,
    pal,
    plt,
    tree_node_centroids,
    tree_w1_embedding,
    trees,
):
    _Z = adata.obsm["X_pca"]
    _levels = trees[0]
    _cent, _parents = tree_node_centroids(_Z, _levels)
    _X = np.asarray(adata[:, [gene_a_dd.value, gene_b_dd.value]].X)
    _A = _X / _X.sum(0, keepdims=True)
    _contrib = np.abs(np.diff(tree_w1_embedding(_Z, _A, _levels), axis=1)).ravel()
    _n = adata.n_obs
    _total, _leaf = _contrib.sum(), _contrib[:_n].sum()
    _nlev = len(_parents)

    # per-level edge terms (node -> parent), in the row order of tree_w1_embedding
    _offs = np.cumsum([_n] + [len(p) for p in _parents])
    _terms = [_contrib[_offs[l]:_offs[l + 1]] for l in range(_nlev)]
    _cmax = max(t.max() for t in _terms)

    # dendrogram layout: finest clusters on the bottom row in depth-first order (children sorted by
    # mean pseudotime), each coarser node centred over its children, root on top
    _t = adata.obs["t"].to_numpy()
    _tnode = [np.bincount(lv, weights=_t) / np.bincount(lv) for lv in _levels]
    _kids = [None] + [[np.where(_parents[l - 1] == k)[0] for k in range(_levels[l].max() + 1)] for l in range(1, _nlev + 1)]
    _xpos = [np.zeros(_levels[l].max() + 1) for l in range(_nlev + 1)]
    # leaves in depth-first order (iterative: older marimo cannot resolve a recursive cell-private function)
    _stack, _next_leaf = [(_nlev, 0)], 0
    while _stack:
        _l, _k = _stack.pop()
        if _l == 0:
            _xpos[0][_k] = _next_leaf
            _next_leaf += 1
        else:
            _order = sorted(_kids[_l][_k], key=lambda c, l=_l: _tnode[l - 1][c])
            _stack.extend((_l - 1, c) for c in reversed(_order))
    for _l in range(1, _nlev + 1):
        _xpos[_l] = np.array([_xpos[_l - 1][_kids[_l][k]].mean() for k in range(len(_xpos[_l]))])

    def _shade(l):
        return str(0.7 - 0.55 * l / max(_nlev - 1, 1))

    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for _row in range(2):
        _ax = _axes[_row, 0]
        _ax.scatter(_Z[:, 0], _Z[:, 1], s=1, color=pal["faint"], lw=0, zorder=0)
        _ax.set(xlabel="PC1", ylabel="PC2")
    for _l, _p in enumerate(_parents):
        _c, _cp = _cent[_l][:, :2], _cent[_l + 1][_p][:, :2]
        _xd, _xdp = _xpos[_l], _xpos[_l + 1][_p]
        for _k in range(len(_p)):
            # structure (top row): thickness and shade by level
            _axes[0, 0].plot([_c[_k, 0], _cp[_k, 0]], [_c[_k, 1], _cp[_k, 1]], color=_shade(_l), lw=0.4 + 0.5 * _l, zorder=1)
            _axes[0, 1].plot([_xd[_k], _xdp[_k]], [_l, _l + 1], color=_shade(_l), lw=0.4 + 0.5 * _l, zorder=1)
            # edge terms (bottom row): width by the edge's share of tree W1
            _w = _terms[_l][_k]
            if _w > 1e-3 * _cmax:
                _lw = 0.3 + 7 * _w / _cmax
                _axes[1, 0].plot([_c[_k, 0], _cp[_k, 0]], [_c[_k, 1], _cp[_k, 1]], color=pal["flow"], lw=_lw, zorder=1, solid_capstyle="round")
                _axes[1, 1].plot([_xd[_k], _xdp[_k]], [_l, _l + 1], color=pal["flow"], lw=_lw, zorder=1, solid_capstyle="round")
            else:
                _axes[1, 1].plot([_xd[_k], _xdp[_k]], [_l, _l + 1], color=pal["faint"], lw=0.4, zorder=0)
        _axes[0, 0].scatter(*_c.T, s=4 + 10 * _l, color=_shade(_l), zorder=2, lw=0)
        if _l > 0:
            _axes[0, 1].scatter(_xd, np.full(len(_xd), _l), s=4 + 10 * _l, color=_shade(_l), zorder=2, lw=0)
    _axes[0, 0].scatter(*_cent[-1][:, :2].T, s=80, marker="*", color="black", zorder=3)
    _axes[0, 1].scatter([_xpos[-1][0]], [_nlev], s=120, marker="*", color="black", zorder=3)
    _leaves = _axes[0, 1].scatter(_xpos[0], np.zeros(len(_xpos[0])), c=_tnode[0], cmap="viridis", s=10, zorder=3, lw=0)
    _fig.colorbar(_leaves, ax=_axes[0, 1], label="mean position along drawing, t [0–1]", pad=0.02)

    _ylabels = [f"{len(p)} clusters" for p in _parents] + ["root"]
    for _ax in _axes[:, 1]:
        _ax.set_yticks(range(_nlev + 1), _ylabels, fontsize=9)
        _ax.set_xticks([])
        _ax.spines["bottom"].set_visible(False)
        _ax.set_xlabel("finest clusters, ordered by pseudotime")
    _axes[0, 0].set_title("cluster tree in PCA", fontsize=11)
    _axes[0, 1].set_title("the same tree as a dendrogram (leaves = finest clusters)", fontsize=11)
    _axes[1, 0].set_title(f"edge terms in PCA: tree $W_1$ = {_total:.2f}", fontsize=11)
    _axes[1, 1].set_title(f"edge terms on the dendrogram ({100 * _leaf / _total:.0f}% of $W_1$ is in the undrawn cell → cluster edges)", fontsize=11)
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A check that the closed form is right: solve the flow problem on the tree's own edges (cells, cluster
    nodes, root) with a generic LP solver, and compare with $\lVert\phi(a) - \phi(b)\rVert_1$.
    """)
    return


@app.cell
def _(
    adata,
    beckmann_w1_edges,
    gene_a_dd,
    gene_b_dd,
    np,
    tree_node_centroids,
    tree_w1_embedding,
    trees,
):
    def verify_tree_closed_form():
        Z = adata.obsm["X_pca"]
        X = np.asarray(adata[:, [gene_a_dd.value, gene_b_dd.value]].X)
        A = X / X.sum(0, keepdims=True)
        levels = trees[0]
        cent, parents = tree_node_centroids(Z, levels)
        n = len(levels[0])
        # global node ids: cells first, then each level's nodes, root last
        offsets = np.cumsum([n] + [lv.max() + 1 for lv in levels])
        child, parent = [np.arange(n)], [offsets[0] + levels[0]]
        ell = [np.linalg.norm(Z - cent[0][levels[0]], axis=1)]
        for l, p in enumerate(parents):
            child.append(offsets[l] + np.arange(len(p)))
            parent.append(offsets[l + 1] + p)
            ell.append(np.linalg.norm(cent[l] - cent[l + 1][p], axis=1))
        child, parent, ell = map(np.concatenate, (child, parent, ell))
        delta = np.zeros(offsets[-1])
        delta[:n] = A[:, 0] - A[:, 1]
        phi = tree_w1_embedding(Z, A, levels)
        return {
            "W1_tree via LP on the tree's edges": beckmann_w1_edges(child, parent, ell, delta)[0],
            "W1_tree via closed form ‖φ(a) − φ(b)‖₁": float(np.abs(phi[:, 0] - phi[:, 1]).sum()),
            "tree edges": len(ell),
        }

    verify_tree_closed_form()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Does tree $W_1$ need tree-shaped biology? A cell-cycle test

    A tree distance can cut a loop: two neighbouring cells on either side of a coarse cluster boundary
    are joined only through the root. If that mattered, tree $W_1$ would overstate the distance between
    genes whose expression sits on opposite sides of a boundary.

    The test is a simulated **cell cycle**: 1,500 cells around a ring, and 40 genes, each peaking at one
    phase. The true distance between two genes is the circular distance between their peak phases.
    Both readouts are computed from one hierarchy (resolutions 0.5, 2, 8, 30). **Left:** cells in PCA,
    coloured by true phase (a cyclic colormap, since phase wraps around). **Middle and right:** each
    gene pair, placed by true distance and by the metric. Dark points are pairs whose genes sit in
    *different* top-level clusters, which is where cutting the loop would show up.
    """)
    return


@app.cell
def _(
    ad,
    build_cluster_tree,
    graph_w1_matrix,
    metacell_graph_from_labels,
    np,
    pdist,
    sc,
    squareform,
    tree_w1_embedding,
):
    def simulate_cycle(n_cells=1500, n_genes=40, n_noise=300, seed=0):
        """Cells uniform on a ring; each gene is a bump in circular distance from its peak phase."""
        rng = np.random.default_rng(seed)
        theta = rng.uniform(0, 2 * np.pi, n_cells)
        peak = np.sort(rng.uniform(0, 2 * np.pi, n_genes))
        width, amp = rng.uniform(0.35, 0.6, n_genes), rng.lognormal(0, 0.4, n_genes)
        d_circ = np.abs(np.angle(np.exp(1j * (theta[:, None] - peak[None]))))
        lib = rng.lognormal(np.log(3), 0.3, n_cells)[:, None]
        counts = np.c_[
            rng.poisson(lib * (0.05 + amp * np.exp(-(d_circ**2) / (2 * width**2)))),
            rng.poisson(lib * rng.lognormal(-1, 1, n_noise)[None]),
        ].astype(np.float32)
        a = ad.AnnData(counts)
        a.var_names = [f"cyc{i:02d}" for i in range(n_genes)] + [f"noise{i:03d}" for i in range(n_noise)]
        a.obs["phase"] = theta
        a.uns["peak"] = peak
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
        sc.pp.pca(a, n_comps=20, random_state=0)
        return a

    def ring_test():
        ring = simulate_cycle()
        genes = [g for g in ring.var_names if g.startswith("cyc")]
        peak = ring.uns["peak"]
        truth = np.abs(np.angle(np.exp(1j * (peak[:, None] - peak[None]))))
        Z, X = ring.obsm["X_pca"], np.asarray(ring[:, genes].X)
        A = X / X.sum(0, keepdims=True)
        levels = build_cluster_tree(ring, [0.5, 2, 8, 30], seed=0)[0]
        D_tree = squareform(pdist(tree_w1_embedding(Z, A, levels).T, "cityblock"))
        mid = [lv for lv in levels if 40 <= lv.max() + 1 <= 150][0]
        m = metacell_graph_from_labels(Z, X, mid)
        D_graph = graph_w1_matrix(m["Xm"] / m["Xm"].sum(0, keepdims=True), m["D_geo"])
        top = levels[-2]  # coarsest level below the root
        home = np.array([np.bincount(top, weights=A[:, j], minlength=top.max() + 1).argmax() for j in range(len(genes))])
        return dict(Z=Z, phase=ring.obs["phase"].to_numpy(), truth=truth, D_tree=D_tree, D_graph=D_graph,
                    split=home[:, None] != home[None], n_mid=int(mid.max() + 1), n_top=int(top.max() + 1))

    ring_res = ring_test()
    return (ring_res,)


@app.cell(hide_code=True)
def _(fig_panels, mo, np, pal, pd, plt, ring_res):
    _iu = np.triu_indices(len(ring_res["truth"]), 1)
    _t, _split = ring_res["truth"][_iu], ring_res["split"][_iu]
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 3.8))
    _sc = _axes[0].scatter(*ring_res["Z"][:, :2].T, c=ring_res["phase"], cmap="hsv", s=3, lw=0)
    _fig.colorbar(_sc, ax=_axes[0], label="true phase [rad]")
    _axes[0].set(xlabel="PC1", ylabel="PC2")
    _axes[0].set_title("simulated cell cycle", fontsize=11)
    for _ax, _key, _name in [
        (_axes[1], "D_tree", "tree $W_1$"),
        (_axes[2], "D_graph", "graph $W_1$"),
    ]:
        _d = ring_res[_key][_iu]
        _rho = pd.Series(_d).corr(pd.Series(_t), method="spearman")
        _ax.scatter(_t[~_split], _d[~_split], s=6, color=pal["light"], lw=0, label="same top-level cluster")
        _ax.scatter(_t[_split], _d[_split], s=6, color=pal["dark"], lw=0, label="different top-level clusters")
        _ax.set(xlabel="true distance between peaks [rad]", ylabel=f"{_name} [PCA units]")
        _ax.set_title(f"{_name}:\nρ = {_rho:.3f}", fontsize=11)
    _axes[2].set_title(f"graph $W_1$, {ring_res['n_mid']}-cluster level:\nρ = {pd.Series(ring_res['D_graph'][_iu]).corr(pd.Series(_t), method='spearman'):.3f}", fontsize=11)
    _axes[2].legend(loc=(0.6, 0.15), markerscale=2.5)
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo, ring_res):
    mo.md(rf"""
    Dark pairs (genes in different top-level clusters, {ring_res["n_top"]} clusters here) and light
    pairs follow the same curve in *both* panels. Whatever small offset there is shows up in exact
    graph $W_1$ too, so it is not caused by the tree cutting the loop. Tree $W_1$ ranks gene pairs
    about as well as exact graph $W_1$. The reason is that
    each gene's expression is spread over many clusters, so a single cut edge carries only a small part
    of the transport. Gene programs much narrower than the coarsest clusters are where this could
    break, and graph $W_1$ at a chosen level (readout 1) remains the topology-agnostic fallback.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is the shortcut faithful, and how much does it save?

    First, all five metrics as gene × gene matrices over the evaluation genes (trajectory genes, twins
    and permuted copies):

    | name | compares | cell distance | how |
    |---|---|---|---|
    | `pearson` | expression vectors over cells | none | $1 - r$ (`find_correlated_features`) |
    | `spearman` | expression vectors over cells | none | $1 - \rho_S$ (`find_correlated_features`) |
    | `W1_values` | value histograms | none (real line of values) | 1D $W_1$ on z-scores (`find_correlated_features`) |
    | `W1_graph` | distributions over metacells | geodesic on the metacell graph | exact OT, one solve per pair (POT `ot.emd2`) |
    | `W1_tree` | distributions over cells | multiscale cluster tree | closed form, one $L_1$ distance computation |
    """)
    return


@app.cell
def _(
    adata,
    find_correlated_features,
    gene_dist,
    genes_eval,
    graph_w1_matrix,
    meta,
    np,
    pd,
    pdist,
    squareform,
    time,
    tree_w1_embedding,
    trees,
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
        secs["pearson + spearman + W1_values"] = time.perf_counter() - _t

        _t = time.perf_counter()
        out["W1_graph"] = graph_w1_matrix(gene_dist, meta["D_geo"])
        secs["W1_graph (exact OT per pair)"] = time.perf_counter() - _t

        _t = time.perf_counter()
        _X = np.asarray(sub.X)
        _A = _X / _X.sum(0, keepdims=True)  # distributions over cells, not metacells
        out["W1_tree"] = np.mean(
            [squareform(pdist(tree_w1_embedding(adata.obsm["X_pca"], _A, lv).T, "cityblock")) for lv in trees],
            axis=0,
        )
        secs["W1_tree (given the tree)"] = time.perf_counter() - _t
        return out, secs

    dists, dist_secs = compute_all_distances()
    pd.Series(dist_secs, name="seconds").round(3).to_frame()
    return dist_secs, dists


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Left:** every gene pair as a dot, placed by its exact and its tree distance. What matters is that
    the *ranking* of pairs is preserved, since distances are used through neighbour lists and
    embeddings; a tight monotone cloud means the tree can stand in for the exact distance. **Right:**
    wall-clock cost on a log scale. The tree is built once (the Leiden runs dominate); after that, the
    tree distance for all pairs is one matrix operation.
    """)
    return


@app.cell(hide_code=True)
def _(
    dist_secs,
    dists,
    fig_panels,
    genes_eval,
    log_ticks,
    mo,
    np,
    pal,
    pd,
    plt,
    tree_build_secs,
):
    _iu = np.triu_indices(len(genes_eval), 1)
    _x, _y = dists["W1_graph"][_iu], dists["W1_tree"][_iu]
    _rho = pd.Series(_x).corr(pd.Series(_y), method="spearman")
    _fig, _axes = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1, 1.35]})
    _axes[0].scatter(_x, _y, s=5, color=pal["dark"], alpha=0.35, lw=0)
    _axes[0].set(xlabel="exact graph $W_1$, metacells [PCA units]", ylabel="tree $W_1$, cells [PCA units]")
    _axes[0].set_title(f"all gene pairs: Spearman ρ = {_rho:.3f}", fontsize=11)
    _rows = [
        ("tree $W_1$: all pairs, given the tree", dist_secs["W1_tree (given the tree)"], dict(color=pal["dark"])),
        ("cluster tree: Leiden runs (once)", tree_build_secs, dict(color=pal["light"], edgecolor=pal["dark"], hatch="///", lw=0)),
        ("exact graph $W_1$: one OT solve per pair", dist_secs["W1_graph (exact OT per pair)"], dict(color=pal["mid"])),
    ]
    for _k, (_name, _v, _style) in enumerate(_rows):
        _axes[1].barh(_k, _v, height=0.55, **_style)
        _axes[1].text(_v * 1.15, _k, f"{_v:.3f} s", va="center", fontsize=10)
    _axes[1].set_yticks(range(len(_rows)), [r[0] for r in _rows])
    _axes[1].set_xscale("log")
    _vals = [r[1] for r in _rows]
    _axes[1].set_xlim(min(_vals) / 3, max(_vals) * 8)
    log_ticks(_axes[1].xaxis)
    _axes[1].set_xlabel(f"runtime for {len(genes_eval)} genes = {len(_x)} pairs [s] (log scale)")
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// tip | Part 4 in one line
    Multiscale coarsening replaces an arbitrary metacell count with a hierarchy the data can choose a
    scale from. Exact graph $W_1$ at the coarsest stable level works for any topology. Tree $W_1$ on
    the same hierarchy keeps most of the ranking (ρ ≈ 0.9 with the exact distances on the default drawing, and it held up on a cell cycle) for the cost of one matrix operation.
    ///

    ## Part 5. Scorecard against the known answer

    Every metric is scored against the true distance between gene peaks (geodesic distance along the drawing):

    * **ρ(GT)**: Spearman correlation between the metric and the true distance, over all
      trajectory-gene pairs. Higher is better.
    * **kNN-5 overlap**: average fraction of each trajectory gene's 5 true nearest genes recovered.
    * **twin rank**: median rank of the twin among its parent's neighbours (1 = nearest). Lower is better.
    * **twin rank beyond ε**: the same, ignoring trajectory genes whose peak lies within ε of the
      parent's (ε = three average gene spacings along the trajectory). Those genes are expressed in
      almost the same cells as the parent, so they are as legitimate a neighbour as the twin; a
      distance that is exact up to sampling noise will order them at random. This is the fair version
      of the positive control.
    * **perm→source rank**: median rank of the source gene among the permuted copy's neighbours.
      **Rank 1 means the metric confuses a gene with a cell-shuffled copy of itself.** Higher is better.

    The chart shows the two numbers that separate the families: how well each metric recovers the
    geometry (left) and whether it passes the shuffle test (right).
    """)
    return


@app.cell
def _(adata, dists, genes_eval, np, pd, true_distance):
    def scorecard(dists, var, genes, k=5):
        v = var.loc[genes]
        traj = np.where(v["kind"].to_numpy() == "trajectory")[0]
        twins = np.where(v["kind"].to_numpy() == "twin")[0]
        perms = np.where(v["kind"].to_numpy() == "permuted")[0]
        pt = v["peak_node"].to_numpy()
        gt = true_distance(pt[traj, None], pt[None, traj])
        iu = np.triu_indices(len(traj), 1)
        gt_nn = np.argsort(gt + np.diag(np.full(len(traj), np.inf)), axis=1)[:, :k]
        parent = v["parent"].to_numpy()
        kind = v["kind"].to_numpy()
        # Twin tolerance: trajectory genes whose peak lies within three average gene spacings of the
        # parent's are as close to it as the twin is; at that distance their order is sampling noise.
        eps = 3 * float(np.max(gt)) / max(len(traj), 1)
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

            def twin_rank_beyond_eps(t):
                p = parent[t]
                d = D[p].copy()
                d[p] = np.inf
                near = (kind == "trajectory") & (true_distance(pt[p], pt) < eps)
                d[near & (np.arange(len(d)) != t)] = np.inf
                return int(np.sum(d < d[t]) + 1)

            twin_r = np.median([rank_of(parent[t], t) for t in twins]) if len(twins) else np.nan
            twin_eps = np.median([twin_rank_beyond_eps(t) for t in twins]) if len(twins) else np.nan
            perm_r = np.median([rank_of(p, parent[p]) for p in perms]) if len(perms) else np.nan
            rows[name] = {"ρ(GT)": rho, "kNN-5 overlap": overlap, "twin rank": twin_r,
                          "twin rank beyond ε": twin_eps, "perm→source rank": perm_r}
        out = pd.DataFrame(rows).T
        out.attrs["eps"] = eps
        return out

    score_df = scorecard(dists, adata.var, genes_eval)
    score_df.round(3).sort_values("ρ(GT)", ascending=False)
    return (score_df,)


@app.cell(hide_code=True)
def _(fig_panels, log_ticks, mo, pal, plt, score_df):
    _s = score_df.sort_values("ρ(GT)")
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.3), sharey=True)
    _axes[0].barh(_s.index, _s["ρ(GT)"], color=pal["dark"], height=0.6)
    for _k, _v in enumerate(_s["ρ(GT)"]):
        _axes[0].text(max(_v, 0) + 0.02, _k, f"{_v:.2f}", va="center", fontsize=10)
    _axes[0].set(xlabel="Spearman ρ with true distance (higher = better)", xlim=(min(0, _s["ρ(GT)"].min()) - 0.05, 1.12))
    _axes[1].barh(_s.index, _s["perm→source rank"], color=pal["light"], edgecolor=pal["dark"], hatch="///", lw=0, height=0.6)
    for _k, _v in enumerate(_s["perm→source rank"]):
        _axes[1].text(_v * 1.1, _k, f"{_v:.0f}", va="center", fontsize=10)
    _axes[1].set(xlabel="rank of source gene for its shuffled copy\n(log scale; 1 = fooled)", xscale="log")
    _axes[1].set_xlim(0.7, _s["perm→source rank"].max() * 3)
    _axes[1].tick_params(labelleft=True)  # keep method names when panels stack on a phone
    log_ticks(_axes[1].xaxis)
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * Exact graph $W_1$ has the highest ρ with the true distance; tree $W_1$ comes next, ahead of the
      correlations, and matches or beats exact $W_1$ on kNN-5 overlap.
    * Strict twin rank can exceed 1 for graph $W_1$: its distance between a gene and its twin is not exactly
      0 (sampling noise on the metacells), while genes peaking a few cells away are just as close. Beyond
      ε, the twin is first or close to it for every metric except `W1_values`.
    * Pearson and Spearman pass the shuffle test but recover the geometry less well, because of the
      saturation in Part 3: beyond overlap range, every distance looks the same.
    * `W1_values` has no relation to the true geometry and ranks each permuted copy as its source's
      nearest neighbour, as Part 2 predicted.

    For the gene pair selected in Part 3, the table compares each metric's distance with gene A's
    typical distance to the other genes. `ratio < 1` means "closer than typical".
    """)
    return


@app.cell
def _(dists, gene_a_dd, gene_b_dd, genes_eval, np, pd):
    _ia, _ib = genes_eval.index(gene_a_dd.value), genes_eval.index(gene_b_dd.value)
    _row = {m: dists[m][_ia, _ib] for m in dists}
    _med = {m: np.median(dists[m][_ia][np.arange(len(genes_eval)) != _ia]) for m in dists}
    _tab = pd.DataFrame({"d(A,B)": _row, "median d(A,·)": _med})
    _tab["ratio"] = _tab["d(A,B)"] / _tab["median d(A,·)"]
    _tab.round(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### "Genes like X": which metric, and why

    The everyday use of `find_correlated_features` is a ranking: give it a target gene and read the top
    of the list. Below is that exact call, with each of its three metrics, for a target gene that has a
    cell-shuffled copy in the data. Next to each hit is what it really is: its true distance to the
    target along the trajectory, or a warning if it is the shuffled copy.
    """)
    return


@app.cell
def _(adata, find_correlated_features, genes_eval, np, pd, true_distance):
    def genes_like(target, k=5):
        """Top-k hits for `target` under each metric of find_correlated_features, annotated with the truth."""
        sub = adata[:, genes_eval]
        v = adata.var
        names = list(v.index)

        def describe(g):
            if v.loc[g, "kind"] == "permuted":
                src = names[int(v.loc[g, "parent"])]
                return f"⚠ {g}: cells shuffled" + (" (copy of the target)" if src == target else "")
            d = float(true_distance(v.loc[target, "peak_node"], v.loc[g, "peak_node"]))
            return f"{g}  (true distance {d:.2f})"

        cols = {}
        for metric, label in [("spearman", "spearman"), ("pearson", "pearson"), ("wasserstein", "wasserstein (1D, on values)")]:
            res = find_correlated_features(sub, target, metrics=(metric,))
            hits = [g for g in res.index if g != target][:k]
            cols[label] = [describe(g) for g in hits]
        return pd.DataFrame(cols, index=pd.Index(np.arange(1, k + 1), name="rank"))

    _perm = adata.var.index[adata.var["kind"] == "permuted"][0]
    query_target = adata.var.index[int(adata.var.loc[_perm, "parent"])]
    genes_like(query_target)
    return (query_target,)


@app.cell(hide_code=True)
def _(mo, query_target):
    mo.md(rf"""
    For the target `{query_target}` (with the default data), every Spearman and Pearson hit is a gene
    expressed nearby along the trajectory, although not in exact nearest-first order, because correlation
    saturates (Part 3). The `wasserstein` metric puts the target's own cell-shuffled copy at rank 1: a
    gene with no biological relation to the target, chosen *because* it has the same histogram. Its
    other hits are genes whose histograms happen to look similar, some of them far away along the trajectory.

    **The recommendation, and where each piece of evidence is in this notebook:**

    | claim | where to see it | what to look for |
    |---|---|---|
    | value-Wasserstein ignores which cell is which | Part 2, quantile and pairing panels | $W_1$ is computed from the *sorted* pairing only; no cell identity survives |
    | so it cannot pass the shuffle test | Part 2, shuffle-test bars; first figure, bottom row | $W_1$ is bit-identical before and after shuffling, and 0 between a gene and its shuffled copy |
    | it has no relation to where genes are expressed | Part 5, scorecard | `W1_values` has ρ ≈ 0 with the true distance and ranks every shuffled copy first |
    | Pearson is pulled down by histogram shape | Part 1, split bars | for a monotone link, Pearson's bar is almost all *marginal* mismatch, which says nothing about co-regulation |
    | Spearman removes that shape term | Part 1, raw vs rank scatter and split bars | the rank cloud is a tight diagonal; Spearman's bar has no marginal part (without ties) |
    | but dropout brings it back | Part 1, dropout slider at 0.7 | a marginal segment appears in Spearman's bar: tied zeros cap it below 1 |
    | correlations are local | Part 3, saturation curve; Part 5, scorecard | $1 - r$ saturates once expression stops overlapping; ρ with truth clearly below graph $W_1$ |

    **What this means for `find_correlated_features`:**

    1. **For "what is co-regulated with X", use `metrics=("spearman",)`.** It answers the question
       asked (do the two genes rise and fall in the same cells?) without penalising differences in
       histogram shape. Pearson is reasonable after a variance-stabilising transform, but it still
       counts shape against the pair.
    2. **Watch zero-inflation.** With heavy dropout, Spearman partly measures *detection* (which cells
       have any expression). Check the hits on a smoothed or imputed matrix, or restrict to cells that
       express the target.
    3. **For "where in cell space is X expressed, and which genes share that region", correlation is the
       wrong tool.** It saturates beyond the overlap range. Use graph $W_1$ (Part 3) or its multiscale
       versions (Part 4).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the distances are for: a gene embedding

    GeneTrajectory's last step embeds the *genes* with a diffusion map built from the gene–gene
    distance. This is the downstream use, and it makes the differences visible at a glance. A good
    metric lines the genes up along one curve, in order of their peak position (colour). Twins
    (positive control) should sit on their parents; permuted copies (negative control) should not sit
    on their sources.
    """)
    return


@app.cell
def _(dists, mo):
    embed_dd = mo.ui.dropdown(list(dists), value="W1_graph", label="metric to embed")
    embed_dd
    return (embed_dd,)


@app.cell(hide_code=True)
def _(Line2D, adata, dists, embed_dd, fig_panels, genes_eval, mo, np, pal, plt):
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
    _kinds = _v["kind"].to_numpy()
    _fig, _ax = plt.subplots(figsize=(9, 4.6))
    _m = _kinds == "trajectory"
    _sc = _ax.scatter(_emb[_m, 0], _emb[_m, 1], c=_v["peak_t"].to_numpy()[_m], cmap="viridis", vmin=0, vmax=1,
                      s=36, edgecolor=pal["dark"], lw=0.4)
    _tw, _pm = _kinds == "twin", _kinds == "permuted"
    _ax.scatter(_emb[_tw, 0], _emb[_tw, 1], marker="D", s=80, facecolor="none", edgecolor="black", lw=1.2)
    _ax.scatter(_emb[_pm, 0], _emb[_pm, 1], marker="x", s=70, color="black", lw=1.5)
    _fig.colorbar(_sc, ax=_ax, label="peak position along drawing, t [0–1]", pad=0.02)
    _handles = [
        Line2D([], [], marker="o", ls="", color=pal["mid"], label="trajectory gene"),
        Line2D([], [], marker="D", ls="", markerfacecolor="none", markeredgecolor="black", label="twin (pos. ctrl)"),
        Line2D([], [], marker="x", ls="", color="black", label="permuted (neg. ctrl)"),
    ]
    _ax.legend(handles=_handles, loc=(1.36, 0.3))
    _ax.set(xlabel="DC1", ylabel="DC2", title=f"gene diffusion map: {embed_dd.value}")
    _fig.tight_layout()
    fig_panels(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaways

    | metric | answers | uses cell space? | fooled by cell shuffling? | cost for $g$ genes | notes |
    |---|---|---|---|---|---|
    | Pearson | linear co-variation in the same cells | no | no | one matrix product | penalises histogram shape, $r\le 1-W_2^2/2$; saturates beyond overlap |
    | Spearman | monotone co-variation | no | no | sort + one matrix product | removes shape; ties from dropout cap it below 1 |
    | 1D $W_1$ on values | same value histogram | no | **yes, by construction** | sort per gene | a shape-similarity score, not co-expression |
    | graph $W_1$ (GeneTrajectory) | same region of cell space, in path length | yes, exact | no | $g^2/2$ OT solves on $M$ metacells | the reference metric; needs coarsening |
    | graph $W_1$ at a hierarchy level | same region, at a data-chosen scale | yes, exact at that scale | no | $g^2/2$ OT solves on a coarser graph | principled coarsening; works for loops and branches |
    | tree $W_1$ (multiscale) | same region, under the hierarchy's own distance | yes, multiscale | no | one clustering + one $L_1$ distance computation | closed form; held up on a cell cycle here |

    /// tip | Recommendations (hypotheses to test on your own data)
    1. For **"what is co-regulated with X"**, use Spearman, and not `wasserstein`, which measures
       distribution shape and ranks a gene's cell-shuffled copy first. The evidence, and what it means
       for `find_correlated_features`, is in "Genes like X" above.
    2. For **"order genes along a trajectory / find gene programs in cell space"**, use graph $W_1$.
       Pick the metacell scale from a multiscale hierarchy: the coarsest level that still agrees with
       finer ones. For many genes, tree $W_1$ on the same hierarchy gives a close approximation
       (ρ ≈ 0.9 with the exact ranking on the default drawing) at a tiny fraction of the cost.
    3. **Always run the two controls**: a cell-shuffled copy (must be far from its source) and an
       amplitude-scaled twin (must be near its parent). They take a few lines and catch most
       metric-choice errors.
    ///

    /// note | A caveat about normalisation
    $a_g = x_g / \sum x_g$ keeps where a gene is expressed, not how much, and turns a constant
    background into mass spread over all of cell space (see Part 3). Consider background subtraction
    or thresholding before forming $a_g$.
    ///

    ### References

    * Qu R., *et al.* Gene trajectory inference for single-cell data by optimal transport metrics. *Nat. Biotechnol.* (2025).
    * Santambrogio F. *Optimal Transport for Applied Mathematicians*, ch. 4 (Beckmann formulation). Birkhäuser (2015).
    * Indyk P. & Thaper N. Fast image retrieval via embeddings. *3rd Int. Workshop on Statistical and Computational Theories of Vision* (2003).
    * Le T., Yamada M., Fukumizu K. & Cuturi M. Tree-sliced variants of Wasserstein distances. *NeurIPS* (2019).
    """)
    return


if __name__ == "__main__":
    app.run()
