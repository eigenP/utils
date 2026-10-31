# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "eigenp-utils[single-cell,bonsai]",
# ]
# ///
"""Benchmark: Bonsai cell trees vs. PaCMAP (``single_cell.tl_pacmap``) on simulated data.

Ground truth
------------
Gene log-expression diffuses along a known tree (Brownian motion, per-gene rate
sigma_g), so the expected squared expression distance between two cells is
``sum_g sigma_g^2 * d_tree`` with ``d_tree`` the along-the-tree (patristic) distance.
UMI counts are Poisson with heterogeneous library sizes. Every representation is scored
against ``d_tree``:

* ``spearman``     Spearman rho over all cell pairs (global structure),
* ``knn_recall``   mean |kNN_rep ∩ kNN_true| / k (local structure),
* ``label_purity`` mean fraction of a cell's k nearest neighbours that share its
  ground-truth label (clade at depth 3 / trajectory segment).

Scenarios
---------
* ``hierarchy``   cells are the leaves of a balanced binary tree with random branch
                  lengths (Bonsai's generative model: favourable to Bonsai).
* ``trajectory``  cells are sampled densely along a branching trajectory (7 segments),
                  i.e. ancestors and descendants are both observed (a continuum, the
                  regime manifold learners such as PaCMAP are designed for).

Representations
---------------
* ``pca``          ``preprocess_subset`` (seurat_v3 HVGs, log1p, scale, PCA 32), Euclidean
                   (PaCMAP's input; reference)
* ``pacmap``       ``tl_pacmap(use_rep='X_pca')`` 2-D, Euclidean
* ``bonsai``       Bonsai on Poisson log-quotients with error bars, patristic distances
* ``bonsai_pca``   Bonsai on the PCA scores without error bars (ablation: tree model
                   without the noise model)
Controls
* ``oracle``       Euclidean on the noise-free log expression (positive control: upper
                   bound for any Euclidean method given finite genes)
* ``shuffled``     ``bonsai`` with cell identities permuted (negative control: chance)

Usage
-----
    BONSAI_DIR=/path/to/Bonsai-data-representation \\
        python benchmarks/bonsai_vs_pacmap.py --out benchmarks/results
    python benchmarks/bonsai_vs_pacmap.py --skip-bonsai   # PaCMAP side only
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from eigenp_utils.bonsai_tree import BonsaiTree, poisson_log_quotients, run_bonsai


# ------------------------- simulation -------------------------


@dataclass
class SimData:
    counts: np.ndarray  # cells x genes
    log_q: np.ndarray  # cells x genes, noise-free log expression (before softmax)
    labels: np.ndarray  # cells
    true_tree: BonsaiTree  # ground-truth tree with the cells as vertices
    scenario: str


def _brownian_on_tree(parents, lengths, n_genes, sigma, mu, rng):
    """Values at vertices 0..V-1 given parent[v] < v (root 0)."""
    V = len(parents)
    vals = np.empty((V, n_genes))
    vals[0] = mu
    for v in range(1, V):
        vals[v] = vals[parents[v]] + np.sqrt(lengths[v]) * sigma * rng.standard_normal(n_genes)
    return vals


def simulate(
    scenario: str,
    n_cells: int,
    n_genes: int = 1000,
    frac_informative: float = 0.2,
    sigma_informative: float = 0.3,
    mean_umi: float = 3000.0,
    seed: int = 0,
) -> SimData:
    rng = np.random.default_rng(seed)
    # baseline expression: log-normal spread of gene abundances
    mu = rng.normal(0.0, 1.5, n_genes)
    sigma = np.zeros(n_genes)
    informative = rng.choice(n_genes, int(round(frac_informative * n_genes)), replace=False)
    sigma[informative] = sigma_informative

    parents, lengths, labels_v = [-1], [0.0], [-1]
    cell_vertices = []
    if scenario == "hierarchy":
        depth = int(np.round(np.log2(n_cells)))
        if 2**depth != n_cells:
            raise ValueError("hierarchy needs n_cells = 2^depth")
        frontier = [(0, 0, -1)]  # (vertex, depth, clade label)
        while frontier:
            v, d, lab = frontier.pop(0)
            if d == depth:
                cell_vertices.append(v)
                continue
            for _ in range(2):
                parents.append(v)
                lengths.append(rng.uniform(0.5, 1.5))
                child_lab = len(parents) - 1 if d + 1 == 3 else lab
                labels_v.append(child_lab)
                frontier.append((len(parents) - 1, d + 1, child_lab))
    elif scenario == "trajectory":
        # backbone: root segment (3) -> 2 (length 2) -> 4 tips (length 2)
        seg_parent = [-1, 0, 0, 1, 1, 2, 2]
        seg_len = np.array([3.0, 2, 2, 2, 2, 2, 2])
        seg = rng.choice(7, n_cells, p=seg_len / seg_len.sum())
        pos = rng.uniform(0, 1, n_cells) * seg_len[seg]
        seg_end = {}
        # segments in topological order; within a segment, cells sorted by position
        for s in range(7):
            start = 0 if seg_parent[s] < 0 else seg_end[seg_parent[s]]
            prev, prev_pos = start, 0.0
            for c in np.flatnonzero(seg == s)[np.argsort(pos[seg == s], kind="stable")]:
                parents.append(prev)
                lengths.append(max(pos[c] - prev_pos, 1e-9))
                labels_v.append(s)
                prev, prev_pos = len(parents) - 1, pos[c]
                cell_vertices.append((c, prev))
            parents.append(prev)  # segment end node
            lengths.append(seg_len[s] - prev_pos)
            labels_v.append(s)
            seg_end[s] = len(parents) - 1
        cell_vertices = [v for _, v in sorted(cell_vertices)]  # back to cell order
    else:
        raise ValueError(scenario)

    parents = np.array(parents)
    lengths = np.array(lengths)
    vals = _brownian_on_tree(parents, lengths, n_genes, sigma, mu, rng)
    cell_vertices = np.array(cell_vertices)
    log_q = vals[cell_vertices]

    q = np.exp(log_q - log_q.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    lib = rng.lognormal(np.log(mean_umi), 0.5, size=n_cells)
    counts = rng.poisson(lib[:, None] * q)
    # drop nothing, but keep every cell non-empty
    counts[counts.sum(1) == 0, 0] = 1

    V = len(parents)
    true_tree = BonsaiTree(
        edges=np.column_stack([parents[1:], np.arange(1, V)]),
        lengths=lengths[1:],
        vert_names=[f"v{i}" for i in range(V)],
        cell_vertices=cell_vertices,
        cell_ids=[f"cell_{i}" for i in range(n_cells)],
    )
    return SimData(counts, log_q, np.array(labels_v)[cell_vertices], true_tree, scenario)


# ------------------------- metrics -------------------------


def _knn_from_D(D, k):
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1, kind="stable")[:, :k]


def score(D, D_true, labels, k=10):
    iu = np.triu_indices_from(D, 1)
    rho = spearmanr(D[iu], D_true[iu]).statistic
    nn, nn_true = _knn_from_D(D, k), _knn_from_D(D_true, k)
    recall = np.mean([len(np.intersect1d(a, b)) / k for a, b in zip(nn, nn_true)])
    purity = np.mean(labels[nn] == labels[:, None])
    return {"spearman": float(rho), "knn_recall": float(recall), "label_purity": float(purity)}


# ------------------------- representations -------------------------


def pca_and_pacmap(counts, seed, n_top_genes):
    """Our standard pipeline: ``preprocess_subset`` (seurat_v3 HVGs on counts, log1p,
    scale, PCA 32) followed by ``tl_pacmap`` on ``X_pca``."""
    import warnings

    import anndata as ad

    from eigenp_utils.single_cell import preprocess_subset, tl_pacmap

    adata = ad.AnnData(counts.astype(np.float32))
    adata.layers["counts"] = adata.X.copy()
    t0 = time.perf_counter()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.filterwarnings("ignore", message="Overwriting .X")
        adata = preprocess_subset(adata, n_top_genes=n_top_genes, random_state=seed)
    t_pca = time.perf_counter() - t0
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        tl_pacmap(adata, use_rep="X_pca", random_state=seed)
    t_pacmap = time.perf_counter() - t0
    return adata.obsm["X_pca"], adata.obsm["X_pacmap"], t_pca, t_pacmap


def run_one(scenario, n_cells, mean_umi, seed, k, skip_bonsai, bonsai_kwargs, workdir, n_top_genes=300):
    sim = simulate(scenario, n_cells, mean_umi=mean_umi, seed=seed)
    D_true = sim.true_tree.patristic_distances()
    rows = []

    def add(method, D, runtime):
        rows.append({
            "scenario": scenario, "n_cells": n_cells, "mean_umi": mean_umi, "seed": seed,
            "method": method, "runtime_s": runtime, **score(D, D_true, sim.labels, k),
        })

    add("oracle", squareform(pdist(sim.log_q)), 0.0)
    X_pca, X_pacmap, t_pca, t_pacmap = pca_and_pacmap(sim.counts, seed, n_top_genes)
    add("pca", squareform(pdist(X_pca)), t_pca)
    add("pacmap", squareform(pdist(X_pacmap)), t_pca + t_pacmap)

    if not skip_bonsai:
        tag = f"{scenario}_n{n_cells}_umi{int(mean_umi)}_s{seed}"
        x, s = poisson_log_quotients(sim.counts)
        t0 = time.perf_counter()
        tree = run_bonsai(x, s, results_folder=workdir / f"{tag}_ltq", **bonsai_kwargs)
        D_b = tree.patristic_distances()
        add("bonsai", D_b, time.perf_counter() - t0)

        perm = np.random.default_rng(seed + 1000).permutation(n_cells)
        add("shuffled", D_b[np.ix_(perm, perm)], 0.0)

        t0 = time.perf_counter()
        tree = run_bonsai(X_pca, None, results_folder=workdir / f"{tag}_pca", **bonsai_kwargs)
        add("bonsai_pca", tree.patristic_distances(), t_pca + time.perf_counter() - t0)
    return rows


# ------------------------- plotting -------------------------


METHOD_ORDER = ["oracle", "pca", "pacmap", "bonsai", "bonsai_pca", "shuffled"]


def plot(df, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["spearman", "knn_recall", "label_purity", "runtime_s"]
    scen = sorted(df["scenario"].unique())
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    fig, axes = plt.subplots(len(scen), len(metrics), figsize=(4 * len(metrics), 3.2 * len(scen)),
                             squeeze=False, constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for i, sname in enumerate(scen):
        sub = df[df["scenario"] == sname]
        umis = sorted(sub["mean_umi"].unique())
        width = 0.8 / len(umis)
        for j, met in enumerate(metrics):
            ax = axes[i, j]
            for u_i, umi in enumerate(umis):
                g = sub[sub["mean_umi"] == umi].groupby("method")[met]
                mean, sd = g.mean().reindex(methods), g.std().reindex(methods).fillna(0)
                x = np.arange(len(methods)) + (u_i - (len(umis) - 1) / 2) * width
                ax.bar(x, mean, width, yerr=sd, color=colors(u_i), label=f"mean UMI {int(umi)}")
                # individual seeds
                for m_i, m in enumerate(methods):
                    vals = sub[(sub["mean_umi"] == umi) & (sub["method"] == m)][met]
                    ax.scatter(np.full(len(vals), x[m_i]), vals, s=8, color="k", zorder=3)
            ax.set_xticks(np.arange(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha="right")
            ax.set_title(f"{sname}: {met}")
            if met == "runtime_s":
                ax.set_yscale("log")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------- main -------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--scenarios", nargs="+", default=["hierarchy", "trajectory"])
    p.add_argument("--n-cells", type=int, default=256)
    p.add_argument("--mean-umi", type=float, nargs="+", default=[1500.0, 6000.0])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-top-genes", type=int, default=300)
    p.add_argument("--skip-bonsai", action="store_true")
    p.add_argument("--bonsai-dir", default=None)
    p.add_argument("--n-procs", type=int, default=1)
    p.add_argument("--nnn-n-randommoves", type=int, default=1000)
    p.add_argument("--nnn-n-randomtrees", type=int, default=10)
    p.add_argument("--timeout", type=float, default=None)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    workdir = args.out / "bonsai_runs"
    bonsai_kwargs = dict(
        bonsai_dir=args.bonsai_dir, n_procs=args.n_procs, timeout=args.timeout,
        nnn_n_randommoves=args.nnn_n_randommoves, nnn_n_randomtrees=args.nnn_n_randomtrees,
    )
    # numba compiles PaCMAP on first use; keep that out of the timed runs
    pca_and_pacmap(simulate("hierarchy", 64, mean_umi=3000.0, seed=123).counts, 0, args.n_top_genes)

    rows = []
    for scenario in args.scenarios:
        for umi in args.mean_umi:
            for seed in range(args.seeds):
                t0 = time.perf_counter()
                rows += run_one(scenario, args.n_cells, umi, seed, args.k,
                                args.skip_bonsai, bonsai_kwargs, workdir, args.n_top_genes)
                print(f"{scenario} umi={umi:g} seed={seed}: {time.perf_counter() - t0:.1f} s", flush=True)
                pd.DataFrame(rows).to_csv(args.out / "bonsai_vs_pacmap.csv", index=False)

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["scenario", "mean_umi", "method"])[["spearman", "knn_recall", "label_purity", "runtime_s"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary = summary.reindex(METHOD_ORDER, level="method")
    print(summary.to_string())
    (args.out / "bonsai_vs_pacmap_summary.txt").write_text(summary.to_string() + "\n")
    meta = {"args": {k: str(v) for k, v in vars(args).items()},
            "chance_knn_recall": args.k / (args.n_cells - 1)}
    (args.out / "bonsai_vs_pacmap_meta.json").write_text(json.dumps(meta, indent=2))
    plot(df, args.out / "bonsai_vs_pacmap.png")


if __name__ == "__main__":
    main()
