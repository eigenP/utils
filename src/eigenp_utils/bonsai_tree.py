# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "scipy",
#     "pandas",
#     # Bonsai's own run-time requirements (it runs in this interpreter by default)
#     "scikit-learn",
#     "matplotlib",
#     "ruamel.yaml",
#     "psutil",
# ]
# ///
"""Run Bonsai cell-tree reconstruction and work with its trees.

Bonsai (de Groot, Morillo Leonardo, Pachkov & van Nimwegen, Nat. Biotechnol. 2026,
https://doi.org/10.1038/s41587-026-03220-2; code https://doi.org/10.5281/zenodo.20370956)
reconstructs the maximum-likelihood tree whose leaves (and possibly internal nodes)
are the cells. Its model is Brownian diffusion of the features along the branches,
observed with Gaussian measurement noise with known per-entry standard deviations, so
it needs a feature estimate *and an error bar* for every (cell, feature).

Bonsai is not a pip package: it is a set of scripts run from a clone of
https://github.com/dhdegroot/Bonsai-data-representation. This module

* writes Bonsai's input files (features x cells TSV matrices, IDs),
* runs ``bonsai/create_config_file.py`` and ``bonsai/bonsai_main.py`` in a subprocess
  (optionally under ``mpiexec`` with ``mpi4py``),
* reads the final tree (``edgeInfo.txt`` / ``vertInfo.txt`` / ``tree.nwk``) into a
  :class:`BonsaiTree`, which gives patristic (along-the-tree) distances and k nearest
  neighbours between cells,
* stores/reloads the tree in an AnnData object (``tl_bonsai`` / ``get_bonsai_tree``).

The location of the clone is given by ``bonsai_dir=`` or the ``BONSAI_DIR`` environment
variable. Install Bonsai's Python requirements with ``pip install eigenp-utils[bonsai]``.
The module itself needs only numpy / scipy / pandas; AnnData is used duck-typed, so it
does not depend on the ``[single-cell]`` extra.

Bonsai is licensed CC BY-NC 4.0 (non-commercial use only); it is not redistributed here.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

__all__ = [
    "BonsaiTree",
    "find_bonsai_dir",
    "poisson_log_quotients",
    "write_bonsai_input",
    "run_bonsai",
    "tl_bonsai",
    "get_bonsai_tree",
]

BONSAI_REPO_URL = "https://github.com/dhdegroot/Bonsai-data-representation"
_CELL_PREFIX = "cell_"
_INTERNAL_PREFIX = "internal_"


# ------------------------- Locating Bonsai -------------------------


def find_bonsai_dir(bonsai_dir: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Return the root of a Bonsai clone (the folder that contains ``bonsai/bonsai_main.py``).

    Looks at ``bonsai_dir`` first, then at the ``BONSAI_DIR`` environment variable.
    """
    candidate = bonsai_dir if bonsai_dir is not None else os.environ.get("BONSAI_DIR")
    if candidate is None or str(candidate) == "":
        raise FileNotFoundError(
            "Bonsai is not pip-installable; clone it and point to the clone with "
            "`bonsai_dir=` or the BONSAI_DIR environment variable:\n"
            f"    git clone {BONSAI_REPO_URL}\n"
            "    pip install 'eigenp-utils[bonsai]'   # Bonsai's Python requirements\n"
            "    export BONSAI_DIR=$PWD/Bonsai-data-representation"
        )
    root = Path(candidate).expanduser().resolve()
    for rel in ("bonsai/bonsai_main.py", "bonsai/create_config_file.py"):
        if not (root / rel).is_file():
            raise FileNotFoundError(
                f"'{root}' does not look like a Bonsai clone: '{rel}' is missing "
                f"(expected a clone of {BONSAI_REPO_URL})."
            )
    return root


# ------------------------- Input preparation -------------------------


def poisson_log_quotients(
    counts, pseudocount: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Log transcription quotients with Poisson error bars, as a quick Bonsai input.

    For a count ``n_gc ~ Poisson(N_c q_gc)`` (``N_c`` the cell's total count), the
    maximum-likelihood estimate of ``log q_gc`` is ``log(n_gc / N_c)`` and its
    asymptotic (Fisher-information) standard deviation is ``1 / sqrt(N_c q_gc)``.
    With a pseudocount ``a`` to keep zeros finite::

        x_gc = log((n_gc + a) / N_c),     sd_gc = 1 / sqrt(n_gc + a)

    This is a Laplace (Gaussian) approximation of the Poisson likelihood in log
    space. It is crude for zero and single counts, where the likelihood is strongly
    one-sided, and it ignores the shrinkage that Sanity's Bayesian model gives; the
    Bonsai authors recommend running Sanity (``-e 1 -v_m MAP``) and passing its output
    folder as ``run_bonsai(sanity_dir=...)``. Use this function when Sanity is not
    available.

    Parameters
    ----------
    counts
        Raw counts, cells x genes (dense or scipy sparse), non-negative.
    pseudocount
        Added to every count (``a`` above). Must be > 0.

    Returns
    -------
    (features, stds)
        Two dense float64 arrays of shape cells x genes.
    """
    if pseudocount <= 0:
        raise ValueError("pseudocount must be > 0 (it keeps log(0) and 1/sqrt(0) finite).")
    if sp.issparse(counts):
        counts = counts.toarray()
    n = np.asarray(counts, dtype=np.float64)
    if n.ndim != 2:
        raise ValueError(f"counts must be 2-D (cells x genes), got shape {n.shape}.")
    if not np.all(np.isfinite(n)) or np.any(n < 0):
        raise ValueError("counts must be finite and non-negative.")
    totals = n.sum(axis=1)
    if np.any(totals <= 0):
        bad = np.flatnonzero(totals <= 0)
        raise ValueError(
            f"{bad.size} cell(s) have zero total counts (e.g. row {bad[0]}); remove them first."
        )
    n_a = n + pseudocount
    features = np.log(n_a) - np.log(totals)[:, None]
    stds = 1.0 / np.sqrt(n_a)
    return features, stds


def _as_dense_2d(x, name: str) -> np.ndarray:
    if sp.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D (cells x features), got shape {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return x


def write_bonsai_input(
    folder: Union[str, os.PathLike],
    features,
    stds=None,
    feature_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Write Bonsai's input files for a general (non-Sanity) dataset.

    Bonsai reads features x cells, tab-separated, header-less matrices. ``features``
    and ``stds`` here are cells x features (the AnnData convention) and are transposed
    on writing. Cells are written as ``cell_0 ... cell_{C-1}`` (Bonsai rejects IDs that
    contain Newick control characters), so results map back to input rows by index.

    Parameters
    ----------
    folder
        Output folder (created if needed).
    features
        Cells x features estimates (dense or sparse).
    stds
        Cells x features standard deviations of the measurement noise, strictly
        positive. ``None``: Bonsai assumes negligible noise and keeps every feature.
    feature_ids
        Optional names for the features (written to ``geneID.txt``).

    Returns
    -------
    dict with ``filenames_data`` (the value for Bonsai's config), ``cell_ids`` and
    ``n_cells`` / ``n_features``.
    """
    X = _as_dense_2d(features, "features")
    n_cells, n_features = X.shape
    if n_cells < 3:
        raise ValueError(f"Bonsai needs at least 3 cells, got {n_cells}.")
    if stds is not None:
        S = _as_dense_2d(stds, "stds")
        if S.shape != X.shape:
            raise ValueError(f"stds shape {S.shape} does not match features shape {X.shape}.")
        if np.any(S <= 0):
            raise ValueError("stds must be strictly positive.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    np.savetxt(folder / "features.txt", X.T, delimiter="\t", fmt="%.10g")
    filenames = "features.txt"
    if stds is not None:
        np.savetxt(folder / "standard_deviations.txt", S.T, delimiter="\t", fmt="%.10g")
        filenames += ",standard_deviations.txt"

    cell_ids = [f"{_CELL_PREFIX}{i}" for i in range(n_cells)]
    (folder / "cellID.txt").write_text("\n".join(cell_ids) + "\n")
    if feature_ids is not None:
        feature_ids = [str(g) for g in feature_ids]
        if len(feature_ids) != n_features:
            raise ValueError(
                f"{len(feature_ids)} feature_ids for {n_features} features."
            )
        (folder / "geneID.txt").write_text("\n".join(feature_ids) + "\n")
    return {
        "filenames_data": filenames,
        "cell_ids": cell_ids,
        "n_cells": n_cells,
        "n_features": n_features,
    }


# ------------------------- Tree container -------------------------


@dataclass
class BonsaiTree:
    """A Bonsai tree: vertices (cells and internal nodes) joined by weighted edges.

    Attributes
    ----------
    edges
        ``(E, 2)`` int array of vertex indices (parent, child).
    lengths
        ``(E,)`` branch lengths (Bonsai's diffusion times, in its rescaled units).
    vert_names
        Name of every vertex, indexed by vertex index. Cells keep their IDs; internal
        nodes are called ``internal_<k>``.
    cell_vertices
        ``(C,)`` vertex index of each cell, in the order of ``cell_ids``. A cell can be
        an internal vertex when Bonsai places it on the path between other cells.
    cell_ids
        Cell IDs in the order of the input rows.
    newick
        Newick string written by Bonsai ('' if not available).
    folder
        Folder the tree was read from, if any.
    """

    edges: np.ndarray
    lengths: np.ndarray
    vert_names: np.ndarray
    cell_vertices: np.ndarray
    cell_ids: np.ndarray
    newick: str = ""
    folder: Optional[str] = None
    _graph: Optional[sp.csr_matrix] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.lengths = np.asarray(self.lengths, dtype=np.float64).ravel()
        self.vert_names = np.asarray(self.vert_names, dtype=str)
        self.cell_vertices = np.asarray(self.cell_vertices, dtype=np.int64).ravel()
        self.cell_ids = np.asarray(self.cell_ids, dtype=str)
        if self.edges.shape[0] != self.lengths.shape[0]:
            raise ValueError("edges and lengths have different lengths.")
        if self.edges.shape[0] != self.n_vertices - 1:
            raise ValueError(
                f"A tree on {self.n_vertices} vertices has {self.n_vertices - 1} edges, "
                f"got {self.edges.shape[0]}."
            )
        if self.cell_vertices.shape[0] != self.cell_ids.shape[0]:
            raise ValueError("cell_vertices and cell_ids have different lengths.")
        if np.any(self.lengths < 0) or not np.all(np.isfinite(self.lengths)):
            raise ValueError("Branch lengths must be finite and non-negative.")

    # -- basic properties --

    @property
    def n_vertices(self) -> int:
        return int(self.vert_names.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cell_vertices.shape[0])

    @property
    def total_length(self) -> float:
        return float(self.lengths.sum())

    def adjacency(self) -> sp.csr_matrix:
        """Symmetric ``(V, V)`` sparse adjacency with branch lengths as weights.

        Zero-length branches (Bonsai merges most of them, but a cell may sit at
        distance 0 from its parent) get a weight of 1e-300: csgraph treats a stored
        zero as an edge, but any ``eliminate_zeros`` or dense round-trip would turn it
        into a missing edge (infinite distance).
        """
        if self._graph is None:
            w = np.maximum(self.lengths, 1e-300)
            i, j = self.edges[:, 0], self.edges[:, 1]
            V = self.n_vertices
            A = sp.coo_matrix(
                (np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                shape=(V, V),
            ).tocsr()
            self._graph = A
        return self._graph

    # -- distances --

    def patristic_distances(
        self,
        rows: Optional[Sequence[int]] = None,
        cols: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Along-the-tree distances between cells (sums of branch lengths on the path).

        Parameters
        ----------
        rows, cols
            Cell indices (positions in ``cell_ids``). ``None`` means all cells.

        Returns
        -------
        ``(len(rows), len(cols))`` dense array. Memory is ``len(rows) * n_vertices``
        floats during the computation; pass ``rows`` in chunks for very large trees.
        """
        rows = np.arange(self.n_cells) if rows is None else np.asarray(rows, dtype=np.int64)
        cols = np.arange(self.n_cells) if cols is None else np.asarray(cols, dtype=np.int64)
        D = dijkstra(
            self.adjacency(), directed=False, indices=self.cell_vertices[rows]
        )
        D = np.atleast_2d(D)[:, self.cell_vertices[cols]]
        # undo the 1e-300 stand-in for zero-length branches
        D[D < 1e-250] = 0.0
        return D

    def kneighbors(
        self, n_neighbors: int = 15, chunk_size: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """k nearest cells of every cell by patristic distance (self excluded).

        Ties are broken by cell index (stable sort), so the result is deterministic.

        Returns
        -------
        (indices, distances), both ``(n_cells, n_neighbors)``, sorted by distance.
        """
        C = self.n_cells
        if not 1 <= n_neighbors < C:
            raise ValueError(f"n_neighbors must be in [1, {C - 1}], got {n_neighbors}.")
        idx = np.empty((C, n_neighbors), dtype=np.int64)
        dist = np.empty((C, n_neighbors), dtype=np.float64)
        for start in range(0, C, chunk_size):
            rows = np.arange(start, min(start + chunk_size, C))
            D = self.patristic_distances(rows=rows)
            D[np.arange(rows.size), rows] = np.inf  # exclude self
            order = np.argsort(D, axis=1, kind="stable")[:, :n_neighbors]
            idx[rows] = order
            dist[rows] = np.take_along_axis(D, order, axis=1)
        return idx, dist

    # -- I/O --

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, os.PathLike],
        cell_ids: Optional[Sequence[str]] = None,
    ) -> "BonsaiTree":
        """Read a Bonsai tree folder (``edgeInfo.txt``, ``vertInfo.txt``, ``tree.nwk``).

        Parameters
        ----------
        folder
            A ``final_bonsai*`` (or ``intermediate_bonsai*``) folder.
        cell_ids
            Cell IDs in the desired order. ``None``: every vertex not named
            ``internal_*``, in vertex order.
        """
        folder = Path(folder)
        edge_df = pd.read_csv(folder / "edgeInfo.txt", sep="\t", header=None)
        if edge_df.shape[1] < 3:
            raise ValueError(f"{folder / 'edgeInfo.txt'} must have 3 columns (vert, vert, length).")
        vert_df = pd.read_csv(
            folder / "vertInfo.txt", sep="\t", dtype={"vertName": str}, keep_default_na=False
        )
        if not {"vertInd", "vertName"} <= set(vert_df.columns):
            raise ValueError(f"{folder / 'vertInfo.txt'} lacks vertInd / vertName columns.")

        V = len(vert_df)
        vert_ind = vert_df["vertInd"].to_numpy(dtype=np.int64)
        if not np.array_equal(np.sort(vert_ind), np.arange(V)):
            raise ValueError("vertInfo.txt vertex indices must be 0..V-1.")
        vert_names = np.empty(V, dtype=object)
        vert_names[vert_ind] = vert_df["vertName"].astype(str).to_numpy()
        vert_names = vert_names.astype(str)

        name_to_vert = {name: i for i, name in enumerate(vert_names)}
        if len(name_to_vert) != V:
            raise ValueError("vertInfo.txt has duplicate vertex names.")
        if cell_ids is None:
            cell_ids = [n for n in vert_names if not n.startswith(_INTERNAL_PREFIX)]
        cell_ids = [str(c) for c in cell_ids]
        missing = [c for c in cell_ids if c not in name_to_vert]
        if missing:
            raise ValueError(
                f"{len(missing)} cell ID(s) are not vertices of the tree, e.g. '{missing[0]}'."
            )
        cell_vertices = np.array([name_to_vert[c] for c in cell_ids], dtype=np.int64)

        nwk_path = folder / "tree.nwk"
        newick = nwk_path.read_text().strip() if nwk_path.is_file() else ""
        return cls(
            edges=edge_df.iloc[:, :2].to_numpy(dtype=np.int64),
            lengths=edge_df.iloc[:, 2].to_numpy(dtype=np.float64),
            vert_names=vert_names,
            cell_vertices=cell_vertices,
            cell_ids=np.asarray(cell_ids, dtype=str),
            newick=newick,
            folder=str(folder),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict of arrays and strings (h5ad-serialisable, e.g. for ``adata.uns``)."""
        return {
            "edges": self.edges,
            "lengths": self.lengths,
            "vert_names": self.vert_names,
            "cell_vertices": self.cell_vertices,
            "cell_ids": self.cell_ids,
            "newick": self.newick,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BonsaiTree":
        return cls(
            edges=np.asarray(d["edges"]),
            lengths=np.asarray(d["lengths"]),
            vert_names=np.asarray(d["vert_names"]),
            cell_vertices=np.asarray(d["cell_vertices"]),
            cell_ids=np.asarray(d["cell_ids"]),
            newick=str(d.get("newick", "")),
        )


# ------------------------- Running Bonsai -------------------------


def _find_final_folder(results_folder: Path, zscore_cutoff: float) -> Path:
    # Bonsai names it 'final_bonsai' + ('_zscore<cutoff>' if cutoff > 0)
    expected = "final_bonsai" + (f"_zscore{zscore_cutoff}" if zscore_cutoff > 0 else "")
    if (results_folder / expected / "edgeInfo.txt").is_file():
        return results_folder / expected
    finals = sorted(
        (p for p in results_folder.glob("final_bonsai*") if (p / "edgeInfo.txt").is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    if not finals:
        raise RuntimeError(
            f"Bonsai finished but no 'final_bonsai*' folder with edgeInfo.txt was found in "
            f"{results_folder}."
        )
    return finals[-1]


def _bool(x: bool) -> str:
    return "True" if x else "False"


def _tail(path: Path, n_lines: int = 40) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


def run_bonsai(
    features=None,
    stds=None,
    *,
    sanity_dir: Optional[Union[str, os.PathLike]] = None,
    feature_ids: Optional[Sequence[str]] = None,
    results_folder: Optional[Union[str, os.PathLike]] = None,
    bonsai_dir: Optional[Union[str, os.PathLike]] = None,
    python: Optional[str] = None,
    n_procs: int = 1,
    mpiexec: str = "mpiexec",
    zscore_cutoff: float = 1.0,
    rescale_by_var: bool = True,
    use_knn: int = 10,
    nnn_n_randommoves: int = 1000,
    nnn_n_randomtrees: int = 10,
    UB_ellipsoid_size: float = 1.0,
    spr_strategy: str = "large_tree",
    dataset: str = "eigenp_bonsai",
    verbose: bool = False,
    timeout: Optional[float] = None,
) -> BonsaiTree:
    """Reconstruct a Bonsai tree and return it as a :class:`BonsaiTree`.

    Give either ``features`` (cells x features estimates, with ``stds`` as error bars)
    or ``sanity_dir`` (an output folder of Sanity run with ``-e 1 -v_m MAP``).

    Parameters
    ----------
    features, stds
        Cells x features estimates and their measurement-noise standard deviations.
        Bonsai's likelihood assumes a Gaussian with these means and SDs per entry,
        independent across entries. Without ``stds`` Bonsai assumes negligible noise
        (and does not filter features by signal-to-noise). See
        :func:`poisson_log_quotients` for a quick estimate from raw counts.
    sanity_dir
        Sanity output folder (Bonsai's recommended scRNA-seq input).
    feature_ids
        Feature names (only with ``features``).
    results_folder
        Where Bonsai writes its results. ``None``: a new temporary folder (kept, its
        path is in ``tree.folder``).
    bonsai_dir
        Root of the Bonsai clone; default: ``$BONSAI_DIR``.
    python
        Interpreter that runs Bonsai (default: this one). Use it to point at a separate
        environment, e.g. the Python 3.14 conda env the Bonsai README recommends.
    n_procs
        > 1 runs ``mpiexec -n n_procs python -m mpi4py bonsai_main.py`` (needs OpenMPI
        and mpi4py; without mpi4py Bonsai silently runs on one core).
    zscore_cutoff, rescale_by_var, use_knn, nnn_n_randommoves, nnn_n_randomtrees,
    UB_ellipsoid_size, spr_strategy
        Bonsai's run configuration; defaults are Bonsai's (see its README). Lowering
        ``nnn_n_randommoves`` / ``nnn_n_randomtrees`` makes small test runs much faster.
    dataset
        Dataset identifier written to Bonsai's metadata.
    verbose
        Bonsai's verbose flag. Bonsai's output goes to ``bonsai_run.log`` in
        ``results_folder`` either way.
    timeout
        Seconds before the run is killed (``subprocess.TimeoutExpired``).

    Returns
    -------
    BonsaiTree
        ``cell_ids`` are ``cell_0 ... cell_{C-1}`` in input row order (or Sanity's
        ``cellID.txt`` order for ``sanity_dir``).
    """
    if (features is None) == (sanity_dir is None):
        raise ValueError("Give exactly one of `features` or `sanity_dir`.")
    if sanity_dir is not None and (stds is not None or feature_ids is not None):
        raise ValueError("`stds` / `feature_ids` are read from `sanity_dir`; do not pass them.")
    if n_procs < 1:
        raise ValueError("n_procs must be >= 1.")

    root = find_bonsai_dir(bonsai_dir)
    python = python or sys.executable
    if results_folder is None:
        results_folder = tempfile.mkdtemp(prefix="bonsai_")
    results_folder = Path(results_folder).expanduser().resolve()
    results_folder.mkdir(parents=True, exist_ok=True)
    stale = [p.name for p in results_folder.glob("*_bonsai*") if p.is_dir()]
    if stale:
        raise FileExistsError(
            f"{results_folder} already holds Bonsai results ({stale[0]}, ...); Bonsai would "
            "pick them up instead of reconstructing. Use an empty results_folder."
        )

    if sanity_dir is not None:
        data_folder = Path(sanity_dir).expanduser().resolve()
        if not data_folder.is_dir():
            raise FileNotFoundError(f"sanity_dir '{data_folder}' does not exist.")
        filenames_data = "delta.txt,d_delta.txt"  # ignored by Bonsai for Sanity input
        cell_file = data_folder / "cellID.txt"
        cell_ids = (
            [line.split("\t")[0] for line in cell_file.read_text().splitlines() if line]
            if cell_file.is_file()
            else None
        )
    else:
        data_folder = results_folder / "input"
        info = write_bonsai_input(data_folder, features, stds, feature_ids=feature_ids)
        filenames_data = info["filenames_data"]
        cell_ids = info["cell_ids"]

    config_path = results_folder / "bonsai_config.yaml"
    log_path = results_folder / "bonsai_run.log"
    config_cmd = [
        python, "bonsai/create_config_file.py",
        "--new_yaml_path", str(config_path),
        "--dataset", dataset,
        "--data_folder", str(data_folder) + os.sep,
        "--filenames_data", filenames_data,
        "--results_folder", str(results_folder) + os.sep,
        "--input_is_sanity_output", _bool(sanity_dir is not None),
        "--verbose", _bool(verbose),
        "--zscore_cutoff", repr(float(zscore_cutoff)),
        "--rescale_by_var", _bool(rescale_by_var),
        "--use_knn", str(int(use_knn)),
        "--nnn_n_randommoves", str(int(nnn_n_randommoves)),
        "--nnn_n_randomtrees", str(int(nnn_n_randomtrees)),
        "--UB_ellipsoid_size", repr(float(UB_ellipsoid_size)),
        "--spr_strategy", spr_strategy,
        "--pickup_intermediate", "False",
    ]
    main_args = ["bonsai/bonsai_main.py", "--config_filepath", str(config_path), "--step", "all"]
    if n_procs > 1:
        if shutil.which(mpiexec) is None:
            raise FileNotFoundError(f"n_procs={n_procs} needs '{mpiexec}' on PATH (OpenMPI).")
        run_cmd = [mpiexec, "-n", str(n_procs), python, "-m", "mpi4py", *main_args]
    else:
        run_cmd = [python, *main_args]

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        for cmd in (config_cmd, run_cmd):
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            proc = subprocess.run(
                cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout
            )
            if proc.returncode != 0:
                log.flush()
                raise RuntimeError(
                    f"Bonsai step `{' '.join(cmd[:3])} ...` failed with exit code "
                    f"{proc.returncode}. Last lines of {log_path}:\n{_tail(log_path)}"
                )
    elapsed = time.perf_counter() - t0

    final = _find_final_folder(results_folder, float(zscore_cutoff))
    tree = BonsaiTree.from_folder(final, cell_ids=cell_ids)
    tree.folder = str(final)
    if verbose:
        print(
            f"Bonsai finished in {elapsed:.1f} s: {tree.n_cells} cells, "
            f"{tree.n_vertices} vertices. Results in {final}."
        )
    return tree


# ------------------------- AnnData convenience -------------------------


def _get_matrix(adata, key, what: str):
    if isinstance(key, (np.ndarray, sp.spmatrix)) or hasattr(key, "toarray"):
        M = key
    elif key == "X":
        M = adata.X
    elif key in adata.obsm:
        M = adata.obsm[key]
    elif key in adata.layers:
        M = adata.layers[key]
    else:
        raise KeyError(f"{what} '{key}' is not 'X', an obsm key or a layer.")
    if hasattr(M, "to_numpy"):
        M = M.to_numpy()
    if M.shape[0] != adata.n_obs:
        raise ValueError(f"{what} has {M.shape[0]} rows for {adata.n_obs} cells.")
    return M


def tl_bonsai(
    adata,
    use_rep: Union[str, np.ndarray] = "X",
    stds: Optional[Union[str, np.ndarray]] = None,
    key_added: str = "bonsai",
    **run_kwargs,
) -> BonsaiTree:
    """Run Bonsai on an AnnData object and store the tree in ``adata.uns[key_added]``.

    Parameters
    ----------
    adata
        AnnData (used duck-typed; ``[single-cell]`` is not required).
    use_rep
        Features: ``'X'``, an ``obsm`` key, a layer name, or an array (cells x features).
        Bonsai is meant for per-feature estimates with error bars (e.g. log
        transcription quotients from :func:`poisson_log_quotients` or Sanity), not for
        PCA scores: PCA mixes features with different noise levels, and the error bars
        are then unknown (``stds=None`` makes Bonsai treat them as negligible).
    stds
        Error bars for ``use_rep``, looked up the same way; same shape.
    key_added
        ``adata.uns`` key. The stored dict holds the tree (``edges``, ``lengths``,
        ``vert_names``, ``cell_vertices``, ``cell_ids``, ``newick``), the cell names
        (``obs_names``) and the run ``params``. Reload with :func:`get_bonsai_tree`.
    **run_kwargs
        Passed to :func:`run_bonsai` (``bonsai_dir``, ``n_procs``, ``zscore_cutoff``, ...).

    Returns
    -------
    BonsaiTree whose ``cell_ids`` are ``adata.obs_names``.
    """
    if "sanity_dir" in run_kwargs:
        raise ValueError(
            "With Sanity output call run_bonsai(sanity_dir=...) directly; tl_bonsai takes "
            "features from adata."
        )
    X = _get_matrix(adata, use_rep, "use_rep")
    S = None if stds is None else _get_matrix(adata, stds, "stds")
    feature_ids = None
    if isinstance(use_rep, str) and (use_rep == "X" or use_rep in adata.layers):
        feature_ids = list(map(str, adata.var_names))
    tree = run_bonsai(X, S, feature_ids=feature_ids, **run_kwargs)
    # Bonsai saw cell_<i>; re-label with the real cell names (row order is preserved)
    tree.cell_ids = np.asarray(adata.obs_names, dtype=str)

    params = {
        k: v for k, v in run_kwargs.items()
        if isinstance(v, (int, float, str, bool)) and k not in ("python", "mpiexec")
    }
    params["use_rep"] = use_rep if isinstance(use_rep, str) else "<array>"
    # h5ad cannot store None
    params["stds"] = "None" if stds is None else (stds if isinstance(stds, str) else "<array>")
    params["results_folder"] = str(tree.folder)
    adata.uns[key_added] = {**tree.to_dict(), "params": params}
    print(
        f"Bonsai tree with {tree.n_vertices} vertices stored in `adata.uns['{key_added}']`."
    )
    return tree


def get_bonsai_tree(adata, key: str = "bonsai") -> BonsaiTree:
    """Rebuild the :class:`BonsaiTree` stored by :func:`tl_bonsai`."""
    if key not in adata.uns:
        raise KeyError(f"No Bonsai tree in adata.uns['{key}']; run tl_bonsai first.")
    tree = BonsaiTree.from_dict(adata.uns[key])
    if tree.n_cells != adata.n_obs or not np.array_equal(
        tree.cell_ids, np.asarray(adata.obs_names, dtype=str)
    ):
        raise ValueError(
            f"adata.uns['{key}'] was computed for different cells (subset or reordered "
            "AnnData?); rerun tl_bonsai."
        )
    return tree
