"""Tests for eigenp_utils.bonsai_tree.

Bonsai itself is a script repository that is not installed in CI, so the subprocess
wiring is tested against a stand-in clone (``fake_bonsai``) that follows Bonsai's file
contract: it reads the config written by ``create_config_file.py`` and writes
``final_bonsai_zscore<c>/{edgeInfo,vertInfo}.txt`` and ``tree.nwk``. A real
end-to-end run is exercised when ``BONSAI_DIR`` points to a clone.
"""
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from eigenp_utils.bonsai_tree import (
    BonsaiTree,
    find_bonsai_dir,
    get_bonsai_tree,
    poisson_log_quotients,
    run_bonsai,
    tl_bonsai,
    write_bonsai_input,
)


# Pyodide (WASM CI job) cannot start subprocesses; the tree utilities still run there.
needs_subprocess = pytest.mark.skipif(
    sys.platform == "emscripten", reason="subprocesses are unavailable in WebAssembly"
)


# ------------------------- helpers -------------------------


def _write_tree_folder(folder: Path, edges, lengths, names, newick="(a:1,b:2)r;"):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "edgeInfo.txt", "w") as f:
        for (u, v), w in zip(edges, lengths):
            f.write("%d\t%d\t%.8e\n" % (u, v, w))
    with open(folder / "vertInfo.txt", "w") as f:
        f.write("vertInd\tnodeInd\tvertName\n")
        for i, n in enumerate(names):
            f.write("%d\t%d\t%s\n" % (i, i, n))
    (folder / "tree.nwk").write_text(newick)


def _small_tree():
    """Tree (vertex: name)

             0 internal_0
           /1.0        \\2.0
       1 internal_1     4 cell_2  ---0.5--- 5 cell_3
       /0.25  \\0.75
     2 cell_0  3 cell_1

    cell_2 is an internal vertex (it has the child cell_3).
    """
    edges = [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5)]
    lengths = [1.0, 0.25, 0.75, 2.0, 0.5]
    names = ["internal_0", "internal_1", "cell_0", "cell_1", "cell_2", "cell_3"]
    D = np.array(
        [
            [0.0, 1.0, 3.25, 3.75],
            [1.0, 0.0, 3.75, 4.25],
            [3.25, 3.75, 0.0, 0.5],
            [3.75, 4.25, 0.5, 0.0],
        ]
    )
    return edges, lengths, names, D


FAKE_CREATE_CONFIG = textwrap.dedent(
    """
    import json, sys
    args = sys.argv[1:]
    cfg = {args[i].lstrip('-'): args[i + 1] for i in range(0, len(args), 2)}
    with open(cfg['new_yaml_path'], 'w') as f:
        json.dump(cfg, f)
    """
)

# Writes a caterpillar tree over the cells found in features.txt; cell k hangs off
# internal vertex k with length 1 + k / 10, consecutive internals are 1 apart.
FAKE_MAIN = textwrap.dedent(
    """
    import json, os, sys
    cfg = json.load(open(sys.argv[sys.argv.index('--config_filepath') + 1]))
    if os.environ.get('FAKE_BONSAI_FAIL'):
        print('simulated Bonsai failure')
        sys.exit(3)
    data = cfg['data_folder']
    cells = [l.strip() for l in open(os.path.join(data, 'cellID.txt')) if l.strip()]
    n_feat = sum(1 for _ in open(os.path.join(data, cfg['filenames_data'].split(',')[0])))
    z = float(cfg['zscore_cutoff'])
    out = os.path.join(cfg['results_folder'], 'final_bonsai' + ('_zscore' + str(z) if z > 0 else ''))
    os.makedirs(out)
    C = len(cells)
    names = ['internal_%d' % k for k in range(C)] + cells
    edges = [(k, k + 1, 1.0) for k in range(C - 1)] + [(k, C + k, 1.0 + k / 10) for k in range(C)]
    with open(os.path.join(out, 'edgeInfo.txt'), 'w') as f:
        for u, v, w in edges:
            f.write('%d\\t%d\\t%.8e\\n' % (u, v, w))
    with open(os.path.join(out, 'vertInfo.txt'), 'w') as f:
        f.write('vertInd\\tnodeInd\\tvertName\\n')
        for i, n in enumerate(names):
            f.write('%d\\t%d\\t%s\\n' % (i, i, n))
    with open(os.path.join(out, 'tree.nwk'), 'w') as f:
        f.write('fake;')
    with open(os.path.join(cfg['results_folder'], 'seen.json'), 'w') as f:
        json.dump({'cfg': cfg, 'n_features': n_feat, 'cwd': os.getcwd()}, f)
    """
)


def _caterpillar_distance(i, j):
    if i == j:
        return 0.0
    return (1 + i / 10) + (1 + j / 10) + abs(i - j)


@pytest.fixture
def fake_bonsai(tmp_path):
    root = tmp_path / "Bonsai-data-representation"
    (root / "bonsai").mkdir(parents=True)
    (root / "bonsai" / "create_config_file.py").write_text(FAKE_CREATE_CONFIG)
    (root / "bonsai" / "bonsai_main.py").write_text(FAKE_MAIN)
    return root


# ------------------------- find_bonsai_dir -------------------------


def test_find_bonsai_dir_explicit_env_and_missing(fake_bonsai, tmp_path, monkeypatch):
    assert find_bonsai_dir(fake_bonsai) == fake_bonsai.resolve()

    monkeypatch.setenv("BONSAI_DIR", str(fake_bonsai))
    assert find_bonsai_dir() == fake_bonsai.resolve()

    monkeypatch.delenv("BONSAI_DIR")
    with pytest.raises(FileNotFoundError, match="git clone"):
        find_bonsai_dir()
    with pytest.raises(FileNotFoundError, match="does not look like a Bonsai clone"):
        find_bonsai_dir(tmp_path)


# ------------------------- poisson_log_quotients -------------------------


def test_poisson_log_quotients_matches_definition():
    counts = np.array([[0, 1, 9], [4, 4, 2]])
    x, s = poisson_log_quotients(counts, pseudocount=0.5)
    totals = counts.sum(1, keepdims=True)
    np.testing.assert_allclose(x, np.log((counts + 0.5) / totals))
    np.testing.assert_allclose(s, 1 / np.sqrt(counts + 0.5))

    xs, ss = poisson_log_quotients(sp.csr_matrix(counts), pseudocount=0.5)
    np.testing.assert_array_equal(xs, x)
    np.testing.assert_array_equal(ss, s)


def test_poisson_log_quotients_error_bar_is_calibrated():
    """For large Poisson means the SD of log(n / N) across replicates equals the
    reported error bar 1/sqrt(n) (delta method)."""
    rng = np.random.default_rng(0)
    lam = 400.0
    n = rng.poisson(lam, size=(20000, 1))
    counts = np.hstack([n, np.full_like(n, 10**6)])  # fixed huge 'other' to pin N
    x, s = poisson_log_quotients(counts)
    empirical = x[:, 0].std()
    assert abs(empirical / np.median(s[:, 0]) - 1) < 0.03


def test_poisson_log_quotients_rejects_bad_input():
    with pytest.raises(ValueError, match="zero total"):
        poisson_log_quotients(np.array([[0, 0], [1, 2]]))
    with pytest.raises(ValueError, match="non-negative"):
        poisson_log_quotients(np.array([[-1, 2], [1, 2]]))
    with pytest.raises(ValueError, match="pseudocount"):
        poisson_log_quotients(np.ones((2, 2)), pseudocount=0)


# ------------------------- write_bonsai_input -------------------------


def test_write_bonsai_input_layout(tmp_path):
    X = np.arange(12, dtype=float).reshape(4, 3)  # 4 cells x 3 features
    S = X + 1
    info = write_bonsai_input(tmp_path, X, S, feature_ids=["g0", "g1", "g2"])
    feats = np.loadtxt(tmp_path / "features.txt", delimiter="\t")
    stds = np.loadtxt(tmp_path / "standard_deviations.txt", delimiter="\t")
    # Bonsai expects features as rows, cells as columns
    np.testing.assert_array_equal(feats, X.T)
    np.testing.assert_array_equal(stds, S.T)
    assert (tmp_path / "cellID.txt").read_text().split() == [f"cell_{i}" for i in range(4)]
    assert (tmp_path / "geneID.txt").read_text().split() == ["g0", "g1", "g2"]
    assert info["filenames_data"] == "features.txt,standard_deviations.txt"

    info = write_bonsai_input(tmp_path / "nostd", sp.csr_matrix(X))
    assert info["filenames_data"] == "features.txt"
    assert not (tmp_path / "nostd" / "standard_deviations.txt").exists()


@pytest.mark.parametrize(
    "X, S, match",
    [
        (np.ones((4, 2)), np.zeros((4, 2)), "strictly positive"),
        (np.ones((4, 2)), np.ones((4, 3)), "does not match"),
        (np.array([[np.nan, 1]] * 4), None, "NaN"),
        (np.ones((2, 2)), None, "at least 3 cells"),
    ],
)
def test_write_bonsai_input_validation(tmp_path, X, S, match):
    with pytest.raises(ValueError, match=match):
        write_bonsai_input(tmp_path, X, S)


# ------------------------- BonsaiTree -------------------------


def test_tree_patristic_distances_and_internal_cell(tmp_path):
    edges, lengths, names, D_true = _small_tree()
    _write_tree_folder(tmp_path, edges, lengths, names)
    tree = BonsaiTree.from_folder(tmp_path)
    assert list(tree.cell_ids) == ["cell_0", "cell_1", "cell_2", "cell_3"]
    assert tree.n_vertices == 6 and tree.total_length == pytest.approx(4.5)
    np.testing.assert_allclose(tree.patristic_distances(), D_true)
    np.testing.assert_allclose(tree.patristic_distances(rows=[3], cols=[0, 2]), D_true[[3]][:, [0, 2]])

    # metric axioms on the full matrix
    D = tree.patristic_distances()
    np.testing.assert_allclose(D, D.T)
    assert np.all(D[:, :, None] <= D[:, None, :] + D.T[None, :, :] + 1e-12)  # triangle


def test_tree_cell_order_follows_cell_ids(tmp_path):
    edges, lengths, names, D_true = _small_tree()
    _write_tree_folder(tmp_path, edges, lengths, names)
    order = ["cell_3", "cell_0", "cell_2", "cell_1"]
    tree = BonsaiTree.from_folder(tmp_path, cell_ids=order)
    perm = [3, 0, 2, 1]
    np.testing.assert_allclose(tree.patristic_distances(), D_true[np.ix_(perm, perm)])
    with pytest.raises(ValueError, match="not vertices"):
        BonsaiTree.from_folder(tmp_path, cell_ids=["cell_0", "nope"])


def test_tree_zero_length_branch_is_kept(tmp_path):
    """A zero-length branch must still connect its endpoints (distance 0, not inf)."""
    edges = [(0, 1), (0, 2), (2, 3)]
    lengths = [1.0, 0.0, 2.0]
    names = ["internal_0", "a", "b", "c"]
    _write_tree_folder(tmp_path, edges, lengths, names)
    D = BonsaiTree.from_folder(tmp_path).patristic_distances()
    np.testing.assert_allclose(D, [[0, 1, 3], [1, 0, 2], [3, 2, 0]])


def test_tree_kneighbors_matches_bruteforce(tmp_path):
    edges, lengths, names, D_true = _small_tree()
    _write_tree_folder(tmp_path, edges, lengths, names)
    tree = BonsaiTree.from_folder(tmp_path)
    idx, dist = tree.kneighbors(n_neighbors=2, chunk_size=3)  # chunking crosses rows
    D = D_true.copy()
    np.fill_diagonal(D, np.inf)
    np.testing.assert_array_equal(idx, np.argsort(D, axis=1, kind="stable")[:, :2])
    np.testing.assert_allclose(dist, np.sort(D, axis=1)[:, :2])
    with pytest.raises(ValueError):
        tree.kneighbors(n_neighbors=4)


def test_tree_validation():
    with pytest.raises(ValueError, match="edges"):
        BonsaiTree(edges=[(0, 1)], lengths=[1.0], vert_names=["a", "b", "c"],
                   cell_vertices=[1, 2], cell_ids=["b", "c"])
    with pytest.raises(ValueError, match="non-negative"):
        BonsaiTree(edges=[(0, 1)], lengths=[-1.0], vert_names=["a", "b"],
                   cell_vertices=[0, 1], cell_ids=["a", "b"])


def test_tree_dict_roundtrip(tmp_path):
    edges, lengths, names, D_true = _small_tree()
    _write_tree_folder(tmp_path, edges, lengths, names)
    tree = BonsaiTree.from_folder(tmp_path)
    tree2 = BonsaiTree.from_dict(tree.to_dict())
    np.testing.assert_allclose(tree2.patristic_distances(), D_true)
    assert tree2.newick == tree.newick


# ------------------------- run_bonsai (stand-in clone) -------------------------


@needs_subprocess
def test_run_bonsai_wiring(fake_bonsai, tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 7))
    S = np.full_like(X, 0.3)
    out = tmp_path / "res"
    tree = run_bonsai(X, S, results_folder=out, bonsai_dir=fake_bonsai,
                      zscore_cutoff=1, nnn_n_randommoves=5, nnn_n_randomtrees=2)

    seen = json.loads((out / "seen.json").read_text())
    cfg = seen["cfg"]
    # Bonsai resolves relative paths against its own root, so everything must be absolute
    assert Path(cfg["data_folder"]).is_absolute() and Path(cfg["results_folder"]).is_absolute()
    assert Path(seen["cwd"]).resolve() == fake_bonsai.resolve()
    assert cfg["input_is_sanity_output"] == "False"
    assert cfg["filenames_data"] == "features.txt,standard_deviations.txt"
    assert cfg["zscore_cutoff"] == "1.0" and cfg["nnn_n_randommoves"] == "5"
    assert seen["n_features"] == 7  # features written as rows

    assert list(tree.cell_ids) == [f"cell_{i}" for i in range(5)]
    D = tree.patristic_distances()
    expected = np.array([[_caterpillar_distance(i, j) for j in range(5)] for i in range(5)])
    np.testing.assert_allclose(D, expected)
    assert Path(tree.folder).name == "final_bonsai_zscore1.0"
    assert (out / "bonsai_run.log").is_file()

    # a second run into the same folder would silently reuse the old tree
    with pytest.raises(FileExistsError):
        run_bonsai(X, S, results_folder=out, bonsai_dir=fake_bonsai)


@needs_subprocess
def test_run_bonsai_failure_reports_log(fake_bonsai, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BONSAI_FAIL", "1")
    with pytest.raises(RuntimeError, match="simulated Bonsai failure"):
        run_bonsai(np.ones((4, 2)), results_folder=tmp_path / "r", bonsai_dir=fake_bonsai)


@needs_subprocess
def test_run_bonsai_argument_checks(fake_bonsai, tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        run_bonsai(bonsai_dir=fake_bonsai)
    with pytest.raises(ValueError, match="exactly one"):
        run_bonsai(np.ones((4, 2)), sanity_dir=tmp_path, bonsai_dir=fake_bonsai)
    with pytest.raises(ValueError, match="sanity_dir"):
        run_bonsai(sanity_dir=tmp_path, stds=np.ones((4, 2)), bonsai_dir=fake_bonsai)


# ------------------------- AnnData round trip -------------------------


@needs_subprocess
def test_tl_bonsai_stores_and_reloads(fake_bonsai, tmp_path):
    anndata = pytest.importorskip("anndata")
    rng = np.random.default_rng(2)
    counts = rng.poisson(3, size=(6, 8)) + 1
    adata = anndata.AnnData(X=counts.astype(np.float32))
    # ':' is illegal in Bonsai IDs; object dtype because anndata 0.10 cannot write
    # pandas 3's default string arrays to h5ad
    adata.obs_names = pd.Index([f"bc:{i}" for i in range(6)], dtype=object)
    adata.var_names = pd.Index([f"g{i}" for i in range(8)], dtype=object)
    x, s = poisson_log_quotients(adata.X)
    adata.layers["ltq"], adata.layers["ltq_sd"] = x, s

    tree = tl_bonsai(adata, use_rep="ltq", stds="ltq_sd", bonsai_dir=fake_bonsai,
                     results_folder=tmp_path / "res", zscore_cutoff=-1)
    assert list(tree.cell_ids) == list(adata.obs_names)
    assert Path(tree.folder).name == "final_bonsai"  # no z-score suffix for cutoff <= 0
    seen = json.loads((tmp_path / "res" / "seen.json").read_text())
    assert (Path(seen["cfg"]["data_folder"]) / "geneID.txt").read_text().split() == list(adata.var_names)

    path = tmp_path / "a.h5ad"
    adata.write_h5ad(path)
    back = anndata.read_h5ad(path)
    tree2 = get_bonsai_tree(back)
    np.testing.assert_allclose(tree2.patristic_distances(), tree.patristic_distances())
    assert back.uns["bonsai"]["params"]["stds"] == "ltq_sd"

    with pytest.raises(ValueError, match="different cells"):
        get_bonsai_tree(back[::-1].copy())
    with pytest.raises(KeyError):
        get_bonsai_tree(back, key="nope")


# ------------------------- real Bonsai (optional) -------------------------


@pytest.mark.skipif(not os.environ.get("BONSAI_DIR"), reason="BONSAI_DIR not set")
def test_real_bonsai_recovers_two_clades(tmp_path):
    """Two well-separated groups of cells: every cell's nearest tree neighbour must be in
    its own group (positive control); this fails if cell order is scrambled anywhere."""
    rng = np.random.default_rng(0)
    n, g = 16, 40
    centers = rng.normal(scale=3.0, size=(2, g))
    labels = np.repeat([0, 1], n // 2)
    X = centers[labels] + rng.normal(scale=0.3, size=(n, g))
    S = np.full_like(X, 0.1)
    tree = run_bonsai(X, S, results_folder=tmp_path / "res", zscore_cutoff=-1,
                      nnn_n_randommoves=20, nnn_n_randomtrees=1, timeout=600)
    idx, _ = tree.kneighbors(n_neighbors=1)
    assert np.all(labels[idx[:, 0]] == labels)
