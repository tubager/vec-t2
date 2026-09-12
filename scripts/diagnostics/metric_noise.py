#!/usr/bin/env python
"""How much of the board's X-side score is row-order luck?

The v2 panel subsamples OUR rows with `np.random.default_rng(0)` before scoring:
  mmd_unbiased  -> 2000 rows (also used by neighborhood_mmd on the nb-pb rows)
  variogram     -> 1500 rows
  d2_distance   -> 200k index pairs
  occupancy_dice-> _match_n, a permutation when the truth has more cells than we do
So the submitted ROW ORDER decides which cells are scored. This script measures, for one file:
  (a) the board-setting value at the as-submitted order and at K random orders,
  (b) an "our-side-noise-free" reference that uses ALL of our rows while keeping the truth's
      subsample, the PCA basis, the bandwidth and the gene pairs bit-identical to (a).
(b) - mean(a) is the systematic penalty our own subsampling noise adds; the spread of (a) is the
luck band. Both are converted to board skill points with the reverse-engineered floor/ceiling.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import anndata as ad

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, "/tmp/t2shim")

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics.pairwise import rbf_kernel  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402
from common.core_metrics import mmd_unbiased, variogram_score  # noqa: E402
from common.shape_metrics import d2_distance, occupancy_dice  # noqa: E402

SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
# heart-interp board floor/ceiling (reverse engineered, see doc sec.10)
FC = {"mmd": (0.05933, 0.00026), "var": (0.059281, 0.000702),
      "nfs": (0.10737, 0.00047), "d2": (0.0461, 0.00213), "ods": (0.81, 0.9528)}


def dense(a):
    X = a.X
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def skill(raw, key):
    f, c = FC[key]
    return 100.0 * (f - raw) / (f - c)


def mmd_parts(pred_X, true_X, n=2000, n_pc=30, seed=0):
    """Return (index set used on pred, callable that scores an arbitrary pred row subset)."""
    rng = np.random.default_rng(seed)
    pca = PCA(n_components=min(n_pc, true_X.shape[1]), random_state=0).fit(true_X)
    I = rng.choice(pred_X.shape[0], min(n, pred_X.shape[0]), replace=False)
    J = rng.choice(true_X.shape[0], min(n, true_X.shape[0]), replace=False)
    B = pca.transform(true_X[J])
    d2 = np.sum((B[:, None] - B[None, :]) ** 2, -1)
    gamma0 = 1.0 / (np.median(d2[d2 > 0]) + 1e-9)
    nb = B.shape[0]
    Kbb_tot = sum(float(np.fill_diagonal(K, 0.0) or K.sum()) for K in (rbf_kernel(B, B, gamma0 * s) for s in SCALES))

    def score(A_rows):
        A = pca.transform(A_rows)
        na = A.shape[0]
        tot = 0.0
        for s in SCALES:
            g = gamma0 * s
            Kaa = rbf_kernel(A, A, g); np.fill_diagonal(Kaa, 0.0)
            Kab = rbf_kernel(A, B, g)
            kbb = rbf_kernel(B, B, g); np.fill_diagonal(kbb, 0.0)
            tot += Kaa.sum() / (na * (na - 1)) + kbb.sum() / (nb * (nb - 1)) - 2 * Kab.mean()
        return float(tot / len(SCALES))
    return I, score


def var_parts(pred_X, true_X, n_cells=1500, n_pairs=20000, p=0.5, seed=0):
    rng = np.random.default_rng(seed)
    I = rng.choice(pred_X.shape[0], min(n_cells, pred_X.shape[0]), replace=False)
    J = rng.choice(true_X.shape[0], min(n_cells, true_X.shape[0]), replace=False)
    G = pred_X.shape[1]
    i = rng.integers(0, G, n_pairs); j = rng.integers(0, G, n_pairs)
    keep = i != j; i, j = i[keep], j[keep]
    vb = (np.abs(true_X[J][:, i] - true_X[J][:, j]) ** p).mean(0)

    def score(A_rows):
        va = (np.abs(A_rows[:, i] - A_rows[:, j]) ** p).mean(0)
        return float(((va - vb) ** 2).mean())
    return I, score


def nbpb(X, C, k=15):
    _, idx = NearestNeighbors(n_neighbors=k).fit(C).kneighbors(C)
    return X[idx].mean(1)


def main(pred_path, truth_path, out_path=None, K=8):
    P = ad.read_h5ad(pred_path); T = ad.read_h5ad(truth_path)
    X = dense(P); C = np.asarray(P.obsm["spatial_3D"], float)
    tX = dense(T); tC = np.asarray(T.obsm["spatial_3D"], float)
    common = [g for g in P.var_names if g in set(T.var_names)]
    if len(common) != X.shape[1] or len(common) != tX.shape[1]:
        print(f"# gene panels differ: aligning on {len(common)} shared genes")
        X = X[:, [list(P.var_names).index(g) for g in common]]
        tX = tX[:, [list(T.var_names).index(g) for g in common]]
    n = len(X)
    print(f"# {Path(pred_path).name}  n={n}   truth={Path(truth_path).name} n={len(tX)}")

    Im, mmd_ref = mmd_parts(X, tX)
    Iv, var_ref = var_parts(X, tX)
    nb_p, nb_t = nbpb(X, C), nbpb(tX, tC)
    In, nfs_ref = mmd_parts(nb_p, nb_t)
    print(f"# I_mmd={len(Im)} I_var={len(Iv)} I_nfs={len(In)}  (I_mmd == I_nfs: {np.array_equal(np.sort(Im), np.sort(In))})")

    pb = nbpb(X, C)
    rows = []
    rng = np.random.default_rng(11)
    orders = [np.arange(n)] + [rng.permutation(n) for _ in range(K)]
    for p in orders:
        rows.append((mmd_ref(X[p][Im]), var_ref(X[p][Iv]), nfs_ref(pb[p][In]),
                     d2_distance(C[p], tC, seed=0), occupancy_dice(C[p], tC, seed=0)[0]))
    a = np.array(rows)
    ref = np.array([mmd_ref(X), var_ref(X), nfs_ref(pb),
                    d2_distance(C, tC, seed=0), occupancy_dice(C, tC, seed=0)[0]])
    names = ["mmd_u", "variogram", "NFS", "d2", "occ_dice"]
    keys = ["mmd", "var", "nfs", "d2", "ods"]
    print(f"{'metric':11s} {'as-subm':>9s} {'rand mean':>10s} {'rand sd':>9s} {'rand min':>9s} {'rand max':>9s} {'full-rows':>10s}")
    for r, nm in enumerate(a.T):
        print(f"{names[r]:11s} {nm[0]:9.5f} {nm[1:].mean():10.5f} {nm[1:].std():9.5f} {nm[1:].min():9.5f} {nm[1:].max():9.5f} {ref[r]:10.5f}")
    print()
    print(f"{'metric':11s} {'skill as-subm':>14s} {'skill rand mean':>16s} {'skill sd':>9s} {'skill full-rows':>16s} {'gain(full-rand)':>16s}")
    tot_gain = 0.0
    weights = {"mmd_u": 0.25 * 0.6, "variogram": 0.25 * 0.4, "NFS": 0.25, "d2": 0.25 / 3, "occ_dice": 0.25 / 3}
    for r, nm in enumerate(names):
        s_sub = skill(a[r][0], keys[r]); s_rnd = skill(a[r][1:].mean(), keys[r])
        s_sd = np.std([skill(v, keys[r]) for v in a[r][1:]]); s_full = skill(ref[r], keys[r])
        g = s_full - s_rnd
        tot_gain += weights[nm] * g
        print(f"{nm:11s} {s_sub:14.1f} {s_rnd:16.1f} {s_sd:9.1f} {s_full:16.1f} {g:16.1f}")
    print(f"\n# total-score effect of removing OUR-side subsample noise: {tot_gain:+.2f}")
    print(f"# total-score luck band (1 sd, sum of order-dependent items): "
          f"{sum(weights[n]*np.std([skill(v,k) for v in a[r][1:]]) for r,(n,k) in enumerate(zip(names,keys))):+.2f}")
    if out_path:
        Path(out_path).write_text("")


if __name__ == "__main__":
    main(*sys.argv[1:])
