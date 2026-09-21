#!/usr/bin/env python
"""Scaffold-local X: pick mid-stage expression onto a board-locked xyz SEQUENCE.

Unlike ``local_pick`` inside OT-place (query = left OT cloud → board ODS collapse) and
unlike post-hoc Hungarian X transplant (anti-transfers), this keeps ``--live`` xyz
bit-exact and queries left/right neighborhoods in a **shared Procrustes frame**.

  1. Kabsch-align left & right clouds onto live (center → RMS → SVD rotation, det=+1)
  2. For each live point, pick X from spatial kNN with progress closest to clock ``w``
     (or pool / neighborhood mean blend)
  3. Write X onto live; assert xyz sequence unchanged

Usage (leave-out):
  .venv/bin/python scripts/31_scaffold_local_x.py \\
    --live outputs/t2/embryo/gate725/sweep_widestrat/bandX_on_spatial_k0.40_b6.h5ad \\
    --setting embryo --left E6.75 --right E8.0 \\
    --mode local_pick --knn 8 --w 0.15 \\
    --out outputs/t2/embryo/gate725/sweep_scaffold/pick_k8_w0.15.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.geometry import rms_radius  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.ot import _progress_axis  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def _align_genes(X: np.ndarray, genes, panel: list[str]) -> np.ndarray:
    genes = list(genes)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def kabsch_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid+scale map of ``src`` onto ``dst``'s frame (no correspondence; cloud moments).

    Centers both, scales src RMS → dst RMS, rotates via Kabsch on cross-covariance of
    *sorted* radial shells is unstable; instead PCA-align both then pick proper flip
    minimising mean NN distance src→dst.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Xs, Xd = src - mu_s, dst - mu_d
    rs, rd = rms_radius(src), rms_radius(dst)
    Xs = Xs * (rd / max(rs, 1e-8))
    # PCA bases
    _, _, Vt_s = np.linalg.svd(Xs, full_matrices=False)
    _, _, Vt_d = np.linalg.svd(Xd, full_matrices=False)
    Zs = Xs @ Vt_s[:3].T
    Zd = Xd @ Vt_d[:3].T
    tree = cKDTree(Zd)
    flips = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    best_f, best_cost = flips[0], np.inf
    # subsample for speed
    rng = np.random.default_rng(0)
    take = rng.choice(len(Zs), size=min(2000, len(Zs)), replace=False)
    for f in flips:
        Z = Zs[take] * np.asarray(f, dtype=np.float64)
        d, _ = tree.query(Z, k=1)
        c = float(np.mean(d))
        if c < best_cost:
            best_cost, best_f = c, f
    Zs = Zs * np.asarray(best_f, dtype=np.float64)
    # map canonical → dst lab: Zd ≈ Xd @ Vt_d[:3].T  ⇒  Xd ≈ Zd @ Vt_d[:3]
    aligned = Zs @ Vt_d[:3] + mu_d
    return aligned.astype(np.float64)


def local_pick_on(
    query: np.ndarray,
    xyz0: np.ndarray,
    X0: np.ndarray,
    s0: np.ndarray,
    xyz1: np.ndarray,
    X1: np.ndarray,
    s1: np.ndarray,
    w: float,
    knn: int,
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    n = len(query)
    k0 = max(1, min(int(knn), len(xyz0)))
    k1 = max(1, min(int(knn), len(xyz1)))
    _, i0 = cKDTree(xyz0).query(query, k=k0)
    _, i1 = cKDTree(xyz1).query(query, k=k1)
    i0 = np.asarray(i0, dtype=np.int64).reshape(n, k0)
    i1 = np.asarray(i1, dtype=np.int64).reshape(n, k1)
    out = np.empty((n, X0.shape[1]), dtype=np.float32)
    w = float(np.clip(w, 0.0, 1.0))
    if mode == "local_mean":
        for j in range(n):
            m0 = X0[i0[j]].mean(0)
            m1 = X1[i1[j]].mean(0)
            out[j] = ((1.0 - w) * m0 + w * m1).astype(np.float32)
        return out
    if mode == "local_pool":
        for j in range(n):
            loc0, loc1 = i0[j], i1[j]
            d0 = np.abs(s0[loc0] - w)
            d1 = np.abs(s1[loc1] - w)
            if float(d0.min()) <= float(d1.min()):
                out[j] = X0[int(loc0[int(np.argmin(d0))])]
            else:
                out[j] = X1[int(loc1[int(np.argmin(d1))])]
        return out
    # local_pick
    take_right = rng.random(n) < w
    for j in range(n):
        if take_right[j]:
            loc = i1[j]
            out[j] = X1[int(loc[int(np.argmin(np.abs(s1[loc] - w)))])]
        else:
            loc = i0[j]
            out[j] = X0[int(loc[int(np.argmin(np.abs(s0[loc] - w)))])]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=Path, required=True)
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--left", type=str, default=None)
    ap.add_argument("--right", type=str, default=None)
    ap.add_argument("--mode", choices=["local_pick", "local_pool", "local_mean"], default="local_pick")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--frac", type=float, default=1.0, help="only replace this fraction of rows (farthest |dX|)")
    ap.add_argument(
        "--partial-genes",
        type=int,
        default=None,
        help="only replace this many genes with largest |μR−μL|; keep live X elsewhere (variogram-safe)",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="blend on replaced genes: X = (1-a)*live + a*scaffold (default 1 = hard swap)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-cluster", action="store_true", help="restrict kNN to same working cluster")
    args = ap.parse_args()
    if not (0.0 < float(args.frac) <= 1.0):
        raise SystemExit("--frac must be in (0,1]")
    if int(args.knn) < 2:
        raise SystemExit("--knn must be >= 2")
    if not (0.0 < float(args.alpha) <= 1.0):
        raise SystemExit("--alpha must be in (0,1]")
    if args.partial_genes is not None and int(args.partial_genes) < 1:
        raise SystemExit("--partial-genes must be >= 1")

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    spec = target_spec(args.setting, dummy_t, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)
    left_name = args.left or ("E6.75" if args.setting == "embryo" else "E8.25_late")
    right_name = args.right or ("E8.0" if args.setting == "embryo" else "E9.5")
    # board embryo often uses E7.25 as left; allow override
    left, right = stages[left_name], stages[right_name]

    live = ad.read_h5ad(args.live)
    Cl = np.asarray(live.obsm["spatial_3D"], dtype=np.float64)
    Xl = _align_genes(to_dense(live.X), live.var_names, panel)
    X0 = _align_genes(to_dense(left.X), left.var_names, panel)
    X1 = _align_genes(to_dense(right.X), right.var_names, panel)
    C0 = kabsch_align(np.asarray(left.obsm["spatial_3D"], dtype=np.float64), Cl)
    C1 = kabsch_align(np.asarray(right.obsm["spatial_3D"], dtype=np.float64), Cl)

    # progress scores in PCA of left/right panel
    pca = ExpressionPCA(n_components=min(32, X0.shape[1]), batch_size=256)
    # fit lightly on subsample
    rng = np.random.default_rng(args.seed)
    fit_idx0 = rng.choice(len(X0), size=min(4000, len(X0)), replace=False)
    fit_idx1 = rng.choice(len(X1), size=min(4000, len(X1)), replace=False)
    from sklearn.decomposition import IncrementalPCA
    ip = IncrementalPCA(n_components=min(32, X0.shape[1]), batch_size=256)
    ip.partial_fit(X0[fit_idx0])
    ip.partial_fit(X1[fit_idx1])
    z0 = ip.transform(X0).astype(np.float64)
    z1 = ip.transform(X1).astype(np.float64)
    prog = _progress_axis(z0, z1)
    if prog is None:
        s0 = np.zeros(len(X0))
        s1 = np.ones(len(X1))
    else:
        s0, s1, _ = prog

    if args.per_cluster:
        # assign live rows to clusters via expr kNN against left+right
        cl0 = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))
        cl1 = np.asarray(labels_to_clusters(right.obs["celltype"], args.setting))
        usable = set(shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting))
        ref_X = np.concatenate([X0[fit_idx0], X1[fit_idx1]], 0)
        ref_c = np.concatenate([cl0[fit_idx0], cl1[fit_idx1]], 0)
        _, nn = cKDTree(ref_X).query(Xl, k=15)
        live_cl = np.empty(len(Xl), dtype=object)
        for i, neigh in enumerate(nn):
            vals, cnt = np.unique(ref_c[neigh], return_counts=True)
            live_cl[i] = vals[int(np.argmax(cnt))]
        Xnew = Xl.copy()
        for name in np.unique(live_cl):
            if str(name) not in usable:
                continue
            m_q = live_cl == name
            m0 = cl0 == name
            m1 = cl1 == name
            if m0.sum() < 4 or m1.sum() < 4 or m_q.sum() == 0:
                continue
            Xnew[m_q] = local_pick_on(
                Cl[m_q], C0[m0], X0[m0], s0[m0], C1[m1], X1[m1], s1[m1],
                args.w, args.knn, rng, args.mode,
            )
    else:
        Xnew = local_pick_on(Cl, C0, X0, s0, C1, X1, s1, args.w, args.knn, rng, args.mode)

    if float(args.frac) < 1.0 - 1e-12:
        dist = np.linalg.norm(Xnew.astype(np.float64) - Xl.astype(np.float64), axis=1)
        k = max(1, int(round(float(args.frac) * len(Xl))))
        take = np.argsort(-dist)[:k]
        blended = Xl.copy()
        blended[take] = Xnew[take]
        Xnew = blended

    gene_mask = None
    if args.partial_genes is not None:
        # global |μR−μL| ranking across shared working clusters
        cl0 = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))
        cl1 = np.asarray(labels_to_clusters(right.obs["celltype"], args.setting))
        usable = shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting)
        dmu = np.zeros(len(panel), dtype=np.float64)
        n = 0
        for name in usable:
            a, b = X0[cl0 == name], X1[cl1 == name]
            if len(a) < 8 or len(b) < 8:
                continue
            dmu += np.abs(a.mean(0) - b.mean(0))
            n += 1
        if n == 0:
            raise SystemExit("no usable clusters for --partial-genes ranking")
        dmu /= float(n)
        top = np.argsort(-dmu)[: int(args.partial_genes)]
        gene_mask = np.zeros(len(panel), dtype=bool)
        gene_mask[top] = True
        blended = Xl.copy()
        a = float(args.alpha)
        blended[:, gene_mask] = (1.0 - a) * Xl[:, gene_mask] + a * Xnew[:, gene_mask]
        Xnew = blended
    elif float(args.alpha) < 1.0 - 1e-12:
        a = float(args.alpha)
        Xnew = ((1.0 - a) * Xl + a * Xnew).astype(np.float32)

    np.clip(Xnew, 0.0, None, out=Xnew)
    out = live.copy()
    out.X = Xnew.astype(np.float32)
    assert np.array_equal(
        np.asarray(out.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(live.obsm["spatial_3D"], dtype=np.float32),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    pg = "all" if gene_mask is None else str(int(gene_mask.sum()))
    print(
        f"scaffold {args.mode} k={args.knn} w={args.w:.3f} frac={args.frac:.2f} "
        f"partial_genes={pg} alpha={float(args.alpha):.2f} "
        f"per_cluster={args.per_cluster}  mean|dX|={float(np.mean(np.abs(Xnew - Xl))):.4f}  "
        f"rms={rms_radius(Cl):.2f}  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
