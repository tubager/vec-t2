#!/usr/bin/env python
"""Per-cluster quadratic-minus-linear curvature. Keeps 194/196 cells.

201 added a *global* μ_quad−μ_lin on 29 genes — cluster-relative DE is invariant,
board de locked, var smashed. This adds the same curvature *inside each working
cluster* and restores original zeros (208).

Two-point leave-out: quadratic ≡ linear ⇒ identity. Dual-gate is tautological.
Board uses E6.75 as the third knot (embryo) or E9.5 (heart interp).

  .venv/bin/python scripts/75_cluster_quad_curv.py --setting embryo --t 7.5 \\
    --pred outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --out outputs/t2/queue/2026-09-20/210_embryo_clust_quadcurv_keepz_on194_n5000.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def _mu_at(ts: np.ndarray, mus: np.ndarray, t: float) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.float64)
    mus = np.asarray(mus, dtype=np.float64)
    order = np.argsort(ts)
    ts, mus = ts[order], mus[order]
    if len(ts) == 1 or t <= ts[0]:
        return mus[0]
    if t >= ts[-1]:
        return mus[-1]
    if len(ts) == 2:
        w = (t - ts[0]) / max(ts[1] - ts[0], 1e-6)
        return (1.0 - w) * mus[0] + w * mus[1]
    T = np.stack([np.ones(len(ts)), ts, ts * ts], axis=1)
    coef, *_ = np.linalg.lstsq(T, mus, rcond=None)
    return coef[0] + coef[1] * t + coef[2] * t * t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")
    args = ap.parse_args()

    cfg = load_config()
    spec = target_spec(args.setting, float(args.t), cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)
    if args.setting == "embryo":
        three = [("E6.75", 6.75), ("E7.25", 7.25), ("E8.0", 8.0)]
        pair = [("E7.25", 7.25), ("E8.0", 8.0)]
        usable = set(
            shared_flow_clusters(
                stages["E7.25"].obs["celltype"], stages["E8.0"].obs["celltype"], "embryo"
            )
        )
    else:
        three = [("E8.25", 8.25), ("E8.75", 8.75), ("E9.5", 9.5)]
        pair = [("E8.25", 8.25), ("E8.75", 8.75)]
        usable = set(
            shared_flow_clusters(
                stages["E8.25"].obs["celltype"], stages["E8.75"].obs["celltype"], "heart"
            )
        )

    packed = {}
    for name, t_s in three:
        a = stages[name]
        X = _align(to_dense(a.X), a.var_names, panel)
        cl = np.asarray(labels_to_clusters(a.obs["celltype"], args.setting))
        packed[name] = (t_s, X, cl)

    Xref = np.concatenate([packed[pair[0][0]][1], packed[pair[1][0]][1]], axis=0)
    cref = np.concatenate([packed[pair[0][0]][2], packed[pair[1][0]][2]])
    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.array(np.asarray(pred.obsm["spatial_3D"], dtype=np.float32), copy=True)
    nn = NearestNeighbors(n_neighbors=min(int(args.knn), len(Xref))).fit(Xref)
    _, idx = nn.kneighbors(Xp)
    names = []
    for row in idx:
        labs = [str(cref[j]) for j in row]
        vals, cnt = np.unique(labs, return_counts=True)
        names.append(str(vals[int(np.argmax(cnt))]))
    names = np.asarray(names)

    Xn = np.array(Xp, copy=True, dtype=np.float64)
    n_used = 0
    a = float(max(0.0, args.alpha))
    for key in sorted(set(names) & usable):
        m = names == key
        if not m.any():
            continue
        ts_q, mus_q = [], []
        ts_l, mus_l = [], []
        ok = True
        for name, t_s in three:
            _, X, cl = packed[name]
            sel = cl == key
            if sel.sum() < 16:
                ok = False
                break
            mu = np.median(X[sel], 0) if args.agg == "median" else X[sel].mean(0)
            ts_q.append(t_s)
            mus_q.append(mu)
            if name in {pair[0][0], pair[1][0]}:
                ts_l.append(t_s)
                mus_l.append(mu)
        if not ok or len(mus_q) < 3 or len(mus_l) < 2:
            continue
        dmu = _mu_at(np.array(ts_q), np.stack(mus_q), float(args.t)) - _mu_at(
            np.array(ts_l), np.stack(mus_l), float(args.t)
        )
        Xn[m] = Xn[m] + a * dmu
        n_used += int(m.sum())

    np.clip(Xn, 0.0, None, out=Xn)
    Xn = Xn.astype(np.float32)
    Xn[Xp <= 0] = 0.0

    out = pred.copy()
    if list(pred.var_names) == panel:
        out.X = Xn
    else:
        live = np.asarray(to_dense(pred.X), dtype=np.float32)
        idxg = {g: i for i, g in enumerate(panel)}
        for j, g in enumerate(pred.var_names):
            live[:, j] = Xn[:, idxg[g]]
        out.X = live
    out.obsm["spatial_3D"] = C0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"], dtype=np.float32), C0)
    print(
        f"quad-curv {args.setting} t={args.t:g} a={a:.2f} agg={args.agg} n={n_used} "
        f"mae={float(np.abs(Xn - Xp).mean()):.4f} frac0={float((Xn <= 0).mean()):.3f}  {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
