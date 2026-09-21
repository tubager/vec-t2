#!/usr/bin/env python
"""Per-cluster PCA residual: keep each cell's residual, retarget the cluster mean.

Three-stage (board): quadratic μ(t) through E6.75/E7.25/E8.0, query t=7.5.
Leave-out: linear μ(t) on E6.75+E8.0, query t=7.25. Not OT-pair interpolants (206).
Keep original zeros (208: identity PCA keepz preserved variogram).

  .venv/bin/python scripts/72_pca_resid.py --t 7.25 --k 64 \\
    --pred outputs/t2/embryo/gate725_repair/k256.h5ad \\
    --out outputs/t2/embryo/gate725_pcaresid/k256.h5ad
  .venv/bin/python scripts/72_pca_resid.py --t 7.5 --k 64 --three-stage \\
    --pred outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --out outputs/t2/queue/2026-09-20/209_embryo_pca64_quadresid_on194_n5000.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA
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
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


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
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--three-stage", action="store_true")
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    spec = target_spec("embryo", float(args.t), cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("embryo", cfg, panel)
    if args.three_stage:
        tags = [("E6.75", 6.75), ("E7.25", 7.25), ("E8.0", 8.0)]
    else:
        tags = [(spec.left, float(spec.t_left)), (spec.right, float(spec.t_right))]
    usable = set(shared_flow_clusters(stages["E7.25"].obs["celltype"], stages["E8.0"].obs["celltype"], "embryo"))

    packed = []
    for name, t_s in tags:
        a = stages[name]
        X = _align(to_dense(a.X), a.var_names, panel)
        cl = np.asarray(labels_to_clusters(a.obs["celltype"], "embryo"))
        keep = np.array([str(c) in usable for c in cl])
        packed.append((t_s, X[keep], cl[keep]))

    Xcat = np.concatenate([p[1] for p in packed], axis=0)
    cl_cat = np.concatenate([p[2] for p in packed])
    Ycat = np.log1p(np.maximum(Xcat, 0.0))

    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.array(np.asarray(pred.obsm["spatial_3D"], dtype=np.float32), copy=True)
    nn = NearestNeighbors(n_neighbors=min(int(args.knn), len(Xcat))).fit(Xcat)
    _, idx = nn.kneighbors(Xp)
    names = []
    for row in idx:
        labs = [str(cl_cat[j]) for j in row]
        vals, cnt = np.unique(labs, return_counts=True)
        names.append(str(vals[int(np.argmax(cnt))]))
    names = np.asarray(names)

    Xn = np.array(Xp, copy=True, dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    n_used = 0
    for key in sorted(set(names) & usable):
        m = names == key
        if not m.any():
            continue
        parts = []
        for t_s, X, cl in packed:
            sel = cl == key
            if sel.sum() < 16:
                continue
            parts.append((t_s, np.log1p(np.maximum(X[sel], 0.0))))
        if len(parts) < 2:
            continue
        Ytr = np.concatenate([p[1] for p in parts], axis=0)
        k = min(int(args.k), Ytr.shape[1], len(Ytr) - 1)
        pca = PCA(n_components=k, random_state=0).fit(Ytr)
        ts = np.array([p[0] for p in parts], dtype=np.float64)
        mus = np.stack([pca.transform(p[1]).mean(0) for p in parts], axis=0)
        mu_star = _mu_at(ts, mus, float(args.t))
        Yp = np.log1p(np.maximum(Xp[m], 0.0))
        zp = pca.transform(Yp)
        # native-t mean: nearest packed stage by expression knn already; use pred cluster mean in this PCA
        mu_src = zp.mean(0)
        z_new = zp - mu_src + mu_star
        Yh = pca.inverse_transform(z_new)
        Xh = np.clip(np.expm1(Yh), 0.0, None).astype(np.float32)
        Xh[Xp[m] <= 0] = 0.0
        Xn[m] = Xh
        n_used += int(m.sum())

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
        f"pca-resid t={args.t:g} k={args.k} three={int(args.three_stage)} n={n_used} "
        f"mae_vs_pred={float(np.abs(Xn - Xp).mean()):.4f} frac0={float((Xn <= 0).mean()):.3f}  {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
