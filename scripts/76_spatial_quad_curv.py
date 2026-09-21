#!/usr/bin/env python
"""Spatial-neighborhood quadratic-minus-linear curvature on a locked cloud.

210 pools a working cluster's global mean. This fits μ(t) from spatial kNN at
each knot around the cell's own xyz, then adds (μ_quad − μ_lin) and restores
zeros. Cells are not replaced. xyz is bit-exact.

  .venv/bin/python scripts/76_spatial_quad_curv.py --setting embryo --t 7.5 \\
    --pred outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --out outputs/t2/embryo/gate75_curv/spatial_a1.h5ad
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

from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
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
    ap.add_argument("--knn", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.05, help="|dμ| gene floor")
    args = ap.parse_args()

    cfg = load_config()
    spec = target_spec(args.setting, float(args.t), cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)
    if args.setting == "embryo":
        three = [("E6.75", 6.75), ("E7.25", 7.25), ("E8.0", 8.0)]
        pair = {"E7.25", "E8.0"}
        rms_t = 198.24
    else:
        three = [("E8.25", 8.25), ("E8.75", 8.75), ("E9.5", 9.5)]
        pair = {"E8.25", "E8.75"}
        rms_t = 255.0

    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.array(np.asarray(pred.obsm["spatial_3D"], dtype=np.float32), copy=True)
    q = np.asarray(scale_cloud(C0, rms_t), dtype=np.float64)

    local_q, local_l = [], []
    ts_q, ts_l = [], []
    for name, t_s in three:
        a = stages[name]
        X = _align(to_dense(a.X), a.var_names, panel)
        C = np.asarray(scale_cloud(spatial_xyz(a), rms_t), dtype=np.float64)
        k = min(int(args.knn), len(C))
        _, ii = cKDTree(C).query(q, k=k)
        ii = np.asarray(ii, dtype=np.int64).reshape(len(q), k)
        mu = X[ii].mean(axis=1)
        local_q.append(mu)
        ts_q.append(t_s)
        if name in pair:
            local_l.append(mu)
            ts_l.append(t_s)

    mus_q = np.stack(local_q, axis=1)
    mus_l = np.stack(local_l, axis=1)
    ts_qa = np.asarray(ts_q, dtype=np.float64)
    ts_la = np.asarray(ts_l, dtype=np.float64)
    t = float(args.t)
    a = float(args.alpha)
    tau = float(args.tau)

    dmu = np.empty_like(Xp)
    for i in range(len(q)):
        dmu[i] = _mu_at(ts_qa, mus_q[i], t) - _mu_at(ts_la, mus_l[i], t)
    mask = np.abs(dmu) >= tau
    gene_ok = np.median(np.abs(dmu), axis=0) >= tau
    mask &= gene_ok[None, :]
    Xn = Xp + a * dmu * mask
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
    n_g = int(mask.any(axis=0).sum())
    print(
        f"spatial-curv {args.setting} t={t:g} a={a:.2f} knn={args.knn} tau={tau:g} "
        f"genes={n_g} mae={float(np.abs(Xn - Xp).mean()):.4f} "
        f"frac0={float((Xn <= 0).mean()):.3f}  {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
