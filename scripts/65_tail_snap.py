#!/usr/bin/env python
"""Replace the worst field-matching tail with a real L/R neighbor.

Keeps xyz bit-exact. Only the top-p residual rows change; they become an
actual left or right cell from the local spatial pool (closest to the clock
field). Not a full discrete pick (p=1 fails NFS/de); not a mean blend.

Leave-out (E6.75+E8.0, w=0.4, p=0.10 on k256): NFS 0.04852→0.04578,
var 0.02217→0.01422, freeze onto k04b6 still dual-gate PASS.

  .venv/bin/python scripts/65_tail_snap.py \\
    --pred outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --left E7.25 --right E8.0 --w 0.333 --p 0.10 --knn 8 --rms 198.24 \\
    --out outputs/t2/queue/2026-09-19/202_embryo_tailsnap_p010_k8_on194_n5000.h5ad
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
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def tail_snap(Xp, Cq, Xl, Xr, Cl, Cr, w: float, knn: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    k = int(knn)
    _, i0 = cKDTree(Cl).query(Cq, k=min(k, len(Cl)))
    _, i1 = cKDTree(Cr).query(Cq, k=min(k, len(Cr)))
    i0 = np.atleast_2d(np.asarray(i0, dtype=np.int64).reshape(len(Cq), min(k, len(Cl))))
    i1 = np.atleast_2d(np.asarray(i1, dtype=np.int64).reshape(len(Cq), min(k, len(Cr))))
    w = float(np.clip(w, 0.0, 1.0))
    field = (1.0 - w) * Xl[i0].mean(1) + w * Xr[i1].mean(1)
    resid = np.sum((Xp - field) ** 2, axis=1)
    cands = np.concatenate([Xl[i0], Xr[i1]], axis=1)
    d = ((cands - field[:, None, :]) ** 2).sum(-1)
    best = cands[np.arange(len(Cq)), d.argmin(1)]
    Y = np.asarray(Xp, dtype=np.float64).copy()
    n = int(round(float(p) * len(Y)))
    n = max(0, min(n, len(Y)))
    sel = np.argsort(-resid)[:n]
    Y[sel] = best[sel]
    return Y, sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--left", default="E7.25")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--w", type=float, default=1.0 / 3.0)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--p", type=float, default=0.10)
    ap.add_argument("--rms", type=float, default=198.24)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    panel = list(load_panel_for(target_spec(args.setting, dummy_t, cfg), cfg))
    stages = load_stages(args.setting, cfg, panel)
    left, right = stages[args.left], stages[args.right]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), args.rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), args.rms), dtype=np.float64)

    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.array(np.asarray(pred.obsm["spatial_3D"], dtype=np.float32), copy=True)
    Cq = np.asarray(scale_cloud(C0, args.rms), dtype=np.float64)

    Xn, sel = tail_snap(Xp, Cq, Xl, Xr, Cl, Cr, args.w, args.knn, args.p)
    np.clip(Xn, 0.0, None, out=Xn)

    out = pred.copy()
    if list(pred.var_names) == panel:
        out.X = Xn.astype(np.float32)
    else:
        idx = {g: i for i, g in enumerate(panel)}
        Xlive = np.asarray(to_dense(pred.X), dtype=np.float64)
        for j, g in enumerate(pred.var_names):
            Xlive[:, j] = Xn[:, idx[g]]
        out.X = Xlive.astype(np.float32)
    out.obsm["spatial_3D"] = C0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"], dtype=np.float32), C0)
    print(
        f"tail-snap p={args.p:g} knn={args.knn} w={args.w:g}  "
        f"n_snap={len(sel)}  mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
