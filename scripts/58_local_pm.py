#!/usr/bin/env python
"""Spatially adaptive partial-mean. xyz sequence bit-exact.

09a uses one global 16-gene mask. Here each cell picks its own k genes from
|μ_right_knn − μ_left_knn|, then takes a small step toward the local mean.
Not Gaussian copula (190 FAIL). Not 1D qmap.

Leave-out:
  .venv/bin/python scripts/58_local_pm.py \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --k-genes 8 --alpha 0.08 --knn 8 \\
    --out outputs/t2/embryo/gate725_localpm/lo_k8_a008.h5ad
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
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def local_pm(Xp, q, Cl, Xl, Cr, Xr, knn: int, k_genes: int, alpha: float) -> np.ndarray:
    n = len(q)
    k0 = min(int(knn), len(Cl))
    k1 = min(int(knn), len(Cr))
    kg = int(max(1, min(int(k_genes), Xl.shape[1])))
    _, i0 = cKDTree(Cl).query(q, k=k0)
    _, i1 = cKDTree(Cr).query(q, k=k1)
    i0 = np.asarray(i0, dtype=np.int64).reshape(n, k0)
    i1 = np.asarray(i1, dtype=np.int64).reshape(n, k1)
    m0 = Xl[i0].mean(axis=1)
    m1 = Xr[i1].mean(axis=1)
    mu = 0.5 * (m0 + m1)
    dmu = np.abs(m1 - m0)
    out = np.asarray(Xp, dtype=np.float32).copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    for i in range(n):
        idx = np.argpartition(-dmu[i], kg - 1)[:kg]
        out[i, idx] = (1.0 - a) * out[i, idx] + a * mu[i, idx]
    np.clip(out, 0.0, None, out=out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--k-genes", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.08)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    panel = list(load_panel_for(target_spec(args.setting, dummy_t, cfg), cfg))
    stages = load_stages(args.setting, cfg, panel)
    Xl = _align(to_dense(stages[args.left].X), stages[args.left].var_names, panel)
    Xr = _align(to_dense(stages[args.right].X), stages[args.right].var_names, panel)
    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.asarray(pred.obsm["spatial_3D"], dtype=np.float64)
    rms = float(args.rms)
    Cl = np.asarray(scale_cloud(spatial_xyz(stages[args.left]), rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(stages[args.right]), rms), dtype=np.float64)
    Cq = np.asarray(scale_cloud(C0, rms), dtype=np.float64)
    Xn = local_pm(Xp, Cq, Cl, Xl, Cr, Xr, args.knn, args.k_genes, args.alpha)
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(
        np.asarray(chk.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(C0, dtype=np.float32),
    )
    print(
        f"local-pm k={args.k_genes} α={args.alpha:g} knn={args.knn}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
