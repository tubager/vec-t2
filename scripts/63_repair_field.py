#!/usr/bin/env python
"""Re-pair existing X onto a locked xyz sequence.

Does not invent expression: the 5000 gene vectors are a permutation of --pred.
Hungarian matches those vectors to an L/R spatial local-pool field at the same
xyz, so neighborhood pairing can move without touching variogram/de/mmd sets.

Leave-out:
  .venv/bin/python scripts/63_repair_field.py \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --w 0.4 \\
    --out outputs/t2/embryo/gate725_repair/pm_hung_pool.h5ad

Board (xyz = 09a bit-exact):
  .venv/bin/python scripts/63_repair_field.py \\
    --pred outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --left E7.25 --right E8.0 --w 0.333 \\
    --out outputs/t2/queue/2026-09-19/192_embryo_repair_hung_pool_w033_on09a_n5000.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.optimize import linear_sum_assignment
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


def _hungarian_rows(X: np.ndarray, field: np.ndarray, xyz: np.ndarray | None = None, lam: float = 0.0) -> np.ndarray:
    a = np.asarray(X, dtype=np.float32)
    b = np.asarray(field, dtype=np.float32)
    cost = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    if xyz is not None and float(lam) > 0:
        z = np.asarray(xyz, dtype=np.float32)
        z2 = (z * z).sum(1)
        sc = z2[:, None] + z2[None, :] - 2.0 * (z @ z.T)
        emed = float(np.median(cost[cost > 0])) if np.any(cost > 0) else 1.0
        smed = float(np.median(sc[sc > 0])) if np.any(sc > 0) else 1.0
        cost = cost / max(emed, 1e-8) + float(lam) * sc / max(smed, 1e-8)
    ri, ci = linear_sum_assignment(cost)
    inv = np.empty(len(X), dtype=np.int64)
    inv[ci] = ri
    return inv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--w", type=float, default=0.4)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")
    ap.add_argument("--space", choices=["gene", "pca"], default="gene")
    ap.add_argument("--lam", type=float, default=0.0, help="spatial identity regularizer (0=192)")
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--seed", type=int, default=0)
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

    k = int(args.knn)
    w = float(np.clip(args.w, 0.0, 1.0))
    _, i0 = cKDTree(Cl).query(Cq, k=min(k, len(Cl)))
    _, i1 = cKDTree(Cr).query(Cq, k=min(k, len(Cr)))
    i0 = np.atleast_2d(np.asarray(i0, dtype=np.int64).reshape(len(Cq), min(k, len(Cl))))
    i1 = np.atleast_2d(np.asarray(i1, dtype=np.int64).reshape(len(Cq), min(k, len(Cr))))
    pool_l, pool_r = Xl[i0], Xr[i1]
    if args.agg == "median":
        field = (1.0 - w) * np.median(pool_l, axis=1) + w * np.median(pool_r, axis=1)
    else:
        field = (1.0 - w) * pool_l.mean(1) + w * pool_r.mean(1)

    match_X, match_F = Xp, field
    if args.space == "pca":
        from t2.pca import ExpressionPCA  # noqa: E402

        pca = ExpressionPCA(n_components=args.pca_dim, batch_size=int(cfg["pca_batch"]))
        pca.fit([left, right], n_per_stage=max(1, int(cfg["pca_fit_cells"]) // 2), seed=args.seed)
        match_X, match_F = pca.encode(Xp), pca.encode(field)

    inv = _hungarian_rows(match_X, match_F, xyz=Cq, lam=float(args.lam))
    Xn = Xp[inv]
    # permutation of the same vectors
    assert np.allclose(np.sort(Xn, axis=0), np.sort(Xp, axis=0), atol=1e-5)

    out = pred.copy()
    out.X = Xn.astype(np.float32)
    out.obsm["spatial_3D"] = C0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"], dtype=np.float32), C0)
    n_moved = float((inv != np.arange(len(inv))).mean())
    print(
        f"repair hung w={w:g} knn={k} agg={args.agg} space={args.space} lam={args.lam:g}  moved={n_moved:.3f}  "
        f"mean|X-field| {np.mean(np.abs(Xp - field)):.4f}→{np.mean(np.abs(Xn - field)):.4f}  "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
