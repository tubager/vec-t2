#!/usr/bin/env python
"""Local operators on a locked submission.

svd: MAGIC-style local SVD denoise. xyz bit-exact. Uses only the pred's own
     spatial neighbors (not a new donor library).
occ: freeze X, replace xyz with occupancy-histogram interpolant of left/right,
     Hungarian-assign to keep pairing as close as possible.

  .venv/bin/python scripts/52_local_ops.py --mode svd --pred ... --k 16 --rank 4 --alpha 0.25 --out ...
  .venv/bin/python scripts/52_local_ops.py --mode occ --pred ... --left E6.75 --right E8.0 --w 0.4 --out ...
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

from t2.geometry import rms_radius, scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.shape import match_axis_flips, occupancy_interpolate_cloud, ot_assign_xyz  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def local_svd(X: np.ndarray, C: np.ndarray, k: int, rank: int, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    n = len(X)
    kk = int(min(max(int(k), 3), n))
    r = int(min(max(int(rank), 1), kk, X.shape[1]))
    a = float(np.clip(alpha, 0.0, 1.0))
    _, idx = cKDTree(C).query(C, k=kk)
    idx = np.atleast_2d(np.asarray(idx, dtype=np.int64).reshape(n, kk))
    out = X.copy()
    for i in range(n):
        loc = X[idx[i]]
        mu = loc.mean(0)
        xc = loc - mu
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
        rr = min(r, int((s > 1e-8).sum()) or 1)
        rec = mu + (xc @ vt[:rr].T) @ vt[:rr]
        # row 0 is the query itself (cKDTree returns self first)
        out[i] = (1.0 - a) * X[i] + a * rec[0]
    np.clip(out, 0.0, None, out=out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["svd", "occ"], required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--w", type=float, default=0.4)
    ap.add_argument("--occ-alpha", type=float, default=1.0, help="occ xyz blend; 1=full replace")
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pred = ad.read_h5ad(args.pred)
    Xp = np.asarray(to_dense(pred.X), dtype=np.float64)
    C0 = np.asarray(pred.obsm["spatial_3D"], dtype=np.float64)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "svd":
        Xn = local_svd(Xp, C0, args.k, args.rank, args.alpha)
        out = pred.copy()
        out.X = Xn.astype(np.float32)
        out.write_h5ad(args.out, compression="gzip")
        chk = ad.read_h5ad(args.out)
        assert np.array_equal(np.asarray(chk.obsm["spatial_3D"]), C0), "xyz changed"
        print(
            f"svd k={args.k} rank={args.rank} α={args.alpha:g}  "
            f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  wrote {args.out}"
        )
        return 0

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    panel = list(load_panel_for(target_spec(args.setting, dummy_t, cfg), cfg))
    stages = load_stages(args.setting, cfg, panel)
    rms = float(rms_radius(C0))
    Cl = np.asarray(scale_cloud(spatial_xyz(stages[args.left]), rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(stages[args.right]), rms), dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    Cr = match_axis_flips(Cl, Cr, rng, n_pair=800)
    occ = occupancy_interpolate_cloud(Cl, Cr, float(args.w), n=len(C0), rng=rng, n_bins=int(args.bins))
    C_occ = ot_assign_xyz(C0, occ)
    a = float(np.clip(args.occ_alpha, 0.0, 1.0))
    C1 = (1.0 - a) * C0 + a * np.asarray(C_occ, dtype=np.float64)
    C1 = np.asarray(scale_cloud(C1, rms), dtype=np.float32)
    out = pred.copy()
    out.obsm["spatial_3D"] = C1
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(to_dense(chk.X), to_dense(pred.X)), "X changed"
    print(
        f"occ w={args.w:g} α={a:g} bins={args.bins} rms={rms:.2f}  "
        f"mean|dC|/rms={float(np.mean(np.linalg.norm(C1 - C0, axis=1)) / max(rms, 1e-8)):.4f}  "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
