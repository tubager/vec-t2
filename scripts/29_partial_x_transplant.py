#!/usr/bin/env python
"""Partial X transplant onto a locked coordinate sequence.

Keeps ``--live`` xyz SEQUENCE bit-exact. Replaces a fraction ``--frac`` of rows'
expression with the OT-matched donor X (Hungarian on coordinates). Use to take
small steps away from a board-locked file without full X swap (which anti-transfers).

Usage:
  .venv/bin/python scripts/29_partial_x_transplant.py \\
    --live outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --donor outputs/t2/embryo/pred_E7.5_local_pick_k8_w0.15_n5000.h5ad \\
    --frac 0.15 --rms 198.24 --out out.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.geometry import scale_cloud  # noqa: E402


def dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True, type=Path)
    ap.add_argument("--donor", required=True, type=Path)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--rms", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--pick",
        choices=["random", "farthest"],
        default="farthest",
        help="which live rows to replace: random, or farthest from matched donor X",
    )
    args = ap.parse_args()
    if not (0.0 < float(args.frac) <= 1.0):
        raise SystemExit(f"--frac must be in (0,1], got {args.frac}")

    live = ad.read_h5ad(args.live)
    donor = ad.read_h5ad(args.donor)
    Cl = np.asarray(live.obsm["spatial_3D"], dtype=np.float64)
    Cd = np.asarray(donor.obsm["spatial_3D"], dtype=np.float64)
    if args.rms is not None:
        Cd = np.asarray(scale_cloud(Cd, float(args.rms)), dtype=np.float64)
    if len(Cl) != len(Cd):
        raise SystemExit(f"n mismatch live {len(Cl)} vs donor {len(Cd)}")

    # Hungarian match donor cloud → live cloud (assign donor rows onto live positions)
    # subsample cost if needed — n=5000 exact is ok but slow; use float32 chunks
    a = Cl.astype(np.float32)
    b = Cd.astype(np.float32)
    # squared distances via (a^2 + b^2 - 2 a·b)
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    cost = a2 + b2 - 2.0 * (a @ b.T)
    ri, ci = linear_sum_assignment(cost)
    # ri should be 0..n-1 permutation; map live row i ← donor row ci[inv]
    order = np.empty(len(Cl), dtype=np.int64)
    order[ri] = ci
    Xd = dense(donor.X)[order]
    Xl = dense(live.X)
    if Xd.shape[1] != Xl.shape[1]:
        raise SystemExit(f"gene mismatch {Xd.shape[1]} vs {Xl.shape[1]}")

    n = len(Cl)
    k = int(round(float(args.frac) * n))
    k = max(1, min(n, k))
    rng = np.random.default_rng(int(args.seed))
    if args.pick == "random":
        take = rng.choice(n, size=k, replace=False)
    else:
        dist = np.linalg.norm(Xl - Xd, axis=1)
        take = np.argsort(-dist)[:k]

    Xnew = Xl.copy()
    Xnew[take] = Xd[take]
    out = live.copy()
    out.X = Xnew.astype(np.float32)
    # xyz unchanged (sequence lock)
    assert np.array_equal(
        np.asarray(out.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(live.obsm["spatial_3D"], dtype=np.float32),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    print(
        f"replaced {k}/{n} rows ({100.0 * k / n:.1f}%) pick={args.pick}  "
        f"mean|dX|={float(np.mean(np.abs(Xnew - Xl))):.4f}  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
