#!/usr/bin/env python
"""Radial-profile matching: fix the over-dispersed tail our own geometry stage creates.

Diagnosis (doc t2_p2_val §13): the live heart cloud occupies 289 voxels at n=5000 while
both real anchors occupy 251 (E8.25) and 238 (E8.75), and its radial profile is 2.2x
farther from the anchor-interpolated profile than either real anchor is. The excess sits
entirely in the tail (q98 1.819 vs 1.53 target). Tracing the pipeline shows the tail is
created by our own geometry stage: E8.25 raw q98 1.591 / nocc 251 -> pipeline output
q98 1.790 / nocc 282. `occupancy_dice` is a density-profile statistic, so those 53-59
spurious voxels cost ODS precision directly (0.80 with recall already 0.95), and the
same excess large-radius mass is what d2_distance integrates.

The fix is a monotone radial transport in the canonical frame: keep every cell's ANGLE
and the RANK ORDER of its radius, remap the radius onto the target radial quantile
function, then renormalise the RMS radius. Consequences, all measured in
scripts/diagnostics/w3_radial_cost.py on the W2 real-truth window:
  * X is never touched  -> de_score / de_direction / mmd_u / variogram frozen byte-exact
  * RMS renormalised    -> scale_log_ratio frozen
  * angles + radial rank preserved -> kNN-15 membership overlap 0.953 (random = 0.003),
    neighborhood_mmd moves <= 1.2% (within-cluster shuffle costs x2.8, so this is the
    gentle end of the coordinate-deformation family)
  * coordinate point set changes -> d2_shape and occupancy_dice move (the target)

Usage:
  .venv/bin/python scripts/24_radial_profile_match.py \
    --in outputs/t2/submit/T2_heart_val_interp.h5ad --rms 255 \
    --anchors data/E8.25_late.h5ad data/E8.75.h5ad --w 0.5 \
    --out outputs/t2/queue/2026-09-11/13_heart_radprof_w050_n5000_rms255.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common.shape_metrics import _canonicalise, d2_distance, occupancy_dice  # noqa: E402
from t2.io import to_dense  # noqa: E402

N_GRID, EXTENT = 16, 3.0
QS = np.array([0.10, 0.50, 0.90, 0.98])


def canon(C):
    Z, rms = _canonicalise(np.asarray(C, float))
    return Z, float(rms)


def nocc(Z):
    v = np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)
    return len(set(map(tuple, v)))


def radii(C):
    Z, _ = canon(C)
    return np.sort(np.linalg.norm(Z, axis=1))


def qfun(r_sorted, m):
    """Resample a sorted radius vector onto a common m-point quantile grid."""
    r_sorted = np.asarray(r_sorted, float)
    n = len(r_sorted)
    if n == m:
        return r_sorted.copy()
    return np.interp((np.arange(m) + 0.5) / m, (np.arange(n) + 0.5) / n, r_sorted)


def sub(C, n, seed=0):
    C = np.asarray(C, float)
    if n is None or len(C) <= n:
        return C
    return C[np.random.default_rng(seed).choice(len(C), n, replace=False)]


def radial_match(C, target_sorted_r, out_rms):
    """Monotone radial transport onto a target radius distribution; angles and rank kept."""
    C = np.asarray(C, float)
    Z, _ = canon(C)
    r = np.linalg.norm(Z, axis=1)
    n, m = len(r), len(target_sorted_r)
    new_r = np.interp((np.arange(n) + 0.5) / n, (np.arange(m) + 0.5) / m, target_sorted_r)
    order = np.argsort(r, kind="stable")
    out = np.empty_like(Z)
    out[order] = (Z[order] / np.maximum(r[order, None], 1e-12)) * new_r[:, None]
    # remapping radii shifts the centroid off zero; _canonicalise re-centres, so centre
    # first and then renormalise, otherwise the written RMS radius misses the target by
    # ~3e-4 and scale_log_ratio (currently a saturated 100.0) would start to drift.
    out = out - out.mean(0)
    out /= max(np.sqrt((out ** 2).sum(1).mean()), 1e-12)
    return out * float(out_rms)


def describe(tag, C, anchors, target_r):
    Z, rms = canon(C)
    row = [f"  {tag:34s} n {len(C):6d} rms {rms:7.2f} nocc {nocc(Z):4d}"]
    q = np.quantile(np.linalg.norm(Z, axis=1), QS)
    row.append("q " + " ".join(f"{v:.3f}" for v in q))
    rz = np.sort(np.linalg.norm(Z, axis=1))
    row.append(f"prof_dist {np.linalg.norm(qfun(rz, len(target_r)) - target_r)/np.sqrt(len(target_r)):.4f}")
    for nm, ac in anchors:
        d, _ = occupancy_dice(C, ac, seed=0)
        row.append(f"| {nm}: dice {d:.4f} d2 {d2_distance(C, ac, seed=0):.5f}")
    print("  ".join(row))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchors", nargs=2, required=True, help="left right anchor h5ad (spatial clouds)")
    ap.add_argument("--w", type=float, default=0.5, help="weight of the RIGHT anchor profile")
    ap.add_argument("--rms", type=float, required=True, help="target RMS radius (keep the live value)")
    ap.add_argument("--anchor-n", type=int, default=None, help="subsample anchors for the profile (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not 0.0 <= args.w <= 1.0:
        raise SystemExit("--w must be in [0,1]")

    a = ad.read_h5ad(args.src)
    X0 = to_dense(a.X)
    C0 = np.asarray(a.obsm["spatial_3D"], float)
    A = np.asarray(ad.read_h5ad(args.anchors[0]).obsm["spatial_3D"], float)
    B = np.asarray(ad.read_h5ad(args.anchors[1]).obsm["spatial_3D"], float)
    A, B = sub(A, args.anchor_n), sub(B, args.anchor_n)
    rA, rB = radii(A), radii(B)
    n5 = min(len(A), 5000)
    anchors = [(Path(args.anchors[0]).stem, sub(A, n5)), (Path(args.anchors[1]).stem, sub(B, n5))]
    m = len(C0)
    target = (1.0 - args.w) * qfun(rA, m) + args.w * qfun(rB, m)

    print(f"target profile = (1-{args.w:g})*{anchors[0][0]} + {args.w:g}*{anchors[1][0]}, "
          f"anchor n={len(rA)}/{len(rB)}")
    describe("BEFORE (live)", C0, anchors, target)
    C1 = radial_match(C0, target, args.rms)
    describe(f"AFTER  (w={args.w:g})", C1, anchors, target)

    Z0, _ = canon(C0); Z1, _ = canon(C1)
    print(f"  nocc {nocc(Z0)} -> {nocc(Z1)}  |  anchor bracket "
          f"{nocc(canon(anchors[0][1])[0])}/{nocc(canon(anchors[1][1])[0])}")

    if args.dry_run:
        return 0
    out = ad.AnnData(np.array(X0, dtype=np.float32))
    out.var_names = list(a.var_names)
    out.obs_names = list(a.obs_names)
    out.obsm["spatial_3D"] = np.asarray(C1, dtype=np.float32)
    for k, v in a.uns.items():
        out.uns[k] = v
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")

    chk = ad.read_h5ad(args.out)
    assert list(chk.var_names) == list(a.var_names), "panel order changed"
    assert chk.shape == a.shape, "shape changed"
    assert np.array_equal(to_dense(chk.X), X0), "X changed"
    assert np.isfinite(to_dense(chk.X)).all() and np.isfinite(np.asarray(chk.obsm["spatial_3D"], float)).all()
    assert to_dense(chk.X).min() >= 0.0, "negative expression"
    r = canon(np.asarray(chk.obsm["spatial_3D"], float))[1]
    assert abs(r - args.rms) < 1e-3, f"rms {r} != {args.rms}"
    print(f"wrote {args.out}  (X byte-identical, panel order kept, rms {r:.2f}, "
          f"nocc {nocc(canon(np.asarray(chk.obsm['spatial_3D'], float))[0])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
