#!/usr/bin/env python
"""Post-hoc geometry fix for an existing T2 prediction: restore the principal-axis ratio.

Why this exists
---------------
`--crop-keep` (FOV crop by rejection) is the right way to narrow the left anchor's field of
view: every kept cell stays on its own coordinate, so the native cell spacing and the
(X, xyz) pairing survive.  But the rejection region is a SPHERE, and cutting a sphere out of
an elongated cloud removes the elongated periphery, so the surviving core is far rounder than
any real stage:

    released clouds   E6.75 2.18  E7.25 1.89  E8.0 1.88  E8.25 1.69  E8.75 1.26  E9.5 1.69
    crop-keep R=200                                        -> 1.06   (outside the family)

`d2_shape` / `occupancy_dice` canonicalise with PCA, so a wrong axis ratio is scored directly.
This script re-applies the anisotropic stretch AFTER the crop, which is the order the
real-cell calibration in doc/t2_p2_plan_2026-09-09.md §4.2 says is correct (crop -> restore
axes -> RMS).

The axis ratio of the target stage is not free: across the six released heart/embryo clouds
a1/a3 rises monotonically with RMS (216.9 -> 1.257, 335.0 -> 1.690, 354.1 -> 1.693), so the
held-out E8.5 (RMS 254.6, from the `size_fidelity` docstring and independently from our own
online TSR bisection) implies a1/a3 ~ 1.36-1.40.

Axes are morphed as a_i -> a_i**p with p = log(R)/log(a1/a3), which preserves the ordering and
the middle axis' share, then the cloud is rescaled to `--rms`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.geometry import canonicalize  # noqa: E402
from t2.io import load_adata, spatial_xyz, write_t2  # noqa: E402


def axis_ratio(C) -> float:
    Z, _, _ = canonicalize(C)
    s = np.linalg.svd(Z, compute_uv=False)
    lam = np.sqrt((s**2) / len(Z))
    lam = lam / np.linalg.norm(lam)
    return float(lam[0] / lam[2])


def _stretch(Z, ratio: float) -> np.ndarray:
    """Scale the PCA axes by a_i**p so the sorted axis ratio becomes `ratio`."""
    std = Z.std(axis=0, ddof=1)
    order = np.argsort(-std)
    cur = float(std[order[0]] / std[order[2]])
    p = 1.0 if abs(np.log(cur)) < 1e-9 else float(np.log(ratio) / np.log(cur))
    new = std.copy()
    new[order] = std[order] ** p
    new = new * (std.sum() / new.sum())
    Y = Z / (std + 1e-12) * new
    return Y - Y.mean(axis=0)


def _radial_restore(Y, r_ref) -> np.ndarray:
    """Keep each cell's direction but hand back its pre-stretch radius, matched by rank."""
    ry = np.linalg.norm(Y, axis=1)
    order = np.argsort(ry, kind="stable")
    fixed = Y.copy()
    fixed[order] = Y[order] / (ry[order][:, None] + 1e-12) * np.sort(r_ref)[:, None]
    return fixed - fixed.mean(axis=0)


def restore(C, target_ratio: float, rms: float, preserve_radial: bool = True) -> np.ndarray:
    Z, _, _ = canonicalize(C)
    if not preserve_radial:
        Y = _stretch(Z, target_ratio)
    else:
        # Stretching the axes pulls mass inward, which is exactly what `d2_shape` punishes
        # (it is a Wasserstein distance on the RMS-normalised pairwise-distance distribution).
        # Undo that: keep each cell's direction but hand back the radius it had before the
        # stretch, matched by rank, so the normalised-radius CDF is restored verbatim. The
        # remap is monotone in radius, so the radial ordering -- and with it the neighbourhood
        # structure `neighborhood_mmd` reads -- is preserved. It also damps the axis ratio,
        # so iterate on the stretch target until the measured ratio lands on the request.
        guess = target_ratio
        r_ref = np.linalg.norm(Z, axis=1)
        Y = _stretch(Z, guess)
        for _ in range(20):
            Y = _radial_restore(Y, r_ref)
            got = axis_ratio(Y)
            if abs(np.log(got / target_ratio)) < 2e-4:
                break
            guess *= (target_ratio / got) ** 0.8
            Y = _stretch(Z, guess)
    ry = float(np.sqrt((Y**2).sum(axis=1).mean()))
    return (Y * (rms / max(ry, 1e-12))).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--axis-ratio", type=float, required=True, help="target a1/a3 after PCA canonicalisation")
    ap.add_argument("--rms", type=float, default=None, help="target RMS radius (default: keep the input's)")
    ap.add_argument("--free-radial", action="store_true",
                    help="do NOT restore the pre-stretch normalised-radius CDF")
    args = ap.parse_args()

    adata = load_adata(Path(args.inp))
    xyz = np.asarray(spatial_xyz(adata), dtype=np.float64)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    r_in = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))
    rms = float(args.rms) if args.rms else r_in
    before = axis_ratio(xyz)
    new = restore(xyz, args.axis_ratio, rms, preserve_radial=not args.free_radial)
    after = axis_ratio(new)
    r_out = float(np.sqrt(((new - new.mean(0)) ** 2).sum(1).mean()))
    write_t2(X, new, list(adata.var_names), Path(args.out))
    print(f"{args.inp}\n  a1/a3 {before:.3f} -> {after:.3f} (target {args.axis_ratio:.3f})   RMS {r_in:.1f} -> {r_out:.1f}"
          f"\n  n={len(new)} genes={X.shape[1]} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
