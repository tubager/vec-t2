#!/usr/bin/env python
"""Coordinate swap: keep one file's X/obs verbatim, take another file's coordinate CLOUD.

Why: `d2_shape` / `occupancy_dice` / `scale_log_ratio` depend only on the submitted
coordinate point set (count-matched, fixed seeds), and the expression/state metrics
depend only on X. Swapping clouds therefore reproduces the donor's shape scores
exactly while freezing the base's expression scores exactly; only
`neighborhood_mmd` (which reads the X<->coord pairing) can move. The donor cloud is
OT-assigned back onto the base cells (exact Hungarian at n<=5000) so the pairing
deformation is minimal.

Usage:
  .venv/bin/python scripts/19_coord_swap.py \
    --base  outputs/t2/heart/pred_E8.5_..._w0.25_n5000_rms255.h5ad \
    --donor /tmp/t2d/repro_w05.h5ad --rms 255 \
    --out   outputs/t2/heart/pred_E8.5_..._xw025_cw05_n5000_rms255.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import anndata as ad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.geometry import scale_cloud  # noqa: E402
from t2.shape import ot_assign_xyz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="file whose X/obs/var are kept verbatim")
    ap.add_argument("--donor", required=True, help="file whose coordinate cloud is taken")
    ap.add_argument("--rms", type=float, default=None, help="rescale donor cloud to this RMS first")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = ad.read_h5ad(args.base)
    donor = ad.read_h5ad(args.donor)
    zb = np.asarray(base.obsm["spatial_3D"], dtype=np.float64)
    zd = np.asarray(donor.obsm["spatial_3D"], dtype=np.float64)
    if len(zb) != len(zd):
        raise SystemExit(f"count mismatch: base {len(zb)} vs donor {len(zd)}")
    if args.rms is not None:
        zd = np.asarray(scale_cloud(zd, float(args.rms)), dtype=np.float64)
    new = np.asarray(ot_assign_xyz(zb, zd), dtype=np.float64)

    keep = {tuple(np.round(r, 3)) for r in new}
    want = {tuple(np.round(r, 3)) for r in zd}
    if keep != want:
        raise SystemExit("internal error: swapped cloud != donor cloud")
    disp = np.linalg.norm(new - zb, axis=1)
    print(f"cloud preserved ({len(keep)} unique), rms={float(np.sqrt(((new-new.mean(0))**2).sum(1).mean())):.2f}, "
          f"median displacement={float(np.median(disp)):.2f}")

    base.obsm["spatial_3D"] = new.astype(np.float32)
    base.write_h5ad(args.out, compression="gzip")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
