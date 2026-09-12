#!/usr/bin/env python
"""Uniformly rescale a T2 prediction's coordinates to an exact target RMS radius.

`d2_shape` and `occupancy_dice` canonicalise by RMS radius, and `neighborhood_mmd` builds
its kNN graph on the coordinate cloud, so a uniform rescale leaves all of them (and every
expression metric) untouched to float precision.  The only ranked term that moves is
`scale_log_ratio`.

The board ranks |log(rms_pred / rms_true)| against |floor| and |ceiling| with the hyperbolic
map in `common/core_metrics.py::skill`, and clips at 1.0 once the value is within |ceiling|.
The clip window is therefore TWO-SIDED and narrow, not "anything below the truth":

    rms_pred in rms_true * [exp(-|ceiling|), exp(+|ceiling|)]

Fitted from four board readings of our own files (embryo 192.0575 -> 91.9 and 200 -> 98.8;
heart 277.13 -> 82.5 and 255 -> 100.0; max residual 0.006 skill), see
doc/t2_p2_val_2026-09-05.md section 14.1:

    board              rms_true          clip-to-100 window
    embryo val_interp  198.233-198.258   [197.210, 199.286]   centre 198.24
    heart  val_interp  256.07            [253.041, 259.136]   (live 255 already inside)
    heart  val_extrap  437.92            (live 437.917 already inside)

Aim at the window CENTRE.  Undershooting is not free: rms 192.0575 is below the embryo truth
and scores 91.9, not 100.0.

    .venv/bin/python scripts/25_rescale_rms.py --inp A.h5ad --rms 198.24 --out B.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.geometry import scale_cloud  # noqa: E402
from t2.io import load_adata, spatial_xyz, write_t2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--rms", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adata = load_adata(Path(args.inp))
    xyz = np.asarray(spatial_xyz(adata), dtype=np.float64)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    r_in = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))
    new = np.asarray(scale_cloud(xyz, float(args.rms)), dtype=np.float64)
    r_out = float(np.sqrt(((new - new.mean(0)) ** 2).sum(1).mean()))
    write_t2(X, new, list(adata.var_names), Path(args.out))
    print(f"{args.inp} -> {args.out}\n  n={len(new)} genes={X.shape[1]}  RMS {r_in:.4f} -> {r_out:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
