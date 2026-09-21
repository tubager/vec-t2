#!/usr/bin/env python
"""Soft occupy calibrate: move each cell toward nearest template point; keep X.

xyz' = (1-α) * xyz + α * NN_template(xyz). Preserves (X, cell) rows; only geometry.
Use true mid cloud for leave-out gate, live09a for board occupy lift.

  .venv/bin/python scripts/41_soft_occupy.py \\
    --base outputs/t2/embryo/gate75_native2/board_spatial.h5ad \\
    --template outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --alpha 0.25 --out out.h5ad
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    a = float(args.alpha)
    if not (0.0 < a <= 1.0):
        raise SystemExit("--alpha in (0,1]")

    base = ad.read_h5ad(args.base)
    tmpl = ad.read_h5ad(args.template)
    Cb = np.asarray(base.obsm["spatial_3D"], dtype=np.float64)
    Ct = np.asarray(scale_cloud(tmpl.obsm["spatial_3D"], float(args.rms)), dtype=np.float64)
    _, nn = cKDTree(Ct).query(Cb, k=1)
    Cnew = (1.0 - a) * Cb + a * Ct[nn]
    Cnew = np.asarray(scale_cloud(Cnew, float(args.rms)), dtype=np.float32)

    out = base.copy()
    X0 = np.asarray(base.X.todense() if hasattr(base.X, "todense") else base.X)
    out.obsm["spatial_3D"] = Cnew
    out.X = base.X  # ensure same object / values
    X1 = np.asarray(out.X.todense() if hasattr(out.X, "todense") else out.X)
    if not np.array_equal(X0, X1):
        raise SystemExit("X changed — abort")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    disp = np.linalg.norm(Cnew.astype(float) - Cb, axis=1)
    print(
        f"wrote {args.out} alpha={a} median|Δxyz|={float(np.median(disp)):.3f} "
        f"rms={rms_radius(Cnew):.2f} X frozen"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
