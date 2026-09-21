#!/usr/bin/env python
"""Heart interp: freeze X, move only interior cells toward a template occupancy.

crop-keep (58.40) dropped peripheral cells → composition smash.
occ full-replace (66.42) refilled the envelope → d2 72.6→36.3.
This keeps every cell and the outer envelope; only the core is softly occupied
onto E8.75 (or an occupancy interpolant), then RMS is restored to 255.

  .venv/bin/python scripts/50_heart_interior_occupy.py \\
    --base outputs/t2/submit/T2_heart_val_interp.h5ad \\
    --template data/E8.75.h5ad --peri 0.35 --alpha 0.25 --rms 255 \\
    --out outputs/t2/heart/interior/live_peri035_a025.h5ad
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
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))

from common.shape_metrics import d2_distance, occupancy_dice  # noqa: E402
from t2.geometry import canonicalize, rms_radius, scale_cloud  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.shape import _PROPER_FLIPS  # noqa: E402

N_GRID, EXTENT = 16, 3.0


def nocc(C: np.ndarray) -> int:
    Z, r, _ = canonicalize(C)
    Z = Z / max(r, 1e-12)
    v = np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)
    return len(set(map(tuple, v)))


def q98(C: np.ndarray) -> float:
    Z, r, _ = canonicalize(C)
    Z = Z / max(r, 1e-12)
    return float(np.quantile(np.linalg.norm(Z, axis=1), 0.98))


def best_flip(Z_src: np.ndarray, Z_ref: np.ndarray) -> np.ndarray:
    """det=+1 axis sign of src that maximises occupied-voxel overlap with ref."""
    def occ(Z):
        v = np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)
        return set(map(tuple, v))

    Zr = Z_ref / max(float(np.sqrt((Z_ref**2).sum(1).mean())), 1e-12)
    Zs0 = Z_src / max(float(np.sqrt((Z_src**2).sum(1).mean())), 1e-12)
    ob = occ(Zr)
    best, best_n = _PROPER_FLIPS[0], -1
    for f in _PROPER_FLIPS:
        n = len(occ(Zs0 * f) & ob)
        if n > best_n:
            best, best_n = f, n
    return Z_src * best


def interior_occupy(
    C: np.ndarray,
    C_tmpl: np.ndarray,
    peri: float,
    alpha: float,
    rms: float,
) -> tuple[np.ndarray, dict]:
    C = np.asarray(C, dtype=np.float64)
    T = np.asarray(C_tmpl, dtype=np.float64)
    mu = C.mean(0)
    Z, r, Vt = canonicalize(C)
    Zt, rt, _ = canonicalize(T)
    Zt = best_flip(Zt, Z)
    Zs = (Z / max(r, 1e-12)) * float(rms)
    Zt_s = (Zt / max(rt, 1e-12)) * float(rms)
    rad = np.linalg.norm(Zs, axis=1)
    order = np.argsort(rad, kind="stable")
    n_peri = int(round(float(np.clip(peri, 0.0, 0.95)) * len(C)))
    n_peri = min(max(n_peri, 0), len(C) - 1)
    peri_idx = order[-n_peri:] if n_peri else np.array([], dtype=np.int64)
    interior = order[:-n_peri] if n_peri else order
    Znew = Zs.copy()
    if len(interior) and float(alpha) > 0:
        _, nn = cKDTree(Zt_s).query(Zs[interior], k=1)
        Znew[interior] = (1.0 - float(alpha)) * Zs[interior] + float(alpha) * Zt_s[nn]
    C_out = (Znew @ Vt) + mu
    C_out = np.asarray(scale_cloud(C_out, float(rms)), dtype=np.float64)
    disp = np.linalg.norm(C_out - C, axis=1)
    info = {
        "n_interior": int(len(interior)),
        "n_peri": int(len(peri_idx)),
        "median_disp": float(np.median(disp)),
        "median_disp_interior": float(np.median(disp[interior])) if len(interior) else 0.0,
        "median_disp_peri": float(np.median(disp[peri_idx])) if len(peri_idx) else 0.0,
        "max_disp_peri": float(disp[peri_idx].max()) if len(peri_idx) else 0.0,
    }
    return C_out, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--peri", type=float, default=0.35, help="outer fraction whose xyz is frozen")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--rms", type=float, default=255.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not (0.0 <= args.alpha <= 1.0):
        raise SystemExit("--alpha in [0,1]")
    if not (0.0 <= args.peri < 1.0):
        raise SystemExit("--peri in [0,1)")

    base = ad.read_h5ad(args.base)
    tmpl = ad.read_h5ad(args.template)
    X0 = to_dense(base.X)
    Cb = np.asarray(base.obsm["spatial_3D"], dtype=np.float64)
    Ct = np.asarray(tmpl.obsm["spatial_3D"], dtype=np.float64)
    Cnew, info = interior_occupy(Cb, Ct, args.peri, args.alpha, args.rms)

    out = base.copy()
    out.obsm["spatial_3D"] = np.asarray(Cnew, dtype=np.float32)
    out.X = base.X
    X1 = to_dense(out.X)
    if not np.array_equal(X0, X1):
        raise SystemExit("X changed — abort")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")

    chk = to_dense(ad.read_h5ad(args.out).X)
    if not np.array_equal(chk, X0):
        raise SystemExit("X changed on disk — abort")
    print(
        f"wrote {args.out} peri={args.peri:.2f} alpha={args.alpha:.2f} "
        f"rms={rms_radius(Cnew):.2f} nocc {nocc(Cb)}→{nocc(Cnew)} "
        f"q98 {q98(Cb):.3f}→{q98(Cnew):.3f} "
        f"d2_vs_live={d2_distance(Cnew, Cb, seed=0):.5f} "
        f"dice_vs_live={occupancy_dice(Cnew, Cb, seed=0)[0]:.4f} "
        f"n_int={info['n_interior']} med|d|_int={info['median_disp_interior']:.3f} "
        f"med|d|_peri={info['median_disp_peri']:.3f} max|d|_peri={info['max_disp_peri']:.3f} "
        f"X frozen"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
