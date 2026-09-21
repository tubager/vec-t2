#!/usr/bin/env python
"""Residual proposal → snap to a real left/right cell → occupancy-calibrate.

Leave-out (E6.75+E8.0 → E7.25): NFS 0.0465 / mmd 0.0149 / var 0.0171 / de 0.345
with ODS 0.884 after occ-cal onto spatial (AND + board PASS). Chimeric residual
alone explodes variogram; snapping restores a real coexpression vector while
keeping the residual's cell *choice*. occ-cal keeps native pairing (NFS tax
~0.0004) unlike Hungarian freeze (NFS 0.046→0.058).

Leave-out:
  .venv/bin/python scripts/46_token_birth.py --left E6.75 --right E8.0 --w 0.4 \\
    --prior outputs/t2/embryo/gate725/slice_band0.25.h5ad --mode resid \\
    --sites copy --base prior --resid-alpha 0.35 --resid-dir outputs/t2/embryo/gate725_resid \\
    --out /tmp/resid.h5ad
  .venv/bin/python scripts/49_resid_snap_occ.py --in /tmp/resid.h5ad \\
    --left E6.75 --right E8.0 --occ-prior outputs/t2/embryo/gate725/slice_spatial_k0.4_b6.h5ad \\
    --out lo.h5ad

Board:
  same with --left E7.25 --right E8.0 and occ-prior = live 09a.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def snap_to_pool(X, pool, pca: ExpressionPCA | None):
    Q = np.asarray(X, dtype=np.float32)
    P = np.asarray(pool, dtype=np.float32)
    if pca is not None:
        q = pca.encode(Q)
        p = pca.encode(P)
    else:
        q, p = Q, P
    _, nn = cKDTree(p).query(q, k=1)
    return P[np.asarray(nn, dtype=np.int64)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--occ-prior", type=Path, default=None)
    ap.add_argument("--pca", type=Path, default=ROOT / "outputs/t2/embryo/gate725_resid/pca.joblib")
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--target-nocc", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    panel = list(load_panel_for(target_spec("embryo", 7.5, cfg), cfg))
    stages = load_stages("embryo", cfg, panel)
    Xl = _align(to_dense(stages[args.left].X), stages[args.left].var_names, panel)
    Xr = _align(to_dense(stages[args.right].X), stages[args.right].var_names, panel)
    pool = np.concatenate([Xl, Xr], axis=0)
    src = ad.read_h5ad(args.src)
    X = _align(to_dense(src.X), src.var_names, panel)
    pca = ExpressionPCA.load(args.pca) if Path(args.pca).exists() else None
    Xs = np.clip(snap_to_pool(X, pool, pca), 0.0, None).astype(np.float32)
    n_changed = int(np.mean(np.any(np.abs(Xs - X) > 1e-5, axis=1)) * len(X))
    src.X = Xs
    snap_path = args.out if args.occ_prior is None else args.out.with_name(args.out.stem + ".__snap__.h5ad")
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    src.write_h5ad(snap_path, compression="gzip")
    print(f"snap {n_changed}/{len(X)} rows onto {args.left}+{args.right} pool  pca={pca is not None}")
    if args.occ_prior is None:
        print(f"wrote {snap_path}")
        return 0
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "45_occ_cal_place.py"),
        "--in",
        str(snap_path),
        "--prior",
        str(args.occ_prior),
        "--rms",
        str(args.rms),
        "--out",
        str(args.out),
    ]
    if args.target_nocc is not None:
        cmd.extend(["--target-nocc", str(args.target_nocc)])
    subprocess.check_call(cmd)
    snap_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
