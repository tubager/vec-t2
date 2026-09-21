#!/usr/bin/env python
"""TPS-cloud native sites: sample birth xyz from TPS/spatial support, attach real neighbor X.

Fixes voxel-histogram ODS collapse by using the leave-out-proven TPS left cloud as the
occupy support. xyz stays on the sampled site (not donor own, not 09a freeze).

  .venv/bin/python scripts/43_tps_site_birth.py --left E6.75 --right E8.0 --w 0.4 \\
    --support spatial --n 5000 --out out.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense, write_t2  # noqa: E402
from t2.ot import _progress_axis  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402
from t2.shape import match_axis_flips, ot_interpolate_clouds, transform_cloud  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--support", choices=["tps", "spatial"], default="tps",
                    help="tps=TPS-morphed left cloud; spatial=same + we still use TPS left as sites")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--mode", choices=["pick", "pool", "slice"], default="pick")
    ap.add_argument("--slice-keep", type=float, default=0.4)
    ap.add_argument("--jitter", type=float, default=0.0, help="fraction of median NN as isotropic jitter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    panel = list(load_panel_for(target_spec("embryo", 7.5, cfg), cfg))
    stages = load_stages("embryo", cfg, panel)
    left, right = stages[args.left], stages[args.right]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Cl_raw, Cr_raw = spatial_xyz(left), spatial_xyz(right)
    tps_n = int((cfg.get("shape") or {}).get("tps_n") or 800)
    Cl = ot_interpolate_clouds(Cl_raw, Cr_raw, float(args.w), float(args.rms), n_pair=tps_n, rng=rng)
    Cr = np.asarray(transform_cloud(Cr_raw, float(args.rms), mode="isotropic"), dtype=np.float32)
    Cr = match_axis_flips(Cl, Cr, rng, n_pair=min(6000, len(Cl), len(Cr)))
    Cl = np.asarray(Cl, dtype=np.float64)
    Cr = np.asarray(Cr, dtype=np.float64)

    # Birth sites = subsample of TPS left cloud (proven occupy family)
    n = int(args.n)
    site_idx = rng.choice(len(Cl), size=n, replace=len(Cl) < n)
    query = Cl[site_idx].copy()
    if args.jitter > 0:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2).fit(query)
        d = nn.kneighbors(query)[0][:, 1]
        med = float(np.median(d))
        query = query + rng.normal(0, args.jitter * med, size=query.shape)

    pca_path = ROOT / "outputs" / "t2" / "embryo" / "pca.joblib"
    pca = ExpressionPCA.load(pca_path) if pca_path.exists() else None
    z0 = pca.encode(Xl) if pca is not None else Xl
    z1 = pca.encode(Xr) if pca is not None else Xr
    prog = _progress_axis(z0, z1)
    s0 = np.zeros(len(Xl)) if prog is None else prog[0]
    s1 = np.ones(len(Xr)) if prog is None else prog[1]

    # Optional global mid-band filter on libraries
    if args.mode == "slice":
        keep = float(args.slice_keep)
        lo, hi = 0.5 - keep / 2, 0.5 + keep / 2
        # progress relative to w: keep near w
        m0 = np.abs(s0 - float(args.w)) <= keep / 2
        m1 = np.abs(s1 - float(args.w)) <= keep / 2
        if m0.sum() < 50:
            m0 = np.ones(len(s0), dtype=bool)
        if m1.sum() < 50:
            m1 = np.ones(len(s1), dtype=bool)
        Xl, Cl, s0 = Xl[m0], Cl[m0], s0[m0]
        Xr, Cr, s1 = Xr[m1], Cr[m1], s1[m1]

    k0 = min(int(args.knn), len(Cl))
    k1 = min(int(args.knn), len(Cr))
    _, i0 = cKDTree(Cl).query(query, k=k0)
    _, i1 = cKDTree(Cr).query(query, k=k1)
    i0 = np.atleast_2d(np.asarray(i0, dtype=np.int64)).reshape(n, k0)
    i1 = np.atleast_2d(np.asarray(i1, dtype=np.int64)).reshape(n, k1)
    w = float(args.w)
    take_right = rng.random(n) < np.clip(w, 0.0, 1.0)
    X = np.empty((n, Xl.shape[1]), dtype=np.float32)
    for t in range(n):
        if args.mode == "pool" or args.mode == "slice":
            # for slice still pick by progress among knn
            d0 = np.abs(s0[i0[t]] - w)
            d1 = np.abs(s1[i1[t]] - w)
            if args.mode == "pool" or True:
                if float(d0.min()) <= float(d1.min()):
                    X[t] = Xl[int(i0[t][int(np.argmin(d0))])]
                else:
                    X[t] = Xr[int(i1[t][int(np.argmin(d1))])]
        elif take_right[t]:
            loc = i1[t]
            X[t] = Xr[int(loc[int(np.argmin(np.abs(s1[loc] - w)))])]
        else:
            loc = i0[t]
            X[t] = Xl[int(loc[int(np.argmin(np.abs(s0[loc] - w)))])]

    xyz = np.asarray(scale_cloud(query, float(args.rms)), dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_t2(X, xyz, panel, args.out)
    print(
        f"wrote {args.out} n={n} w={w} knn={args.knn} mode={args.mode} "
        f"jitter={args.jitter} anchors={args.left}+{args.right} tps-site"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
