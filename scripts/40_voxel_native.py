#!/usr/bin/env python
"""Native voxel birth: sample xyz from interpolated occupancy, pick a real nearby cell's X.

Unlike ``--ot-xyz occ`` this does NOT Hungarian-assign onto the density cloud after
OT-place. Each submitted cell is born at its sampled voxel with a true coexpression
vector from a spatial neighbor. Do not freeze onto 09a.

Leave-out:
  .venv/bin/python scripts/40_voxel_native.py --left E6.75 --right E8.0 --w 0.4 \\
    --n 5000 --out outputs/t2/embryo/gate725_voxel/lo_pick_k8.h5ad

Board (only after gate):
  .venv/bin/python scripts/40_voxel_native.py --left E7.25 --right E8.0 --w 0.333 \\
    --n 5000 --out outputs/t2/embryo/gate75_voxel/board_pick_k8.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense, write_t2  # noqa: E402
from t2.ot import _progress_axis  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402
from t2.shape import match_axis_flips, occupancy_interpolate_cloud  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--bins", type=int, default=48)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--mode", choices=["pick", "pool"], default="pick")
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    spec = target_spec("embryo", 7.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("embryo", cfg, panel)
    left, right = stages[args.left], stages[args.right]

    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), args.rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), args.rms), dtype=np.float64)
    Cr = np.asarray(match_axis_flips(Cl, Cr, rng, n_pair=min(6000, len(Cl), len(Cr))), dtype=np.float64)

    pca_path = ROOT / "outputs" / "t2" / "embryo" / "pca.joblib"
    pca = ExpressionPCA.load(pca_path) if pca_path.exists() else None
    if pca is not None:
        z0, z1 = pca.encode(Xl), pca.encode(Xr)
    else:
        z0, z1 = Xl, Xr
    prog = _progress_axis(z0, z1)
    if prog is None:
        s0 = np.zeros(len(Xl))
        s1 = np.ones(len(Xr))
    else:
        s0, s1, _ = prog

    query = occupancy_interpolate_cloud(Cl, Cr, float(args.w), int(args.n), rng, n_bins=int(args.bins))
    query = np.asarray(query, dtype=np.float64)

    k0 = min(int(args.knn), len(Cl))
    k1 = min(int(args.knn), len(Cr))
    _, i0 = cKDTree(Cl).query(query, k=k0)
    _, i1 = cKDTree(Cr).query(query, k=k1)
    i0 = np.asarray(i0, dtype=np.int64).reshape(len(query), k0)
    i1 = np.asarray(i1, dtype=np.int64).reshape(len(query), k1)

    used0, used1 = set(), set()
    X = np.empty((len(query), Xl.shape[1]), dtype=np.float32)
    xyz = query.astype(np.float32)
    w = float(args.w)
    take_right = rng.random(len(query)) < np.clip(w, 0.0, 1.0)

    def choose(side_idx, scores, Xside, used):
        order = np.argsort(np.abs(scores[side_idx] - w))
        for j in order:
            idx = int(side_idx[j])
            if not args.unique or idx not in used:
                used.add(idx)
                return Xside[idx]
        return Xside[int(side_idx[int(order[0])])]

    for t in range(len(query)):
        if args.mode == "pool":
            d0 = np.abs(s0[i0[t]] - w)
            d1 = np.abs(s1[i1[t]] - w)
            if float(d0.min()) <= float(d1.min()):
                X[t] = choose(i0[t], s0, Xl, used0)
            else:
                X[t] = choose(i1[t], s1, Xr, used1)
        elif take_right[t]:
            X[t] = choose(i1[t], s1, Xr, used1)
        else:
            X[t] = choose(i0[t], s0, Xl, used0)

    xyz = np.asarray(scale_cloud(xyz, args.rms), dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_t2(X, xyz, panel, args.out)
    print(
        f"wrote {args.out} n={len(X)} w={w} bins={args.bins} knn={args.knn} "
        f"mode={args.mode} unique={args.unique} rms={args.rms} "
        f"anchors={args.left}+{args.right} native-voxel"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
