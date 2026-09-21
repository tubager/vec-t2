#!/usr/bin/env python
"""Composition-correct native birth: real cells keep own (X, xyz), framed to target RMS.

Unlike freeze-to-09a, each submitted cell is an intact donor cell. Composition follows
π(t*). Left cells are TPS-morphed toward the right frame (same as OT-place left cloud);
right/birth cells are scaled into the same RMS.

Leave-out:
  .venv/bin/python scripts/42_comp_native_birth.py --left E6.75 --right E8.0 --t 7.25 \\
    --n 5000 --out outputs/t2/embryo/gate725_compnat/lo.h5ad

Board:
  .venv/bin/python scripts/42_comp_native_birth.py --left E7.25 --right E8.0 --t 7.5 \\
    --n 5000 --out outputs/t2/embryo/gate75_compnat/board.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import birth_clusters, labels_to_clusters  # noqa: E402
from t2.expression import CompositionT2  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.growth import interp_weight  # noqa: E402
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense, write_t2  # noqa: E402
from t2.paths import load_config  # noqa: E402
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
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--w", type=float, default=None, help="override Bernoulli right weight")
    ap.add_argument("--tps-w", type=float, default=None)
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    spec = target_spec("embryo", float(args.t) if abs(float(args.t) - 7.5) < 1e-9 else 7.5, cfg)
    # For leave-out t=7.25, force proxy-like panel/stages
    panel = list(load_panel_for(target_spec("embryo", 7.5, cfg), cfg))
    stages = load_stages("embryo", cfg, panel)
    times = stage_times("embryo", cfg)
    left, right = stages[args.left], stages[args.right]
    t_l, t_r = float(times[args.left]), float(times[args.right])
    t = float(args.t)
    w = float(args.w) if args.w is not None else interp_weight(t_l, t_r, t)
    tps_w = float(args.tps_w) if args.tps_w is not None else w

    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Cl_raw = spatial_xyz(left)
    Cr_raw = spatial_xyz(right)
    # Frame: TPS-morph left toward right at tps_w; scale right to rms
    tps_n = int((cfg.get("shape") or {}).get("tps_n") or 800)
    Cl = ot_interpolate_clouds(Cl_raw, Cr_raw, tps_w, float(args.rms), n_pair=tps_n, rng=rng)
    Cr = np.asarray(transform_cloud(Cr_raw, float(args.rms), mode="isotropic"), dtype=np.float32)
    Cr = match_axis_flips(Cl, Cr, rng, n_pair=min(6000, len(Cl), len(Cr)))

    cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], "embryo"))
    cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], "embryo"))
    birth = birth_clusters("embryo")

    comp = CompositionT2("embryo", {args.left: t_l, args.right: t_r})
    comp.fit(
        {args.left: left.obs["celltype"], args.right: right.obs["celltype"]},
        {args.left: t_l, args.right: t_r},
    )
    alloc = comp.allocate(t, int(args.n), rng, allowed_times=[t_l, t_r])

    rows_x, rows_xyz, rows_cl = [], [], []
    used_l, used_r = set(), set()

    def draw(side_idx, Xside, Cside, used, count):
        if len(side_idx) == 0:
            return None
        out_i = []
        for _ in range(count):
            pool = [i for i in side_idx if (not args.unique or i not in used)]
            if not pool:
                pool = list(side_idx)
            i = int(rng.choice(pool))
            used.add(i)
            out_i.append(i)
        return Xside[out_i], Cside[out_i]

    for cluster, count in alloc.items():
        if count <= 0:
            continue
        i0 = np.flatnonzero(cl_l == cluster)
        i1 = np.flatnonzero(cl_r == cluster)
        # birth clusters: prefer right
        if cluster in birth and len(i1):
            n_right = count
        elif len(i0) == 0 and len(i1):
            n_right = count
        elif len(i1) == 0 and len(i0):
            n_right = 0
        else:
            n_right = int(rng.binomial(count, np.clip(w, 0.0, 1.0)))
        n_left = count - n_right
        if n_left:
            got = draw(i0, Xl, Cl, used_l, n_left)
            if got is None:
                got = draw(i1, Xr, Cr, used_r, n_left)
            if got is not None:
                x, z = got
                rows_x.append(x)
                rows_xyz.append(z)
                rows_cl.append(np.full(len(x), cluster))
        if n_right:
            got = draw(i1, Xr, Cr, used_r, n_right)
            if got is None:
                got = draw(i0, Xl, Cl, used_l, n_right)
            if got is not None:
                x, z = got
                rows_x.append(x)
                rows_xyz.append(z)
                rows_cl.append(np.full(len(x), cluster))

    X = np.concatenate(rows_x, axis=0).astype(np.float32)
    xyz = np.concatenate(rows_xyz, axis=0).astype(np.float32)
    if len(X) != int(args.n):
        # pad/truncate
        if len(X) > args.n:
            take = rng.choice(len(X), size=args.n, replace=False)
            X, xyz = X[take], xyz[take]
        else:
            pad = args.n - len(X)
            take = rng.choice(len(X), size=pad, replace=True)
            X = np.concatenate([X, X[take]], axis=0)
            xyz = np.concatenate([xyz, xyz[take]], axis=0)
    xyz = np.asarray(scale_cloud(xyz, float(args.rms)), dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_t2(X, xyz, panel, args.out)
    print(
        f"wrote {args.out} n={len(X)} t={t} w={w:.3f} tps_w={tps_w:.3f} "
        f"unique={args.unique} anchors={args.left}+{args.right} comp-native"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
