#!/usr/bin/env python
"""Composition quotas placed on TPS occupy sites (native couple, not donor-own xyz).

Fixes the complementary failure modes:
  - 42_comp_native: good NFS/mmd/de, bad ODS (donor own geometry)
  - 43_tps_site: good ODS, bad NFS/mmd (composition-blind local_pick)

Each cell keeps a real donor X from its allocated cluster; xyz is a TPS-cloud site
(optionally soft-matched within-cluster to the donor's framed position).

Leave-out:
  .venv/bin/python scripts/44_comp_tps_place.py --left E6.75 --right E8.0 --t 7.25 \\
    --w 0.4 --n 5000 --out outputs/t2/embryo/gate725_newpaths/lo_comp_tps.h5ad

Board:
  .venv/bin/python scripts/44_comp_tps_place.py --left E7.25 --right E8.0 --t 7.5 \\
    --w 0.333 --n 5000 --out outputs/t2/embryo/gate75_comptps/board.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import birth_clusters, labels_to_clusters  # noqa: E402
from t2.expression import CompositionT2  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.growth import interp_weight  # noqa: E402
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
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
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--w", type=float, default=None, help="Bernoulli right weight")
    ap.add_argument("--tps-w", type=float, default=None, help="TPS morph weight (default=w)")
    ap.add_argument(
        "--place",
        choices=["random", "nn", "otsoft"],
        default="nn",
        help="random=uniform sites; nn=site near donor framed xyz; otsoft=nn then mild pull",
    )
    ap.add_argument("--otsoft-alpha", type=float, default=0.35)
    ap.add_argument(
        "--progress-keep",
        type=float,
        default=None,
        help="if set, keep only donors with |progress−w| within keep/2 (mid-band)",
    )
    ap.add_argument("--tps-cache", type=Path, default=None, help="load/save Cl,Cr float64 npz")
    ap.add_argument("--unique", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
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
    Cl_raw, Cr_raw = spatial_xyz(left), spatial_xyz(right)
    tps_n = int((cfg.get("shape") or {}).get("tps_n") or 800)
    if args.tps_cache is not None and args.tps_cache.exists():
        z = np.load(args.tps_cache)
        Cl = np.asarray(z["Cl"], dtype=np.float64)
        Cr = np.asarray(z["Cr"], dtype=np.float64)
    else:
        Cl = ot_interpolate_clouds(Cl_raw, Cr_raw, tps_w, float(args.rms), n_pair=tps_n, rng=rng)
        Cr = np.asarray(transform_cloud(Cr_raw, float(args.rms), mode="isotropic"), dtype=np.float32)
        Cr = match_axis_flips(Cl, Cr, rng, n_pair=min(6000, len(Cl), len(Cr)))
        Cl = np.asarray(Cl, dtype=np.float64)
        Cr = np.asarray(Cr, dtype=np.float64)
        if args.tps_cache is not None:
            args.tps_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.tps_cache, Cl=Cl, Cr=Cr)
    sites = Cl.copy()
    site_tree = cKDTree(sites)

    cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], "embryo"))
    cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], "embryo"))
    birth = birth_clusters("embryo")

    s0 = np.zeros(len(Xl), dtype=np.float64)
    s1 = np.ones(len(Xr), dtype=np.float64)
    if args.progress_keep is not None:
        pca_path = ROOT / "outputs" / "t2" / "embryo" / "pca.joblib"
        pca = ExpressionPCA.load(pca_path) if pca_path.exists() else None
        z0 = pca.encode(Xl) if pca is not None else Xl
        z1 = pca.encode(Xr) if pca is not None else Xr
        prog = _progress_axis(z0, z1)
        if prog is not None:
            s0, s1 = np.asarray(prog[0], dtype=np.float64), np.asarray(prog[1], dtype=np.float64)

    comp = CompositionT2("embryo", {args.left: t_l, args.right: t_r})
    comp.fit(
        {args.left: left.obs["celltype"], args.right: right.obs["celltype"]},
        {args.left: t_l, args.right: t_r},
    )
    alloc = comp.allocate(t, int(args.n), rng, allowed_times=[t_l, t_r])

    rows_x, rows_xyz = [], []
    used_l, used_r = set(), set()
    used_sites: set[int] = set()

    def draw_idx(side_idx, used, count, scores):
        out = []
        keep = args.progress_keep
        for _ in range(count):
            pool = [i for i in side_idx if (not args.unique or i not in used)]
            if not pool:
                pool = list(side_idx)
            if keep is not None and len(pool) > 1:
                half = float(keep) / 2.0
                band = [i for i in pool if abs(float(scores[i]) - w) <= half]
                if len(band) >= max(1, len(pool) // 20):
                    pool = band
                else:
                    # fall back to closest-to-w ranking
                    pool = sorted(pool, key=lambda i: abs(float(scores[i]) - w))[
                        : max(1, int(round(float(keep) * len(pool))))
                    ]
            i = int(rng.choice(pool))
            used.add(i)
            out.append(i)
        return out

    def take_site(near_xyz: np.ndarray | None):
        if args.place == "random" or near_xyz is None:
            pool = [i for i in range(len(sites)) if i not in used_sites]
            if not pool:
                pool = list(range(len(sites)))
            i = int(rng.choice(pool))
            used_sites.add(i)
            return sites[i].copy()
        _, nn = site_tree.query(near_xyz, k=min(32, len(sites)))
        nn = np.atleast_1d(np.asarray(nn, dtype=np.int64))
        for j in nn:
            j = int(j)
            if j not in used_sites:
                used_sites.add(j)
                xyz = sites[j].copy()
                if args.place == "otsoft":
                    a = float(np.clip(args.otsoft_alpha, 0.0, 1.0))
                    xyz = (1.0 - a) * xyz + a * near_xyz
                return xyz
        # fallback
        j = int(nn[0])
        used_sites.add(j)
        return sites[j].copy()

    for cluster, count in alloc.items():
        if count <= 0:
            continue
        i0 = np.flatnonzero(cl_l == cluster)
        i1 = np.flatnonzero(cl_r == cluster)
        if cluster in birth and len(i1):
            n_right = count
        elif len(i0) == 0 and len(i1):
            n_right = count
        elif len(i1) == 0 and len(i0):
            n_right = 0
        else:
            n_right = int(rng.binomial(count, np.clip(w, 0.0, 1.0)))
        n_left = count - n_right

        for side, idxs, Xs, Cs, used, scores, n_take in (
            ("L", i0, Xl, Cl, used_l, s0, n_left),
            ("R", i1, Xr, Cr, used_r, s1, n_right),
        ):
            if n_take <= 0:
                continue
            if len(idxs) == 0:
                # swap side
                if side == "L" and len(i1):
                    idxs, Xs, Cs, used, scores = i1, Xr, Cr, used_r, s1
                elif side == "R" and len(i0):
                    idxs, Xs, Cs, used, scores = i0, Xl, Cl, used_l, s0
                else:
                    continue
            di = draw_idx(idxs.tolist(), used, n_take, scores)
            for i in di:
                donor_xyz = Cs[i]
                xyz = take_site(donor_xyz)
                rows_x.append(Xs[i])
                rows_xyz.append(xyz)

    if not rows_x:
        raise SystemExit("no cells allocated")
    X = np.stack(rows_x, axis=0).astype(np.float32)
    xyz = np.stack(rows_xyz, axis=0).astype(np.float32)
    if len(X) > args.n:
        take = rng.choice(len(X), size=args.n, replace=False)
        X, xyz = X[take], xyz[take]
    elif len(X) < args.n:
        pad = args.n - len(X)
        take = rng.choice(len(X), size=pad, replace=True)
        X = np.concatenate([X, X[take]], axis=0)
        xyz = np.concatenate([xyz, xyz[take]], axis=0)
    xyz = np.asarray(scale_cloud(xyz, float(args.rms)), dtype=np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_t2(X, xyz, panel, args.out)
    pk = "none" if args.progress_keep is None else f"{float(args.progress_keep):g}"
    print(
        f"wrote {args.out} n={len(X)} w={w:.3f} tps_w={tps_w:.3f} place={args.place} "
        f"prog_keep={pk} unique={args.unique} anchors={args.left}+{args.right}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
