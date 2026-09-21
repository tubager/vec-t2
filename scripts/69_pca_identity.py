#!/usr/bin/env python
"""PCA-256 identity codec: within-cluster OT interpolant at t*.

Linear PCA-256 on log1p reconstructs holdout cells at mae ~0.006 with parent
zeros. Neural AE could not. Birth mid-time cells as convex combinations in that
PCA, not clock-mean pull on 194 (201/204 family) and not intact L/R inject (202).

Leave-out (dual-gate before any board file):
  .venv/bin/python scripts/69_pca_identity.py --t 7.25 --mode local \\
    --pred outputs/t2/embryo/gate725_repair/k256.h5ad \\
    --native outputs/t2/embryo/gate725_pcaot/local_k256.h5ad \\
    --freeze-onto outputs/t2/embryo/gate725/slice_spatial_k0.4_b6.h5ad \\
    --out outputs/t2/embryo/gate725_pcaot/local_freeze_k04b6.h5ad

Board (only after scripts/54_dual_gate.py PASS):
  .venv/bin/python scripts/69_pca_identity.py --t 7.5 --mode local \\
    --pred outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --out outputs/t2/queue/2026-09-20/206_embryo_pca384_localinterp_on194_n5000.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense, write_t2  # noqa: E402
from t2.nbhd_pick import freeze_x_onto  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def _ot_idx(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = a[:, None, :].astype(np.float32) - b[None, :, :].astype(np.float32)
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    ri, ci = linear_sum_assignment(cost)
    return ri.astype(np.int64), ci.astype(np.int64)


def _pair_cluster(z0, z1, rng, cap: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(z0), len(z1), int(cap))
    if n < 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    i = rng.choice(len(z0), size=n, replace=False)
    j = rng.choice(len(z1), size=n, replace=False)
    ri, ci = _ot_idx(z0[i], z1[j])
    return i[ri], j[ci]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--native", type=Path, default=None)
    ap.add_argument("--pred", type=Path, default=None, help="xyz skeleton (local birth sites, or OT freeze target)")
    ap.add_argument("--freeze-onto", type=Path, default=None, help="optional second xyz for gate-2 freeze")
    ap.add_argument("--mode", choices=["ot", "local"], default="ot")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--cap", type=int, default=800, help="max Hungarian size per cluster batch")
    ap.add_argument("--knn-local", type=int, default=1, help="spatial neighbors per side in local mode")
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--zero", choices=["parent", "quantile", "none"], default="parent")
    ap.add_argument("--replace-p", type=float, default=0.0, help="replace top residual fraction of --pred with interpolants")
    ap.add_argument("--blend", type=float, default=0.0, help="X=(1-b)*pred + b*hung-interpolant; ignored if replace-p>0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-recon-gate", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    spec = target_spec("embryo", float(args.t), cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("embryo", cfg, panel)
    left, right = stages[spec.left], stages[spec.right]
    t0, t1 = float(spec.t_left), float(spec.t_right)
    w = float(np.clip((float(args.t) - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))

    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Cl = spatial_xyz(left).astype(np.float32)
    Cr = spatial_xyz(right).astype(np.float32)
    cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], "embryo"))
    cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], "embryo"))
    usable = sorted(shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], "embryo"))
    keep_l = np.array([str(c) in set(usable) for c in cl_l])
    keep_r = np.array([str(c) in set(usable) for c in cl_r])
    Xl, Xr = Xl[keep_l], Xr[keep_r]
    Cl, Cr = Cl[keep_l], Cr[keep_r]
    cl_l, cl_r = cl_l[keep_l], cl_r[keep_r]
    Yl = np.log1p(np.maximum(Xl, 0.0))
    Yr = np.log1p(np.maximum(Xr, 0.0))
    Ycat = np.concatenate([Yl, Yr], axis=0)
    Xcat = np.concatenate([Xl, Xr], axis=0)

    rng = np.random.default_rng(args.seed)
    n_all = len(Ycat)
    tr = rng.choice(n_all, int(0.85 * n_all), replace=False)
    ho = np.setdiff1d(np.arange(n_all), tr)
    gdim = int(Ycat.shape[1])
    k_try = [int(args.k)]
    for k in (256, 384, gdim):
        if k not in k_try and k <= gdim:
            k_try.append(k)
    pca = None
    mae = frac0_h = frac0_t = 0.0
    ok = False
    for k in k_try:
        pca = PCA(n_components=min(k, gdim, len(tr) - 1), random_state=0).fit(Ycat[tr])
        Yh = pca.inverse_transform(pca.transform(Ycat[ho]))
        Xh = np.clip(np.expm1(Yh), 0.0, None)
        Xz = Xh.copy()
        Xz[Xcat[ho] <= 0] = 0.0
        mae = float(np.abs(Xz - Xcat[ho]).mean())
        frac0_h = float((Xz <= 0).mean())
        frac0_t = float((Xcat[ho] <= 0).mean())
        ok = mae <= 0.03 and abs(frac0_h - frac0_t) <= 0.05
        print(
            f"RECON GATE PCA-{pca.n_components_} keep-zeros mae={mae:.4f} "
            f"frac0={frac0_h:.3f} vs {frac0_t:.3f} => {'PASS' if ok else 'FAIL'}"
        )
        if ok:
            break
    if not ok and not args.skip_recon_gate:
        print("RECON GATE FAIL — do not submit")
        return 2

    Zl = pca.transform(Yl)
    Zr = pca.transform(Yr)

    def _apply_zero(Xm, i0, i1):
        if args.zero == "parent":
            Xp = Xl[i0] if w <= 0.5 else Xr[i1]
            Xm[Xp <= 0] = 0.0
        elif args.zero == "quantile":
            mix = (1.0 - w) * (Xl <= 0).mean(0) + w * (Xr <= 0).mean(0)
            for j, p in enumerate(mix):
                if p <= 0:
                    continue
                thr = np.quantile(Xm[:, j], float(p))
                Xm[Xm[:, j] <= thr, j] = 0.0
        return Xm

    if args.mode == "local":
        if args.pred is None:
            raise SystemExit("--mode local requires --pred sites")
        live = ad.read_h5ad(args.pred)
        C0 = np.array(np.asarray(live.obsm["spatial_3D"], dtype=np.float32), copy=True)
        Cq = np.asarray(scale_cloud(C0, float(args.rms)), dtype=np.float64)
        Cl_s = np.asarray(scale_cloud(Cl, float(args.rms)), dtype=np.float64)
        Cr_s = np.asarray(scale_cloud(Cr, float(args.rms)), dtype=np.float64)
        Ccat = np.concatenate([Cl_s, Cr_s], axis=0)
        cl_cat = np.concatenate([cl_l, cl_r])
        _, nn = cKDTree(Ccat).query(Cq, k=min(15, len(Ccat)))
        nn = np.atleast_2d(np.asarray(nn, dtype=np.int64))
        names = []
        for row in nn:
            labs = [str(cl_cat[j]) for j in row]
            vals, cnt = np.unique(labs, return_counts=True)
            names.append(str(vals[int(np.argmax(cnt))]))
        names = np.asarray(names)
        idx_l = {name: np.flatnonzero(cl_l == name) for name in usable}
        idx_r = {name: np.flatnonzero(cl_r == name) for name in usable}
        trees_l = {
            name: cKDTree(Cl_s[idx_l[name]]) if len(idx_l[name]) >= 8 else None for name in usable
        }
        trees_r = {
            name: cKDTree(Cr_s[idx_r[name]]) if len(idx_r[name]) >= 8 else None for name in usable
        }
        tree_l = cKDTree(Cl_s)
        tree_r = cKDTree(Cr_s)
        kloc = max(1, int(args.knn_local))
        i0 = np.empty((len(Cq), kloc), dtype=np.int64)
        i1 = np.empty((len(Cq), kloc), dtype=np.int64)
        for i, name in enumerate(names):
            tl, tr_ = trees_l.get(name), trees_r.get(name)
            if tl is not None:
                _, j = tl.query(Cq[i], k=min(kloc, tl.n))
                j = np.atleast_1d(np.asarray(j, dtype=np.int64))
                take = idx_l[name][j]
                if len(take) < kloc:
                    take = np.pad(take, (0, kloc - len(take)), mode="edge")
                i0[i] = take[:kloc]
            else:
                _, j = tree_l.query(Cq[i], k=kloc)
                i0[i] = np.atleast_1d(np.asarray(j, dtype=np.int64))
            if tr_ is not None:
                _, j = tr_.query(Cq[i], k=min(kloc, tr_.n))
                j = np.atleast_1d(np.asarray(j, dtype=np.int64))
                take = idx_r[name][j]
                if len(take) < kloc:
                    take = np.pad(take, (0, kloc - len(take)), mode="edge")
                i1[i] = take[:kloc]
            else:
                _, j = tree_r.query(Cq[i], k=kloc)
                i1[i] = np.atleast_1d(np.asarray(j, dtype=np.int64))
        Zm = (1.0 - w) * Zl[i0].mean(1) + w * Zr[i1].mean(1)
        Xm = np.clip(np.expm1(pca.inverse_transform(Zm)), 0.0, None).astype(np.float32)
        Xm = _apply_zero(Xm, i0[:, 0], i1[:, 0])
        Cm = C0
        native_path = args.native or args.out
        out_n = live.copy()
        if list(live.var_names) == panel:
            out_n.X = Xm
        else:
            live_x = np.asarray(to_dense(live.X), dtype=np.float32)
            idxg = {g: i for i, g in enumerate(panel)}
            for j, g in enumerate(live.var_names):
                live_x[:, j] = Xm[:, idxg[g]]
            out_n.X = live_x
        out_n.obsm["spatial_3D"] = C0
        Path(native_path).parent.mkdir(parents=True, exist_ok=True)
        out_n.write_h5ad(native_path, compression="gzip")
        print(
            f"local t={args.t:g} {spec.left}+{spec.right} w={w:.3f} knn={kloc} zero={args.zero}  "
            f"n={len(Xm)} mean={float(Xm.mean()):.4f} frac0={float((Xm <= 0).mean()):.3f}  {native_path}"
        )
        freeze_src = args.freeze_onto
        if freeze_src is None:
            if str(args.out) != str(native_path):
                out_n.write_h5ad(args.out, compression="gzip")
            return 0
        live2 = ad.read_h5ad(freeze_src)
        frozen = freeze_x_onto(Xm, Cm, live2)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_h5ad(args.out, compression="gzip")
        chk = ad.read_h5ad(args.out)
        assert np.array_equal(
            np.asarray(chk.obsm["spatial_3D"], dtype=np.float32),
            np.asarray(live2.obsm["spatial_3D"], dtype=np.float32),
        )
        print(f"freeze xyz from {freeze_src}  wrote {args.out}")
        return 0

    i0_all: list[np.ndarray] = []
    i1_all: list[np.ndarray] = []
    for name in usable:
        m0 = np.flatnonzero(cl_l == name)
        m1 = np.flatnonzero(cl_r == name)
        if len(m0) < 8 or len(m1) < 8:
            continue
        a, b = _pair_cluster(Zl[m0], Zr[m1], rng, args.cap)
        if len(a) == 0:
            continue
        i0_all.append(m0[a])
        i1_all.append(m1[b])
    if not i0_all:
        raise SystemExit("no within-cluster OT pairs")
    i0 = np.concatenate(i0_all)
    i1 = np.concatenate(i1_all)
    if len(i0) < int(args.n):
        extra0: list[np.ndarray] = []
        extra1: list[np.ndarray] = []
        fat = [name for name in usable if (cl_l == name).sum() >= 8 and (cl_r == name).sum() >= 8]
        tries = 0
        while fat and len(i0) + sum(len(x) for x in extra0) < int(args.n) and tries < 64:
            tries += 1
            name = fat[int(rng.integers(len(fat)))]
            m0 = np.flatnonzero(cl_l == name)
            m1 = np.flatnonzero(cl_r == name)
            need = int(args.n) - len(i0) - sum(len(x) for x in extra0)
            a, b = _pair_cluster(Zl[m0], Zr[m1], rng, min(args.cap, need))
            if len(a) == 0:
                continue
            extra0.append(m0[a])
            extra1.append(m1[b])
        if extra0:
            i0 = np.concatenate([i0, *extra0])
            i1 = np.concatenate([i1, *extra1])
        if len(i0) < int(args.n):
            pad = rng.integers(len(i0), size=int(args.n) - len(i0))
            i0 = np.concatenate([i0, i0[pad]])
            i1 = np.concatenate([i1, i1[pad]])
    if len(i0) > int(args.n):
        take = rng.choice(len(i0), size=int(args.n), replace=False)
        i0, i1 = i0[take], i1[take]

    Zm = (1.0 - w) * Zl[i0] + w * Zr[i1]
    Xm = np.clip(np.expm1(pca.inverse_transform(Zm)), 0.0, None).astype(np.float32)
    Xm = _apply_zero(Xm, i0, i1)
    Cm = scale_cloud((1.0 - w) * Cl[i0] + w * Cr[i1], float(args.rms))

    native_path = args.native or (args.out if args.pred is None else args.out.with_name(args.out.stem + "_native.h5ad"))
    live = ad.read_h5ad(args.pred) if args.pred is not None else None
    if live is not None and (float(args.replace_p) > 0 or float(args.blend) > 0):
        Xp = _align(to_dense(live.X), live.var_names, panel)
        C0 = np.array(np.asarray(live.obsm["spatial_3D"], dtype=np.float32), copy=True)
        if len(Xp) != len(Xm):
            raise SystemExit(f"n mismatch pred {len(Xp)} interpolant {len(Xm)}")
        a = Xp.astype(np.float32)
        b = Xm.astype(np.float32)
        cost = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
        ri, ci = linear_sum_assignment(cost)
        order = np.empty(len(a), dtype=np.int64)
        order[ri] = ci
        Xh = Xm[order]
        if float(args.replace_p) > 0:
            p = float(np.clip(args.replace_p, 0.0, 1.0))
            n_rep = int(round(p * len(Xp)))
            resid = np.abs(Xp - Xh).mean(1)
            take = np.argsort(-resid)[:n_rep]
            Xn = Xp.copy()
            Xn[take] = Xh[take]
            tag = f"replace-p={p:g} n={n_rep}"
        else:
            bgt = float(np.clip(args.blend, 0.0, 1.0))
            Xn = (1.0 - bgt) * Xp + bgt * Xh
            tag = f"blend={bgt:g}"
        Xm, Cm = Xn.astype(np.float32), C0
        out_n = live.copy()
        if list(live.var_names) == panel:
            out_n.X = Xm
        else:
            live_x = np.asarray(to_dense(live.X), dtype=np.float32)
            idxg = {g: i for i, g in enumerate(panel)}
            for j, g in enumerate(live.var_names):
                live_x[:, j] = Xm[:, idxg[g]]
            out_n.X = live_x
        out_n.obsm["spatial_3D"] = C0
        Path(native_path).parent.mkdir(parents=True, exist_ok=True)
        out_n.write_h5ad(native_path, compression="gzip")
        print(
            f"{tag} t={args.t:g} {spec.left}+{spec.right} w={w:.3f}  "
            f"mae_vs_pred={float(np.abs(Xm - Xp).mean()):.4f} frac0={float((Xm <= 0).mean()):.3f}  {native_path}"
        )
        freeze_src = args.freeze_onto
        if freeze_src is None:
            if str(args.out) != str(native_path):
                out_n.write_h5ad(args.out, compression="gzip")
            return 0
        live2 = ad.read_h5ad(freeze_src)
        frozen = freeze_x_onto(Xm, Cm, live2)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_h5ad(args.out, compression="gzip")
        chk = ad.read_h5ad(args.out)
        assert np.array_equal(
            np.asarray(chk.obsm["spatial_3D"], dtype=np.float32),
            np.asarray(live2.obsm["spatial_3D"], dtype=np.float32),
        )
        print(f"freeze xyz from {freeze_src}  wrote {args.out}")
        return 0

    write_t2(Xm, Cm, panel, native_path)
    print(
        f"native t={args.t:g} {spec.left}+{spec.right} w={w:.3f} zero={args.zero}  "
        f"n={len(Xm)} mean={float(Xm.mean()):.4f} frac0={float((Xm <= 0).mean()):.3f}  {native_path}"
    )

    if args.pred is None:
        return 0
    live = ad.read_h5ad(args.pred)
    if live.n_obs != len(Xm):
        raise SystemExit(f"n mismatch pred {live.n_obs} interpolant {len(Xm)}")
    frozen = freeze_x_onto(Xm, Cm, live)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(
        np.asarray(chk.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(live.obsm["spatial_3D"], dtype=np.float32),
    )
    print(f"freeze xyz from {args.pred}  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
