#!/usr/bin/env python
"""Occupancy-native spatial-token birth: real-cell X at sites sampled from a prior occupancy.

Not freeze-09a (voxel interiors, not bit-exact xyz). Not TPS morph (board nocc prior).
X is a spatial-neighborhood token: pick / pool / local partial-mean / trained nbhd MLP.

Leave-out:
  .venv/bin/python scripts/46_token_birth.py --left E6.75 --right E8.0 --t 7.25 --w 0.4 \\
    --prior outputs/t2/embryo/gate725/mix.h5ad --mode pm --out lo.h5ad

Board:
  .venv/bin/python scripts/46_token_birth.py --left E7.25 --right E8.0 --t 7.5 --w 0.333 \\
    --prior outputs/t2/submit/T2_embryo_val_interp.h5ad --mode pm --out board.h5ad
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
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense, write_t2  # noqa: E402
from t2.occ_sample import sample_occupancy_sites  # noqa: E402
from t2.ot import _local_pick_rows, _progress_axis  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def _xyz(path: Path) -> np.ndarray:
    a = ad.read_h5ad(path)
    return np.asarray(a.obsm["spatial_3D"], dtype=np.float64)


def _local_pool_rows(q, xyz0, X0, xyz1, X1, w, knn):
    n = len(q)
    k0 = min(int(knn), len(xyz0))
    k1 = min(int(knn), len(xyz1))
    _, i0 = cKDTree(xyz0).query(q, k=k0)
    _, i1 = cKDTree(xyz1).query(q, k=k1)
    i0 = np.atleast_2d(np.asarray(i0, dtype=np.int64).reshape(n, k0))
    i1 = np.atleast_2d(np.asarray(i1, dtype=np.int64).reshape(n, k1))
    m0 = X0[i0].mean(axis=1)
    m1 = X1[i1].mean(axis=1)
    return ((1.0 - w) * m0 + w * m1).astype(np.float32)


def _nbhd_rows(q, xyz0, X0, xyz1, X1, w, knn, nbhd_dir, device, project: bool):
    import torch

    from t2.flow import torch_pca_decode
    from t2.nbhd_mid import load_nbhd

    pca = ExpressionPCA.load(Path(nbhd_dir) / "pca.joblib")
    model, meta = load_nbhd(Path(nbhd_dir) / "ckpt" / "nbhd.pt", device=device)
    k = int(meta.get("knn", knn))
    zl, zr = pca.encode(X0), pca.encode(X1)
    k0 = min(k, len(xyz0))
    k1 = min(k, len(xyz1))
    _, i0 = cKDTree(xyz0).query(q, k=k0)
    _, i1 = cKDTree(xyz1).query(q, k=k1)
    i0 = np.atleast_2d(np.asarray(i0, dtype=np.int64).reshape(len(q), k0))
    i1 = np.atleast_2d(np.asarray(i1, dtype=np.int64).reshape(len(q), k1))
    z_l = zl[i0].mean(axis=1)
    z_r = zr[i1].mean(axis=1)
    with torch.no_grad():
        pred_z = model(
            torch.as_tensor(z_l, device=device, dtype=torch.float32),
            torch.as_tensor(z_r, device=device, dtype=torch.float32),
            torch.full((len(q), 1), float(w), device=device, dtype=torch.float32),
        )
        comp = torch.as_tensor(pca.model.components_, device=device, dtype=torch.float32)
        mean = torch.as_tensor(pca.model.mean_, device=device, dtype=torch.float32)
        X = torch_pca_decode(pred_z, comp, mean).cpu().numpy().astype(np.float32)
    if project:
        from t2.joint_flow import project_x_to_pool

        pool = np.concatenate([X0, X1], axis=0)
        if len(pool) > 20000:
            take = np.random.default_rng(0).choice(len(pool), size=20000, replace=False)
            pool = pool[take]
        X = project_x_to_pool(X, pool)
    return np.clip(X, 0.0, None).astype(np.float32)


def _snap_rows(q, xyz0, X0, xyz1, X1, w, knn, nbhd_dir, device):
    """Predict mid-z from neighbors, then copy the single real neighbor closest in z.

    Convex mixing of L+R cells is chimeric (simplex oracle: mmd 0.13 / var 0.11).
    Discrete snap keeps a real coexpression vector.
    """
    import torch

    from t2.nbhd_mid import load_nbhd

    pca = ExpressionPCA.load(Path(nbhd_dir) / "pca.joblib")
    model, meta = load_nbhd(Path(nbhd_dir) / "ckpt" / "nbhd.pt", device=device)
    k = int(meta.get("knn", knn))
    z0, z1 = pca.encode(X0), pca.encode(X1)
    k0 = min(k, len(xyz0))
    k1 = min(k, len(xyz1))
    _, i0 = cKDTree(xyz0).query(q, k=k0)
    _, i1 = cKDTree(xyz1).query(q, k=k1)
    i0 = np.asarray(i0, dtype=np.int64).reshape(len(q), k0)
    i1 = np.asarray(i1, dtype=np.int64).reshape(len(q), k1)
    z_l = z0[i0].mean(axis=1)
    z_r = z1[i1].mean(axis=1)
    with torch.no_grad():
        pred_z = (
            model(
                torch.as_tensor(z_l, device=device, dtype=torch.float32),
                torch.as_tensor(z_r, device=device, dtype=torch.float32),
                torch.full((len(q), 1), float(w), device=device, dtype=torch.float32),
            )
            .cpu()
            .numpy()
        )
    z_nb = np.concatenate([z0[i0], z1[i1]], axis=1)
    idx_nb = np.concatenate([i0, i1], axis=1)
    side = np.concatenate(
        [np.zeros((len(q), k0), dtype=np.int8), np.ones((len(q), k1), dtype=np.int8)],
        axis=1,
    )
    d = ((z_nb - pred_z[:, None, :]) ** 2).sum(axis=2)
    j = np.argmin(d, axis=1)
    rows = np.arange(len(q))
    take = idx_nb[rows, j]
    from_right = side[rows, j] == 1
    X = np.empty((len(q), X0.shape[1]), dtype=np.float32)
    X[~from_right] = X0[take[~from_right]]
    X[from_right] = X1[take[from_right]]
    return X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--prior", type=Path, required=True, help="occupancy prior cloud")
    ap.add_argument("--mode", choices=["pick", "pool", "pm", "nbhd", "simplex", "snap", "resid"], default="pm")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--k-genes", type=int, default=16)
    ap.add_argument("--pm-w", type=float, default=0.15)
    ap.add_argument("--nbhd-dir", type=Path, default=ROOT / "outputs/t2/embryo/gate725_nbhd")
    ap.add_argument("--simplex-dir", type=Path, default=ROOT / "outputs/t2/embryo/gate725_simplex")
    ap.add_argument("--resid-dir", type=Path, default=ROOT / "outputs/t2/embryo/gate725_resid")
    ap.add_argument("--project-pool", action="store_true")
    ap.add_argument("--alpha", type=float, default=1.0, help="nbhd: blend with local pick")
    ap.add_argument("--sites", choices=["copy", "occ"], default="copy",
                    help="copy=prior xyz (density); occ=one draw per occupied voxel (flattens density)")
    ap.add_argument("--base", choices=["pick", "prior"], default="pick",
                    help="resid: add residual onto local pick or onto the prior cloud's own X")
    ap.add_argument("--resid-alpha", type=float, default=1.0, help="scale residual when --mode resid")
    ap.add_argument("--interior", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    panel = list(load_panel_for(target_spec(args.setting, dummy_t, cfg), cfg))
    stages = load_stages(args.setting, cfg, panel)
    left, right = stages[args.left], stages[args.right]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    prior_ad = ad.read_h5ad(args.prior)
    C_prior = np.asarray(prior_ad.obsm["spatial_3D"], dtype=np.float64)
    X_prior = _align(
        to_dense(prior_ad.X),
        list(prior_ad.var_names),
        panel,
    )
    rms_use = float(rms_radius(C_prior)) if args.sites == "copy" else float(args.rms)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), rms_use), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), rms_use), dtype=np.float64)
    if args.sites == "copy":
        if int(args.n) == len(C_prior):
            rng_idx = np.arange(len(C_prior), dtype=np.int64)
        else:
            rng_idx = rng.choice(len(C_prior), size=int(args.n), replace=len(C_prior) < int(args.n))
        xyz = np.asarray(C_prior[rng_idx], dtype=np.float32)
        X_base = np.asarray(X_prior[rng_idx], dtype=np.float32)
    else:
        C_prior_s = np.asarray(scale_cloud(C_prior, float(args.rms)), dtype=np.float64)
        xyz = sample_occupancy_sites(C_prior_s, int(args.n), rng, float(args.rms), interior=float(args.interior))
        X_base = None
    q = np.asarray(xyz, dtype=np.float64)
    w = float(args.w)

    pca_path = Path(args.resid_dir) / "pca.joblib"
    if not pca_path.exists():
        pca_path = ROOT / (
            "outputs/t2/heart/gate_w2/pca.joblib"
            if args.setting == "heart"
            else "outputs/t2/embryo/gate725/pca.joblib"
        )
    pca = ExpressionPCA.load(pca_path) if pca_path.exists() else None
    s0 = np.zeros(len(Xl), dtype=np.float64)
    s1 = np.ones(len(Xr), dtype=np.float64)
    if pca is not None:
        prog = _progress_axis(pca.encode(Xl), pca.encode(Xr))
        if prog is not None:
            s0, s1 = np.asarray(prog[0], dtype=np.float64), np.asarray(prog[1], dtype=np.float64)

    if args.mode == "pick":
        X = _local_pick_rows(q, Cl, Xl, s0, Cr, Xr, s1, w, rng, knn=args.knn, pool=True)
    elif args.mode == "pool":
        X = _local_pool_rows(q, Cl, Xl, Cr, Xr, w, args.knn)
    elif args.mode == "pm":
        X_pick = _local_pick_rows(q, Cl, Xl, s0, Cr, Xr, s1, w, rng, knn=args.knn, pool=True)
        mu_loc = _local_pool_rows(q, Cl, Xl, Cr, Xr, w, args.knn)
        d_mu = np.abs(Xr.mean(0) - Xl.mean(0))
        k = int(max(1, min(int(args.k_genes), Xl.shape[1])))
        mask = np.zeros(Xl.shape[1], dtype=np.float32)
        mask[np.argsort(-d_mu)[:k]] = 1.0
        a = float(np.clip(args.pm_w, 0.0, 1.0))
        X = (X_pick + a * (mu_loc - X_pick) * mask).astype(np.float32)
    elif args.mode == "snap":
        from t2.flow import get_device

        X = _snap_rows(q, Cl, Xl, Cr, Xr, w, args.knn, args.nbhd_dir, get_device())
    elif args.mode == "resid":
        import torch

        from t2.flow import get_device
        from t2.pick_resid import load_pick_resid

        if args.base == "prior":
            if X_base is None:
                raise SystemExit("--base prior requires --sites copy")
            X_pk = X_base
        else:
            X_pk = _local_pick_rows(q, Cl, Xl, s0, Cr, Xr, s1, w, rng, knn=args.knn, pool=True)
        device = get_device()
        model, meta = load_pick_resid(Path(args.resid_dir) / "ckpt" / "resid.pt", device=device)
        pca_r = ExpressionPCA.load(Path(args.resid_dir) / "pca.joblib")
        k = int(meta.get("knn", args.knn))
        zl, zr = pca_r.encode(Xl), pca_r.encode(Xr)
        k0 = min(k, len(Cl))
        k1 = min(k, len(Cr))
        _, i0 = cKDTree(Cl).query(q, k=k0)
        _, i1 = cKDTree(Cr).query(q, k=k1)
        i0 = np.asarray(i0, dtype=np.int64).reshape(len(q), k0)
        i1 = np.asarray(i1, dtype=np.int64).reshape(len(q), k1)
        z_l = zl[i0].mean(axis=1)
        z_r = zr[i1].mean(axis=1)
        comp = torch.as_tensor(pca_r.model.components_, device=device, dtype=torch.float32)
        chunks = []
        bs = 1024
        with torch.no_grad():
            for s in range(0, len(q), bs):
                e = min(s + bs, len(q))
                pred, _ = model(
                    torch.as_tensor(z_l[s:e], device=device, dtype=torch.float32),
                    torch.as_tensor(z_r[s:e], device=device, dtype=torch.float32),
                    torch.full((e - s, 1), w, device=device, dtype=torch.float32),
                    torch.as_tensor(X_pk[s:e], device=device, dtype=torch.float32),
                    comp,
                )
                chunks.append(pred.cpu().numpy())
        X_full = np.concatenate(chunks, axis=0)
        a = float(np.clip(args.resid_alpha, 0.0, 1.0))
        X = np.clip(X_pk + a * (X_full - X_pk), 0.0, None).astype(np.float32)
    elif args.mode == "simplex":
        import torch

        from t2.flow import get_device
        from t2.simplex_token import load_simplex

        device = get_device()
        model, meta = load_simplex(Path(args.simplex_dir) / "ckpt" / "simplex.pt", device=device)
        pca_s = ExpressionPCA.load(Path(args.simplex_dir) / "pca.joblib")
        k = int(meta.get("knn", args.knn))
        zl, zr = pca_s.encode(Xl), pca_s.encode(Xr)
        k0 = min(k, len(Cl))
        k1 = min(k, len(Cr))
        _, i0 = cKDTree(Cl).query(q, k=k0)
        _, i1 = cKDTree(Cr).query(q, k=k1)
        i0 = np.atleast_2d(np.asarray(i0, dtype=np.int64).reshape(len(q), k0))
        i1 = np.atleast_2d(np.asarray(i1, dtype=np.int64).reshape(len(q), k1))
        z_l = zl[i0].mean(axis=1)
        z_r = zr[i1].mean(axis=1)
        X_slot = np.concatenate([Xl[i0], Xr[i1]], axis=1).astype(np.float32)
        chunks = []
        bs = 1024
        with torch.no_grad():
            for s in range(0, len(q), bs):
                e = min(s + bs, len(q))
                pred, _ = model(
                    torch.as_tensor(z_l[s:e], device=device, dtype=torch.float32),
                    torch.as_tensor(z_r[s:e], device=device, dtype=torch.float32),
                    torch.full((e - s, 1), w, device=device, dtype=torch.float32),
                    torch.as_tensor(X_slot[s:e], device=device, dtype=torch.float32),
                )
                chunks.append(pred.cpu().numpy())
        X = np.clip(np.concatenate(chunks, axis=0), 0.0, None).astype(np.float32)
    else:
        from t2.flow import get_device

        X_nb = _nbhd_rows(
            q, Cl, Xl, Cr, Xr, w, args.knn, args.nbhd_dir, get_device(), bool(args.project_pool)
        )
        if float(args.alpha) < 1.0:
            X_pk = _local_pick_rows(q, Cl, Xl, s0, Cr, Xr, s1, w, rng, knn=args.knn, pool=True)
            a = float(np.clip(args.alpha, 0.0, 1.0))
            X_nb = (1.0 - a) * X_pk + a * X_nb
        X = np.clip(X_nb, 0.0, None).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_t2(X, np.asarray(xyz, dtype=np.float32), panel, args.out)
    print(
        f"wrote {args.out} n={len(X)} mode={args.mode} sites={args.sites} base={args.base} "
        f"w={w:.3f} knn={args.knn} prior={args.prior.name} anchors={args.left}+{args.right}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
