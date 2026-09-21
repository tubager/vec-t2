#!/usr/bin/env python
"""Sample PCA-384 CFM at t* onto a frozen xyz backbone. No target-stage X.

Board (only if 70_train_pca_cfm RECON GATE PASS):
  .venv/bin/python scripts/71_predict_pca_cfm.py \\
    --ckpt-dir outputs/t2/embryo/board75_pcacfm --t 7.5 \\
    --pred outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --out outputs/t2/queue/2026-09-20/208_embryo_pca384_cfm_t75_on194_n5000.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters  # noqa: E402
from t2.cond_flow import integrate, load_cond_flow  # noqa: E402
from t2.flow import get_device  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    model, meta = load_cond_flow(args.ckpt_dir / "ckpt" / "cond_flow.pt", device=device)
    if not bool(meta.get("recon_pass", False)):
        raise SystemExit("checkpoint recon_pass is false — do not submit")
    blob = joblib.load(args.ckpt_dir / "pca.joblib")
    pca, z_mu, z_sd = blob["pca"], np.asarray(blob["z_mu"]), np.asarray(blob["z_sd"])
    mu_pack = blob.get("mu_pack")
    panel = list(meta["panel"])
    ids = dict(meta["cluster_to_id"])
    usable = set(meta.get("usable") or [])
    rms = float(meta.get("rms") or 198.24)
    arch = str(meta.get("arch") or "flow")
    mu_skip = bool(meta.get("mu_skip", False))
    z_dim = int(meta.get("z_dim") or 64)

    cfg = load_config()
    spec = target_spec("embryo", float(args.t), cfg)
    stages = load_stages("embryo", cfg, panel)
    left, right = stages[spec.left], stages[spec.right]
    t0, t1 = float(spec.t_left), float(spec.t_right)
    w = float(np.clip((float(args.t) - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], "embryo"))
    cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], "embryo"))
    keep_l = np.array([str(c) in usable for c in cl_l])
    keep_r = np.array([str(c) in usable for c in cl_r])
    Xcat = np.concatenate([Xl[keep_l], Xr[keep_r]], axis=0)
    cl_cat = np.concatenate([cl_l[keep_l], cl_r[keep_r]])

    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.array(np.asarray(pred.obsm["spatial_3D"], dtype=np.float32), copy=True)
    nn = NearestNeighbors(n_neighbors=min(int(args.knn), len(Xcat))).fit(Xcat)
    _, idx = nn.kneighbors(Xp)
    names = []
    for neigh in idx:
        labs = [str(cl_cat[j]) for j in neigh]
        vals, cnt = np.unique(labs, return_counts=True)
        names.append(str(vals[int(np.argmax(cnt))]))
    cid = np.array([ids.get(n, 0) for n in names], dtype=np.int64)
    xyz = np.asarray(scale_cloud(C0, rms), dtype=np.float32) / float(rms)

    torch.manual_seed(int(args.seed))
    t_bio = torch.full((len(Xp),), float(args.t), device=device)
    cid_t = torch.as_tensor(cid, device=device)
    xyz_t = torch.as_tensor(xyz, device=device)
    mu_np = np.zeros((len(Xp), int(pca.n_components_)), dtype=np.float32)
    if mu_skip and mu_pack is not None:
        mt = np.asarray(mu_pack["t"])
        mc = np.asarray(mu_pack["c"])
        mm = np.asarray(mu_pack["mu"])
        for i, cv in enumerate(cid):
            sel = np.flatnonzero(mc == int(cv))
            if len(sel) == 0:
                continue
            ts = mt[sel]
            order = np.argsort(ts)
            ts, mus = ts[order], mm[sel][order]
            if len(ts) == 1 or float(args.t) <= ts[0]:
                mu_np[i] = mus[0]
            elif float(args.t) >= ts[-1]:
                mu_np[i] = mus[-1]
            else:
                j = int(np.searchsorted(ts, float(args.t)))
                t0m, t1m = float(ts[j - 1]), float(ts[j])
                ww = float(np.clip((float(args.t) - t0m) / max(t1m - t0m, 1e-6), 0.0, 1.0))
                mu_np[i] = (1.0 - ww) * mus[j - 1] + ww * mus[j]
    with torch.no_grad():
        if arch == "gen":
            eps = torch.randn(len(Xp), z_dim, device=device)
            resid = model(eps, t_bio, cid_t, xyz_t)
            zhat = resid + torch.as_tensor(mu_np, device=device) if mu_skip else resid
        else:
            z0 = torch.randn(len(Xp), int(pca.n_components_), device=device)
            zhat = integrate(model, z0, t_bio, cid_t, xyz_t, steps=int(args.steps))
            if mu_skip:
                zhat = zhat + torch.as_tensor(mu_np, device=device)
        z = zhat.cpu().numpy() * z_sd + z_mu
    Xn = np.clip(np.expm1(pca.inverse_transform(z)), 0.0, None).astype(np.float32)
    mix = (1.0 - w) * (Xl[keep_l] <= 0).mean(0) + w * (Xr[keep_r] <= 0).mean(0)
    for j, p in enumerate(mix):
        if p <= 0:
            continue
        thr = np.quantile(Xn[:, j], float(p))
        Xn[Xn[:, j] <= thr, j] = 0.0

    out = pred.copy()
    if list(pred.var_names) == panel:
        out.X = Xn
    else:
        live = np.asarray(to_dense(pred.X), dtype=np.float32)
        idxg = {g: i for i, g in enumerate(panel)}
        for j, g in enumerate(pred.var_names):
            live[:, j] = Xn[:, idxg[g]]
        out.X = live
    out.obsm["spatial_3D"] = C0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"], dtype=np.float32), C0)
    print(
        f"pca-cfm t={args.t:g} w={w:.3f} mean={float(Xn.mean()):.4f} frac0={float((Xn <= 0).mean()):.3f} "
        f"mae_vs_pred={float(np.abs(Xn - Xp).mean()):.4f} wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
