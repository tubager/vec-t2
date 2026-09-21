#!/usr/bin/env python
"""Sample CondFlow at t* onto a frozen xyz backbone. No target-stage X.

Leave-out:
  .venv/bin/python scripts/67_predict_cond_flow.py \\
    --ckpt-dir outputs/t2/embryo/gate725_condflow \\
    --pred outputs/t2/embryo/gate725_repair/k256.h5ad \\
    --t 7.25 --out outputs/t2/embryo/gate725_condflow/k256_t725.h5ad

Board (only after dual-gate PASS):
  --pred outputs/t2/submit/T2_embryo_val_interp.h5ad --t 7.5 --mode skip_euler --alpha 0.15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
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
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument(
        "--mode",
        choices=["query", "endpoint_lerp", "skip_euler"],
        default="endpoint_lerp",
        help="query: decode/integrate at --t. endpoint_lerp: Φ(t0),Φ(t1) mix. "
        "skip_euler: log1p(X)+dt*(y1-y0) from encoded identity (VAE)",
    )
    ap.add_argument("--w", type=float, default=None, help="clock mix; default (t-t0)/(t1-t0)")
    ap.add_argument(
        "--match-moments",
        choices=["none", "global", "mean", "meanstd"],
        default="none",
        help="match generated moments to L/R clock-mix (no target-stage X). global=one scale (variogram-safe)",
    )
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-scale", type=float, default=1.0, help="scale of z ~ N(0,I) for CondGen; 0=mean field")
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="skip_euler step size on dt*(y1-y0); 0=keep backbone X",
    )
    ap.add_argument(
        "--snap-sparsity",
        action="store_true",
        help="per-gene quantile-zero to L/R clock-mix zero rate (no target-stage X)",
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    model, meta = load_cond_flow(args.ckpt_dir / "ckpt" / "cond_flow.pt", device=device)
    panel = list(meta["panel"])
    ids = dict(meta["cluster_to_id"])
    usable = set(meta.get("usable") or [])
    sigma = np.asarray(meta["sigma"], dtype=np.float32)
    rms = float(meta.get("rms") or 198.24)
    setting = str(meta.get("setting") or "embryo")

    cfg = load_config()
    dummy_t = 7.5 if setting == "embryo" else 8.5
    stages = load_stages(setting, cfg, panel)
    left = stages[str(meta["left"])]
    right = stages[str(meta["right"])]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    cl = np.concatenate(
        [
            np.asarray(labels_to_clusters(left.obs["celltype"], setting)),
            np.asarray(labels_to_clusters(right.obs["celltype"], setting)),
        ]
    )
    ref_X = np.concatenate([Xl, Xr], axis=0)
    keep = np.array([str(c) in usable and str(c) in ids for c in cl])
    ref_X, cl = ref_X[keep], cl[keep]

    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.array(np.asarray(pred.obsm["spatial_3D"], dtype=np.float32), copy=True)
    xyz = np.asarray(scale_cloud(C0, rms), dtype=np.float32) / rms

    nn = NearestNeighbors(n_neighbors=min(int(args.knn), len(ref_X))).fit(ref_X)
    _, idx = nn.kneighbors(Xp)
    cid = np.empty(len(Xp), dtype=np.int64)
    fallback = ids[sorted(usable)[0]] if usable else 0
    for i, neigh in enumerate(idx):
        labs = [str(cl[j]) for j in neigh]
        vals, cnt = np.unique(labs, return_counts=True)
        name = str(vals[int(np.argmax(cnt))])
        cid[i] = int(ids[name]) if name in ids and name in usable else int(fallback)

    t0 = float(meta.get("t0", 6.75))
    t1 = float(meta.get("t1", 8.0))
    w = float(args.w) if args.w is not None else (float(args.t) - t0) / max(t1 - t0, 1e-6)
    w = float(np.clip(w, 0.0, 1.0))

    n_cl = max(int(c) for c in ids.values()) + 1 if ids else 1
    mu_l = np.zeros((n_cl, len(panel)), dtype=np.float32)
    mu_r = np.zeros((n_cl, len(panel)), dtype=np.float32)
    if bool(meta.get("mu_skip")):
        cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], setting))
        cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], setting))
        for name, cid_i in ids.items():
            ml = cl_l == str(name)
            mr = cl_r == str(name)
            if ml.any():
                mu_l[int(cid_i)] = Xl[ml].mean(0)
            if mr.any():
                mu_r[int(cid_i)] = Xr[mr].mean(0)
    mu_mix = (1.0 - w) * mu_l + w * mu_r

    rng = np.random.default_rng(args.seed)
    Xn = np.empty_like(Xp)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(Xp), args.batch):
            sl = slice(start, min(start + args.batch, len(Xp)))
            n = sl.stop - sl.start
            c = torch.as_tensor(cid[sl], device=device)
            zxyz = torch.as_tensor(xyz[sl], device=device)
            if str(meta.get("arch") or "flow") == "gen":
                zdim = int(meta.get("z_dim") or 32)
                z = torch.as_tensor(rng.standard_normal((n, zdim)).astype(np.float32), device=device) * float(args.z_scale)
                use_mu = bool(meta.get("mu_skip"))
                mu0 = torch.as_tensor(mu_l[cid[sl]], device=device) if use_mu else None
                mu1 = torch.as_tensor(mu_r[cid[sl]], device=device) if use_mu else None
                mut = torch.as_tensor(mu_mix[cid[sl]], device=device) if use_mu else None
                if args.mode == "endpoint_lerp":
                    x0 = model.decode(z, z.new_full((n,), t0), c, zxyz, mu0)
                    x1 = model.decode(z, z.new_full((n,), t1), c, zxyz, mu1)
                    xg = (1.0 - w) * x0 + w * x1
                else:
                    xg = model.decode(z, z.new_full((n,), float(args.t)), c, zxyz, mut)
            elif str(meta.get("arch") or "flow") == "vae":
                x_in = torch.as_tensor(Xp[sl], device=device)
                mu_z, _ = model.encode(x_in, c, zxyz)
                z = mu_z
                if args.mode == "skip_euler":
                    y0 = model.decode_log(z, z.new_full((n,), t0), c, zxyz)
                    y1 = model.decode_log(z, z.new_full((n,), t1), c, zxyz)
                    yx = torch.log1p(x_in.clamp_min(0.0))
                    e0 = ((y0 - yx) ** 2).mean(1)
                    e1 = ((y1 - yx) ** 2).mean(1)
                    dt = torch.where(e0 < e1, yx.new_full((n,), w), yx.new_full((n,), w - 1.0))
                    xg = torch.expm1(yx + float(args.alpha) * dt.unsqueeze(1) * (y1 - y0))
                    xg = xg.masked_fill(x_in <= 0, 0.0)
                elif args.mode == "endpoint_lerp":
                    x0 = model.decode(z, z.new_full((n,), t0), c, zxyz)
                    x1 = model.decode(z, z.new_full((n,), t1), c, zxyz)
                    xg = (1.0 - w) * x0 + w * x1
                else:
                    xg = model.decode(z, z.new_full((n,), float(args.t)), c, zxyz)
            else:
                sig_t = torch.as_tensor(sigma, device=device)
                noise = torch.as_tensor(rng.standard_normal((n, len(panel))).astype(np.float32), device=device) * sig_t
                if args.mode == "endpoint_lerp":
                    x0 = integrate(model, noise, noise.new_full((n,), t0), c, zxyz, steps=args.steps)
                    x1 = integrate(model, noise, noise.new_full((n,), t1), c, zxyz, steps=args.steps)
                    xg = (1.0 - w) * x0 + w * x1
                else:
                    tb = noise.new_full((n,), float(args.t))
                    xg = integrate(model, noise, tb, c, zxyz, steps=args.steps)
            Xn[sl] = xg.clamp_min(0.0).cpu().numpy().astype(np.float32)

    if args.snap_sparsity:
        p = (1.0 - w) * (Xl <= 0).mean(0) + w * (Xr <= 0).mean(0)
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 0.99)
        for j in range(Xn.shape[1]):
            pj = float(p[j])
            if pj <= 0:
                continue
            thr = np.quantile(Xn[:, j], pj)
            Xn[Xn[:, j] <= thr, j] = 0.0
        Xn = Xn.astype(np.float32)

    if args.match_moments != "none":
        mu_lr = (1.0 - w) * Xl.mean() + w * Xr.mean()
        if args.match_moments == "global":
            s = float(mu_lr) / max(float(Xn.mean()), 1e-6)
            Xn = np.clip(Xn * s, 0.0, None).astype(np.float32)
        else:
            id_to_name = {int(v): str(k) for k, v in ids.items()}
            names = np.array([id_to_name.get(int(c), "") for c in cid])
            cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], setting))
            cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], setting))
            for name in sorted(set(names) & usable):
                m = names == name
                if not m.any():
                    continue
                ml = cl_l == name
                mr = cl_r == name
                if ml.sum() < 8 or mr.sum() < 8:
                    continue
                mu = (1.0 - w) * Xl[ml].mean(0) + w * Xr[mr].mean(0)
                x = Xn[m]
                mu_g = x.mean(0)
                if args.match_moments == "meanstd":
                    sd_t = np.sqrt((1.0 - w) * Xl[ml].var(0) + w * Xr[mr].var(0) + 1e-8)
                    sd_g = x.std(0) + 1e-6
                    x = mu + (x - mu_g) * (sd_t / sd_g).astype(np.float32)
                else:
                    scale = (mu / np.maximum(mu_g, 1e-6)).astype(np.float32)
                    x = x * scale
                Xn[m] = np.clip(x, 0.0, None).astype(np.float32)

    out = pred.copy()
    if list(pred.var_names) == panel:
        out.X = Xn
    else:
        idxg = {g: i for i, g in enumerate(panel)}
        Xlive = np.asarray(to_dense(pred.X), dtype=np.float32)
        for j, g in enumerate(pred.var_names):
            Xlive[:, j] = Xn[:, idxg[g]]
        out.X = Xlive
    out.obsm["spatial_3D"] = C0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"], dtype=np.float32), C0)
    print(
        f"cond-flow t={args.t:g} mode={args.mode} w={w:.3f} match={args.match_moments} "
        f"z={args.z_scale:g} snap={int(args.snap_sparsity)} alpha={args.alpha:g}  "
        f"mean|X|={float(Xn.mean()):.4f}  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
