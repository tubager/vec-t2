#!/usr/bin/env python
"""Apply residual VAE (or nonparametric residual Gauss) onto a locked xyz scaffold.

  .venv/bin/python scripts/33_predict_residual_vae.py \\
    --live outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --vae-dir outputs/t2/embryo/gate725_rvae \\
    --w 0.4 --mode vae --out out.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters  # noqa: E402
from t2.geometry import rms_radius  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.residual_vae import load_vae  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def assign_clusters(X, ref_X, ref_c, k=15):
    _, idx = cKDTree(ref_X).query(X, k=min(k, len(ref_X)))
    out = np.empty(len(X), dtype=object)
    for i, neigh in enumerate(np.asarray(idx).reshape(len(X), -1)):
        vals, cnt = np.unique(ref_c[neigh], return_counts=True)
        out[i] = vals[int(np.argmax(cnt))]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=Path, required=True)
    ap.add_argument("--vae-dir", type=Path, required=True)
    ap.add_argument("--w", type=float, required=True, help="clock weight toward right μ")
    ap.add_argument(
        "--mode",
        choices=["vae", "gauss", "bootstrap", "blend", "transport"],
        default="vae",
        help="transport: encode live residual at t0, decode at t* (keeps cell identity)",
    )
    ap.add_argument("--alpha", type=float, default=1.0, help="X=(1-a)*live + a*gen (all modes)")
    ap.add_argument("--t-star", type=float, default=None, help="condition time (default lerp t0,t1 by w)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ckpt = Path(args.vae_dir) / "residual_vae.pt"
    model, extra = load_vae(ckpt, device=torch.device(args.device))
    panel = list(extra["panel"])
    ids = dict(extra["cluster_to_id"])
    id_to_name = {int(v): k for k, v in ids.items()}
    mu_l = {k: np.asarray(v, dtype=np.float32) for k, v in extra["mu_left"].items()}
    mu_r = {k: np.asarray(v, dtype=np.float32) for k, v in extra["mu_right"].items()}
    t0, t1 = float(extra["t0"]), float(extra["t1"])
    w = float(np.clip(args.w, 0.0, 1.0))
    t_star = float(args.t_star) if args.t_star is not None else (1.0 - w) * t0 + w * t1

    cfg = load_config()
    setting = str(extra.get("setting") or "embryo")
    dummy_t = 7.5 if setting == "embryo" else 8.5
    spec = target_spec(setting, dummy_t, cfg)
    # panel from checkpoint is authoritative
    stages = load_stages(setting, cfg, list(load_panel_for(spec, cfg)))
    left = stages[str(extra["left"])]
    right = stages[str(extra["right"])]
    X0 = _align(to_dense(left.X), left.var_names, panel)
    X1 = _align(to_dense(right.X), right.var_names, panel)
    cl0 = np.asarray(labels_to_clusters(left.obs["celltype"], setting))
    cl1 = np.asarray(labels_to_clusters(right.obs["celltype"], setting))

    live = ad.read_h5ad(args.live)
    Xl = _align(to_dense(live.X), live.var_names, panel)
    Cl = np.asarray(live.obsm["spatial_3D"], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    i0 = rng.choice(len(X0), min(4000, len(X0)), replace=False)
    i1 = rng.choice(len(X1), min(4000, len(X1)), replace=False)
    live_cl = assign_clusters(
        Xl, np.concatenate([X0[i0], X1[i1]], 0), np.concatenate([cl0[i0], cl1[i1]], 0), k=15
    )

    # precompute per-cluster residual pools / cov for nonparametric modes
    resid_pool: dict[str, np.ndarray] = {}
    resid_cov: dict[str, np.ndarray] = {}
    for name in mu_l:
        a = X0[cl0 == name] - mu_l[name]
        b = X1[cl1 == name] - mu_r[name]
        pool = np.concatenate([a, b], 0)
        resid_pool[name] = pool
        if len(pool) >= 8:
            cov = np.cov(pool.T) + 1e-4 * np.eye(pool.shape[1])
            # PSD
            cov = 0.5 * (cov + cov.T)
            resid_cov[name] = cov

    Xnew = Xl.copy()
    device = torch.device(args.device)
    model.eval()
    for name in np.unique(live_cl):
        key = str(name)
        if key not in mu_l or key not in ids:
            continue
        rows = np.flatnonzero(live_cl == name)
        if len(rows) == 0:
            continue
        mu_star = (1.0 - w) * mu_l[key] + w * mu_r[key]
        n = len(rows)
        if args.mode == "gauss":
            cov = resid_cov.get(key)
            if cov is None:
                r = np.zeros((n, len(panel)), dtype=np.float32)
            else:
                r = rng.multivariate_normal(np.zeros(len(panel)), cov, size=n).astype(np.float32)
            gen = mu_star + r
        elif args.mode == "bootstrap":
            pool = resid_pool[key]
            pick = rng.choice(len(pool), size=n, replace=True)
            gen = mu_star + pool[pick]
        elif args.mode == "transport":
            # Keep each cell's identity: r_live vs left μ → encode@t0 → decode@t*
            r_live = Xl[rows] - mu_l[key]
            cid = torch.full((n,), int(ids[key]), device=device, dtype=torch.long)
            t_src = torch.full((n, 1), t0, device=device, dtype=torch.float32)
            t_dst = torch.full((n, 1), t_star, device=device, dtype=torch.float32)
            with torch.no_grad():
                rt = torch.from_numpy(np.asarray(r_live, dtype=np.float32)).to(device)
                mu_z, logvar = model.encode(rt, t_src, cid)
                # use mean latent (deterministic transport)
                r_hat = model.decode(mu_z, t_dst, cid).cpu().numpy().astype(np.float32)
            gen = mu_star + r_hat
        else:
            # vae prior sample / blend
            cid = torch.full((n,), int(ids[key]), device=device, dtype=torch.long)
            t = torch.full((n, 1), t_star, device=device, dtype=torch.float32)
            with torch.no_grad():
                r = model.sample(t, cid, n=n).cpu().numpy().astype(np.float32)
            gen = mu_star + r
        a = float(np.clip(args.alpha, 0.0, 1.0))
        Xnew[rows] = (1.0 - a) * Xl[rows] + a * gen if a < 1.0 - 1e-12 else gen

    np.clip(Xnew, 0.0, None, out=Xnew)
    out = live.copy()
    out.X = Xnew.astype(np.float32)
    assert np.array_equal(
        np.asarray(out.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(live.obsm["spatial_3D"], dtype=np.float32),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    print(
        f"residual-{args.mode} w={w:.3f} t*={t_star:.3f} alpha={args.alpha:.2f}  "
        f"mean|dX|={float(np.mean(np.abs(Xnew - Xl))):.4f}  rms={rms_radius(Cl):.2f}  "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
