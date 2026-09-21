#!/usr/bin/env python
"""PCA-384 codec + cluster-time residual generator (board: all three stages).

Noise CFM on raw z underfit (mean_mae 0.105). Predict residual around μ[t,cluster]
in PCA, decode with linear inverse. Independent cells, not OT-pair interpolants.

  .venv/bin/python scripts/70_train_pca_cfm.py --arch gen --mu-skip \\
    --stages E6.75 E7.25 E8.0 --out-dir outputs/t2/embryo/board75_pcagen --epochs 40 --device mps
Leave-out:
  --stages E6.75 E8.0 --out-dir outputs/t2/embryo/gate725_pcagen
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import cluster_id_map, labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.cond_flow import CondFlow, CondGen, integrate, save_cond_flow, variogram_loss  # noqa: E402
from t2.flow import get_device  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def _snap(xg, xt):
    out = np.array(xg, copy=True)
    for j in range(out.shape[1]):
        p = float((xt[:, j] <= 0).mean())
        if p <= 0:
            continue
        thr = np.quantile(out[:, j], p)
        out[out[:, j] <= thr, j] = 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stages", nargs="+", default=("E6.75", "E7.25", "E8.0"))
    ap.add_argument("--arch", choices=["flow", "gen"], default="gen")
    ap.add_argument("--mu-skip", action="store_true", default=True)
    ap.add_argument("--no-mu-skip", action="store_false", dest="mu_skip")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--z-dim", type=int, default=64)
    ap.add_argument("--k", type=int, default=384)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--var-weight", type=float, default=2.0)
    ap.add_argument("--mean-weight", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--max-per-stage", type=int, default=8000)
    ap.add_argument("--holdout", type=float, default=0.12)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    spec = target_spec("embryo", 7.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("embryo", cfg, panel)
    times = stage_times("embryo", cfg)
    usable = set(shared_flow_clusters(stages["E7.25"].obs["celltype"], stages["E8.0"].obs["celltype"], "embryo"))
    ids = cluster_id_map("embryo")
    rng = np.random.default_rng(args.seed)
    tags = [str(s) for s in args.stages]
    t_lo = min(float(times[s]) for s in tags)
    t_hi = max(float(times[s]) for s in tags)

    Xs, Ts, Cs, Zs = [], [], [], []
    for tag in tags:
        adata = stages[tag]
        X = _align(to_dense(adata.X), adata.var_names, panel)
        cl = np.asarray(labels_to_clusters(adata.obs["celltype"], "embryo"))
        xyz = np.asarray(scale_cloud(spatial_xyz(adata), args.rms), dtype=np.float32) / float(args.rms)
        keep = np.array([c in usable and c in ids for c in cl])
        idx = np.flatnonzero(keep)
        if args.max_per_stage > 0 and len(idx) > args.max_per_stage:
            idx = rng.choice(idx, args.max_per_stage, replace=False)
        Xs.append(X[idx])
        Ts.append(np.full(len(idx), float(times[tag]), dtype=np.float32))
        Cs.append(np.array([ids[str(c)] for c in cl[idx]], dtype=np.int64))
        Zs.append(xyz[idx])
        print(f"  {tag}: n={len(idx)} t={times[tag]:g}")

    X = np.concatenate(Xs, axis=0)
    T = np.concatenate(Ts, axis=0)
    C = np.concatenate(Cs, axis=0)
    xyz = np.concatenate(Zs, axis=0)
    Y = np.log1p(np.maximum(X, 0.0))
    perm = rng.permutation(len(Y))
    n_ho = max(512, int(len(Y) * float(args.holdout)))
    ho, tr = perm[:n_ho], perm[n_ho:]
    pca = PCA(n_components=min(int(args.k), Y.shape[1], len(tr) - 1), random_state=0).fit(Y[tr])
    Zraw = pca.transform(Y)
    z_mu = Zraw[tr].mean(0).astype(np.float32)
    z_sd = np.maximum(Zraw[tr].std(0, ddof=1), 0.05).astype(np.float32)
    Z = ((Zraw - z_mu) / z_sd).astype(np.float32)
    print(f"PCA-{pca.n_components_} train n={len(tr)} holdout n={len(ho)} evr={float(pca.explained_variance_ratio_.sum()):.3f}")

    mu_tab = {}
    sd_tab = {}
    for t in np.unique(T[tr]):
        for c in np.unique(C[tr]):
            m = (T[tr] == t) & (C[tr] == c)
            if m.sum() < 8:
                continue
            mu_tab[(float(t), int(c))] = Z[tr][m].mean(0).astype(np.float32)
            sd_tab[(float(t), int(c))] = np.maximum(Z[tr][m].std(0, ddof=1), 0.05).astype(np.float32)

    def lookup_mu(t_arr, c_arr):
        out = np.zeros((len(t_arr), Z.shape[1]), dtype=np.float32)
        for i, (tv, cv) in enumerate(zip(t_arr, c_arr, strict=True)):
            key = (float(tv), int(cv))
            if key in mu_tab:
                out[i] = mu_tab[key]
                continue
            same = [(tt, mu) for (tt, cc), mu in mu_tab.items() if cc == int(cv)]
            if not same:
                continue
            same.sort()
            ts = np.array([s[0] for s in same], dtype=np.float64)
            w = np.clip((float(tv) - ts[0]) / max(ts[-1] - ts[0], 1e-6), 0.0, 1.0) if len(ts) > 1 else 0.0
            if len(same) == 1:
                out[i] = same[0][1]
            else:
                # linear in t between first and last available
                j = int(np.searchsorted(ts, float(tv)))
                j = min(max(j, 1), len(ts) - 1)
                t0, t1 = ts[j - 1], ts[j]
                ww = float(np.clip((float(tv) - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
                out[i] = (1.0 - ww) * same[j - 1][1] + ww * same[j][1]
        return out

    device = torch.device(args.device) if args.device else get_device()
    W = torch.as_tensor(pca.components_.astype(np.float32), device=device)
    b0 = torch.as_tensor(pca.mean_.astype(np.float32), device=device)
    zmu_t = torch.as_tensor(z_mu, device=device)
    zsd_t = torch.as_tensor(z_sd, device=device)

    def decode_x(znorm: torch.Tensor) -> torch.Tensor:
        z = znorm * zsd_t + zmu_t
        y = z @ W + b0
        return torch.expm1(y).clamp_min(0.0)

    if args.arch == "gen":
        model = CondGen(
            n_genes=int(pca.n_components_),
            n_clusters=len(ids),
            hidden=int(args.hidden),
            emb_dim=32,
            xyz_hidden=64,
            z_dim=int(args.z_dim),
            t_anchor=t_lo,
            t_scale=max(t_hi - t_lo, 1e-6),
        ).to(device)
    else:
        model = CondFlow(
            n_genes=int(pca.n_components_),
            n_clusters=len(ids),
            hidden=int(args.hidden),
            emb_dim=16,
            xyz_hidden=32,
            t_anchor=t_lo,
            t_scale=max(t_hi - t_lo, 1e-6),
        ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    Zt = torch.as_tensor(Z)
    Xt = torch.as_tensor(X)
    Tt = torch.as_tensor(T)
    Ct = torch.as_tensor(C)
    Yt = torch.as_tensor(xyz)
    Mu = torch.as_tensor(lookup_mu(T, C))

    model.train()
    for epoch in range(1, int(args.epochs) + 1):
        logs = []
        order = rng.permutation(len(tr))
        for s in range(0, len(tr), int(args.batch)):
            sl = tr[order[s : s + int(args.batch)]]
            if len(sl) < 8:
                continue
            z1 = Zt[sl].to(device)
            x1 = Xt[sl].to(device)
            t_bio = Tt[sl].to(device)
            cid = Ct[sl].to(device)
            xyz_b = Yt[sl].to(device)
            mu = Mu[sl].to(device) if args.mu_skip else None
            if args.arch == "gen":
                eps = torch.randn(len(sl), int(args.z_dim), device=device)
                resid = model(eps, t_bio, cid, xyz_b)
                zhat = resid + mu if mu is not None else resid
                loss = ((zhat - z1) ** 2).mean()
                stats = {"mse": float(loss.detach())}
                xg = decode_x(zhat)
            else:
                z0 = torch.randn_like(z1)
                target = z1 - mu if mu is not None else z1
                tau = torch.rand(len(sl), 1, device=device)
                z_tau = (1.0 - tau) * z0 + tau * target
                u = target - z0
                v = model(z_tau, tau, t_bio, cid, xyz_b)
                loss = ((v - u) ** 2).mean()
                stats = {"cfm": float(loss.detach())}
                zhat = integrate(model, z0.detach(), t_bio, cid, xyz_b, steps=args.steps)
                if mu is not None:
                    zhat = zhat + mu
                xg = decode_x(zhat)
            if args.var_weight > 0:
                lv = variogram_loss(xg, x1)
                loss = loss + float(args.var_weight) * lv
                stats["var"] = float(lv.detach())
            if args.mean_weight > 0:
                lm = ((xg.mean(0) - x1.mean(0)) ** 2).mean()
                loss = loss + float(args.mean_weight) * lm
                stats["mean"] = float(lm.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            stats["loss"] = float(loss.detach())
            logs.append(stats)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            keys = sorted({k for s in logs for k in s})
            msg = " ".join(f"{k}={np.mean([s.get(k, 0.0) for s in logs]):.4f}" for k in keys)
            print(f"epoch {epoch:3d}/{args.epochs}  {msg}", flush=True)

    model.eval()
    with torch.no_grad():
        mu_ho = torch.as_tensor(lookup_mu(T[ho], C[ho]), device=device)
        if args.arch == "gen":
            eps = torch.randn(len(ho), int(args.z_dim), device=device)
            resid = model(eps, Tt[ho].to(device), Ct[ho].to(device), Yt[ho].to(device))
            zhat = resid + mu_ho if args.mu_skip else resid
        else:
            z0 = torch.randn(len(ho), pca.n_components_, device=device)
            zhat = integrate(model, z0, Tt[ho].to(device), Ct[ho].to(device), Yt[ho].to(device), steps=max(int(args.steps), 8))
            if args.mu_skip:
                zhat = zhat + mu_ho
        xg = decode_x(zhat).cpu().numpy()
    xt = X[ho]
    xg_s = _snap(xg, xt)
    mean_mae = float(np.abs(xg_s.mean(0) - xt.mean(0)).mean())
    std_ratio = float(xg_s.std() / max(float(xt.std()), 1e-8))
    frac0_h = float((xg_s <= 0).mean())
    frac0_t = float((xt <= 0).mean())
    rng2 = np.random.default_rng(0)
    i = rng2.integers(0, xg_s.shape[1], size=2048)
    j = rng2.integers(0, xg_s.shape[1], size=2048)
    va = np.mean(np.abs(xg_s[:, i] - xg_s[:, j]) ** 0.5, axis=0)
    vb = np.mean(np.abs(xt[:, i] - xt[:, j]) ** 0.5, axis=0)
    vratio = float(np.mean(va) / max(float(np.mean(vb)), 1e-8))
    ok = (
        mean_mae <= 0.05
        and abs(frac0_h - frac0_t) <= 0.08
        and vratio < 2.0
        and 0.35 <= std_ratio <= 1.8
    )
    print(
        f"RECON GATE native-t SET mean_mae={mean_mae:.4f} std_ratio={std_ratio:.3f} "
        f"frac0={frac0_h:.3f} vs {frac0_t:.3f} var_ratio={vratio:.3f} "
        f"=> {'PASS' if ok else 'FAIL — do not submit'}"
    )

    keys = sorted(mu_tab)
    mu_pack = {
        "t": np.array([k[0] for k in keys], dtype=np.float32),
        "c": np.array([k[1] for k in keys], dtype=np.int64),
        "mu": np.stack([mu_tab[k] for k in keys], axis=0),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pca": pca, "z_mu": z_mu, "z_sd": z_sd, "panel": panel, "mu_pack": mu_pack},
        args.out_dir / "pca.joblib",
    )
    extra = {
        "n_genes": int(pca.n_components_),
        "n_clusters": len(ids),
        "hidden": int(args.hidden),
        "emb_dim": 32 if args.arch == "gen" else 16,
        "xyz_hidden": 64 if args.arch == "gen" else 32,
        "z_dim": int(args.z_dim),
        "arch": args.arch,
        "t_anchor": t_lo,
        "t_scale": float(max(t_hi - t_lo, 1e-6)),
        "setting": "embryo",
        "panel": panel,
        "cluster_to_id": ids,
        "rms": float(args.rms),
        "usable": sorted(usable),
        "epochs": int(args.epochs),
        "recon_mae": mean_mae,
        "recon_pass": bool(ok),
        "std_ratio": std_ratio,
        "var_ratio": vratio,
        "k": int(pca.n_components_),
        "mu_skip": bool(args.mu_skip),
        "stages": tags,
    }
    ckpt = args.out_dir / "ckpt" / "cond_flow.pt"
    save_cond_flow(ckpt, model, extra)
    (args.out_dir / "train.json").write_text(json.dumps({k: v for k, v in extra.items() if k != "panel"}, indent=2, default=str))
    print(f"wrote {ckpt}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
