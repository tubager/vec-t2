#!/usr/bin/env python
"""Train spatial/time/cluster-conditional noise→X flow (leave-out: no E7.25).

  .venv/bin/python scripts/66_train_cond_flow.py \\
    --out-dir outputs/t2/embryo/gate725_condflow --epochs 40 --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scipy.optimize import linear_sum_assignment  # noqa: E402
from t2.clusters import cluster_id_map, labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.cond_flow import CondAE, CondFlow, CondGen, CondVAE, integrate, save_cond_flow, variogram_loss  # noqa: E402
from t2.flow import cov_frobenius_loss, get_device  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
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
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--exclude", nargs="*", default=("E7.25",), help="stages to drop; pass NONE for board-window train")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--arch", choices=["flow", "gen", "vae", "ae"], default="gen")
    ap.add_argument("--l1-weight", type=float, default=0.0)
    ap.add_argument("--holdout", type=float, default=0.15, help="fraction held out for recon gate")
    ap.add_argument("--kl-weight", type=float, default=0.05)
    ap.add_argument("--pair-weight", type=float, default=2.0, help="VAE same-cluster OT time-transfer loss")
    ap.add_argument("--pair-cap", type=int, default=1200)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--var-weight", type=float, default=5.0)
    ap.add_argument("--cov-weight", type=float, default=1.0)
    ap.add_argument("--mean-weight", type=float, default=8.0)
    ap.add_argument("--mse-weight", type=float, default=1.0)
    ap.add_argument("--zero-weight", type=float, default=2.0)
    ap.add_argument("--mu-skip", action="store_true", default=True)
    ap.add_argument("--no-mu-skip", action="store_false", dest="mu_skip")
    ap.add_argument("--steps", type=int, default=4, help="Euler steps for endpoint regularizer")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-stage", type=int, default=12000)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    spec = target_spec(args.setting, dummy_t, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)
    times = stage_times(args.setting, cfg)
    for name in args.exclude:
        if not name or str(name).upper() == "NONE":
            continue
        if name in stages:
            del stages[name]
            print(f"excluded {name}")

    left, right = stages[args.left], stages[args.right]
    t0, t1 = float(times[args.left]), float(times[args.right])
    usable = set(shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting))
    ids = cluster_id_map(args.setting)
    rng = np.random.default_rng(args.seed)

    Xs, Ts, Cs, Zs = [], [], [], []
    for tag, adata, t in ((args.left, left, t0), (args.right, right, t1)):
        X = _align(to_dense(adata.X), adata.var_names, panel)
        cl = np.asarray(labels_to_clusters(adata.obs["celltype"], args.setting))
        xyz = np.asarray(scale_cloud(spatial_xyz(adata), args.rms), dtype=np.float32) / float(args.rms)
        keep = np.array([c in usable and c in ids for c in cl])
        idx = np.flatnonzero(keep)
        if args.max_per_stage > 0 and len(idx) > args.max_per_stage:
            idx = rng.choice(idx, args.max_per_stage, replace=False)
        Xs.append(X[idx])
        Ts.append(np.full(len(idx), t, dtype=np.float32))
        Cs.append(np.array([ids[str(c)] for c in cl[idx]], dtype=np.int64))
        Zs.append(xyz[idx])
        print(f"  {tag}: n={len(idx)} t={t:g} clusters={len({str(c) for c in cl[idx] if c in usable})}")

    X = np.concatenate(Xs, axis=0)
    T = np.concatenate(Ts, axis=0)
    C = np.concatenate(Cs, axis=0)
    Z = np.concatenate(Zs, axis=0)
    sigma = np.maximum(X.std(0, ddof=1), 0.05).astype(np.float32)
    mu_tab = {}
    for t in np.unique(T):
        for c in np.unique(C):
            m = (T == t) & (C == c)
            if m.any():
                mu_tab[(float(t), int(c))] = X[m].mean(0).astype(np.float32)
    print(f"train n={len(X)} G={X.shape[1]} shared={sorted(usable)}")
    ho_X = ho_T = ho_C = ho_Z = None
    if args.holdout > 0 and args.arch in {"ae", "vae"}:
        perm = rng.permutation(len(X))
        n_ho = max(256, int(len(X) * float(args.holdout)))
        n_ho = min(n_ho, len(X) // 5)
        ho, tr = perm[:n_ho], perm[n_ho:]
        ho_X, ho_T, ho_C, ho_Z = X[ho], T[ho], C[ho], Z[ho]
        X, T, C, Z = X[tr], T[tr], C[tr], Z[tr]
        print(f"holdout n={len(ho_X)} train n={len(X)}")

    pair_X0 = pair_X1 = pair_C = pair_Z = None
    if args.arch == "vae" and args.pair_weight > 0:
        p0, p1, pc, pz = [], [], [], []
        for name in sorted(usable):
            cid_i = int(ids[str(name)])
            i0 = np.flatnonzero((np.isclose(T, t0)) & (C == cid_i))
            i1 = np.flatnonzero((np.isclose(T, t1)) & (C == cid_i))
            if len(i0) < 8 or len(i1) < 8:
                continue
            m = int(min(args.pair_cap, len(i0), len(i1)))
            s0 = rng.choice(i0, size=m, replace=False)
            s1 = rng.choice(i1, size=m, replace=False)
            a = X[s0].astype(np.float64)
            b = X[s1].astype(np.float64)
            cost = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
            ri, ci = linear_sum_assignment(cost)
            p0.append(X[s0[ri]])
            p1.append(X[s1[ci]])
            pc.append(np.full(m, cid_i, dtype=np.int64))
            pz.append(0.5 * (Z[s0[ri]] + Z[s1[ci]]))
        pair_X0 = np.concatenate(p0, axis=0)
        pair_X1 = np.concatenate(p1, axis=0)
        pair_C = np.concatenate(pc, axis=0)
        pair_Z = np.concatenate(pz, axis=0)
        print(f"OT pairs n={len(pair_X0)}")

    device = torch.device(args.device) if args.device else get_device()
    if args.arch == "gen":
        model = CondGen(
            n_genes=X.shape[1],
            n_clusters=len(ids),
            hidden=args.hidden,
            z_dim=args.z_dim,
            t_anchor=t0,
            t_scale=abs(t1 - t0) or 1.0,
        ).to(device)
    elif args.arch == "ae":
        model = CondAE(
            n_genes=X.shape[1],
            n_clusters=len(ids),
            hidden=args.hidden,
            z_dim=args.z_dim,
            t_anchor=t0,
            t_scale=abs(t1 - t0) or 1.0,
        ).to(device)
    elif args.arch == "vae":
        model = CondVAE(
            n_genes=X.shape[1],
            n_clusters=len(ids),
            hidden=args.hidden,
            z_dim=args.z_dim,
            t_anchor=t0,
            t_scale=abs(t1 - t0) or 1.0,
        ).to(device)
    else:
        model = CondFlow(
            n_genes=X.shape[1],
            n_clusters=len(ids),
            hidden=args.hidden,
            t_anchor=t0,
            t_scale=abs(t1 - t0) or 1.0,
        ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sig_t = torch.as_tensor(sigma, device=device)
    n = len(X)
    model.train()
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(n)
        logs = []
        for start in range(0, n, args.batch):
            sl = order[start : start + args.batch]
            x1 = torch.as_tensor(X[sl], device=device)
            t_bio = torch.as_tensor(T[sl], device=device)
            cid = torch.as_tensor(C[sl], device=device)
            xyz = torch.as_tensor(Z[sl], device=device)
            if args.arch in {"vae", "ae"}:
                mu_z, logv = model.encode(x1, cid, xyz)
                z = mu_z if args.arch == "ae" else model.reparam(mu_z, logv)
                yhat = model.decode_log(z, t_bio, cid, xyz)
                xg = torch.expm1(yhat).clamp_min(0.0)
                recon = ((yhat - torch.log1p(x1)) ** 2).mean()
                loss = float(args.mse_weight) * recon
                stats = {"mse": float(recon.detach())}
                if args.arch == "vae":
                    kl = 0.5 * (mu_z.pow(2) + logv.exp() - logv - 1).mean()
                    loss = loss + float(args.kl_weight) * kl
                    stats["kl"] = float(kl.detach())
                if float(args.l1_weight) > 0:
                    l1 = (xg - x1).abs().mean()
                    loss = loss + float(args.l1_weight) * l1
                    stats["l1"] = float(l1.detach())
                if pair_X0 is not None and args.pair_weight > 0 and args.arch == "vae":
                    pb = rng.integers(0, len(pair_X0), size=min(args.batch, len(pair_X0)))
                    xa = torch.as_tensor(pair_X0[pb], device=device)
                    xb = torch.as_tensor(pair_X1[pb], device=device)
                    cp = torch.as_tensor(pair_C[pb], device=device)
                    zp = torch.as_tensor(pair_Z[pb], device=device)
                    za, _ = model.encode(xa, cp, zp)
                    yb = model.decode_log(za, xa.new_full((len(pb),), t1), cp, zp)
                    zb, _ = model.encode(xb, cp, zp)
                    ya = model.decode_log(zb, xb.new_full((len(pb),), t0), cp, zp)
                    lp = ((yb - torch.log1p(xb)) ** 2).mean() + ((ya - torch.log1p(xa)) ** 2).mean()
                    loss = loss + float(args.pair_weight) * lp
                    stats["pair"] = float(lp.detach())
            elif args.arch == "gen":
                z = torch.randn(len(sl), args.z_dim, device=device)
                mu = None
                if args.mu_skip:
                    mu = torch.as_tensor(
                        np.stack([mu_tab[(float(T[i]), int(C[i]))] for i in sl], axis=0),
                        device=device,
                    )
                yhat = model(z, t_bio, cid, xyz, mu)
                xg = torch.expm1(yhat).clamp_min(0.0)
                loss = float(args.mse_weight) * ((yhat - torch.log1p(x1)) ** 2).mean()
                stats = {"mse": float(loss.detach())}
            else:
                x0 = torch.randn_like(x1) * sig_t
                tau = torch.rand(len(sl), 1, device=device)
                x_tau = (1.0 - tau) * x0 + tau * x1
                u = x1 - x0
                v = model(x_tau, tau, t_bio, cid, xyz)
                loss = ((v - u) ** 2).mean()
                stats = {"cfm": float(loss.detach())}
                xg = integrate(model, x0.detach(), t_bio, cid, xyz, steps=args.steps)
                xg = xg.clamp_min(0.0)
            if args.var_weight > 0:
                lv = variogram_loss(xg, x1)
                loss = loss + float(args.var_weight) * lv
                stats["var"] = float(lv.detach())
            if args.cov_weight > 0:
                lc = cov_frobenius_loss(xg, x1)
                loss = loss + float(args.cov_weight) * lc
                stats["cov"] = float(lc.detach())
            if args.mean_weight > 0:
                lm = ((xg.mean(0) - x1.mean(0)) ** 2).mean()
                loss = loss + float(args.mean_weight) * lm
                stats["mean"] = float(lm.detach())
            if args.arch in {"gen", "vae", "ae"} and args.zero_weight > 0:
                lz = (xg * (x1 <= 0).float()).mean()
                loss = loss + float(args.zero_weight) * lz
                stats["zero"] = float(lz.detach())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            stats["loss"] = float(loss.detach())
            logs.append(stats)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            keys = sorted({k for s in logs for k in s})
            msg = " ".join(f"{k}={np.mean([s.get(k, 0) for s in logs]):.4f}" for k in keys)
            print(f"epoch {epoch:3d}/{args.epochs}  {msg}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "n_genes": int(X.shape[1]),
        "n_clusters": len(ids),
        "hidden": int(args.hidden),
        "emb_dim": 32 if args.arch in {"gen", "vae", "ae"} else 16,
        "xyz_hidden": 64 if args.arch in {"gen", "vae", "ae"} else 32,
        "z_dim": int(args.z_dim),
        "arch": args.arch,
        "t_anchor": float(t0),
        "t_scale": float(abs(t1 - t0) or 1.0),
        "setting": args.setting,
        "left": args.left,
        "right": args.right,
        "t0": t0,
        "t1": t1,
        "panel": panel,
        "cluster_to_id": ids,
        "sigma": sigma,
        "rms": float(args.rms),
        "usable": sorted(usable),
        "epochs": int(args.epochs),
        "var_weight": float(args.var_weight),
        "cov_weight": float(args.cov_weight),
        "mean_weight": float(args.mean_weight),
        "kl_weight": float(getattr(args, "kl_weight", 0.0)),
        "pair_weight": float(getattr(args, "pair_weight", 0.0)),
        "mu_skip": bool(args.mu_skip),
    }
    ckpt = args.out_dir / "ckpt" / "cond_flow.pt"
    save_cond_flow(ckpt, model, extra)
    (args.out_dir / "train_meta.json").write_text(json.dumps({k: extra[k] for k in extra if k != "sigma"}, indent=2, default=str))
    print(f"saved {ckpt}")
    if ho_X is not None:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(ho_X), 512):
                sl = slice(start, min(start + 512, len(ho_X)))
                x = torch.as_tensor(ho_X[sl], device=device)
                t_bio = torch.as_tensor(ho_T[sl], device=device)
                cid = torch.as_tensor(ho_C[sl], device=device)
                xyz = torch.as_tensor(ho_Z[sl], device=device)
                z, _ = model.encode(x, cid, xyz)
                preds.append(model.decode(z, t_bio, cid, xyz).cpu().numpy())
        xg_np = np.concatenate(preds, axis=0)
        mae = float(np.abs(xg_np - ho_X).mean())
        frac0_h = float((xg_np <= 0).mean())
        frac0_t = float((ho_X <= 0).mean())
        corr = float(np.corrcoef(xg_np.mean(0), ho_X.mean(0))[0, 1])
        ok_mae = mae <= 0.03
        ok_sp = abs(frac0_h - frac0_t) <= 0.05
        print(
            f"RECON GATE holdout mae={mae:.4f} ({'PASS' if ok_mae else 'FAIL'} ≤0.03)  "
            f"frac0={frac0_h:.3f} vs {frac0_t:.3f} ({'PASS' if ok_sp else 'FAIL'})  "
            f"mean-corr={corr:.3f}  => {'PASS' if ok_mae and ok_sp else 'FAIL — do not submit'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
