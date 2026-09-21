#!/usr/bin/env python
"""Train cluster-conditional residual VAE (gene space) for T2 mid-stage generation.

Leave-out embryo:
  .venv/bin/python scripts/32_train_residual_vae.py --setting embryo \\
    --left E6.75 --right E8.0 --exclude E7.25 \\
    --out-dir outputs/t2/embryo/gate725_rvae --epochs 80 --device mps
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

from t2.clusters import cluster_id_map, labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.flow import get_device  # noqa: E402
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.residual_vae import ResidualVAE, save_vae, vae_loss  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--left", type=str, default=None)
    ap.add_argument("--right", type=str, default=None)
    ap.add_argument("--exclude", nargs="*", default=())
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.01, help="KL weight")
    ap.add_argument("--cov-weight", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-cluster", type=int, default=4000)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    spec = target_spec(args.setting, dummy_t, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)
    times = stage_times(args.setting, cfg)
    for name in args.exclude:
        if name in stages:
            del stages[name]
            print(f"excluded {name}")

    left_name = args.left or ("E6.75" if args.setting == "embryo" else "E8.25_late")
    right_name = args.right or ("E8.0" if args.setting == "embryo" else "E9.5")
    # normalize heart left key
    if left_name == "E8.25" and "E8.25_late" in stages:
        left_name = "E8.25_late"
    left, right = stages[left_name], stages[right_name]
    t0, t1 = float(times[left_name]), float(times[right_name])
    usable = shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting)
    ids = cluster_id_map(args.setting)
    cl0 = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))
    cl1 = np.asarray(labels_to_clusters(right.obs["celltype"], args.setting))

    def align(X, genes):
        genes = list(genes)
        if genes == panel:
            return np.asarray(X, dtype=np.float32)
        idx = {g: i for i, g in enumerate(genes)}
        return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)

    X0 = align(to_dense(left.X), left.var_names)
    X1 = align(to_dense(right.X), right.var_names)
    rng = np.random.default_rng(args.seed)

    mu_by: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
    rows_r, rows_t, rows_c = [], [], []
    for name in usable:
        if name not in ids:
            continue
        a = X0[cl0 == name]
        b = X1[cl1 == name]
        if len(a) < 16 or len(b) < 16:
            print(f"  skip {name}: n0={len(a)} n1={len(b)}")
            continue
        if len(a) > args.max_per_cluster:
            a = a[rng.choice(len(a), args.max_per_cluster, replace=False)]
        if len(b) > args.max_per_cluster:
            b = b[rng.choice(len(b), args.max_per_cluster, replace=False)]
        mu_a, mu_b = a.mean(0), b.mean(0)
        mu_by["left"][name] = mu_a.astype(np.float32)
        mu_by["right"][name] = mu_b.astype(np.float32)
        r0 = a - mu_a
        r1 = b - mu_b
        rows_r.append(r0)
        rows_r.append(r1)
        rows_t.append(np.full(len(r0), t0, dtype=np.float32))
        rows_t.append(np.full(len(r1), t1, dtype=np.float32))
        rows_c.append(np.full(len(r0), ids[name], dtype=np.int64))
        rows_c.append(np.full(len(r1), ids[name], dtype=np.int64))
        print(f"  {name}: n0={len(a)} n1={len(b)}")

    R = np.concatenate(rows_r, axis=0)
    T = np.concatenate(rows_t, axis=0)
    C = np.concatenate(rows_c, axis=0)
    print(f"train rows={len(R)} genes={R.shape[1]} clusters={len(mu_by['left'])}")

    device = torch.device(args.device) if args.device else get_device()
    model = ResidualVAE(
        n_genes=R.shape[1],
        n_clusters=len(ids),
        latent=args.latent,
        hidden=args.hidden,
        t_anchor=float(min(t0, t1)),
        t_scale=float(abs(t1 - t0) or 1.0),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    n = len(R)
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, args.batch):
            sl = order[start : start + args.batch]
            r = torch.from_numpy(R[sl]).to(device)
            t = torch.from_numpy(T[sl]).to(device)
            cid = torch.from_numpy(C[sl]).to(device)
            r_hat, mu, logvar = model(r, t, cid)
            loss, _ = vae_loss(r, r_hat, mu, logvar, beta=args.beta, cov_w=args.cov_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:4d}/{args.epochs}  loss={np.mean(losses):.5f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "residual_vae.pt"
    save_vae(
        ckpt,
        model,
        extra={
            "n_genes": int(R.shape[1]),
            "n_clusters": len(ids),
            "latent": int(args.latent),
            "hidden": int(args.hidden),
            "emb_dim": 16,
            "t_anchor": float(min(t0, t1)),
            "t_scale": float(abs(t1 - t0) or 1.0),
            "setting": args.setting,
            "left": left_name,
            "right": right_name,
            "t0": t0,
            "t1": t1,
            "cluster_to_id": ids,
            "panel": panel,
            "mu_left": mu_by["left"],
            "mu_right": mu_by["right"],
            "beta": float(args.beta),
            "cov_weight": float(args.cov_weight),
            "epochs": int(args.epochs),
        },
    )
    meta = {
        "left": left_name,
        "right": right_name,
        "t0": t0,
        "t1": t1,
        "n_rows": int(n),
        "clusters": sorted(mu_by["left"]),
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
