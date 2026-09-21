#!/usr/bin/env python
"""Simplex neighborhood mixer: X is a convex combination of real spatial neighbors.

Oracle nbhd-MLP at true E7.25 xyz hits NFS 0.049 but mmd/var explode (PCA decode).
This model cannot leave the convex hull of real cells, so variogram is bounded.

Train (leave-out):
  .venv/bin/python scripts/47_train_simplex_token.py \\
    --out-dir outputs/t2/embryo/gate725_simplex --epochs 40

Predict via scripts/46_token_birth.py --mode simplex --simplex-dir ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.flow import get_device  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402
from t2.simplex_token import SimplexMixer  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--w", type=float, default=0.4)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--cov-weight", type=float, default=0.05)
    ap.add_argument("--clock-weight", type=float, default=0.1)
    ap.add_argument("--ent-weight", type=float, default=0.01)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device(args.device) if args.device else get_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    spec = target_spec("embryo", 7.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("embryo", cfg, panel)
    left, right, mid = stages["E6.75"], stages["E8.0"], stages["E7.25"]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Xm = _align(to_dense(mid.X), mid.var_names, panel)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), args.rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), args.rms), dtype=np.float64)
    Cm = np.asarray(scale_cloud(spatial_xyz(mid), args.rms), dtype=np.float64)

    pca = ExpressionPCA(n_components=args.pca_dim, batch_size=int(cfg["pca_batch"]))
    pca.fit([left, right, mid], n_per_stage=max(1, int(cfg["pca_fit_cells"]) // 3), seed=args.seed)
    zl, zr = pca.encode(Xl), pca.encode(Xr)

    k = int(args.knn)
    _, i0 = cKDTree(Cl).query(Cm, k=k)
    _, i1 = cKDTree(Cr).query(Cm, k=k)
    i0 = np.asarray(i0, dtype=np.int64).reshape(len(Cm), k)
    i1 = np.asarray(i1, dtype=np.int64).reshape(len(Cm), k)
    z_l = zl[i0].mean(axis=1)
    z_r = zr[i1].mean(axis=1)
    n_nb = int(i0.shape[1] + i1.shape[1])
    z_t = Xm.astype(np.float32)
    print(f"device={device} n={len(z_t)} genes={z_t.shape[1]} knn={k} n_nb={n_nb}", flush=True)

    n = len(z_t)
    idx = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_i, tr_i = idx[:n_val], idx[n_val:]

    model = SimplexMixer(d=args.pca_dim, n_nb=n_nb, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca.save(args.out_dir / "pca.joblib")
    log = args.out_dir / "train.log"

    def batch_loss(ii):
        zl_b = torch.as_tensor(z_l[ii], device=device, dtype=torch.float32)
        zr_b = torch.as_tensor(z_r[ii], device=device, dtype=torch.float32)
        xt = torch.as_tensor(z_t[ii], device=device, dtype=torch.float32)
        xnb = np.concatenate([Xl[i0[ii]], Xr[i1[ii]]], axis=1).astype(np.float32)
        xnb = torch.as_tensor(xnb, device=device, dtype=torch.float32)
        ww = torch.full((len(ii), 1), float(args.w), device=device, dtype=torch.float32)
        pred, wt = model(zl_b, zr_b, ww, xnb)
        mse = ((pred - xt) ** 2).mean()
        p = pred - pred.mean(0, keepdim=True)
        t = xt - xt.mean(0, keepdim=True)
        g = pred.size(1)
        gi = torch.randperm(g, device=device)[: min(64, g)]
        denom = float(max(len(ii) - 1, 1))
        cp = (p[:, gi].T @ p[:, gi]) / denom
        ct = (t[:, gi].T @ t[:, gi]) / denom
        cov = ((cp - ct) ** 2).mean()
        left_mass = wt[:, : i0.shape[1]].sum(dim=-1)
        clock = ((left_mass - (1.0 - float(args.w))) ** 2).mean()
        ent = -(wt * (wt.clamp_min(1e-8).log())).sum(dim=-1).mean()
        loss = (
            mse
            + float(args.cov_weight) * cov
            + float(args.clock_weight) * clock
            - float(args.ent_weight) * ent
        )
        return loss, mse, cov

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_i)
        losses = []
        for s in range(0, len(tr_i), args.batch):
            ii = tr_i[s : s + args.batch]
            loss, mse, cov = batch_loss(ii)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vloss, vmse, vcov = batch_loss(val_i)
        msg = (
            f"epoch {epoch:03d}  train={np.mean(losses):.5f}  "
            f"val={float(vloss):.5f}  vmse={float(vmse):.5f}  vcov={float(vcov):.5f}"
        )
        print(msg, flush=True)
        with log.open("a") as f:
            f.write(msg + "\n")
        if float(vloss) < best:
            best = float(vloss)
            ckpt = args.out_dir / "ckpt" / "simplex.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "d": args.pca_dim,
                    "n_nb": n_nb,
                    "knn": k,
                    "k_left": int(i0.shape[1]),
                    "hidden": args.hidden,
                    "w_train": float(args.w),
                    "rms": float(args.rms),
                    "epoch": epoch,
                    "best_val": best,
                },
                ckpt,
            )
            print(f"  saved best val={best:.5f}", flush=True)
    print("done", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
