#!/usr/bin/env python
"""Train neighborhood mid mapper on true E7.25 from E6.75+E8.0 kNN features.

  .venv/bin/python scripts/38_train_nbhd_mid.py \\
    --out-dir outputs/t2/embryo/gate725_nbhd --epochs 40 --device cpu
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

from t2.flow import get_device, torch_pca_decode  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.nbhd_mid import NbhdMidMLP, save_nbhd  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


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
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--w", type=float, default=0.4, help="clock (7.25-6.75)/(8-6.75)=0.4")
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--gene-cov-weight", type=float, default=0.25)
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
    zl, zr, zm = pca.encode(Xl), pca.encode(Xr), pca.encode(Xm)

    tree_l, tree_r = cKDTree(Cl), cKDTree(Cr)
    _, i_l = tree_l.query(Cm, k=args.knn)
    _, i_r = tree_r.query(Cm, k=args.knn)
    # use NN mean in PCA as features (stable)
    z_l = zl[i_l].mean(axis=1)
    z_r = zr[i_r].mean(axis=1)
    z_t = zm
    n = len(z_t)
    idx = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_i, tr_i = idx[:n_val], idx[n_val:]

    model = NbhdMidMLP(d=args.pca_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    comp = torch.as_tensor(pca.model.components_, device=device, dtype=torch.float32)
    mean = torch.as_tensor(pca.model.mean_, device=device, dtype=torch.float32)
    w = float(args.w)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca.save(args.out_dir / "pca.joblib")
    log = args.out_dir / "train.log"

    def batch_loss(ii):
        zl_b = torch.as_tensor(z_l[ii], device=device, dtype=torch.float32)
        zr_b = torch.as_tensor(z_r[ii], device=device, dtype=torch.float32)
        zt_b = torch.as_tensor(z_t[ii], device=device, dtype=torch.float32)
        ww = torch.full((len(ii), 1), w, device=device, dtype=torch.float32)
        pred = model(zl_b, zr_b, ww)
        mse = ((pred - zt_b) ** 2).mean()
        # gene-space cov match (hard-ish)
        x_p = torch_pca_decode(pred, comp, mean)
        x_t = torch_pca_decode(zt_b, comp, mean)
        # subsample genes
        g = x_p.size(1)
        k = min(96, g)
        gi = torch.randperm(g, device=device)[:k]
        xp, xt = x_p[:, gi], x_t[:, gi]
        xp = xp - xp.mean(0, keepdim=True)
        xt = xt - xt.mean(0, keepdim=True)
        denom = float(max(len(ii) - 1, 1))
        cp = (xp.T @ xp) / denom
        ct = (xt.T @ xt) / denom
        cov = ((cp - ct) ** 2).mean()
        return mse + float(args.gene_cov_weight) * cov, mse, cov

    best_val = float("inf")
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
        if float(vloss) < best_val:
            best_val = float(vloss)
            save_nbhd(
                args.out_dir / "ckpt" / "nbhd.pt",
                model,
                {
                    "d": args.pca_dim,
                    "hidden": args.hidden,
                    "dropout": 0.0,
                    "w_train": w,
                    "knn": args.knn,
                    "rms": float(args.rms),
                    "left": "E6.75",
                    "right": "E8.0",
                    "mid": "E7.25",
                    "epoch": epoch,
                    "best_val": best_val,
                },
            )
            print(f"  saved best val={best_val:.5f}", flush=True)

    print("done", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
