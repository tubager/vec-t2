#!/usr/bin/env python
"""Count-space neighborhood generator: p(X_k | t, spatial nbhd) on expm1(X).

Not PCA decode (nbhd_mid). Not residual-on-wrong-backbone (gene-resid).
Trains on true E7.25 cells from E6.75+E8.0 neighborhoods; at infer blends the
predicted k-gene log1p onto a frozen-xyz backbone (lo-pm / 09a).

  .venv/bin/python scripts/61_count_gen.py train \\
    --out-dir outputs/t2/embryo/gate725_countgen --epochs 30 --k-genes 16

  .venv/bin/python scripts/61_count_gen.py predict \\
    --ckpt-dir outputs/t2/embryo/gate725_countgen \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --w 0.4 --alpha 0.25 \\
    --out outputs/t2/embryo/gate725_countgen/lo_pm_a025.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.flow import get_device  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def _gene_mask(Xl, Xr, k):
    dmu = np.abs(np.asarray(Xr).mean(0) - np.asarray(Xl).mean(0))
    return np.argsort(-dmu)[: int(k)].astype(np.int64)


class CountHead(nn.Module):
    """Predict nonnegative normalized counts for k genes; expose log1p."""

    def __init__(self, d: int, k: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(d) + 1, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(k)),
        )

    def rates(self, z_l, z_r, w):
        if w.ndim == 1:
            w = w[:, None]
        return F.softplus(self.net(torch.cat([z_l, z_r, w], dim=-1))) + 1e-4

    def log1p(self, z_l, z_r, w):
        return torch.log1p(self.rates(z_l, z_r, w))


def cmd_train(args) -> int:
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
    Cl = np.asarray(scale_cloud(spatial_xyz(left), args.rms), np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), args.rms), np.float64)
    Cm = np.asarray(scale_cloud(spatial_xyz(mid), args.rms), np.float64)
    mask = _gene_mask(Xl, Xr, args.k_genes)
    pca = ExpressionPCA(n_components=args.pca_dim, batch_size=int(cfg["pca_batch"]))
    pca.fit([left, right, mid], n_per_stage=max(1, int(cfg["pca_fit_cells"]) // 3), seed=args.seed)
    zl, zr = pca.encode(Xl), pca.encode(Xr)
    k = int(args.knn)
    _, i0 = cKDTree(Cl).query(Cm, k=k)
    _, i1 = cKDTree(Cr).query(Cm, k=k)
    z_l = zl[np.asarray(i0, np.int64).reshape(len(Cm), k)].mean(1).astype(np.float32)
    z_r = zr[np.asarray(i1, np.int64).reshape(len(Cm), k)].mean(1).astype(np.float32)
    y_log = Xm[:, mask].astype(np.float32)
    y_cnt = np.expm1(np.clip(y_log, 0.0, None)).astype(np.float32)

    n = len(y_log)
    idx = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    model = CountHead(d=args.pca_dim, k=len(mask), hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca.save(args.out_dir / "pca.joblib")
    np.save(args.out_dir / "gene_mask.npy", mask)
    log = args.out_dir / "train.log"

    def batch_loss(ii):
        zl_b = torch.as_tensor(z_l[ii], device=device, dtype=torch.float32)
        zr_b = torch.as_tensor(z_r[ii], device=device, dtype=torch.float32)
        ww = torch.full((len(ii), 1), float(args.w), device=device, dtype=torch.float32)
        rate = model.rates(zl_b, zr_b, ww)
        pred = torch.log1p(rate)
        yt = torch.as_tensor(y_log[ii], device=device, dtype=torch.float32)
        ct = torch.as_tensor(y_cnt[ii], device=device, dtype=torch.float32)
        mse = ((pred - yt) ** 2).mean()
        pois = (rate - ct * torch.log(rate)).mean()
        mean_pen = ((pred.mean(0) - yt.mean(0)) ** 2).mean()
        if len(ii) >= 4:
            j = torch.randperm(len(ii), device=device)
            dx = (pred - pred[j]).pow(2).mean(-1)
            dt = (yt - yt[j]).pow(2).mean(-1)
            var = ((dx - dt) / (dt.mean() + 1e-6)).pow(2).mean()
        else:
            var = pred.new_zeros(())
        loss = (
            mse
            + float(args.pois_weight) * pois
            + float(args.mean_weight) * mean_pen
            + float(args.var_weight) * var
        )
        return loss, mse, mean_pen, var

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_i)
        losses = []
        for s in range(0, len(tr_i), args.batch):
            ii = tr_i[s : s + args.batch]
            loss, mse, mean_pen, var = batch_loss(ii)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vloss, vmse, vmean, vvar = batch_loss(val_i)
        msg = (
            f"epoch {epoch:03d}  train={np.mean(losses):.5f}  val={float(vloss):.5f}  "
            f"vmse={float(vmse):.5f}  vmean={float(vmean):.5f}  vvar={float(vvar):.5f}"
        )
        print(msg, flush=True)
        with log.open("a") as f:
            f.write(msg + "\n")
        if float(vloss) < best:
            best = float(vloss)
            ckpt = args.out_dir / "ckpt" / "count_gen.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "d": args.pca_dim,
                    "k": int(len(mask)),
                    "hidden": args.hidden,
                    "knn": args.knn,
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


def cmd_predict(args) -> int:
    cfg = load_config()
    device = torch.device(args.device) if args.device else get_device()
    spec = target_spec("embryo", 7.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("embryo", cfg, panel)
    left, right = stages[args.left], stages[args.right]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.array(np.asarray(pred.obsm["spatial_3D"], np.float32), copy=True)
    pca = ExpressionPCA.load(args.ckpt_dir / "pca.joblib")
    mask = np.load(args.ckpt_dir / "gene_mask.npy")
    blob = torch.load(args.ckpt_dir / "ckpt" / "count_gen.pt", map_location=device, weights_only=False)
    model = CountHead(d=int(blob["d"]), k=int(blob["k"]), hidden=int(blob.get("hidden", 256)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    rms = float(blob.get("rms", args.rms))
    knn = int(blob.get("knn", args.knn))
    Cl = np.asarray(scale_cloud(spatial_xyz(left), rms), np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), rms), np.float64)
    Cq = np.asarray(scale_cloud(C0, rms), np.float64)
    zl, zr = pca.encode(Xl), pca.encode(Xr)
    _, i0 = cKDTree(Cl).query(Cq, k=knn)
    _, i1 = cKDTree(Cr).query(Cq, k=knn)
    z_l = zl[np.asarray(i0, np.int64).reshape(len(Cq), knn)].mean(1)
    z_r = zr[np.asarray(i1, np.int64).reshape(len(Cq), knn)].mean(1)
    chunks = []
    with torch.no_grad():
        for s in range(0, len(Cq), 1024):
            e = min(s + 1024, len(Cq))
            logp = model.log1p(
                torch.as_tensor(z_l[s:e], device=device, dtype=torch.float32),
                torch.as_tensor(z_r[s:e], device=device, dtype=torch.float32),
                torch.full((e - s, 1), float(args.w), device=device, dtype=torch.float32),
            )
            chunks.append(logp.cpu().numpy())
    gen = np.concatenate(chunks, axis=0)
    a = float(np.clip(args.alpha, 0.0, 1.0))
    Xn = Xp.copy()
    Xn[:, mask] = (1.0 - a) * Xp[:, mask] + a * gen
    np.clip(Xn, 0.0, None, out=Xn)
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    out.obsm["spatial_3D"] = C0
    assert np.array_equal(np.asarray(out.obsm["spatial_3D"], np.float32), C0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    print(
        f"wrote {args.out}  alpha={a:g} k={len(mask)}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  xyz bit-exact"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--out-dir", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--pca-dim", type=int, default=32)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--knn", type=int, default=8)
    tr.add_argument("--k-genes", type=int, default=16)
    tr.add_argument("--w", type=float, default=0.4)
    tr.add_argument("--rms", type=float, default=198.24)
    tr.add_argument("--pois-weight", type=float, default=0.05)
    tr.add_argument("--mean-weight", type=float, default=2.0)
    tr.add_argument("--var-weight", type=float, default=1.0)
    tr.add_argument("--val-frac", type=float, default=0.2)
    tr.add_argument("--device", default=None)
    tr.add_argument("--seed", type=int, default=0)
    pr = sub.add_parser("predict")
    pr.add_argument("--ckpt-dir", type=Path, required=True)
    pr.add_argument("--pred", type=Path, required=True)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--left", default="E6.75")
    pr.add_argument("--right", default="E8.0")
    pr.add_argument("--w", type=float, default=0.4)
    pr.add_argument("--alpha", type=float, default=0.25)
    pr.add_argument("--knn", type=int, default=8)
    pr.add_argument("--rms", type=float, default=198.24)
    pr.add_argument("--device", default=None)
    args = ap.parse_args()
    return cmd_train(args) if args.cmd == "train" else cmd_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
