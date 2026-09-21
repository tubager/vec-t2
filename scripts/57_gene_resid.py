#!/usr/bin/env python
"""Gene-space partial residual on a real-cell / pm backbone. Not PCA decode.

Only the k genes with largest |μR−μL| get a learned additive residual. The rest
of X is bit-frozen so variogram cannot freely explode. xyz of --pred is bit-exact.

Train (leave-out E6.75+E8.0 → true E7.25):
  .venv/bin/python scripts/57_gene_resid.py train \\
    --out-dir outputs/t2/embryo/gate725_genresid --epochs 25 --k-genes 16

Apply onto frozen k04b6/09a analog:
  .venv/bin/python scripts/57_gene_resid.py predict \\
    --ckpt-dir outputs/t2/embryo/gate725_genresid \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --w 0.4 --alpha 0.15 \\
    --out outputs/t2/embryo/gate725_genresid/lo_pm_a015.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
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


class GeneResidMLP(nn.Module):
    def __init__(self, d: int, k: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(d) + 1, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(k)),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, z_l, z_r, w):
        if w.ndim == 1:
            w = w[:, None]
        return self.scale * self.net(torch.cat([z_l, z_r, w], dim=-1))


def _gene_mask(Xl, Xr, k):
    dmu = np.abs(np.asarray(Xr).mean(0) - np.asarray(Xl).mean(0))
    return np.argsort(-dmu)[: int(k)].astype(np.int64)


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
    # backbone = spatial-NN mix of left/right gene vectors (real cells on the k genes
    # come from the nearer-progress neighbor mean is chimeric — use left NN as CSS-safe base)
    i0_1 = np.asarray(i0, np.int64).reshape(len(Cm), k)[:, 0]
    i1_1 = np.asarray(i1, np.int64).reshape(len(Cm), k)[:, 0]
    take_r = rng.random(len(Cm)) < float(args.w)
    X_base = Xl[i0_1].copy()
    X_base[take_r] = Xr[i1_1[take_r]]
    X_true = Xm.astype(np.float32)
    n = len(X_true)
    idx = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    model = GeneResidMLP(d=args.pca_dim, k=len(mask), hidden=args.hidden).to(device)
    with torch.no_grad():
        model.scale.fill_(float(args.freeze_scale))
    model.scale.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca.save(args.out_dir / "pca.joblib")
    np.save(args.out_dir / "gene_mask.npy", mask)
    log = args.out_dir / "train.log"

    def apply_resid(ii, resid):
        x = torch.as_tensor(X_base[ii], device=device, dtype=torch.float32).clone()
        x[:, torch.as_tensor(mask, device=device)] = x[:, torch.as_tensor(mask, device=device)] + resid
        return torch.clamp(x, 0.0)

    def batch_loss(ii):
        zl_b = torch.as_tensor(z_l[ii], device=device, dtype=torch.float32)
        zr_b = torch.as_tensor(z_r[ii], device=device, dtype=torch.float32)
        ww = torch.full((len(ii), 1), float(args.w), device=device, dtype=torch.float32)
        resid = model(zl_b, zr_b, ww)
        pred = apply_resid(ii, resid)
        xt = torch.as_tensor(X_true[ii], device=device, dtype=torch.float32)
        mse = ((pred[:, torch.as_tensor(mask, device=device)] - xt[:, torch.as_tensor(mask, device=device)]) ** 2).mean()
        gi = torch.randperm(pred.size(1), device=device)[: min(64, pred.size(1))]
        p = pred[:, gi] - pred[:, gi].mean(0, keepdim=True)
        t = xt[:, gi] - xt[:, gi].mean(0, keepdim=True)
        denom = float(max(len(ii) - 1, 1))
        cov = (((p.T @ p) / denom) - ((t.T @ t) / denom)).pow(2).mean()
        if len(ii) >= 2:
            j = torch.randperm(len(ii), device=device)
            dx = (pred - pred[j]).pow(2).mean(-1)
            dt = (xt - xt[j]).pow(2).mean(-1)
            var = ((dx - dt) / (dt.mean() + 1e-6)).pow(2).mean()
        else:
            var = pred.new_zeros(())
        l2 = resid.pow(2).mean()
        loss = mse + float(args.cov_weight) * cov + float(args.var_weight) * var + float(args.l2_weight) * l2
        return loss, mse, var

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_i)
        losses = []
        for s in range(0, len(tr_i), args.batch):
            ii = tr_i[s : s + args.batch]
            loss, mse, var = batch_loss(ii)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vloss, vmse, vvar = batch_loss(val_i)
        msg = (
            f"epoch {epoch:03d}  train={np.mean(losses):.5f}  "
            f"val={float(vloss):.5f}  vmse={float(vmse):.5f}  vvar={float(vvar):.5f}"
        )
        print(msg, flush=True)
        with log.open("a") as f:
            f.write(msg + "\n")
        if float(vloss) < best:
            best = float(vloss)
            ckpt = args.out_dir / "ckpt" / "gene_resid.pt"
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
                    "freeze_scale": float(args.freeze_scale),
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
    blob = torch.load(args.ckpt_dir / "ckpt" / "gene_resid.pt", map_location=device, weights_only=False)
    model = GeneResidMLP(d=int(blob["d"]), k=int(blob["k"]), hidden=int(blob.get("hidden", 256)))
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
            resid = model(
                torch.as_tensor(z_l[s:e], device=device, dtype=torch.float32),
                torch.as_tensor(z_r[s:e], device=device, dtype=torch.float32),
                torch.full((e - s, 1), float(args.w), device=device, dtype=torch.float32),
            )
            chunks.append(resid.cpu().numpy())
    resid = np.concatenate(chunks, axis=0) * float(args.alpha)
    Xn = Xp.copy()
    Xn[:, mask] = np.clip(Xn[:, mask] + resid, 0.0, None)
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    out.obsm["spatial_3D"] = C0
    assert np.array_equal(np.asarray(out.obsm["spatial_3D"], np.float32), C0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    print(
        f"wrote {args.out}  alpha={args.alpha:g} k={len(mask)}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  xyz bit-exact"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--out-dir", type=Path, required=True)
    tr.add_argument("--epochs", type=int, default=25)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--pca-dim", type=int, default=32)
    tr.add_argument("--hidden", type=int, default=256)
    tr.add_argument("--knn", type=int, default=8)
    tr.add_argument("--k-genes", type=int, default=16)
    tr.add_argument("--w", type=float, default=0.4)
    tr.add_argument("--rms", type=float, default=198.24)
    tr.add_argument("--freeze-scale", type=float, default=0.05)
    tr.add_argument("--cov-weight", type=float, default=0.25)
    tr.add_argument("--var-weight", type=float, default=1.0)
    tr.add_argument("--l2-weight", type=float, default=0.2)
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
    pr.add_argument("--alpha", type=float, default=0.15)
    pr.add_argument("--knn", type=int, default=8)
    pr.add_argument("--rms", type=float, default=198.24)
    pr.add_argument("--device", default=None)
    args = ap.parse_args()
    return cmd_train(args) if args.cmd == "train" else cmd_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
