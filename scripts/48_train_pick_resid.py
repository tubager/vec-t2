#!/usr/bin/env python
"""Train a small PCA residual on top of a real spatial-neighbor pick.

Simplex mixing of L+R is chimeric. PCA-decode nbhd hits NFS~0.049 at true xyz
but blows mmd/var. This keeps a real cell and only adds a regularized residual.

Hard losses (leave-out E6.75+E8.0 → true E7.25):
  MSE to true X, gene-cov, kNN-15 pseudobulk (NFS proxy), pairwise ||ΔX|| (variogram),
  L2 on residual.

  .venv/bin/python scripts/48_train_pick_resid.py \\
    --out-dir outputs/t2/embryo/gate725_resid --epochs 40

Heart W2 (E8.25+E9.5 → E8.75):
  .venv/bin/python scripts/48_train_pick_resid.py --setting heart \\
    --left E8.25 --right E9.5 --mid E8.75 --w 0.4 --rms 346.33 \\
    --freeze-scale 0.05 --max-n 8000 \\
    --out-dir outputs/t2/heart/gate_w2_resid --epochs 30
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
from t2.ot import _local_pick_rows, _progress_axis  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402
from t2.pick_resid import PickResidualMLP  # noqa: E402


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
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--left", default=None)
    ap.add_argument("--right", default=None)
    ap.add_argument("--mid", default=None)
    ap.add_argument("--max-n", type=int, default=None, help="subsample mid cells for train speed")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--nfs-k", type=int, default=15)
    ap.add_argument("--w", type=float, default=0.4)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--cov-weight", type=float, default=0.25)
    ap.add_argument("--nfs-weight", type=float, default=2.0)
    ap.add_argument("--var-weight", type=float, default=0.15)
    ap.add_argument("--l2-weight", type=float, default=0.1)
    ap.add_argument("--freeze-scale", type=float, default=None,
                    help="if set, keep residual scale fixed (blocks CSS blow-up)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device(args.device) if args.device else get_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    spec = target_spec(args.setting, dummy_t, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)
    left_n = args.left or ("E6.75" if args.setting == "embryo" else "E8.25")
    right_n = args.right or ("E8.0" if args.setting == "embryo" else "E9.5")
    mid_n = args.mid or ("E7.25" if args.setting == "embryo" else "E8.75")
    left, right, mid = stages[left_n], stages[right_n], stages[mid_n]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Xm = _align(to_dense(mid.X), mid.var_names, panel)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), args.rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), args.rms), dtype=np.float64)
    Cm = np.asarray(scale_cloud(spatial_xyz(mid), args.rms), dtype=np.float64)

    pca = ExpressionPCA(n_components=args.pca_dim, batch_size=int(cfg["pca_batch"]))
    pca.fit([left, right, mid], n_per_stage=max(1, int(cfg["pca_fit_cells"]) // 3), seed=args.seed)
    zl, zr = pca.encode(Xl), pca.encode(Xr)
    prog = _progress_axis(zl, zr)
    s0 = np.asarray(prog[0], dtype=np.float64) if prog is not None else np.zeros(len(Xl))
    s1 = np.asarray(prog[1], dtype=np.float64) if prog is not None else np.ones(len(Xr))

    k = int(args.knn)
    _, i0 = cKDTree(Cl).query(Cm, k=k)
    _, i1 = cKDTree(Cr).query(Cm, k=k)
    i0 = np.asarray(i0, dtype=np.int64).reshape(len(Cm), k)
    i1 = np.asarray(i1, dtype=np.int64).reshape(len(Cm), k)
    z_l = zl[i0].mean(axis=1).astype(np.float32)
    z_r = zr[i1].mean(axis=1).astype(np.float32)

    X_pick = _local_pick_rows(
        Cm, Cl, Xl, s0, Cr, Xr, s1, float(args.w), rng, knn=k, pool=True
    ).astype(np.float32)
    X_true = Xm.astype(np.float32)
    if args.max_n is not None and len(X_true) > int(args.max_n):
        take = rng.choice(len(X_true), size=int(args.max_n), replace=False)
        take.sort()
        z_l, z_r = z_l[take], z_r[take]
        X_pick, X_true = X_pick[take], X_true[take]
        Cm = Cm[take]
        print(f"subsampled mid {len(take)}/{len(Xm)}", flush=True)
    _, knn_pb = cKDTree(Cm).query(Cm, k=int(args.nfs_k))
    knn_pb = np.asarray(knn_pb, dtype=np.int64).reshape(len(Cm), int(args.nfs_k))

    n = len(X_true)
    idx = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    print(
        f"device={device} n={n} genes={X_true.shape[1]} pick_mse="
        f"{float(((X_pick - X_true) ** 2).mean()):.5f}",
        flush=True,
    )

    model = PickResidualMLP(d=args.pca_dim, hidden=args.hidden).to(device)
    if args.freeze_scale is not None:
        with torch.no_grad():
            model.scale.fill_(float(args.freeze_scale))
        model.scale.requires_grad_(False)
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    comp = torch.as_tensor(pca.model.components_, device=device, dtype=torch.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca.save(args.out_dir / "pca.joblib")
    log = args.out_dir / "train.log"

    def batch_loss(ii):
        nb = knn_pb[ii]
        nodes = np.unique(np.concatenate([np.asarray(ii, dtype=np.int64), nb.ravel()]))
        loc = {int(v): j for j, v in enumerate(nodes)}
        pred_n, dz = model(
            torch.as_tensor(z_l[nodes], device=device, dtype=torch.float32),
            torch.as_tensor(z_r[nodes], device=device, dtype=torch.float32),
            torch.full((len(nodes), 1), float(args.w), device=device, dtype=torch.float32),
            torch.as_tensor(X_pick[nodes], device=device, dtype=torch.float32),
            comp,
        )
        b = np.asarray([loc[int(v)] for v in ii], dtype=np.int64)
        pred = pred_n[b]
        xt = torch.as_tensor(X_true[ii], device=device, dtype=torch.float32)
        mse = ((pred - xt) ** 2).mean()
        g = pred.size(1)
        gi = torch.randperm(g, device=device)[: min(64, g)]
        p = pred[:, gi] - pred[:, gi].mean(0, keepdim=True)
        t = xt[:, gi] - xt[:, gi].mean(0, keepdim=True)
        denom = float(max(len(ii) - 1, 1))
        cov = (((p.T @ p) / denom) - ((t.T @ t) / denom)).pow(2).mean()
        nb_loc = np.asarray([[loc[int(v)] for v in row] for row in nb], dtype=np.int64)
        pb_p = pred_n[torch.as_tensor(nb_loc, device=device)].mean(dim=1)
        pb_t = torch.as_tensor(X_true[nb], device=device, dtype=torch.float32).mean(dim=1)
        nfs = ((pb_p - pb_t) ** 2).mean()
        if len(ii) >= 2:
            j = torch.randperm(len(ii), device=device)
            dx = (pred - pred[j]).pow(2).mean(-1)
            dt = (xt - xt[j]).pow(2).mean(-1)
            var = ((dx - dt) / (dt.mean() + 1e-6)).pow(2).mean()
        else:
            var = pred.new_zeros(())
        l2 = dz[b].pow(2).mean()
        loss = (
            mse
            + float(args.cov_weight) * cov
            + float(args.nfs_weight) * nfs
            + float(args.var_weight) * var
            + float(args.l2_weight) * l2
        )
        return loss, mse, nfs, var

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_i)
        losses = []
        for s in range(0, len(tr_i), args.batch):
            ii = tr_i[s : s + args.batch]
            loss, mse, nfs, var = batch_loss(ii)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vloss, vmse, vnfs, vvar = batch_loss(val_i)
        msg = (
            f"epoch {epoch:03d}  train={np.mean(losses):.5f}  val={float(vloss):.5f}  "
            f"vmse={float(vmse):.5f}  vnfs={float(vnfs):.5f}  vvar={float(vvar):.5f}  "
            f"scale={float(model.scale.detach().cpu()):.4f}"
        )
        print(msg, flush=True)
        with log.open("a") as f:
            f.write(msg + "\n")
        if float(vloss) < best:
            best = float(vloss)
            ckpt = args.out_dir / "ckpt" / "resid.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "d": args.pca_dim,
                    "hidden": args.hidden,
                    "knn": k,
                    "nfs_k": int(args.nfs_k),
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
