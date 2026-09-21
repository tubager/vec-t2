#!/usr/bin/env python
"""Full-panel gene-program residual on a frozen-xyz backbone.

Not PCA decode. Not k16 independent genes. Residual is rank-r in gene space
(X += u(z_l,z_r) @ V), V from SVD of Hungarian(mid X) − backbone, so train and
infer share the same host. Hard variogram + gene-cov + mean lock.

  .venv/bin/python scripts/62_gene_program.py train \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --out-dir outputs/t2/embryo/gate725_program --rank 4 --epochs 40

  .venv/bin/python scripts/62_gene_program.py predict \\
    --ckpt-dir outputs/t2/embryo/gate725_program \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --alpha 0.25 --out outputs/t2/embryo/gate725_program/lo_pm_a025.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.flow import cov_frobenius_loss, get_device  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def _hungarian_mid(Cp, Cm, Xm, n_mid_sub: int, seed: int) -> np.ndarray:
    """Assign a mid expression vector to each backbone row (xyz Hungarian)."""
    rng = np.random.default_rng(seed)
    if len(Cm) > n_mid_sub:
        take = rng.choice(len(Cm), size=n_mid_sub, replace=False)
        Cm, Xm = Cm[take], Xm[take]
    a = Cp.astype(np.float32)
    b = Cm.astype(np.float32)
    # subsample assignment: for n=5000 vs 5000, full cost is 25e6 — OK
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    cost = a2 + b2 - 2.0 * (a @ b.T)
    if cost.shape[0] == cost.shape[1]:
        ri, ci = linear_sum_assignment(cost)
        order = np.empty(len(a), dtype=np.int64)
        order[ri] = ci
        return Xm[order]
    # rectangular: each pred row takes its min mid (not a bijection)
    return Xm[np.argmin(cost, axis=1)]


def vario_loss(xp: torch.Tensor, xt: torch.Tensor, n_pairs: int = 4096, p: float = 0.5) -> torch.Tensor:
    if xp.size(0) < 2:
        return xp.new_zeros(())
    g = xp.size(1)
    i = torch.randint(0, g, (n_pairs,), device=xp.device)
    j = torch.randint(0, g, (n_pairs,), device=xp.device)
    va = ((xp[:, i] - xp[:, j]).abs() + 1e-4).pow(p).mean(0)
    vb = ((xt[:, i] - xt[:, j]).abs() + 1e-4).pow(p).mean(0)
    return ((va - vb) ** 2).mean()


class ProgramHead(nn.Module):
    def __init__(self, d: int, rank: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(d) + 1, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(rank)),
        )

    def forward(self, z_l, z_r, w):
        if w.ndim == 1:
            w = w[:, None]
        return self.net(torch.cat([z_l, z_r, w], dim=-1))


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
    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    Cp = np.asarray(pred.obsm["spatial_3D"], np.float64)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), args.rms), np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), args.rms), np.float64)
    Cm = np.asarray(scale_cloud(spatial_xyz(mid), args.rms), np.float64)
    Cq = np.asarray(scale_cloud(Cp, args.rms), np.float64)

    n_sub = min(int(args.n_mid), len(Cm))
    Y = _hungarian_mid(Cq, Cm, Xm, n_sub, args.seed)
    R = Y - Xp
    mu = R.mean(0)
    Rc = R - mu
    _, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    r = int(min(args.rank, len(S)))
    # Fold singular values into V and z-score the cell scores so the MLP stays O(1).
    std = np.maximum(S[:r] / np.sqrt(max(len(Rc), 1)), 1e-3)
    V = (Vt[:r] * std[:, None]).astype(np.float64)
    energy = float((S[:r] ** 2).sum() / max((S ** 2).sum(), 1e-12))
    u_true = Rc @ np.linalg.pinv(V)
    print(
        f"n={len(Xp)} G={Xp.shape[1]} rank={r} energy={energy:.3f} "
        f"mean|R|={np.mean(np.abs(R)):.4f}  mean|u|={np.mean(np.abs(u_true)):.4f}",
        flush=True,
    )

    pca = ExpressionPCA(n_components=args.pca_dim, batch_size=int(cfg["pca_batch"]))
    pca.fit([left, right, mid], n_per_stage=max(1, int(cfg["pca_fit_cells"]) // 3), seed=args.seed)
    zl_all, zr_all = pca.encode(Xl), pca.encode(Xr)
    k = int(args.knn)
    _, i0 = cKDTree(Cl).query(Cq, k=k)
    _, i1 = cKDTree(Cr).query(Cq, k=k)
    z_l = zl_all[np.asarray(i0, np.int64).reshape(len(Cq), k)].mean(1).astype(np.float32)
    z_r = zr_all[np.asarray(i1, np.int64).reshape(len(Cq), k)].mean(1).astype(np.float32)

    n = len(Xp)
    idx = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    model = ProgramHead(d=args.pca_dim, rank=r, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    V_t = torch.as_tensor(V, device=device, dtype=torch.float32)
    mu_t = torch.as_tensor(mu, device=device, dtype=torch.float32)
    Xp_t = torch.as_tensor(Xp, device=device, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, device=device, dtype=torch.float32)
    u_t = torch.as_tensor(u_true, device=device, dtype=torch.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca.save(args.out_dir / "pca.joblib")
    np.save(args.out_dir / "V.npy", V)
    np.save(args.out_dir / "mu.npy", mu)
    log = args.out_dir / "train.log"

    def batch_loss(ii):
        ii_t = torch.as_tensor(ii, device=device, dtype=torch.long)
        zl_b = torch.as_tensor(z_l[ii], device=device, dtype=torch.float32)
        zr_b = torch.as_tensor(z_r[ii], device=device, dtype=torch.float32)
        ww = torch.full((len(ii), 1), float(args.w), device=device, dtype=torch.float32)
        u = model(zl_b, zr_b, ww)
        xhat = torch.clamp(Xp_t[ii_t] + u @ V_t + mu_t, min=0.0)
        yt = Y_t[ii_t]
        mse = ((xhat - yt) ** 2).mean()
        u_mse = ((u - u_t[ii_t]) ** 2).mean()
        var = vario_loss(xhat, yt)
        cov = cov_frobenius_loss(xhat, yt)
        mean_pen = ((xhat.mean(0) - yt.mean(0)) ** 2).mean()
        loss = (
            mse
            + float(args.u_weight) * u_mse
            + float(args.var_weight) * var
            + float(args.cov_weight) * cov
            + float(args.mean_weight) * mean_pen
        )
        return loss, mse, var, mean_pen

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_i)
        losses = []
        for s in range(0, len(tr_i), args.batch):
            ii = tr_i[s : s + args.batch]
            loss, mse, var, mean_pen = batch_loss(ii)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            vloss, vmse, vvar, vmean = batch_loss(val_i)
        msg = (
            f"epoch {epoch:03d}  train={np.mean(losses):.5f}  val={float(vloss):.5f}  "
            f"vmse={float(vmse):.5f}  vvar={float(vvar):.5f}  vmean={float(vmean):.5f}"
        )
        print(msg, flush=True)
        with log.open("a") as f:
            f.write(msg + "\n")
        if float(vloss) < best:
            best = float(vloss)
            ckpt = args.out_dir / "ckpt" / "program.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "d": args.pca_dim,
                    "rank": r,
                    "hidden": args.hidden,
                    "knn": args.knn,
                    "w_train": float(args.w),
                    "rms": float(args.rms),
                    "epoch": epoch,
                    "best_val": best,
                    "energy": energy,
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
    V = np.load(args.ckpt_dir / "V.npy")
    mu = np.load(args.ckpt_dir / "mu.npy")
    blob = torch.load(args.ckpt_dir / "ckpt" / "program.pt", map_location=device, weights_only=False)
    model = ProgramHead(d=int(blob["d"]), rank=int(blob["rank"]), hidden=int(blob.get("hidden", 128)))
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
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
    V_t = torch.as_tensor(V, device=device, dtype=torch.float32)
    mu_t = torch.as_tensor(mu, device=device, dtype=torch.float32)
    chunks = []
    with torch.no_grad():
        for s in range(0, len(Cq), 1024):
            e = min(s + 1024, len(Cq))
            u = model(
                torch.as_tensor(z_l[s:e], device=device, dtype=torch.float32),
                torch.as_tensor(z_r[s:e], device=device, dtype=torch.float32),
                torch.full((e - s, 1), float(args.w), device=device, dtype=torch.float32),
            )
            delta = u @ V_t
            if args.use_mu:
                delta = delta + mu_t
            chunks.append(delta.cpu().numpy())
    delta = np.concatenate(chunks, axis=0)
    a = float(np.clip(args.alpha, 0.0, 1.0))
    Xn = np.clip(Xp + a * delta, 0.0, None)
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    out.obsm["spatial_3D"] = C0
    assert np.array_equal(np.asarray(out.obsm["spatial_3D"], np.float32), C0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    print(
        f"wrote {args.out}  alpha={a:g} rank={len(V)} use_mu={args.use_mu}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  xyz bit-exact"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--pred", type=Path, required=True)
    tr.add_argument("--out-dir", type=Path, required=True)
    tr.add_argument("--rank", type=int, default=4)
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch", type=int, default=256)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--pca-dim", type=int, default=32)
    tr.add_argument("--hidden", type=int, default=128)
    tr.add_argument("--knn", type=int, default=8)
    tr.add_argument("--w", type=float, default=0.4)
    tr.add_argument("--rms", type=float, default=198.24)
    tr.add_argument("--n-mid", type=int, default=5000)
    tr.add_argument("--u-weight", type=float, default=0.25)
    tr.add_argument("--var-weight", type=float, default=5.0)
    tr.add_argument("--cov-weight", type=float, default=1.0)
    tr.add_argument("--mean-weight", type=float, default=2.0)
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
    pr.add_argument(
        "--use-mu",
        action="store_true",
        help="add SVD mean residual (LEAVE-OUT LEAK from Hungarian mid X). Do not board.",
    )
    pr.add_argument("--knn", type=int, default=8)
    pr.add_argument("--rms", type=float, default=198.24)
    pr.add_argument("--device", default=None)
    args = ap.parse_args()
    return cmd_train(args) if args.cmd == "train" else cmd_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
