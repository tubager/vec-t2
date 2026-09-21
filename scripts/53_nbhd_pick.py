#!/usr/bin/env python
"""Train / apply a discrete neighborhood ranker (one real cell, never a mix).

Leave-out train (E6.75+E8.0 neighborhoods → true E7.25 cells):
  .venv/bin/python scripts/53_nbhd_pick.py train \\
    --out-dir outputs/t2/embryo/gate725_pick --epochs 30

Predict both native (donor xyz) and freeze (prior xyz bit-exact):
  .venv/bin/python scripts/53_nbhd_pick.py predict \\
    --ckpt-dir outputs/t2/embryo/gate725_pick \\
    --prior outputs/t2/embryo/gate725/slice_spatial_k0.4_b6.h5ad \\
    --left E6.75 --right E8.0 --w 0.4 --recipe rank \\
    --out-dir outputs/t2/embryo/gate725_pick/lo_k04b6

snapmean is the no-train discrete baseline (neighbor closest to local PCA lerp).
Do not submit unless scripts/54_dual_gate.py PASSes both gates.
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

from t2.flow import get_device  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense, write_t2  # noqa: E402
from t2.nbhd_pick import (  # noqa: E402
    NeighborRanker,
    freeze_x_onto,
    gather_picks,
    load_ranker,
    neighbor_labels,
    save_ranker,
    snapmean_indices,
)
from t2.ot import _local_pick_rows, _progress_axis  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def _knn_pack(Cl, Cr, Cq, zl, zr, k):
    k0 = min(int(k), len(Cl))
    k1 = min(int(k), len(Cr))
    _, i0 = cKDTree(Cl).query(Cq, k=k0)
    _, i1 = cKDTree(Cr).query(Cq, k=k1)
    i0 = np.asarray(i0, dtype=np.int64).reshape(len(Cq), k0)
    i1 = np.asarray(i1, dtype=np.int64).reshape(len(Cq), k1)
    z_l = zl[i0].mean(axis=1).astype(np.float32)
    z_r = zr[i1].mean(axis=1).astype(np.float32)
    z_nb = np.concatenate([zl[i0], zr[i1]], axis=1).astype(np.float32)
    return i0, i1, z_l, z_r, z_nb


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
    Cl = np.asarray(scale_cloud(spatial_xyz(left), args.rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), args.rms), dtype=np.float64)
    Cm = np.asarray(scale_cloud(spatial_xyz(mid), args.rms), dtype=np.float64)

    pca = ExpressionPCA(n_components=args.pca_dim, batch_size=int(cfg["pca_batch"]))
    pca.fit([left, right, mid], n_per_stage=max(1, int(cfg["pca_fit_cells"]) // 3), seed=args.seed)
    zl, zr = pca.encode(Xl), pca.encode(Xr)
    i0, i1, z_l, z_r, z_nb = _knn_pack(Cl, Cr, Cm, zl, zr, args.knn)
    X_nb = np.concatenate([Xl[i0], Xr[i1]], axis=1).astype(np.float32)
    y = neighbor_labels(X_nb, Xm)
    n_nb = int(z_nb.shape[1])
    acc0 = float((y == snapmean_indices(z_nb, z_l, z_r, args.w)).mean())
    print(
        f"device={device} n={len(y)} knn={args.knn} n_nb={n_nb} "
        f"snapmean_hit={acc0:.3f} labels={np.bincount(y, minlength=n_nb).tolist()}",
        flush=True,
    )

    n = len(y)
    idx = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_i, tr_i = idx[:n_val], idx[n_val:]
    model = NeighborRanker(d=args.pca_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca.save(args.out_dir / "pca.joblib")
    log = args.out_dir / "train.log"

    def batch_stats(ii):
        zl_b = torch.as_tensor(z_l[ii], device=device, dtype=torch.float32)
        zr_b = torch.as_tensor(z_r[ii], device=device, dtype=torch.float32)
        znb = torch.as_tensor(z_nb[ii], device=device, dtype=torch.float32)
        yy = torch.as_tensor(y[ii], device=device, dtype=torch.long)
        ww = torch.full((len(ii), 1), float(args.w), device=device, dtype=torch.float32)
        logits = model(zl_b, zr_b, ww, znb)
        loss = torch.nn.functional.cross_entropy(logits, yy)
        acc = float((logits.argmax(-1) == yy).float().mean().detach().cpu())
        return loss, acc

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_i)
        losses, accs = [], []
        for s in range(0, len(tr_i), args.batch):
            ii = tr_i[s : s + args.batch]
            loss, acc = batch_stats(ii)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(acc)
        model.eval()
        with torch.no_grad():
            vloss, vacc = batch_stats(val_i)
        msg = (
            f"epoch {epoch:03d}  train={np.mean(losses):.5f}  acc={np.mean(accs):.3f}  "
            f"val={float(vloss):.5f}  vacc={vacc:.3f}"
        )
        print(msg, flush=True)
        with log.open("a") as f:
            f.write(msg + "\n")
        if float(vloss) < best:
            best = float(vloss)
            save_ranker(
                args.out_dir / "ckpt" / "ranker.pt",
                model,
                {
                    "d": args.pca_dim,
                    "hidden": args.hidden,
                    "knn": args.knn,
                    "w_train": float(args.w),
                    "rms": float(args.rms),
                    "epoch": epoch,
                    "best_val": best,
                    "n_nb": n_nb,
                },
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

    prior = ad.read_h5ad(args.prior)
    C_prior = np.array(np.asarray(prior.obsm["spatial_3D"], dtype=np.float32), copy=True)
    rms = float(args.rms)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), rms), dtype=np.float64)
    Cq = np.asarray(scale_cloud(C_prior, rms), dtype=np.float64)

    knn = int(args.knn)
    model = None
    if args.recipe == "rank":
        if args.ckpt_dir is None:
            raise SystemExit("rank recipe needs --ckpt-dir")
        pca = ExpressionPCA.load(args.ckpt_dir / "pca.joblib")
        model, meta = load_ranker(args.ckpt_dir / "ckpt" / "ranker.pt", device=device)
        knn = int(meta.get("knn", knn))
    else:
        pca_path = ROOT / "outputs/t2/embryo/gate725/pca.joblib"
        if args.ckpt_dir is not None and (args.ckpt_dir / "pca.joblib").exists():
            pca_path = args.ckpt_dir / "pca.joblib"
        pca = ExpressionPCA.load(pca_path)

    zl, zr = pca.encode(Xl), pca.encode(Xr)
    i0, i1, z_l, z_r, z_nb = _knn_pack(Cl, Cr, Cq, zl, zr, knn)

    k0 = int(i0.shape[1])
    rng = np.random.default_rng(args.seed)
    if args.recipe == "rank":
        picks = []
        bs = 1024
        with torch.no_grad():
            for s in range(0, len(Cq), bs):
                e = min(s + bs, len(Cq))
                logits = model(
                    torch.as_tensor(z_l[s:e], device=device, dtype=torch.float32),
                    torch.as_tensor(z_r[s:e], device=device, dtype=torch.float32),
                    torch.full((e - s, 1), float(args.w), device=device, dtype=torch.float32),
                    torch.as_tensor(z_nb[s:e], device=device, dtype=torch.float32),
                )
                picks.append(logits.cpu().numpy())
        logits_np = np.concatenate(picks, axis=0)
        if args.side_clock:
            take_right = rng.random(len(Cq)) < float(args.w)
            indices = np.empty(len(Cq), dtype=np.int64)
            indices[~take_right] = logits_np[~take_right, :k0].argmax(axis=1)
            indices[take_right] = k0 + logits_np[take_right, k0:].argmax(axis=1)
        else:
            indices = logits_np.argmax(axis=1).astype(np.int64)
        X, xyz_own = gather_picks(indices, i0, i1, Xl, Xr, Cl.astype(np.float32), Cr.astype(np.float32))
    elif args.recipe == "progress":
        prog = _progress_axis(zl, zr)
        s0 = np.asarray(prog[0], dtype=np.float64) if prog is not None else np.zeros(len(Xl))
        s1 = np.asarray(prog[1], dtype=np.float64) if prog is not None else np.ones(len(Xr))
        X, xyz_own = _local_pick_rows(
            Cq, Cl, Xl, s0, Cr, Xr, s1, float(args.w), rng, knn=knn, pool=not args.side_clock,
            return_xyz=True,
        )
        indices = np.zeros(len(X), dtype=np.int64)
        n_right = None
    else:
        indices = snapmean_indices(z_nb, z_l, z_r, float(args.w))
        if args.side_clock:
            take_right = rng.random(len(Cq)) < float(args.w)
            lerp = (1.0 - float(args.w)) * z_l + float(args.w) * z_r
            d2 = ((z_nb - lerp[:, None, :]) ** 2).sum(axis=2)
            indices = np.empty(len(Cq), dtype=np.int64)
            indices[~take_right] = d2[~take_right, :k0].argmin(axis=1)
            indices[take_right] = k0 + d2[take_right, k0:].argmin(axis=1)
        X, xyz_own = gather_picks(indices, i0, i1, Xl, Xr, Cl.astype(np.float32), Cr.astype(np.float32))

    n_right = int((indices >= i0.shape[1]).sum())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    native_path = args.out_dir / "native.h5ad"
    freeze_path = args.out_dir / "freeze.h5ad"
    write_t2(X, xyz_own, panel, native_path)
    frozen = freeze_x_onto(X, C_prior, prior)
    if list(frozen.var_names) != list(panel) or int(frozen.n_vars) != int(X.shape[1]):
        frozen = ad.AnnData(np.asarray(X, dtype=np.float32))
        frozen.var_names = panel
        frozen.obs_names = prior.obs_names.copy()
        frozen.obsm["spatial_3D"] = np.array(
            np.asarray(prior.obsm["spatial_3D"], dtype=np.float32), copy=True
        )
    frozen.write_h5ad(freeze_path, compression="gzip")
    print(
        f"wrote {native_path} (donor xyz) and {freeze_path} (prior xyz bit-exact)  "
        f"recipe={args.recipe} n={len(X)} right={n_right}/{len(X)} "
        f"anchors={args.left}+{args.right} w={args.w:.3f} knn={knn}"
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
    tr.add_argument("--w", type=float, default=0.4)
    tr.add_argument("--rms", type=float, default=198.24)
    tr.add_argument("--val-frac", type=float, default=0.2)
    tr.add_argument("--device", default=None)
    tr.add_argument("--seed", type=int, default=0)

    pr = sub.add_parser("predict")
    pr.add_argument("--out-dir", type=Path, required=True)
    pr.add_argument("--prior", type=Path, required=True)
    pr.add_argument("--ckpt-dir", type=Path, default=None)
    pr.add_argument("--recipe", choices=["rank", "snapmean", "progress"], default="rank")
    pr.add_argument(
        "--side-clock",
        action="store_true",
        help="Bernoulli(w) chooses left vs right neighborhood, then argmax inside the side. "
        "Blocks collapse-to-left (ranker labels are ~99% left on E7.25).",
    )
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--left", default="E6.75")
    pr.add_argument("--right", default="E8.0")
    pr.add_argument("--w", type=float, default=0.4)
    pr.add_argument("--knn", type=int, default=8)
    pr.add_argument("--rms", type=float, default=198.24)
    pr.add_argument("--device", default=None)

    args = ap.parse_args()
    if args.cmd == "train":
        return cmd_train(args)
    return cmd_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
