#!/usr/bin/env python
"""Train joint OT-CFM on (PCA-z || xyz) for embryo leave-out E6.75→E8.0.

  .venv/bin/python scripts/36_train_joint_flow.py \\
    --out-dir outputs/t2/embryo/gate725_joint --epochs 60 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import cluster_id_map, labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.flow import get_device, t_anchor_for  # noqa: E402
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
from t2.io import spatial_xyz  # noqa: E402
from t2.joint_flow import (  # noqa: E402
    JointVelocity,
    canonicalize_xyz_np,
    gene_cov_from_state,
    joint_cfm_loss,
    pack_state,
    save_joint_ckpt,
)
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", default="embryo", choices=["embryo"])
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--exclude", nargs="*", default=["E7.25"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pca-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, nargs=2, default=[192, 192])
    ap.add_argument("--gene-cov-weight", type=float, default=0.5)
    ap.add_argument("--gene-cov-genes", type=int, default=96)
    ap.add_argument("--xyz-rms", type=float, default=1.0,
                    help="canonicalize training xyz to this RMS (1.0 keeps z/xyz scales comparable)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device(args.device) if args.device else get_device()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    spec = target_spec(args.setting, 7.5, cfg)
    panel = load_panel_for(spec, cfg)
    stages = load_stages(args.setting, cfg, panel)
    times = stage_times(args.setting, cfg)
    for name in args.exclude:
        if name in stages:
            del stages[name]
            print(f"excluded {name}")

    left, right = stages[args.left], stages[args.right]
    t0 = float(times[args.left])
    dt = float(times[args.right]) - t0
    usable = shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting)
    ids = cluster_id_map(args.setting)
    cl0 = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))
    cl1 = np.asarray(labels_to_clusters(right.obs["celltype"], args.setting))

    pca = ExpressionPCA(n_components=int(args.pca_dim), batch_size=int(cfg["pca_batch"]))
    n_per = max(1, int(cfg["pca_fit_cells"]) // 2)
    pca.fit([left, right], n_per_stage=n_per, seed=args.seed)
    z0_all = pca.encode_adata(left)
    z1_all = pca.encode_adata(right)
    xyz0_all = canonicalize_xyz_np(spatial_xyz(left), args.xyz_rms)
    xyz1_all = canonicalize_xyz_np(spatial_xyz(right), args.xyz_rms)

    by = {}
    for name in usable:
        if name not in ids:
            continue
        a = np.where(cl0 == name)[0]
        b = np.where(cl1 == name)[0]
        if len(a) < 32 or len(b) < 32:
            print(f"  skip {name}: n0={len(a)} n1={len(b)}")
            continue
        by[name] = (a, b)
        print(f"  {name}: n0={len(a)} n1={len(b)}")
    if not by:
        raise SystemExit("no usable clusters")

    model = JointVelocity(
        z_dim=int(args.pca_dim),
        emb_dim=16,
        n_clusters=max(ids.values()) + 1,
        hidden=list(args.hidden),
        t_anchor=t_anchor_for(args.setting),
        t_scale=2.0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    comp = torch.as_tensor(pca.model.components_, device=device, dtype=torch.float32)
    mean = torch.as_tensor(pca.model.mean_, device=device, dtype=torch.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pca.save(args.out_dir / "pca.joblib")
    log_path = args.out_dir / "train.log"
    names = list(by.keys())

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        covs = []
        rng.shuffle(names)
        for name in names:
            a_idx, b_idx = by[name]
            n = min(args.batch, len(a_idx), len(b_idx))
            ia = rng.choice(a_idx, size=n, replace=len(a_idx) < n)
            ib = rng.choice(b_idx, size=n, replace=len(b_idx) < n)
            za = z0_all[ia]
            zb = z1_all[ib]
            xa = xyz0_all[ia]
            xb = xyz1_all[ib]
            # Hungarian re-pair on z; carry xyz with the same row permutation
            diff = za[:, None, :].astype(np.float64) - zb[None, :, :].astype(np.float64)
            cost = np.einsum("ijk,ijk->ij", diff, diff)
            ri, ci = linear_sum_assignment(cost)
            za, zb = za[ri], zb[ci]
            xa, xb = xa[ri], xb[ci]

            s0 = pack_state(
                torch.as_tensor(za, device=device, dtype=torch.float32),
                torch.as_tensor(xa, device=device, dtype=torch.float32),
            )
            s1 = pack_state(
                torch.as_tensor(zb, device=device, dtype=torch.float32),
                torch.as_tensor(xb, device=device, dtype=torch.float32),
            )
            cid = torch.full((n,), int(ids[name]), device=device, dtype=torch.long)
            t0_t = torch.full((n, 1), t0, device=device, dtype=torch.float32)
            dt_t = torch.full((n, 1), dt, device=device, dtype=torch.float32)

            loss = joint_cfm_loss(model, s0, s1, t0_t, dt_t, cid)
            if args.gene_cov_weight > 0:
                cov = gene_cov_from_state(
                    model, s0, t0_t, dt_t, cid, s1, comp, mean,
                    z_dim=int(args.pca_dim), steps=4, n_genes=args.gene_cov_genes,
                )
                loss = loss + float(args.gene_cov_weight) * cov
                covs.append(float(cov.detach().cpu()))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        msg = f"epoch {epoch:03d}  loss={np.mean(losses):.5f}"
        if covs:
            msg += f"  gene_cov={np.mean(covs):.5f}"
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = args.out_dir / "ckpt" / f"joint_ep{epoch:03d}.pt"
            save_joint_ckpt(
                ckpt,
                model,
                {
                    "z_dim": int(args.pca_dim),
                    "emb_dim": 16,
                    "n_clusters": max(ids.values()) + 1,
                    "hidden": list(args.hidden),
                    "t_anchor": t_anchor_for(args.setting),
                    "t_scale": 2.0,
                    "left": args.left,
                    "right": args.right,
                    "t0": t0,
                    "dt": dt,
                    "xyz_rms": float(args.xyz_rms),
                    "epoch": epoch,
                    "cluster_ids": dict(ids),
                },
            )
            save_joint_ckpt(args.out_dir / "ckpt" / "joint.pt", model, {
                "z_dim": int(args.pca_dim),
                "emb_dim": 16,
                "n_clusters": max(ids.values()) + 1,
                "hidden": list(args.hidden),
                "t_anchor": t_anchor_for(args.setting),
                "t_scale": 2.0,
                "left": args.left,
                "right": args.right,
                "t0": t0,
                "dt": dt,
                "xyz_rms": float(args.xyz_rms),
                "epoch": epoch,
                "cluster_ids": dict(ids),
            })
            print(f"  wrote {ckpt}", flush=True)

    print("done", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
