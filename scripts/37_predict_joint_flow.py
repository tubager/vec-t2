#!/usr/bin/env python
"""Predict with joint flow + optional manifold projection onto real cells.

Leave-out E7.25 (fraction tau along E6.75→E8.0):
  .venv/bin/python scripts/37_predict_joint_flow.py \\
    --joint-dir outputs/t2/embryo/gate725_joint_v2 --target 7.25 --n 5000 \\
    --project pool --out outputs/t2/embryo/gate725_joint_v2/pred_lo_pool.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters  # noqa: E402
from t2.flow import euler_integrate, get_device, torch_pca_decode  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense, write_t2  # noqa: E402
from t2.joint_flow import (  # noqa: E402
    canonicalize_xyz_np,
    load_joint_ckpt,
    pack_state,
    project_x_to_pool,
    unpack_state,
)
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint-dir", type=Path, required=True)
    ap.add_argument("--setting", default="embryo")
    ap.add_argument("--target", type=float, default=7.25)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument(
        "--project",
        choices=["none", "pool", "decode"],
        default="pool",
        help="decode=continuous PCA decode; pool=NN onto real left∪right",
    )
    ap.add_argument("--pool-stages", nargs="*", default=None)
    ap.add_argument(
        "--include-mid",
        action="store_true",
        help="add E7.25 into projection pool (board only; leaks leave-out)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device(args.device) if args.device else get_device()
    rng = np.random.default_rng(args.seed)

    model, meta = load_joint_ckpt(args.joint_dir / "ckpt" / "joint.pt", device=device)
    z_dim = int(meta["z_dim"])
    t0 = float(meta["t0"])
    dt_full = float(meta["dt"])
    xyz_rms = float(meta.get("xyz_rms", 1.0))
    ids = {str(k): int(v) for k, v in dict(meta["cluster_ids"]).items()}

    spec = target_spec(args.setting, float(args.target), cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)

    left_name = str(meta["left"])
    right_name = str(meta["right"])
    left = stages[left_name]

    pca = ExpressionPCA.load(args.joint_dir / "pca.joblib")
    z0 = pca.encode_adata(left)
    xyz0 = canonicalize_xyz_np(spatial_xyz(left), xyz_rms)
    cl0 = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))

    n = int(args.n)
    take = rng.choice(len(z0), size=n, replace=len(z0) < n)
    z0, xyz0, cl0 = z0[take], xyz0[take], cl0[take]

    t_tgt = float(args.target)
    tau = float(np.clip((t_tgt - t0) / max(dt_full, 1e-8), 0.0, 1.0))
    dt = torch.full((n, 1), tau * dt_full, device=device, dtype=torch.float32)
    t_t = torch.full((n, 1), t0, device=device, dtype=torch.float32)
    cid = torch.as_tensor(
        np.array([ids.get(str(c), 0) for c in cl0], dtype=np.int64),
        device=device,
    )

    s0 = pack_state(
        torch.as_tensor(z0, device=device, dtype=torch.float32),
        torch.as_tensor(xyz0, device=device, dtype=torch.float32),
    )
    with torch.no_grad():
        s1 = euler_integrate(model, s0, t_t, dt, cid, steps=args.steps)
    z_hat, xyz_hat = unpack_state(s1, z_dim)
    z_np = z_hat.cpu().numpy()
    xyz_np = np.asarray(scale_cloud(xyz_hat.cpu().numpy(), float(args.rms)), dtype=np.float32)

    comp = torch.as_tensor(pca.model.components_, device=device, dtype=torch.float32)
    mean = torch.as_tensor(pca.model.mean_, device=device, dtype=torch.float32)
    X_dec = (
        torch_pca_decode(torch.as_tensor(z_np, device=device, dtype=torch.float32), comp, mean)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    if args.project in ("none", "decode"):
        X_out = X_dec
    else:
        pool_names = list(args.pool_stages) if args.pool_stages else [left_name, right_name]
        if args.include_mid and "E7.25" in stages and "E7.25" not in pool_names:
            pool_names.append("E7.25")
        pools = []
        for name in pool_names:
            a = stages[name]
            X = to_dense(a.X).astype(np.float32)
            genes = list(a.var_names)
            if genes != panel:
                idx = {g: i for i, g in enumerate(genes)}
                X = X[:, [idx[g] for g in panel]]
            pools.append(X)
        X_pool = np.concatenate(pools, axis=0)
        if len(X_pool) > 20000:
            X_pool = X_pool[rng.choice(len(X_pool), size=20000, replace=False)]
        X_out = project_x_to_pool(X_dec, X_pool)
        print(f"projected onto pool n={len(X_pool)} stages={pool_names}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_t2(X_out, xyz_np, panel, args.out)
    print(
        f"wrote {args.out} n={len(X_out)} tau={tau:.3f} project={args.project} "
        f"train_xyz_rms={xyz_rms} out_rms={float(np.sqrt((xyz_np.astype(float) ** 2).sum(1).mean())):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
