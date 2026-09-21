#!/usr/bin/env python
"""OT-CFM on log1p PCA (not raw panel PCA, not noise→z).

Leave-out (no E7.25):
  .venv/bin/python scripts/73_train_hop_logpca.py --exclude E7.25 --hop E6.75,E8.0 \\
    --out-dir outputs/t2/embryo/gate725_hoplog --epochs 40 --device mps
Board hops:
  .venv/bin/python scripts/73_train_hop_logpca.py --exclude NONE --hop E6.75,E7.25 --hop E7.25,E8.0 \\
    --out-dir outputs/t2/embryo/board75_hoplog --epochs 40 --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import cluster_id_map, labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.flow import (  # noqa: E402
    Velocity,
    cfm_loss,
    cov_frobenius_loss,
    euler_integrate_grad,
    gene_cov_loss,
    get_device,
    save_checkpoint,
    t_anchor_for,
)
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.ot import minibatch_ot_pairs  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def _parse_hops(raw: list[str] | None) -> list[tuple[str, str]]:
    out = []
    for item in raw or []:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise SystemExit(f"--hop expects LEFT,RIGHT, got {item!r}")
        out.append((parts[0], parts[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--hop", action="append", default=None)
    ap.add_argument("--exclude", nargs="*", default=("NONE",))
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cov-weight", type=float, default=0.5)
    ap.add_argument("--gene-cov-weight", type=float, default=1.0)
    ap.add_argument("--z-noise", type=float, default=0.05)
    ap.add_argument("--max-per-stage", type=int, default=8000)
    ap.add_argument("--holdout", type=float, default=0.12)
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
    times = stage_times(args.setting, cfg)
    for name in args.exclude:
        if not name or str(name).upper() == "NONE":
            continue
        if name in stages:
            del stages[name]
            print(f"excluded {name}")
    hops = _parse_hops(args.hop) or (
        [("E6.75", "E8.0")] if args.setting == "embryo" else [("E8.25", "E8.75")]
    )
    ids = cluster_id_map(args.setting)

    packed = {}
    Ys = []
    for tag, adata in stages.items():
        X = _align(to_dense(adata.X), adata.var_names, panel)
        cl = np.asarray(labels_to_clusters(adata.obs["celltype"], args.setting))
        idx = np.arange(len(X))
        if args.max_per_stage > 0 and len(idx) > args.max_per_stage:
            idx = rng.choice(idx, args.max_per_stage, replace=False)
        Y = np.log1p(np.maximum(X[idx], 0.0))
        packed[tag] = (Y, cl[idx], np.asarray(X[idx], dtype=np.float32))
        Ys.append(Y)
    Ycat = np.concatenate(Ys, axis=0)
    k = min(int(args.k), Ycat.shape[1], len(Ycat) - 1)
    pca = PCA(n_components=k, random_state=args.seed).fit(Ycat)

    # recon gate on a holdout slice of the first packed stage
    tag0 = next(iter(packed))
    Y0, _, X0 = packed[tag0]
    n_ho = max(32, int(len(Y0) * float(args.holdout)))
    ho = rng.choice(len(Y0), n_ho, replace=False)
    rec = np.clip(np.expm1(pca.inverse_transform(pca.transform(Y0[ho]))), 0.0, None)
    recz = rec.copy()
    recz[X0[ho] <= 0] = 0.0
    mean_mae = float(np.abs(recz - X0[ho]).mean())
    recon_pass = bool(mean_mae <= 0.05)
    print(f"pca k={k} expl={pca.explained_variance_ratio_.sum():.3f} keepz_mae={mean_mae:.4f} pass={recon_pass}")

    hop_pack = []
    trained = set()
    for left, right in hops:
        if left not in packed or right not in packed:
            print(f"skip hop {left}→{right}")
            continue
        Yl, cl, _ = packed[left]
        Yr, cr, _ = packed[right]
        usable = set(shared_flow_clusters(
            stages[left].obs["celltype"] if left in stages else [],
            stages[right].obs["celltype"] if right in stages else [],
            args.setting,
        ))
        # packed already filtered by max-per-stage; use labels present
        z0 = pca.transform(Yl).astype(np.float32)
        z1 = pca.transform(Yr).astype(np.float32)
        z0_by, z1_by = {}, {}
        for name in sorted(set(cl) | set(cr)):
            if name not in ids:
                continue
            a = z0[cl == name]
            b = z1[cr == name]
            if len(a) < 8 or len(b) < 8:
                continue
            if usable and name not in usable:
                continue
            z0_by[name] = a
            z1_by[name] = b
            trained.add(name)
            print(f"  {left}→{right} {name}: n0={len(a)} n1={len(b)}")
        if not z0_by:
            continue
        hop_pack.append(
            {
                "left": left,
                "right": right,
                "t0": float(times[left]),
                "dt": float(times[right]) - float(times[left]),
                "z0_by": z0_by,
                "z1_by": z1_by,
            }
        )
    if not hop_pack:
        raise SystemExit("no hops")

    t_anchor = t_anchor_for(args.setting)
    t_scale = 1.25 if args.setting == "embryo" else 0.5
    model = Velocity(
        d=k,
        emb_dim=16,
        n_clusters=len(ids),
        hidden=[int(args.hidden), int(args.hidden)],
        t_anchor=t_anchor,
        t_scale=t_scale,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    components_t = torch.tensor(np.asarray(pca.components_, dtype=np.float32), device=device)
    mean_t = torch.tensor(np.asarray(pca.mean_, dtype=np.float32), device=device)
    units = [(h, c) for h in hop_pack for c in h["z0_by"]]
    print(f"train {device} epochs={args.epochs} units/epoch={len(units)} hops={[ (h['left'], h['right']) for h in hop_pack ]}")
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for hop, cluster in [units[i] for i in rng.permutation(len(units))]:
            z0, z1 = minibatch_ot_pairs(hop["z0_by"][cluster], hop["z1_by"][cluster], args.batch, rng)
            z0_t = torch.from_numpy(z0).to(device)
            z1_t = torch.from_numpy(z1).to(device)
            n = z0_t.size(0)
            t0_t = torch.full((n, 1), hop["t0"], device=device, dtype=z0_t.dtype)
            dt_t = torch.full((n, 1), hop["dt"], device=device, dtype=z0_t.dtype)
            cid = torch.full((n,), ids[cluster], device=device, dtype=torch.long)
            loss = cfm_loss(model, z0_t, z1_t, t0_t, dt_t, cid, z_noise=float(args.z_noise))
            if args.cov_weight > 0 or args.gene_cov_weight > 0:
                z_hat = euler_integrate_grad(model, z0_t, t0_t, dt_t, cid, steps=4)
                if args.cov_weight > 0:
                    loss = loss + float(args.cov_weight) * cov_frobenius_loss(z_hat, z1_t)
                if args.gene_cov_weight > 0:
                    loss = loss + float(args.gene_cov_weight) * gene_cov_loss(
                        z_hat, z1_t, components_t, mean_t, n_genes=64
                    )
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={float(np.mean(losses)):.5f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pca": pca, "k": k, "panel": panel}, args.out_dir / "pca.joblib")
    save_checkpoint(
        args.out_dir / "ckpt" / "flow.pt",
        model,
        extra={
            "d": k,
            "emb_dim": 16,
            "n_clusters": len(ids),
            "hidden": [int(args.hidden), int(args.hidden)],
            "t_anchor": t_anchor,
            "t_scale": t_scale,
            "setting": args.setting,
            "cluster_to_id": ids,
            "trained_clusters": sorted(trained),
            "hops": [(h["left"], h["right"]) for h in hop_pack],
            "epochs": int(args.epochs),
            "residual": False,
            "velocity": True,
            "log1p": True,
            "recon_mae": mean_mae,
            "recon_pass": recon_pass,
            "panel": panel,
        },
    )
    (args.out_dir / "meta.json").write_text(
        json.dumps({"recon_mae": mean_mae, "recon_pass": recon_pass, "k": k, "hops": [(h["left"], h["right"]) for h in hop_pack]}, indent=2)
    )
    print(f"saved {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
