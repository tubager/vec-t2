#!/usr/bin/env python
"""Fit T2 panel PCA and train OT-CFM per setting (independent of T1).

Default is now *raw* (non-residual) transport: left latents → right latents.
Each epoch walks every shared cluster on every hop (the 2026-09-03 trainer
did one cluster / epoch and never drove CFM loss down).

Leave-out gates:
  embryo E7.25:  --hop E6.75,E8.0 --exclude E7.25 --out-dir outputs/t2/embryo/gate725
  heart W2:      --hop E8.25,E9.5 --exclude E8.75 --out-dir outputs/t2/heart/gate_w2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import cluster_id_map, labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.flow import (  # noqa: E402
    Velocity,
    cfm_loss,
    flow_artifact_paths,
    get_device,
    hops_for,
    mmd_unbiased,
    save_checkpoint,
    t_anchor_for,
)
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
from t2.io import mean_X, to_dense  # noqa: E402
from t2.ot import minibatch_ot_pairs  # noqa: E402
from t2.paths import ensure_out_dirs, load_config, pred_dir  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def _t_scale(setting: str, cfg: dict) -> float:
    fcfg = cfg.get("flow") or {}
    key = "t_scale_embryo" if setting == "embryo" else "t_scale_heart"
    default = 2.0 if setting == "embryo" else 4.0
    return float(fcfg.get(key) or default)


def _parse_hops(raw: list[str] | None) -> list[tuple[str, str]] | None:
    if not raw:
        return None
    out = []
    for item in raw:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise SystemExit(f"--hop expects LEFT,RIGHT, got {item!r}")
        out.append((parts[0], parts[1]))
    return out


def _fit_setting(
    setting: str,
    cfg: dict,
    epochs: int,
    mmd_w: float,
    device: torch.device,
    seed: int,
    *,
    hop_pairs: list[tuple[str, str]] | None,
    exclude: list[str],
    residual: bool,
    out_dir: Path | None,
) -> None:
    rng = np.random.default_rng(seed)
    dummy_t = 7.5 if setting == "embryo" else 8.5
    spec = target_spec(setting, dummy_t, cfg)
    panel = load_panel_for(spec, cfg)
    stages = load_stages(setting, cfg, panel)
    times = stage_times(setting, cfg)
    exclude_set = {str(s) for s in exclude}
    for name in exclude_set:
        if name in stages:
            del stages[name]
            print(f"  excluded stage {name} from PCA/hops")
    names = list(stages)
    if not names:
        raise RuntimeError(f"no stages left for setting={setting}")
    n_per = max(1, int(cfg["pca_fit_cells"]) // max(len(names), 1))

    pca = ExpressionPCA(n_components=int(cfg["pca_dim"]), batch_size=int(cfg["pca_batch"]))
    pca.fit([stages[k] for k in names], n_per_stage=n_per, seed=seed)
    x_chk = to_dense(stages[names[0]][: min(500, stages[names[0]].n_obs)].X)
    pca_mse, mean_mse = pca.reconstruction_mse(x_chk)
    print(f"  PCA recon MSE={pca_mse:.5f}  mean-baseline MSE={mean_mse:.5f}")
    if pca_mse >= mean_mse:
        print("  [warn] PCA reconstruction is not better than the mean; consider raising pca_dim")

    clip = float(cfg["shift"]["clip_min"])
    hops = []
    trained = set()
    for left, right in hops_for(setting, hop_pairs):
        if left not in stages or right not in stages:
            print(f"  [warn] skip hop {left}→{right} (missing stage)")
            continue
        delta = mean_X(stages[right]) - mean_X(stages[left])
        tag = "residual" if residual else "raw"
        print(f"  encoding {left} → {tag} {right} ...")
        z0 = pca.encode_adata(stages[left])
        z1 = (
            pca.encode_residual(stages[right], delta, clip_min=clip)
            if residual
            else pca.encode_adata(stages[right])
        )
        labels_l = stages[left].obs["celltype"]
        labels_r = stages[right].obs["celltype"]
        usable = shared_flow_clusters(labels_l, labels_r, setting)
        c0 = np.asarray(labels_to_clusters(labels_l, setting))
        c1 = np.asarray(labels_to_clusters(labels_r, setting))
        z0_by, z1_by = {}, {}
        keep = []
        for name in usable:
            a = z0[c0 == name]
            b = z1[c1 == name]
            print(f"    {left}→{right}  {name}: n0={len(a)} n1={len(b)}")
            if len(a) == 0 or len(b) == 0:
                continue
            z0_by[name] = a
            z1_by[name] = b
            keep.append(name)
        if not keep:
            print(f"  [warn] no shared flow clusters for {left}→{right}")
            continue
        hops.append(
            {
                "left": left,
                "right": right,
                "t0": float(times[left]),
                "dt": float(times[right]) - float(times[left]),
                "z0_by": z0_by,
                "z1_by": z1_by,
                "usable": keep,
            }
        )
        trained.update(keep)

    if not hops:
        raise RuntimeError(f"No OT-CFM hops for setting={setting}")

    ids = cluster_id_map(setting)
    fcfg = cfg.get("flow") or {}
    t_anchor = t_anchor_for(setting)
    t_scale = _t_scale(setting, cfg)
    model = Velocity(
        d=int(cfg["pca_dim"]),
        emb_dim=int(fcfg.get("emb_dim") or 16),
        n_clusters=len(ids),
        hidden=list(fcfg.get("hidden") or [128, 128]),
        t_anchor=t_anchor,
        t_scale=t_scale,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(fcfg.get("lr") or 1e-3))
    m_pairs = int(fcfg.get("batch_pairs") or 256)
    z_noise = float(fcfg.get("z_noise") or 0.05)

    units = [(h, c) for h in hops for c in h["usable"]]
    print(
        f"  training on {device} for {epochs} epochs  "
        f"hops={[(h['left'], h['right']) for h in hops]}  "
        f"steps/epoch={len(units)}  residual={residual}"
    )
    model.train()
    for epoch in range(1, epochs + 1):
        order = [units[i] for i in rng.permutation(len(units))]
        losses = []
        last = order[-1]
        for hop, cluster in order:
            z0, z1 = minibatch_ot_pairs(hop["z0_by"][cluster], hop["z1_by"][cluster], m_pairs, rng)
            z0_t = torch.from_numpy(z0).to(device)
            z1_t = torch.from_numpy(z1).to(device)
            n = z0_t.size(0)
            t0_t = torch.full((n, 1), hop["t0"], device=device, dtype=z0_t.dtype)
            dt_t = torch.full((n, 1), hop["dt"], device=device, dtype=z0_t.dtype)
            cid = torch.full((n,), ids[cluster], device=device, dtype=torch.long)
            loss = cfm_loss(model, z0_t, z1_t, t0_t, dt_t, cid, z_noise=z_noise)
            if mmd_w > 0:
                tau = torch.rand(n, 1, device=device)
                z_tau = (1 - tau) * z0_t + tau * z1_t
                loss = loss + mmd_w * mmd_unbiased(z_tau, z1_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            last = (hop, cluster)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            hop, cluster = last
            print(
                f"  epoch {epoch:4d}/{epochs}  {hop['left']}→{hop['right']}  "
                f"cluster={cluster:10s}  loss={float(np.mean(losses)):.5f}"
            )

    dest = Path(out_dir) if out_dir is not None else pred_dir(cfg, setting)
    dest.mkdir(parents=True, exist_ok=True)
    pca_path, ckpt_path = flow_artifact_paths(dest)
    pca.save(pca_path)
    save_checkpoint(
        ckpt_path,
        model,
        extra={
            "d": int(cfg["pca_dim"]),
            "emb_dim": int(fcfg.get("emb_dim") or 16),
            "n_clusters": len(ids),
            "hidden": list(fcfg.get("hidden") or [128, 128]),
            "t_anchor": t_anchor,
            "t_scale": t_scale,
            "setting": setting,
            "cluster_to_id": ids,
            "trained_clusters": sorted(trained),
            "hops": [(h["left"], h["right"]) for h in hops],
            "epochs": epochs,
            "residual": bool(residual),
            "velocity": True,
        },
    )
    print(f"  saved {pca_path}")
    print(f"  saved {ckpt_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", choices=["embryo", "heart", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--mmd-weight", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--hop",
        action="append",
        default=None,
        help="LEFT,RIGHT hop (repeatable). Default: all adjacent hops for the setting.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=(),
        help="stage names to drop from PCA fit and hops (leave-out gates).",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="encode right as clip(x−Δ) (old 09-03 recipe). Default: raw latents.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="write pca.joblib + ckpt/flow.pt here (default: outputs/t2/<setting>/)",
    )
    args = parser.parse_args()

    cfg = load_config()
    ensure_out_dirs(cfg)
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    fcfg = cfg.get("flow") or {}
    epochs = args.epochs if args.epochs is not None else int(fcfg.get("epochs") or 200)
    mmd_w = args.mmd_weight if args.mmd_weight is not None else float(fcfg.get("mmd_weight") or 0.0)
    device = torch.device(args.device) if args.device else get_device()
    hop_pairs = _parse_hops(args.hop)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if args.setting == "both" and out_dir is not None:
        raise SystemExit("--out-dir requires a single --setting")

    settings = ("embryo", "heart") if args.setting == "both" else (args.setting,)
    for setting in settings:
        print(f"=== {setting} OT-CFM ===")
        _fit_setting(
            setting,
            cfg,
            epochs=epochs,
            mmd_w=mmd_w,
            device=device,
            seed=seed,
            hop_pairs=hop_pairs,
            exclude=list(args.exclude),
            residual=bool(args.residual),
            out_dir=out_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
