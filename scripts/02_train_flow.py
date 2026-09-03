#!/usr/bin/env python
"""Fit composition + PCA and train residual OT-CFM on E8.5 -> E9.5."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t1.clusters import FLOW_CLUSTERS, cluster_id_map, labels_to_clusters  # noqa: E402
from t1.composition import CompositionModel  # noqa: E402
from t1.flow import Velocity, cfm_loss, get_device, save_checkpoint  # noqa: E402
from t1.infer import encode_stage_residual  # noqa: E402
from t1.io import align_to_panel, load_adata, load_panel, mean_X  # noqa: E402
from t1.losses import mmd_unbiased  # noqa: E402
from t1.ot import minibatch_ot_pairs  # noqa: E402
from t1.paths import ROOT as PROJ, ensure_out_dirs, load_config, resolve  # noqa: E402
from t1.pca import ExpressionPCA  # noqa: E402


def _panel(cfg, adata):
    panel = load_panel(resolve(cfg, "panel")) or load_panel(resolve(cfg, "panel_fallback"))
    return panel or list(adata.var_names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--mmd-weight", type=float, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config()
    ensure_out_dirs(cfg)
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    a85 = load_adata(PROJ / cfg["paths"]["adata_85"])
    a95 = load_adata(PROJ / cfg["paths"]["adata_95"])
    panel = _panel(cfg, a85)
    a85 = align_to_panel(a85, panel)
    a95 = align_to_panel(a95, panel)

    tcfg = cfg["time"]
    ccfg = cfg["composition"]
    composition = CompositionModel(
        t_85=tcfg["t_85"],
        t_95=tcfg["t_95"],
        rho=ccfg["rho"],
        pi_max=ccfg["pi_max"],
        eps=ccfg["eps"],
    ).fit(a85.obs["celltype"], a95.obs["celltype"])
    composition.save(resolve(cfg, "composition"))
    print("composition π(9.5):", {k: round(v, 4) for k, v in composition.probs(9.5).items() if v > 1e-4})
    print("composition π(10.5):", {k: round(v, 4) for k, v in composition.probs(10.5).items() if v > 1e-4})

    n_per = int(cfg["pca_fit_cells"]) // 2
    pca = ExpressionPCA(n_components=int(cfg["pca_dim"]), batch_size=int(cfg["pca_batch"]))
    pca.fit([a85, a95], n_per_stage=n_per, seed=seed)
    pca.save(resolve(cfg, "pca"))
    x_chk = np.asarray(a85[:500].X.toarray() if hasattr(a85[:500].X, "toarray") else a85[:500].X)
    pca_mse, mean_mse = pca.reconstruction_mse(x_chk.astype(np.float32))
    print(f"PCA recon MSE={pca_mse:.5f}  mean-baseline MSE={mean_mse:.5f}")
    if pca_mse >= mean_mse:
        print("[warn] PCA reconstruction is not better than the mean; consider raising pca_dim")

    delta = mean_X(a95) - mean_X(a85)
    np.save(PROJ / "outputs/t1/delta.npy", delta)

    print("encoding E8.5 ...")
    z85 = pca.encode_adata(a85)
    print("encoding residual E9.5 ...")
    z95_res = encode_stage_residual(a95, pca, delta, clip_min=float(cfg["shift"]["clip_min"]))
    z95 = pca.encode_adata(a95)
    np.save(PROJ / "outputs/t1/z_85.npy", z85)
    np.save(PROJ / "outputs/t1/z_95.npy", z95)
    np.save(PROJ / "outputs/t1/z_95_residual.npy", z95_res)

    c85 = np.array(labels_to_clusters(a85.obs["celltype"]))
    c95 = np.array(labels_to_clusters(a95.obs["celltype"]))
    ids = cluster_id_map()

    z0_by, z1_by = {}, {}
    for name in FLOW_CLUSTERS:
        z0_by[name] = z85[c85 == name]
        z1_by[name] = z95_res[c95 == name]
        print(f"  flow cluster {name}: n0={len(z0_by[name])} n1={len(z1_by[name])}")
        if len(z0_by[name]) == 0 or len(z1_by[name]) == 0:
            print(f"  [warn] skip {name} (empty side)")

    usable = [k for k in FLOW_CLUSTERS if len(z0_by[k]) and len(z1_by[k])]
    if not usable:
        raise RuntimeError("No flow clusters with cells on both stages")

    fcfg = cfg["flow"]
    epochs = args.epochs if args.epochs is not None else int(fcfg["epochs"])
    mmd_w = args.mmd_weight if args.mmd_weight is not None else float(fcfg["mmd_weight"])
    device = torch.device(args.device) if args.device else get_device()
    print(f"training on {device} for {epochs} epochs, mmd_weight={mmd_w}")

    model = Velocity(
        d=int(cfg["pca_dim"]),
        emb_dim=int(fcfg["emb_dim"]),
        n_clusters=len(ids),
        hidden=list(fcfg["hidden"]),
        t_anchor=float(tcfg["t_85"]),
        t_scale=4.0,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(fcfg["lr"]))
    m_pairs = int(fcfg["batch_pairs"])
    t0 = float(tcfg["t_85"])
    dt = float(fcfg["delta_t_unit"])
    z_noise = float(fcfg["z_noise"])

    model.train()
    for epoch in range(1, epochs + 1):
        cluster = usable[int(rng.integers(len(usable)))]
        z0, z1 = minibatch_ot_pairs(z0_by[cluster], z1_by[cluster], m_pairs, rng)
        z0_t = torch.from_numpy(z0).to(device)
        z1_t = torch.from_numpy(z1).to(device)
        n = z0_t.size(0)
        t0_t = torch.full((n, 1), t0, device=device, dtype=z0_t.dtype)
        dt_t = torch.full((n, 1), dt, device=device, dtype=z0_t.dtype)
        cid = torch.full((n,), ids[cluster], device=device, dtype=torch.long)
        loss = cfm_loss(model, z0_t, z1_t, t0_t, dt_t, cid, z_noise=z_noise)
        if mmd_w > 0:
            # cheap regularizer: push one Euler-free linear mix toward target z
            tau = torch.rand(n, 1, device=device)
            z_tau = (1 - tau) * z0_t + tau * z1_t
            loss = loss + mmd_w * mmd_unbiased(z_tau, z1_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}/{epochs}  cluster={cluster:8s}  loss={float(loss):.5f}")

    save_checkpoint(
        resolve(cfg, "ckpt"),
        model,
        extra={
            "d": int(cfg["pca_dim"]),
            "emb_dim": int(fcfg["emb_dim"]),
            "n_clusters": len(ids),
            "hidden": list(fcfg["hidden"]),
            "t_anchor": float(tcfg["t_85"]),
            "t_scale": 4.0,
            "cluster_to_id": ids,
            "trained_clusters": usable,
            "epochs": epochs,
        },
    )
    print(f"saved {resolve(cfg, 'ckpt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
