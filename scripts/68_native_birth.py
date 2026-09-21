#!/usr/bin/env python
"""Near-window native birth: p(X | t*, cluster, xyz) on occupancy sites.

Train (board anchors, E7.25 is allowed):
  .venv/bin/python scripts/66_train_cond_flow.py --arch vae \\
    --left E7.25 --right E8.0 --exclude NONE \\
    --out-dir outputs/t2/embryo/board75_condvae --epochs 50 \\
    --mse-weight 2 --kl-weight 0.02 --pair-weight 3 --no-mu-skip

Predict t=7.5, xyz from occupancy interpolant (not 194 freeze):
  .venv/bin/python scripts/68_native_birth.py \\
    --ckpt-dir outputs/t2/embryo/board75_condvae --t 7.5 \\
    --out outputs/t2/embryo/board75_condvae/native.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters  # noqa: E402
from t2.cond_flow import load_cond_flow  # noqa: E402
from t2.flow import get_device  # noqa: E402
from t2.geometry import rms_radius, scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense, write_t2  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.shape import occupancy_interpolate_cloud  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=float, default=7.5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--sites", type=Path, default=None, help="optional xyz skeleton (e.g. 194); default=occupancy interpolant")
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--occ-bins", type=int, default=48)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snap-sparsity", action="store_true")
    ap.add_argument("--match-mean", action="store_true", help="per-gene scale to L/R clock mix after snap")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    model, meta = load_cond_flow(args.ckpt_dir / "ckpt" / "cond_flow.pt", device=device)
    if str(meta.get("arch") or "") != "vae":
        raise SystemExit("68_native_birth expects a CondVAE checkpoint")
    panel = list(meta["panel"])
    ids = dict(meta["cluster_to_id"])
    usable = set(meta.get("usable") or [])
    rms = float(meta.get("rms") or args.rms)
    setting = str(meta.get("setting") or "embryo")
    t0 = float(meta.get("t0", 7.25))
    t1 = float(meta.get("t1", 8.0))
    w = float(np.clip((float(args.t) - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))

    cfg = load_config()
    spec = target_spec(setting, float(args.t), cfg)
    stages = load_stages(setting, cfg, panel)
    left = stages[str(meta["left"])]
    right = stages[str(meta["right"])]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), rms), dtype=np.float32)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), rms), dtype=np.float32)
    cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], setting))
    cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], setting))
    keep_l = np.array([str(c) in usable and str(c) in ids for c in cl_l])
    keep_r = np.array([str(c) in usable and str(c) in ids for c in cl_r])
    ref_X = np.concatenate([Xl[keep_l], Xr[keep_r]], axis=0)
    ref_C = np.concatenate([Cl[keep_l], Cr[keep_r]], axis=0)
    ref_cid = np.concatenate(
        [
            np.array([ids[str(c)] for c in cl_l[keep_l]], dtype=np.int64),
            np.array([ids[str(c)] for c in cl_r[keep_r]], dtype=np.int64),
        ]
    )

    rng = np.random.default_rng(args.seed)
    if args.sites is not None:
        import anndata as ad

        live = ad.read_h5ad(args.sites)
        C = np.array(np.asarray(live.obsm["spatial_3D"], dtype=np.float32), copy=True)
        print(f"xyz skeleton {args.sites} n={len(C)}")
    else:
        C = occupancy_interpolate_cloud(Cl, Cr, w, int(args.n), rng, n_bins=int(args.occ_bins))
        C = np.asarray(scale_cloud(C, rms), dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=1).fit(ref_C)
    _, idx = nn.kneighbors(C)
    take = idx[:, 0]
    Xd = ref_X[take]
    cid = ref_cid[take]
    xyz_n = (C / rms).astype(np.float32)

    # Reconstruction fuse at observed endpoints (no E7.5).
    model.eval()
    with torch.no_grad():
        sl = slice(0, min(1024, len(ref_X)))
        x = torch.as_tensor(ref_X[sl], device=device)
        c = torch.as_tensor(ref_cid[sl], device=device)
        zref = torch.as_tensor(ref_C[sl] / rms, device=device)
        mu_z, _ = model.encode(x, c, zref)
        n_l = int(keep_l.sum())
        t_nat = torch.as_tensor(
            np.concatenate(
                [
                    np.full(n_l, t0, dtype=np.float32),
                    np.full(len(ref_X) - n_l, t1, dtype=np.float32),
                ]
            )[sl],
            device=device,
        )
        xr = model.decode(mu_z, t_nat, c, zref)
        mae = float((xr - x).abs().mean().cpu())
        corr = float(
            np.corrcoef(xr.mean(0).cpu().numpy(), x.mean(0).cpu().numpy())[0, 1]
        )
        print(f"recon native-t mae={mae:.4f} mean-corr={corr:.3f}")

    Xn = np.empty((len(C), len(panel)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(C), args.batch):
            sl = slice(start, min(start + args.batch, len(C)))
            n = sl.stop - sl.start
            x_in = torch.as_tensor(Xd[sl], device=device)
            c = torch.as_tensor(cid[sl], device=device)
            zxyz = torch.as_tensor(xyz_n[sl], device=device)
            mu_z, _ = model.encode(x_in, c, zxyz)
            xg = model.decode(mu_z, x_in.new_full((n,), float(args.t)), c, zxyz)
            Xn[sl] = xg.clamp_min(0.0).cpu().numpy().astype(np.float32)

    if args.snap_sparsity:
        p = (1.0 - w) * (Xl <= 0).mean(0) + w * (Xr <= 0).mean(0)
        p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 0.99)
        for j in range(Xn.shape[1]):
            pj = float(p[j])
            if pj <= 0:
                continue
            thr = np.quantile(Xn[:, j], pj)
            Xn[Xn[:, j] <= thr, j] = 0.0
        Xn = Xn.astype(np.float32)

    if args.match_mean:
        mu = (1.0 - w) * Xl.mean(0) + w * Xr.mean(0)
        mu_g = np.maximum(Xn.mean(0), 1e-6)
        Xn = np.clip(Xn * (mu / mu_g).astype(np.float32), 0.0, None)

    write_t2(Xn, C, panel, args.out)
    if args.sites is not None:
        import anndata as ad

        chk = ad.read_h5ad(args.out)
        assert np.array_equal(
            np.asarray(chk.obsm["spatial_3D"], dtype=np.float32),
            C,
        )
    print(
        f"native-birth t={args.t:g} w={w:.3f} n={len(C)} rms={rms_radius(C):.2f} "
        f"mean|X|={float(Xn.mean()):.4f} frac0={float((Xn == 0).mean()):.3f} wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
