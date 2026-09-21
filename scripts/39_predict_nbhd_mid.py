#!/usr/bin/env python
"""Apply neighborhood mid mapper onto a locked xyz template (bit-exact).

Leave-out (anchors E6.75+E8.0, template=band/mix, w=0.4):
  .venv/bin/python scripts/39_predict_nbhd_mid.py \\
    --nbhd-dir outputs/t2/embryo/gate725_nbhd \\
    --live outputs/t2/embryo/gate725/sweep_widestrat/bandX_on_spatial_k0.40_b6.h5ad \\
    --left E6.75 --right E8.0 --w 0.4 \\
    --out outputs/t2/embryo/gate725_nbhd/pred_lo_band.h5ad

Board (anchors E7.25+E8.0, template=09a, w=1/3):
  .venv/bin/python scripts/39_predict_nbhd_mid.py \\
    --nbhd-dir outputs/t2/embryo/gate725_nbhd \\
    --live outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --left E7.25 --right E8.0 --w 0.333 \\
    --out outputs/t2/queue/2026-09-15/110_....h5ad
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

from t2.flow import get_device, torch_pca_decode  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.nbhd_mid import bit_exact_xyz_from, load_nbhd  # noqa: E402
from t2.paths import load_config  # noqa: E402
from t2.pca import ExpressionPCA  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbhd-dir", type=Path, required=True)
    ap.add_argument("--live", type=Path, required=True, help="xyz template (bit-exact kept)")
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=1.0, help="blend with live X: (1-a)*live + a*pred")
    ap.add_argument("--project-pool", action="store_true",
                    help="NN-project decoded X onto left∪right real cells (coexpression lock)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not (0.0 < float(args.alpha) <= 1.0):
        raise SystemExit("--alpha in (0,1]")

    cfg = load_config()
    device = torch.device(args.device) if args.device else get_device()
    model, meta = load_nbhd(args.nbhd_dir / "ckpt" / "nbhd.pt", device=device)
    knn = int(meta.get("knn", 8))
    rms = float(meta.get("rms", 198.24))
    pca = ExpressionPCA.load(args.nbhd_dir / "pca.joblib")

    spec = target_spec("embryo", 7.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("embryo", cfg, panel)
    left, right = stages[args.left], stages[args.right]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    Cl = np.asarray(scale_cloud(spatial_xyz(left), rms), dtype=np.float64)
    Cr = np.asarray(scale_cloud(spatial_xyz(right), rms), dtype=np.float64)
    zl, zr = pca.encode(Xl), pca.encode(Xr)

    live = ad.read_h5ad(args.live)
    Clive = bit_exact_xyz_from(live.obsm["spatial_3D"])
    # query in same RMS frame as training
    Cq = np.asarray(scale_cloud(Clive, rms), dtype=np.float64)
    tree_l, tree_r = cKDTree(Cl), cKDTree(Cr)
    _, i_l = tree_l.query(Cq, k=knn)
    _, i_r = tree_r.query(Cq, k=knn)
    z_l = zl[i_l].mean(axis=1)
    z_r = zr[i_r].mean(axis=1)

    with torch.no_grad():
        pred_z = model(
            torch.as_tensor(z_l, device=device, dtype=torch.float32),
            torch.as_tensor(z_r, device=device, dtype=torch.float32),
            torch.full((len(z_l), 1), float(args.w), device=device, dtype=torch.float32),
        )
        comp = torch.as_tensor(pca.model.components_, device=device, dtype=torch.float32)
        mean = torch.as_tensor(pca.model.mean_, device=device, dtype=torch.float32)
        X_pred = torch_pca_decode(pred_z, comp, mean).cpu().numpy().astype(np.float32)

    if args.project_pool:
        from t2.joint_flow import project_x_to_pool

        pool = np.concatenate([Xl, Xr], axis=0)
        if len(pool) > 20000:
            rng = np.random.default_rng(0)
            pool = pool[rng.choice(len(pool), size=20000, replace=False)]
        X_pred = project_x_to_pool(X_pred, pool)

    X_live = _align(
        np.asarray(live.X.todense() if hasattr(live.X, "todense") else live.X, dtype=np.float32),
        live.var_names,
        panel,
    )
    a = float(args.alpha)
    X_out = (1.0 - a) * X_live + a * X_pred
    X_out = np.clip(X_out, 0.0, None).astype(np.float32)

    out = live.copy()
    out.X = X_out
    out.obsm["spatial_3D"] = Clive  # bit-exact
    assert np.array_equal(out.obsm["spatial_3D"], live.obsm["spatial_3D"].astype(np.float32)) or np.array_equal(
        out.obsm["spatial_3D"], np.asarray(live.obsm["spatial_3D"], dtype=np.float32)
    )
    # stricter: identical bytes after float32 cast
    assert np.array_equal(
        np.asarray(out.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(live.obsm["spatial_3D"], dtype=np.float32),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    print(
        f"wrote {args.out}  w={args.w} alpha={a} project={args.project_pool}  "
        f"anchors={args.left}+{args.right}  xyz bit-exact"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
