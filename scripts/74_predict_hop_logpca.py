#!/usr/bin/env python
"""Flow a real left-anchor cell a clock fraction of its trained hop; keep donor zeros.

Leave-out:
  .venv/bin/python scripts/74_predict_hop_logpca.py --ckpt-dir outputs/t2/embryo/gate725_hoplog \\
    --t 7.25 --left E6.75 --right E8.0 \\
    --pred outputs/t2/embryo/gate725_repair/k256.h5ad \\
    --out outputs/t2/embryo/gate725_hoplog/k256.h5ad
Board (adjacent hop, not 206 interpolant):
  .venv/bin/python scripts/74_predict_hop_logpca.py --ckpt-dir outputs/t2/embryo/board75_hoplog \\
    --t 7.5 --left E7.25 --right E8.0 \\
    --pred outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --out outputs/t2/queue/2026-09-20/209_embryo_hoplog128_e725_w033_on194_n5000.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.flow import get_device, load_velocity  # noqa: E402
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float32)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float32)


@torch.no_grad()
def _partial_euler(model, z, t0, dt_hop, cid, w, steps: int):
    """Integrate fraction w of a hop, always feeding the trained hop dt."""
    h = dt_hop / max(int(steps), 1)
    n_run = max(1, int(round(float(w) * int(steps))))
    z = z.clone()
    for i in range(n_run):
        t_i = t0 + i * h
        z = z + model(z, t_i, dt_hop, cid) * h
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--bidir", action="store_true", help="flow nearer L/R endpoint toward t*")
    ap.add_argument("--alpha", type=float, default=1.0, help="blend with pred: (1-a)*pred + a*flowed")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    blob = joblib.load(args.ckpt_dir / "pca.joblib")
    pca = blob["pca"]
    model, extra = load_velocity(args.ckpt_dir / "ckpt" / "flow.pt", device=device)
    panel = list(extra.get("panel") or blob["panel"])
    ids = dict(extra["cluster_to_id"])
    trained = set(extra.get("trained_clusters") or [])

    cfg = load_config()
    spec = target_spec(args.setting, float(args.t), cfg)
    stages = load_stages(args.setting, cfg, panel)
    times = stage_times(args.setting, cfg)
    left, right = stages[args.left], stages[args.right]
    usable = set(
        shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting)
    )
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))
    cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], args.setting))
    Yl = np.log1p(np.maximum(Xl, 0.0))
    Yr = np.log1p(np.maximum(Xr, 0.0))
    zl = pca.transform(Yl).astype(np.float32)
    zr = pca.transform(Yr).astype(np.float32)

    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.array(np.asarray(pred.obsm["spatial_3D"], dtype=np.float32), copy=True)
    Yp = np.log1p(np.maximum(Xp, 0.0))

    Xcat = np.concatenate([Xl, Xr], axis=0)
    cl_cat = np.concatenate([cl_l, cl_r])
    nn_c = NearestNeighbors(n_neighbors=min(int(args.knn), len(Xcat))).fit(Xcat)
    _, idxc = nn_c.kneighbors(Xp)
    names = []
    for row in idxc:
        labs = [str(cl_cat[j]) for j in row]
        vals, cnt = np.unique(labs, return_counts=True)
        names.append(str(vals[int(np.argmax(cnt))]))
    names = np.asarray(names)

    t0 = float(times[args.left])
    t1 = float(times[args.right])
    dt_hop = t1 - t0
    w = float(np.clip((float(args.t) - t0) / max(dt_hop, 1e-6), 0.0, 1.0))
    t0_t = torch.full((1, 1), t0, device=device)
    dt_t = torch.full((1, 1), dt_hop, device=device)

    Xn = np.array(Xp, copy=True, dtype=np.float32)
    n_used = 0
    model.eval()
    with torch.no_grad():
        for key in sorted(set(names) & (trained or usable) & usable):
            m = names == key
            if not m.any():
                continue
            if key not in ids:
                continue
            src_l = np.flatnonzero(cl_l == key)
            src_r = np.flatnonzero(cl_r == key)
            if len(src_l) < 8:
                continue
            zp = pca.transform(Yp[m]).astype(np.float32)
            nn_l = NearestNeighbors(n_neighbors=1).fit(zl[src_l])
            _, loc_l = nn_l.kneighbors(zp)
            loc_l = src_l[loc_l.ravel()]
            use_right = np.zeros(int(m.sum()), dtype=bool)
            loc_r = loc_l
            if args.bidir and len(src_r) >= 8:
                nn_r = NearestNeighbors(n_neighbors=1).fit(zr[src_r])
                _, loc_r = nn_r.kneighbors(zp)
                loc_r = src_r[loc_r.ravel()]
                d_l = np.linalg.norm(zp - zl[loc_l], axis=1)
                d_r = np.linalg.norm(zp - zr[loc_r], axis=1)
                use_right = d_r < d_l
            z_in = np.array(zl[loc_l], copy=True)
            donor = np.array(Xl[loc_l], copy=True)
            if args.bidir and use_right.any():
                z_in[use_right] = zr[loc_r[use_right]]
                donor[use_right] = Xr[loc_r[use_right]]
            cid = torch.full((len(z_in),), ids[key], device=device, dtype=torch.long)
            z1 = np.empty_like(z_in)
            left_m = ~use_right
            if left_m.any():
                z1[left_m] = (
                    _partial_euler(
                        model,
                        torch.as_tensor(z_in[left_m], device=device),
                        t0_t.expand(int(left_m.sum()), 1),
                        dt_t.expand(int(left_m.sum()), 1),
                        cid[left_m],
                        float(w),
                        args.steps,
                    )
                    .cpu()
                    .numpy()
                )
            if use_right.any():
                zr_t = torch.as_tensor(z_in[use_right], device=device)
                n_r = int(use_right.sum())
                h = dt_t[:1] / max(int(args.steps), 1)
                n_run = max(1, int(round((1.0 - float(w)) * int(args.steps))))
                t_i = t0 + dt_hop
                for i in range(n_run):
                    tt = torch.full((n_r, 1), float(t_i) - i * float(h.item()), device=device)
                    zr_t = zr_t - model(zr_t, tt, dt_t.expand(n_r, 1), cid[use_right]) * h
                z1[use_right] = zr_t.cpu().numpy()
            Yh = pca.inverse_transform(z1)
            Xh = np.clip(np.expm1(Yh), 0.0, None).astype(np.float32)
            Xh[donor <= 0] = 0.0
            Xn[m] = Xh
            n_used += int(m.sum())
    a = float(np.clip(args.alpha, 0.0, 1.0))
    if a < 1.0 - 1e-12:
        Xn = ((1.0 - a) * Xp + a * Xn).astype(np.float32)
        Xn[Xp <= 0] = 0.0

    out = pred.copy()
    if list(pred.var_names) == panel:
        out.X = Xn
    else:
        live = np.asarray(to_dense(pred.X), dtype=np.float32)
        idxg = {g: i for i, g in enumerate(panel)}
        for j, g in enumerate(pred.var_names):
            live[:, j] = Xn[:, idxg[g]]
        out.X = live
    out.obsm["spatial_3D"] = C0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"], dtype=np.float32), C0)
    print(
        f"hop-logpca t={args.t:g} {args.left}→{args.right} w={w:.3f} bidir={int(args.bidir)} a={a:.2f} "
        f"n={n_used} mae_vs_pred={float(np.abs(Xn - Xp).mean()):.4f} frac0={float((Xn <= 0).mean()):.3f}  {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
