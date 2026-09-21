#!/usr/bin/env python
"""Mid-stage spatial teacher: pull locked-xyz live X toward E7.25 neighborhood.

Board path (E7.5): teacher = observed E7.25 (allowed training data). Leaves xyz
SEQUENCE of --live bit-exact. Do NOT score this recipe on leave-out E7.25 vs
true E7.25 (that leaks the teacher).

  .venv/bin/python scripts/35_mid_teacher_x.py \\
    --live outputs/t2/submit/T2_embryo_val_interp.h5ad \\
    --teacher data/E7.25.h5ad --alpha 0.20 --mode nn \\
    --out outputs/t2/embryo/gate725_teacher/board_a0.20_nn.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, target_spec  # noqa: E402
from t2.paths import load_config  # noqa: E402


def dense(X):
    return np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)


def align_genes(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    miss = [g for g in panel if g not in idx]
    if miss:
        raise SystemExit(f"teacher missing genes e.g. {miss[:3]}")
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, default=ROOT / "data" / "E7.25.h5ad")
    ap.add_argument("--alpha", type=float, required=True, help="blend weight toward teacher")
    ap.add_argument("--mode", choices=["nn", "mean", "pick"], default="nn")
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--rms", type=float, default=198.24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not (0.0 < float(args.alpha) <= 1.0):
        raise SystemExit("--alpha must be in (0,1]")

    cfg = load_config()
    live = ad.read_h5ad(args.live)
    teacher = ad.read_h5ad(args.teacher)
    panel = list(load_panel_for(target_spec("embryo", 7.5, cfg), cfg))

    Cl = np.asarray(live.obsm["spatial_3D"], dtype=np.float64)
    Ct = np.asarray(scale_cloud(teacher.obsm["spatial_3D"], float(args.rms)), dtype=np.float64)
    Xl = align_genes(dense(live.X), live.var_names, panel)
    Xt = align_genes(dense(teacher.X), teacher.var_names, panel)

    rng = np.random.default_rng(args.seed)
    n = len(Cl)
    k = min(int(args.knn), len(Ct))
    t2 = (Ct * Ct).sum(1)
    nn_idx = np.empty((n, k), dtype=np.int64)
    chunk = 256
    for i0 in range(0, n, chunk):
        q = Cl[i0 : i0 + chunk]
        q2 = (q * q).sum(1)[:, None]
        d = q2 + t2[None, :] - 2.0 * (q @ Ct.T)
        nn_idx[i0 : i0 + chunk] = np.argpartition(d, kth=k - 1, axis=1)[:, :k]

    if args.mode == "nn":
        Xteach = np.empty_like(Xl)
        for i in range(n):
            ids = nn_idx[i]
            sub = Ct[ids]
            d = ((sub - Cl[i]) ** 2).sum(1)
            Xteach[i] = Xt[ids[int(np.argmin(d))]]
    elif args.mode == "mean":
        Xteach = Xt[nn_idx].mean(axis=1)
    else:
        choice = rng.integers(0, k, size=n)
        Xteach = Xt[nn_idx[np.arange(n), choice]]

    a = float(args.alpha)
    Xnew = (1.0 - a) * Xl + a * Xteach

    out = live.copy()
    out.X = Xnew.astype(np.float32)
    if list(out.var_names) != panel:
        # rare: live already panel-aligned in this repo
        raise SystemExit("live gene order != panel; refusing to write misaligned X")

    assert np.array_equal(
        np.asarray(out.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(live.obsm["spatial_3D"], dtype=np.float32),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    delta = np.linalg.norm(Xnew - Xl, axis=1)
    print(
        f"wrote {args.out}  alpha={a} mode={args.mode} knn={k}  "
        f"median|ΔX|={float(np.median(delta)):.4f}  xyz locked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
