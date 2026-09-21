#!/usr/bin/env python
"""Rank-preserving 1D Wasserstein barycenter on genes. xyz sequence bit-exact.

Not RNA copula. Not PCA residual. Not cell swap.
Each gene: Q*(p)=(1-w) Q_left(p)+w Q_right(p), then X'=(1-α)X + α Q*(F(X)).

Leave-out (on 09a analog / band):
  .venv/bin/python scripts/51_gene_qmap.py \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --w 0.15 --alpha 0.15 \\
    --out outputs/t2/embryo/gate725_qmap/lo_pm_a015_w015.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align(X, genes, panel):
    genes = list(genes)
    panel = list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def _qfun(sorted_r: np.ndarray, m: int) -> np.ndarray:
    r = np.asarray(sorted_r, dtype=np.float64)
    n = len(r)
    if n == m:
        return r.copy()
    return np.interp((np.arange(m) + 0.5) / m, (np.arange(n) + 0.5) / n, r)


def qmap(X: np.ndarray, Xl: np.ndarray, Xr: np.ndarray, w: float, alpha: float, n_genes: int | None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    g = X.shape[1]
    idx = np.arange(g)
    if n_genes is not None:
        dmu = np.abs(Xr.mean(0) - Xl.mean(0))
        idx = np.argsort(-dmu)[: int(min(n_genes, g))]
    n = len(X)
    p = (np.arange(n) + 0.5) / n
    w = float(np.clip(w, 0.0, 1.0))
    a = float(np.clip(alpha, 0.0, 1.0))
    for j in idx:
        q = (1.0 - w) * _qfun(np.sort(Xl[:, j]), n) + w * _qfun(np.sort(Xr[:, j]), n)
        order = np.argsort(X[:, j], kind="stable")
        mapped = np.empty(n, dtype=np.float64)
        mapped[order] = q
        out[:, j] = (1.0 - a) * X[:, j] + a * mapped
    np.clip(out, 0.0, None, out=out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--partial-genes", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    panel = list(load_panel_for(target_spec(args.setting, dummy_t, cfg), cfg))
    stages = load_stages(args.setting, cfg, panel)
    left, right = stages[args.left], stages[args.right]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)

    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.asarray(pred.obsm["spatial_3D"])
    Xn = qmap(Xp, Xl, Xr, args.w, args.alpha, args.partial_genes)

    out = pred.copy()
    out.X = Xn.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"]), C0), "xyz changed"
    print(
        f"qmap w={args.w:g} α={args.alpha:g} genes={args.partial_genes or 'all'}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
