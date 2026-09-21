#!/usr/bin/env python
"""Joint qmap on the PCA of the partial-mean genes. xyz bit-exact.

1D qmap (51) and Gaussian copula (56/190) are closed. This transports the
k-gene *subspace* along its own principal axes, then decodes.

  .venv/bin/python scripts/59_pc_qmap.py \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --k-genes 16 --rank 4 --alpha 0.10 --w 0.4 \\
    --out outputs/t2/embryo/gate725_pcqmap/lo_k16_r4_a010.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA

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


def _qfun(sorted_r, m):
    n = len(sorted_r)
    if n == m:
        return sorted_r.copy()
    return np.interp((np.arange(m) + 0.5) / m, (np.arange(n) + 0.5) / n, sorted_r)


def pc_qmap(X, Xl, Xr, k_genes: int, rank: int, w: float, alpha: float) -> np.ndarray:
    dmu = np.abs(Xr.mean(0) - Xl.mean(0))
    idx = np.argsort(-dmu)[: int(k_genes)]
    Xs, Xls, Xrs = X[:, idx], Xl[:, idx], Xr[:, idx]
    pool = np.concatenate([Xls, Xrs], axis=0)
    r = int(max(1, min(int(rank), Xs.shape[1], len(pool) - 1)))
    pca = PCA(n_components=r, svd_solver="full")
    pca.fit(pool)
    zp, zl, zr = pca.transform(Xs), pca.transform(Xls), pca.transform(Xrs)
    n = len(zp)
    zq = np.empty_like(zp)
    for j in range(r):
        q = (1.0 - float(w)) * _qfun(np.sort(zl[:, j]), n) + float(w) * _qfun(np.sort(zr[:, j]), n)
        order = np.argsort(zp[:, j], kind="stable")
        mapped = np.empty(n, dtype=np.float64)
        mapped[order] = q
        zq[:, j] = mapped
    rec = pca.inverse_transform(zq)
    out = X.copy()
    a = float(np.clip(alpha, 0.0, 1.0))
    out[:, idx] = (1.0 - a) * Xs + a * rec
    np.clip(out, 0.0, None, out=out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--k-genes", type=int, default=16)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--w", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    panel = list(load_panel_for(target_spec(args.setting, dummy_t, cfg), cfg))
    stages = load_stages(args.setting, cfg, panel)
    Xl = _align(to_dense(stages[args.left].X), stages[args.left].var_names, panel)
    Xr = _align(to_dense(stages[args.right].X), stages[args.right].var_names, panel)
    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.asarray(pred.obsm["spatial_3D"])
    Xn = pc_qmap(Xp, Xl, Xr, args.k_genes, args.rank, args.w, args.alpha)
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"]), C0)
    print(
        f"pc-qmap k={args.k_genes} r={args.rank} α={args.alpha:g} w={args.w:g}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
