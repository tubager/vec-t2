#!/usr/bin/env python
"""Gaussian-copula barycenter on a frozen xyz sequence.

Not 1D qmap (that reassigns each gene's values independently and blew variogram).
This recolors the *joint* Gaussian scores so the k-gene correlation moves toward
the clock mix of left/right copulas. Optional margin retarget is separate.

  .venv/bin/python scripts/56_copula_bary.py \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --w 0.4 --alpha 0.15 --k-genes 16 --mode corr \\
    --out outputs/t2/embryo/gate725_copula/lo_pm_corr_k16_a015.h5ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

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


def _gauss_scores(X: np.ndarray) -> np.ndarray:
    n, g = X.shape
    U = np.empty((n, g), dtype=np.float64)
    for j in range(g):
        r = rankdata(X[:, j], method="average")
        U[:, j] = (r - 0.5) / n
    return ndtri(np.clip(U, 1e-6, 1.0 - 1e-6))


def _spd(C: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    C = 0.5 * (C + C.T)
    d = np.sqrt(np.clip(np.diag(C), 1e-8, None))
    corr = C / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    e, V = np.linalg.eigh(corr)
    e = np.maximum(e, eps)
    corr = (V * e) @ V.T
    np.fill_diagonal(corr, 1.0)
    return corr


def _recolor(Z: np.ndarray, C_from: np.ndarray, C_to: np.ndarray) -> np.ndarray:
    Lf = np.linalg.cholesky(_spd(C_from))
    Lt = np.linalg.cholesky(_spd(C_to))
    white = np.linalg.solve(Lf, Z.T).T
    return white @ Lt.T


def _quantile_decode(U: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    n, k = U.shape
    out = np.empty((n, k), dtype=np.float64)
    m = len(sorted_ref)
    grid = (np.arange(m) + 0.5) / m
    for j in range(k):
        out[:, j] = np.interp(U[:, j], grid, sorted_ref[:, j])
    return out


def bary(X, Xl, Xr, w: float, alpha: float, k_genes: int | None, mode: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    out = X.copy()
    g = X.shape[1]
    if k_genes is None or int(k_genes) >= g:
        idx = np.arange(g)
    else:
        dmu = np.abs(Xr.mean(0) - Xl.mean(0))
        idx = np.argsort(-dmu)[: int(k_genes)]
    Xs, Xls, Xrs = X[:, idx], Xl[:, idx], Xr[:, idx]
    Zp = _gauss_scores(Xs)
    Zl = _gauss_scores(Xls)
    Zr = _gauss_scores(Xrs)
    Cp = np.corrcoef(Zp.T)
    Ct = (1.0 - float(w)) * np.corrcoef(Zl.T) + float(w) * np.corrcoef(Zr.T)
    Zt = _recolor(Zp, Cp, Ct)
    a = float(np.clip(alpha, 0.0, 1.0))
    n = len(Xs)
    k = int(Xs.shape[1])
    if mode == "corr":
        Zmix = (1.0 - a) * Zp + a * Zt
        decoded = _quantile_decode(ndtr(Zmix), np.sort(Xs, axis=0))
        out[:, idx] = decoded
    elif mode == "full":
        qT = np.empty((n, k), dtype=np.float64)
        p = (np.arange(n) + 0.5) / n
        for j in range(k):
            gL = (np.arange(len(Xls)) + 0.5) / len(Xls)
            gR = (np.arange(len(Xrs)) + 0.5) / len(Xrs)
            qL = np.interp(p, gL, np.sort(Xls[:, j]))
            qR = np.interp(p, gR, np.sort(Xrs[:, j]))
            qT[:, j] = (1.0 - float(w)) * qL + float(w) * qR
        decoded = _quantile_decode(ndtr(Zt), qT)
        out[:, idx] = (1.0 - a) * Xs + a * decoded
    else:
        raise ValueError(mode)
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
    ap.add_argument("--k-genes", type=int, default=16)
    ap.add_argument("--mode", choices=["corr", "full"], default="corr")
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
    Xn = bary(Xp, Xl, Xr, args.w, args.alpha, args.k_genes, args.mode)
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"]), C0), "xyz changed"
    print(
        f"copula {args.mode} w={args.w:g} α={args.alpha:g} k={args.k_genes}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
