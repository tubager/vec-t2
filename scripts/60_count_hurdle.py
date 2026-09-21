#!/usr/bin/env python
"""Count-space / hurdle retarget on frozen xyz. Not Gaussian copula, not log-std.

X on disk is log1p of library-normalized counts. 09a moved log-means; this moves
the *count* means (Jensen gap) or only the detected (X>0) mass.

  .venv/bin/python scripts/60_count_hurdle.py \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --w 0.15 --k-genes 16 --mode count \\
    --out outputs/t2/embryo/gate725_count/pm_count_k16_w015.h5ad

Modes:
  count   — cluster mean in expm1(X), add delta on k genes, log1p back
  hurdle  — same but only on X>0 rows (zeros stay zero)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from t2.clusters import labels_to_clusters, shared_flow_clusters  # noqa: E402
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


def _assign(Xp, ref_X, ref_c, k=15):
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_X))).fit(ref_X)
    _, ix = nn.kneighbors(Xp)
    out = np.empty(len(Xp), dtype=object)
    for i, neigh in enumerate(ix):
        labs, cnt = np.unique(ref_c[neigh], return_counts=True)
        out[i] = labs[int(np.argmax(cnt))]
    return out.astype(str)


def retarget(Xp, clusters, mu_l, mu_r, w, genes_idx, mode: str) -> np.ndarray:
    w = float(np.clip(w, 0.0, 1.0))
    C = np.expm1(np.clip(Xp, 0.0, None))
    Cn = C.copy()
    for name in np.unique(clusters):
        key = str(name)
        if key not in mu_l or key not in mu_r:
            continue
        rows = np.flatnonzero(clusters == name)
        mu_star = (1.0 - w) * mu_l[key] + w * mu_r[key]
        block = Cn[np.ix_(rows, genes_idx)]
        if mode == "hurdle":
            pos = block > 1e-8
            if not np.any(pos):
                continue
            src = np.zeros(len(genes_idx), dtype=np.float64)
            for j in range(len(genes_idx)):
                col = block[:, j]
                m = col > 1e-8
                src[j] = col[m].mean() if np.any(m) else 0.0
            delta = mu_star[genes_idx] - src
            block = block + pos.astype(np.float64) * delta
        else:
            src = block.mean(0)
            block = block + (mu_star[genes_idx] - src)
        np.clip(block, 0.0, None, out=block)
        Cn[np.ix_(rows, genes_idx)] = block
    return np.log1p(Cn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--left", default="E6.75")
    ap.add_argument("--right", default="E8.0")
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--w", type=float, default=0.15)
    ap.add_argument("--k-genes", type=int, default=16)
    ap.add_argument("--mode", choices=["count", "hurdle"], default="count")
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    panel = list(load_panel_for(target_spec(args.setting, dummy_t, cfg), cfg))
    stages = load_stages(args.setting, cfg, panel)
    left, right = stages[args.left], stages[args.right]
    Xl = _align(to_dense(left.X), left.var_names, panel)
    Xr = _align(to_dense(right.X), right.var_names, panel)
    cl = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))
    cr = np.asarray(labels_to_clusters(right.obs["celltype"], args.setting))
    usable = shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting)

    # cluster means in count (expm1) space, optionally hurdle = detected mean
    mu_l, mu_r = {}, {}
    for name in usable:
        a = np.expm1(np.clip(Xl[cl == name], 0.0, None))
        b = np.expm1(np.clip(Xr[cr == name], 0.0, None))
        if len(a) < 8 or len(b) < 8:
            continue
        if args.mode == "hurdle":
            def pos_mean(M):
                out = np.zeros(M.shape[1], dtype=np.float64)
                for j in range(M.shape[1]):
                    col = M[:, j]
                    m = col > 1e-8
                    out[j] = col[m].mean() if np.any(m) else 0.0
                return out
            mu_l[name], mu_r[name] = pos_mean(a), pos_mean(b)
        else:
            mu_l[name], mu_r[name] = a.mean(0), b.mean(0)

    dmu = np.zeros(len(panel), dtype=np.float64)
    n = 0
    for name in mu_l:
        dmu += np.abs(mu_r[name] - mu_l[name])
        n += 1
    dmu /= max(n, 1)
    genes_idx = np.argsort(-dmu)[: int(args.k_genes)]

    rng = np.random.default_rng(args.seed)
    n_ref = min(8000, len(Xl) + len(Xr))
    i_l = rng.choice(len(Xl), size=min(len(Xl), n_ref // 2), replace=False)
    i_r = rng.choice(len(Xr), size=min(len(Xr), n_ref - len(i_l)), replace=False)
    ref_X = np.concatenate([Xl[i_l], Xr[i_r]], axis=0)
    ref_c = np.concatenate([cl[i_l], cr[i_r]], axis=0)

    pred = ad.read_h5ad(args.pred)
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    C0 = np.asarray(pred.obsm["spatial_3D"])
    clusters = _assign(Xp, ref_X, ref_c, k=args.knn)
    Xn = retarget(Xp, clusters, mu_l, mu_r, args.w, genes_idx, args.mode)
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    chk = ad.read_h5ad(args.out)
    assert np.array_equal(np.asarray(chk.obsm["spatial_3D"]), C0), "xyz changed"
    print(
        f"{args.mode} w={args.w:g} k={args.k_genes}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
