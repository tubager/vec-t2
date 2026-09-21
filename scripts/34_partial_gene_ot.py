#!/usr/bin/env python
"""Partial-gene OT morph: transport only top-|Δμ| genes; lock all other genes + xyz.

Leave-out honest (E6.75+E8.0 → score E7.25). Board: same lo anchors on live 68.26 xyz.
Unlike full-panel CFM, untouched genes keep the live coexpression shell.
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
from t2.geometry import rms_radius  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.ot import minibatch_ot_pairs  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align_genes(X: np.ndarray, genes, panel: list[str]) -> np.ndarray:
    genes = list(genes)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def assign_clusters(X, ref_X, ref_clusters, k: int = 15) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_X)), algorithm="auto").fit(ref_X)
    _, idx = nn.kneighbors(X)
    out = np.empty(len(X), dtype=object)
    for i, neigh in enumerate(idx):
        vals, cnt = np.unique(ref_clusters[neigh], return_counts=True)
        out[i] = vals[int(np.argmax(cnt))]
    return out


def morph_partial_ot(
    X: np.ndarray,
    clusters: np.ndarray,
    X0: np.ndarray,
    cl0: np.ndarray,
    X1: np.ndarray,
    cl1: np.ndarray,
    genes: np.ndarray,
    w: float,
    alpha: float,
    rng: np.random.Generator,
    ot_cap: int = 800,
) -> np.ndarray:
    """Replace ``genes`` columns via OT-interpolated left/right; blend with alpha."""
    out = np.asarray(X, dtype=np.float64).copy()
    w = float(np.clip(w, 0.0, 1.0))
    a = float(np.clip(alpha, 0.0, 1.0))
    genes = np.asarray(genes, dtype=int)
    for name in np.unique(clusters):
        m_q = clusters == name
        m0 = cl0 == name
        m1 = cl1 == name
        if m_q.sum() == 0 or m0.sum() < 4 or m1.sum() < 4:
            continue
        A = X0[m0][:, genes]
        B = X1[m1][:, genes]
        n_pair = min(ot_cap, len(A), len(B))
        try:
            A_ot, B_ot = minibatch_ot_pairs(A, B, n_pair, rng)
        except ValueError:
            mid = ((1.0 - w) * A.mean(0) + w * B.mean(0)).astype(np.float64)
            block = out[m_q]
            block[:, genes] = (1.0 - a) * block[:, genes] + a * mid
            out[m_q] = block
            continue
        Z = (1.0 - w) * A_ot + w * B_ot
        # assign each query cell to nearest OT mid in gene subspace
        Q = out[m_q][:, genes]
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(Z)
        _, idx = nn.kneighbors(Q)
        mid = Z[idx[:, 0]]
        block = out[m_q]
        block[:, genes] = (1.0 - a) * block[:, genes] + a * mid
        out[m_q] = block
    np.clip(out, 0.0, None, out=out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--left", type=str, default="E6.75")
    ap.add_argument("--right", type=str, default="E8.0")
    ap.add_argument("--partial-genes", type=int, default=16)
    ap.add_argument("--w", type=float, required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    spec = target_spec(args.setting, dummy_t, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)
    left, right = stages[args.left], stages[args.right]
    pred = ad.read_h5ad(args.pred)
    Xp = _align_genes(to_dense(pred.X), pred.var_names, panel)
    X0 = _align_genes(to_dense(left.X), left.var_names, panel)
    X1 = _align_genes(to_dense(right.X), right.var_names, panel)
    cl0 = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))
    cl1 = np.asarray(labels_to_clusters(right.obs["celltype"], args.setting))
    usable = shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting)
    dmu = np.zeros(len(panel), dtype=np.float64)
    n = 0
    for name in usable:
        a, b = X0[cl0 == name], X1[cl1 == name]
        if len(a) < 8 or len(b) < 8:
            continue
        dmu += np.abs(a.mean(0) - b.mean(0))
        n += 1
    dmu /= max(n, 1)
    genes = np.argsort(-dmu)[: int(args.partial_genes)]

    rng = np.random.default_rng(args.seed)
    i_l = rng.choice(len(X0), size=min(4000, len(X0)), replace=False)
    i_r = rng.choice(len(X1), size=min(4000, len(X1)), replace=False)
    clusters = assign_clusters(
        Xp,
        np.concatenate([X0[i_l], X1[i_r]], 0),
        np.concatenate([cl0[i_l], cl1[i_r]], 0),
        k=15,
    )
    Xn = morph_partial_ot(
        Xp, clusters, X0, cl0, X1, cl1, genes, args.w, args.alpha, rng
    )
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    assert np.array_equal(
        np.asarray(out.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(pred.obsm["spatial_3D"], dtype=np.float32),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    C = np.asarray(pred.obsm["spatial_3D"], dtype=np.float64)
    print(
        f"partial-gene OT pg={len(genes)} w={args.w:.3f} a={args.alpha:.2f} "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f} rms={rms_radius(C):.2f} "
        f"wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
