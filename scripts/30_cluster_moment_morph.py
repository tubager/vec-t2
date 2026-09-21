#!/usr/bin/env python
"""Per-cluster gene-space moment morph (no PCA bottleneck).

Preserves each cell's residual relative to its source-cluster mean, then retargets
mean (and optional diagonal std) to a clock-interpolated (left, right) moment.
Designed to move mix/band predictions toward mid-stage marginals without the
variogram collapse of PCA-32 OT-CFM.

Usage (leave-out E7.25):
  .venv/bin/python scripts/30_cluster_moment_morph.py \\
    --pred outputs/t2/embryo/gate725/mix.h5ad \\
    --left data/E6.75.h5ad --right data/E8.0.h5ad \\
    --setting embryo --w 0.4 --std-blend 1 \\
    --out outputs/t2/embryo/gate725/mix_moment_w0.4.h5ad

Std-only on the 09a analog (k=16 genes, keep pm means):
  .venv/bin/python scripts/30_cluster_moment_morph.py \\
    --pred outputs/t2/embryo/gate725/lo_pm_k16_w015.h5ad \\
    --left E6.75 --right E8.0 --w 0.15 --std-blend 0.5 --partial-genes 16 --std-only \\
    --out outputs/t2/embryo/gate725_std/pm_w015_sb050.h5ad
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
from t2.io import to_dense  # noqa: E402
from t2.infer import load_panel_for, load_stages, stage_times, target_spec  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std = np.maximum(std, 1e-4)
    return mu.astype(np.float64), std.astype(np.float64)


def _align_genes(X: np.ndarray, genes, panel: list[str]) -> np.ndarray:
    genes = list(genes)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def assign_clusters(
    X: np.ndarray,
    ref_X: np.ndarray,
    ref_clusters: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """Majority vote of kNN in expression space against a labelled reference."""
    nn = NearestNeighbors(n_neighbors=min(k, len(ref_X)), algorithm="auto").fit(ref_X)
    _, idx = nn.kneighbors(X)
    out = np.empty(len(X), dtype=object)
    for i, neigh in enumerate(idx):
        labs = ref_clusters[neigh]
        # mode
        vals, cnt = np.unique(labs, return_counts=True)
        out[i] = vals[int(np.argmax(cnt))]
    return out


def morph(
    X: np.ndarray,
    clusters: np.ndarray,
    mu_l: dict[str, np.ndarray],
    std_l: dict[str, np.ndarray],
    mu_r: dict[str, np.ndarray],
    std_r: dict[str, np.ndarray],
    w: float,
    std_blend: float,
    clip_min: float = 0.0,
    *,
    source: str = "pred",
) -> np.ndarray:
    """Retarget each cluster toward clock-interpolated moments.

    ``source=pred`` (default): keep each cell's residual vs the *prediction's*
    empirical cluster mean, then move that mean to μ*. Safe for OT-place mixes.
    ``source=left``: residual vs μ_L (only for left-born clouds).
    """
    w = float(np.clip(w, 0.0, 1.0))
    sb = float(np.clip(std_blend, 0.0, 1.0))
    out = np.asarray(X, dtype=np.float64).copy()
    for name in np.unique(clusters):
        key = str(name)
        if key not in mu_l or key not in mu_r:
            continue
        mask = clusters == name
        mu_star = (1.0 - w) * mu_l[key] + w * mu_r[key]
        std_star = (1.0 - w) * std_l[key] + w * std_r[key]
        if source == "left":
            mu_src = mu_l[key]
            std_src = std_l[key]
        else:
            mu_src = out[mask].mean(axis=0)
            std_src = np.maximum(out[mask].std(axis=0, ddof=1), 1e-4)
        scale = (std_star / std_src) ** sb if sb > 0 else 1.0
        if sb > 0:
            scale = np.clip(scale, 0.25, 4.0)
        resid = out[mask] - mu_src
        out[mask] = mu_star + resid * scale
    if clip_min is not None:
        np.clip(out, clip_min, None, out=out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--left", type=str, default=None, help="stage name or h5ad path")
    ap.add_argument("--right", type=str, default=None)
    ap.add_argument("--setting", choices=["embryo", "heart"], default="embryo")
    ap.add_argument("--w", type=float, required=True, help="clock weight toward right")
    ap.add_argument("--std-blend", type=float, default=0.0, help="0=mean-only (default), 1=diag-std retarget")
    ap.add_argument(
        "--source",
        choices=["pred", "left"],
        default="pred",
        help="residual center: pred empirical mean (safe for mixes) or left-stage μ",
    )
    ap.add_argument(
        "--partial-genes",
        type=int,
        default=None,
        help="if set, only retarget this many genes with largest |μR−μL| (variogram-safe)",
    )
    ap.add_argument(
        "--std-only",
        action="store_true",
        help="with --partial-genes, scale residuals only (keep the pred's cluster means)",
    )
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    dummy_t = 7.5 if args.setting == "embryo" else 8.5
    spec = target_spec(args.setting, dummy_t, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages(args.setting, cfg, panel)
    times = stage_times(args.setting, cfg)

    def resolve(tag: str | None, default: str) -> ad.AnnData:
        name = tag or default
        if name in stages:
            return stages[name]
        p = Path(name)
        if not p.exists():
            raise SystemExit(f"stage/path not found: {name}")
        return ad.read_h5ad(p)

    left_name = args.left or ("E6.75" if args.setting == "embryo" else "E8.25_late")
    right_name = args.right or ("E8.0" if args.setting == "embryo" else "E9.5")
    # allow bare stage names
    if left_name in stages:
        left = stages[left_name]
        t0 = float(times[left_name])
    else:
        left = resolve(left_name, left_name)
        t0 = float(times.get(Path(left_name).stem, 0.0))
    if right_name in stages:
        right = stages[right_name]
        t1 = float(times[right_name])
    else:
        right = resolve(right_name, right_name)
        t1 = float(times.get(Path(right_name).stem, 1.0))

    Xl = _align_genes(to_dense(left.X), left.var_names, panel)
    Xr = _align_genes(to_dense(right.X), right.var_names, panel)
    cl = np.asarray(labels_to_clusters(left.obs["celltype"], args.setting))
    cr = np.asarray(labels_to_clusters(right.obs["celltype"], args.setting))
    usable = shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], args.setting)

    mu_l, std_l, mu_r, std_r = {}, {}, {}, {}
    for name in usable:
        a = Xl[cl == name]
        b = Xr[cr == name]
        if len(a) < 8 or len(b) < 8:
            continue
        mu_l[name], std_l[name] = _moments(a)
        mu_r[name], std_r[name] = _moments(b)

    pred = ad.read_h5ad(args.pred)
    Xp = _align_genes(to_dense(pred.X), pred.var_names, panel)
    # reference for kNN: subsample left+right
    rng = np.random.default_rng(args.seed)
    n_ref = min(8000, len(Xl) + len(Xr))
    i_l = rng.choice(len(Xl), size=min(len(Xl), n_ref // 2), replace=False)
    i_r = rng.choice(len(Xr), size=min(len(Xr), n_ref - len(i_l)), replace=False)
    ref_X = np.concatenate([Xl[i_l], Xr[i_r]], axis=0)
    ref_c = np.concatenate([cl[i_l], cr[i_r]], axis=0)
    clusters = assign_clusters(Xp, ref_X, ref_c, k=args.knn)

    if args.partial_genes is None:
        Xn = morph(
            Xp, clusters, mu_l, std_l, mu_r, std_r, args.w, args.std_blend, source=args.source
        )
    else:
        dmu = np.zeros(len(panel), dtype=np.float64)
        n = 0
        for name in mu_l:
            dmu += np.abs(mu_r[name] - mu_l[name])
            n += 1
        dmu /= max(n, 1)
        k = min(int(args.partial_genes), len(panel))
        genes_idx = np.argsort(-dmu)[:k]
        Xn = Xp.copy()
        w = float(np.clip(args.w, 0.0, 1.0))
        sb = float(np.clip(args.std_blend, 0.0, 1.0))
        for name in np.unique(clusters):
            key = str(name)
            if key not in mu_l or key not in mu_r:
                continue
            rows = np.flatnonzero(clusters == name)
            mu_star = (1.0 - w) * mu_l[key] + w * mu_r[key]
            std_star = (1.0 - w) * std_l[key] + w * std_r[key]
            mu_src = Xn[rows].mean(0)
            std_src = np.maximum(Xn[rows].std(0, ddof=1), 1e-4)
            block = Xn[np.ix_(rows, genes_idx)].copy()
            if not args.std_only:
                block = block + (mu_star - mu_src)[genes_idx]
            if sb > 0:
                mu_block = block.mean(0)
                scale = np.clip((std_star / std_src) ** sb, 0.5, 2.0)
                block = mu_block + (block - mu_block) * scale[genes_idx]
            Xn[np.ix_(rows, genes_idx)] = block
        np.clip(Xn, 0.0, None, out=Xn)
    out = pred.copy()
    out.X = Xn.astype(np.float32)
    # keep spatial sequence bit-exact
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(args.out, compression="gzip")
    dt = t1 - t0
    print(
        f"moment morph w={args.w:.3f} std_blend={args.std_blend:.2f} source={args.source}  "
        f"partial_genes={args.partial_genes} std_only={args.std_only}  clusters_used={len(mu_l)}  "
        f"mean|dX|={float(np.mean(np.abs(Xn - Xp))):.4f}  "
        f"t_span={t0:.2f}→{t1:.2f} (dt={dt:.2f})  wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
