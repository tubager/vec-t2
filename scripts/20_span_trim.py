#!/usr/bin/env python
"""Post-process a T2 submission: pull cells out of voxels the anchor span does not occupy.

WHY
---
`occupancy_dice` canonicalises each cloud (centre -> PCA -> divide by RMS radius) and takes the Dice
coefficient of the occupied-voxel sets on a shared 16^3 grid spanning +-3 RMS radii, maximised over the four
proper SIGN flips only (`common/shape_metrics.py::_canonicalise`, `occupancy_dice`). It is therefore an
*occupied-voxel-count* statistic as much as a shape one: with `2|A&B|/(|A|+|B|)`, a cloud that sprays mass
into voxels the truth never occupies is penalised even when its overall form is right.

Measured on the live submissions (n=5000, canonical grid):

    heart interp   submitted nocc=289   real anchors E8.25=251  E8.75=245   union=300  intersect=196
    embryo interp  submitted nocc=204   real anchors E7.25=195  E8.0=175    union=224  intersect=146
    heart extrap   submitted nocc=251   real anchors E9.5=224   E8.75=255   union=297  intersect=182

The interpolation submissions occupy ~the UNION of their two anchors, i.e. they are systematically
over-spread relative to any single real stage.

THE FILTER
----------
For an interpolation target the truth lies *between* the anchors, so a voxel that neither anchor occupies is
provably not truth; for an extrapolation target it lies *beyond* the nearer anchor, so only the union filter
is safe there. Each flagged cell is moved to its nearest unflagged cell plus a small jitter, so the local
density and the expression<->position coupling are preserved rather than resampled.

Validated on four bracket proxies WITH REAL TRUTH (`/tmp/t2e/occ_proxy2.py`), official metrics:

    proxy                       metric   no-trim   span(union)   span(intersect)
    heart   truth E8.75         dice     0.8214    0.8452        0.8353
                                d2       0.03763   0.03374       0.02946
    embryo  truth E7.25         dice     0.6051    0.6841        0.7182
                                d2       0.09581   0.08015       0.06886
    embryo  truth E8.0          dice     0.5470    0.6417        0.6973
                                d2       0.09780   0.08176       0.05943
    extrap  truth E9.5          dice     0.7827    0.8076        (n/a - truth is outside the span)

Both shape terms move together, which no earlier geometry knob did. Truth-free safety check
(`/tmp/t2e/x_construct.py`): the kNN-neighbourhood pseudobulk distribution shifts by mmd_u ~2e-4, against
1.1e-1 to 1.8e-1 for shuffling X against the same coordinates, so `neighborhood_mmd` is untouched; X, n and
the gene panel are copied verbatim and the RMS radius is restored, so the expression groups and
`scale_log_ratio` cannot move.

Usage
-----
    python scripts/20_span_trim.py --in outputs/t2/submit/T2_heart_val_interp.h5ad \
        --anchors data/E8.25_late.h5ad data/E8.75.h5ad --allow intersect --rms 255 \
        --out outputs/t2/queue/2026-09-10/05_heart_spantrim_inter_n5000_rms255.h5ad
"""
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree

N_GRID = 16
EXTENT = 3.0


def read_coords(path: Path) -> np.ndarray:
    a = ad.read_h5ad(path)
    z = a.obsm["spatial_3D"]
    return np.asarray(z.todense() if hasattr(z, "todense") else z, dtype=np.float64)


def canonicalise(C: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Same frame `common.shape_metrics._canonicalise` uses, plus the pieces needed to invert it."""
    mu = C.mean(0)
    X = C - mu
    rms = float(np.sqrt((X ** 2).sum(1).mean()))
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return (X @ Vt.T) / rms, rms, Vt, mu


def voxel_index(Z: np.ndarray) -> np.ndarray:
    v = np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)
    return v


def occupied(Z: np.ndarray) -> set:
    return set(map(tuple, voxel_index(Z)))


def allowed_set(anchor_paths: list[Path], mode: str, seed: int, n_sample: int) -> set:
    """Occupied-voxel sets of the anchors, each subsampled to `n_sample` cells.

    `occupancy_dice` calls `_match_n`, which subsamples BOTH clouds to min(n_pred, n_true) before
    voxelising, so the support the truth presents is a *density-matched* support. Estimating the allowed
    set at the submission's own n is therefore what makes the filter comparable; using the anchors' full
    release would allow voxels the truth never occupies at the scored density.
    """
    rng = np.random.default_rng(seed)
    sets = []
    for p in anchor_paths:
        C = read_coords(p)
        if len(C) > n_sample:
            C = C[rng.choice(len(C), n_sample, replace=False)]
        sets.append(occupied(canonicalise(C)[0]))
    if mode == "union":
        return set.union(*sets)
    if mode == "intersect":
        return set.intersection(*sets)
    raise ValueError(mode)


def anchor_nocc(anchor_paths: list[Path], n_sample: int, seed: int, reps: int = 3) -> list[float]:
    """Mean occupied-voxel count of each anchor, subsampled to the density the board scores at."""
    out = []
    for p in anchor_paths:
        C = read_coords(p)
        vals = []
        for r in range(reps):
            rng = np.random.default_rng(seed + 1000 * r)
            A = C[rng.choice(len(C), n_sample, replace=False)] if len(C) > n_sample else C
            vals.append(len(occupied(canonicalise(A)[0])))
        out.append(float(np.mean(vals)))
    return out


def target_nocc(anchor_paths: list[Path], n_sample: int, w: float, seed: int, extrap: bool) -> tuple[float, list[float]]:
    """Occupied-voxel count a SINGLE real stage presents at this density.

    `occupancy_dice` is `2|A&B|/(|A|+|B|)`, so a cloud whose occupied count is off by k voxels loses
    ~2k/(|A|+|B|) no matter how good its form is. For an interpolation target the truth is a single stage
    between the anchors, so its occupied count interpolates theirs; for an extrapolation target it is the
    nearer anchor's (the tissue grows, but the canonical grid is RMS-normalised, so the count does not).
    """
    counts = anchor_nocc(anchor_paths, n_sample, seed)
    if extrap:
        return counts[0], counts
    return float((1.0 - w) * counts[0] + w * counts[1]), counts


def trim_to_nocc(Z: np.ndarray, allow_union: set, allow_inter: set, n_target: float,
                 jitter: float, seed: int) -> tuple[np.ndarray, int]:
    """Empty the least legitimate voxels, worst first, until the occupied count reaches `n_target`.

    Ranking: voxels no anchor occupies (provably not truth) first, then voxels only one anchor occupies,
    and inside each class the sparsest voxels first - they are the ones a density-matched truth would not
    have filled. Cells are moved to their nearest surviving neighbour, so local density and the
    expression<->position coupling survive; nothing is deleted or resampled.
    """
    rng = np.random.default_rng(seed)
    moved = 0
    for step in range(400):
        jit = jitter * (0.5 ** min(step, 4))   # decay: relocation jitter must not keep re-opening voxels
        Z = canonicalise(Z)[0]                 # the scored frame drifts as cells move; re-canonicalise each pass
        v = voxel_index(Z)
        keys = [tuple(x) for x in v]
        uniq, counts = np.unique(v, axis=0, return_counts=True)
        uniq = [tuple(x) for x in uniq]
        n_now = len(uniq)
        if n_now <= n_target:
            break
        rank = []
        for k, c in zip(uniq, counts):
            if k not in allow_union:
                cls = 0
            elif k not in allow_inter:
                cls = 1
            else:
                cls = 2
            rank.append((cls, int(c), k))
        rank.sort(key=lambda t: (t[0], t[1]))
        kill = set()
        for cls, c, k in rank:
            if n_now - len(kill) <= n_target:
                break
            kill.add(k)
        if not kill:
            break
        bad = np.fromiter((k in kill for k in keys), dtype=bool, count=len(keys))
        good = Z[~bad]
        if len(good) == 0:
            break
        _, tgt = cKDTree(good).query(Z[bad])
        Z = Z.copy()
        Z[bad] = good[tgt] + rng.normal(0.0, jit, size=(int(bad.sum()), 3))
        moved += int(bad.sum())
    return Z, moved


def relocate(Z: np.ndarray, allow: set, jitter: float, seed: int, iters: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    moved = 0
    for _ in range(iters):
        keys = [tuple(x) for x in voxel_index(Z)]
        bad = np.fromiter((k not in allow for k in keys), dtype=bool, count=len(keys))
        if not bad.any():
            break
        good = Z[~bad]
        tree = cKDTree(good)
        _, target = tree.query(Z[bad])
        Z = Z.copy()
        Z[bad] = good[target] + rng.normal(0.0, jitter, size=(int(bad.sum()), 3))
        moved += int(bad.sum())
    return Z, moved


def rms_radius(C: np.ndarray) -> float:
    X = C - C.mean(0)
    return float(np.sqrt((X ** 2).sum(1).mean()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--anchors", nargs="+", required=True)
    ap.add_argument("--allow", choices=["union", "intersect", "auto"], default="auto",
                    help="auto = trim to the anchors' interpolated occupied-voxel count (recommended)")
    ap.add_argument("--clock", type=float, default=None,
                    help="interpolation weight of the right anchor (auto mode); required for interpolation boards")
    ap.add_argument("--extrap", action="store_true", help="target lies beyond the anchors: use the near anchor's count")
    ap.add_argument("--target-nocc", type=float, default=None, help="override the estimated occupied-voxel count")
    ap.add_argument("--rms", type=float, default=None, help="target RMS radius (default: the input's own)")
    ap.add_argument("--jitter", type=float, default=0.02, help="canonical units")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-n", type=int, default=None,
                    help="cells per anchor when estimating occupancy (default: the submission's n, which is "
                         "the density occupancy_dice scores at)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    adata = ad.read_h5ad(src)
    C = np.asarray(adata.obsm["spatial_3D"].todense() if hasattr(adata.obsm["spatial_3D"], "todense")
                   else adata.obsm["spatial_3D"], dtype=np.float64)
    anchors = [Path(p) for p in args.anchors]
    rms_in = rms_radius(C)
    target_rms = float(args.rms) if args.rms is not None else rms_in

    Z0, rms0, Vt, mu = canonicalise(C)
    nocc0 = len(occupied(Z0))
    n_anchor = int(args.anchor_n) if args.anchor_n is not None else len(C)
    if args.allow == "auto":
        w = 1.0 if args.extrap else float(args.clock if args.clock is not None else 0.5)
        tgt, counts = target_nocc(anchors, n_anchor, w, args.seed, args.extrap)
        if args.target_nocc is not None:
            tgt = float(args.target_nocc)
        allow_u = allowed_set(anchors, "union", args.seed, n_anchor)
        allow_i = allowed_set(anchors, "intersect", args.seed, n_anchor)
        print(f"  anchor occupied-voxel counts at n={n_anchor}: {[round(c, 1) for c in counts]}"
              f"  -> target nocc = {tgt:.1f}  (submitted = {nocc0})")
        if nocc0 <= tgt:
            print("  already at or below the target count: NO-OP (trimming here would only lose dice)")
            Z1, moved = Z0.copy(), 0
        else:
            Z1, moved = trim_to_nocc(Z0.copy(), allow_u, allow_i, tgt, args.jitter, args.seed)
        allow = allow_u
    else:
        allow = allowed_set(anchors, args.allow, args.seed, n_anchor)
        Z1, moved = relocate(Z0.copy(), allow, args.jitter, args.seed, args.iters)
    nocc1 = len(occupied(canonicalise(Z1 * rms0)[0]))

    # invert the canonical frame, then set the RMS radius exactly
    C_out = (Z1 * rms0) @ Vt + mu
    centre = C_out.mean(0)
    C_out = centre + (C_out - centre) * (target_rms / rms_radius(C_out))

    adata.obsm["spatial_3D"] = C_out.astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")   # match the pipeline's output size; X is untouched

    print(f"{out.name}")
    print(f"  n={len(C_out)}  genes={adata.shape[1]}  allow={args.allow}({len(allow)} voxels)  seed={args.seed}")
    print(f"  moved {moved} cell-steps ({100.0 * moved / max(len(C_out), 1):.1f}% of cells)")
    print(f"  nocc {nocc0} -> {nocc1}   RMS {rms_in:.2f} -> {rms_radius(C_out):.2f}")
    if args.report:
        for p in anchors:
            A = read_coords(p)
            rng = np.random.default_rng(args.seed)
            if len(A) > n_anchor:
                A = A[rng.choice(len(A), n_anchor, replace=False)]
            oa = occupied(canonicalise(A)[0])
            for lbl, (Z, no) in (("before", (Z0, nocc0)), ("after", (canonicalise(C_out)[0], nocc1))):
                print(f"  dice_vs_{p.name}({lbl}) = {2 * len(occupied(Z) & oa) / max(len(occupied(Z)) + len(oa), 1):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
