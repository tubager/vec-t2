from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from t2.geometry import canonicalize
from t2.shape import morph_cluster_cloud


def minibatch_ot_pairs(
    z0: np.ndarray,
    z1: np.ndarray,
    m: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair m cells from z0 with m cells from z1 via squared Euclidean OT (Hungarian)."""
    if len(z0) == 0 or len(z1) == 0:
        raise ValueError("Empty source or target for OT pairing")
    i = rng.choice(len(z0), size=min(m, len(z0)), replace=len(z0) < m)
    j = rng.choice(len(z1), size=min(m, len(z1)), replace=len(z1) < m)
    a = z0[i]
    b = z1[j]
    diff = a[:, None, :] - b[None, :, :]
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    ri, ci = linear_sum_assignment(cost)
    return a[ri], b[ci]


def _hungarian(z0: np.ndarray, z1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = z0[:, None, :].astype(np.float64) - z1[None, :, :].astype(np.float64)
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    return linear_sum_assignment(cost)


def _cluster_index(clusters: np.ndarray, name: str) -> np.ndarray:
    return np.flatnonzero(np.asarray(clusters) == name)


def _progress_axis(z0: np.ndarray, z1: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Unitless progress: left mean → 0, right mean → 1, along μ_R − μ_L."""
    mu0 = z0.mean(axis=0)
    mu1 = z1.mean(axis=0)
    axis = mu1 - mu0
    denom = float(np.dot(axis, axis))
    if denom < 1e-12:
        return None
    s0 = ((z0 - mu0) @ axis) / denom
    s1 = ((z1 - mu0) @ axis) / denom
    return s0, s1, axis


def _slice_eligible(
    scores: np.ndarray,
    keep: float,
    *,
    target: float,
    side: str,
    mode: str,
) -> np.ndarray:
    n = len(scores)
    n_keep = max(1, int(round(float(keep) * n)))
    n_keep = min(n_keep, n)
    mode_n = (mode or "band").lower()
    if mode_n == "band":
        return np.argsort(np.abs(scores - float(target)))[:n_keep]
    if mode_n == "tail":
        if side == "left":
            return np.argsort(-scores)[:n_keep]
        if side == "right":
            return np.argsort(scores)[:n_keep]
        raise ValueError(f"side must be left|right, got {side}")
    raise ValueError(f"slice_mode must be band|tail, got {mode}")


def _backfill_empty_bins(
    chosen: np.ndarray,
    bins: np.ndarray,
    scores: np.ndarray,
    *,
    target: float,
    side: str,
    mode: str,
    max_frac: float = 0.08,
) -> np.ndarray:
    """Keep a global expression slice, then plug the largest empty occupied bins.

    ``max_frac`` caps how many extra cells can be added (fraction of the cluster),
    so the global band still dominates expression.
    """
    n = len(bins)
    if n == 0:
        return chosen
    selected = np.zeros(n, dtype=bool)
    selected[np.asarray(chosen, dtype=np.int64)] = True
    uniq, occ = np.unique(bins, return_counts=True)
    empty = []
    for b, c in zip(uniq.tolist(), occ.tolist()):
        loc = np.flatnonzero(bins == b)
        if selected[loc].any():
            continue
        empty.append((int(c), int(b), loc))
    empty.sort(key=lambda t: -t[0])
    budget = max(1, int(round(float(max_frac) * n)))
    extra: list[int] = []
    for _c, _b, loc in empty:
        if len(extra) >= budget:
            break
        pick = _slice_eligible(scores[loc], keep=1e-9, target=target, side=side, mode=mode)
        extra.append(int(loc[int(pick[0])]))
    if not extra:
        return chosen
    return np.unique(np.concatenate([np.asarray(chosen, dtype=np.int64), np.asarray(extra, dtype=np.int64)]))


def _spatial_bin_ids(C: np.ndarray, n_grid: int = 8, extent: float = 3.0) -> np.ndarray:
    """Voxel ids in the cluster's canonical frame (occupancy_dice-style, coarser grid)."""
    Z, r, _ = canonicalize(C)
    n = len(Z)
    if n == 0 or r < 1e-8:
        return np.zeros(n, dtype=np.int64)
    Zn = Z / r
    v = np.floor((np.clip(Zn, -extent, extent - 1e-9) + extent) / (2.0 * extent) * n_grid).astype(np.int64)
    v = np.clip(v, 0, n_grid - 1)
    return (v[:, 0] * n_grid + v[:, 1]) * n_grid + v[:, 2]


def _apportion(weights: np.ndarray, total: int) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    s = float(w.sum())
    if total <= 0 or len(w) == 0 or s <= 0:
        return np.zeros(len(w), dtype=np.int64)
    raw = w / s * int(total)
    out = np.floor(raw).astype(np.int64)
    rem = int(total) - int(out.sum())
    if rem > 0:
        out[np.argsort(-(raw - out))[:rem]] += 1
    return out


def _spatial_stratified_pairs(
    i0: np.ndarray,
    i1: np.ndarray,
    s0: np.ndarray,
    s1: np.ndarray,
    xyz0: np.ndarray,
    xyz1: np.ndarray,
    expr0: np.ndarray,
    expr1: np.ndarray,
    keep: float,
    target: float,
    mode: str,
    n_pair: int,
    rng: np.random.Generator,
    n_grid: int = 8,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Expression-band inside spatial bins; pair quota follows full left occupancy.

    Returns global stage indices ``(p0, p1)`` already Hungarian-paired, or
    ``None`` to fall back to a global slice.
    """
    if len(i0) < 2 or len(i1) < 2:
        return None
    bins0 = _spatial_bin_ids(xyz0, n_grid=n_grid)
    nn = cKDTree(np.asarray(xyz0, dtype=np.float64)).query(np.asarray(xyz1, dtype=np.float64), k=1)[1]
    bins1 = bins0[np.asarray(nn, dtype=np.int64)]
    uniq, occ = np.unique(bins0, return_counts=True)
    occ_map = {int(b): int(c) for b, c in zip(uniq.tolist(), occ.tolist())}

    e0_by: dict[int, np.ndarray] = {}
    e1_by: dict[int, np.ndarray] = {}
    for b, _ in occ_map.items():
        loc0 = np.flatnonzero(bins0 == b)
        e0_by[b] = loc0[_slice_eligible(s0[loc0], keep, target=target, side="left", mode=mode)]
        loc1 = np.flatnonzero(bins1 == b)
        if len(loc1) == 0:
            e1_by[b] = np.array([], dtype=np.int64)
        else:
            e1_by[b] = loc1[_slice_eligible(s1[loc1], keep, target=target, side="right", mode=mode)]

    e1_global = _slice_eligible(s1, keep, target=target, side="right", mode=mode)
    valid = [b for b in occ_map if len(e0_by[b]) > 0]
    if not valid:
        return None
    weights = np.asarray([occ_map[b] for b in valid], dtype=np.float64)
    m_cap = int(min(n_pair, int(sum(len(e0_by[b]) for b in valid)), max(len(e1_global), 1)))
    if m_cap < 1:
        return None
    alloc = _apportion(weights, m_cap)
    for i, b in enumerate(valid):
        alloc[i] = min(int(alloc[i]), len(e0_by[b]))
    leftover = int(m_cap - int(alloc.sum()))
    order = np.argsort(-weights)
    while leftover > 0:
        progressed = False
        for i in order:
            if leftover <= 0:
                break
            ii = int(i)
            b = valid[ii]
            if alloc[ii] < len(e0_by[b]):
                alloc[ii] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break

    p0_parts, p1_parts = [], []
    for b, k in zip(valid, alloc.tolist()):
        k = int(k)
        if k <= 0:
            continue
        loc0 = e0_by[b]
        loc1 = e1_by[b] if len(e1_by[b]) else e1_global
        if len(loc1) == 0:
            continue
        a0 = rng.choice(loc0, size=k, replace=len(loc0) < k)
        a1 = rng.choice(loc1, size=k, replace=len(loc1) < k)
        if k == 1:
            ri, ci = np.array([0]), np.array([0])
        else:
            ri, ci = _hungarian(expr0[a0], expr1[a1])
        p0_parts.append(i0[a0[ri]])
        p1_parts.append(i1[a1[ci]])
    if not p0_parts:
        return None
    return np.concatenate(p0_parts), np.concatenate(p1_parts)


def draw_pair_indices(
    m: int,
    count: int,
    rng: np.random.Generator,
    unique: bool = False,
) -> np.ndarray:
    """Indices into the m paired templates for ``count`` cells.

    ``unique`` exhausts a random permutation of the m pairs before reusing any,
    so no submitted cell is duplicated more than ``ceil(count / m)`` times.
    With ``unique=False`` this is the historical with-replacement draw, which
    stacks ~16-24% of cells onto an already-used (X, xyz) pair.
    """
    if m <= 0:
        raise ValueError(f"need at least one OT pair, got m={m}")
    if not unique:
        return rng.integers(0, m, size=count)
    reps = -(-int(count) // int(m))
    return np.concatenate([rng.permutation(m) for _ in range(reps)])[:count]


def ot_interpolate_alloc(
    stage_X: dict[str, np.ndarray],
    stage_xyz: dict[str, np.ndarray],
    stage_cl: dict[str, np.ndarray],
    alloc: dict[str, int],
    left: str,
    right: str,
    w: float,
    rng: np.random.Generator,
    *,
    pca=None,
    birth: frozenset[str] | None = None,
    progenitors: dict[str, str] | None = None,
    n_pair: int = 800,
    sigma_birth: float = 0.0,
    xyz_mode: str = "lerp",
    x_mode: str = "lerp",
    xyz_w: float | None = None,
    xyz_rng: np.random.Generator | None = None,
    unique_pick: bool = False,
    slice_keep: float = 0.4,
    slice_mode: str = "band",
    slice_stratify: str = "none",
    slice_bins: int = 8,
    slice_fill_frac: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cluster Hungarian pairing, then place cells. No global Δ.

    ``x_mode``:
      - ``lerp``: gene-wise ``(1-w) X_L + w X_R`` (smears covariance).
      - ``pick``: keep a real cell, Bernoulli(w) left vs right.
      - ``slice``: same as pick, but only from cells whose within-type progress
        along (μ_R−μ_L) is near the clock weight (``band``) or from the late
        left tail + early right tail (``tail``). Real rows only.
        ``slice_stratify='spatial'`` applies that band *inside* spatial voxels
        of the left cloud, allocating templates by full-cloud occupancy so a
        global expression slice cannot empty the periphery.
        ``slice_stratify='fill'`` keeps the global band, then adds one
        best-progress cell from each occupied voxel the band missed.

    ``xyz_mode`` uses the matched partner: ``lerp`` / ``left`` / ``right`` / ``morph`` / ``own``.
    ``own``: each picked cell keeps its own coordinate (native pairing).
    ``morph``: per-cluster spatial OT displacement on the left cloud, then
    place at the matched left cell's morphed coordinate.
    ``xyz_w`` is the lerp/morph weight for coordinates (default: same as ``w``).
    Pairing uses panel PCA if ``pca`` is given, otherwise raw expression.
    Birth / one-sided clusters copy that side (right for birth). Returns
    ``(X, xyz, cluster)``.
    """
    birth = birth or frozenset()
    progenitors = progenitors or {}
    w = float(w)
    w_xyz = float(w if xyz_w is None else xyz_w)
    xyz_rng = xyz_rng or np.random.default_rng(2_000_007)
    rows_x, rows_xyz, rows_cl = [], [], []

    def encode(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if pca is None:
            return X
        return pca.encode(X)

    def take_side(stage: str, keys: list[str], n: int) -> tuple[np.ndarray, np.ndarray] | None:
        if stage not in stage_cl:
            return None
        for key in keys:
            idx = _cluster_index(stage_cl[stage], key)
            if len(idx) == 0:
                continue
            pick = rng.choice(idx, size=n, replace=len(idx) < n)
            return stage_X[stage][pick], stage_xyz[stage][pick]
        return None

    for cluster, count in alloc.items():
        if count <= 0:
            continue
        keys = [cluster]
        if cluster in progenitors:
            keys.append(progenitors[cluster])
        i0 = _cluster_index(stage_cl[left], cluster) if left in stage_cl else np.array([], dtype=int)
        i1 = _cluster_index(stage_cl[right], cluster) if right in stage_cl else np.array([], dtype=int)

        if len(i0) and len(i1):
            x_mode_n = (x_mode or "lerp").lower()
            if x_mode_n == "slice":
                keep = float(slice_keep)
                if not (0.0 < keep <= 1.0):
                    raise ValueError(f"slice_keep must be in (0, 1], got {keep}")
                z0_all = encode(stage_X[left][i0])
                z1_all = encode(stage_X[right][i1])
                prog = _progress_axis(z0_all, z1_all)
                if prog is None:
                    s0 = np.zeros(len(i0))
                    s1 = np.ones(len(i1))
                else:
                    s0, s1, _ = prog
                strat = (slice_stratify or "none").lower()
                paired = None
                if strat == "spatial":
                    n_grid = int(slice_bins)
                    if n_grid < 2:
                        raise ValueError(f"slice_bins must be >= 2, got {n_grid}")
                    paired = _spatial_stratified_pairs(
                        i0, i1, s0, s1,
                        stage_xyz[left][i0], stage_xyz[right][i1],
                        z0_all, z1_all,
                        keep, w, slice_mode, n_pair, rng, n_grid=n_grid,
                    )
                if paired is not None:
                    p0, p1 = paired
                    m = len(p0)
                else:
                    e0 = _slice_eligible(s0, keep, target=w, side="left", mode=slice_mode)
                    e1 = _slice_eligible(s1, keep, target=w, side="right", mode=slice_mode)
                    if strat == "fill":
                        n_grid = int(slice_bins)
                        if n_grid < 2:
                            raise ValueError(f"slice_bins must be >= 2, got {n_grid}")
                        bins0 = _spatial_bin_ids(stage_xyz[left][i0], n_grid=n_grid)
                        nn = cKDTree(np.asarray(stage_xyz[left][i0], dtype=np.float64)).query(
                            np.asarray(stage_xyz[right][i1], dtype=np.float64), k=1
                        )[1]
                        bins1 = bins0[np.asarray(nn, dtype=np.int64)]
                        e0 = _backfill_empty_bins(
                            e0, bins0, s0, target=w, side="left", mode=slice_mode,
                            max_frac=float(slice_fill_frac),
                        )
                        e1 = _backfill_empty_bins(
                            e1, bins1, s1, target=w, side="right", mode=slice_mode,
                            max_frac=float(slice_fill_frac),
                        )
                    pool0, pool1 = i0[e0], i1[e1]
                    m = int(min(n_pair, len(pool0), len(pool1)))
                    s0p = rng.choice(pool0, size=m, replace=False)
                    s1p = rng.choice(pool1, size=m, replace=False)
                    ri, ci = _hungarian(encode(stage_X[left][s0p]), encode(stage_X[right][s1p]))
                    p0, p1 = s0p[ri], s1p[ci]
            else:
                pool0, pool1 = i0, i1
                m = int(min(n_pair, len(pool0), len(pool1)))
                s0p = rng.choice(pool0, size=m, replace=False)
                s1p = rng.choice(pool1, size=m, replace=False)
                ri, ci = _hungarian(encode(stage_X[left][s0p]), encode(stage_X[right][s1p]))
                p0, p1 = s0p[ri], s1p[ci]
            pick = draw_pair_indices(m, count, rng, unique_pick)
            if x_mode_n == "pick" or x_mode_n == "slice":
                n_right = int(rng.binomial(count, np.clip(w, 0.0, 1.0)))
                take_right = np.zeros(count, dtype=bool)
                take_right[:n_right] = True
                rng.shuffle(take_right)
                x = np.empty((count, stage_X[left].shape[1]), dtype=np.float32)
                x[take_right] = stage_X[right][p1[pick][take_right]]
                x[~take_right] = stage_X[left][p0[pick][~take_right]]
            elif x_mode_n == "lerp":
                x = (1.0 - w) * stage_X[left][p0[pick]] + w * stage_X[right][p1[pick]]
            else:
                raise ValueError(f"x_mode must be lerp|pick|slice, got {x_mode}")
            mode = (xyz_mode or "lerp").lower()
            if mode == "own" and x_mode_n in {"pick", "slice"}:
                xyz = np.empty((count, stage_xyz[left].shape[1]), dtype=np.float32)
                xyz[take_right] = stage_xyz[right][p1[pick][take_right]]
                xyz[~take_right] = stage_xyz[left][p0[pick][~take_right]]
            elif mode == "left":
                xyz = stage_xyz[left][p0[pick]]
            elif mode == "right":
                xyz = stage_xyz[right][p1[pick]]
            elif mode == "lerp":
                xyz = (1.0 - w_xyz) * stage_xyz[left][p0[pick]] + w_xyz * stage_xyz[right][p1[pick]]
            elif mode == "morph":
                if len(i0) < 3 or len(i1) < 3:
                    xyz = stage_xyz[left][p0[pick]]
                else:
                    morphed = morph_cluster_cloud(
                        stage_xyz[left][i0],
                        stage_xyz[right][i1],
                        w_xyz,
                        n_pair=m,
                        rng=xyz_rng,
                    )
                    loc = np.searchsorted(i0, p0[pick])
                    xyz = morphed[loc]
            else:
                raise ValueError(f"xyz_mode must be lerp|left|right|morph|own, got {xyz_mode}")
        else:
            prefer_right = cluster in birth or (len(i1) and not len(i0))
            got = take_side(right if prefer_right else left, keys, count)
            if got is None:
                got = take_side(left if prefer_right else right, keys, count)
            if got is None:
                raise RuntimeError(f"No OT template for cluster {cluster}")
            x, xyz = got
            if cluster in birth and sigma_birth > 0:
                x = np.asarray(x, dtype=np.float32) + rng.normal(0.0, sigma_birth, size=x.shape)

        rows_x.append(np.asarray(x, dtype=np.float32))
        rows_xyz.append(np.asarray(xyz, dtype=np.float32))
        rows_cl.append(np.full(count, cluster, dtype=object))

    return (
        np.concatenate(rows_x, axis=0),
        np.concatenate(rows_xyz, axis=0),
        np.concatenate(rows_cl, axis=0),
    )
