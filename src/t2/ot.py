from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cluster Hungarian pairing, then place cells. No global Δ.

    ``x_mode``:
      - ``lerp``: gene-wise ``(1-w) X_L + w X_R`` (smears covariance).
      - ``pick``: keep a real cell, Bernoulli(w) left vs right.

    ``xyz_mode`` uses the matched partner: ``lerp`` / ``left`` / ``right`` / ``morph``.
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
            m = int(min(n_pair, len(i0), len(i1)))
            s0 = rng.choice(i0, size=m, replace=False)
            s1 = rng.choice(i1, size=m, replace=False)
            ri, ci = _hungarian(encode(stage_X[left][s0]), encode(stage_X[right][s1]))
            p0, p1 = s0[ri], s1[ci]
            pick = draw_pair_indices(m, count, rng, unique_pick)
            x_mode_n = (x_mode or "lerp").lower()
            if x_mode_n == "pick":
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
                raise ValueError(f"x_mode must be lerp|pick, got {x_mode}")
            mode = (xyz_mode or "lerp").lower()
            if mode == "left":
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
                raise ValueError(f"xyz_mode must be lerp|left|right|morph, got {xyz_mode}")
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
