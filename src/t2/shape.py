"""Point-cloud interpolation: isotropic scale → anisotropic → OT/TPS (gated)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors

from t2.geometry import canonicalize, scale_cloud, transform_anisotropic, transform_isotropic


def nearest_side(t: float, t_left: float, t_right: float) -> str:
    """Ties go to the left anchor (E8.5 copies E8.25 then shrinks)."""
    if abs(t - t_left) <= abs(t - t_right):
        return "left"
    return "right"


def transform_cloud(
    C,
    r_target: float,
    mode: str = "isotropic",
    axis_std_target=None,
) -> np.ndarray:
    mode = mode.lower()
    if mode == "isotropic":
        return transform_isotropic(C, r_target)
    if mode == "anisotropic":
        if axis_std_target is None:
            raise ValueError("anisotropic mode needs axis_std_target")
        return transform_anisotropic(C, axis_std_target, r_target)
    if mode == "tps":
        # TPS is two-cloud; single-cloud fallback is anisotropic then isotropic.
        if axis_std_target is not None:
            return transform_anisotropic(C, axis_std_target, r_target)
        return transform_isotropic(C, r_target)
    raise ValueError(f"Unknown shape mode {mode}")


# Official SDD tries these four det=+1 axis flips. Independent PCA on
# left vs right can pick opposite signs; lerp then smears through the origin.
_PROPER_FLIPS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)


def match_axis_flips(
    Z_left,
    Z_right,
    rng: np.random.Generator,
    n_pair: int = 800,
) -> np.ndarray:
    """Flip PCA axes of ``Z_right`` (det=+1) to minimize subsample OT cost vs ``Z_left``."""
    z0 = np.asarray(Z_left, dtype=np.float64)
    z1 = np.asarray(Z_right, dtype=np.float64)
    if z0.ndim != 2 or z1.ndim != 2 or z0.shape[1] != 3 or z1.shape[1] != 3:
        raise ValueError("match_axis_flips expects [n, 3] clouds")
    m = int(min(n_pair, len(z0), len(z1)))
    i = rng.choice(len(z0), size=m, replace=False)
    j = rng.choice(len(z1), size=m, replace=False)
    a = z0[i]
    b0 = z1[j]
    best_s = _PROPER_FLIPS[0]
    best_cost = np.inf
    for s in _PROPER_FLIPS:
        b = b0 * s
        diff = a[:, None, :] - b[None, :, :]
        cost = np.einsum("ijk,ijk->ij", diff, diff)
        ri, ci = linear_sum_assignment(cost)
        tot = float(cost[ri, ci].sum())
        if tot < best_cost:
            best_cost = tot
            best_s = s
    return (z1 * best_s).astype(np.float32)


def dens_keep_cloud(
    xyz,
    keep: float,
    rng: np.random.Generator,
    k: int = 15,
) -> np.ndarray:
    """Drop the sparsest points by kNN density, then OT-assign everyone onto a resample of the core.

    ``keep`` is the fraction of points to retain (1.0 = no-op). Same count as ``xyz``.
    """
    keep = float(keep)
    if keep >= 1.0 - 1e-12:
        return np.asarray(xyz, dtype=np.float32)
    if not (0.0 < keep <= 1.0):
        raise ValueError(f"dens_keep must be in (0, 1], got {keep}")
    P = np.asarray(xyz, dtype=np.float64)
    n = len(P)
    if n < k + 2:
        return P.astype(np.float32)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(P)
    dist = nn.kneighbors(P, return_distance=True)[0][:, 1:]
    rho = 1.0 / (dist.mean(axis=1) + 1e-8)
    core = P[rho >= np.quantile(rho, 1.0 - keep)]
    if len(core) < 3:
        return P.astype(np.float32)
    idx = rng.choice(len(core), size=n, replace=len(core) < n)
    return ot_assign_xyz(P, core[idx])


def radius_crop_indices(xyz, r_target: float) -> np.ndarray:
    """Innermost indices whose RMS first reaches ``r_target``. All indices if already smaller."""
    P = np.asarray(xyz, dtype=np.float64)
    n = len(P)
    if n < 4:
        return np.arange(n)
    X = P - P.mean(axis=0)
    sq = (X * X).sum(axis=1)
    r_full = float(np.sqrt(sq.mean()))
    if r_full <= float(r_target) + 1e-8:
        return np.arange(n)
    order = np.argsort(sq)
    lo, hi = 3, n
    while lo < hi:
        mid = (lo + hi) // 2
        if float(np.sqrt(sq[order[:mid]].mean())) < float(r_target):
            lo = mid + 1
        else:
            hi = mid
    k = max(int(lo), 3)
    return np.sort(order[:k])


def crop_place_xyz(
    xyz,
    template,
    r_target: float,
    rng: np.random.Generator,
    mode: str = "isotropic",
    axis_std_target=None,
) -> np.ndarray:
    """OT-assign ``xyz`` onto a unique sample of the radius-cropped ``template``.

    Does not drop expression rows. ``template`` should be the unscaled source cloud.
    """
    tmpl = np.asarray(template, dtype=np.float64)
    keep = radius_crop_indices(tmpl, r_target)
    core = transform_cloud(tmpl[keep], r_target, mode=mode, axis_std_target=axis_std_target)
    n = len(xyz)
    idx = rng.choice(len(core), size=n, replace=len(core) < n)
    return ot_assign_xyz(xyz, np.asarray(core, dtype=np.float64)[idx])


def radius_crop_resample(
    xyz,
    r_target: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Drop outermost points until RMS matches ``r_target``, then OT-assign onto a resample.

    Unlike isotropic scale, this is a FOV crop: peripheral mass is removed rather
    than crushed into the target radius. Same count as ``xyz``.
    """
    P = np.asarray(xyz, dtype=np.float64)
    n = len(P)
    if n < 4:
        return scale_cloud(P, r_target)
    X = P - P.mean(axis=0)
    sq = (X * X).sum(axis=1)
    r_full = float(np.sqrt(sq.mean()))
    if r_full <= float(r_target) + 1e-8:
        return scale_cloud(P, r_target)
    order = np.argsort(sq)
    lo, hi = 3, n
    while lo < hi:
        mid = (lo + hi) // 2
        if float(np.sqrt(sq[order[:mid]].mean())) < float(r_target):
            lo = mid + 1
        else:
            hi = mid
    k = max(int(lo), 3)
    core = scale_cloud(X[order[:k]], r_target)
    idx = rng.choice(len(core), size=n, replace=len(core) < n)
    return ot_assign_xyz(P, np.asarray(core, dtype=np.float64)[idx])


def occupancy_interpolate_cloud(
    C_left,
    C_right,
    w: float,
    n: int,
    rng: np.random.Generator,
    n_bins: int = 48,
) -> np.ndarray:
    """Sample ``n`` points from a voxel occupancy interpolated between two clouds.

    Both clouds should already share a frame (PCA + sign match). Interpolates
    normalized histograms, not cell coordinates, so the result is one density
    rather than chord-smeared pairs.
    """
    left = np.asarray(C_left, dtype=np.float64)
    right = np.asarray(C_right, dtype=np.float64)
    n = int(n)
    n_bins = int(n_bins)
    if n <= 0:
        raise ValueError("occupancy_interpolate_cloud n must be positive")
    pts = np.concatenate([left, right], axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    pad = 0.02 * (hi - lo + 1e-8)
    lo = lo - pad
    hi = hi + pad
    ranges = [(float(lo[d]), float(hi[d])) for d in range(3)]
    h0, edges = np.histogramdd(left, bins=n_bins, range=ranges)
    h1, _ = np.histogramdd(right, bins=n_bins, range=ranges)
    p0 = h0 / max(float(h0.sum()), 1.0)
    p1 = h1 / max(float(h1.sum()), 1.0)
    dens = (1.0 - float(w)) * p0 + float(w) * p1
    dens = np.clip(dens, 0.0, None)
    total = float(dens.sum())
    if total <= 0:
        i = rng.choice(len(left), size=n, replace=len(left) < n)
        return left[i].astype(np.float32)
    dens = dens / total
    flat_idx = rng.choice(dens.size, size=n, p=dens.ravel())
    ix, iy, iz = np.unravel_index(flat_idx, dens.shape)
    xs = rng.uniform(edges[0][ix], edges[0][ix + 1])
    ys = rng.uniform(edges[1][iy], edges[1][iy + 1])
    zs = rng.uniform(edges[2][iz], edges[2][iz + 1])
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


# Exact Hungarian assignment is O(n^2) memory; at the heart leaderboard cap
# (n=17,616) the dense cost matrix alone is 2.5 GB. Above this size fall back
# to a greedy kNN assignment, which keeps every point within a few percent of
# its optimal partner when both clouds sample the same region.
_EXACT_ASSIGN_MAX = 2000


def greedy_assign_xyz(src, tgt, k: int = 24) -> np.ndarray:
    """Bijection ``src`` -> ``tgt``: tightest pairs claim their nearest free target first."""
    from scipy.spatial import cKDTree

    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(tgt, dtype=np.float64)
    n = len(a)
    kk = int(min(max(k, 2), len(b)))
    dist, idx = cKDTree(b).query(a, k=kk)
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    taken = np.zeros(len(b), dtype=bool)
    out = np.empty_like(b)
    for i in np.argsort(dist[:, 0], kind="stable"):
        row = idx[i]
        free_row = row[~taken[row]]
        if free_row.size:
            j = int(free_row[0])
        else:
            rest = np.flatnonzero(~taken)
            j = int(rest[0]) if rest.size else 0
        taken[j] = True
        out[i] = b[j]
    return out.astype(np.float32)


def ot_assign_xyz(src, tgt) -> np.ndarray:
    """Assign each ``src`` point a unique ``tgt`` point by squared-Euclidean OT."""
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(tgt, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("ot_assign_xyz needs equal-length clouds")
    if len(a) == 0:
        return b.astype(np.float32)
    if len(a) > _EXACT_ASSIGN_MAX:
        return greedy_assign_xyz(a, b)
    diff = a[:, None, :] - b[None, :, :]
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    ri, ci = linear_sum_assignment(cost)
    out = np.empty_like(b)
    out[ri] = b[ci]
    return out.astype(np.float32)


def ot_interpolate_clouds(
    C_left,
    C_right,
    w: float,
    r_target: float,
    n_pair: int = 800,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """PCA-align both clouds, Hungarian-pair a subsample, linearly interpolate, scale RMS.

    Used only after isotropic/anisotropic already beat copy on local SDD/ODS.
    """
    rng = rng or np.random.default_rng(0)
    Z0, r0, _ = canonicalize(C_left)
    Z1, r1, _ = canonicalize(C_right)
    n0, n1 = len(Z0), len(Z1)
    m = int(min(n_pair, n0, n1))
    i = rng.choice(n0, size=m, replace=False)
    j = rng.choice(n1, size=m, replace=False)
    a = Z0[i] / max(r0, 1e-8)
    b = Z1[j] / max(r1, 1e-8)
    diff = a[:, None, :] - b[None, :, :]
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    ri, ci = linear_sum_assignment(cost)
    mid = (1.0 - w) * a[ri] + w * b[ci]
    # Map every left point to the displacement of its nearest paired source.
    src = a[ri]
    disp = mid - src
    # Assign remaining left (unit) points via nearest paired source.
    Z0u = Z0 / max(r0, 1e-8)
    d2 = ((Z0u[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(axis=1)
    Y = Z0u + disp[nn]
    return scale_cloud(Y, r_target)


def morph_cluster_cloud(
    C_left,
    C_right,
    w: float,
    n_pair: int = 800,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Displace a cluster's left xyz toward the right cluster. Stays in the original frame.

    Spatial Hungarian on centered clouds, Kabsch-rotate the right sample, lerp
    paired positions (including centroids), then apply NN displacement to every
    left point. Unlike ``ot_interpolate_clouds``, does not re-canonicalize into
    a per-cluster PCA frame (that would stack organs at the origin).
    """
    from t2.geometry import kabsch_rotation

    rng = rng or np.random.default_rng(0)
    left = np.asarray(C_left, dtype=np.float64)
    right = np.asarray(C_right, dtype=np.float64)
    w = float(w)
    n0, n1 = len(left), len(right)
    m = int(min(n_pair, n0, n1))
    if m < 3:
        return left.astype(np.float32)
    i = rng.choice(n0, size=m, replace=False)
    j = rng.choice(n1, size=m, replace=False)
    mu_l = left.mean(axis=0)
    mu_r = right.mean(axis=0)
    a = left[i] - mu_l
    b = right[j] - mu_r
    ra = float(np.sqrt((a**2).sum(axis=1).mean()))
    rb = float(np.sqrt((b**2).sum(axis=1).mean()))
    au = a / max(ra, 1e-8)
    bu = b / max(rb, 1e-8)
    diff = au[:, None, :] - bu[None, :, :]
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    ri, ci = linear_sum_assignment(cost)
    rot = kabsch_rotation(b[ci], a[ri])
    mid_c = (1.0 - w) * a[ri] + w * (b[ci] @ rot)
    mu_mid = (1.0 - w) * mu_l + w * mu_r
    mid = mid_c + mu_mid
    src = left[i][ri]
    disp = mid - src
    d2 = ((left[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    nn = d2.argmin(axis=1)
    return (left + disp[nn]).astype(np.float32)


def choose_template_cloud(
    C_left,
    C_right,
    t: float,
    t_left: float,
    t_right: float,
    r_target: float,
    mode: str = "isotropic",
    mix: bool = False,
    w: float | None = None,
    axis_std_target=None,
    rng: np.random.Generator | None = None,
    tps_n: int = 800,
) -> np.ndarray:
    """Build a single whole-embryo template at r_target."""
    w = interp_w(t, t_left, t_right) if w is None else float(w)
    if mode == "tps" and C_right is not None:
        return ot_interpolate_clouds(C_left, C_right, w, r_target, n_pair=tps_n, rng=rng)
    side = nearest_side(t, t_left, t_right)
    src = C_left if side == "left" else C_right
    if mix and C_right is not None:
        L = transform_cloud(C_left, r_target, mode="isotropic" if mode == "tps" else mode, axis_std_target=axis_std_target)
        R = transform_cloud(C_right, r_target, mode="isotropic" if mode == "tps" else mode, axis_std_target=axis_std_target)
        rng = rng or np.random.default_rng(0)
        n_l = max(1, int(round(len(L) * (1.0 - w))))
        n_r = max(1, int(round(len(R) * w)))
        i = rng.choice(len(L), size=min(n_l, len(L)), replace=n_l > len(L))
        j = rng.choice(len(R), size=min(n_r, len(R)), replace=n_r > len(R))
        return np.concatenate([L[i], R[j]], axis=0)
    return transform_cloud(src, r_target, mode=mode, axis_std_target=axis_std_target)


def interp_w(t: float, t_left: float, t_right: float) -> float:
    return float((t - t_left) / (t_right - t_left))
