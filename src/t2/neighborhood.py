"""Place cells on same-cluster point clouds + isotropic jitter (NFS channel)."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def median_nn_dist(xyz, k: int = 1, max_n: int = 2000, rng: np.random.Generator | None = None) -> float:
    P = np.asarray(xyz, dtype=np.float64)
    if len(P) < 2:
        return 1.0
    if len(P) > max_n:
        rng = rng or np.random.default_rng(0)
        P = P[rng.choice(len(P), size=max_n, replace=False)]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(P)
    d = nn.kneighbors(P, return_distance=True)[0][:, 1:]
    med = float(np.median(d[:, 0]))
    return med if med > 0 else 1.0


def sample_from_cloud(P: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    replace = len(P) < n
    idx = rng.choice(len(P), size=n, replace=replace)
    return np.asarray(P[idx], dtype=np.float32)


def build_cluster_templates(
    stage_xyz: dict[str, np.ndarray],
    stage_clusters: dict[str, np.ndarray],
    weights: dict[str, float],
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate (optionally subsampled) same-cluster coords across stages.

    `stage_xyz` values must already be scaled/deformed to the target RMS.
    """
    rng = rng or np.random.default_rng(0)
    keys = set()
    for cl in stage_clusters.values():
        keys.update(np.unique(cl).tolist())
    templates: dict[str, np.ndarray] = {}
    for k in keys:
        parts = []
        for stage, w in weights.items():
            if w <= 0 or stage not in stage_xyz:
                continue
            mask = np.asarray(stage_clusters[stage]) == k
            P = np.asarray(stage_xyz[stage])[mask]
            if len(P) == 0:
                continue
            n_keep = max(1, int(round(len(P) * float(w))))
            parts.append(sample_from_cloud(P, n_keep, rng))
        if not parts:
            for stage in stage_xyz:
                mask = np.asarray(stage_clusters[stage]) == k
                P = np.asarray(stage_xyz[stage])[mask]
                if len(P) > 0:
                    parts.append(np.asarray(P, dtype=np.float32))
        if parts:
            templates[k] = np.concatenate(parts, axis=0)
    return templates


def place_by_cluster(
    expr_clusters,
    templates: dict[str, np.ndarray],
    rng: np.random.Generator,
    jitter: float,
    progenitors: dict[str, str] | None = None,
) -> np.ndarray:
    """templates[k] = (n_k, 3) scaled point cloud for cluster k.

    Missing clusters use the progenitor template. Never falls back to the
    concatenated whole-embryo blob (that flattens NFS).
    """
    expr_clusters = np.asarray(expr_clusters)
    n = len(expr_clusters)
    xyz = np.empty((n, 3), dtype=np.float32)
    progenitors = progenitors or {}

    def _template(k: str) -> np.ndarray:
        P = templates.get(k)
        if P is not None and len(P) > 0:
            return P
        p = progenitors.get(k)
        if p is not None:
            P = templates.get(p)
            if P is not None and len(P) > 0:
                return P
        raise RuntimeError(f"No spatial template for cluster {k} (no progenitor fallback)")

    for k in np.unique(expr_clusters):
        mask = expr_clusters == k
        n_k = int(mask.sum())
        P = _template(str(k))
        xyz[mask] = sample_from_cloud(P, n_k, rng)

    nn = median_nn_dist(xyz, k=1, rng=rng)
    if jitter > 0 and nn > 0:
        xyz = xyz + rng.normal(0.0, jitter * nn, size=xyz.shape).astype(np.float32)
    return xyz


def add_jitter(xyz, rng: np.random.Generator, jitter: float) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    if jitter <= 0:
        return xyz
    nn = median_nn_dist(xyz, k=1, rng=rng)
    if nn <= 0:
        return xyz
    return xyz + rng.normal(0.0, jitter * nn, size=xyz.shape).astype(np.float32)


def shuffle_xyz(xyz, rng: np.random.Generator) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32).copy()
    rng.shuffle(xyz)
    return xyz
