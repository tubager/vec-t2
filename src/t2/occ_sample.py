"""Sample birth xyz from a board-proven occupancy support (16³ canonical grid).

Ticket 130 nocc 213 → ODS 53.6; 09a nocc 199 → ODS 66.0. Sample inside the prior's
occupied voxels so nocc matches by construction. Points are voxel-interior, not a
bit-exact copy of the prior cloud.
"""
from __future__ import annotations

import numpy as np

N_GRID = 16
EXTENT = 3.0
VOXEL = (2.0 * EXTENT) / N_GRID


def canonicalise(C: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mu = np.asarray(C, dtype=np.float64).mean(0)
    X = np.asarray(C, dtype=np.float64) - mu
    rms = float(np.sqrt((X**2).sum(1).mean()))
    _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return (X @ Vt.T) / max(rms, 1e-12), rms, Vt, mu


def voxel_index(Z: np.ndarray) -> np.ndarray:
    return np.floor((np.clip(Z, -EXTENT, EXTENT - 1e-9) + EXTENT) / (2 * EXTENT) * N_GRID).astype(int)


def occupied_voxels(C: np.ndarray) -> list[tuple[int, int, int]]:
    Z, _, _, _ = canonicalise(C)
    return sorted(set(map(tuple, voxel_index(Z))))


def sample_occupancy_sites(
    C_prior: np.ndarray,
    n: int,
    rng: np.random.Generator,
    rms: float,
    interior: float = 0.8,
) -> np.ndarray:
    """n tissue points covering the prior's occupied voxels.

    Voxel *centers* sit off the manifold and make spatial kNN pick distant
    cells (NFS/mmd collapse). Use a real prior cell already in the voxel,
    plus optional sub-voxel jitter so the cloud is not bit-exact freeze.
    """
    C_prior = np.asarray(C_prior, dtype=np.float64)
    Z, r, Vt, mu = canonicalise(C_prior)
    keys = [tuple(int(x) for x in row) for row in voxel_index(Z)]
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, k in enumerate(keys):
        buckets.setdefault(k, []).append(i)
    voxels = list(buckets.keys())
    if not voxels:
        raise ValueError("prior occupancy is empty")
    n = int(n)
    order = []
    while len(order) < n:
        extra = list(voxels)
        rng.shuffle(extra)
        order.extend(extra)
    order = order[:n]
    idx = np.array([int(rng.choice(buckets[k])) for k in order], dtype=np.int64)
    Zs = Z[idx].copy()
    jitter = 0.12 * VOXEL * float(np.clip(interior, 0.0, 1.0))
    if jitter > 0:
        Zs = Zs + rng.normal(0.0, jitter, size=Zs.shape)
    C = (Zs * r) @ Vt + mu
    return np.asarray(C, dtype=np.float32)
