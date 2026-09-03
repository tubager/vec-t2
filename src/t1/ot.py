from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


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
    # (m0, 1, d) - (1, m1, d)
    diff = a[:, None, :] - b[None, :, :]
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    ri, ci = linear_sum_assignment(cost)
    return a[ri], b[ci]
