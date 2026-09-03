"""Point-cloud interpolation: isotropic scale → anisotropic → OT/TPS (gated)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

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
