"""Log-linear RMS(t) and principal-axis length ratios. No bounding boxes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from t2.geometry import canonicalize, rms_radius


def loglin(t0: float, t1: float, t: float, y0: float, y1: float) -> float:
    if y0 <= 0 or y1 <= 0:
        raise ValueError("log-linear interpolation requires positive y")
    w = (t - t0) / (t1 - t0)
    return float(np.exp((1.0 - w) * np.log(y0) + w * np.log(y1)))


def interp_weight(t_left: float, t_right: float, t: float) -> float:
    return float((t - t_left) / (t_right - t_left))


def r_interp(t_left: float, t_right: float, t: float, r_left: float, r_right: float) -> float:
    return loglin(t_left, t_right, t, r_left, r_right)


def r_extrap(r_anchor: float, beta: float, alpha: float = 1.0) -> float:
    """log r* = log r_anchor + alpha * beta. First version: beta=0."""
    return float(r_anchor * np.exp(alpha * beta))


def axis_interp(
    t_left: float,
    t_right: float,
    t: float,
    std_left,
    std_right,
) -> np.ndarray:
    a = np.asarray(std_left, dtype=np.float64).reshape(3)
    b = np.asarray(std_right, dtype=np.float64).reshape(3)
    return np.array(
        [loglin(t_left, t_right, t, a[i], b[i]) for i in range(3)],
        dtype=np.float64,
    )


def stage_geometry(xyz) -> dict:
    xyz = np.asarray(xyz, dtype=np.float64)
    Z, r, _ = canonicalize(xyz)
    std = Z.std(axis=0, ddof=1)
    return {
        "n": int(xyz.shape[0]),
        "rms": float(r),
        "axis_std": [float(x) for x in std],
        "centroid": [float(x) for x in xyz.mean(axis=0)],
        "has_nan": bool(np.isnan(xyz).any()),
    }


def save_geometry(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_geometry(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def target_rms(
    setting: str,
    t: float,
    rms_by_t: dict[float, float],
    cfg: dict,
) -> float:
    g = cfg["growth"]
    if setting == "embryo":
        t0, t1 = g["embryo_interp_anchors"]
        return r_interp(t0, t1, t, rms_by_t[t0], rms_by_t[t1])
    if t <= float(cfg["time"]["heart"]["t_875"]) + 1e-9:
        t0, t1 = g["heart_interp_anchors"]
        return r_interp(t0, t1, t, rms_by_t[t0], rms_by_t[t1])
    # Heart extrapolation: E9.5 anchor + beta. Do not use 8.75→9.5 slope.
    r_anchor = rms_by_t[float(g["heart_extrap_anchor"])]
    if abs(t - 10.5) < 1e-9:
        beta = float(g.get("heart_extrap_beta_105") or 0.0)
        alpha = float(g.get("heart_extrap_alpha_105") or 1.0)
        return r_extrap(r_anchor, beta, alpha)
    if abs(t - 12.5) < 1e-9:
        beta = g.get("heart_extrap_beta_125")
        if beta is None:
            beta = 0.0
        alpha = g.get("heart_extrap_alpha_125")
        if alpha is None:
            alpha = 1.0
        return r_extrap(r_anchor, float(beta), float(alpha))
    # Heart one-step proxy toward E9.5: use the true E9.5 RMS if present
    # (geometry diagnostic); otherwise freeze the E9.5-or-last available RMS.
    if t in rms_by_t:
        return float(rms_by_t[t])
    return r_anchor
