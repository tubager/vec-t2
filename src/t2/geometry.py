"""Procrustes-style geometry: RMS radius, PCA alignment, isotropic / anisotropic scale."""

from __future__ import annotations

import numpy as np


def rms_radius(C) -> float:
    X = np.asarray(C, dtype=np.float64)
    X = X - X.mean(axis=0)
    return float(np.sqrt((X**2).sum(axis=1).mean()))


def canonicalize(C):
    """Center, PCA-align to principal axes. Returns (Z, rms, Vt[:3]). Z is not unit-RMS."""
    X = np.asarray(C, dtype=np.float64)
    X = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    axes = Vt[:3]
    Z = X @ axes.T
    r = float(np.sqrt((Z**2).sum(axis=1).mean()))
    return Z, r, axes


def axis_stds(C) -> np.ndarray:
    Z, _, _ = canonicalize(C)
    return Z.std(axis=0, ddof=1).astype(np.float64)


def scale_cloud(C, r_target: float) -> np.ndarray:
    X = np.asarray(C, dtype=np.float64)
    X = X - X.mean(axis=0)
    r = float(np.sqrt((X**2).sum(axis=1).mean()))
    return (X * (r_target / max(r, 1e-8))).astype(np.float32)


def anisotropic_scale(Z, axis_std_target, r_target: float) -> np.ndarray:
    """Z is already PCA-aligned. Stretch axes to `axis_std_target`, then set RMS to r_target."""
    s = np.asarray(axis_std_target, dtype=np.float64).reshape(3)
    Z = np.asarray(Z, dtype=np.float64)
    cur = Z.std(axis=0, ddof=1)
    Y = Z / (cur + 1e-8) * s
    Y = Y - Y.mean(axis=0)
    r = float(np.sqrt((Y**2).sum(axis=1).mean()))
    return (Y * (r_target / max(r, 1e-8))).astype(np.float32)


def kabsch_rotation(source, target) -> np.ndarray:
    """det=+1 rotation mapping centered ``source`` rows toward centered ``target``."""
    p = np.asarray(source, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    if len(p) < 3:
        return np.eye(3, dtype=np.float64)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt = vt.copy()
        vt[-1] *= -1
        r = u @ vt
    return r


def rigid_align(source, target) -> np.ndarray:
    """Map ``source`` onto ``target``: det=+1 rotation, isotropic scale, translation.

    Fits on all rows jointly (Kabsch). If fewer than 3 points or a cloud is
    degenerate, returns the target centroid broadcast to ``source`` length.
    """
    p = np.asarray(source, dtype=np.float64)
    q = np.asarray(target, dtype=np.float64)
    if p.ndim != 2 or q.ndim != 2 or p.shape[1] != 3 or q.shape[1] != 3:
        raise ValueError("rigid_align expects [n, 3] clouds")
    if len(p) != len(q):
        raise ValueError("rigid_align needs paired clouds of equal length")
    if len(p) < 3:
        return np.broadcast_to(q.mean(axis=0), p.shape).astype(np.float32)
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    rp = float(np.sqrt((pc**2).sum(axis=1).mean()))
    rq = float(np.sqrt((qc**2).sum(axis=1).mean()))
    if rp < 1e-8:
        return np.broadcast_to(q.mean(axis=0), p.shape).astype(np.float32)
    r = kabsch_rotation(pc, qc)
    scale = rq / rp
    aligned = (pc @ r) * scale + q.mean(axis=0)
    return aligned.astype(np.float32)


def transform_isotropic(C, r_target: float) -> np.ndarray:
    return scale_cloud(C, r_target)


def transform_anisotropic(C, axis_std_target, r_target: float) -> np.ndarray:
    Z, _, _ = canonicalize(C)
    return anisotropic_scale(Z, axis_std_target, r_target)
