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


def transform_isotropic(C, r_target: float) -> np.ndarray:
    return scale_cloud(C, r_target)


def transform_anisotropic(C, axis_std_target, r_target: float) -> np.ndarray:
    Z, _, _ = canonicalize(C)
    return anisotropic_scale(Z, axis_std_target, r_target)
