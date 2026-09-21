"""Discrete neighborhood pick: one real left/right neighbor, never a mix.

Trained as a ranker over spatial kNN. Inference is argmax, so X is always a
true MERFISH vector. Pair with freeze_x_onto() for the second dual-gate.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class NeighborRanker(nn.Module):
    """Score each candidate given local left/right context and clock w."""

    def __init__(self, d: int = 32, hidden: int = 256):
        super().__init__()
        self.d = int(d)
        in_dim = 4 * int(d) + 1  # z_l, z_r, w, z_cand, z_cand - lerp
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(
        self,
        z_l: torch.Tensor,
        z_r: torch.Tensor,
        w: torch.Tensor,
        z_nb: torch.Tensor,
    ) -> torch.Tensor:
        if w.ndim == 1:
            w = w[:, None]
        b, k, d = z_nb.shape
        lerp = (1.0 - w) * z_l + w * z_r
        ctx = torch.cat([z_l, z_r, w], dim=-1).unsqueeze(1).expand(b, k, -1)
        delta = z_nb - lerp.unsqueeze(1)
        inp = torch.cat([ctx, z_nb, delta], dim=-1)
        return self.net(inp).squeeze(-1)


def save_ranker(path: Path, model: NeighborRanker, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_ranker(path: Path, device: torch.device | None = None) -> tuple[NeighborRanker, dict]:
    device = device or torch.device("cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = NeighborRanker(d=int(blob.get("d", 32)), hidden=int(blob.get("hidden", 256)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob


def neighbor_labels(X_nb: np.ndarray, X_true: np.ndarray) -> np.ndarray:
    """Per row, index of the real neighbor closest to the true mid cell."""
    d2 = ((X_nb.astype(np.float32) - X_true[:, None, :].astype(np.float32)) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def snapmean_indices(z_nb: np.ndarray, z_l: np.ndarray, z_r: np.ndarray, w: float) -> np.ndarray:
    """No-train discrete pick: neighbor closest to local PCA lerp."""
    lerp = (1.0 - float(w)) * z_l + float(w) * z_r
    d2 = ((z_nb - lerp[:, None, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def gather_picks(
    indices: np.ndarray,
    i0: np.ndarray,
    i1: np.ndarray,
    X0: np.ndarray,
    X1: np.ndarray,
    C0: np.ndarray,
    C1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k0 = int(i0.shape[1])
    n = len(indices)
    X = np.empty((n, X0.shape[1]), dtype=np.float32)
    xyz = np.empty((n, C0.shape[1]), dtype=np.float32)
    from_right = indices >= k0
    take = np.empty(n, dtype=np.int64)
    take[~from_right] = i0[np.flatnonzero(~from_right), indices[~from_right]]
    take[from_right] = i1[np.flatnonzero(from_right), indices[from_right] - k0]
    X[~from_right] = X0[take[~from_right]]
    X[from_right] = X1[take[from_right]]
    xyz[~from_right] = C0[take[~from_right]]
    xyz[from_right] = C1[take[from_right]]
    return X, xyz


def freeze_x_onto(X: np.ndarray, cand_xyz: np.ndarray, live):
    """Put ``X`` onto ``live`` spatial sequence (bit-exact). Hungarian if clouds differ."""
    live = live.copy()
    C_live = np.asarray(live.obsm["spatial_3D"], dtype=np.float32)
    C_cand = np.asarray(cand_xyz, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if len(C_live) != len(C_cand) or len(X) != len(C_cand):
        raise ValueError(f"n mismatch freeze {len(C_live)} cand {len(C_cand)} X {len(X)}")
    if np.array_equal(C_live, C_cand):
        order = np.arange(len(X), dtype=np.int64)
    else:
        a = C_live.astype(np.float32)
        b = C_cand.astype(np.float32)
        cost = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
        ri, ci = linear_sum_assignment(cost)
        order = np.empty(len(X), dtype=np.int64)
        order[ri] = ci
    out = live.copy()
    out.X = X[order]
    out.obsm["spatial_3D"] = np.array(np.asarray(live.obsm["spatial_3D"], dtype=np.float32), copy=True)
    if not np.array_equal(
        np.asarray(out.obsm["spatial_3D"], dtype=np.float32),
        np.asarray(live.obsm["spatial_3D"], dtype=np.float32),
    ):
        raise RuntimeError("freeze xyz is not bit-exact with live")
    return out
