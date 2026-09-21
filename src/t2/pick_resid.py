"""Pick + small PCA residual. Base is a real cell; residual is NFS/variogram-regularized."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class PickResidualMLP(nn.Module):
    """X = X_pick + scale * (dz @ components). Starts near the real cell."""

    def __init__(self, d: int = 32, hidden: int = 256):
        super().__init__()
        self.d = int(d)
        self.net = nn.Sequential(
            nn.Linear(2 * int(d) + 1, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(d)),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, z_l, z_r, w, X_pick, components):
        if w.ndim == 1:
            w = w[:, None]
        dz = self.scale * self.net(torch.cat([z_l, z_r, w], dim=-1))
        return X_pick + dz @ components, dz


def load_pick_resid(path: Path, device: torch.device | None = None) -> tuple[PickResidualMLP, dict]:
    device = device or torch.device("cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = PickResidualMLP(d=int(blob.get("d", 32)), hidden=int(blob.get("hidden", 256)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob
