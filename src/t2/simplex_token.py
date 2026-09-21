"""Convex combination of real spatial neighbors (variogram-safe X)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class SimplexMixer(nn.Module):
    def __init__(self, d: int, n_nb: int, hidden: int = 256):
        super().__init__()
        self.n_nb = int(n_nb)
        self.net = nn.Sequential(
            nn.Linear(2 * int(d) + 1, int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(n_nb)),
        )

    def forward(self, z_l, z_r, w, X_nb):
        if w.ndim == 1:
            w = w[:, None]
        logits = self.net(torch.cat([z_l, z_r, w], dim=-1))
        wt = torch.softmax(logits, dim=-1)
        return (wt.unsqueeze(-1) * X_nb).sum(dim=1), wt


def load_simplex(path: Path, device: torch.device | None = None) -> tuple[SimplexMixer, dict]:
    device = device or torch.device("cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = SimplexMixer(d=int(blob["d"]), n_nb=int(blob["n_nb"]), hidden=int(blob.get("hidden", 256)))
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob
