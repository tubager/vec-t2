"""Neighborhood mid-stage mapper: predict mid X from left/right spatial neighbors.

Supervised on true E7.25 cells (features from E6.75+E8.0 neighborhoods). At board
time the same weights map (E7.25, E8.0) neighborhoods → E7.5 slots on a locked xyz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class NbhdMidMLP(nn.Module):
    """Residual on PCA lerp: z = (1-w) z_l + w z_r + scale * mlp([z_l,z_r,w])."""

    def __init__(self, d: int = 32, hidden: int = 256, drop_p: float = 0.0):
        super().__init__()
        self.d = int(d)
        layers: list[nn.Module] = [
            nn.Linear(2 * int(d) + 1, int(hidden)),
            nn.SiLU(),
        ]
        if float(drop_p) > 0:
            layers.append(nn.Dropout(p=float(drop_p)))
        layers.extend(
            [
                nn.Linear(int(hidden), int(hidden)),
                nn.SiLU(),
                nn.Linear(int(hidden), int(d)),
            ]
        )
        self.net = nn.Sequential(*layers)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_l: torch.Tensor, z_r: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if w.ndim == 1:
            w = w[:, None]
        inp = torch.cat([z_l, z_r, w], dim=-1)
        base = (1.0 - w) * z_l + w * z_r
        return base + self.res_scale * self.net(inp)


def save_nbhd(path: Path, model: NbhdMidMLP, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_nbhd(path: Path, device: torch.device | None = None) -> tuple[NbhdMidMLP, dict]:
    device = device or torch.device("cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    model = NbhdMidMLP(
        d=int(blob.get("d", 32)),
        hidden=int(blob.get("hidden", 256)),
        drop_p=float(blob.get("dropout", blob.get("drop_p", 0.0))),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob


def bit_exact_xyz_from(donor: np.ndarray) -> np.ndarray:
    """Copy spatial cloud with no rescale / OT — float32 bit-identical."""
    return np.array(np.asarray(donor, dtype=np.float32), copy=True)
