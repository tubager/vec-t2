from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Velocity(nn.Module):
    """Non-autonomous OT-CFM velocity: v(z, t, Δt, cluster)."""

    def __init__(
        self,
        d: int = 64,
        emb_dim: int = 16,
        n_clusters: int = 15,
        hidden: list[int] | None = None,
        t_anchor: float = 8.5,
        t_scale: float = 4.0,
    ):
        super().__init__()
        hidden = hidden or [256, 256]
        self.d = d
        self.t_anchor = t_anchor
        self.t_scale = t_scale
        self.emb = nn.Embedding(n_clusters, emb_dim)
        layers: list[nn.Module] = []
        in_dim = d + 2 + emb_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.SiLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, d))
        self.net = nn.Sequential(*layers)

    def _norm_t(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.t_anchor) / self.t_scale

    def _norm_dt(self, dt: torch.Tensor) -> torch.Tensor:
        return dt / self.t_scale

    def forward(self, z, t, dt, cluster_id):
        e = self.emb(cluster_id)
        t_n = self._norm_t(t)
        dt_n = self._norm_dt(dt)
        inp = torch.cat([z, t_n, dt_n, e], dim=-1)
        return self.net(inp)


def cfm_loss(model: Velocity, z0, z1, t0, dt, cid, z_noise: float = 0.0):
    tau = torch.rand(z0.size(0), 1, device=z0.device)
    if z_noise > 0:
        z0 = z0 + z_noise * torch.randn_like(z0)
        z1 = z1 + z_noise * torch.randn_like(z1)
    z_tau = (1 - tau) * z0 + tau * z1
    u = z1 - z0
    t = t0 + tau * dt
    v = model(z_tau, t, dt, cid)
    return ((v - u) ** 2).mean()


@torch.no_grad()
def euler_integrate(
    model: Velocity,
    z: torch.Tensor,
    t: torch.Tensor,
    dt: torch.Tensor,
    cid: torch.Tensor,
    steps: int = 10,
) -> torch.Tensor:
    z = z.clone()
    h = dt / steps
    for i in range(steps):
        t_i = t + i * h
        z = z + model(z, t_i, dt, cid) * h
    return z


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path: Path, model: Velocity, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), **extra}
    torch.save(payload, path)


def load_velocity(path: Path, device: torch.device | None = None) -> tuple[Velocity, dict]:
    device = device or get_device()
    payload = torch.load(path, map_location=device, weights_only=False)
    extra = {k: v for k, v in payload.items() if k != "state_dict"}
    model = Velocity(
        d=extra.get("d", 64),
        emb_dim=extra.get("emb_dim", 16),
        n_clusters=extra.get("n_clusters", 15),
        hidden=extra.get("hidden", [256, 256]),
        t_anchor=extra.get("t_anchor", 8.5),
        t_scale=extra.get("t_scale", 4.0),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, extra
