"""T2 non-autonomous OT-CFM on the MERFISH panel PCA. Not T1's 32k-gene network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def hops_for(setting: str) -> list[tuple[str, str]]:
    """Adjacent training hops. Embryo interpolates; heart has interp + one-step extrap."""
    if setting == "embryo":
        return [("E6.75", "E7.25"), ("E7.25", "E8.0")]
    if setting == "heart":
        return [("E8.25", "E8.75"), ("E8.75", "E9.5")]
    raise ValueError(f"Unknown setting {setting}")


def t_anchor_for(setting: str) -> float:
    return 6.75 if setting == "embryo" else 8.25


class Velocity(nn.Module):
    """Non-autonomous OT-CFM velocity: v(z, t, Δt, cluster)."""

    def __init__(
        self,
        d: int = 32,
        emb_dim: int = 16,
        n_clusters: int = 15,
        hidden: list[int] | None = None,
        t_anchor: float = 6.75,
        t_scale: float = 2.0,
    ):
        super().__init__()
        hidden = hidden or [128, 128]
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
        d=extra.get("d", 32),
        emb_dim=extra.get("emb_dim", 16),
        n_clusters=extra.get("n_clusters", 15),
        hidden=extra.get("hidden", [128, 128]),
        t_anchor=extra.get("t_anchor", 6.75),
        t_scale=extra.get("t_scale", 2.0),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, extra


def _rbf(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    xx = (x * x).sum(-1, keepdim=True)
    yy = (y * y).sum(-1, keepdim=True).T
    dist = (xx + yy - 2.0 * x @ y.T).clamp_min(0.0)
    return torch.exp(-gamma * dist)


def mmd_unbiased(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: list[float] | None = None,
) -> torch.Tensor:
    n, m = x.size(0), y.size(0)
    if n < 2 or m < 2:
        return x.new_zeros(())
    with torch.no_grad():
        yy = ((y.unsqueeze(0) - y.unsqueeze(1)) ** 2).sum(-1)
        med = yy[yy > 0].median() if (yy > 0).any() else y.new_tensor(1.0)
        g0 = 1.0 / (2.0 * med.clamp_min(1e-8))
    if gammas is None:
        gammas = [s * float(g0) for s in (0.25, 0.5, 1.0, 2.0, 4.0)]
    acc = x.new_zeros(())
    for g in gammas:
        gamma = x.new_tensor(g)
        kxx = _rbf(x, x, gamma)
        kyy = _rbf(y, y, gamma)
        kxy = _rbf(x, y, gamma)
        mmd = kxx.fill_diagonal_(0).sum() / (n * (n - 1))
        mmd = mmd + kyy.fill_diagonal_(0).sum() / (m * (m - 1))
        mmd = mmd - 2.0 * kxy.mean()
        acc = acc + mmd
    return acc / len(gammas)


def flow_artifact_paths(out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    return out_dir / "pca.joblib", out_dir / "ckpt" / "flow.pt"
