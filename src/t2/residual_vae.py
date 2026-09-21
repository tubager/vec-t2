"""Cluster-conditional residual VAE in gene space (no PCA-32 bottleneck).

X = μ(t, cluster) + r, where r is decoded from a latent conditioned on (t, cluster).
Preserves gene-gene structure better than OT-CFM on PCA-32 absolute expression.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class ResidualVAE(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_clusters: int,
        latent: int = 64,
        hidden: int = 512,
        emb_dim: int = 16,
        t_anchor: float = 6.75,
        t_scale: float = 2.0,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.latent = int(latent)
        self.t_anchor = float(t_anchor)
        self.t_scale = float(t_scale)
        self.cluster_emb = nn.Embedding(n_clusters, emb_dim)
        cond = 2 + emb_dim  # t_n, ones, emb
        self.enc = nn.Sequential(
            nn.Linear(n_genes + cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent + cond, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )

    def _cond(self, t: torch.Tensor, cid: torch.Tensor) -> torch.Tensor:
        t_n = (t - self.t_anchor) / max(self.t_scale, 1e-6)
        if t_n.ndim == 1:
            t_n = t_n.unsqueeze(-1)
        ones = torch.ones_like(t_n)
        return torch.cat([t_n, ones, self.cluster_emb(cid)], dim=-1)

    def encode(self, r: torch.Tensor, t: torch.Tensor, cid: torch.Tensor):
        h = self.enc(torch.cat([r, self._cond(t, cid)], dim=-1))
        return self.mu(h), self.logvar(h)

    def reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor, t: torch.Tensor, cid: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.cat([z, self._cond(t, cid)], dim=-1))

    def forward(self, r: torch.Tensor, t: torch.Tensor, cid: torch.Tensor):
        mu, logvar = self.encode(r, t, cid)
        z = self.reparam(mu, logvar)
        return self.decode(z, t, cid), mu, logvar

    @torch.no_grad()
    def sample(self, t: torch.Tensor, cid: torch.Tensor, n: int | None = None) -> torch.Tensor:
        if n is None:
            n = int(cid.size(0))
        z = torch.randn(n, self.latent, device=cid.device, dtype=next(self.parameters()).dtype)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        if t.size(0) == 1 and n > 1:
            t = t.expand(n, -1)
        if cid.size(0) == 1 and n > 1:
            cid = cid.expand(n)
        return self.decode(z, t, cid)


def vae_loss(
    r: torch.Tensor,
    r_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 0.01,
    cov_w: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    recon = ((r_hat - r) ** 2).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    loss = recon + float(beta) * kl
    stats = {"recon": float(recon.detach()), "kl": float(kl.detach())}
    if cov_w > 0 and r.size(0) >= 4:
        p = r_hat - r_hat.mean(0, keepdim=True)
        t = r - r.mean(0, keepdim=True)
        # gene subset for speed
        g = r.size(1)
        k = min(96, g)
        idx = torch.randperm(g, device=r.device)[:k]
        p, t = p[:, idx], t[:, idx]
        denom = float(max(r.size(0) - 1, 1))
        cp = (p.T @ p) / denom
        ct = (t.T @ t) / denom
        cov = ((cp - ct) ** 2).mean()
        loss = loss + float(cov_w) * cov
        stats["cov"] = float(cov.detach())
    stats["loss"] = float(loss.detach())
    return loss, stats


def save_vae(path: Path, model: ResidualVAE, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_vae(path: Path, device: torch.device | None = None) -> tuple[ResidualVAE, dict]:
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    extra = {k: v for k, v in payload.items() if k != "state_dict"}
    model = ResidualVAE(
        n_genes=int(extra["n_genes"]),
        n_clusters=int(extra["n_clusters"]),
        latent=int(extra.get("latent", 64)),
        hidden=int(extra.get("hidden", 512)),
        emb_dim=int(extra.get("emb_dim", 16)),
        t_anchor=float(extra.get("t_anchor", 6.75)),
        t_scale=float(extra.get("t_scale", 2.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, extra
