"""Conditional noise→X flow. Time, cluster, and xyz; not L/R pair lerp.

OT-CFM interpolants (1-τ)x0+τ x1 of two real cells are chimeric (variogram).
This learns p(X | t, cluster, xyz) by flowing Gaussian noise to observed cells
at their native stage time. Endpoint batches are regularized with gene-cov and
variogram so interpolated t* cannot freely explode coexpression.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from t2.flow import cov_frobenius_loss  # noqa: F401


class CondGen(nn.Module):
    """Direct p(X | t, cluster, xyz): z + conditions → log1p, then expm1.

    CFM from isotropic noise under-shoots sparse MERFISH means. This decoder
    is trained on observed stages only; infer interpolates t in the embedding.
    """

    def __init__(
        self,
        n_genes: int,
        n_clusters: int,
        hidden: int = 512,
        emb_dim: int = 32,
        xyz_hidden: int = 64,
        z_dim: int = 32,
        t_anchor: float = 6.75,
        t_scale: float = 1.25,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.z_dim = int(z_dim)
        self.t_anchor = float(t_anchor)
        self.t_scale = float(t_scale)
        self.emb = nn.Embedding(n_clusters, emb_dim)
        self.xyz_net = nn.Sequential(
            nn.Linear(3, xyz_hidden),
            nn.SiLU(),
            nn.Linear(xyz_hidden, xyz_hidden),
            nn.SiLU(),
        )
        in_dim = z_dim + 1 + emb_dim + xyz_hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )

    def _t_n(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return (t - self.t_anchor) / max(self.t_scale, 1e-6)

    def forward(self, z, t_bio, cid, xyz, mu=None):
        e = self.emb(cid)
        hid = self.xyz_net(xyz)
        inp = torch.cat([z, self._t_n(t_bio), e, hid], dim=-1)
        pred = self.net(inp)
        if mu is not None:
            pred = pred + torch.log1p(mu.clamp_min(0.0))
        return pred

    def decode(self, z, t_bio, cid, xyz, mu=None):
        return torch.expm1(self.forward(z, t_bio, cid, xyz, mu)).clamp_min(0.0)


class CondVAE(nn.Module):
    """Encode identity from X (no t); decode p(X | z, t, cluster, xyz).

    Infer encodes the frozen backbone cell (L/R, not target-stage) and decodes
    at t0/t1 with the same z, then clock-lerps. Not a residual around a hop mix.
    """

    def __init__(
        self,
        n_genes: int,
        n_clusters: int,
        hidden: int = 512,
        emb_dim: int = 32,
        xyz_hidden: int = 64,
        z_dim: int = 32,
        t_anchor: float = 6.75,
        t_scale: float = 1.25,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.z_dim = int(z_dim)
        self.t_anchor = float(t_anchor)
        self.t_scale = float(t_scale)
        self.emb = nn.Embedding(n_clusters, emb_dim)
        self.xyz_net = nn.Sequential(
            nn.Linear(3, xyz_hidden),
            nn.SiLU(),
            nn.Linear(xyz_hidden, xyz_hidden),
            nn.SiLU(),
        )
        enc_in = n_genes + emb_dim + xyz_hidden
        self.enc = nn.Sequential(nn.Linear(enc_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, z_dim)
        self.logvar = nn.Linear(hidden, z_dim)
        dec_in = z_dim + 1 + emb_dim + xyz_hidden
        self.dec = nn.Sequential(
            nn.Linear(dec_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )

    def _t_n(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return (t - self.t_anchor) / max(self.t_scale, 1e-6)

    def encode(self, x, cid, xyz):
        e = self.emb(cid)
        h = self.xyz_net(xyz)
        hid = self.enc(torch.cat([torch.log1p(x.clamp_min(0.0)), e, h], dim=-1))
        return self.mu(hid), self.logvar(hid)

    def reparam(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar.clamp(-8, 8)) * torch.randn_like(mu)

    def decode_log(self, z, t_bio, cid, xyz):
        e = self.emb(cid)
        h = self.xyz_net(xyz)
        return self.dec(torch.cat([z, self._t_n(t_bio), e, h], dim=-1))

    def decode(self, z, t_bio, cid, xyz, mu=None):
        del mu
        return torch.expm1(self.decode_log(z, t_bio, cid, xyz)).clamp_min(0.0)


class CondAE(nn.Module):
    """Deterministic identity AE. z from X+cluster only; decode uses t and xyz.

    Reconstruction gate: encode→decode at native t must copy the cell
    (mae ~0.02, sparsity ~true) before any t* sampling is allowed.
    """

    def __init__(
        self,
        n_genes: int,
        n_clusters: int,
        hidden: int = 1024,
        emb_dim: int = 32,
        xyz_hidden: int = 64,
        z_dim: int = 128,
        t_anchor: float = 7.25,
        t_scale: float = 0.75,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.z_dim = int(z_dim)
        self.t_anchor = float(t_anchor)
        self.t_scale = float(t_scale)
        self.emb = nn.Embedding(n_clusters, emb_dim)
        self.xyz_net = nn.Sequential(
            nn.Linear(3, xyz_hidden),
            nn.SiLU(),
            nn.Linear(xyz_hidden, xyz_hidden),
            nn.SiLU(),
        )
        enc_in = n_genes + emb_dim
        self.enc = nn.Sequential(
            nn.Linear(enc_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(z_dim + emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )
        self.time_net = nn.Sequential(
            nn.Linear(z_dim + 1 + xyz_hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, n_genes),
        )
        nn.init.zeros_(self.time_net[-1].weight)
        nn.init.zeros_(self.time_net[-1].bias)

    def _t_n(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return (t - self.t_anchor) / max(self.t_scale, 1e-6)

    def encode(self, x, cid, xyz=None):
        del xyz
        e = self.emb(cid)
        z = self.enc(torch.cat([torch.log1p(x.clamp_min(0.0)), e], dim=-1))
        return z, z.new_zeros(z.shape)

    def decode_log(self, z, t_bio, cid, xyz):
        e = self.emb(cid)
        base = self.dec(torch.cat([z, e], dim=-1))
        resid = self.time_net(torch.cat([z, self._t_n(t_bio), self.xyz_net(xyz)], dim=-1))
        return base + resid

    def decode(self, z, t_bio, cid, xyz, mu=None):
        del mu
        return torch.expm1(self.decode_log(z, t_bio, cid, xyz)).clamp_min(0.0)


class CondFlow(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_clusters: int,
        hidden: int = 256,
        emb_dim: int = 16,
        xyz_hidden: int = 32,
        t_anchor: float = 6.75,
        t_scale: float = 1.25,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.t_anchor = float(t_anchor)
        self.t_scale = float(t_scale)
        self.emb = nn.Embedding(n_clusters, emb_dim)
        self.xyz_net = nn.Sequential(
            nn.Linear(3, xyz_hidden),
            nn.SiLU(),
            nn.Linear(xyz_hidden, xyz_hidden),
            nn.SiLU(),
        )
        in_dim = n_genes + 2 + emb_dim + xyz_hidden  # x, tau, t_n, emb, xyz
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_genes),
        )

    def _t_n(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return (t - self.t_anchor) / max(self.t_scale, 1e-6)

    def forward(self, x, tau, t_bio, cid, xyz):
        if tau.ndim == 1:
            tau = tau.unsqueeze(-1)
        e = self.emb(cid)
        z = self.xyz_net(xyz)
        inp = torch.cat([x, tau, self._t_n(t_bio), e, z], dim=-1)
        return self.net(inp)


def variogram_loss(xp: torch.Tensor, xt: torch.Tensor, n_pairs: int = 2048, p: float = 0.5) -> torch.Tensor:
    if xp.size(0) < 2:
        return xp.new_zeros(())
    g = xp.size(1)
    i = torch.randint(0, g, (n_pairs,), device=xp.device)
    j = torch.randint(0, g, (n_pairs,), device=xp.device)
    va = ((xp[:, i] - xp[:, j]).abs() + 1e-4).pow(p).mean(0)
    vb = ((xt[:, i] - xt[:, j]).abs() + 1e-4).pow(p).mean(0)
    return ((va - vb) ** 2).mean()


def integrate(model: CondFlow, x: torch.Tensor, t_bio, cid, xyz, steps: int = 8) -> torch.Tensor:
    n = x.size(0)
    h = 1.0 / max(int(steps), 1)
    for i in range(int(steps)):
        tau = x.new_full((n, 1), i * h)
        x = x + model(x, tau, t_bio, cid, xyz) * h
    return x


def save_cond_flow(path: Path, model: CondFlow, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_cond_flow(path: Path, device: torch.device | None = None):
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    extra = {k: v for k, v in payload.items() if k != "state_dict"}
    arch = str(extra.get("arch") or "flow")
    common = dict(
        n_genes=int(extra["n_genes"]),
        n_clusters=int(extra["n_clusters"]),
        t_anchor=float(extra.get("t_anchor", 6.75)),
        t_scale=float(extra.get("t_scale", 1.25)),
    )
    if arch == "gen":
        model = CondGen(
            **common,
            hidden=int(extra.get("hidden", 512)),
            emb_dim=int(extra.get("emb_dim", 32)),
            xyz_hidden=int(extra.get("xyz_hidden", 64)),
            z_dim=int(extra.get("z_dim", 32)),
        )
    elif arch == "ae":
        model = CondAE(
            **common,
            hidden=int(extra.get("hidden", 1024)),
            emb_dim=int(extra.get("emb_dim", 32)),
            xyz_hidden=int(extra.get("xyz_hidden", 64)),
            z_dim=int(extra.get("z_dim", 128)),
        )
    else:
        model = CondFlow(
            **common,
            hidden=int(extra.get("hidden", 256)),
            emb_dim=int(extra.get("emb_dim", 16)),
            xyz_hidden=int(extra.get("xyz_hidden", 32)),
        )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, extra
