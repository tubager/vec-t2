"""Joint OT-CFM on (PCA-z || xyz): geometry flows continuously; X can be manifold-projected.

Failed panel CFM decoded continuous z→X and blew variogram. Here the state includes
canonical xyz so transport is native ``(z, xyz)``. At predict time we optionally replace
decoded X with a nearest real cell from an anchor pool (preserves coexpression).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from t2.flow import Velocity, cfm_loss, euler_integrate, gene_cov_loss, torch_pca_decode


class JointVelocity(Velocity):
    """Same MLP as Velocity but input/output dim = z_dim + 3."""

    def __init__(self, z_dim: int = 32, **kwargs):
        super().__init__(d=int(z_dim) + 3, **kwargs)
        self.z_dim = int(z_dim)


def pack_state(z: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
    return torch.cat([z, xyz], dim=-1)


def unpack_state(s: torch.Tensor, z_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return s[:, :z_dim], s[:, z_dim:]


def canonicalize_xyz_np(C: np.ndarray, r_target: float) -> np.ndarray:
    """Center + isotropic scale to r_target (float64)."""
    X = np.asarray(C, dtype=np.float64)
    X = X - X.mean(axis=0)
    r = float(np.sqrt((X**2).sum(axis=1).mean()))
    return X * (float(r_target) / max(r, 1e-8))


def joint_cfm_loss(model: JointVelocity, s0, s1, t0, dt, cid, z_noise: float = 0.0):
    return cfm_loss(model, s0, s1, t0, dt, cid, z_noise=z_noise)


def gene_cov_from_state(
    model: JointVelocity,
    s0: torch.Tensor,
    t0: torch.Tensor,
    dt: torch.Tensor,
    cid: torch.Tensor,
    s1: torch.Tensor,
    components: torch.Tensor,
    mean: torch.Tensor,
    z_dim: int,
    steps: int = 4,
    n_genes: int = 64,
) -> torch.Tensor:
    """Endpoint gene-cov after Euler on joint state (z part only)."""
    from t2.flow import euler_integrate_grad

    s_hat = euler_integrate_grad(model, s0, t0, dt, cid, steps=steps)
    z_hat, _ = unpack_state(s_hat, z_dim)
    z_tgt, _ = unpack_state(s1, z_dim)
    return gene_cov_loss(z_hat, z_tgt, components, mean, n_genes=n_genes)


def project_x_to_pool(X_query: np.ndarray, X_pool: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Replace each query row with nearest pool row (L2). CPU chunked."""
    Q = np.asarray(X_query, dtype=np.float32)
    P = np.asarray(X_pool, dtype=np.float32)
    out = np.empty_like(Q)
    p2 = (P * P).sum(1)
    for i0 in range(0, len(Q), chunk):
        q = Q[i0 : i0 + chunk]
        q2 = (q * q).sum(1)[:, None]
        # (a-b)^2 = a2 + b2 - 2ab
        d = q2 + p2[None, :] - 2.0 * (q @ P.T)
        nn = np.argmin(d, axis=1)
        out[i0 : i0 + chunk] = P[nn]
    return out


def save_joint_ckpt(path: Path, model: JointVelocity, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_joint_ckpt(path: Path, device: torch.device | None = None) -> tuple[JointVelocity, dict]:
    device = device or torch.device("cpu")
    blob = torch.load(path, map_location=device, weights_only=False)
    z_dim = int(blob.get("z_dim", blob.get("d", 35) - 3))
    model = JointVelocity(
        z_dim=z_dim,
        emb_dim=int(blob.get("emb_dim", 16)),
        n_clusters=int(blob.get("n_clusters", 15)),
        hidden=list(blob.get("hidden", [128, 128])),
        t_anchor=float(blob.get("t_anchor", 6.75)),
        t_scale=float(blob.get("t_scale", 2.0)),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    model.eval()
    return model, blob
