from __future__ import annotations

import numpy as np
import torch


def rbf_kernel(x: torch.Tensor, y: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    # x: (n, d), y: (m, d), gamma: scalar
    xx = (x * x).sum(-1, keepdim=True)
    yy = (y * y).sum(-1, keepdim=True).T
    dist = xx + yy - 2.0 * x @ y.T
    dist = dist.clamp_min(0.0)
    return torch.exp(-gamma * dist)


def mmd_unbiased(
    x: torch.Tensor,
    y: torch.Tensor,
    gammas: list[float] | None = None,
) -> torch.Tensor:
    """Average unbiased MMD^2 over a few RBF bandwidths (scorer-like)."""
    n, m = x.size(0), y.size(0)
    if n < 2 or m < 2:
        return x.new_zeros(())
    with torch.no_grad():
        # median heuristic on y
        yy = ((y.unsqueeze(0) - y.unsqueeze(1)) ** 2).sum(-1)
        med = yy[yy > 0].median() if (yy > 0).any() else y.new_tensor(1.0)
        g0 = 1.0 / (2.0 * med.clamp_min(1e-8))
    if gammas is None:
        gammas = [s * float(g0) for s in (0.25, 0.5, 1.0, 2.0, 4.0)]
    acc = x.new_zeros(())
    for g in gammas:
        gamma = x.new_tensor(g)
        kxx = rbf_kernel(x, x, gamma)
        kyy = rbf_kernel(y, y, gamma)
        kxy = rbf_kernel(x, y, gamma)
        # drop diagonal
        mmd = kxx.fill_diagonal_(0).sum() / (n * (n - 1))
        mmd = mmd + kyy.fill_diagonal_(0).sum() / (m * (m - 1))
        mmd = mmd - 2.0 * kxy.mean()
        acc = acc + mmd
    return acc / len(gammas)


def cosine_lfc_loss(pred: torch.Tensor, ref: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """1 - cosine(mean(pred)-mean(ref), mean(true)-mean(ref))."""
    lfc_p = pred.mean(0) - ref.mean(0)
    lfc_t = true.mean(0) - ref.mean(0)
    num = (lfc_p * lfc_t).sum()
    den = lfc_p.norm() * lfc_t.norm() + 1e-8
    return 1.0 - num / den


def variogram_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    n_pairs: int = 4096,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    g = pred.size(1)
    device = pred.device
    i = torch.randint(0, g, (n_pairs,), device=device, generator=generator)
    j = torch.randint(0, g, (n_pairs,), device=device, generator=generator)
    same = i == j
    if same.any():
        j = torch.where(same, (j + 1) % g, j)
    vg_p = 0.5 * (pred[:, i] - pred[:, j]).abs().mean(0)
    vg_t = 0.5 * (true[:, i] - true[:, j]).abs().mean(0)
    return ((vg_p - vg_t) ** 2).mean()


def subsample_rows(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]
