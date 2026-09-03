from __future__ import annotations

import anndata as ad
import numpy as np

from t1.io import mean_X, to_dense, write_submission


def subsample(
    adata: ad.AnnData,
    n: int,
    seed: int = 0,
    stratify: bool = True,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n_obs = adata.n_obs
    if n_obs == 0:
        raise ValueError("Cannot subsample empty AnnData")
    if stratify and "celltype" in adata.obs:
        idx = []
        types = adata.obs["celltype"].astype(str)
        props = types.value_counts(normalize=True)
        for t, p in props.items():
            k = max(1, int(round(p * n)))
            cand = np.flatnonzero(types.values == t)
            take = rng.choice(cand, size=min(k, len(cand)), replace=len(cand) < k)
            idx.append(take)
        idx = np.concatenate(idx)
        if len(idx) > n:
            idx = rng.choice(idx, size=n, replace=False)
        elif len(idx) < n:
            extra = rng.choice(n_obs, size=n - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
    else:
        idx = rng.choice(n_obs, size=n, replace=n_obs < n)
    return adata[idx].copy()


def copy_last(
    adata: ad.AnnData,
    n: int = 3000,
    seed: int = 0,
    stratify: bool = True,
) -> ad.AnnData:
    sub = subsample(adata, n=n, seed=seed, stratify=stratify)
    X = to_dense(sub.X)
    out = ad.AnnData(X)
    out.var_names = adata.var_names
    out.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    return out


def add_delta(X: np.ndarray, delta: np.ndarray, alpha: float = 1.0, clip_min: float = 0.0) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32).ravel()
    return np.clip(X + alpha * delta, clip_min, None)


def pseudobulk_shift(
    src: ad.AnnData,
    ref_prev: ad.AnnData,
    ref_next: ad.AnnData,
    n: int = 3000,
    alpha: float = 1.0,
    seed: int = 0,
    clip_min: float = 0.0,
    stratify: bool = True,
) -> ad.AnnData:
    """Translate subsampled `src` cells by alpha * (mean(ref_next) - mean(ref_prev))."""
    sub = subsample(src, n=n, seed=seed, stratify=stratify)
    delta = mean_X(ref_next) - mean_X(ref_prev)
    X = add_delta(to_dense(sub.X), delta, alpha=alpha, clip_min=clip_min)
    out = ad.AnnData(X)
    out.var_names = src.var_names
    out.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    return out


def resample_types_then_noise(
    adata: ad.AnnData,
    n: int = 3000,
    seed: int = 0,
    noise_std: float = 0.05,
    clip_min: float = 0.0,
) -> ad.AnnData:
    """Fair one-step baseline: stratified resample + independent gene noise."""
    sub = copy_last(adata, n=n, seed=seed, stratify=True)
    rng = np.random.default_rng(seed + 1)
    noise = rng.normal(0.0, noise_std, size=sub.X.shape).astype(np.float32)
    X = np.clip(to_dense(sub.X) + noise, clip_min, None)
    sub.X = X
    return sub


def save_pred(adata: ad.AnnData, path, clip_min: float = 0.0):
    return write_submission(to_dense(adata.X), list(adata.var_names), path, clip_min=clip_min)
