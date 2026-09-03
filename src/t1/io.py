from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np


def to_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def mean_X(adata: ad.AnnData) -> np.ndarray:
    mu = np.asarray(adata.X.mean(axis=0)).ravel()
    return mu.astype(np.float32, copy=False)


def load_adata(path: Path, backed: bool | str | None = None) -> ad.AnnData:
    if backed:
        return ad.read_h5ad(path, backed="r")
    return ad.read_h5ad(path)


def load_panel(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    genes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return genes or None


def write_gene_order(genes: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(genes) + "\n")
    np.save(path.with_suffix(".npy"), np.asarray(genes))


def align_to_panel(adata: ad.AnnData, panel: list[str]) -> ad.AnnData:
    current = list(adata.var_names)
    if current == panel:
        return adata
    missing = [g for g in panel if g not in adata.var_names]
    extra = [g for g in current if g not in panel]
    if missing:
        raise ValueError(
            f"Panel has {len(missing)} genes absent from AnnData "
            f"(first: {missing[:5]}). Extra in AnnData: {len(extra)}."
        )
    return adata[:, panel].copy()


def write_submission(
    X: np.ndarray,
    var_names: list[str],
    path: Path,
    clip_min: float = 0.0,
) -> ad.AnnData:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X, got {X.shape}")
    if X.shape[1] != len(var_names):
        raise ValueError(f"n_vars {X.shape[1]} != panel {len(var_names)}")
    if not np.isfinite(X).all():
        raise ValueError("Submission X contains NaN or Inf")
    X = np.clip(X, clip_min, None)
    adata = ad.AnnData(X)
    adata.var_names = var_names
    adata.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression="gzip")
    return adata


def assert_t1_submission(adata: ad.AnnData, cfg: dict) -> None:
    n, g = adata.n_obs, adata.n_vars
    if g != cfg["n_genes"]:
        raise AssertionError(f"n_vars={g}, expected {cfg['n_genes']}")
    if not (cfg["n_submit_min"] <= n <= cfg["n_submit_max"]):
        raise AssertionError(
            f"n_obs={n} outside [{cfg['n_submit_min']}, {cfg['n_submit_max']}]"
        )
    X = adata.X
    if hasattr(X, "min"):
        lo = float(X.min()) if not hasattr(X, "toarray") else float(X.min())
    else:
        lo = float(to_dense(X).min())
    if lo < -1e-6:
        raise AssertionError(f"negative values in X (min={lo})")
