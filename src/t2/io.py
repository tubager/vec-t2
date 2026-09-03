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


def spatial_xyz(adata: ad.AnnData) -> np.ndarray:
    if "spatial_3D" not in adata.obsm:
        raise KeyError("AnnData is missing obsm['spatial_3D']")
    xyz = np.asarray(adata.obsm["spatial_3D"], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"spatial_3D must be (n, 3), got {xyz.shape}")
    return xyz


def reindex_X(X, var_names: list[str], panel: list[str], fill: float = 0.0) -> np.ndarray:
    """Reorder / subset gene columns to `panel`. Missing genes are filled."""
    name_to_idx = {g: i for i, g in enumerate(var_names)}
    dense = to_dense(X) if any(g not in name_to_idx for g in panel) else None
    if dense is None and list(var_names) == list(panel):
        return to_dense(X)
    if dense is None:
        present = [g for g in panel if g in name_to_idx]
        if len(present) == len(panel):
            cols = [name_to_idx[g] for g in panel]
            if hasattr(X, "toarray"):
                return np.asarray(X[:, cols].toarray(), dtype=np.float32)
            return np.asarray(X[:, cols], dtype=np.float32)
        dense = to_dense(X)
    out = np.full((dense.shape[0], len(panel)), fill, dtype=np.float32)
    for j, g in enumerate(panel):
        i = name_to_idx.get(g)
        if i is not None:
            out[:, j] = dense[:, i]
    return out


def reindex_adata(adata: ad.AnnData, panel: list[str], fill: float = 0.0) -> ad.AnnData:
    current = list(adata.var_names)
    if current == panel:
        return adata
    missing = [g for g in panel if g not in adata.var_names]
    if not missing:
        out = adata[:, panel].copy()
        if "spatial_3D" in adata.obsm and "spatial_3D" not in out.obsm:
            out.obsm["spatial_3D"] = np.asarray(adata.obsm["spatial_3D"])
        return out
    X = reindex_X(adata.X, current, panel, fill=fill)
    out = ad.AnnData(X)
    out.var_names = panel
    out.obs = adata.obs.copy()
    out.obs_names = adata.obs_names.copy()
    for key, val in adata.obsm.items():
        out.obsm[key] = np.asarray(val)
    return out


def write_t2(X, xyz, var_names, path, clip_min: float = 0.0) -> ad.AnnData:
    X = np.clip(np.asarray(X, dtype=np.float32), clip_min, None)
    xyz = np.asarray(xyz, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X, got {X.shape}")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"spatial_3D must be (n, 3), got {xyz.shape}")
    if X.shape[0] != xyz.shape[0]:
        raise ValueError(f"n_obs mismatch: X={X.shape[0]} xyz={xyz.shape[0]}")
    if X.shape[1] != len(var_names):
        raise ValueError(f"n_vars {X.shape[1]} != panel {len(var_names)}")
    if not np.isfinite(X).all():
        raise ValueError("Submission X contains NaN or Inf")
    if not np.isfinite(xyz).all():
        raise ValueError("spatial_3D contains NaN or Inf")
    adata = ad.AnnData(X)
    adata.var_names = list(var_names)
    adata.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    adata.obsm["spatial_3D"] = xyz
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, compression="gzip")
    return adata


def assert_t2_submission(
    adata: ad.AnnData,
    n_genes: int,
    n_min: int,
    n_max: int,
) -> None:
    n, g = adata.n_obs, adata.n_vars
    if g != n_genes:
        raise AssertionError(f"n_vars={g}, expected {n_genes}")
    if not (n_min <= n <= n_max):
        raise AssertionError(f"n_obs={n} outside [{n_min}, {n_max}]")
    if "spatial_3D" not in adata.obsm:
        raise AssertionError("missing obsm['spatial_3D']")
    xyz = np.asarray(adata.obsm["spatial_3D"])
    if xyz.shape != (n, 3):
        raise AssertionError(f"spatial_3D shape {xyz.shape} != ({n}, 3)")
    if not np.isfinite(xyz).all():
        raise AssertionError("spatial_3D has NaN/Inf")
    X = adata.X
    lo = float(X.min()) if not hasattr(X, "toarray") else float(np.asarray(X.min()))
    if lo < -1e-6:
        raise AssertionError(f"negative values in X (min={lo})")
    if hasattr(X, "toarray"):
        block = np.asarray(X[: min(32, n)].toarray())
    else:
        block = np.asarray(X[: min(32, n)])
    if not np.isfinite(block).all():
        raise AssertionError("X contains NaN/Inf")
