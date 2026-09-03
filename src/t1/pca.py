from __future__ import annotations

from pathlib import Path

import anndata as ad
import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA

from t1.io import to_dense


class ExpressionPCA:
    """Incremental PCA over log1p expression; decode is clipped to be non-negative."""

    def __init__(self, n_components: int = 64, batch_size: int = 256):
        self.n_components = n_components
        self.batch_size = batch_size
        self.model = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        self.n_features_: int | None = None
        self.fitted_ = False

    def _iter_dense(self, adata: ad.AnnData, idx: np.ndarray):
        for start in range(0, len(idx), self.batch_size):
            sl = idx[start : start + self.batch_size]
            yield to_dense(adata[sl].X)

    def fit(
        self,
        stages: list[ad.AnnData],
        n_per_stage: int,
        seed: int = 0,
    ) -> "ExpressionPCA":
        rng = np.random.default_rng(seed)
        chunks = []
        for adata in stages:
            n = min(n_per_stage, adata.n_obs)
            idx = rng.choice(adata.n_obs, size=n, replace=False)
            chunks.append((adata, np.sort(idx)))
        # Partial fit in batches to avoid a 8k x 32k dense block.
        for adata, idx in chunks:
            for batch in self._iter_dense(adata, idx):
                self.model.partial_fit(batch)
                self.n_features_ = batch.shape[1]
        self.fitted_ = True
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        self._check()
        X = np.asarray(X, dtype=np.float32)
        return self.model.transform(X).astype(np.float32)

    def encode_adata(self, adata: ad.AnnData, idx: np.ndarray | None = None) -> np.ndarray:
        self._check()
        if idx is None:
            idx = np.arange(adata.n_obs)
        zs = []
        for start in range(0, len(idx), self.batch_size):
            sl = idx[start : start + self.batch_size]
            zs.append(self.encode(to_dense(adata[sl].X)))
        return np.concatenate(zs, axis=0)

    def decode(self, z: np.ndarray, clip_min: float = 0.0) -> np.ndarray:
        self._check()
        z = np.asarray(z, dtype=np.float32)
        x = self.model.inverse_transform(z).astype(np.float32)
        if clip_min is not None:
            np.clip(x, clip_min, None, out=x)
        return x

    def reconstruction_mse(self, X: np.ndarray) -> tuple[float, float]:
        """Return (pca_mse, mean_baseline_mse) on a dense block."""
        X = np.asarray(X, dtype=np.float32)
        rec = self.decode(self.encode(X), clip_min=None)
        pca_mse = float(np.mean((rec - X) ** 2))
        mu = X.mean(axis=0, keepdims=True)
        mean_mse = float(np.mean((X - mu) ** 2))
        return pca_mse, mean_mse

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ExpressionPCA":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        return obj

    def _check(self) -> None:
        if not self.fitted_:
            raise RuntimeError("ExpressionPCA is not fitted")
