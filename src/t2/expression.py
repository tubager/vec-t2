"""Expression channel on the MERFISH panel: composition, residual shift, optional PCA OT-CFM."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np

from t1.baselines import add_delta
from t2.clusters import (
    birth_clusters,
    birth_progenitor,
    cluster_id_map,
    cluster_order,
    fov_lost_clusters,
    labels_to_clusters,
)
from t2.geometry import scale_cloud
from t2.io import mean_X, spatial_xyz, to_dense


class CompositionT2:
    """Simplex interpolation on working clusters; freeze + FOV-zero on heart extrap."""

    def __init__(self, setting: str, times: dict[str, float], eps: float = 1e-6):
        self.setting = setting
        self.times = {k: float(v) for k, v in times.items()}
        self.eps = eps
        self.order = cluster_order(setting)
        self.pi_by_t: dict[float, dict[str, float]] = {}
        self.counts_by_t: dict[float, dict[str, int]] = {}
        self.stage_t: dict[str, float] = {}

    def fit(self, stage_labels: dict[str, object], stage_times: dict[str, float]) -> "CompositionT2":
        self.stage_t = {k: float(v) for k, v in stage_times.items()}
        for stage, labels in stage_labels.items():
            t = self.stage_t[stage]
            clusters = labels_to_clusters(labels, self.setting)
            counts = dict(Counter(clusters))
            n = max(len(clusters), 1)
            pi = {k: counts.get(k, 0) / n for k in self.order}
            self.counts_by_t[t] = {k: int(counts.get(k, 0)) for k in self.order}
            self.pi_by_t[t] = pi
        return self

    def _anchors(self, t: float, allowed_times: list[float] | None = None) -> tuple[float, ...]:
        ts = sorted(self.pi_by_t)
        if allowed_times is not None:
            allow = set(float(x) for x in allowed_times)
            ts = [x for x in ts if x in allow]
        if not ts:
            raise RuntimeError("CompositionT2 has no fitted stages")
        return tuple(ts)

    def probs(self, t: float, allowed_times: list[float] | None = None) -> dict[str, float]:
        ts = self._anchors(t, allowed_times)
        if t <= ts[0]:
            pi = dict(self.pi_by_t[ts[0]])
        elif t >= ts[-1]:
            pi = dict(self.pi_by_t[ts[-1]])
        else:
            t_left = max(x for x in ts if x <= t)
            t_right = min(x for x in ts if x >= t)
            if t_left == t_right:
                pi = dict(self.pi_by_t[t_left])
            else:
                w = (t - t_left) / (t_right - t_left)
                pl = self.pi_by_t[t_left]
                pr = self.pi_by_t[t_right]
                pi = {k: (1.0 - w) * pl[k] + w * pr[k] for k in self.order}

        if self.setting == "heart" and t >= 9.5 - 1e-9:
            for k in fov_lost_clusters("heart"):
                pi[k] = 0.0
            # Freeze at E9.5 mix (already the case when 9.5 is the last fitted time).
            if 9.5 in self.pi_by_t and t > 9.5 + 1e-9:
                pi = dict(self.pi_by_t[9.5])
                for k in fov_lost_clusters("heart"):
                    pi[k] = 0.0
        return self._renorm(pi)

    @staticmethod
    def _renorm(pi: dict[str, float]) -> dict[str, float]:
        clipped = {k: max(float(v), 0.0) for k, v in pi.items()}
        total = sum(clipped.values())
        if total <= 0:
            raise RuntimeError("Composition probabilities summed to 0")
        return {k: v / total for k, v in clipped.items()}

    def allocate(self, t: float, n: int, rng: np.random.Generator, allowed_times=None) -> dict[str, int]:
        pi = self.probs(t, allowed_times=allowed_times)
        keys = [k for k, p in pi.items() if p > 0]
        p = np.array([pi[k] for k in keys], dtype=np.float64)
        p = p / p.sum()
        counts = rng.multinomial(n, p)
        return {k: int(c) for k, c in zip(keys, counts) if c > 0}

    def to_dict(self) -> dict:
        return {
            "setting": self.setting,
            "times": self.times,
            "eps": self.eps,
            "order": list(self.order),
            "pi_by_t": {str(k): v for k, v in self.pi_by_t.items()},
            "counts_by_t": {str(k): v for k, v in self.counts_by_t.items()},
            "stage_t": self.stage_t,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "CompositionT2":
        payload = json.loads(Path(path).read_text())
        obj = cls(payload["setting"], payload["times"], eps=payload.get("eps", 1e-6))
        obj.pi_by_t = {float(k): v for k, v in payload["pi_by_t"].items()}
        obj.counts_by_t = {float(k): {kk: int(vv) for kk, vv in v.items()} for k, v in payload["counts_by_t"].items()}
        obj.stage_t = payload.get("stage_t", {})
        return obj


def subsample(adata: ad.AnnData, n: int, seed: int = 0, stratify: bool = True) -> ad.AnnData:
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
    out = adata[idx].copy()
    out.obs_names_make_unique()
    return out


def copy_last(adata: ad.AnnData, n: int, seed: int = 0, stratify: bool = True) -> ad.AnnData:
    sub = subsample(adata, n=n, seed=seed, stratify=stratify)
    X = to_dense(sub.X)
    xyz = spatial_xyz(sub).astype(np.float32)
    out = ad.AnnData(X)
    out.var_names = list(adata.var_names)
    out.obs_names = [f"pred_{i}" for i in range(X.shape[0])]
    out.obsm["spatial_3D"] = xyz
    if "celltype" in sub.obs:
        out.obs["celltype"] = sub.obs["celltype"].to_numpy()
    return out


def scale_copy(adata: ad.AnnData, n: int, r_target: float, seed: int = 0) -> ad.AnnData:
    out = copy_last(adata, n=n, seed=seed)
    out.obsm["spatial_3D"] = scale_cloud(out.obsm["spatial_3D"], r_target)
    return out


def shift_scale(
    src: ad.AnnData,
    ref_prev: ad.AnnData,
    ref_next: ad.AnnData,
    r_target: float,
    n: int,
    alpha: float = 1.0,
    seed: int = 0,
    clip_min: float = 0.0,
) -> ad.AnnData:
    """Expression: clip(src + α Δ). Geometry: scale_copy of src."""
    out = scale_copy(src, n=n, r_target=r_target, seed=seed)
    delta = mean_X(ref_next) - mean_X(ref_prev)
    out.X = add_delta(to_dense(out.X), delta, alpha=alpha, clip_min=clip_min)
    return out


def libraries_from_adata(adata: ad.AnnData, setting: str) -> dict[str, np.ndarray]:
    clusters = np.asarray(labels_to_clusters(adata.obs["celltype"], setting))
    X = to_dense(adata.X)
    lib = {}
    for k in np.unique(clusters):
        lib[k] = X[clusters == k]
    return lib


def cluster_mean_delta(
    left: ad.AnnData,
    right: ad.AnnData,
    setting: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Global Δ and per-working-cluster Δ = mean(right_k) − mean(left_k).

    Clusters present on only one side fall back to the global Δ.
    """
    global_delta = mean_X(right) - mean_X(left)
    cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], setting))
    cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], setting))
    Xl = to_dense(left.X)
    Xr = to_dense(right.X)
    by_k: dict[str, np.ndarray] = {}
    for k in set(cl_l.tolist()) | set(cl_r.tolist()):
        m_l = cl_l == k
        m_r = cl_r == k
        if m_l.any() and m_r.any():
            by_k[str(k)] = (Xr[m_r].mean(axis=0) - Xl[m_l].mean(axis=0)).astype(np.float32)
        else:
            by_k[str(k)] = global_delta
    return global_delta.astype(np.float32, copy=False), by_k


def apply_cluster_delta(
    X: np.ndarray,
    clusters: np.ndarray,
    global_delta: np.ndarray,
    delta_by_cluster: dict[str, np.ndarray] | None,
    alpha: float,
    clip_min: float,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    clusters = np.asarray(clusters)
    if not delta_by_cluster:
        return add_delta(X, global_delta, alpha=alpha, clip_min=clip_min)
    out = np.empty_like(X)
    for k in np.unique(clusters):
        mask = clusters == k
        d = delta_by_cluster.get(str(k), global_delta)
        out[mask] = add_delta(X[mask], d, alpha=alpha, clip_min=clip_min)
    return out


def _index_for_cluster(clusters: np.ndarray, name: str) -> np.ndarray:
    return np.flatnonzero(np.asarray(clusters) == name)


def sample_paired_cells(
    stage_X: dict[str, np.ndarray],
    stage_xyz: dict[str, np.ndarray],
    stage_clusters: dict[str, np.ndarray],
    alloc: dict[str, int],
    rng: np.random.Generator,
    *,
    geom_stage: str,
    expr_weights: dict[str, float],
    progenitors: dict[str, str] | None = None,
    birth: frozenset[str] | None = None,
    sigma_birth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw (X, xyz) from the same source cell whenever the cluster exists on geom_stage.

    Birth / missing clusters take xyz from the progenitor on `geom_stage` and X
    from any stage that actually has the cluster. Never samples xyz from the
    whole-embryo blob.
    """
    progenitors = progenitors or {}
    birth = birth or frozenset()
    expr_stages = [s for s, w in expr_weights.items() if w > 0 and s in stage_X]

    def _xyz_pool(cluster: str) -> tuple[str, np.ndarray] | None:
        for key in (cluster, progenitors.get(cluster)):
            if not key:
                continue
            if geom_stage in stage_clusters:
                idx = _index_for_cluster(stage_clusters[geom_stage], key)
                if len(idx):
                    return geom_stage, idx
            for stage, cl in stage_clusters.items():
                idx = _index_for_cluster(cl, key)
                if len(idx):
                    return stage, idx
        return None

    def _expr_pool(cluster: str) -> list[tuple[str, np.ndarray, float]]:
        keys = [cluster]
        if cluster in progenitors:
            keys.append(progenitors[cluster])
        for key in keys:
            parts = []
            for stage in expr_stages:
                idx = _index_for_cluster(stage_clusters.get(stage, np.array([])), key)
                if len(idx):
                    parts.append((stage, idx, float(expr_weights.get(stage, 0.0))))
            if parts:
                return parts
            parts = []
            for stage, cl in stage_clusters.items():
                idx = _index_for_cluster(cl, key)
                if len(idx):
                    parts.append((stage, idx, 1.0))
            if parts:
                return parts
        return []

    rows_x, rows_xyz, rows_cl = [], [], []
    for cluster, count in alloc.items():
        xyz_src = _xyz_pool(cluster)
        expr_src = _expr_pool(cluster)
        if xyz_src is None or not expr_src:
            raise RuntimeError(f"No paired template for cluster {cluster}")
        xyz_stage, xyz_idx = xyz_src
        w = np.array([p[2] for p in expr_src], dtype=np.float64)
        w = np.clip(w, 0, None)
        if w.sum() <= 0:
            w = np.ones(len(expr_src), dtype=np.float64)
        w = w / w.sum()
        n_from = rng.multinomial(count, w)
        x_chunks, y_chunks = [], []
        for (st, idx, _), n_i in zip(expr_src, n_from):
            if n_i == 0:
                continue
            take = rng.choice(idx, size=n_i, replace=len(idx) < n_i)
            x_chunks.append(stage_X[st][take])
            if st == xyz_stage:
                y_chunks.append(stage_xyz[st][take])
            else:
                take_xyz = rng.choice(xyz_idx, size=n_i, replace=len(xyz_idx) < n_i)
                y_chunks.append(stage_xyz[xyz_stage][take_xyz])
        x = np.concatenate(x_chunks, axis=0)
        xyz = np.concatenate(y_chunks, axis=0)
        if cluster in birth and sigma_birth > 0:
            x = x + rng.normal(0.0, sigma_birth, size=x.shape).astype(np.float32)
        rows_x.append(np.asarray(x, dtype=np.float32))
        rows_xyz.append(np.asarray(xyz, dtype=np.float32))
        rows_cl.append(np.full(count, cluster, dtype=object))

    return (
        np.concatenate(rows_x, axis=0),
        np.concatenate(rows_xyz, axis=0),
        np.concatenate(rows_cl, axis=0),
    )


def _blocks_for_cluster(
    libraries: dict[str, dict[str, np.ndarray]],
    cluster: str,
    source_weights: dict[str, float],
    progenitors: dict[str, str],
) -> tuple[list[np.ndarray], list[float]]:
    stages = [s for s, w in source_weights.items() if w > 0]
    keys = [cluster]
    if cluster in progenitors:
        keys.append(progenitors[cluster])
    for key in keys:
        parts, weights = [], []
        for stage in stages:
            block = libraries.get(stage, {}).get(key)
            if block is not None and len(block) > 0:
                parts.append(block)
                weights.append(source_weights[stage])
        if parts:
            return parts, weights
        for stage, lib in libraries.items():
            block = lib.get(key)
            if block is not None and len(block) > 0:
                parts.append(block)
                weights.append(1.0)
        if parts:
            return parts, weights
    # Last resort: any cells from weighted stages, then any stage.
    for stage in stages:
        lib = libraries.get(stage, {})
        if lib:
            return [next(iter(lib.values()))], [1.0]
    for lib in libraries.values():
        if lib:
            return [next(iter(lib.values()))], [1.0]
    raise RuntimeError(f"No expression library for cluster {cluster}")


def sample_expression(
    libraries: dict[str, dict[str, np.ndarray]],
    alloc: dict[str, int],
    rng: np.random.Generator,
    *,
    source_weights: dict[str, float],
    delta: np.ndarray,
    alpha: float,
    clip_min: float,
    birth: frozenset[str],
    sigma_birth: float,
    progenitors: dict[str, str] | None = None,
    apply_delta: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, cluster_id per cell). libraries[stage][cluster] = (n, G)."""
    progenitors = progenitors or {}
    rows = []
    cl_out = []
    for cluster, count in alloc.items():
        parts, weights = _blocks_for_cluster(libraries, cluster, source_weights, progenitors)
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        n_from = rng.multinomial(count, w)
        chunks = []
        for block, n_i in zip(parts, n_from):
            if n_i == 0:
                continue
            idx = rng.choice(len(block), size=n_i, replace=len(block) < n_i)
            chunks.append(block[idx])
        x = np.concatenate(chunks, axis=0).astype(np.float32, copy=True)
        if cluster in birth and sigma_birth > 0:
            x = x + rng.normal(0.0, sigma_birth, size=x.shape).astype(np.float32)
        rows.append(x)
        cl_out.append(np.full(count, cluster, dtype=object))
    X = np.concatenate(rows, axis=0)
    clusters = np.concatenate(cl_out, axis=0)
    if apply_delta:
        X = add_delta(X, delta, alpha=alpha, clip_min=clip_min)
    return X, clusters


def transport_expression(
    X: np.ndarray,
    clusters: np.ndarray,
    pca,
    model,
    trained: set[str],
    setting: str,
    t: float,
    dt: float,
    steps: int = 10,
    skip: frozenset[str] | None = None,
) -> np.ndarray:
    """Euler-integrate panel PCA latents within trained working clusters, then decode.

    Birth clusters and anything not in `trained` are left as sampled expression.
    """
    if model is None or abs(dt) < 1e-12:
        return X
    import torch

    from t2.flow import euler_integrate

    skip = skip or birth_clusters(setting)
    ids = cluster_id_map(setting)
    device = next(model.parameters()).device
    z = pca.encode(np.asarray(X, dtype=np.float32))
    clusters = np.asarray(clusters)
    for k in np.unique(clusters):
        name = str(k)
        if name not in trained or name in skip or name not in ids:
            continue
        mask = clusters == k
        zt = torch.from_numpy(z[mask]).to(device)
        n = zt.size(0)
        t_t = torch.full((n, 1), t, device=device, dtype=zt.dtype)
        dt_t = torch.full((n, 1), dt, device=device, dtype=zt.dtype)
        cid = torch.full((n,), ids[name], device=device, dtype=torch.long)
        with torch.no_grad():
            z[mask] = euler_integrate(model, zt, t_t, dt_t, cid, steps=steps).cpu().numpy()
    return pca.decode(z, clip_min=None)


def project_t1_delta(panel: list[str], delta_path: Path, genes_path: Path) -> np.ndarray | None:
    if not delta_path.exists() or not genes_path.exists():
        return None
    delta = np.load(delta_path).astype(np.float32).ravel()
    genes = [line.strip() for line in genes_path.read_text().splitlines() if line.strip()]
    if len(genes) != len(delta):
        return None
    idx = {g: i for i, g in enumerate(genes)}
    out = np.zeros(len(panel), dtype=np.float32)
    n_hit = 0
    for j, g in enumerate(panel):
        i = idx.get(g)
        if i is not None:
            out[j] = delta[i]
            n_hit += 1
    if n_hit == 0:
        return None
    return out


def combine_delta(merfish_delta: np.ndarray, t1_delta: np.ndarray | None, weight: float) -> np.ndarray:
    merfish_delta = np.asarray(merfish_delta, dtype=np.float32).ravel()
    if t1_delta is None or weight <= 0:
        return merfish_delta
    t1_delta = np.asarray(t1_delta, dtype=np.float32).ravel()
    w = float(weight)
    return ((1.0 - w) * merfish_delta + w * t1_delta).astype(np.float32)
