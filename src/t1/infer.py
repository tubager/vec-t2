from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anndata as ad
import numpy as np

from t1.baselines import add_delta
from t1.clusters import (
    BIRTH_CLUSTERS,
    BIRTH_PROGENITOR,
    CLUSTER_ORDER,
    DISSECT_CLUSTER,
    FLOW_CLUSTERS,
    cluster_id_map,
    labels_to_clusters,
)
from t1.composition import CompositionModel
from t1.io import to_dense
from t1.pca import ExpressionPCA

if TYPE_CHECKING:
    from t1.flow import Velocity


@dataclass
class Libraries:
    z_by_stage: dict[str, dict[str, np.ndarray]]  # stage -> cluster -> z


def build_libraries(
    z_85: np.ndarray,
    z_95: np.ndarray,
    labels_85,
    labels_95,
) -> Libraries:
    c85 = labels_to_clusters(labels_85)
    c95 = labels_to_clusters(labels_95)
    lib: dict[str, dict[str, np.ndarray]] = {"8.5": {}, "9.5": {}}
    for name in CLUSTER_ORDER:
        i85 = np.array([k == name for k in c85])
        i95 = np.array([k == name for k in c95])
        if i85.any():
            lib["8.5"][name] = z_85[i85]
        if i95.any():
            lib["9.5"][name] = z_95[i95]
    return Libraries(z_by_stage=lib)


def _sample_z(
    lib: dict[str, np.ndarray],
    cluster: str,
    n: int,
    rng: np.random.Generator,
    fallback: str | None = None,
) -> np.ndarray:
    if cluster in lib and len(lib[cluster]) > 0:
        src = lib[cluster]
    elif fallback and fallback in lib and len(lib[fallback]) > 0:
        src = lib[fallback]
    else:
        # last resort: concatenate all available
        src = np.concatenate(list(lib.values()), axis=0)
    idx = rng.choice(len(src), size=n, replace=len(src) < n)
    return src[idx].astype(np.float32, copy=True)


def _maybe_flow(
    z: np.ndarray,
    cluster: str,
    t: float,
    dt: float,
    model: Any,
    trained: set[str],
    steps: int,
) -> np.ndarray:
    if model is None or cluster not in trained or dt == 0:
        return z
    import torch

    from t1.flow import euler_integrate

    ids = cluster_id_map()
    device = next(model.parameters()).device
    zt = torch.from_numpy(z).to(device)
    n = zt.size(0)
    t_t = torch.full((n, 1), t, device=device, dtype=zt.dtype)
    dt_t = torch.full((n, 1), dt, device=device, dtype=zt.dtype)
    cid = torch.full((n,), ids[cluster], device=device, dtype=torch.long)
    with torch.no_grad():
        out = euler_integrate(model, zt, t_t, dt_t, cid, steps=steps)
    return out.cpu().numpy().astype(np.float32)


def generate(
    target_t: float,
    n: int,
    cfg: dict,
    composition: CompositionModel,
    pca: ExpressionPCA,
    libraries: Libraries,
    delta: np.ndarray,
    var_names: list[str],
    *,
    source: str = "9.5",
    use_flow: bool = False,
    model: Any = None,
    seed: int = 0,
) -> ad.AnnData:
    """Generate n cells at target_t.

    source='8.5': one-step proxy toward E9.5 (birth from progenitors).
    source='9.5': extrapolation from E9.5 (E10.5 / E12.5 submissions).
    """
    rng = np.random.default_rng(seed)
    t_cfg = cfg["time"]
    clip_min = float(cfg["shift"]["clip_min"])
    sigma_birth = float(cfg["composition"]["sigma_birth"])
    steps = int(cfg["flow"]["euler_steps"])
    trained = set(FLOW_CLUSTERS)

    if target_t <= t_cfg["t_95"] + 1e-9:
        alpha = 1.0 if source == "8.5" else 0.0
        t_start = t_cfg["t_85"] if source == "8.5" else t_cfg["t_95"]
        dt = target_t - t_start
    elif abs(target_t - t_cfg["t_105"]) < 1e-9:
        alpha = float(cfg["shift"]["alpha_105"])
        t_start = t_cfg["t_95"]
        dt = target_t - t_start
        source = "9.5"
    else:
        alpha = float(cfg["shift"]["alpha_125"])
        t_start = t_cfg["t_95"]
        dt = target_t - t_start
        source = "9.5"

    alloc = composition.allocate(target_t, n, rng)
    alloc.pop(DISSECT_CLUSTER, None)
    # Fix rounding so n matches exactly
    got = sum(alloc.values())
    if got < n:
        keys = list(alloc.keys()) or [k for k in CLUSTER_ORDER if k != DISSECT_CLUSTER]
        alloc[keys[0]] = alloc.get(keys[0], 0) + (n - got)
    elif got > n:
        for k in list(alloc):
            extra = got - n
            take = min(extra, alloc[k])
            alloc[k] -= take
            got -= take
            if alloc[k] == 0:
                del alloc[k]
            if got == n:
                break

    lib = libraries.z_by_stage[source]
    zs = []
    for cluster, count in alloc.items():
        fallback = None
        birth_noise = 0.0
        if source == "8.5" and cluster in BIRTH_CLUSTERS:
            fallback = BIRTH_PROGENITOR.get(cluster)
            birth_noise = sigma_birth
        elif source == "9.5" and cluster in BIRTH_CLUSTERS:
            birth_noise = sigma_birth
        z = _sample_z(lib, cluster, count, rng, fallback=fallback)
        if birth_noise > 0:
            z = z + rng.normal(0.0, birth_noise, size=z.shape).astype(np.float32)
        flow_cluster = cluster
        if source == "8.5" and cluster in BIRTH_CLUSTERS:
            flow_cluster = BIRTH_PROGENITOR.get(cluster, cluster)
        z = _maybe_flow(
            z,
            flow_cluster,
            t=t_start,
            dt=dt,
            model=model if use_flow else None,
            trained=trained,
            steps=steps,
        )
        zs.append(z)

    z_all = np.concatenate(zs, axis=0)[:n]
    X = pca.decode(z_all, clip_min=None)
    X = add_delta(X, delta, alpha=alpha, clip_min=clip_min)
    return _as_adata(X, var_names)


def _as_adata(X: np.ndarray, var_names: list[str]) -> ad.AnnData:
    adata = ad.AnnData(np.asarray(X, dtype=np.float32))
    adata.var_names = var_names
    adata.obs_names = [f"pred_{i}" for i in range(adata.n_obs)]
    return adata


def encode_stage_residual(
    adata: ad.AnnData,
    pca: ExpressionPCA,
    delta: np.ndarray,
    clip_min: float = 0.0,
) -> np.ndarray:
    """Encode x - Δ (clipped) so CFM learns residual transport."""
    zs = []
    batch = pca.batch_size
    for start in range(0, adata.n_obs, batch):
        sl = slice(start, min(start + batch, adata.n_obs))
        x = to_dense(adata[sl].X)
        x = np.clip(x - delta, clip_min, None)
        zs.append(pca.encode(x))
    return np.concatenate(zs, axis=0)
