#!/usr/bin/env python
"""Local gates for remaining first-place levers (2026-09-20 evening).

1. embryo: 210 curvature WITHOUT keep-zeros (allow genes to turn on)
2. W3 analog of G0+last-step cluster Δ keep-zeros (E8.75 → E9.5, no E9.5 means in Δ)
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from common.core_metrics import de_direction, de_score, mmd_unbiased, variogram_score  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402
from t2.clusters import labels_to_clusters, shared_flow_clusters  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _dense(a):
    X = a.X
    return np.asarray(X.todense() if hasattr(X, "todense") else X, np.float64)


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def _sc(z):
    return z["score"] if isinstance(z, dict) else z


def _mu_at(ts, mus, t):
    ts = np.asarray(ts, dtype=np.float64)
    mus = np.asarray(mus, dtype=np.float64)
    order = np.argsort(ts)
    ts, mus = ts[order], mus[order]
    if len(ts) == 1:
        return mus[0]
    if t <= ts[0]:
        return mus[0]
    if t >= ts[-1] and len(ts) == 2:
        w = (t - ts[0]) / max(ts[1] - ts[0], 1e-6)
        return (1.0 - w) * mus[0] + w * mus[1]
    T = np.stack([np.ones(len(ts)), ts, ts * ts], axis=1)
    coef, *_ = np.linalg.lstsq(T, mus, rcond=None)
    return coef[0] + coef[1] * t + coef[2] * t * t


def embryo_nokeepz() -> None:
    cfg = load_config()
    spec = target_spec("embryo", 7.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("embryo", cfg, panel)
    three = [("E6.75", 6.75), ("E7.25", 7.25), ("E8.0", 8.0)]
    pair = {"E7.25", "E8.0"}
    usable = set(
        shared_flow_clusters(
            stages["E7.25"].obs["celltype"], stages["E8.0"].obs["celltype"], "embryo"
        )
    )
    packed = {}
    for name, t_s in three:
        a = stages[name]
        packed[name] = (
            t_s,
            _align(to_dense(a.X), a.var_names, panel),
            np.asarray(labels_to_clusters(a.obs["celltype"], "embryo")),
        )
    Xref = np.concatenate([packed["E7.25"][1], packed["E8.0"][1]], 0)
    cref = np.concatenate([packed["E7.25"][2], packed["E8.0"][2]])
    pred = ad.read_h5ad(ROOT / "outputs/t2/queue/2026-09-19/194_embryo_repair_hung_pool_k256_w033_on09a_n5000.h5ad")
    keepz = ad.read_h5ad(ROOT / "outputs/t2/submit/T2_embryo_val_interp.h5ad")
    Xp = _align(to_dense(pred.X), pred.var_names, panel)
    Xk = _align(to_dense(keepz.X), keepz.var_names, panel)
    C = np.asarray(pred.obsm["spatial_3D"], float)
    nn = NearestNeighbors(n_neighbors=15).fit(Xref)
    _, idx = nn.kneighbors(Xp)
    names = []
    for row in idx:
        labs = [str(cref[j]) for j in row]
        vals, cnt = np.unique(labs, return_counts=True)
        names.append(str(vals[int(np.argmax(cnt))]))
    names = np.asarray(names)
    Xn = np.array(Xp, copy=True, dtype=np.float64)
    for key in sorted(set(names) & usable):
        m = names == key
        ts_q, mus_q, ts_l, mus_l = [], [], [], []
        ok = True
        for name, t_s in three:
            _, X, cl = packed[name]
            sel = cl == key
            if sel.sum() < 16:
                ok = False
                break
            mu = X[sel].mean(0)
            ts_q.append(t_s)
            mus_q.append(mu)
            if name in pair:
                ts_l.append(t_s)
                mus_l.append(mu)
        if not ok:
            continue
        dmu = _mu_at(np.array(ts_q), np.stack(mus_q), 7.5) - _mu_at(
            np.array(ts_l), np.stack(mus_l), 7.5
        )
        Xn[m] = Xn[m] + dmu
    np.clip(Xn, 0.0, None, out=Xn)
    fill = int(((Xp <= 0) & (Xn > 0)).sum())
    E675 = packed["E6.75"][1]
    E725 = packed["E7.25"][1]
    E80 = packed["E8.0"][1]
    print("\n=== embryo α=1 curvature, keepz vs fill ===")
    print(f"fill zeros {fill}  mae vs 194 {float(np.abs(Xn - Xp).mean()):.4f}  mae vs 210 {float(np.abs(Xn - Xk).mean()):.4f}")
    for tag, X in [("194", Xp), ("210 keepz", Xk), ("fill", Xn)]:
        print(
            f"{tag:12s} de725 {de_score(X, E725, E675)['score']:.3f} dir725 {de_direction(X, E725, E675):.3f}  "
            f"de80 {de_score(X, E80, E725)['score']:.3f} dir80 {de_direction(X, E80, E725):.3f}  "
            f"mmd210 {_sc(mmd_unbiased(X, Xk)):.5f} var210 {_sc(variogram_score(X, Xk)):.5f}  "
            f"nfs210 {_sc(neighborhood_mmd(X, C, Xk, C)):.5f}"
        )


def w3_last_step() -> None:
    cfg = load_config()
    spec = target_spec("heart", 9.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    A, B, T = stages["E8.25"], stages["E8.75"], stages["E9.5"]
    XA = _align(to_dense(A.X), A.var_names, panel)
    XB = _align(to_dense(B.X), B.var_names, panel)
    XT = _align(to_dense(T.X), T.var_names, panel)
    CA = np.asarray(A.obsm["spatial_3D"] if "spatial_3D" in A.obsm else A.obsm["spatial"], float)[:, :3]
    CB = np.asarray(B.obsm["spatial_3D"] if "spatial_3D" in B.obsm else B.obsm["spatial"], float)[:, :3]
    CT = np.asarray(T.obsm["spatial_3D"] if "spatial_3D" in T.obsm else T.obsm["spatial"], float)[:, :3]
    from t2.io import spatial_xyz

    CA, CB, CT = spatial_xyz(A), spatial_xyz(B), spatial_xyz(T)
    clA = np.asarray(labels_to_clusters(A.obs["celltype"], "heart"))
    clB = np.asarray(labels_to_clusters(B.obs["celltype"], "heart"))
    usable = set(shared_flow_clusters(A.obs["celltype"], B.obs["celltype"], "heart"))
    rng = np.random.default_rng(0)
    n = 5000
    pick = rng.choice(len(XB), n, replace=False)
    X0, C0, cl0 = XB[pick], np.asarray(CB[pick], float), clB[pick]
    dmu = {}
    for k in sorted(usable):
        mA, mB = clA == k, clB == k
        if mA.sum() < 16 or mB.sum() < 16:
            continue
        dmu[k] = XB[mB].mean(0) - XA[mA].mean(0)
    dt_obs, dt_fwd = 8.75 - 8.25, 9.5 - 8.75
    scale = dt_fwd / max(dt_obs, 1e-6)  # 1.5 if applying full continuation
    print("\n=== W3 E8.75 copy + last-step cluster Δ keepz vs E9.5 ===")
    print(f"clusters with dmu {len(dmu)}  scale_full={scale:.3f}")
    base_de = de_score(X0, XT, XA)
    print(
        f"{'alpha':>6} {'de':>7} {'dir':>7} {'mmd':>8} {'var':>8} {'nfs':>8} {'mae':>7} {'fill0':>6}"
    )
    print(
        f"{'copy':>6} {base_de['score']:7.3f} {de_direction(X0, XT, XA):7.3f} "
        f"{_sc(mmd_unbiased(X0, XT)):8.4f} {_sc(variogram_score(X0, XT)):8.4f} "
        f"{_sc(neighborhood_mmd(X0, C0, XT, CT)):8.4f} {0:7.4f} {0:6d}"
    )
    for a in (0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50):
        Y = np.array(X0, copy=True, dtype=np.float64)
        for k, d in dmu.items():
            m = cl0 == k
            if m.any():
                Y[m] = Y[m] + a * scale * d
        np.clip(Y, 0.0, None, out=Y)
        fill = int(((X0 <= 0) & (Y > 0)).sum())
        Y[X0 <= 0] = 0.0  # keepz
        print(
            f"{a:6.2f} {de_score(Y, XT, XA)['score']:7.3f} {de_direction(Y, XT, XA):7.3f} "
            f"{_sc(mmd_unbiased(Y, XT)):8.4f} {_sc(variogram_score(Y, XT)):8.4f} "
            f"{_sc(neighborhood_mmd(Y, C0, XT, CT)):8.4f} {float(np.abs(Y - X0).mean()):7.4f} {fill:6d}"
        )


if __name__ == "__main__":
    embryo_nokeepz()
    w3_last_step()
