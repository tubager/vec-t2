#!/usr/bin/env python
"""G0-chassis per-cluster last-step Δ with keep-zeros (heart extrap).

Board: freeze G0 xyz, add α·(μ_E9.5 − μ_E8.75) inside each working cluster.
W3 gate: same operator on an E8.75 copy predicting E9.5 (truth in hand).

  .venv/bin/python scripts/78_extrap_g0_clust_delta.py --eval-w3 --alpha 0.35
  .venv/bin/python scripts/78_extrap_g0_clust_delta.py --alpha 0.75 --include-birth \\
    --out outputs/t2/queue/2026-09-20/214_extrap_clustdelta_a075_keepz_birth_onG0_n25179.h5ad
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from t2.clusters import (  # noqa: E402
    birth_clusters,
    birth_progenitor,
    labels_to_clusters,
    shared_flow_clusters,
)
from t2.expression import cluster_mean_delta  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402


def _align(X, genes, panel):
    genes, panel = list(genes), list(panel)
    if genes == panel:
        return np.asarray(X, dtype=np.float64)
    idx = {g: i for i, g in enumerate(genes)}
    return np.asarray(X[:, [idx[g] for g in panel]], dtype=np.float64)


def _usable(
    left,
    right,
    include_birth: bool,
    only: set[str] | None,
) -> set[str]:
    u = set(shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], "heart"))
    if include_birth:
        n_r = Counter(labels_to_clusters(right.obs["celltype"], "heart"))
        for k in birth_clusters("heart"):
            if n_r.get(k, 0) >= 16:
                u.add(k)
    if only:
        u &= only
    return u


def _curvature_delta(
    prev_left,
    prev_right,
    by_last: dict[str, np.ndarray],
    include_birth: bool,
) -> dict[str, np.ndarray]:
    """Discrete d²: last-step − previous-step. Birth uses progenitor's previous hop."""
    _, by_prev = cluster_mean_delta(prev_left, prev_right, "heart")
    prog = birth_progenitor("heart") if include_birth else {}
    out: dict[str, np.ndarray] = {}
    for key, dlast in by_last.items():
        src = prog.get(str(key), str(key))
        dprev = np.asarray(by_prev.get(src, by_prev.get(str(key), dlast)), dtype=np.float64).ravel()
        out[str(key)] = (np.asarray(dlast, dtype=np.float64).ravel() - dprev).astype(np.float32)
    return out


def _birth_continuation_delta(left, right, by_k: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """μ_E9.5[birth] − μ_E8.75[progenitor] instead of the global-Δ fallback."""
    cl_l = np.asarray(labels_to_clusters(left.obs["celltype"], "heart"))
    cl_r = np.asarray(labels_to_clusters(right.obs["celltype"], "heart"))
    xl = to_dense(left.X)
    xr = to_dense(right.X)
    out = dict(by_k)
    for birth, prog in birth_progenitor("heart").items():
        m_r = cl_r == birth
        m_l = cl_l == prog
        if not (m_r.any() and m_l.any()):
            continue
        out[str(birth)] = (
            np.asarray(xr[m_r].mean(axis=0)) - np.asarray(xl[m_l].mean(axis=0))
        ).astype(np.float32)
    return out


def _assign_clusters(Xp: np.ndarray, Xref: np.ndarray, cref: np.ndarray, knn: int) -> np.ndarray:
    k = min(int(knn), len(Xref))
    _, idx = NearestNeighbors(n_neighbors=k).fit(Xref).kneighbors(Xp)
    names = []
    for row in idx:
        labs = [str(cref[j]) for j in row]
        vals, cnt = np.unique(labs, return_counts=True)
        names.append(str(vals[int(np.argmax(cnt))]))
    return np.asarray(names)


def _apply(
    Xp: np.ndarray,
    names: np.ndarray,
    by_k: dict[str, np.ndarray],
    global_delta: np.ndarray,
    usable: set[str],
    alpha: float,
    rest: set[str] | None = None,
    alpha_rest: float = 0.0,
    curv: dict[str, np.ndarray] | None = None,
    curv_beta: float = 0.0,
    proj_rest: float = 0.0,
    proj_clusters: set[str] | None = None,
    proj_knn: int = 0,
    coords: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    Xn = np.array(Xp, copy=True, dtype=np.float64)
    n_used = 0
    n_rest = 0
    a = float(alpha)
    ar = float(alpha_rest)
    cb = float(curv_beta)
    g = np.asarray(global_delta, dtype=np.float64).ravel()
    for key in sorted(set(names) & usable):
        m = names == key
        if not m.any():
            continue
        d = np.asarray(by_k.get(key, g), dtype=np.float64).ravel()
        Xn[m] = Xn[m] + a * d
        if cb != 0.0 and curv is not None and key in curv:
            Xn[m] = Xn[m] + cb * np.asarray(curv[key], dtype=np.float64).ravel()
        n_used += int(m.sum())
    if ar != 0.0 and rest:
        for key in sorted(set(names) & rest):
            m = names == key
            if not m.any():
                continue
            d = np.asarray(by_k.get(key, g), dtype=np.float64).ravel()
            Xn[m] = Xn[m] + ar * d
            n_rest += int(m.sum())
    pr = float(proj_rest)
    if pr != 0.0:
        m_rest = ~np.isin(names, list(usable))
        if proj_clusters:
            m_rest = m_rest & np.isin(names, list(proj_clusters))
        if m_rest.any():
            pk = int(proj_knn)
            if pk > 0:
                if coords is None:
                    raise SystemExit("proj-knn requires coordinates")
                m_keep = np.isin(names, list(usable))
                keep_idx = np.flatnonzero(m_keep)
                rest_idx = np.flatnonzero(m_rest)
                if len(keep_idx) == 0:
                    raise SystemExit("proj-knn: no keeper cells")
                k = min(pk, len(keep_idx))
                nn = NearestNeighbors(n_neighbors=k).fit(
                    np.asarray(coords, dtype=np.float64)[keep_idx]
                )
                _, nbr = nn.kneighbors(np.asarray(coords, dtype=np.float64)[rest_idx])
                knames = names[keep_idx]
                dkeep = np.stack(
                    [
                        np.asarray(by_k.get(str(nm), g), dtype=np.float64).ravel()
                        for nm in knames
                    ]
                )
                Xn[rest_idx] = Xn[rest_idx] + pr * dkeep[nbr].mean(axis=1)
            else:
                wsum = 0.0
                d_proj = np.zeros_like(g)
                for key in sorted(usable):
                    m = names == key
                    n = int(m.sum())
                    if n == 0:
                        continue
                    d_proj = d_proj + n * np.asarray(by_k.get(key, g), dtype=np.float64).ravel()
                    wsum += n
                if wsum > 0:
                    Xn[m_rest] = Xn[m_rest] + pr * (d_proj / wsum)
            n_rest += int(m_rest.sum())
    np.clip(Xn, 0.0, None, out=Xn)
    Xn[Xp <= 0] = 0.0
    return Xn.astype(np.float32), n_used, n_rest


def eval_w3(alpha: float, knn: int, n: int, seed: int) -> None:
    from common.core_metrics import de_direction, de_score, mmd_unbiased, variogram_score
    from T2.metrics import neighborhood_mmd

    cfg = load_config()
    spec = target_spec("heart", 9.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    left, right, truth = stages["E8.25"], stages["E8.75"], stages["E9.5"]
    XA = _align(to_dense(left.X), left.var_names, panel)
    XB = _align(to_dense(right.X), right.var_names, panel)
    XT = _align(to_dense(truth.X), truth.var_names, panel)
    CB = np.asarray(spatial_xyz(right), float)
    CT = np.asarray(spatial_xyz(truth), float)
    clB = np.asarray(labels_to_clusters(right.obs["celltype"], "heart"))
    usable = set(shared_flow_clusters(left.obs["celltype"], right.obs["celltype"], "heart"))
    gdel, by_k = cluster_mean_delta(left, right, "heart")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(XB), min(n, len(XB)), replace=False)
    X0, C0, names = XB[pick], CB[pick], np.asarray([str(x) for x in clB[pick]])
    Y, n_used, _ = _apply(X0, names, by_k, gdel, usable, alpha)

    def sc(z):
        return z["score"] if isinstance(z, dict) else z

    print("=== W3 gate  E8.75 copy + cluster Δ keepz  vs truth E9.5 ===")
    print(f"alpha={alpha:g} n={len(X0)} used={n_used} mae={float(np.abs(Y - X0).mean()):.4f} fill0=0")
    rows = [("copy", X0), (f"a={alpha:g}", Y)]
    print(f"{'tag':>8} {'de':>7} {'dir':>7} {'mmd':>8} {'var':>8} {'nfs':>8}")
    for tag, X in rows:
        print(
            f"{tag:>8} {sc(de_score(X, XT, XA)):7.3f} {float(de_direction(X, XT, XA)):7.3f} "
            f"{sc(mmd_unbiased(X, XT)):8.4f} {sc(variogram_score(X, XT)):8.4f} "
            f"{sc(neighborhood_mmd(X, C0, XT, CT)):8.4f}"
        )
    de0, de1 = sc(de_score(X0, XT, XA)), sc(de_score(Y, XT, XA))
    var0, var1 = sc(variogram_score(X0, XT)), sc(variogram_score(Y, XT))
    mmd0, mmd1 = sc(mmd_unbiased(X0, XT)), sc(mmd_unbiased(Y, XT))
    ok = (de1 >= de0 - 1e-6) and (var1 <= var0 * 1.10) and (mmd1 <= mmd0 * 1.10)
    print("W3 AND:", "PASS" if ok else "FAIL", "(de not down, var/mmd ≤10% worse)")


def write_board(
    pred: Path,
    out: Path,
    alpha: float,
    knn: int,
    include_birth: bool = False,
    only: set[str] | None = None,
    rest: set[str] | None = None,
    alpha_rest: float = 0.0,
    curv_beta: float = 0.0,
    keeper_boost: float = 1.0,
    seed: int = 0,
    proj_rest: float = 0.0,
    proj_clusters: set[str] | None = None,
    proj_knn: int = 0,
) -> None:
    cfg = load_config()
    spec = target_spec("heart", 10.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    left, right = stages["E8.75"], stages["E9.5"]
    gdel, by_k = cluster_mean_delta(left, right, "heart")
    if include_birth:
        by_k = _birth_continuation_delta(left, right, by_k)
    curv = None
    if curv_beta != 0.0:
        curv = _curvature_delta(stages["E8.25"], left, by_k, include_birth)
    usable = _usable(left, right, include_birth, only)
    if rest:
        overlap = usable & rest
        if overlap:
            raise SystemExit(f"keeper/rest overlap: {sorted(overlap)}")
    if proj_clusters:
        overlap = usable & proj_clusters
        if overlap:
            raise SystemExit(f"keeper/proj-clusters overlap: {sorted(overlap)}")
    Xref = _align(to_dense(right.X), right.var_names, panel)
    cref = np.asarray(labels_to_clusters(right.obs["celltype"], "heart")).astype(str)

    pred_ad = ad.read_h5ad(pred)
    Xp = _align(to_dense(pred_ad.X), pred_ad.var_names, panel)
    C0 = np.array(np.asarray(pred_ad.obsm["spatial_3D"], dtype=np.float32), copy=True)
    names = _assign_clusters(Xp, Xref, cref, knn)
    n_before = {k: int((names == k).sum()) for k in sorted(set(names))}
    if float(keeper_boost) != 1.0:
        if keeper_boost <= 1.0 or keeper_boost >= 1.5:
            raise SystemExit("keeper-boost must be in (1, 1.5); G1 L1 0.1 ≈ −1.3")
        w = np.where(np.isin(names, list(usable)), float(keeper_boost), 1.0 / float(keeper_boost))
        w = w / w.sum()
        idx = np.random.default_rng(int(seed)).choice(len(Xp), size=len(Xp), replace=True, p=w)
        Xp = Xp[idx]
        C0 = C0[idx]
        names = names[idx]
        pred_ad = pred_ad[idx].copy()
        pred_ad.obs_names_make_unique()
        target_rms = 437.92
        rms0 = float(np.sqrt((C0.astype(np.float64) ** 2).sum(1).mean()))
        C0 = (C0.astype(np.float64) * (target_rms / max(rms0, 1e-6))).astype(np.float32)
    Xn, n_used, n_rest = _apply(
        Xp,
        names,
        by_k,
        gdel,
        usable,
        alpha,
        rest,
        alpha_rest,
        curv,
        curv_beta,
        proj_rest,
        proj_clusters,
        int(proj_knn),
        C0,
    )
    n_by = {k: int((names == k).sum()) for k in sorted(set(names))}
    used_by = {k: n_by[k] for k in sorted(usable) if n_by.get(k, 0)}
    rest_by = {
        k: n_by[k]
        for k in sorted(set(rest or ()) | set(proj_clusters or ()))
        if n_by.get(k, 0)
    }
    keys = sorted(set(n_before) | set(n_by))
    l1 = 0.5 * sum(abs(n_by.get(k, 0) / len(names) - n_before.get(k, 0) / sum(n_before.values())) for k in keys)

    fill = int(((Xp <= 0) & (Xn > 0)).sum())
    assert fill == 0, fill
    assert Xn.shape == Xp.shape
    np.clip(Xn, 0.0, None, out=Xn)

    out_ad = pred_ad.copy()
    if list(pred_ad.var_names) == panel:
        out_ad.X = Xn
    else:
        live = np.asarray(to_dense(pred_ad.X), dtype=np.float32)
        idxg = {g: i for i, g in enumerate(panel)}
        for j, g in enumerate(pred_ad.var_names):
            live[:, j] = Xn[:, idxg[g]]
        out_ad.X = live
    out_ad.obsm["spatial_3D"] = C0
    out.parent.mkdir(parents=True, exist_ok=True)
    out_ad.write_h5ad(out, compression="gzip")
    chk = ad.read_h5ad(out)
    rms = float(np.sqrt((np.asarray(chk.obsm["spatial_3D"], dtype=np.float64) ** 2).sum(1).mean()))
    if float(keeper_boost) == 1.0:
        assert np.array_equal(np.asarray(chk.obsm["spatial_3D"], dtype=np.float32), C0)
    birth_n = {k: n_by.get(k, 0) for k in sorted(birth_clusters("heart"))}
    print(
        f"extrap clust-Δ a={alpha:.2f} ar={alpha_rest:.2f} curv={curv_beta:.2f} "
        f"boost={keeper_boost:.2f} proj={proj_rest:.2f} pknn={int(proj_knn)} l1={l1:.4f} birth={int(include_birth)} "
        f"n_used={n_used}/{len(Xp)} n_rest={n_rest} used_by={used_by} rest_by={rest_by} "
        f"assigned_birth={birth_n} "
        f"mae={float(np.abs(Xn - Xp).mean()):.4f} frac0={float((Xn <= 0).mean()):.3f} "
        f"rms={rms:.2f} genes|d|>0.05={int((np.abs(Xn - Xp).mean(0) > 0.05).sum())}  {out}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred",
        type=Path,
        default=ROOT / "outputs/t2/heart/pred_E10.5_full_nodelta_n25179_rms438.h5ad",
        help="G0 chassis (not the live submit, which may already have Δ stacked)",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--eval-w3", action="store_true")
    ap.add_argument("--w3-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--include-birth",
        action="store_true",
        help="also Δ CM_OFT/branch/st via progenitor continuation (shared_flow skips them)",
    )
    ap.add_argument(
        "--clusters",
        type=str,
        default="",
        help="comma-separated cluster names to restrict Δ (default: all usable)",
    )
    ap.add_argument(
        "--alpha-rest",
        type=float,
        default=0.0,
        help="signed Δ on --rest-clusters (in (-1,0); 0=off). Not the banned global α=1.",
    )
    ap.add_argument(
        "--rest-clusters",
        type=str,
        default="",
        help="comma-separated clusters for --alpha-rest (default: gut,ncc,phm,se,endo,CM_V)",
    )
    ap.add_argument(
        "--curv-beta",
        type=float,
        default=0.0,
        help="add β·(last-step − previous-step) on keeper clusters only (in [0,1); 0=off)",
    )
    ap.add_argument(
        "--keeper-boost",
        type=float,
        default=1.0,
        help="resample G0 with keeper weight ×boost, others ×1/boost (1=off; (1,1.5))",
    )
    ap.add_argument(
        "--proj-rest",
        type=float,
        default=0.0,
        help="add ε·(n-weighted keeper last-step) to non-keeper cells; xyz frozen. in [0,1)",
    )
    ap.add_argument(
        "--proj-clusters",
        type=str,
        default="",
        help="restrict --proj-rest to these non-keeper clusters (default: all non-keepers)",
    )
    ap.add_argument(
        "--proj-knn",
        type=int,
        default=0,
        help="if >0, each rest cell gets mean last-step of k spatial-nearest keepers (0=global n-weighted)",
    )
    args = ap.parse_args()
    a = float(args.alpha)
    if a <= 0 or a >= 1.0:
        raise SystemExit("alpha must be in (0, 1); α=1 crashed the board to 42.5")
    ar = float(args.alpha_rest)
    if ar > 0 or ar <= -1.0:
        raise SystemExit("alpha-rest must be in (-1, 0]; 0 disables it")
    cb = float(args.curv_beta)
    if cb < 0 or cb >= 1.0:
        raise SystemExit("curv-beta must be in [0, 1)")
    pr = float(args.proj_rest)
    if pr < 0 or pr >= 1.0:
        raise SystemExit("proj-rest must be in [0, 1)")
    only = {s.strip() for s in args.clusters.split(",") if s.strip()} or None
    rest = {s.strip() for s in args.rest_clusters.split(",") if s.strip()}
    proj_only = {s.strip() for s in args.proj_clusters.split(",") if s.strip()} or None
    if proj_only and pr == 0.0:
        raise SystemExit("proj-clusters requires --proj-rest > 0")
    pknn = int(args.proj_knn)
    if pknn < 0:
        raise SystemExit("proj-knn must be >= 0")
    if pknn > 0 and pr == 0.0:
        raise SystemExit("proj-knn requires --proj-rest > 0")
    if ar != 0.0 and not rest:
        rest = {"gut", "ncc", "phm", "se", "endo", "CM_V"}
    if args.eval_w3:
        eval_w3(a, args.knn, args.w3_n, args.seed)
    if args.out is not None:
        write_board(
            args.pred,
            args.out,
            a,
            args.knn,
            args.include_birth,
            only,
            rest,
            ar,
            cb,
            float(args.keeper_boost),
            int(args.seed),
            pr,
            proj_only,
            pknn,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
