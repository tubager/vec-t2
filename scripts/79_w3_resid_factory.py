#!/usr/bin/env python
"""W3-gated cell-level residual factory on the 234 analog (heart extrap).

last-step / 投影族已关。本脚本不扫 α、ε、proj-kNN。
底盘：E8.75 拷贝 + 载体 α=0.95 last-step + 只对 ncc ε=0.20，keep-zeros，对真 E9.5。
候选是均值位移之后的簇条件残差（细胞 NN 速度、簇内体积比），不是 VAE。

AND vs 234 analog：de 不降，var/mmd ≤10% 更差。过门才允许写板。

  .venv/bin/python scripts/79_w3_resid_factory.py
  .venv/bin/python scripts/79_w3_resid_factory.py --write-board --op nn_resid --beta 0.25
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from common.core_metrics import de_direction, de_score, mmd_unbiased, variogram_score  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402
from t2.clusters import labels_to_clusters  # noqa: E402
from t2.expression import cluster_mean_delta  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402

KEEPERS = frozenset({"CM_IFT", "CM_OFT", "peri", "pam", "branch", "st"})
ALPHA = 0.95
PROJ_EPS = 0.20
PROJ = frozenset({"ncc"})


def _load_78():
    spec = importlib.util.spec_from_file_location(
        "s78_extrap", ROOT / "scripts/78_extrap_g0_clust_delta.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sc(z):
    return z["score"] if isinstance(z, dict) else z


def keepz(Xn: np.ndarray, X0: np.ndarray) -> np.ndarray:
    out = np.clip(np.asarray(Xn, dtype=np.float64), 0.0, None)
    out[np.asarray(X0) <= 0] = 0.0
    return out


def nn_velocity(
    X_dst: np.ndarray,
    names: np.ndarray,
    X_src: np.ndarray,
    names_src: np.ndarray,
    usable: set[str],
    pca_k: int = 32,
) -> np.ndarray:
    """vel_i = X_dst_i − X_src[NN]. PCA-32 只用于找邻，残差仍在全 panel。"""
    vel = np.zeros_like(X_dst, dtype=np.float64)
    for key in sorted(usable):
        md = names == key
        ms = names_src == key
        n_d, n_s = int(md.sum()), int(ms.sum())
        if n_d == 0 or n_s < 16:
            continue
        src = np.asarray(X_src[ms], dtype=np.float64)
        dst = np.asarray(X_dst[md], dtype=np.float64)
        k = int(min(pca_k, n_s - 1, src.shape[1], 32))
        k = max(k, 2)
        pca = PCA(n_components=k, random_state=0)
        z_s = pca.fit_transform(src)
        z_d = pca.transform(dst)
        _, idx = NearestNeighbors(n_neighbors=1).fit(z_s).kneighbors(z_d)
        vel[md] = dst - src[idx.ravel()]
    return vel


def center_by_cluster(V: np.ndarray, names: np.ndarray, usable: set[str]) -> np.ndarray:
    out = np.array(V, copy=True, dtype=np.float64)
    for key in sorted(usable):
        m = names == key
        if not m.any():
            continue
        out[m] = out[m] - out[m].mean(axis=0)
    return out


def std_ratio_scale(
    X: np.ndarray,
    names: np.ndarray,
    X_left: np.ndarray,
    names_left: np.ndarray,
    X_right: np.ndarray,
    names_right: np.ndarray,
    usable: set[str],
    beta: float,
) -> np.ndarray:
    """簇内、按基因：把 234 后的偏差按 last-hop std 比缩放。均值不变。"""
    Y = np.array(X, copy=True, dtype=np.float64)
    eps = 1e-6
    for key in sorted(usable):
        m = names == key
        ml = names_left == key
        mr = names_right == key
        if not m.any() or ml.sum() < 16 or mr.sum() < 16:
            continue
        sl = np.std(X_left[ml], axis=0) + eps
        sr = np.std(X_right[mr], axis=0) + eps
        ratio = np.clip(sr / sl, 0.5, 2.0)
        # β=1 全量体积迁移；β<1 收缩
        r = 1.0 + float(beta) * (ratio - 1.0)
        mu = Y[m].mean(axis=0)
        Y[m] = mu + (Y[m] - mu) * r
    return Y


def scores(X, C, XT, CT, XA):
    return {
        "de": _sc(de_score(X, XT, XA)),
        "dir": float(de_direction(X, XT, XA)),
        "mmd": _sc(mmd_unbiased(X, XT)),
        "var": _sc(variogram_score(X, XT)),
        "nfs": _sc(neighborhood_mmd(X, C, XT, CT)),
    }


def gate_and(row: dict, base: dict) -> bool:
    return (
        row["de"] >= base["de"] - 1e-6
        and row["var"] <= base["var"] * 1.10
        and row["mmd"] <= base["mmd"] * 1.10
    )


def print_row(tag: str, row: dict, mae: float) -> None:
    print(
        f"{tag:>16} {row['de']:7.3f} {row['dir']:7.3f} {row['mmd']:8.4f} "
        f"{row['var']:8.4f} {row['nfs']:8.4f} {mae:7.4f}"
    )


def eval_w3(n: int, seed: int) -> dict[str, dict]:
    s78 = _load_78()
    cfg = load_config()
    spec = target_spec("heart", 9.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    left, right, truth = stages["E8.25"], stages["E8.75"], stages["E9.5"]
    XA = s78._align(to_dense(left.X), left.var_names, panel)
    XB = s78._align(to_dense(right.X), right.var_names, panel)
    XT = s78._align(to_dense(truth.X), truth.var_names, panel)
    CB = np.asarray(spatial_xyz(right), float)
    CT = np.asarray(spatial_xyz(truth), float)
    clA = np.asarray(labels_to_clusters(left.obs["celltype"], "heart")).astype(str)
    clB = np.asarray(labels_to_clusters(right.obs["celltype"], "heart")).astype(str)
    clT = np.asarray(labels_to_clusters(truth.obs["celltype"], "heart")).astype(str)
    gdel, by_k = cluster_mean_delta(left, right, "heart")
    by_k = s78._birth_continuation_delta(left, right, by_k)
    usable = s78._usable(left, right, include_birth=True, only=set(KEEPERS))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(XB), min(n, len(XB)), replace=False)
    X0, C0, names = XB[pick], CB[pick], clB[pick]
    Y234, n_used, n_rest = s78._apply(
        X0, names, by_k, gdel, usable, ALPHA,
        None, 0.0, None, 0.0, PROJ_EPS, set(PROJ), 0, C0,
    )
    Y234 = keepz(Y234, X0)

    print("=== W3 residual factory  E8.75 + 234 analog → truth E9.5 ===")
    print(
        f"n={len(X0)} keepers_used={n_used} ncc_proj={n_rest} "
        f"usable={sorted(usable)} fill0=0"
    )
    print(f"{'tag':>16} {'de':>7} {'dir':>7} {'mmd':>8} {'var':>8} {'nfs':>8} {'mae':>7}")

    rows: dict[str, dict] = {}
    catalog: list[tuple[str, np.ndarray]] = [("copy", X0), ("234", Y234)]

    vel = nn_velocity(X0, names, XA, clA, usable)
    vel_c = center_by_cluster(vel, names, usable)
    Y_vel = keepz(X0 + ALPHA * vel, X0)
    Y_vel, _, _ = s78._apply(
        Y_vel, names, by_k, gdel, usable, 0.0,
        None, 0.0, None, 0.0, PROJ_EPS, set(PROJ), 0, C0,
    )
    Y_vel = keepz(Y_vel, X0)
    catalog.append(("nn_vel", Y_vel))

    for beta in (0.15, 0.25, 0.40):
        Y = keepz(Y234 + beta * vel_c, X0)
        catalog.append((f"nn_resid_{beta:g}", Y))

    for beta in (0.25, 0.50, 1.00):
        Y = keepz(std_ratio_scale(Y234, names, XA, clA, XB, clB, usable, beta), X0)
        catalog.append((f"std_ratio_{beta:g}", Y))

    for lam in (0.15, 0.25, 0.35, 0.50):
        Y = np.array(Y234, copy=True, dtype=np.float64)
        for key in sorted(usable):
            m = names == key
            if not m.any():
                continue
            mu = Y[m].mean(axis=0)
            Y[m] = mu + (1.0 - lam) * (Y[m] - mu)
        catalog.append((f"shrink_{lam:g}", keepz(Y, X0)))

    def _nn_truth(src: np.ndarray, names_s: np.ndarray, keys: set[str] | None) -> np.ndarray:
        nn_t = NearestNeighbors(n_neighbors=1)
        out = np.array(src, copy=True, dtype=np.float64)
        use = set(names_s) if keys is None else keys
        for key in sorted(use):
            m = names_s == key
            mt = clT == key
            if not m.any() or mt.sum() < 16:
                continue
            _, idx = nn_t.fit(XT[mt]).kneighbors(src[m])
            out[m] = XT[np.flatnonzero(mt)][idx.ravel()]
        return out

    tgt_all = _nn_truth(Y234, names, None)
    tgt_k = _nn_truth(Y234, names, set(usable))
    catalog.append(("oracle_l015", keepz(0.85 * Y234 + 0.15 * tgt_all, X0)))
    catalog.append(("oracle_k015", keepz(0.85 * Y234 + 0.15 * tgt_k, X0)))

    Y_pb = np.array(Y234, copy=True, dtype=np.float64)
    Y_pb_k = np.array(Y234, copy=True, dtype=np.float64)
    Y_rs = np.array(Y234, copy=True, dtype=np.float64)
    for key in sorted(set(names)):
        m = names == key
        mt = clT == key
        if not m.any() or not mt.any():
            continue
        mu_p, mu_t = Y234[m].mean(axis=0), XT[mt].mean(axis=0)
        Y_pb[m] = Y234[m] - mu_p + mu_t
        if key in usable:
            Y_pb_k[m] = Y234[m] - mu_p + mu_t
        nn = NearestNeighbors(n_neighbors=1).fit(XT[mt])
        _, idx = nn.kneighbors(Y234[m])
        tgt = XT[np.flatnonzero(mt)][idx.ravel()]
        Y_rs[m] = tgt - tgt.mean(axis=0) + mu_p
    catalog.append(("pb_oracle", keepz(Y_pb, X0)))
    catalog.append(("pb_oracle_k", keepz(Y_pb_k, X0)))
    catalog.append(("resid_oracle", keepz(Y_rs, X0)))

    for tag, X in catalog:
        row = scores(X, C0, XT, CT, XA)
        rows[tag] = row
        print_row(tag, row, float(np.abs(X - X0).mean()))

    base = rows["234"]
    print("\nAND vs 234 analog (de not down, var/mmd ≤10% worse):")
    for tag, row in rows.items():
        if tag in {"copy", "234"} or tag.startswith("oracle") or tag.startswith("pb_oracle") or tag == "resid_oracle":
            continue
        ok = gate_and(row, base)
        de_up = row["de"] > base["de"] + 1e-6
        var_dn = row["var"] < base["var"] - 1e-8
        print(
            f"  {tag:>16}  {'PASS' if ok else 'FAIL'}"
            f"  de {row['de'] - base['de']:+.3f}  var {row['var'] - base['var']:+.4f}"
            f"  mmd {row['mmd'] - base['mmd']:+.4f}"
            f"{'  improve' if ok and (de_up or var_dn) else ''}"
        )

    print("\nper-cluster |μ_234 − μ_truth| (keepers / ncc / other):")
    for key in sorted(set(names)):
        m = names == key
        mt = clT == key
        if not m.any() or not mt.any():
            continue
        d0 = float(np.abs(X0[m].mean(0) - XT[mt].mean(0)).mean())
        d1 = float(np.abs(Y234[m].mean(0) - XT[mt].mean(0)).mean())
        mark = "K" if key in usable else ("P" if key in PROJ else ".")
        print(f"  {mark} {key:12s} n={int(m.sum()):4d}  copy {d0:.4f}  234 {d1:.4f}  Δ {d1-d0:+.4f}")
    return rows


def write_board(op: str, beta: float, out: Path) -> None:
    s78 = _load_78()
    pred = ROOT / "outputs/t2/heart/pred_E10.5_full_nodelta_n25179_rms438.h5ad"
    tmp = out.with_name(out.stem + ".__234__.h5ad")
    s78.write_board(
        pred,
        tmp,
        alpha=ALPHA,
        knn=15,
        include_birth=True,
        only=set(KEEPERS),
        rest=None,
        alpha_rest=0.0,
        curv_beta=0.0,
        keeper_boost=1.0,
        seed=0,
        proj_rest=PROJ_EPS,
        proj_clusters=set(PROJ),
        proj_knn=0,
    )
    cfg = load_config()
    spec = target_spec("heart", 10.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    left, right = stages["E8.75"], stages["E9.5"]
    XA = s78._align(to_dense(left.X), left.var_names, panel)
    XB = s78._align(to_dense(right.X), right.var_names, panel)
    clA = np.asarray(labels_to_clusters(left.obs["celltype"], "heart")).astype(str)
    clB = np.asarray(labels_to_clusters(right.obs["celltype"], "heart")).astype(str)
    usable = s78._usable(left, right, include_birth=True, only=set(KEEPERS))
    shrink_keys = s78._usable(left, right, include_birth=False, only=set(KEEPERS))

    g0 = ad.read_h5ad(pred)
    Xp = s78._align(to_dense(g0.X), g0.var_names, panel)
    ad234 = ad.read_h5ad(tmp)
    Y = s78._align(to_dense(ad234.X), ad234.var_names, panel)
    Xref = XB
    cref = clB
    names = s78._assign_clusters(Xp, Xref, cref, 15)

    if op == "nn_resid":
        vel = nn_velocity(Xp, names, XA, clA, usable)
        Y = keepz(Y + float(beta) * center_by_cluster(vel, names, usable), Xp)
    elif op == "std_ratio":
        Y = keepz(std_ratio_scale(Y, names, XA, clA, XB, clB, usable, float(beta)), Xp)
    elif op == "shrink":
        Yn = np.array(Y, copy=True, dtype=np.float64)
        for key in sorted(shrink_keys):
            m = names == key
            if not m.any():
                continue
            mu = Yn[m].mean(axis=0)
            Yn[m] = mu + (1.0 - float(beta)) * (Yn[m] - mu)
        Y = keepz(Yn, Xp)
    elif op == "nn_vel":
        gdel, by_k = cluster_mean_delta(left, right, "heart")
        by_k = s78._birth_continuation_delta(left, right, by_k)
        C0 = np.asarray(ad234.obsm["spatial_3D"], dtype=np.float64)
        Y = keepz(Xp + ALPHA * nn_velocity(Xp, names, XA, clA, usable), Xp)
        Y, _, _ = s78._apply(
            Y, names, by_k, gdel, usable, 0.0,
            None, 0.0, None, 0.0, PROJ_EPS, set(PROJ), 0, C0,
        )
        Y = keepz(Y, Xp)
    else:
        raise SystemExit(f"unknown op {op}")

    fill = int(((Xp <= 0) & (Y > 0)).sum())
    assert fill == 0, fill
    C0 = np.asarray(ad234.obsm["spatial_3D"], dtype=np.float32)
    if list(ad234.var_names) == panel:
        ad234.X = Y.astype(np.float32)
    else:
        live = np.asarray(to_dense(ad234.X), dtype=np.float32)
        idxg = {g: i for i, g in enumerate(panel)}
        for j, g in enumerate(ad234.var_names):
            live[:, j] = Y[:, idxg[g]]
        ad234.X = live
    ad234.obsm["spatial_3D"] = C0
    out.parent.mkdir(parents=True, exist_ok=True)
    ad234.write_h5ad(out, compression="gzip")
    tmp.unlink(missing_ok=True)
    rms = float(np.sqrt((C0.astype(np.float64) ** 2).sum(1).mean()))
    print(
        f"board {op} β={beta:g} mae_vs_g0={float(np.abs(Y - Xp).mean()):.4f} "
        f"fill0=0 rms={rms:.2f}  {out}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w3-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-board", action="store_true")
    ap.add_argument("--op", choices=("nn_resid", "std_ratio", "nn_vel", "shrink"), default="")
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/t2/queue/2026-09-21/237_extrap_234_resid_onG0_n25179.h5ad",
    )
    args = ap.parse_args()
    rows = eval_w3(int(args.w3_n), int(args.seed))
    if not args.write_board:
        return 0
    if not args.op:
        raise SystemExit("--write-board requires --op")
    if args.op == "nn_vel":
        tag = "nn_vel"
    elif args.op == "nn_resid":
        tag = f"nn_resid_{float(args.beta):g}"
    elif args.op == "shrink":
        tag = f"shrink_{float(args.beta):g}"
    else:
        tag = f"std_ratio_{float(args.beta):g}"
    if tag not in rows:
        raise SystemExit(f"W3 did not evaluate {tag}")
    base = rows["234"]
    row = rows[tag]
    if not gate_and(row, base):
        raise SystemExit(f"REFUSE board: {tag} FAIL W3 AND vs 234")
    if not (row["de"] > base["de"] + 1e-6 or row["var"] < base["var"] - 1e-8):
        raise SystemExit(f"REFUSE board: {tag} AND but no de/var improve")
    write_board(args.op, float(args.beta), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
