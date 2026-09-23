#!/usr/bin/env python
"""W3-gated hop-CFM continuation past the last observed stage (heart extrap).

Not bidir interpolant (209 closed). Not VAE. Not shrink (237 anti-indicator).
Train hop E8.25→E8.75 (no E9.5). From E8.75 cells, Euler dt=E9.5−E8.75, keep-zeros.
AND vs 234 analog: de not down, var/mmd ≤10% worse.

  .venv/bin/python scripts/73_train_hop_logpca.py --setting heart --exclude E9.5 \\
    --hop E8.25,E8.75 --t-scale 4.0 --epochs 40 --device mps \\
    --out-dir outputs/t2/heart/gate95_hoplog
  .venv/bin/python scripts/80_w3_hop_continue.py --ckpt-dir outputs/t2/heart/gate95_hoplog
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from common.core_metrics import de_direction, de_score, mmd_unbiased, variogram_score  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402
from t2.clusters import labels_to_clusters  # noqa: E402
from t2.expression import cluster_mean_delta  # noqa: E402
from t2.flow import get_device, load_velocity  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402

KEEPERS = frozenset({"CM_IFT", "CM_OFT", "peri", "pam", "branch", "st"})
ALPHA_234 = 0.95
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


@torch.no_grad()
def continue_euler(model, z, t_start, dt_hop, dt_fwd, cid, steps: int):
    """v is per-day; keep the trained dt in the dt-channel, integrate dt_fwd."""
    z = z.clone()
    h = float(dt_fwd) / max(int(steps), 1)
    n = z.size(0)
    dt_t = torch.full((n, 1), float(dt_hop), device=z.device, dtype=z.dtype)
    for i in range(int(steps)):
        t_i = torch.full((n, 1), float(t_start) + i * h, device=z.device, dtype=z.dtype)
        z = z + model(z, t_i, dt_t, cid) * h
    return z


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
        f"{tag:>14} {row['de']:7.3f} {row['dir']:7.3f} {row['mmd']:8.4f} "
        f"{row['var']:8.4f} {row['nfs']:8.4f} {mae:7.4f}"
    )


def flow_keep(
    X: np.ndarray,
    names: np.ndarray,
    pca,
    model,
    device,
    ids: dict,
    trained: set[str],
    keys: set[str],
    t_start: float,
    dt_hop: float,
    dt_fwd: float,
    steps: int,
) -> np.ndarray:
    Y = np.log1p(np.maximum(X, 0.0))
    Xn = np.array(X, copy=True, dtype=np.float64)
    use = keys & trained & set(ids)
    model.eval()
    with torch.no_grad():
        for key in sorted(set(names) & use):
            m = names == key
            if not m.any():
                continue
            z_np = pca.transform(Y[m]).astype(np.float32)
            z = torch.as_tensor(z_np, device=device)
            cid = torch.full((int(m.sum()),), int(ids[key]), device=device, dtype=torch.long)
            z1 = continue_euler(model, z, t_start, dt_hop, dt_fwd, cid, steps).cpu().numpy()
            dlog = (z1 - z_np) @ np.asarray(pca.components_, dtype=np.float64)
            Xh = np.clip(np.expm1(Y[m] + dlog), 0.0, None)
            Xn[m] = Xh
    return keepz(Xn, X)


def eval_w3(ckpt_dir: Path, n: int, seed: int, steps: int, blend: tuple[float, ...]) -> dict:
    s78 = _load_78()
    device = get_device()
    blob = joblib.load(ckpt_dir / "pca.joblib")
    pca = blob["pca"]
    model, extra = load_velocity(ckpt_dir / "ckpt" / "flow.pt", device=device)
    ids = dict(extra["cluster_to_id"])
    trained = set(extra.get("trained_clusters") or [])

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
    clB = np.asarray(labels_to_clusters(right.obs["celltype"], "heart")).astype(str)
    gdel, by_k = cluster_mean_delta(left, right, "heart")
    by_k = s78._birth_continuation_delta(left, right, by_k)
    usable = s78._usable(left, right, include_birth=True, only=set(KEEPERS))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(XB), min(n, len(XB)), replace=False)
    X0, C0, names = XB[pick], CB[pick], clB[pick]
    Y234, _, _ = s78._apply(
        X0, names, by_k, gdel, usable, ALPHA_234,
        None, 0.0, None, 0.0, PROJ_EPS, set(PROJ), 0, C0,
    )
    Y234 = keepz(Y234, X0)

    dt_hop = 8.75 - 8.25
    dt_fwd = 9.5 - 8.75
    Yflow = flow_keep(
        X0, names, pca, model, device, ids, trained, usable,
        8.75, dt_hop, dt_fwd, steps,
    )
    Ystack = flow_keep(
        Y234, names, pca, model, device, ids, trained, usable,
        8.75, dt_hop, dt_fwd, steps,
    )

    print("=== W3 hop-CFM continue  E8.75 → dt=0.75 vs truth E9.5 ===")
    print(f"device={device} trained={sorted(trained & usable)} dt_hop={dt_hop:g} dt_fwd={dt_fwd:g}")
    print(f"{'tag':>14} {'de':>7} {'dir':>7} {'mmd':>8} {'var':>8} {'nfs':>8} {'mae':>7}")
    rows = {}
    catalog = [("copy", X0), ("234", Y234), ("flow_copy", Yflow), ("flow_on234", Ystack)]
    for a in blend:
        if a <= 0 or a >= 1:
            continue
        Yb = keepz((1.0 - a) * Y234 + a * Ystack, X0)
        catalog.append((f"blend_{a:g}", Yb))
        Yc = keepz((1.0 - a) * Y234 + a * Yflow, X0)
        catalog.append((f"mixflow_{a:g}", Yc))
    for tag, X in catalog:
        row = scores(X, C0, XT, CT, XA)
        rows[tag] = row
        print_row(tag, row, float(np.abs(X - X0).mean()))

    base = rows["234"]
    print("\nAND vs 234 analog:")
    for tag, row in rows.items():
        if tag in {"copy", "234"}:
            continue
        ok = gate_and(row, base)
        improve = ok and (row["de"] > base["de"] + 1e-6 or row["var"] < base["var"] - 1e-8)
        print(
            f"  {tag:>14}  {'PASS' if ok else 'FAIL'}"
            f"  de {row['de'] - base['de']:+.3f}  var {row['var'] - base['var']:+.4f}"
            f"  mmd {row['mmd'] - base['mmd']:+.4f}"
            f"{'  improve' if improve else ''}"
        )
    return rows


def write_board(
    ckpt_dir: Path,
    blend: float,
    steps: int,
    out: Path,
) -> None:
    s78 = _load_78()
    pred = ROOT / "outputs/t2/heart/pred_E10.5_full_nodelta_n25179_rms438.h5ad"
    tmp = out.with_name(out.stem + ".__234__.h5ad")
    s78.write_board(
        pred,
        tmp,
        alpha=ALPHA_234,
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
    device = get_device()
    blob = joblib.load(ckpt_dir / "pca.joblib")
    pca = blob["pca"]
    model, extra = load_velocity(ckpt_dir / "ckpt" / "flow.pt", device=device)
    ids = dict(extra["cluster_to_id"])
    trained = set(extra.get("trained_clusters") or [])

    cfg = load_config()
    spec = target_spec("heart", 10.5, cfg)
    panel = list(load_panel_for(spec, cfg))
    stages = load_stages("heart", cfg, panel)
    left, right = stages["E8.75"], stages["E9.5"]
    g0 = ad.read_h5ad(pred)
    Xp = s78._align(to_dense(g0.X), g0.var_names, panel)
    ad234 = ad.read_h5ad(tmp)
    Y234 = s78._align(to_dense(ad234.X), ad234.var_names, panel)
    Xref = s78._align(to_dense(right.X), right.var_names, panel)
    cref = np.asarray(labels_to_clusters(right.obs["celltype"], "heart")).astype(str)
    names = s78._assign_clusters(Xp, Xref, cref, 15)
    usable = s78._usable(left, right, include_birth=False, only=set(KEEPERS))
    dt_hop = 9.5 - 8.75
    dt_fwd = 10.5 - 9.5
    Yflow = flow_keep(
        Y234, names, pca, model, device, ids, trained, usable,
        9.5, dt_hop, dt_fwd, steps,
    )
    a = float(blend)
    if a < 1.0 - 1e-12:
        Y = keepz((1.0 - a) * Y234 + a * Yflow, Xp)
    else:
        Y = keepz(Yflow, Xp)
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
    print(
        f"board hop-cont blend={a:g} mae_g0={float(np.abs(Y - Xp).mean()):.4f} "
        f"mae_234={float(np.abs(Y - Y234).mean()):.4f} fill0=0 "
        f"rms={float(np.sqrt((C0.astype(np.float64) ** 2).sum(1).mean())):.2f}  {out}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--w3-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--write-board", action="store_true")
    ap.add_argument("--blend", type=float, default=0.50)
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/t2/queue/2026-09-21/238_extrap_234_hopcont_b050_onG0_n25179.h5ad",
    )
    args = ap.parse_args()
    if args.write_board:
        b = float(args.blend)
        if b <= 0 or b > 1.0:
            raise SystemExit("blend must be in (0, 1]")
        write_board(args.ckpt_dir, b, int(args.steps), args.out)
        return 0
    eval_w3(args.ckpt_dir, int(args.w3_n), int(args.seed), int(args.steps), (0.15, 0.35, 0.50))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
