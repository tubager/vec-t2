#!/usr/bin/env python
"""W3 gate for CondAE identity→future-t residual (heart extrap).

Encode the real cell (no t); decode at t* with a zero-init time head.
Not hop-CFM, not noise→X, not VAE sampling.

Native-t recon must PASS (mae ≤ 0.03, sparsity Δ ≤ 0.05) before W3 counts.
W3: encode E8.75 → decode t=9.5, keep-zeros, vs 234 analog.
Chimera veto: var > copy×1.5. Stack veto: mae vs 234 > 0.04.
Polluter-only de wins do not board.

  .venv/bin/python scripts/66_train_cond_flow.py --setting heart --left E8.25 --right E8.75 \\
    --exclude E9.5 --arch ae --z-dim 128 --hidden 1024 --t-scale 4.0 --rms 216.9 \\
    --epochs 40 --device mps --out-dir outputs/t2/heart/gate95_condae
  .venv/bin/python scripts/82_w3_condae.py --ckpt-dir outputs/t2/heart/gate95_condae
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".venv/lib/python3.13/site-packages"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

from common.core_metrics import de_direction, de_score, mmd_unbiased, variogram_score  # noqa: E402
from T2.metrics import neighborhood_mmd  # noqa: E402
from t2.clusters import labels_to_clusters  # noqa: E402
from t2.cond_flow import load_cond_flow  # noqa: E402
from t2.expression import cluster_mean_delta  # noqa: E402
from t2.flow import get_device  # noqa: E402
from t2.geometry import scale_cloud  # noqa: E402
from t2.infer import load_panel_for, load_stages, target_spec  # noqa: E402
from t2.io import spatial_xyz, to_dense  # noqa: E402
from t2.paths import load_config  # noqa: E402

KEEPERS = frozenset({"CM_IFT", "CM_OFT", "peri", "pam", "branch", "st"})
ALPHA_234 = 0.95
PROJ_EPS = 0.20
PROJ = frozenset({"ncc"})
MAE_CAP = 0.04
RECON_MAE = 0.03


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


def scores(X, C, XT, CT, XA):
    return {
        "de": _sc(de_score(X, XT, XA)),
        "dir": float(de_direction(X, XT, XA)),
        "mmd": _sc(mmd_unbiased(X, XT)),
        "var": _sc(variogram_score(X, XT)),
        "nfs": _sc(neighborhood_mmd(X, C, XT, CT)),
    }


def decode_at(
    model,
    meta: dict,
    X: np.ndarray,
    names: np.ndarray,
    coords: np.ndarray,
    t: float,
    device,
    batch: int = 512,
) -> np.ndarray:
    ids = dict(meta["cluster_to_id"])
    usable = set(meta.get("usable") or [])
    rms = float(meta.get("rms") or 216.9)
    xyz = np.asarray(scale_cloud(coords, rms), dtype=np.float32) / rms
    cid = np.array([int(ids[str(n)]) if str(n) in ids else 0 for n in names], dtype=np.int64)
    out = np.array(X, copy=True, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch):
            sl = slice(start, min(start + batch, len(X)))
            names_sl = names[sl]
            use = np.array([str(nm) in usable and str(nm) in ids for nm in names_sl])
            if not use.any():
                continue
            x_in = torch.as_tensor(X[sl][use], device=device, dtype=torch.float32)
            c = torch.as_tensor(cid[sl][use], device=device)
            zx = torch.as_tensor(xyz[sl][use], device=device)
            z, _ = model.encode(x_in, c, zx)
            tb = x_in.new_full((int(use.sum()),), float(t))
            xg = model.decode(z, tb, c, zx).clamp_min(0.0).cpu().numpy()
            dest = np.array(out[sl], copy=True)
            dest[use] = xg
            out[sl] = dest
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--w3-n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    s78 = _load_78()
    device = get_device()
    model, meta = load_cond_flow(args.ckpt_dir / "ckpt" / "cond_flow.pt", device=device)
    if str(meta.get("arch")) != "ae":
        raise SystemExit(f"expected arch=ae, got {meta.get('arch')}")
    print(
        f"arch=ae z={meta.get('z_dim')} hidden={meta.get('hidden')} "
        f"t_scale={meta.get('t_scale')} t0={meta.get('t0')} t1={meta.get('t1')}"
    )

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
    usable_k = s78._usable(left, right, include_birth=True, only=set(KEEPERS))
    rng = np.random.default_rng(int(args.seed))
    pick = rng.choice(len(XB), min(int(args.w3_n), len(XB)), replace=False)
    X0, C0, names = XB[pick], CB[pick], clB[pick]
    Y234, _, _ = s78._apply(
        X0, names, by_k, gdel, usable_k, ALPHA_234,
        None, 0.0, None, 0.0, PROJ_EPS, set(PROJ), 0, C0,
    )
    Y234 = keepz(Y234, X0)

    rec = keepz(decode_at(model, meta, X0, names, C0, 8.75, device), X0)
    mae_rec = float(np.abs(rec - X0).mean())
    f0_r, f0_t = float((rec <= 0).mean()), float((X0 <= 0).mean())
    rec_ok = mae_rec <= RECON_MAE and abs(f0_r - f0_t) <= 0.05
    print(
        f"RECON native t=8.75 mae={mae_rec:.4f} ({'PASS' if mae_rec <= RECON_MAE else 'FAIL'} ≤{RECON_MAE})  "
        f"frac0={f0_r:.3f} vs {f0_t:.3f}  => {'PASS' if rec_ok else 'FAIL — stop, no W3 board'}"
    )
    if not rec_ok:
        return 0

    fut = keepz(decode_at(model, meta, X0, names, C0, 9.5, device), X0)
    catalog = [("copy", X0), ("234", Y234), ("ae_t95", fut)]
    for a in (0.10, 0.25, 1.00):
        catalog.append((f"stack_{a:g}", keepz(Y234 + a * (fut - Y234), X0)))
    m_k = np.isin(names, list(usable_k))
    Yk = np.array(Y234, copy=True)
    Yk[m_k] = Y234[m_k] + 0.10 * (fut[m_k] - Y234[m_k])
    catalog.append(("stack_k0.1", keepz(Yk, X0)))

    print("=== W3 CondAE  encode E8.75 → decode t=9.5 vs truth E9.5 ===")
    print(f"{'tag':>12} {'de':>7} {'dir':>7} {'mmd':>8} {'var':>8} {'nfs':>8} {'mae234':>7} {'mae0':>7}")
    rows = {}
    by_tag = dict(catalog)
    copy_var = None
    for tag, X in catalog:
        row = scores(X, C0, XT, CT, XA)
        rows[tag] = row
        if tag == "copy":
            copy_var = row["var"]
        mae234 = float(np.abs(X - Y234).mean())
        mae0 = float(np.abs(X - X0).mean())
        print(
            f"{tag:>12} {row['de']:7.3f} {row['dir']:7.3f} {row['mmd']:8.4f} "
            f"{row['var']:8.4f} {row['nfs']:8.4f} {mae234:7.4f} {mae0:7.4f}"
        )

    base = rows["234"]
    print("\nAND vs 234 analog + chimera/mae/polluter veto:")
    for tag, row in rows.items():
        if tag in {"copy", "234"}:
            continue
        and_ok = (
            row["de"] >= base["de"] - 1e-6
            and row["var"] <= base["var"] * 1.10
            and row["mmd"] <= base["mmd"] * 1.10
        )
        chimera = row["var"] > copy_var * 1.50
        mae234 = float(np.abs(by_tag[tag] - Y234).mean())
        stack = tag.startswith("stack")
        mae_ok = mae234 <= MAE_CAP if stack else True
        replace = tag == "ae_t95"
        if replace:
            mae_ok = mae234 <= MAE_CAP
        polluter_only = tag.startswith("stack_") and tag != "stack_k0.1" and rows.get("stack_k0.1", {}).get("de", 0) <= base["de"] + 1e-6 and row["de"] > base["de"] + 1e-6
        improve = and_ok and (row["de"] > base["de"] + 1e-6 or row["var"] < base["var"] - 1e-8)
        board = and_ok and (not chimera) and mae_ok and improve and (not polluter_only) and rec_ok
        if tag == "stack_k0.1" and row["de"] <= base["de"] + 1e-6:
            board = False
        print(
            f"  {tag:>12}  AND={'PASS' if and_ok else 'FAIL'}"
            f"  chimera={'YES' if chimera else 'no'}"
            f"  mae234={mae234:.4f}"
            f"{'  polluter-de' if polluter_only else ''}"
            f"  de {row['de'] - base['de']:+.3f}  var {row['var'] - base['var']:+.4f}"
            f"  => {'BOARD' if board else 'no-board'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
